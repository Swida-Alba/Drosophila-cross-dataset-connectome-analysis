#!/usr/bin/env python
"""
Regression tests for batched NeuPrint neuron fetching.

The neuprint client issues ONE Cypher query per fetch_neurons() call. A single
call with tens of thousands of bodyIds makes the server evaluate a giant
IN-list and returns a massive payload that the client parses for minutes at
~100% CPU (perceived as a hang). coana._fetch_neurons_batched() chunks the
list so every query and response stays bounded.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import coana  # noqa: E402


def _make_chunk_df(ids):
    return pd.DataFrame({
        'bodyId': [str(i) for i in ids],
        'type': ['T'] * len(ids),
        'instance': ['I'] * len(ids),
    })


class _StubCriteria:
    """Stand-in for neuprint's `NeuronCriteria`.

    Building a real one resolves neuprint's DEFAULT CLIENT
    (`neuprint/client.py:106`), so in a process that never created a Client
    the chunk builder raises before the fetch is attempted, the retry loop
    sleeps its 5→40 s backoff, and the patched `fetch_neurons` is never
    reached — which made these four tests fail as `calls == []` AND cost
    minutes of sleeping each. To the batching contract under test the
    criteria is a value object, so a stand-in with the same attributes keeps
    the assertion honest and the run offline."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestFetchNeuronsBatched:
    @pytest.fixture(autouse=True)
    def _no_default_client_needed(self, monkeypatch):
        """Every case here drives `_fetch_neurons_batched`, which builds a
        `NeuronCriteria` per chunk — and building one resolves neuprint's
        default client BEFORE any fetch is attempted. Autouse, because a test
        that patches only `fetch_neurons` still falls into that wall and then
        reads as "the patched client was never called"."""
        monkeypatch.setattr(coana, 'NeuronCriteria', _StubCriteria)

    def _client(self, monkeypatch):
        calls = []

        def fake_fetch_neurons(criteria):
            ids = criteria.bodyId if isinstance(criteria.bodyId, list) else [criteria.bodyId]
            calls.append(len(ids))
            return _make_chunk_df(ids), pd.DataFrame()

        monkeypatch.setattr(coana, 'fetch_neurons', fake_fetch_neurons)
        return calls

    def test_small_list_is_single_call(self, monkeypatch):
        calls = self._client(monkeypatch)
        fc = coana.FindNeuronConnection(dataset='male-cns:v1.0')
        out = fc._fetch_neurons_batched([1, 2, 3, 4, 5])
        assert calls == [5]
        assert len(out) == 5

    def test_large_list_is_chunked(self, monkeypatch):
        calls = self._client(monkeypatch)
        fc = coana.FindNeuronConnection(dataset='male-cns:v1.0')
        ids = list(range(5000))
        out = fc._fetch_neurons_batched(ids, batch_size=2000)
        assert calls == [2000, 2000, 1000], f'chunk sizes: {calls}'
        assert len(out) == 5000
        assert sorted(out['bodyId'].astype(int).tolist()) == ids

    def test_empty_input_returns_empty_frame(self, monkeypatch):
        calls = self._client(monkeypatch)
        fc = coana.FindNeuronConnection(dataset='male-cns:v1.0')
        out = fc._fetch_neurons_batched([])
        assert calls == []
        assert out.empty

    def test_chunk_errors_retry_then_continue_with_remaining_chunks(self, monkeypatch):
        """A chunk that keeps failing after retries is reported and SKIPPED;
        the remaining chunks still download (resilient pull)."""
        import time as _time
        monkeypatch.setattr(_time, 'sleep', lambda s: None)  # no backoff wait
        calls = []
        status = []

        def flaky_fetch(criteria):
            ids = criteria.bodyId
            calls.append(len(ids))
            if len(ids) == 2000 and len(calls) <= 5:
                raise RuntimeError('server timeout')
            return _make_chunk_df(ids), pd.DataFrame()

        monkeypatch.setattr(coana, 'fetch_neurons', flaky_fetch)
        fc = coana.FindNeuronConnection(dataset='male-cns:v1.0')
        out = fc._fetch_neurons_batched(
            list(range(4000)), batch_size=2000, status_callback=status.append
        )
        assert len(out) == 2000  # only the second chunk survived
        assert len(calls) == 6  # 5 attempts on chunk 1 + chunk 2
        assert any('Server not responding' in s for s in status), status
        assert any('failed after retries' in s for s in status), status
        assert any('attempt 1/5' in s for s in status), status

    def test_neurons_batched_reports_downloading_progress(self, monkeypatch):
        """The first-run neuron pull shows a 'Pulling neurons from server'
        progress bar over the total neuron count, updated per chunk."""
        class _FakeTqdm:
            instances = []

            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs
                self.updates = []
                _FakeTqdm.instances.append(self)

            def update(self, n):
                self.updates.append(n)

            def close(self):
                pass

        def fake_fetch(criteria):
            return _make_chunk_df(criteria.bodyId), pd.DataFrame()

        # Patch at BOTH levels: coana's binding (what fetch_chunk resolves)
        # and neuprint's own attribute (leaked executor workers from earlier
        # tests may call the restored module-level function after their
        # test's monkeypatch is undone).
        import neuprint as np_mod
        monkeypatch.setattr(coana, 'fetch_neurons', fake_fetch)
        monkeypatch.setattr(np_mod, 'fetch_neurons', fake_fetch)
        # Even a fully unpatched call must not fail: stub the default client.
        class _StubClient:
            def fetch_neurons(self, criteria=None):
                return _make_chunk_df(criteria.bodyId if criteria else []), pd.DataFrame()

            def fetch_custom(self, *a, **k):
                return pd.DataFrame()

        monkeypatch.setattr(np_mod, 'default_client', lambda: _StubClient())

        monkeypatch.setattr('tqdm.tqdm', _FakeTqdm)
        fc = coana.FindNeuronConnection(dataset='male-cns:v1.0')
        _FakeTqdm.instances.clear()
        fc._fetch_neurons_batched(list(range(5000)), batch_size=2000)
        assert len(_FakeTqdm.instances) == 1
        bar = _FakeTqdm.instances[0]
        assert bar.kwargs['total'] == 5000
        assert bar.kwargs['desc'] == 'Pulling neurons from server'
        assert bar.updates == [2000, 2000, 1000]

    def test_neurons_batched_cancel_stops_between_chunks(self, monkeypatch):
        """A set cancel event aborts the pull between chunks (raises
        _FetchCancelled, never recorded as a fetch failure)."""
        import threading
        ev = threading.Event()
        ev.set()
        monkeypatch.setattr(
            coana, 'fetch_neurons',
            lambda criteria: (_make_chunk_df(criteria.bodyId), pd.DataFrame()),
        )
        fc = coana.FindNeuronConnection(dataset='male-cns:v1.0')
        with pytest.raises(coana._FetchCancelled):
            fc._fetch_neurons_batched(list(range(4000)), batch_size=2000,
                                      cancel_event=ev)


# =============================================================================
# Bulk connection fetch: timeout/retry (reconnect) + cooperative cancel
# =============================================================================

def _conn_df(ids):
    return pd.DataFrame({
        'bodyId_pre': [str(i) for i in ids],
        'bodyId_post': ['999'] * len(ids),
        'weight': [1] * len(ids),
        'roi': ['AL'] * len(ids),
    })


class TestFetchConnectionsBulk:
    def _fc(self, monkeypatch):
        import time as _time
        monkeypatch.setattr(_time, 'sleep', lambda s: None)  # no backoff wait
        fc = coana.FindNeuronConnection(dataset='male-cns:v1.0')
        fc._ensure_neuprint_client = lambda: None  # no real login in tests
        fc.simple_fetch = False  # route through fetch_adjacencies (patchable)
        return fc

    def test_bulk_fetch_reconnects_after_server_failure(self, monkeypatch):
        """A server that fails twice is reconnected: the batch succeeds on
        the third attempt and the user is told about the reconnection."""
        import neuprint as np_mod
        attempts = {'n': 0}

        def flaky_adj(sources, targets=None, min_total_weight=1, **kw):
            attempts['n'] += 1
            assert kw.get('batch_size') == len(sources)
            if attempts['n'] < 3:
                raise ConnectionError('server not responding')
            return pd.DataFrame(), _conn_df(sources)

        monkeypatch.setattr(np_mod, 'fetch_adjacencies', flaky_adj)
        fc = self._fc(monkeypatch)
        status = []
        out = fc._fetch_connections_bulk(['1', '2', '3'], status_callback=status.append)
        assert attempts['n'] == 3
        assert len(out) == 3
        assert any('Server not responding' in s for s in status), status
        assert any('reconnecting' in s for s in status), status

    def test_bulk_fetch_cancel_during_retry(self, monkeypatch):
        """Cancel is honoured even while a batch is retrying: the retry
        notice raises _FetchCancelled instead of waiting for the next
        attempt."""
        import threading
        import neuprint as np_mod
        ev = threading.Event()

        def failing_adj(sources, targets=None, min_total_weight=1, **kw):
            ev.set()  # simulate the user cancelling after the failure
            raise ConnectionError('server down')

        monkeypatch.setattr(np_mod, 'fetch_adjacencies', failing_adj)
        fc = self._fc(monkeypatch)
        with pytest.raises(coana._FetchCancelled):
            fc._fetch_connections_bulk(['1'], cancel_event=ev)

    def test_build_cache_cancel_mid_batch_not_marked_failed(self, tmp_path, monkeypatch):
        """Cancelling while a batch is fetching stops the build cleanly:
        the summary says cancelled=True and the in-flight batch is NOT
        recorded as failed (a re-run resumes it from the checkpoint)."""
        import threading
        fc = coana.FindNeuronConnection(dataset='male-cns:v1.0')
        fc.use_cache = True
        fc._get_all_dataset_bodyids = lambda: ['1', '2', '3']
        fc._load_neuron_index = lambda force_reload=False: pd.DataFrame()
        fc._neuron_index_dict = {}
        fc._get_connection_db_path = lambda: str(tmp_path / 'connections.parquet')
        fc._get_neuron_index_path = lambda: str(tmp_path / 'neuron_index.parquet')
        fc._consolidate_batch_files = lambda deduplicate=False: None
        fc._count_cached_connections = lambda: 0
        fc._append_connections_to_cache = lambda conns, batch: None
        fc._update_neuron_index_batch = lambda batch: None

        ev = threading.Event()

        def cancel_bulk(*args, **kwargs):
            ev.set()
            raise coana._FetchCancelled('cancelled')

        fc._fetch_connections_bulk = cancel_bulk
        summary = fc.build_connection_cache(batch_size=2, quiet=True, cancel_event=ev)
        assert summary['cancelled'] is True
        assert summary['failed_neurons'] == []
        assert summary['newly_cached'] == 0
