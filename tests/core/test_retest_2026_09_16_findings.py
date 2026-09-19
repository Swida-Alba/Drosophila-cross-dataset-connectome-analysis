"""Regression tests for the 2026-09-16 Windows re-test findings.

The re-test report (`DROCAT_retest_report_2026-09-16.md`, tested revision
``df0600e``) confirmed every fix from the 2026-09-12 round and reported four
new findings; these tests pin the FIXES for those findings (implemented
2026-09-17).  The pre-fix behavior is documented in each docstring so the
regression value of every assertion stays clear.

- F2 (high) — the cache-only coverage gate missed a 77% cache loss through
  three holes: raw-row counting vs the loader's deduplicated counting (hole
  1), comparison restricted to downstream_complete-flagged neurons (hole 2),
  and "any cached rows = complete" classification that never reaches the
  fetch-time gate for row-level loss (hole 3).  Fixes: loader-faithful dedup
  in ``_check_cache_coverage`` + the ``cache_manifest.json`` whole-cache
  layer; hole 3's fetch-time gate stays as belt-and-suspenders.
- F1 (medium) — a fetch-time refusal used to leave the run folder (and its
  bootstrap files) behind.  Fix: folders this run created are tracked and
  removed on refusal; user folders (saveas) are never touched.
- F3 (medium) — ``banc_public_data.http_get`` did not retry
  ``http.client.IncompleteRead`` (field: a BANC download died mid-body after
  67.8 MB despite ``attempts=3``).  Fix: ``http.client.HTTPException`` joined
  the retry except tuple.
- F4 (low) — the refusal message spells out the remedy instead of assuming
  an opt-in: the 2026-09-18 round removed ``allow_incomplete_cache`` and its
  INCOMPLETE_CACHE.txt stamping entirely, so no run can proceed on an
  incomplete cache and every refusal has to name a reachable fix.  The
  static check below pins that the bypass is gone from the whole tree.
"""

import http.client
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import banc_public_data as bpd
import coana
from coana import cache_coverage_mismatches


# ---------------------------------------------------------------------------
# Synthetic cache helpers (main table + batch files, like the real layout)
# ---------------------------------------------------------------------------

DATASET = 'test:v1.0'


def _cache_paths(tmp_path):
    dataset_safe = coana.dataset_folder(DATASET)
    cache_dir = tmp_path / 'cache' / dataset_safe
    index_dir = tmp_path / 'neuron_indexes' / dataset_safe
    return dataset_safe, cache_dir, index_dir


def _write_cache(tmp_path, main_rows, batch_rows_by_file, index_rows):
    """Write connections.parquet, batch files, and the neuron index.

    ``main_rows``/batch rows: (bodyId_pre, bodyId_post, roi, weight) tuples.
    ``index_rows``: (bodyId, downstream_complete, connection_count).
    """
    dataset_safe, cache_dir, index_dir = _cache_paths(tmp_path)
    cache_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)

    def frame(rows):
        return pl.DataFrame({
            'bodyId_pre': [r[0] for r in rows],
            'bodyId_post': [r[1] for r in rows],
            'roi': [r[2] for r in rows],
            'weight': [r[3] for r in rows],
        })

    frame(main_rows).write_parquet(cache_dir / 'connections.parquet')
    batch_dir = cache_dir / '_batch_files'
    batch_dir.mkdir()
    for name, rows in batch_rows_by_file.items():
        frame(rows).write_parquet(batch_dir / name)
    pl.DataFrame({
        'bodyId': [r[0] for r in index_rows],
        'downstream_complete': [r[1] for r in index_rows],
        'connection_count': [r[2] for r in index_rows],
    }).write_parquet(index_dir / 'neuron_index.parquet')


def _write_manifest(tmp_path, distinct):
    import json
    _, cache_dir, _ = _cache_paths(tmp_path)
    (cache_dir / 'cache_manifest.json').write_text(json.dumps({
        'schema': 1, 'dataset': DATASET, 'distinct_connections': distinct,
        'built_at': '2026-09-17T00:00:00', 'source': 'test'}))


def _connection_files(tmp_path):
    _, cache_dir, _ = _cache_paths(tmp_path)
    files = [str(cache_dir / 'connections.parquet')]
    batch_dir = cache_dir / '_batch_files'
    files.extend(str(p) for p in sorted(batch_dir.glob('batch_*.parquet')))
    return files


def _dedup_pre_counts(tmp_path):
    """What the loader serves: distinct (pre, post, roi) rows per source."""
    counts = (
        pl.scan_parquet(_connection_files(tmp_path))
        .unique(subset=['bodyId_pre', 'bodyId_post', 'roi'], keep='last')
        .group_by('bodyId_pre')
        .len()
        .collect()
    )
    return {str(k): v for k, v in zip(counts['bodyId_pre'].to_list(),
                                      counts['len'].to_list())}


def _make_fc(tmp_path):
    fc = object.__new__(coana.FindNeuronConnection)
    fc.script_path = str(tmp_path)
    fc.dataset = DATASET
    fc._dataset_safe = coana.dataset_folder(DATASET)
    fc.use_cache = True
    fc._vprint = lambda *a, **k: None
    return fc


@pytest.fixture(autouse=True)
def _clear_coverage_memo():
    coana._CACHE_COVERAGE_CACHE.clear()
    yield
    coana._CACHE_COVERAGE_CACHE.clear()


# ---------------------------------------------------------------------------
# F2 hole 1 — the gate now counts like the loader (deduplicated)
# ---------------------------------------------------------------------------

class TestF2Hole1UnitMismatch:
    def test_pristine_dedup_matches_recorded_while_raw_is_inflated(self, tmp_path):
        """Healthy cache shaped like the field one: main + batch hold the
        SAME rows, so raw = 2x dedup.  Field-measured on male-cns: raw
        3,047,740 vs dedup 1,523,870 vs recorded 1,523,870.  The gate must
        stay silent here (no false positive) despite the raw inflation."""
        rows = [(100, 1000 + i, f'ROI_{i}', 5) for i in range(100)]
        _write_cache(tmp_path, main_rows=rows,
                     batch_rows_by_file={'batch_0.parquet': rows},
                     index_rows=[(100, True, 100)])
        dedup = _dedup_pre_counts(tmp_path)
        raw = sum(
            pl.scan_parquet(f).select(pl.len()).collect().item()
            for f in _connection_files(tmp_path))
        assert raw == 200 and dedup['100'] == 100  # exactly the old 2.0x cushion
        fc = _make_fc(tmp_path)
        coverage = fc._check_cache_coverage()
        assert coverage['mismatches'] == []
        assert coverage['distinct_connections'] == 100
        # The pure helper agrees with the deduplicated count.
        assert cache_coverage_mismatches([(100, True, 100)], dedup) == ([], 1)

    def test_gate_detects_distinct_loss_after_dedup_correction(self, tmp_path):
        """Regression for hole 1: 45 of 100 distinct connections dropped from
        BOTH files.  Pre-fix, the raw count (110) still exceeded the recorded
        count (100) and the gate stayed silent; the dedup-corrected gate
        reports the shortfall."""
        keep = [(100, 1000 + i, f'ROI_{i}', 5) for i in range(55)]
        _write_cache(tmp_path, main_rows=keep,
                     batch_rows_by_file={'batch_0.parquet': keep},
                     index_rows=[(100, True, 100)])
        fc = _make_fc(tmp_path)
        coverage = fc._check_cache_coverage()
        assert coverage['mismatches'] == [('100', 100, 55)]
        # Enforcement refuses (this fixture also lacks a manifest).
        with pytest.raises(RuntimeError, match='incomplete'):
            fc._enforce_cache_coverage('user_requested')


# ---------------------------------------------------------------------------
# F2 hole 2 — the manifest layer covers non-flagged neurons
# ---------------------------------------------------------------------------

class TestF2Hole2Scope:
    def test_unflagged_neuron_loss_is_caught_by_the_manifest(self, tmp_path):
        """Regression for hole 2: neuron B is not flagged
        downstream_complete (like 94.7% of the male-cns index), so the
        per-neuron layer never compares it — and must stay blind (widening
        it would false-positive: FlyWire-imported index rows record the
        dataset ``post`` column, not cached rows).  The whole-cache manifest
        layer is the detector: on-disk distinct collapses against the
        recorded baseline."""
        a_rows = [(100, 1000 + i, 'ROI', 5) for i in range(10)]
        b_rows = [(200, 5000 + i, 'ROI', 5) for i in range(5)]
        _write_cache(tmp_path, main_rows=a_rows + b_rows, batch_rows_by_file={},
                     index_rows=[(100, True, 10), (200, False, 1_000_000)])
        fc = _make_fc(tmp_path)
        # Per-neuron layer: blind to B by design.
        assert fc._check_cache_coverage()['mismatches'] == []
        # Whole-cache layer: catches the loss.
        _write_manifest(tmp_path, distinct=1_000_005)
        with pytest.raises(RuntimeError, match='integrity manifest records'):
            fc._enforce_cache_coverage('user_requested')


# ---------------------------------------------------------------------------
# F2 hole 3 — fetch-time classification (belt-and-suspenders layer)
# ---------------------------------------------------------------------------

class TestF2Hole3DetectionDepth:
    def test_partial_rows_classify_neuron_as_cached(self, tmp_path):
        """Field shape (shallow query, truncated cache): the source keeps
        507 of its 2,201 downstream partners.  ``_query_connection_db``
        classifies the neuron as cached because rows exist — the "rows prove
        completeness" assumption is deliberately kept as the fast path, and
        the manifest gate now catches this scenario BEFORE the run starts
        (whole-cache loss).  This test pins the fetch-time classification so
        the layered design stays explicit."""
        fc = object.__new__(coana.FindNeuronConnection)
        fc.use_cache = True
        fc._vprint = lambda *a, **k: None
        fc._conn_db_pre_id_cache = None
        fc._load_neuron_index = lambda: pl.DataFrame()
        fc._neuron_index_dict = {
            '100': {'downstream_complete': True, 'connection_count': 2201,
                    'type': 'aMe4', 'instance': '', 'post': 2201,
                    'last_fetched': '', 'row_idx': 0},
        }
        rows = 507
        fc._load_connection_db = lambda: pl.DataFrame({
            'bodyId_pre': [100] * rows,
            'bodyId_post': [300000 + i for i in range(rows)],
            'roi': [''] * rows,
            'weight': [5] * rows,
        })
        fc._conn_index = {'100': list(range(rows))}

        cached_conn, uncached, partial = fc._query_connection_db([100], None)

        assert uncached == []          # no whole-neuron miss at fetch time
        assert cached_conn.height == rows  # the 507 surviving rows are served
        assert partial == []


# ---------------------------------------------------------------------------
# F1 — refusal removes the folders this run created
# ---------------------------------------------------------------------------

class TestF1RefusalCleansRunFolder:
    def test_refusal_removes_folders_created_by_this_run(self, tmp_path):
        run_dir = tmp_path / 'find-paths-complete_TEST_src_to_tgt_L2w3_ts'
        run_dir.mkdir()
        (run_dir / 'parameters.txt').write_text('params')
        fc = object.__new__(coana.FindNeuronConnection)
        fc.dataset = DATASET
        fc._vprint = lambda *a, **k: None
        fc._run_created_folders = [str(run_dir)]

        with pytest.raises(RuntimeError, match='not in the local cache'):
            fc._handle_cache_only_miss([512925, 72227])

        assert not run_dir.exists()

    def test_refusal_never_touches_untracked_folders(self, tmp_path):
        """User-pointed folders (saveas) already existed and are never
        tracked — a refusal must leave them alone."""
        user_dir = tmp_path / 'user_saveas'
        user_dir.mkdir()
        (user_dir / 'precious.csv').write_text('data')
        fc = object.__new__(coana.FindNeuronConnection)
        fc.dataset = DATASET
        fc._vprint = lambda *a, **k: None
        fc._run_created_folders = []

        with pytest.raises(RuntimeError):
            fc._handle_cache_only_miss([1])

        assert (user_dir / 'precious.csv').exists()


# ---------------------------------------------------------------------------
# F3 — http_get retries http.client.IncompleteRead
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload=None, exc=None):
        self._payload = payload
        self._exc = exc

    def read(self):
        if self._exc is not None:
            raise self._exc
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestF3IncompleteRead:
    def test_incomplete_read_is_retried(self, monkeypatch):
        """Regression: the field failure (response.read() raising
        IncompleteRead mid-body) now exhausts attempts and raises the
        aggregated RuntimeError instead of escaping on the first attempt."""
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(request)
            # partial must be bytes-like: IncompleteRead's repr formats its
            # length (the field error read "67845400 bytes read, 8280407
            # more expected").
            return _FakeResponse(
                exc=http.client.IncompleteRead(b'x' * 6784, 8280407))

        monkeypatch.setattr(bpd.urllib.request, 'urlopen', fake_urlopen)
        monkeypatch.setattr(bpd.time, 'sleep', lambda seconds: None)

        with pytest.raises(RuntimeError, match='after 3 attempts'):
            bpd.http_get('https://example.invalid/product', attempts=3,
                         timeout=1, note='connections product')
        assert len(calls) == 3

    def test_incomplete_read_retry_succeeds(self, monkeypatch):
        """A body interrupted once and completed on the next attempt now
        succeeds (pre-fix: hard IncompleteRead on attempt 1)."""
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(request)
            if len(calls) < 2:
                return _FakeResponse(exc=http.client.IncompleteRead(b'x' * 10, 5))
            return _FakeResponse(payload=b'ok')

        monkeypatch.setattr(bpd.urllib.request, 'urlopen', fake_urlopen)
        monkeypatch.setattr(bpd.time, 'sleep', lambda seconds: None)

        assert bpd.http_get('https://example.invalid/product', attempts=3,
                            timeout=1) == b'ok'
        assert len(calls) == 2

    def test_covered_exceptions_are_still_retried(self, monkeypatch):
        """Control for the harness above: an exception type inside the
        except tuple IS retried and a later attempt succeeds."""
        import urllib.error as urllib_error
        attempts = {'n': 0}

        def fake_urlopen(request, timeout=None):
            attempts['n'] += 1
            if attempts['n'] < 3:
                raise urllib_error.URLError('transient')
            return _FakeResponse(payload=b'ok')

        monkeypatch.setattr(bpd.urllib.request, 'urlopen', fake_urlopen)
        monkeypatch.setattr(bpd.time, 'sleep', lambda seconds: None)

        assert bpd.http_get('https://example.invalid/product', attempts=3,
                            timeout=1) == b'ok'
        assert attempts['n'] == 3


# ---------------------------------------------------------------------------
# F4 — the UI has no opt-in control; the refusal message carries the remedy
# ---------------------------------------------------------------------------

class TestF4NoIncompleteCacheBypass:
    def test_incomplete_cache_bypass_is_gone_tree_wide(self):
        """An incomplete cache is never runnable (2026-09-18 decision), so
        neither the opt-in flag nor its PARTIAL-results marker may come
        back — not in the product, and not as documented advice."""
        repo = Path(__file__).resolve().parents[2]
        hits = []
        for folder in ('src', 'ui', 'scripts', 'docs', 'skills'):
            for path in (repo / folder).rglob('*'):
                if not path.is_file() or path.suffix not in {'.py', '.md'}:
                    continue
                text = path.read_text(errors='ignore')
                if 'allow_incomplete_cache' in text or 'INCOMPLETE_CACHE' in text:
                    hits.append(str(path.relative_to(repo)))
        assert hits == []
