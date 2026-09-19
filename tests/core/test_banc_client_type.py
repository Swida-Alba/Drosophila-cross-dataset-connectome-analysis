"""Standalone ``client_type='banc'`` migration tests (integration plan §I,
issue-report BANC-02 and BANC-09).

Covers:
* the BANC client-type normalization (default/'neuprint'/'banc' all land
  on 'banc'; other values are rejected; FAFB/NeuPrint datasets are
  untouched);
* ``_ensure_complete_dataset`` early-returning for 'banc' (the doomed
  ``sv.pull_dataset`` per-init regression);
* ``_fetch_api_connections`` serving BANC cache misses from the SAME
  per-version merged table and never reaching the NeuPrint fetch (with
  bucket guidance, not FAFB codex instructions, on a total miss);
* ``_ensure_banc_connection_cache`` honoring the False return of the
  table builder (fresh-install recovery: ensure -> rebuild; hard exit
  when the tables stay absent).

Heavy collaborators are monkeypatched; no network access happens.
"""

import sys
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import coana  # noqa: E402

BANC = 'banc_v888'
MERGED = PROJECT_ROOT / 'datasets' / 'banc_v888' / \
    'banc_v888_merged_connections.parquet'

pytestmark = pytest.mark.skipif(
    not MERGED.exists(),
    reason='local banc_v888 merged table not available')


def _make_fc(**attrs):
    """Minimal copy of the make_fc convention in test_coana_coverage."""
    fc = object.__new__(coana.FindNeuronConnection)
    fc._vprint = lambda msg="", level="full", end="\n", flush=False: None
    fc.verbose_mode = "silent"
    fc.progress_events = False
    fc._warn_notes = []
    fc.dataset = BANC
    fc.client_type = 'neuprint'
    fc.script_path = str(PROJECT_ROOT)
    for key, value in attrs.items():
        setattr(fc, key, value)
    return fc


# ---------------------------------------------------------------------------
# client_type normalization (§I.2)
# ---------------------------------------------------------------------------

def test_banc_default_neuprint_normalizes_to_banc():
    fc = _make_fc(client_type='neuprint')
    fc._normalize_client_type()
    assert fc.client_type == 'banc'


def test_banc_explicit_banc_stays():
    fc = _make_fc(client_type='banc')
    fc._normalize_client_type()
    assert fc.client_type == 'banc'


def test_banc_rejects_other_client_types():
    fc = _make_fc(client_type='flywire')
    with pytest.raises(ValueError, match="not valid for BANC"):
        fc._normalize_client_type()


def test_non_banc_datasets_unaffected():
    fc = _make_fc(dataset='hemibrain:v1.2.1', client_type='neuprint')
    fc._normalize_client_type()
    assert fc.client_type == 'neuprint'

    fc = _make_fc(dataset='flywire_FAFB_v783', client_type='neuprint')
    fc._normalize_client_type()
    assert fc.client_type == 'flywire'


def test_visualizer_and_validation_accept_banc():
    """Source-level contract (the visualizer's __post_init__ is too heavy
    to construct in a unit test): the normalization + no-client guard and
    the validation whitelist must name 'banc'."""
    source = (PROJECT_ROOT / 'src' / 'visualize_skeleton.py').read_text(
        encoding='utf-8')
    assert "client_type='banc'" in source
    assert "not is_banc_dataset(self.dataset)" in source
    assert "('neuprint', 'flywire', 'banc')" in source


# ---------------------------------------------------------------------------
# _ensure_complete_dataset: the doomed sv.pull_dataset regression (BANC-09 #2)
# ---------------------------------------------------------------------------

def test_ensure_complete_dataset_early_returns_for_banc(monkeypatch):
    # statvis.pull_dataset is the doomed call on the NeuPrint path; with
    # the 'banc' early-return it must never be reached.
    import statvis as sv
    called = []
    monkeypatch.setattr(sv, 'pull_dataset',
                        lambda *a, **k: called.append(a))
    fc = _make_fc(client_type='banc', cache_only=False)
    # Must return before any download machinery runs.
    assert fc._ensure_complete_dataset() is None
    assert called == []


# ---------------------------------------------------------------------------
# _fetch_api_connections: BANC cache misses served from the merged table
# ---------------------------------------------------------------------------

def _one_upstream_id():
    frame = pl.scan_parquet(MERGED).select(
        pl.col('bodyId_pre').cast(pl.Utf8).alias('bodyId_pre')).head(1).collect()
    return frame['bodyId_pre'][0]


def test_fetch_api_connections_serves_banc_from_merged_table():
    fc = _make_fc(client_type='banc', use_cache=True, cache_only=False,
                  _banc_local_conn_cache=None)
    upstream = _one_upstream_id()
    api_conn = fc._fetch_api_connections([upstream], None)
    assert api_conn is not None and not api_conn.empty
    assert set(api_conn['bodyId_pre'].astype(str)) == {str(upstream)}
    assert {'bodyId_pre', 'bodyId_post', 'weight', 'roi'} <= set(api_conn.columns)
    # the mtime-aware snapshot is populated for subsequent layer fetches
    assert fc._banc_local_conn_cache is not None


def test_fetch_api_connections_banc_never_reaches_neuprint(monkeypatch):
    fc = _make_fc(client_type='banc', use_cache=True, cache_only=False,
                  _banc_local_conn_cache=None)
    upstream = _one_upstream_id()
    # If the NeuPrint fetchers were ever reached, these raise.
    monkeypatch.setattr(coana, 'fetch_adjacencies',
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError('neuprint reached')))
    monkeypatch.setattr(coana, 'fetch_simple_connections',
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError('neuprint reached')))
    notes = []
    fc._vprint = lambda msg="", level="full", end="\n", flush=False: (
        notes.append(str(msg)))
    api_conn = fc._fetch_api_connections([upstream], None)
    assert api_conn is not None
    assert not any('codex.flywire.ai' in n for n in notes)


def test_fetch_api_connections_banc_total_miss_names_bucket_guidance():
    fc = _make_fc(client_type='banc', use_cache=True, cache_only=False,
                  _banc_local_conn_cache=None, dataset='banc_v999')
    notes = []
    fc._vprint = lambda msg="", level="full", end="\n", flush=False: (
        notes.append(str(msg)))
    result = fc._fetch_api_connections(['123'], None)
    assert result is None  # guidance + skip, never an online fetch
    joined = '\n'.join(notes)
    assert 'codex.flywire.ai' not in joined
    assert 'BANC' in joined


# ---------------------------------------------------------------------------
# _ensure_banc_connection_cache: honors the False return (BANC-02)
# ---------------------------------------------------------------------------

def test_cache_refresh_true_short_circuits(monkeypatch):
    import BANC_file_converter as bfc
    calls = {'build': 0, 'ensure': 0}
    monkeypatch.setattr(bfc, 'build_connection_cache_from_tables',
                        lambda d, c: calls.__setitem__('build', calls['build'] + 1) or True)
    monkeypatch.setattr(bfc, 'ensure_banc_data',
                        lambda ds, d: calls.__setitem__('ensure', calls['ensure'] + 1) or True)
    fc = _make_fc(client_type='banc', use_cache=False)
    fc._ensure_banc_connection_cache()
    assert calls == {'build': 1, 'ensure': 0}


def test_cache_refresh_false_triggers_ensure_and_rebuild(monkeypatch):
    import BANC_file_converter as bfc
    calls = {'build': 0, 'ensure': 0}

    def _build(dataset_dir, cache_dir):
        calls['build'] += 1
        return calls['build'] > 1  # False first, True after the ensure

    monkeypatch.setattr(bfc, 'build_connection_cache_from_tables', _build)
    monkeypatch.setattr(bfc, 'ensure_banc_data',
                        lambda ds, d: calls.__setitem__('ensure', calls['ensure'] + 1) or True)
    fc = _make_fc(client_type='banc', use_cache=False)
    fc._ensure_banc_connection_cache()
    assert calls == {'build': 2, 'ensure': 1}


def test_cache_refresh_absent_tables_exit_with_instructions(monkeypatch):
    import BANC_file_converter as bfc
    monkeypatch.setattr(bfc, 'build_connection_cache_from_tables',
                        lambda d, c: False)
    monkeypatch.setattr(bfc, 'ensure_banc_data', lambda ds, d: False)
    monkeypatch.setattr(coana, 'print_download_instructions',
                        lambda ds, d: None, raising=False)
    fc = _make_fc(client_type='banc', use_cache=False)
    # A library failure raises; it must not kill the host process.
    with pytest.raises(RuntimeError):
        fc._ensure_banc_connection_cache()
