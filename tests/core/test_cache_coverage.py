"""Cache-only integrity gates (Defect A from the 2026-09-12 Windows report,
extended per the 2026-09-16 re-test findings F2/F4).

A cache whose rows were truncated (or whose fetches never finished) passes
every file-presence check, so cache-only runs must verify:

- per-neuron coverage: the neuron index's recorded ``connection_count``
  against the DEDUPLICATED connection rows actually present (the loader
  dedups on ``(bodyId_pre, bodyId_post, roi)`` — counting raw rows gives a
  ~2x cushion on consolidated caches);
- whole-cache coverage: on-disk distinct connections vs the
  ``cache_manifest.json`` baseline written at consolidation / first online
  run — the only detector for loss outside the flagged-neuron subset.

Runs without any problem pass; on a problem the run is refused unless
``allow_incomplete_cache`` opted in (warnings + INCOMPLETE_CACHE.txt
stamping).  Mirrors the report's §2.3 experiment: pristine cache -> 0
mismatches; truncated cache -> mismatches.
"""

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

import coana
from coana import cache_coverage_mismatches


@pytest.fixture(autouse=True)
def _clear_coverage_memo():
    coana._CACHE_COVERAGE_CACHE.clear()
    yield
    coana._CACHE_COVERAGE_CACHE.clear()


# ---------------------------------------------------------------------------
# Pure helper
# ---------------------------------------------------------------------------

def test_pristine_cache_yields_no_mismatches():
    rows = [(100, True, 189), (101, True, 0), (102, False, 500)]
    counts = {100: 189, 101: 0, 102: 3}
    mismatches, complete_flagged = cache_coverage_mismatches(rows, counts)
    assert mismatches == []
    assert complete_flagged == 2


def test_truncated_cache_is_detected():
    rows = [(100, True, 189), (101, True, 125), (102, True, 13)]
    counts = {100: 12, 101: 6, 102: 0}
    mismatches, complete_flagged = cache_coverage_mismatches(rows, counts)
    assert complete_flagged == 3
    assert mismatches == [('100', 189, 12), ('101', 125, 6), ('102', 13, 0)]


def test_extra_cached_rows_are_not_flagged():
    # Duplicated main+batch rows can only inflate counts; only
    # cached < recorded signals a truncated cache.
    rows = [(100, True, 10)]
    counts = {100: 15}
    assert cache_coverage_mismatches(rows, counts) == ([], 1)


def test_missing_neuron_counts_as_zero_cached():
    rows = [(100, True, 13)]
    mismatches, _ = cache_coverage_mismatches(rows, {})
    assert mismatches == [('100', 13, 0)]


# ---------------------------------------------------------------------------
# Instance-level _check_cache_coverage over a synthetic cache
# ---------------------------------------------------------------------------

def _write_cache(tmp_path, connection_rows, index_rows, with_dataset=True):
    dataset = 'test:v1.0'
    dataset_safe = coana.dataset_folder(dataset)
    cache_dir = tmp_path / 'cache' / dataset_safe
    cache_dir.mkdir(parents=True)
    pl.DataFrame({
        'bodyId_pre': [r[0] for r in connection_rows],
        'bodyId_post': [r[1] for r in connection_rows],
        'weight': [r[2] for r in connection_rows],
    }).write_parquet(cache_dir / 'connections.parquet')
    index_dir = tmp_path / 'neuron_indexes' / dataset_safe
    index_dir.mkdir(parents=True)
    pl.DataFrame({
        'bodyId': [r[0] for r in index_rows],
        'downstream_complete': [r[1] for r in index_rows],
        'connection_count': [r[2] for r in index_rows],
    }).write_parquet(index_dir / 'neuron_index.parquet')
    if with_dataset:
        ds_dir = tmp_path / 'datasets' / dataset_safe
        ds_dir.mkdir(parents=True)
        pd.DataFrame({'bodyId': [1], 'type': ['T']}).to_parquet(
            ds_dir / f'{dataset_safe}_allneurons_neuron_df.parquet')
    return dataset, dataset_safe


def _make_fc(tmp_path, dataset, allow_incomplete_cache=False, notes=None):
    fc = object.__new__(coana.FindNeuronConnection)
    fc.script_path = str(tmp_path)
    fc.dataset = dataset
    fc._dataset_safe = coana.dataset_folder(dataset)
    fc.allow_incomplete_cache = allow_incomplete_cache
    fc.use_cache = True
    fc._vprint = (lambda message, level='full', **k: notes.append(message)) \
        if notes is not None else (lambda *a, **k: None)
    return fc


def _write_manifest(tmp_path, dataset, distinct):
    dataset_safe = coana.dataset_folder(dataset)
    path = tmp_path / 'cache' / dataset_safe / 'cache_manifest.json'
    path.write_text(json.dumps({
        'schema': 1, 'dataset': dataset, 'distinct_connections': distinct,
        'built_at': '2026-09-17T00:00:00', 'source': 'test'}))


def test_check_cache_coverage_pristine(tmp_path):
    dataset, _ = _write_cache(
        tmp_path,
        connection_rows=[(100, 200 + i, 5) for i in range(189)],
        index_rows=[(100, True, 189), (101, True, 0)],
    )
    fc = _make_fc(tmp_path, dataset)
    coverage = fc._check_cache_coverage()
    assert coverage is not None
    assert coverage['mismatches'] == []
    assert coverage['complete_flagged'] == 2
    assert coverage['distinct_connections'] == 189
    assert coverage['manifest'] is None


def test_check_cache_coverage_truncated(tmp_path):
    dataset, _ = _write_cache(
        tmp_path,
        connection_rows=(
            [(100, 200 + i, 10) for i in range(12)]
            + [(101, 300 + i, 10) for i in range(6)]
            + [(102, 400 + i, 10) for i in range(3)]),
        index_rows=[(100, True, 189), (101, True, 125), (102, False, 500)],
    )
    fc = _make_fc(tmp_path, dataset)
    coverage = fc._check_cache_coverage()
    assert coverage['mismatches'] == [('100', 189, 12), ('101', 125, 6)]
    # Only complete-flagged neurons are counted (102 is ignored).
    assert coverage['complete_flagged'] == 2


def test_check_cache_coverage_none_when_cache_missing(tmp_path):
    fc = _make_fc(tmp_path, 'test:v1.0')
    assert fc._check_cache_coverage() is None


def test_check_cache_coverage_memoized_until_files_change(tmp_path):
    dataset, _ = _write_cache(
        tmp_path,
        connection_rows=[(100, 200, 5)],
        index_rows=[(100, True, 189)],
    )
    fc = _make_fc(tmp_path, dataset)
    first = fc._check_cache_coverage()
    # Touch the connections file: the signature changes and the verdict is
    # recomputed instead of served from the memo.
    (tmp_path / 'cache' / coana.dataset_folder(dataset) / 'connections.parquet') \
        .write_bytes(
            (tmp_path / 'cache' / coana.dataset_folder(dataset)
             / 'connections.parquet').read_bytes())
    second = fc._check_cache_coverage()
    assert first == second


# ---------------------------------------------------------------------------
# Gate behavior
# ---------------------------------------------------------------------------

def test_enforce_refuses_incomplete_cache(tmp_path):
    dataset, _ = _write_cache(
        tmp_path,
        connection_rows=[(100, 200 + i, 10) for i in range(12)],
        index_rows=[(100, True, 189)],
    )
    fc = _make_fc(tmp_path, dataset)
    with pytest.raises(RuntimeError, match='incomplete'):
        fc._enforce_cache_coverage('user_requested')


def test_enforce_refuses_legacy_cache_without_manifest(tmp_path):
    """A cache without an integrity manifest cannot be verified whole-cache;
    cache-only runs are refused until an online run writes the baseline."""
    dataset, _ = _write_cache(
        tmp_path,
        connection_rows=[(100, 200 + i, 5) for i in range(189)],
        index_rows=[(100, True, 189)],
    )
    fc = _make_fc(tmp_path, dataset)
    with pytest.raises(RuntimeError, match='no integrity manifest'):
        fc._enforce_cache_coverage('user_requested')


def test_enforce_passes_complete_cache_with_matching_manifest(tmp_path):
    dataset, _ = _write_cache(
        tmp_path,
        connection_rows=[(100, 200 + i, 5) for i in range(189)],
        index_rows=[(100, True, 189)],
    )
    _write_manifest(tmp_path, dataset, distinct=189)
    fc = _make_fc(tmp_path, dataset)
    fc._enforce_cache_coverage('user_requested')  # must not raise


def test_enforce_passes_when_cache_grew_beyond_manifest(tmp_path):
    """Batch files appended after the last consolidation grow the cache;
    growth is never a refusal."""
    dataset, _ = _write_cache(
        tmp_path,
        connection_rows=[(100, 200 + i, 5) for i in range(189)],
        index_rows=[(100, True, 189)],
    )
    _write_manifest(tmp_path, dataset, distinct=100)
    fc = _make_fc(tmp_path, dataset)
    fc._enforce_cache_coverage('user_requested')  # must not raise


def test_enforce_refuses_on_manifest_loss(tmp_path):
    """The field T6 shape: flagged neurons' rows survive in batch files while
    the rest of the cache is truncated — the manifest layer is the only
    detector (re-test finding F2)."""
    dataset, _ = _write_cache(
        tmp_path,
        connection_rows=[(100, 200 + i, 10) for i in range(12)],
        index_rows=[(100, True, 12)],  # flagged subset intact
    )
    _write_manifest(tmp_path, dataset, distinct=1_000_000)
    fc = _make_fc(tmp_path, dataset)
    with pytest.raises(RuntimeError, match='integrity manifest records'):
        fc._enforce_cache_coverage('user_requested')


def test_enforce_error_names_remediation(tmp_path):
    dataset, _ = _write_cache(
        tmp_path,
        connection_rows=[(100, 200 + i, 10) for i in range(12)],
        index_rows=[(100, True, 189)],
    )
    fc = _make_fc(tmp_path, dataset)
    with pytest.raises(RuntimeError) as excinfo:
        fc._enforce_cache_coverage('server_unavailable')
    message = str(excinfo.value)
    assert 'allow_incomplete_cache=True' in message
    # The UI remedy is spelled out because the opt-in flag is library-only
    # (re-test finding F4).
    assert 'Cache Only' in message


def test_enforce_allows_opted_in_partial_run(tmp_path):
    dataset, _ = _write_cache(
        tmp_path,
        connection_rows=[(100, 200 + i, 10) for i in range(12)],
        index_rows=[(100, True, 189)],
    )
    notes = []
    fc = _make_fc(tmp_path, dataset, allow_incomplete_cache=True, notes=notes)
    fc._enforce_cache_coverage('user_requested')  # must not raise
    assert any('allow_incomplete_cache=True' in n for n in notes)
    # The summary is stored for the INCOMPLETE_CACHE.txt stamp.
    assert 'flagged complete' in fc._cache_incomplete_summary


def test_ensure_cache_manifest_writes_baseline_once(tmp_path):
    dataset, _ = _write_cache(
        tmp_path,
        connection_rows=[(100, 200 + i, 5) for i in range(189)],
        index_rows=[(100, True, 189)],
    )
    fc = _make_fc(tmp_path, dataset)
    assert fc._load_cache_manifest() is None
    fc._ensure_cache_manifest()
    manifest = fc._load_cache_manifest()
    assert manifest is not None
    assert manifest['distinct_connections'] == 189
    assert manifest['source'] == 'baseline'
    # Second call is a no-op: the baseline is never refreshed silently.
    built_at = manifest['built_at']
    fc._ensure_cache_manifest()
    assert fc._load_cache_manifest()['built_at'] == built_at


def test_incomplete_marker_carries_coverage_numbers(tmp_path):
    """A stamped partial run documents HOW incomplete it was (F4)."""
    dataset, _ = _write_cache(
        tmp_path,
        connection_rows=[(100, 200 + i, 10) for i in range(12)],
        index_rows=[(100, True, 189)],
    )
    fc = _make_fc(tmp_path, dataset, allow_incomplete_cache=True)
    fc.cache_only = True
    fc._warn_notes = []
    fc._enforce_cache_coverage('user_requested')
    fc._handle_cache_only_miss([512925, 72227], has_cached=False)
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    fc._write_user_warning_notes(str(run_dir))
    marker = (run_dir / 'INCOMPLETE_CACHE.txt').read_text(encoding='utf-8')
    assert 'Coverage:' in marker
    assert 'flagged complete' in marker
    assert '2 neuron(s) absent' in marker


def test_handle_cache_only_miss_refuses_without_opt_in():
    fc = _make_fc(tmp_path_factory(), 'test:v1.0')
    with pytest.raises(RuntimeError, match='not in the local cache'):
        fc._handle_cache_only_miss([512925, 72227], has_cached=False)


def test_handle_cache_only_miss_warns_with_opt_in():
    notes = []
    fc = _make_fc(tmp_path_factory(), 'test:v1.0',
                  allow_incomplete_cache=True, notes=notes)
    fc._handle_cache_only_miss([512925], has_cached=False)
    assert any('not in cache' in n for n in notes)
    assert any('No cached connections' in n for n in notes)


def tmp_path_factory():
    import tempfile
    return tempfile.mkdtemp()


# ---------------------------------------------------------------------------
# is_usable now requires the dataset table
# ---------------------------------------------------------------------------

def test_check_cache_exists_requires_dataset_table(tmp_path):
    dataset, dataset_safe = _write_cache(
        tmp_path,
        connection_rows=[(100, 200, 5)],
        index_rows=[(100, True, 1)],
        with_dataset=False,
    )
    fc = _make_fc(tmp_path, dataset)
    status = fc._check_cache_exists()
    assert status['has_connections'] and status['has_neuron_index']
    assert not status['has_dataset']
    assert not status['is_usable']


def test_check_cache_exists_usable_with_dataset_table(tmp_path):
    dataset, dataset_safe = _write_cache(
        tmp_path,
        connection_rows=[(100, 200, 5)],
        index_rows=[(100, True, 1)],
        with_dataset=True,
    )
    fc = _make_fc(tmp_path, dataset)
    status = fc._check_cache_exists()
    assert status['is_usable'] is True
