"""Pair-collapsed main connection cache (plan §12.3, 2026-10-12 root
fix for the per-ROI slice-count defect class): consolidation emits ONE
row per (pre, post) at the pair's physical (slice-summed) weight plus a
connections_by_roi.parquet sidecar; the D_t map skips stale batches on
the pair-level layout (their rows are already summed into the main)."""
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vispath-subproject" / "src"))

import coana  # noqa: E402
from connection_map import ThresholdedConnectionMap  # noqa: E402


def _fc_for(tmp_path):
    fc = coana.FindNeuronConnection.__new__(coana.FindNeuronConnection)
    fc.use_cache = True
    fc.dataset = "male-cns:v1.0"
    fc._get_connection_db_path = lambda: str(tmp_path / "connections.parquet")
    fc._invalidate_connection_memory_cache = lambda: None
    fc._write_cache_manifest = lambda source: None
    return fc


def _write_batch(cache_dir, name, rows):
    batch_dir = cache_dir / "_batch_files"
    batch_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(batch_dir / name)


def test_consolidation_emits_pair_level_main_and_sidecar(tmp_path):
    fc = _fc_for(tmp_path)
    _write_batch(tmp_path, "batch_000000.parquet", {
        "bodyId_pre": ["A", "A", "B"],
        "bodyId_post": ["B", "B", "C"],
        "roi": ["R1", "R2", "R1"],
        "weight": [2, 5, 3]})
    # a second batch re-listing one row verbatim (cross-batch duplicate)
    _write_batch(tmp_path, "batch_000001.parquet", {
        "bodyId_pre": ["B"],
        "bodyId_post": ["C"],
        "roi": ["R1"],
        "weight": [3]})
    total = fc._consolidate_batch_files(deduplicate=True)
    assert total == 2  # two PAIRS
    main = pl.read_parquet(tmp_path / "connections.parquet")
    got = {(r[0], r[1]): r[2] for r in main.select(
        "bodyId_pre", "bodyId_post", "weight").iter_rows()}
    assert got == {("A", "B"): 7, ("B", "C"): 3}
    assert set(main.columns) >= {"bodyId_pre", "bodyId_post", "weight"}
    # sidecar: the dedup'd per-ROI rows verbatim (the audit surface)
    side = pl.read_parquet(tmp_path / "connections_by_roi.parquet")
    srows = {(r[0], r[1], r[2]): r[3] for r in side.select(
        "bodyId_pre", "bodyId_post", "roi", "weight").iter_rows()}
    assert srows == {("A", "B", "R1"): 2, ("A", "B", "R2"): 5,
                     ("B", "C", "R1"): 3}
    # batches cleaned up
    assert not (tmp_path / "_batch_files").exists()


def test_dt_map_skips_stale_batches_on_pair_level_layout(tmp_path):
    """The crash window with the pair-level layout: the main file is
    written before the batch dir is removed, so batches coexisting with
    it are consolidation leftovers — their slices are already summed
    into the main and must NOT double-count the D_t totals."""
    fc = _fc_for(tmp_path)
    _write_batch(tmp_path, "batch_000000.parquet", {
        "bodyId_pre": ["A", "A", "B"],
        "bodyId_post": ["B", "B", "C"],
        "roi": ["R1", "R2", "R1"],
        "weight": [2, 5, 3]})
    fc._consolidate_batch_files(deduplicate=True)
    # simulate the crash window: resurrect a stale batch AFTER consolidating
    _write_batch(tmp_path, "batch_000001.parquet", {
        "bodyId_pre": ["A", "A", "B"],
        "bodyId_post": ["B", "B", "C"],
        "roi": ["R1", "R2", "R1"],
        "weight": [2, 5, 3]})
    cm = ThresholdedConnectionMap(
        str(tmp_path / "connections.parquet"),
        str(tmp_path / "neuron_index.parquet"), min_weight=1)
    idx = pl.DataFrame({"bodyId": ["A", "B", "C"],
                        "type": ["TA", "TB", "TC"]})
    idx.write_parquet(tmp_path / "neuron_index.parquet")
    body = cm.total_incoming_by_bodyid()
    got = dict(body.iter_rows())
    assert got == {"B": 7, "C": 3}  # no double-count


def test_dt_map_includes_batches_without_main(tmp_path):
    """A fetch interrupted before the first consolidation: the batch
    files are the only data and must be read."""
    fc = _fc_for(tmp_path)
    _write_batch(tmp_path, "batch_000000.parquet", {
        "bodyId_pre": ["A", "A"],
        "bodyId_post": ["B", "B"],
        "roi": ["R1", "R2"],
        "weight": [2, 5]})
    pl.DataFrame({"bodyId": ["A", "B"], "type": ["TA", "TB"]}).write_parquet(
        tmp_path / "neuron_index.parquet")
    cm = ThresholdedConnectionMap(
        str(tmp_path / "connections.parquet"),
        str(tmp_path / "neuron_index.parquet"), min_weight=1)
    body = cm.total_incoming_by_bodyid()
    assert dict(body.iter_rows()) == {"B": 7}
