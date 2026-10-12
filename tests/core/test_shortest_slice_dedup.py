"""Engine `_dedup_physical_edge_rows` slice-sum semantics (plan §10,
original-data adjudication 2026-10-12): shortest-mode fetch frames are
verbatim per-ROI slices refetched across BFS layers — the physical pair
weight is the slice SUM; the I1 keep-one-row dedup sampled a different
slice per layer."""
import polars as pl
import pytest

import coana  # noqa: E402


def _owner():
    return coana.FindNeuronConnection


@pytest.fixture(scope="module")
def helper():
    # the helper reads no instance state — a bare __new__ instance is safe
    return _owner().__new__(_owner())._dedup_physical_edge_rows


def _warn_holder():
    class _W:
        _warn_notes = []
    return _W()


def test_slice_sum_polars(helper):
    holder = _warn_holder()
    frame = pl.DataFrame({
        "bodyId_pre": ["a1", "a1", "a1", "a1", "b1"],
        "bodyId_post": ["b1", "b1", "b1", "b1", "c1"],
        "roi": ["A", "B", "A", "B", "A"],
        "weight": [2, 7, 2, 7, 3],          # L0 slices + L1 refetches
        "conn_layer": ["L0", "L0", "L1", "L1", "L1"],
    })
    out = helper(frame, "test")
    got = {tuple(r[:2]): r[2] for r in out.select(
        "bodyId_pre", "bodyId_post", "weight").iter_rows()}
    # a1->b1 = 2 + 7 (distinct ROI slices; refetch duplicates collapse)
    assert got[("a1", "b1")] == 9
    assert got[("b1", "c1")] == 3
    assert holder._warn_notes == []


def test_slice_sum_with_ratio_recompute(helper):
    # two slices of one pair, same denominator (w/r = 10), refetched in
    # a second layer: summed weight 3+7=10 must keep ratio 10/10 = 1.0
    frame = pl.DataFrame({
        "bodyId_pre": ["a1", "a1", "a1"],
        "bodyId_post": ["b1", "b1", "b1"],
        "roi": ["A", "B", "A"],
        "weight": [3, 7, 3],
        "weight_ratio": [0.3, 0.7, 0.3],
        "conn_layer": ["L0", "L0", "L1"],
    })
    out = helper(frame, "test")
    row = out.row(0, named=True)
    assert row["weight"] == 10
    assert row["weight_ratio"] == pytest.approx(1.0)


def test_slice_sum_pandas(helper):
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame({
        "bodyId_pre": ["a1", "a1", "a1"],
        "bodyId_post": ["b1", "b1", "b1"],
        "roi": ["A", "B", "A"],
        "weight": [2, 7, 2],
        "conn_layer": ["L0", "L0", "L1"],
    })
    out = helper(frame, "test")
    assert len(out) == 1
    assert out.iloc[0]["weight"] == 9
