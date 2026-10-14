#!/usr/bin/env python
"""
Regression tests for the FindAllPath memory fix (2026-08 ArrowMemoryError).

Covers:
  - Connection-layer column trimming (_PATH_CONN_KEEP_COLS contract) so the
    pandas -> Polars conversion no longer duplicates enrichment-only width
  - The Polars-native fetch pipeline (return_polars=True) producing the same
    rows/columns as the pandas pipeline for cached, mixed, filtered,
    hemisphere-separated, label-mapped, and bodyId-thresholded runs
  - Memory hygiene of the discovery loops via the shared conversion helper
"""

import sys
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))

import coana  # noqa: E402
from coana import _PATH_CONN_KEEP_COLS  # noqa: E402


def _make_fc(**attrs):
    """Minimal copy of the make_fc convention in test_coana_coverage."""
    fc = object.__new__(coana.FindNeuronConnection)
    fc._vprint = lambda msg="", level="full", end="\n", flush=False: None
    fc.verbose_mode = "silent"
    fc.progress_events = False
    fc._warn_notes = []
    for key, value in attrs.items():
        setattr(fc, key, value)
    return fc


def _neuron_info(extra_cols=None):
    info = pd.DataFrame({
        "bodyId": ["1", "2", "3", "4"],
        "type": ["TA", "TB", None, "TD"],
        "instance": ["TA_L", "TB_R", None, "TD_R"],
        "hemisphere": ["left", "right", "", "right"],
        "hemisphere_code": ["L", "R", "U", "R"],
    })
    if extra_cols:
        info = info.merge(extra_cols, on="bodyId", how="left")
    return info


def _cached_pl():
    # bodyIds are int64 on purpose: the connection DB stores them numeric
    return pl.DataFrame({
        "bodyId_pre": [1, 1, 2],
        "bodyId_post": [2, 3, 3],
        "weight": [5, 2, 8],
        "roi": ["ROIA", "ROIB", "ROIA"],
        "cached_date": ["2026-01-01"] * 3,
    })


def _api_pd():
    return pd.DataFrame({
        "bodyId_pre": [4, 4],
        "bodyId_post": [2, 3],
        "weight": [6, 1],
        "roi": ["ROIC", "ROIC"],
    })


def _make_fetch_fc(monkeypatch, cached, uncached, api_pd, total_incoming=None,
                   **attrs):
    defaults = dict(
        dataset="test:v1",
        use_cache=True,
        force_API_fetching=False,
        cache_only=False,
        label_mapper=None,
        separate_hemispheres=False,
        hemisphere_filter="none",
        exclude_intra_type_connections=False,
        filter_by="bodyid",
        aggregate_method="product",
        min_synapse_num=1,
        min_ratio=0.0,
        min_traversal_probability=0.0,
        script_path=str(PROJECT_ROOT),
        source_df=pd.DataFrame(columns=["bodyId"]),
        target_df=pd.DataFrame(columns=["bodyId"]),
    )
    defaults.update(attrs)
    fc = _make_fc(**defaults)
    monkeypatch.setattr(
        fc, "_query_connection_db",
        lambda upstream, downstream=None: (cached, list(uncached), []),
    )
    monkeypatch.setattr(
        fc, "_build_neuron_info_frame",
        lambda bodyids, is_flywire: _neuron_info(),
    )
    monkeypatch.setattr(
        fc, "_fetch_api_connections",
        lambda upstream, downstream=None: None if api_pd is None else api_pd.copy(),
    )
    monkeypatch.setattr(fc, "_save_connections_only", lambda *a, **k: None)
    marks = []
    monkeypatch.setattr(
        fc, "_mark_neurons_as_cached",
        lambda *a, **k: marks.append(a),
    )
    if total_incoming is not None:
        monkeypatch.setattr(
            fc, "_fetch_total_incoming_weight",
            lambda posts, min_weight: total_incoming.copy(),
        )
    return fc, marks


def _normalized(df):
    cols = [c for c in _PATH_CONN_KEEP_COLS if c in df.columns]
    df = df[cols].copy()
    for col in cols:
        if df[col].dtype == object:
            df[col] = df[col].where(df[col].notna(), None)
    return (
        df.sort_values(["bodyId_pre", "bodyId_post"], kind="mergesort")
        .reset_index(drop=True)
    )


def _assert_equivalent(pd_result, pl_result):
    assert isinstance(pl_result, pl.DataFrame)
    left = _normalized(pd_result)
    right = _normalized(pl_result.to_pandas())
    pd.testing.assert_frame_equal(left, right, check_dtype=False)


# =============================================================================
# Column trimming
# =============================================================================

class TestPathColumnTrim:
    def test_trim_keeps_required_drops_enrichment_only(self):
        fc = _make_fc()
        wide = pd.DataFrame({
            "bodyId_pre": ["1"], "bodyId_post": ["2"], "weight": [3],
            "roi": ["R"], "type_pre": ["TA"], "type_post": ["TB"],
            "instance_pre": ["i1"], "instance_post": ["i2"],
            "nt_type_pre": ["ACh"], "nt_type_post": ["GABA"],
            "hemisphere_pre": ["left"], "hemisphere_post": ["right"],
            "hemisphere_code_pre": ["L"], "hemisphere_code_post": ["R"],
            "std_label_pre": ["x"], "std_label_post": ["y"],
            "cached_date": ["d"],
        })
        trimmed = fc._trim_conn_df_for_path_discovery(wide)
        assert set(trimmed.columns) <= set(_PATH_CONN_KEEP_COLS)
        assert {"bodyId_pre", "bodyId_post", "weight",
                "type_pre", "type_post"} <= set(trimmed.columns)
        for dropped in ("hemisphere_pre", "hemisphere_code_post",
                        "nt_type_post", "std_label_pre", "cached_date"):
            assert dropped not in trimmed.columns

    def test_trim_tolerates_missing_columns(self):
        fc = _make_fc()
        bare = pd.DataFrame({
            "bodyId_pre": ["1"], "bodyId_post": ["2"], "weight": [3],
        })
        trimmed = fc._trim_conn_df_for_path_discovery(bare)
        assert list(trimmed.columns) == ["bodyId_pre", "bodyId_post", "weight"]

    def test_trim_on_canonical_empty_frame(self):
        fc = _make_fc()
        empty = fc._empty_path_connection_frame()
        trimmed = fc._trim_conn_df_for_path_discovery(empty)
        assert trimmed.empty

    def test_as_polars_conn_frame_pandas_path(self):
        fc = _make_fc()
        wide = pd.DataFrame({
            "bodyId_pre": [1, None], "bodyId_post": [2, 3], "weight": [3, 9],
            "type_pre": ["TA", None], "type_post": ["TB", "TD"],
            "hemisphere_pre": ["left", "left"],
        })
        out = fc._as_polars_conn_frame(wide)
        assert isinstance(out, pl.DataFrame)
        assert out["bodyId_pre"].dtype == pl.Utf8
        # [1, None] is float64 in pandas, so astype(str) yields '1.0';
        # the helper reproduces exactly that pandas parity
        assert out["bodyId_pre"].to_list() == ["1.0", "nan"]
        assert "hemisphere_pre" not in out.columns
        assert {"bodyId_pre", "bodyId_post", "weight",
                "type_pre", "type_post"} == set(out.columns)

    def test_as_polars_conn_frame_polars_input_normalized(self):
        fc = _make_fc()
        frame = pl.DataFrame({
            "bodyId_pre": [1], "bodyId_post": [2], "weight": [3],
            "cached_date": ["d"],
        })
        out = fc._as_polars_conn_frame(frame)
        assert out["bodyId_pre"].dtype == pl.Utf8
        assert "cached_date" not in out.columns


# =============================================================================
# Polars-native fetch pipeline equivalence
# =============================================================================

class TestPolarsFetchEquivalence:
    UPSTREAM = ["1", "2"]

    def _run_both(self, monkeypatch, min_weight=1, min_traversal_prob=0.0,
                  min_conn_ratio=0.0, **attrs):
        cached, uncached, api, totals = (
            attrs.pop("cached", _cached_pl()),
            attrs.pop("uncached", []),
            attrs.pop("api", _api_pd()),
            attrs.pop("totals", None),
        )
        fc, marks = _make_fetch_fc(
            monkeypatch, cached, uncached, api, totals, **attrs
        )
        pd_result = fc._fetch_connections_with_cache(
            list(self.UPSTREAM), None,
            min_weight=min_weight,
            min_traversal_prob=min_traversal_prob,
            min_conn_ratio=min_conn_ratio,
        )
        pl_result = fc._fetch_connections_with_cache(
            list(self.UPSTREAM), None,
            min_weight=min_weight,
            min_traversal_prob=min_traversal_prob,
            min_conn_ratio=min_conn_ratio,
            return_polars=True,
        )
        return fc, marks, pd_result, pl_result

    def test_fully_cached(self, monkeypatch):
        _, _, pd_result, pl_result = self._run_both(monkeypatch)
        _assert_equivalent(pd_result, pl_result)
        # cached_date is DB bookkeeping and must not leak into either
        # trimmed result
        assert "cached_date" not in _normalized(pd_result).columns

    def test_partial_cache_with_weight_filter(self, monkeypatch):
        _, _, pd_result, pl_result = self._run_both(
            monkeypatch, uncached=["4"], min_weight=3,
        )
        _assert_equivalent(pd_result, pl_result)
        kept_weights = sorted(pl_result["weight"].to_list())
        assert kept_weights == [5, 6, 8]  # weight 1 and 2 filtered out

    def test_empty_result_returns_polars_frame(self, monkeypatch):
        _, _, pd_result, pl_result = self._run_both(
            monkeypatch, cached=pl.DataFrame(), uncached=["9"],
            api=pd.DataFrame(),
        )
        assert isinstance(pl_result, pl.DataFrame)
        assert pl_result.is_empty()
        assert isinstance(pd_result, pd.DataFrame)
        assert pd_result.empty

    def test_fafb_abort_signal_returns_polars_empty(self, monkeypatch):
        _, _, pd_result, pl_result = self._run_both(
            monkeypatch, cached=pl.DataFrame(), uncached=["9"], api=None,
        )
        assert isinstance(pl_result, pl.DataFrame)
        assert pl_result.is_empty()
        assert isinstance(pd_result, pd.DataFrame)

    def test_marking_receives_enriched_columns(self, monkeypatch):
        _, marks, _, _ = self._run_both(monkeypatch, uncached=["4"])
        assert marks, "polars path must still mark fetched neurons cached"
        connections = marks[0][1]
        for col in ("bodyId_pre", "bodyId_post", "weight",
                    "type_pre", "instance_pre"):
            assert col in connections.columns

    def test_bodyid_level_filters(self, monkeypatch):
        totals = pd.DataFrame({
            "bodyId_post": ["2", "3"],
            "total_incoming_weight": [10, 4],
        })
        _, _, pd_result, pl_result = self._run_both(
            monkeypatch, uncached=["4"], min_traversal_prob=0.5,
            totals=totals,
        )
        _assert_equivalent(pd_result, pl_result)
        assert "connection_ratio" in pl_result.columns
        # PROBABILITY SUSPENSION (2026-10-06): traversal_probability is
        # no longer exported; min_traversal_probability filters degrade
        # to no-ops on the suspended column.
        assert "traversal_probability" not in pl_result.columns
        # ratio = weight/total. Rows from
        # post=2: 5/10=0.5; 6/10=0.6.
        # post=3: 2/4=0.5; 8/4=2; 1/4=0.25
        # -> 0.83 kept.  A 0.5 prob floor keeps everything here, so make it
        # stricter and compare against the pandas outcome explicitly.
        _, _, pd_strict, pl_strict = self._run_both(
            monkeypatch, uncached=["4"], min_traversal_prob=0.9,
            totals=totals,
        )
        _assert_equivalent(pd_strict, pl_strict)

    def test_separate_hemispheres(self, monkeypatch):
        _, _, pd_result, pl_result = self._run_both(
            monkeypatch, uncached=["4"], separate_hemispheres=True,
        )
        _assert_equivalent(pd_result, pl_result)
        # pre neurons 1/2/4 carry codes L/R/R -> suffixes appended
        assert set(pl_result["type_pre"].to_list()) == {
            "TA_L", "TB_R", "TD_R",
        }

    def test_label_mapper_fallback(self, monkeypatch):
        class FakeMapper:
            """Mapper groups keyed by native type name (the real shape).

            The fetch lane types its frame through the run's registry, not
            through ``apply_to_dataframe``: both engines must reach the same
            keys, and only the registry can also express a `T*` residual.
            """

            is_empty = False

            def __init__(self):
                groups = {f"std_{t}": {"test:v1": [t]}
                          for t in ("TA", "TB", "TD")}
                self._source_mapping = groups
                self._target_mapping = groups
                self._intermediate_mapping = {}

            def _dataset_keys(self, dataset):
                return [dataset]

        _, _, pd_result, pl_result = self._run_both(
            monkeypatch, uncached=["4"], label_mapper=FakeMapper(),
        )
        _assert_equivalent(pd_result, pl_result)
        # the effective key lives in type_pre; no std_label twin is exported
        assert "std_label_pre" not in pl_result.columns
        assert set(pl_result["type_pre"].to_list()) == {
            "std_TA", "std_TB", "std_TD",
        }
        # ...and the raw native type survives beside it for provenance (D2)
        assert set(str(v) for v in pl_result["native_type_pre"].to_list()) <= {
            "TA", "TB", "TD", "None"}

    def test_intra_type_exclusion(self, monkeypatch):
        # 1->1 style self loop via same type on both endpoints: give
        # neurons 2 and 4 the same type so pre=TA/post=TA pairs exist.
        info = _neuron_info()
        info.loc[info["bodyId"] == "2", "type"] = "TA"

        fc, marks = _make_fetch_fc(
            monkeypatch, _cached_pl(), ["4"], _api_pd(), None,
            exclude_intra_type_connections=True,
        )
        monkeypatch.setattr(
            fc, "_build_neuron_info_frame",
            lambda bodyids, is_flywire: info.copy(),
        )
        pd_result = fc._fetch_connections_with_cache(
            list(self.UPSTREAM), None,
            min_weight=1, min_traversal_prob=0.0, min_conn_ratio=0.0,
        )
        pl_result = fc._fetch_connections_with_cache(
            list(self.UPSTREAM), None,
            min_weight=1, min_traversal_prob=0.0, min_conn_ratio=0.0,
            return_polars=True,
        )
        _assert_equivalent(pd_result, pl_result)
        # 1->2 (TA->TA after the retype) must be excluded on both paths,
        # while null-type edges (x->3) are always kept
        kept_pairs = set(
            zip(pl_result["bodyId_pre"].to_list(),
                pl_result["bodyId_post"].to_list())
        )
        assert ("1", "2") not in kept_pairs
        assert kept_pairs == {("1", "3"), ("2", "3"), ("4", "2"), ("4", "3")}

    def test_type_level_filter_fallback(self, monkeypatch):
        totals = pd.DataFrame({
            "type_post": ["TB", "TD"],
            "total_incoming_weight": [100, 100],
        })

        def fake_type_filters(combined, min_weight, min_conn_ratio,
                              min_traversal_prob, total_before_filter,
                              aggregate_method="product"):
            return combined[combined["weight"] >= min_weight].copy()

        cached, uncached, api = _cached_pl(), ["4"], _api_pd()
        fc, marks = _make_fetch_fc(
            monkeypatch, cached, uncached, api, None, filter_by="type",
        )
        monkeypatch.setattr(
            fc, "_apply_type_level_filters", fake_type_filters
        )
        pd_result = fc._fetch_connections_with_cache(
            list(self.UPSTREAM), None,
            min_weight=3, min_traversal_prob=0.0, min_conn_ratio=0.0,
        )
        pl_result = fc._fetch_connections_with_cache(
            list(self.UPSTREAM), None,
            min_weight=3, min_traversal_prob=0.0, min_conn_ratio=0.0,
            return_polars=True,
        )
        _assert_equivalent(pd_result, pl_result)


# =============================================================================
# Plumbing
# =============================================================================

class TestFetchPlumbing:
    def test_fetch_path_connections_forwards_return_polars(self):
        captured = {}

        def fake_fetch(**kwargs):
            captured.update(kwargs)
            return "SENTINEL"

        fc = _make_fc(
            min_synapse_num=7, min_ratio=0.25, min_traversal_probability=0.5,
        )
        fc._fetch_connections_with_cache = fake_fetch
        out = fc._fetch_path_connections(
            upstream_bodyIds=["1"], return_polars=True,
        )
        assert out == "SENTINEL"
        assert captured["return_polars"] is True
        assert captured["min_weight"] == 7
        assert captured["min_conn_ratio"] == 0.25
        assert captured["min_traversal_prob"] == 0.5

    def test_fetch_path_connections_defaults_to_pandas(self):
        captured = {}

        def fake_fetch(**kwargs):
            captured.update(kwargs)
            return pd.DataFrame()

        fc = _make_fc(
            min_synapse_num=1, min_ratio=0.0, min_traversal_probability=0.0,
        )
        fc._fetch_connections_with_cache = fake_fetch
        fc._fetch_path_connections(upstream_bodyIds=["1"])
        assert captured["return_polars"] is False

    def test_keep_cols_constant_contract(self):
        for col in ("bodyId_pre", "bodyId_post", "weight",
                    "type_pre", "type_post"):
            assert col in _PATH_CONN_KEEP_COLS
        # output pass-through columns stay so saved CSV schemas do not change
        assert "roi" in _PATH_CONN_KEEP_COLS
        assert "instance_pre" in _PATH_CONN_KEEP_COLS
        assert "instance_post" in _PATH_CONN_KEEP_COLS
