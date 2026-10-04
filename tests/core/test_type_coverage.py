"""Hermetic tests for the union type-resolution coverage pass.

The coverage pass (comparison/type_coverage.py) resolves the union of
types that appeared in ANY dataset into EVERY dataset and classifies the
absences.  All local data access (neuron tables, connection cache,
searched-graph lists) is monkeypatched; no dataset folder or network is
touched.  auto_type_mapping=False throughout, so resolution runs the
same-name lane and resolver verdicts are simulated via patched
``resolve_valid_targets`` where needed.
"""

import pandas as pd
import pytest

from comparison import type_coverage as tc
from comparison.comparison_analyzer import ComparisonAnalyzer
from comparison.comparison_parameters import ComparisonParameters
from comparison.type_resolver import (
    STATUS_CONFLICT,
    STATUS_UNMAPPED,
    TypeResolution,
)

DS1 = "hemibrain:v1.2.1"   # recruits every type in the scenario
DS2 = "male-cns:v0.9"      # lacks APL / Ghost / Island / Late

SAFE1 = "hemibrain_v1_2_1"
SAFE2 = "male-cns_v0_9"

_PARAMS_OUTPUT = {"folder": ""}


@pytest.fixture(autouse=True)
def _params_write_under_tmp(tmp_path):
    _PARAMS_OUTPUT["folder"] = str(tmp_path)


@pytest.fixture(autouse=True)
def _isolate_type_coverage_caches():
    tc.clear_caches()
    yield
    tc.clear_caches()


@pytest.fixture(autouse=True)
def _no_real_local_data(monkeypatch):
    """Default the loaders to empty: dataset names like hemibrain:v1.2.1
    resolve to REAL local tables/caches in the repo, which unit tests must
    never read.  ``coverage_env`` re-patches with data-returning fakes."""
    monkeypatch.setattr(tc, "load_neuron_table", lambda root, ds: None)
    monkeypatch.setattr(tc, "load_connection_frame", lambda root, ds: None)
    monkeypatch.setattr(tc, "load_in_graph_roles", lambda folder: None)


def _params(**overrides):
    defaults = dict(
        datasets=[DS1, DS2],
        source_neurons=["Src"],
        target_neurons=["Tgt"],
        max_interlayer=1,
        thresholds=[3],
        output_folder=_PARAMS_OUTPUT["folder"],
        auto_type_mapping=False,
        verbose=False,
    )
    defaults.update(overrides)
    return ComparisonParameters(**defaults)


@pytest.fixture
def analyzer():
    return ComparisonAnalyzer(_params(), verbose=False)


def _edge_df(edges):
    return pd.DataFrame(edges, columns=["type_pre", "type_post", "weight"])


def _results():
    """DS1 surfaces five types; DS2 only Src -> Tgt.  "Late(Hold)" is a
    merged-display canonical name — its alternates never exist in any
    neuron table, so classification must run on the base name."""
    d1 = _edge_df([
        ("Src", "Tgt", 10),
        ("APL", "Tgt", 5),
        ("Ghost", "Tgt", 2),
        ("Island", "Tgt", 3),
        ("Late(Hold)", "Tgt", 4),
    ])
    d2 = _edge_df([("Src", "Tgt", 4)])
    return {DS1: {3: d1.copy()}, DS2: {3: d2.copy()}}


# type -> bodyIds per dataset; bodyIds never overlap across datasets.
_TABLES = {
    DS1: pd.DataFrame({
        "bodyId": [101, 102, 103, 104, 105, 106],
        "type": ["Src", "Tgt", "APL", "Ghost", "Island", "Late"],
    }),
    DS2: pd.DataFrame({
        "bodyId": [201, 202, 203, 204, 205],
        "type": ["Src", "Tgt", "APL", "Late", "Island"],
    }),
}

# Edges between the searched graph and candidate types:
# DS2 APL(203) receives weight 2 from Src (below threshold 3) and feeds
# back onto Src with weight 50 — the feedback edge must NOT upgrade the
# verdict (recruitment depends on the ENTRY leg direction);
# DS2 Late(204) receives weight 9 (>= threshold) yet was not recruited;
# DS2 Island(205) has no edges at all.
_CONNS = {
    DS1: pd.DataFrame({
        "bodyId_pre": ["101", "101", "101", "101", "101"],
        "bodyId_post": ["102", "103", "104", "105", "106"],
        "weight": [10, 8, 2, 3, 4],
    }),
    DS2: pd.DataFrame({
        "bodyId_pre": ["201", "201", "201", "203"],
        "bodyId_post": ["202", "203", "204", "201"],
        "weight": [4, 2, 9, 50],
    }),
}

_IN_GRAPH = {
    DS1: {'source': {'101'}, 'intermediate': {'102', '103', '104', '105', '106'},
          'target': set()},
    DS2: {'source': {'201'}, 'intermediate': {'202'}, 'target': set()},
}


@pytest.fixture
def coverage_env(monkeypatch):
    monkeypatch.setattr(
        tc, "load_neuron_table", lambda root, ds: _TABLES.get(ds))
    monkeypatch.setattr(
        tc, "load_connection_frame", lambda root, ds: _CONNS.get(ds))

    def _in_graph(folder):
        if "hemibrain" in str(folder):
            return _IN_GRAPH[DS1]
        if "male-cns" in str(folder):
            return _IN_GRAPH[DS2]
        return None

    monkeypatch.setattr(tc, "load_in_graph_roles", _in_graph)


def _query():
    return {
        "id": "threshold_3",
        "label": "N=3",
        "thresholds": {DS1: 3, DS2: 3},
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_union_classification(analyzer, coverage_env):
    analyzer.raw_results = _results()
    coverage = tc.build_query_type_coverage(analyzer, _query())

    # Present in both, present in one only.
    assert coverage[("Src", DS1)].present
    assert coverage[("Src", DS2)].present
    assert coverage[("APL", DS1)].present

    # APL: neurons exist in DS2, an edge from a path-source neuron exists
    # but is below the applied threshold (the real-run MCNS case).
    apl = coverage[("APL", DS2)]
    assert not apl.present
    assert apl.status == tc.STATUS_BELOW_THRESHOLD
    assert apl.status_label == "below threshold"
    assert "max edge weight from path sources 2 < threshold 3" in apl.detail

    # A qualifying source-side edge exists, yet the searched graph never
    # included the type.  "Late(Hold)" carries a merged-display alternate:
    # classification runs on the base name, so it is diagnosed like
    # "Late" instead of collapsing to mapper-unavailable.
    late = coverage[("Late(Hold)", DS2)]
    assert late.status == tc.STATUS_NOT_RECRUITED
    assert "edge weight 9 >= threshold 3 from path sources" in late.detail
    assert late.resolved_type == "Late"

    # Neurons exist but no edge touches the searched graph.
    assert coverage[("Island", DS2)].status == tc.STATUS_NO_EDGES

    # Same-name lane (mapper off): no such type in DS2 -> the mapper-off
    # verdict surfaces rather than a guessed not_in_dataset.
    ghost = coverage[("Ghost", DS2)]
    assert ghost.status == tc.STATUS_MAPPER_UNAVAILABLE
    assert "no same-name neurons" in ghost.detail

    # The present side never gets an absence diagnosis.
    assert coverage[("Tgt", DS2)].status == tc.STATUS_PRESENT


def test_union_resolves_into_every_dataset(analyzer, coverage_env):
    """The union covers DS1-only types: every union type has a DS2 entry."""
    analyzer.raw_results = _results()
    coverage = tc.build_query_type_coverage(analyzer, _query())
    union = {"Src", "Tgt", "APL", "Ghost", "Island", "Late(Hold)"}
    for name in union:
        for ds in (DS1, DS2):
            assert (name, ds) in coverage, (name, ds)


def test_missing_diagnosis_inputs_degrade(analyzer, monkeypatch):
    """No neuron table / no searched-graph list -> resolved_absent, never
    an exception and never a fabricated below-threshold verdict."""
    monkeypatch.setattr(tc, "load_neuron_table", lambda root, ds: None)
    monkeypatch.setattr(tc, "load_connection_frame", lambda root, ds: None)
    monkeypatch.setattr(tc, "load_in_graph_roles", lambda folder: None)
    analyzer.raw_results = _results()
    coverage = tc.build_query_type_coverage(analyzer, _query())
    assert coverage[("APL", DS2)].status == tc.STATUS_RESOLVED_ABSENT


def test_resolver_unmapped_surfaces_when_same_name_missing(
        analyzer, coverage_env, monkeypatch):
    """mapper ON: an explicit unmapped verdict + no same-name neurons is
    reported as unmapped, while a same-name hit still gets diagnosed."""

    def fake_resolve(mapper, source_type, source_dataset, target_dataset,
                     *, snapshot=None, alias_cache=None, bridge_cache=None):
        return TypeResolution(
            status=STATUS_UNMAPPED,
            source_type=str(source_type),
            source_dataset=str(source_dataset or ""),
            target_dataset=str(target_dataset),
            reason="not in any crosswalk",
        )

    monkeypatch.setattr(tc, "resolve_valid_targets", fake_resolve)
    analyzer.raw_results = _results()
    coverage = tc.build_query_type_coverage(analyzer, _query())
    # Ghost has no same-name neurons in DS2 -> unmapped verdict.
    assert coverage[("Ghost", DS2)].status == STATUS_UNMAPPED
    # APL exists same-name in DS2 despite "unmapped" -> still diagnosed
    # via the long-tail raw-name fallback (below threshold).
    assert coverage[("APL", DS2)].status == tc.STATUS_BELOW_THRESHOLD


def test_resolver_conflict_surfaces_verbatim(analyzer, coverage_env,
                                             monkeypatch):
    def fake_resolve(mapper, source_type, source_dataset, target_dataset,
                     *, snapshot=None, alias_cache=None, bridge_cache=None):
        return TypeResolution(
            status=STATUS_CONFLICT,
            source_type=str(source_type),
            source_dataset=str(source_dataset or ""),
            target_dataset=str(target_dataset),
            reason="rival claims",
        )

    monkeypatch.setattr(tc, "resolve_valid_targets", fake_resolve)
    analyzer.raw_results = _results()
    coverage = tc.build_query_type_coverage(analyzer, _query())
    entry = coverage[("APL", DS2)]
    assert entry.status == STATUS_CONFLICT
    assert entry.detail == "rival claims"
    # Present entries are untouched by the resolver.
    assert coverage[("APL", DS1)].present


# ---------------------------------------------------------------------------
# Exports / report surfaces
# ---------------------------------------------------------------------------

def test_presence_matrix_gains_status_columns(analyzer, coverage_env,
                                              tmp_path):
    analyzer.raw_results = _results()
    out_dir = tmp_path / "comparison_results"
    out_dir.mkdir()
    analyzer._export_presence_matrix(str(out_dir), 3)
    df = pd.read_csv(out_dir / "edge_presence_matrix_minsyn_3.csv")
    for col in (f"source_status_{SAFE2}", f"target_status_{SAFE2}",
                f"source_status_{SAFE1}"):
        assert col in df.columns, col
    apl = df[df["edge_key"] == "APL -> Tgt"].iloc[0]
    assert apl[f"source_status_{SAFE2}"] == "below_threshold"
    # The endpoint that IS present reads as present.
    tgt_row = df[df["edge_key"] == "Src -> Tgt"].iloc[0]
    assert tgt_row[f"target_status_{SAFE2}"] == "present"


def test_type_resolution_union_csv(analyzer, coverage_env, tmp_path):
    analyzer.raw_results = _results()
    out_dir = tmp_path / "comparison_results"
    out_dir.mkdir()
    analyzer._export_type_resolution_union(str(out_dir))
    df = pd.read_csv(out_dir / "type_resolution_union.csv")
    assert set(df["query_id"]) == {"threshold_3"}
    # One row per (query, union type, dataset).
    assert len(df) == 12
    apl = df[(df["type"] == "APL") & (df["dataset"] == DS2)].iloc[0]
    assert apl["present"] == False  # noqa: E712 - CSV round-trip
    assert apl["resolution_status"] == "below_threshold"
    present = df[(df["type"] == "Src") & (df["dataset"] == DS2)].iloc[0]
    assert present["present"] == True  # noqa: E712 - CSV round-trip
    assert present["resolution_status"] == "present"


def test_txt_report_type_coverage_section(analyzer, coverage_env):
    analyzer.raw_results = _results()
    lines = analyzer._type_coverage_txt_lines()
    text = "\n".join(lines)
    assert "TYPE COVERAGE (UNION RESOLUTION)" in text
    assert "MCNS below threshold" in text
    assert "max edge weight from path sources 2 < threshold 3" in text
    # Present pairs are omitted from the txt summary.
    assert "Src:" not in text


def test_txt_report_without_coverage_has_no_section(analyzer):
    analyzer.raw_results = {}
    assert analyzer._type_coverage_txt_lines() == []


# ---------------------------------------------------------------------------
# Report annotations
# ---------------------------------------------------------------------------

def test_network_hover_and_node_titles_annotated(analyzer, coverage_env):
    analyzer.raw_results = _results()
    html = __import__(
        "comparison.html_report_generator", fromlist=["x"]
    )._generate_conservation_network(
        analyzer, [DS1, DS2], 3, {DS1: "HEMI", DS2: "MCNS"})
    # The APL -> Tgt edge is DS2-absent: the hover names the failing
    # endpoint with its coverage verdict.
    assert "APL: below threshold" in html
    # The APL node title carries the per-dataset coverage line.
    assert "Coverage: MCNS below threshold" in html
    # A present dataset's weight line stays untouched.
    assert "HEMI: 5" in html


def test_network_hover_plain_without_coverage(analyzer, monkeypatch):
    analyzer.raw_results = _results()
    monkeypatch.setattr(analyzer, "_type_coverage", lambda: {})
    html = __import__(
        "comparison.html_report_generator", fromlist=["x"]
    )._generate_conservation_network(
        analyzer, [DS1, DS2], 3, {DS1: "HEMI", DS2: "MCNS"})
    assert "below threshold" not in html
    # The em-dash is JSON-escaped inside the embedded vis.js payload.
    assert "MCNS: \\u2014" in html


def test_coverage_card_renders_absent_rows(analyzer, coverage_env):
    hrg = __import__("comparison.html_report_generator", fromlist=["x"])
    analyzer.raw_results = _results()
    card = hrg._generate_type_coverage_card(
        analyzer, "threshold_3", [DS1, DS2],
        {DS1: "HEMI", DS2: "MCNS"}, display_label="N=3")
    assert "Type coverage" in card
    assert "below threshold" in card
    # Present-in-both types are not rows of the absent table.
    assert ">Src</td>" not in card


def test_coverage_card_empty_without_coverage(analyzer):
    hrg = __import__("comparison.html_report_generator", fromlist=["x"])
    card = hrg._generate_type_coverage_card(
        analyzer, "threshold_3", [DS1, DS2], {DS1: "HEMI", DS2: "MCNS"})
    assert card == ""


# ---------------------------------------------------------------------------
# Combinations mode wiring
# ---------------------------------------------------------------------------

def test_combinations_edge_rows_gain_status_columns(coverage_env, tmp_path):
    analyzer = ComparisonAnalyzer(_params(
        threshold_mode="combinations",
        threshold_combinations=[
            {"id": "combo_001", "label": "Combo 1",
             "thresholds": {DS1: 3, DS2: 3}},
        ],
    ), verbose=False)
    analyzer.raw_results = _results()
    out_dir = tmp_path / "comparison_results"
    out_dir.mkdir()
    analyzer._export_cross_dataset_combinations(str(out_dir))
    df = pd.read_csv(out_dir / "unified_edge_comparison.csv")
    for col in (f"source_status_{SAFE2}", f"target_status_{SAFE2}"):
        assert col in df.columns, col
    apl = df[df["edge_key"] == "APL -> Tgt"].iloc[0]
    assert apl[f"source_status_{SAFE2}"] == "below_threshold"


# ---------------------------------------------------------------------------
# Ratio-basis absence diagnosis: the tier gates w / total_incoming(post),
# so the below-threshold / not-recruited split must compare RATIOS, not
# raw synapse weights (implemented 2026-10-04; previously degraded to
# resolved_absent).
# ---------------------------------------------------------------------------
class TestRatioDiagnosis:
    def _ratio_analyzer(self, monkeypatch, conns, in_graph, tier=0.01):
        params = _params(
            weight_basis='connection_ratio', thresholds=[tier])
        analyzer = ComparisonAnalyzer(params, verbose=False)
        monkeypatch.setattr(
            tc, "load_neuron_table", lambda root, ds: _TABLES.get(ds))
        monkeypatch.setattr(
            tc, "load_connection_frame", lambda root, ds: conns.get(ds))
        monkeypatch.setattr(tc, "load_in_graph_roles", in_graph)
        return analyzer, {
            "id": "threshold_0.01", "label": "N=0.01",
            "thresholds": {DS1: tier, DS2: tier},
        }

    def test_ratio_below_threshold_and_not_recruited(self, monkeypatch):
        # DS1 adds a Weak type: bodyId 107 with 3 x 400-syn entry legs and
        # total incoming 1200 -> best ratio 400/1200 = 0.3333.
        # DS2 adds 10 x 5-syn inputs onto APL(203) from bodyId 208: APL's
        # total incoming becomes 2 + 50 = 52, so the Src->APL entry leg is
        # 2/52 = 0.0385. At tier 0.5: Weak (DS1) and APL (DS2) are BELOW;
        # Late(204) in DS1 is 4/4 = 1.0 >= tier -> NOT recruited.
        conns = {
            DS1: pd.DataFrame({
                "bodyId_pre": ["101", "101", "101", "101", "101",
                               "101", "101", "101"],
                "bodyId_post": ["102", "103", "104", "105", "106",
                                "107", "107", "107"],
                "weight": [10, 8, 2, 3, 4, 400, 400, 400],
            }),
            DS2: pd.DataFrame({
                "bodyId_pre": (["201", "201", "201", "203"]
                               + ["208"] * 10),
                "bodyId_post": (["202", "203", "204", "201"]
                                + ["203"] * 10),
                "weight": ([4, 2, 9, 50] + [5] * 10),
            }),
        }
        tables = {
            DS1: pd.DataFrame({
                "bodyId": [101, 102, 103, 104, 105, 106, 107],
                "type": ["Src", "Tgt", "APL", "Ghost", "Island", "Late",
                         "Weak"],
            }),
            DS2: _TABLES[DS2],
        }

        seen_folders = []

        def _ig(folder):
            seen_folders.append(str(folder))
            if "hemibrain" in str(folder):
                return _IN_GRAPH[DS1]
            if "male-cns" in str(folder):
                return _IN_GRAPH[DS2]
            return None

        analyzer, query = self._ratio_analyzer(monkeypatch, conns, _ig,
                                               tier=0.5)
        # AFTER the helper (it installs its own plain table loader).
        monkeypatch.setattr(
            tc, "load_neuron_table", lambda root, ds: tables.get(ds))
        # Union = types surfaced anywhere; each is diagnosed in the
        # datasets where it did NOT appear.
        analyzer.raw_results = {
            DS1: {0.5: _edge_df([("Src", "Tgt", 10), ("APL", "Tgt", 5)])},
            DS2: {0.5: _edge_df([("Src", "Tgt", 4), ("Weak", "Tgt", 2),
                                 ("Late", "Tgt", 3)])},
        }
        coverage = tc.build_query_type_coverage(analyzer, query)
        weak = coverage[("Weak", DS1)]
        assert weak.status == "below_threshold"
        assert "0.3333" in weak.detail and "400/1200" in weak.detail
        apl2 = coverage[("APL", DS2)]
        assert apl2.status == "below_threshold"
        assert "0.03846" in apl2.detail and "2/52" in apl2.detail
        late1 = coverage[("Late", DS1)]
        assert late1.status == "not_recruited"
        assert "1" in late1.detail and "4/4" in late1.detail
        # Types present where they surfaced stay untouched.
        assert coverage[("APL", DS1)].status == "present"
        assert coverage[("Weak", DS2)].status == "present"
        # The searched-graph folder must resolve through the ratio
        # grammar (a bare int(0.5) once produced minratio_0_0 — never on
        # disk — and silently degraded every diagnosis to unavailable).
        hemi = [f for f in seen_folders if "hemibrain" in f]
        assert hemi and all(f.endswith("minratio_0_5") for f in hemi), hemi

    def test_synapse_diagnosis_unchanged(self, analyzer, coverage_env):
        analyzer.raw_results = _results()
        coverage = tc.build_query_type_coverage(analyzer, _query())
        assert coverage[("APL", DS2)].status == "below_threshold"
        assert coverage[("APL", DS2)].detail == (
            "max edge weight from path sources 2 < threshold 3")
