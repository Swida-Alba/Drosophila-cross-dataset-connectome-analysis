"""Tests for the paths pair report (plan:
_plan/plan-paths-pair-report.md §6).

Pins: per-length top-10 cap math, shared/unique classification, unit
discovery across all three run kinds, additive-only + byte-stable
re-generation, CSV↔HTML consistency, the cross-unit presence-matrix join,
payload integrity (selects / deep-link slugs), and SVG layout
determinism (min-hop placement, first-seen column order).
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paths_pair_report import (  # noqa: E402
    DEFAULT_MATRIX_ROWS,
    DEFAULT_TOP_PER_LENGTH,
    INTERMEDIATES_CSV_NAME,
    OUTPUT_DIR_NAME,
    PATHS_CSV_NAME,
    REPORT_NAME,
    build_unit_breakdown,
    build_viz,
    discover_units,
    generate_paths_pair_report,
    load_unit_paths,
    run_kind,
)

CSV_COLUMNS = [
    "path", "weights", "probabilities", "ratios", "min_weight",
    "path_prob", "min_ratio", "length", "nt_types",
]


def _row(path, mw, weights=None, pp=None):
    hops = len(path.split("->")) - 1
    if weights is None:
        weights = "[" + ", ".join(str(mw) for _ in range(hops)) + "]"
    if pp is None:
        pp = 0.5
    return {
        "path": path, "weights": weights,
        "probabilities": "[0.5]", "ratios": "[0.1]",
        "min_weight": mw, "path_prob": pp, "min_ratio": 0.1,
        "length": hops, "nt_types": '["ACH"]',
    }


def _write_csv(folder: Path, rows, name=None):
    folder.mkdir(parents=True, exist_ok=True)
    name = name or f"{folder.name}_allpaths_type.csv"
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(folder / name, index=False)
    return folder / name


def _payload(html_text) -> dict:
    match = re.search(
        r'<script type="application/json" id="pair-data">(.*?)</script>',
        html_text, re.DOTALL)
    assert match, "payload script tag missing"
    return json.loads(match.group(1))


@pytest.fixture
def single_run(tmp_path) -> Path:
    """Complete-paths run: pair (R1,T1) has 30 paths at EACH of lengths
    2/3/4; pair (R1,T2) is a direct L1 connection."""
    run = tmp_path / "find-paths-complete_FAFB_R1_etc_to_T1_L4w3_20260101_000000"
    rows = []
    # length 2: R1->M{i}->T1, min_weight 100..71 (desc)
    for i in range(30):
        rows.append(_row(f"R1->M{i:02d}->T1", 100 - i))
    # length 3: R1->A{i}->B{i}->T1, min_weight 200-i (so the length-3 group
    # outranks length-2 in the table despite longer routes)
    for i in range(30):
        rows.append(_row(f"R1->A{i:02d}->B{i:02d}->T1", 200 - i))
    # length 4
    for i in range(30):
        rows.append(_row(f"R1->P{i:02d}->Q{i:02d}->Z{i:02d}->T1", 300 - i))
    # direct pair, exercises the L0-style single-edge shape
    rows.append(_row("R1->T2", 50))
    _write_csv(run, rows, name="R1_etc_to_T1_allpaths_type.csv")
    return run


@pytest.fixture
def cross_run(tmp_path) -> Path:
    """Cross-dataset run with a skipped folder, an applied_floor delegate,
    and paths shared across units for the presence-matrix join."""
    run = tmp_path / "cross-dataset_S_to_T_20260101_000000"
    _write_csv(run / "dataset_data" / "dsA" / "minsyn_3", [
        _row("S->X->T", 10),
        _row("S->Y->T", 8),
    ])
    _write_csv(run / "dataset_data" / "dsA" / "minsyn_5_applied_floor", [
        _row("S->Y->T", 8),
    ])
    _write_csv(run / "dataset_data" / "dsB" / "minsyn_3", [
        _row("S->X->T", 12),
        _row("S->Z->T", 5),
    ])
    (run / "dataset_data" / "dsA" / "minsyn_4_skipped").mkdir(
        parents=True, exist_ok=True)
    (run / "dataset_data" / "dsA" / "minsyn_4_skipped" / "README.txt").write_text(
        "skipped", encoding="utf-8")
    return run


# ---------------------------------------------------------------------------
# (c) discovery + run kind
# ---------------------------------------------------------------------------

def test_report_file_is_path_report_html():
    assert REPORT_NAME == "path_report.html"


def test_interactive_network_and_fallback_wiring(single_run):
    """Round 4/5: interactive vis-network rendered from the VENDORED
    library (fully offline), with the static-SVG fallback pattern."""
    report = generate_paths_pair_report(single_run, log=None)
    text = report.read_text(encoding="utf-8")
    assert "new vis.Network" in text                # interactive renderer
    assert "vis-network could not initialize" in text  # SVG-fallback note
    assert 'id="global-network"' in text            # global network holder
    # vendored library is INLINED, not CDN-loaded
    assert "unpkg.com/vis-network" not in text
    assert "visjs.github.io/vis-network" in text    # vendored banner present


def test_run_kind_detection():
    assert run_kind("find-paths-complete_X") == "Complete Paths"
    assert run_kind("find-paths-shortest_X") == "Shortest Paths"
    assert run_kind("cross-dataset_X") == "Cross-Dataset > Paths"


def test_shortest_run_note_and_flag(tmp_path):
    run = tmp_path / "find-paths-shortest_FAFB_A_to_B_L5w3_20260101_000000"
    _write_csv(run, [_row("A->M->B", 4), _row("A->N->B", 3)])
    report = generate_paths_pair_report(run, log=None)
    text = report.read_text(encoding="utf-8")
    payload = _payload(text)
    assert payload["run"]["shortest"] is True
    assert "Shortest path mode: results contain only" in text
    assert "DATA.run.shortest" in text  # pair-pane note hook
    # shortest is per bodyId pair — the type-level note must NOT claim a
    # flat top-10 (real-data finding 2026-10-02: 42/63 pairs span lengths)
    assert "degenerates" not in text


def test_hemisphere_badge_from_parameters(tmp_path):
    run = tmp_path / "find-paths-complete_FAFB_A_to_B_L2w3_20260101_000000"
    _write_csv(run, [_row("A_L->B_L", 4)])
    (run / "parameters.txt").write_text(
        "separate hemispheres:          True\n", encoding="utf-8")
    report = generate_paths_pair_report(run, log=None)
    text = report.read_text(encoding="utf-8")
    payload = _payload(text)
    assert payload["run"]["hemispheres"] is True
    assert "Hemisphere-aware run" in text


def test_discovery_single_run_root_unit(single_run):
    units = discover_units(single_run)
    assert len(units) == 1
    assert units[0].unit_id == "" and units[0].label == "(run root)"
    assert units[0].dataset == ""


def test_discovery_cross_run_labels_and_skips(cross_run):
    units = discover_units(cross_run)
    by_id = {u.unit_id: u for u in units}
    assert set(by_id) == {
        "dsA/minsyn_3", "dsA/minsyn_5_applied_floor", "dsB/minsyn_3"}
    floor = by_id["dsA/minsyn_5_applied_floor"]
    assert floor.label == "dsA/minsyn_5_applied_floor"  # raw name intact
    assert floor.threshold == "5" and floor.dataset == "dsA"
    # sorted deterministically: dataset, threshold, unit id
    assert [u.unit_id for u in units] == [
        "dsA/minsyn_3", "dsA/minsyn_5_applied_floor", "dsB/minsyn_3"]


def test_discovery_ignores_data_details(tmp_path):
    run = tmp_path / "find-paths-shortest_FAFB_A_to_B_L5w3_20260101_000000"
    _write_csv(run, [_row("A->B", 4)], name="A_to_B_allpaths_type.csv")
    (run / "data_details").mkdir()
    _write_csv(run / "data_details", [_row("A->B", 4)],
               name="A_to_B_allpaths_type_excluded.csv")
    units = discover_units(run)
    assert len(units) == 1 and units[0].unit_id == ""


# ---------------------------------------------------------------------------
# (a) cap math + rank order  /  (b) shared-unique  /  (h) layout
# ---------------------------------------------------------------------------

def test_cap_math_top10_per_length(single_run):
    report = generate_paths_pair_report(single_run, log=None)
    payload = _payload(report.read_text(encoding="utf-8"))
    pair = next(p for p in payload["pairs"]
                if (p["source"], p["target"]) == ("R1", "T1"))
    groups = pair["table"]["groups"]
    assert [g["len"] for g in groups] == [2, 3, 4]
    assert sum(len(g["rows"]) for g in groups) == 30  # 3 lengths x top-10
    for group in groups:
        assert len(group["rows"]) == DEFAULT_TOP_PER_LENGTH
        assert group["total"] == 30
        ranks = [row["rank"] for row in group["rows"]]
        assert ranks == list(range(1, 11))
        weights = [row["mw"] for row in group["rows"]]
        assert weights == sorted(weights, reverse=True)  # bottleneck-first
    # the length-3 group's top row outranks any length-2 row (200 > 100)
    assert groups[1]["rows"][0]["path"] == "R1->A00->B00->T1"
    # cap note numbers
    shown = sum(len(g["rows"]) for g in groups)
    assert shown < pair["paths_total"] == 90


def test_shared_unique_classification(tmp_path):
    run = tmp_path / "find-paths-complete_FAFB_S_to_T_L2w3_20260101_000000"
    _write_csv(run, [
        _row("S->A->T", 30),          # A unique
        _row("S->B->C->T", 25),       # B shared, C unique
        _row("S->B->D->T", 20),       # B shared, D unique
    ])
    frame = load_unit_paths(next(run.glob("*_allpaths_type.csv")))
    _, inter_rows, pair_stats = build_unit_breakdown(frame, "min_weight")
    by_node = {r["intermediate"]: r for r in inter_rows}
    assert by_node["B"]["classification"] == "shared"
    assert by_node["B"]["n_paths_using"] == 2
    assert by_node["A"]["classification"] == "unique"
    assert by_node["C"]["classification"] == "unique"
    stats = pair_stats[("S", "T")]
    assert stats["shared"] == 1 and stats["unique"] == 3
    assert stats["paths"] == 3


def test_rank_uses_bottleneck_then_prob_then_path(tmp_path):
    run = tmp_path / "find-paths-complete_FAFB_S_to_T_L2w3_20260101_000000"
    _write_csv(run, [
        _row("S->Lo->T", 5, pp=0.9),
        _row("S->Hi->T", 9, pp=0.1),
        _row("S->HiB->T", 9, pp=0.1),   # same mw+pp -> path asc tiebreak
        _row("S->HiA->T", 9, pp=0.1),
    ])
    frame = load_unit_paths(next(run.glob("*_allpaths_type.csv")))
    path_rows, _, _ = build_unit_breakdown(frame, "min_weight")
    order = [r["path"] for r in sorted(
        (r for r in path_rows if r["length"] == 2),
        key=lambda r: r["rank_in_pair_length"])]
    assert order == ["S->Hi->T", "S->HiA->T", "S->HiB->T", "S->Lo->T"]


def test_viz_min_hop_placement_and_first_seen_order():
    rows = [
        {"path": "S->M->T", "weights": "[5, 5]"},
        {"path": "S->N->M->T", "weights": "[4, 4, 4]"},  # M now at hop 2
        {"path": "S->K->T", "weights": "[3, 3]"},
        {"path": "S->T", "weights": "[2]"},
    ]
    counts = {"M": 2, "N": 1, "K": 1}
    minhop = {"M": 1, "N": 1, "K": 1}
    viz = build_viz(rows, counts, minhop)
    by_id = {n["id"]: n for n in viz["nodes"]}
    assert by_id["S"]["cls"] == "source" and by_id["S"]["col"] == 0
    assert by_id["T"]["cls"] == "target" and by_id["T"]["col"] == viz["ncols"] - 1
    assert by_id["M"]["cls"] == "shared"
    assert by_id["M"]["col"] == 1          # MIN hop position, not hop 2
    assert by_id["N"]["cls"] == "unique" and by_id["N"]["col"] == 1
    # first-seen order within the intermediate column: M, N, K
    assert [by_id[x]["row"] for x in ("M", "N", "K")] == [0, 1, 2]
    # columns = source + hop positions + target (4 nodes in the longest path)
    assert viz["ncols"] == 4
    # aggregated edges
    keys = {(e["f"], e["t"]): e for e in viz["edges"]}
    assert keys[("S", "M")]["c"] == 1
    assert keys[("M", "T")]["c"] == 2      # direct route + via N
    assert keys[("M", "T")]["w"] == 9.0    # Σw = 5 (hop 2 of path 1) + 4
    assert keys[("S", "T")]["c"] == 1
    assert ("S", "N") in keys and ("N", "M") in keys


def test_viz_l0_keeps_source_and_target_apart():
    rows = [{"path": "R1->T2", "weights": "[50]"}]
    viz = build_viz(rows, {}, {})
    assert viz["ncols"] == 2
    by_id = {n["id"]: n for n in viz["nodes"]}
    assert by_id["R1"]["col"] == 0 and by_id["T2"]["col"] == 1


# ---------------------------------------------------------------------------
# (d) additive-only + byte-stable re-generation
# ---------------------------------------------------------------------------

def test_regeneration_is_additive_and_byte_stable(single_run):
    sentinel = single_run / "pre_existing.txt"
    sentinel.write_text("do not touch", encoding="utf-8")

    def snapshot():
        return {p.relative_to(single_run): p.read_bytes()
                for p in single_run.rglob("*") if p.is_file()}

    before = snapshot()
    first = generate_paths_pair_report(single_run, log=None)
    after_first = snapshot()
    # only the report + the two breakdown CSVs appeared
    new_paths = set(after_first) - set(before)
    assert new_paths == {
        Path(REPORT_NAME),
        Path(OUTPUT_DIR_NAME) / PATHS_CSV_NAME,
        Path(OUTPUT_DIR_NAME) / INTERMEDIATES_CSV_NAME,
    }
    assert sentinel.read_text(encoding="utf-8") == "do not touch"

    text1 = after_first[Path(REPORT_NAME)].decode("utf-8")
    second = generate_paths_pair_report(single_run, log=None)
    after_second = snapshot()
    assert set(after_second) == set(after_first)
    text2 = after_second[Path(REPORT_NAME)].decode("utf-8")

    stamp = re.compile(r"Generated: [\d\- :]+")
    assert stamp.sub("Generated: X", text1) == stamp.sub("Generated: X", text2)
    assert second == first


# ---------------------------------------------------------------------------
# (e)/(f) CSV<->HTML consistency, presence-matrix join
# ---------------------------------------------------------------------------

def test_embedded_rows_are_in_csv(single_run):
    report = generate_paths_pair_report(single_run, log=None)
    html_text = report.read_text(encoding="utf-8")
    payload = _payload(html_text)
    csv = pd.read_csv(
        single_run / OUTPUT_DIR_NAME / PATHS_CSV_NAME, dtype={"path": str},
        keep_default_na=False)  # the root unit's empty unit id stays ''
    for pair in payload["pairs"]:
        sub = csv[(csv["source"] == pair["source"])
                  & (csv["target"] == pair["target"])
                  & (csv["unit"] == pair["unit"])]
        for group in pair["table"]["groups"]:
            for row in group["rows"]:
                hit = sub[sub["path"] == row["path"]]
                assert len(hit) >= 1, f"{row['path']} missing from CSV"
                assert int(hit.iloc[0]["rank_in_pair_length"]) == row["rank"]
        matrix = payload["matrices"][pair["mt"]]
        assert matrix["shown"] <= DEFAULT_MATRIX_ROWS
        assert matrix["shown"] <= matrix["total"]
        allowed = set(csv[(csv["source"] == pair["source"])
                          & (csv["target"] == pair["target"])]["path"])
        for row in matrix["rows"]:
            assert row["path"] in allowed


def test_presence_matrix_cross_unit_join(cross_run):
    report = generate_paths_pair_report(cross_run, log=None)
    payload = _payload(report.read_text(encoding="utf-8"))
    pair = next(p for p in payload["pairs"]
                if (p["source"], p["target"]) == ("S", "T"))
    matrix = payload["matrices"][pair["mt"]]
    assert matrix["units"] == [
        "dsA/minsyn_3", "dsA/minsyn_5_applied_floor", "dsB/minsyn_3"]
    by_path = {row["path"]: row for row in matrix["rows"]}
    assert by_path["S->X->T"]["cons"] == 2   # dsA/minsyn_3 + dsB/minsyn_3
    assert by_path["S->X->T"]["cells"]["dsB/minsyn_3"] == 12
    assert by_path["S->X->T"]["cells"]["dsA/minsyn_5_applied_floor"] is None
    assert by_path["S->Y->T"]["cons"] == 2   # dsA/minsyn_3 + applied_floor
    assert by_path["S->Z->T"]["cons"] == 1
    # conservation-first, then max-weight desc: X (2, max 12) before Y (2, 8)
    order = [row["path"] for row in matrix["rows"]]
    assert order.index("S->X->T") < order.index("S->Y->T") < order.index("S->Z->T")


def test_cross_run_writes_nested_per_delegate_reports(cross_run):
    """Round 7: cross-dataset runs also write a single-unit path_report.html
    into each dataset_data/<dataset>/<delegate>/ folder."""
    report = generate_paths_pair_report(cross_run, log=None)
    payload = _payload(report.read_text(encoding="utf-8"))
    unit_ids = [u["id"] for u in payload["run"]["units"]]
    assert unit_ids == [
        "dsA/minsyn_3", "dsA/minsyn_5_applied_floor", "dsB/minsyn_3"]
    for unit_id in unit_ids:
        nested = cross_run / "dataset_data" / unit_id
        assert (nested / REPORT_NAME).exists(), unit_id
        assert (nested / OUTPUT_DIR_NAME / PATHS_CSV_NAME).exists(), unit_id
        nested_payload = _payload(
            (nested / REPORT_NAME).read_text(encoding="utf-8"))
        # single-unit, self-consistent report: the delegate folder IS the
        # nested report's run root (unit_id '')
        assert [u["id"] for u in nested_payload["run"]["units"]] == ['']
        assert all(p["unit"] == '' for p in nested_payload["pairs"])
        nested_csv = pd.read_csv(
            nested / OUTPUT_DIR_NAME / PATHS_CSV_NAME, keep_default_na=False)
        root_sub = pd.read_csv(
            cross_run / OUTPUT_DIR_NAME / PATHS_CSV_NAME,
            keep_default_na=False)
        root_rows = root_sub[root_sub["unit"] == unit_id]
        assert len(nested_csv) == len(root_rows) > 0
        assert set(nested_csv["path"]) == set(root_rows["path"])
    # the root Data tab links every nested report (resolvable relative hrefs)
    root_html = report.read_text(encoding="utf-8")
    for unit_id in unit_ids:
        assert f'dataset_data/{unit_id}/path_report.html' in root_html


def test_single_run_writes_no_nested_reports(single_run):
    report = generate_paths_pair_report(single_run, log=None)
    text = report.read_text(encoding="utf-8")
    assert 'href="dataset_data/' not in text  # no nested report links
    nested = list(single_run.glob("dataset_data"))
    assert nested == []


# ---------------------------------------------------------------------------
# (g) payload integrity: selects + deep-link slugs
# ---------------------------------------------------------------------------

def test_payload_select_and_slug_integrity(single_run):
    report = generate_paths_pair_report(single_run, log=None)
    html_text = report.read_text(encoding="utf-8")
    payload = _payload(html_text)
    # payload parses (done by _payload) and its pair ids are slugs
    slug_re = re.compile(r"^[A-Za-z0-9._-]+$")
    ids = set()
    for pair in payload["pairs"]:
        assert slug_re.match(pair["id"]), pair["id"]
        ids.add(pair["id"])
    # every deep link on the Overview round-trips to a payload pair id
    overview = re.search(
        r'<template id="tpl-overview">(.*?)</template>', html_text, re.DOTALL)
    linked = set(re.findall(r"TAB\.gotoPair\('([^']+)'\)", overview.group(1)))
    assert linked == ids
    # select option universes: one row per pair in the Overview pair table
    assert html_text.count("data-pair=") == len(payload["pairs"])
    # unit select only when multi-unit
    assert ('id="sel-unit"' in html_text) == (len(payload["run"]["units"]) > 1)


def test_intermediates_csv_written(single_run):
    generate_paths_pair_report(single_run, log=None)
    inter = pd.read_csv(single_run / OUTPUT_DIR_NAME / INTERMEDIATES_CSV_NAME)
    assert set(inter.columns) == {
        "dataset", "threshold", "unit", "source", "target", "intermediate",
        "n_paths_using", "classification", "min_hop_position"}
    assert (inter["classification"].isin(["shared", "unique"])).all()
    # a length-4 intermediate can sit as deep as hop 3, never deeper
    assert inter["min_hop_position"].between(1, 3).all()


# ---------------------------------------------------------------------------
# global tab (round-3 addition: global presentation before the per-pair
# breakdown, cross-dataset-analysis style)
# ---------------------------------------------------------------------------

def test_global_tab_present_and_unit_stats(single_run):
    report = generate_paths_pair_report(single_run, log=None)
    text = report.read_text(encoding="utf-8")
    assert "TAB.show('global')" in text
    assert '<template id="tpl-global">' in text
    payload = _payload(text)
    glob = payload["global"]
    assert set(glob) >= {"units", "pair_matrix", "pair_units_hist",
                         "network", "edge_stats"}
    unit = glob["units"][0]
    assert unit["paths"] == 91 and unit["pairs"] == 2
    assert unit["sources"] == 1 and unit["targets"] == 2
    assert unit["max_weight"] == 300
    rows = {(r["source"], r["target"]): r for r in glob["pair_matrix"]["rows"]}
    assert rows[("R1", "T1")]["total"] == 90
    assert rows[("R1", "T1")]["cons"] == 1
    assert rows[("R1", "T2")]["total"] == 1
    assert glob["pair_units_hist"] == {"1": 2}


def test_global_pair_matrix_and_network_cross_unit_join(cross_run):
    report = generate_paths_pair_report(cross_run, log=None)
    payload = _payload(report.read_text(encoding="utf-8"))
    glob = payload["global"]
    matrix = glob["pair_matrix"]
    assert matrix["units"] == [
        "dsA/minsyn_3", "dsA/minsyn_5_applied_floor", "dsB/minsyn_3"]
    assert matrix["total_pairs"] == 1
    row = matrix["rows"][0]
    assert (row["source"], row["target"]) == ("S", "T")
    assert row["cells"] == {
        "dsA/minsyn_3": 2, "dsA/minsyn_5_applied_floor": 1, "dsB/minsyn_3": 2}
    assert row["cons"] == 3 and row["total"] == 5
    assert glob["pair_units_hist"] == {"3": 1}
    # edge aggregation across units: S->X traversed by dsA/minsyn_3 + dsB
    edges = {(e["f"], e["t"]): e["c"] for e in glob["network"]["edges"]}
    assert edges[("S", "X")] == 2 and edges[("X", "T")] == 2
    assert edges[("S", "Y")] == 2 and edges[("Y", "T")] == 2
    assert edges[("S", "Z")] == 1 and edges[("Z", "T")] == 1
    assert glob["edge_stats"] == {"total": 6, "shown": 6}
    by_id = {n["id"]: n for n in glob["network"]["nodes"]}
    assert by_id["S"]["cls"] == "source" and by_id["T"]["cls"] == "target"
    # each intermediate is touched by exactly one pair -> unique (gray)
    assert by_id["X"]["cls"] == "unique"
    assert by_id["X"]["col"] == 1


def test_global_node_role_precedence(tmp_path):
    """A node that is a target in one path and an intermediate in another
    stays purple (target precedence), not blue."""
    run = tmp_path / "cross-dataset_R_to_U_20260101_000000"
    _write_csv(run / "dataset_data" / "dsA" / "minsyn_3", [
        _row("R->M->U", 10),   # U as target
        _row("R->U->U2", 8),   # U as intermediate, U2 as target
    ])
    report = generate_paths_pair_report(run, log=None)
    payload = _payload(report.read_text(encoding="utf-8"))
    by_id = {n["id"]: n for n in payload["global"]["network"]["nodes"]}
    assert by_id["R"]["cls"] == "source"
    assert by_id["U"]["cls"] == "target"
    assert by_id["U"]["col"] == 1        # placed at its intermediate hop
    assert by_id["U2"]["cls"] == "target"
    assert by_id["M"]["cls"] == "unique"


def test_meta_summary_is_escaped_once(tmp_path):
    """The header must not double-escape (no literal &amp;#x27; renders)."""
    run = tmp_path / "cross-dataset_S_to_T_20260101_000000"
    _write_csv(run / "dataset_data" / "dsA" / "minsyn_3", [_row("S->T", 5)])
    (run / "run_manifest.json").write_text(json.dumps({
        "datasets": ["male-cns:v1.0"], "nicknames": ["MCN'S"],
        "parameters": {}}), encoding="utf-8")
    report = generate_paths_pair_report(run, log=None)
    text = report.read_text(encoding="utf-8")
    assert "MCN&#x27;S" in text
    assert "&amp;#x27;" not in text
