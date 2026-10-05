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
    "path_prob", "min_ratio", "length", "nt_types", "coverage",
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
    # per-dataset enrollment files (bodyId coverage source)
    pd.DataFrame([
        {"bodyId": 1, "type": "S", "isInPath": True},
        {"bodyId": 2, "type": "S", "isInPath": True},
        {"bodyId": 3, "type": "S", "isInPath": False},
    ]).to_csv(run / "dataset_data" / "dsA" / "minsyn_3" / "source_neurons.csv",
              index=False)
    pd.DataFrame([
        {"bodyId": 4, "type": "T", "Checked": True},
        {"bodyId": 5, "type": "T", "Checked": True},
    ]).to_csv(run / "dataset_data" / "dsA" / "minsyn_3" / "target_neurons.csv",
              index=False)
    pd.DataFrame([
        {"bodyId": 6, "type": "S", "isInPath": True},
    ]).to_csv(run / "dataset_data" / "dsB" / "minsyn_3" / "source_neurons.csv",
              index=False)
    # per-delegate provenance blocks (round 9b: aggregated on the root card)
    (run / "dataset_data" / "dsA" / "minsyn_3" / "parameters.txt").write_text(
        "requested_threshold:           3\n"
        "applied_threshold:             5\n"
        "applied_threshold_source:      edge_budget\n"
        "strongest_first_budget_bitten: False\n"
        "strongest_first_tau:           5\n"
        "paths_complete:                True\n", encoding="utf-8")
    (run / "dataset_data" / "dsB" / "minsyn_3" / "parameters.txt").write_text(
        "requested_threshold:           3\n"
        "applied_threshold:             3\n"
        "applied_threshold_source:      requested\n"
        "strongest_first_budget_bitten: True\n"
        "strongest_first_tau:           3\n"
        "paths_complete:                False\n", encoding="utf-8")
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
# (e)/(f) CSV<->HTML consistency, bodyId coverage (round 8)
# ---------------------------------------------------------------------------

def test_embedded_rows_are_in_csv(single_run):
    report = generate_paths_pair_report(single_run, log=None)
    html_text = report.read_text(encoding="utf-8")
    payload = _payload(html_text)
    csv = pd.read_csv(
        single_run / OUTPUT_DIR_NAME / PATHS_CSV_NAME, dtype={"path": str},
        keep_default_na=False)  # the root unit's empty unit id stays ''
    assert "source_bodyid_coverage" in csv.columns
    assert "target_bodyid_coverage" in csv.columns
    for pair in payload["pairs"]:
        sub = csv[(csv["source"] == pair["source"])
                  & (csv["target"] == pair["target"])
                  & (csv["unit"] == pair["unit"])]
        for group in pair["table"]["groups"]:
            for row in group["rows"]:
                hit = sub[sub["path"] == row["path"]]
                assert len(hit) >= 1, f"{row['path']} missing from CSV"
                assert int(hit.iloc[0]["rank_in_pair_length"]) == row["rank"]
                # the two coverage columns embedded == CSV (pair-level n/N)
                assert (str(hit.iloc[0]["source_bodyid_coverage"])
                        == (row["scov"] or ""))
                assert (str(hit.iloc[0]["target_bodyid_coverage"])
                        == (row["tcov"] or ""))


def test_bodyid_coverage_source_target_n_over_n(tmp_path):
    """Round-8b item 1: coverage = bodyId-level n/N for source (isInPath /
    enrolled, from source_neurons.csv) and target (Checked / resolved, from
    target_neurons.csv) — two columns, per pair."""
    run = tmp_path / "find-paths-complete_FAFB_S_to_T_L2w3_20260101_000000"
    _write_csv(run, [_row("S->M->T", 30), _row("S->N->M->T", 20)])
    pd.DataFrame([
        {"bodyId": 1, "type": "S", "isInPath": True},
        {"bodyId": 2, "type": "S", "isInPath": False},
        {"bodyId": 3, "type": "S", "isInPath": True},
    ]).to_csv(run / "source_neurons.csv", index=False)
    pd.DataFrame([
        {"bodyId": 4, "type": "T", "Checked": True},
        {"bodyId": 5, "type": "T", "Checked": True},
        {"bodyId": 6, "type": "T", "Checked": False},
    ]).to_csv(run / "target_neurons.csv", index=False)
    report = generate_paths_pair_report(run, log=None)
    payload = _payload(report.read_text(encoding="utf-8"))
    pair = payload["pairs"][0]
    for g in pair["table"]["groups"]:
        for row in g["rows"]:
            assert row["scov"] == "2/3"
            assert row["tcov"] == "2/3"
    csv = pd.read_csv(run / OUTPUT_DIR_NAME / PATHS_CSV_NAME,
                      keep_default_na=False)
    assert (csv["source_bodyid_coverage"] == "2/3").all()
    assert (csv["target_bodyid_coverage"] == "2/3").all()


def test_bodyid_coverage_missing_files(tmp_path):
    run = tmp_path / "find-paths-complete_FAFB_S_to_T_L2w3_20260101_000000"
    _write_csv(run, [_row("S->T", 4)])
    report = generate_paths_pair_report(run, log=None)
    payload = _payload(report.read_text(encoding="utf-8"))
    for g in payload["pairs"][0]["table"]["groups"]:
        for row in g["rows"]:
            assert row["scov"] == "" and row["tcov"] == ""


def test_no_presence_matrix_in_payload_and_pane(single_run):
    """Round 8 item 1: the per-pair presence matrix is gone; the pane keeps
    the top-paths-per-length table."""
    report = generate_paths_pair_report(single_run, log=None)
    text = report.read_text(encoding="utf-8")
    payload = _payload(text)
    assert "matrices" not in payload
    assert "renderMatrix" not in text
    assert "Paths presence matrix" not in text
    assert "Top paths per length" in text
    # the single-dataset global tab keeps the pair × unit matrix
    assert "pair_matrix" in payload["global"]


def test_informative_unit_labels(single_run):
    """Round 8 item 5: '(run root)' is retired for dataset/threshold labels."""
    (single_run / "parameters.txt").write_text(
        "min synapse number:            3\n"
        "dataset:                       flywire_FAFB_v783\n", encoding="utf-8")
    report = generate_paths_pair_report(single_run, log=None)
    text = report.read_text(encoding="utf-8")
    payload = _payload(text)
    assert "(run root)" not in text
    assert payload["run"]["units"][0]["label"] == (
        "flywire_FAFB_v783 · min synapse 3")


def test_unit_label_fallback_to_folder_name(tmp_path):
    run = tmp_path / "minsyn_5"   # nested-delegate style: no parameters.txt
    _write_csv(run, [_row("A->B", 4)], name="A_to_B_allpaths_type.csv")
    report = generate_paths_pair_report(run, log=None)
    payload = _payload(report.read_text(encoding="utf-8"))
    assert payload["run"]["units"][0]["label"] == "minsyn_5"


def test_vispath_links_in_payload_and_pane(tmp_path):
    run = tmp_path / "find-paths-complete_FAFB_A_to_B_L1w3_20260101_000000"
    _write_csv(run, [_row("A->B", 4)])
    (run / "visualization").mkdir()
    (run / "visualization" / "Network_run.html").write_text(
        "<html></html>", encoding="utf-8")
    report = generate_paths_pair_report(run, log=None)
    text = report.read_text(encoding="utf-8")
    payload = _payload(text)
    assert payload["run"]["vispath"] == [
        {"href": "visualization/Network_run.html", "label": "🕸️ vispath network"}]
    # the pair footer renders the link from the payload
    assert "item.href" in text and "vispath network" in text


def test_multi_select_template_and_explorer(single_run):
    """Round 8 item 3: source/target selects are multiple; the explorer
    renders the cross product of the selections."""
    report = generate_paths_pair_report(single_run, log=None)
    text = report.read_text(encoding="utf-8")
    assert 'id="sel-source" multiple' in text
    assert 'id="sel-target" multiple' in text
    assert "selectedOptions" in text
    assert "currentPairs" in text      # cross-product renderer
    assert "Ctrl/Cmd-click" in text


def test_global_pair_matrix_has_hop_ranges(cross_run):
    """Round 8 item 6: pair × unit table carries a Lengths column
    (min–max across units) with per-unit hop-range tooltips."""
    report = generate_paths_pair_report(cross_run, log=None)
    text = report.read_text(encoding="utf-8")
    payload = _payload(text)
    rows = {(r["source"], r["target"]): r
            for r in payload["global"]["pair_matrix"]["rows"]}
    # every path in the fixture is 2 hops; the per-unit range collapses to "2"
    assert rows[("S", "T")]["length_range"] == "2–2"
    assert rows[("S", "T")]["ranges"]["dsA/minsyn_3"] == "2"
    assert "<th>Lengths</th>" in text
    assert 'title="hops 2"' in text


def test_global_pair_matrix_bodyid_coverage_columns(cross_run):
    """Round 9: the pair × unit table's Coverage column is replaced by
    Source/Target bodyId n/N columns (same values as the Pair Explorer)."""
    report = generate_paths_pair_report(cross_run, log=None)
    text = report.read_text(encoding="utf-8")
    payload = _payload(text)
    row = payload["global"]["pair_matrix"]["rows"][0]
    # primary unit = most paths (dsA/minsyn_3, 2 paths) → its enrollment n/N
    assert row["scov"] == "2/3" and row["tcov"] == "2/2"
    # dsB has no target_neurons.csv → target coverage unknown (—)
    # cov_detail values carry an internal scope tag ([pair]/[type]);
    # the rendered badges/hovers strip it.
    assert row["cov_detail"]["dsB/minsyn_3"] == "1/1 / — [type]"
    assert "Source coverage" in text and "Target coverage" in text
    assert "Coverage</th>" not in text          # old unit-coverage column gone
    # The fixture writes type-level paths only → the pair values fall
    # back to the type query-scope coverage, scope-labeled in the hover.
    assert row["cov_scope"] == "type"
    assert ('title="dsA/minsyn_3: 2/3 / 2/2; '
            'dsA/minsyn_5_applied_floor: — / —; '
            'dsB/minsyn_3: 1/1 / — '
            '— type query-scope (run skipped bodyId output)"' in text)
    # Query-scope coverage totals render per unit.
    units = payload["global"]["units"]
    assert units[0]["src_cov"] == [2, 3]  # JSON list from the payload
    assert "Query bodyId coverage" in text


def test_provenance_aggregated_per_delegate_on_cross_root(cross_run):
    """Round 9b: cross-dataset roots aggregate the per-delegate provenance
    blocks into one applied-threshold table; each delegate also carries its
    own parameters.txt so nested reports show their own card."""
    report = generate_paths_pair_report(cross_run, log=None)
    text = report.read_text(encoding="utf-8")
    payload = _payload(text)
    delegate_prov = payload["run"]["meta"]["delegate_provenance"]
    assert delegate_prov["dsA/minsyn_3"]["applied_threshold"] == "5"
    assert delegate_prov["dsA/minsyn_3"]["applied_threshold_source"] == (
        "edge_budget")
    assert delegate_prov["dsB/minsyn_3"]["applied_threshold"] == "3"
    # the applied-thresholds-per-delegate table renders on the root
    assert "Applied thresholds (per delegate)" in text
    assert "edge_budget" in text
    # nested delegate reports still show their own single-run card
    nested = (cross_run / "dataset_data" / "dsA" / "minsyn_3" / REPORT_NAME
              ).read_text(encoding="utf-8")
    assert "Applied threshold" in nested
    # the old single-run card (root provenance) does not render on the root
    assert "Threshold actually used" not in text


def test_provenance_card_marks_applied_threshold(tmp_path):
    """Round 9: applied threshold + StrongestFirst/Edge budget provenance
    rendered explicitly, like the cross-dataset report."""
    run = tmp_path / "find-paths-complete_FAFB_A_to_B_L5w3_20260101_000000"
    _write_csv(run, [_row("A->M->B", 30), _row("A->N->M->B", 20)])
    (run / "parameters.txt").write_text(
        "min synapse number:            3\n"
        "requested_threshold:           3\n"
        "applied_threshold:             17\n"
        "applied_threshold_source:      strongest_first_budget+edge_budget\n"
        "strongest_first_budget:        1000000\n"
        "strongest_first_budget_bitten: True\n"
        "strongest_first_tau:           17\n"
        "tau_canonical:                 17\n"
        "strongest_dropped_bottleneck:  16\n"
        "edge_budget:                   1000000\n"
        "edge_budget_applied:           True\n"
        "edge_budget_landing:           7\n"
        "edge_weight_floor:             8\n"
        "strongest_retained_bottleneck: 38\n"
        "paths_complete:                False\n", encoding="utf-8")
    report = generate_paths_pair_report(run, log=None)
    text = report.read_text(encoding="utf-8")
    payload = _payload(text)
    prov = payload["run"]["meta"]["provenance"]
    assert prov["applied_threshold"] == "17"
    assert prov["strongest_first_budget_bitten"] == "True"
    assert "⚙️ Applied threshold &amp; budget provenance" in text
    assert "Threshold actually used" in text
    assert "applied threshold: 17  (strongest_first_budget+edge_budget)" in text
    assert "Canonical tau" in text and "strongest_retained_bottleneck" in text


def test_provenance_absent_runs_no_card(tmp_path):
    run = tmp_path / "find-paths-complete_FAFB_A_to_B_L1w3_20260101_000000"
    _write_csv(run, [_row("A->B", 4)])
    report = generate_paths_pair_report(run, log=None)
    text = report.read_text(encoding="utf-8")
    assert "Applied threshold" not in text


def test_data_tab_rebuilt(single_run):
    """Round 8 item 4: the Data tab carries breakdown CSVs, artifacts, and
    vispath sections (not just one bare link)."""
    (single_run / "parameters.txt").write_text(
        "min synapse number:            3\n", encoding="utf-8")
    report = generate_paths_pair_report(single_run, log=None)
    text = report.read_text(encoding="utf-8")
    data_tab = re.search(
        r'<template id="tpl-data">(.*?)</template>', text, re.DOTALL).group(1)
    assert "Breakdown CSVs (lossless)" in data_tab
    assert "Run artifacts" in data_tab
    assert "parameters.txt" in data_tab
    # every href in the data tab resolves
    run_dir = single_run
    for h in re.findall(r'href="([^"#]+)"', data_tab):
        assert (run_dir / h.split("#")[0]).exists(), h


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


# ---------------------------------------------------------------------------
# Per-path / per-pair / query-scope bodyId coverage (2026-10-04): rows
# carry the FIRST/LAST entries of their own per-node coverage list; pair
# stats and the Global matrix count DISTINCT endpoint bodyIds per pair
# when bodyId-level output exists (else the type query-scope value,
# scope-labeled); the Global tab gains query-scope totals.
# ---------------------------------------------------------------------------
def _write_enrollment_csv(folder, rows, name):
    """source/target_neurons.csv (NOT the paths-table column schema)."""
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / name, index=False)


def test_pair_exact_coverage_when_bodyid_paths_exist(tmp_path):
    run = tmp_path / "find-paths-complete_FAFB_S_to_T_L2w3_20260101_000000"
    rows = [
        _row("S->M1->T", 10),
        _row("S->M2->T", 8),
    ]
    rows[0]["coverage"] = "[1/2, 1/1, 1/3]"
    rows[1]["coverage"] = "[2/2, 1/1, 2/3]"
    _write_csv(run, rows)
    # bodyId-level paths: three distinct source bodyIds realize the
    # pair (1 -> path A, 2 and 3 -> path B); one target bodyId.
    _write_csv(
        run,
        [
            {"path": "1->10->20", "weights": "[10, 10]",
             "probabilities": "[1, 1]", "ratios": "[1, 1]",
             "min_weight": 10, "path_prob": 1.0, "length": 2},
            {"path": "2->10->20", "weights": "[8, 8]",
             "probabilities": "[1, 1]", "ratios": "[1, 1]",
             "min_weight": 8, "path_prob": 1.0, "length": 2},
            {"path": "3->11->20", "weights": "[8, 8]",
             "probabilities": "[1, 1]", "ratios": "[1, 1]",
             "min_weight": 8, "path_prob": 1.0, "length": 2},
        ],
        name=f"{run.name}_allpaths_bodyId_paths.csv",
    )
    _write_enrollment_csv(run, [
        {"bodyId": 1, "type": "S", "isInPath": True},
        {"bodyId": 2, "type": "S", "isInPath": True},
        {"bodyId": 3, "type": "S", "isInPath": False},
        {"bodyId": 4, "type": "S", "isInPath": False},
    ], "source_neurons.csv")
    _write_enrollment_csv(run, [
        {"bodyId": 20, "type": "T", "Checked": True},
        {"bodyId": 21, "type": "T", "Checked": False},
        {"bodyId": 22, "type": "T", "Checked": False},
    ], "target_neurons.csv")
    from paths_pair_report import (
        build_unit_breakdown, load_unit_paths, _load_pair_bodyid_coverage,
        _load_query_coverage,
    )
    frame = load_unit_paths(run / f"{run.name}_allpaths_type.csv")
    pair_bodyid = _load_pair_bodyid_coverage(run)
    assert pair_bodyid[("S", "T")] == (3, 1)  # distinct endpoints per pair
    path_rows, _, pair_stats = build_unit_breakdown(
        frame, "min_weight", pair_bodyid=pair_bodyid,
        source_cov={"S": "2/4"}, target_cov={"T": "1/3"})
    stats = pair_stats[("S", "T")]
    assert stats["scov"] == "3/4" and stats["tcov"] == "1/3"
    assert stats["cov_scope"] == "pair"
    # per-path rows: first/last coverage entries, NOT the pair value
    assert path_rows[0]["source_bodyid_coverage"] == "1/2"
    assert path_rows[1]["source_bodyid_coverage"] == "2/2"
    query = _load_query_coverage(run)
    assert query["source"] == (2, 4) and query["target"] == (1, 3)


def test_query_coverage_card_and_unit_columns(tmp_path):
    """Full-report rendering on a run WITH enrollment CSVs (and per-pair
    bodyId output): the Global tab carries the query-scope card and the
    Units table's two coverage columns."""
    run = tmp_path / "find-paths-complete_FAFB_S2_to_T2_L2w3_20260101_000000"
    rows = [_row("S2->M->T2", 10)]
    rows[0]["coverage"] = "[1/2, 1/1, 1/3]"
    _write_csv(run, rows)
    _write_enrollment_csv(run, [
        {"bodyId": 1, "type": "S2", "isInPath": True},
        {"bodyId": 2, "type": "S2", "isInPath": False},
    ], "source_neurons.csv")
    _write_enrollment_csv(run, [
        {"bodyId": 20, "type": "T2", "Checked": True},
        {"bodyId": 21, "type": "T2", "Checked": False},
    ], "target_neurons.csv")
    from paths_pair_report import generate_paths_pair_report
    report = generate_paths_pair_report(run, log=None)
    text = report.read_text(encoding="utf-8")
    assert "Query bodyId coverage" in text
    assert "sources 1/2" in text and "targets 1/2" in text
    assert "Src on paths</th>" in text and "Tgt reached</th>" in text


def test_pair_summary_fallback_when_no_bodyid_paths(tmp_path):
    """skip_bodyId runs persist per-pair distinct endpoint counts at RUN
    time (data_details/pair_bodyid_coverage.csv) — the report reads that
    summary when the bodyId paths table is absent and still gets EXACT
    pair-scope coverage (scope='pair', not the query-scope fallback)."""
    run = tmp_path / "find-paths-complete_FAFB_S3_to_T3_L2w3_20260101_000000"
    rows = [_row("S3->M->T3", 10)]
    rows[0]["coverage"] = "[1/2, 1/1, 1/3]"
    _write_csv(run, rows)
    _write_enrollment_csv(run, [
        {"bodyId": 1, "type": "S3", "isInPath": True},
        {"bodyId": 2, "type": "S3", "isInPath": False},
    ], "source_neurons.csv")
    _write_enrollment_csv(run, [
        {"bodyId": 20, "type": "T3", "Checked": True},
        {"bodyId": 21, "type": "T3", "Checked": False},
        {"bodyId": 22, "type": "T3", "Checked": False},
    ], "target_neurons.csv")
    # NO bodyId paths CSV — the run-time summary stands in for it.
    (run / "data_details").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"source_type": "S3", "target_type": "T3",
         "distinct_source_bodyids": 2, "distinct_target_bodyids": 1},
    ]).to_csv(run / "data_details" / "pair_bodyid_coverage.csv",
              index=False)
    from paths_pair_report import (
        _load_pair_bodyid_coverage, build_unit_breakdown, load_unit_paths,
    )
    pair_bodyid = _load_pair_bodyid_coverage(run)
    assert pair_bodyid[("S3", "T3")] == (2, 1)
    frame = load_unit_paths(run / f"{run.name}_allpaths_type.csv")
    _, _, pair_stats = build_unit_breakdown(
        frame, "min_weight", pair_bodyid=pair_bodyid,
        source_cov={"S3": "1/2"}, target_cov={"T3": "1/3"})
    stats = pair_stats[("S3", "T3")]
    assert stats["scov"] == "2/2" and stats["tcov"] == "1/3"
    assert stats["cov_scope"] == "pair"


def test_coverage_stop_surfaces_in_provenance(tmp_path):
    """The single-run provenance card names the realized coverage stop
    (layer + requirements) and the knobs; cross roots get per-delegate
    Cov src / Cov tgt / Coverage stop columns (2026-10-04)."""
    run = tmp_path / "find-paths-shortest_FAFB_S4_to_T4_L3w3_20260101_000000"
    rows = [_row("S4->M->T4", 10)]
    _write_csv(run, rows)
    _write_enrollment_csv(run, [
        {"bodyId": 1, "type": "S4", "isInPath": True},
        {"bodyId": 2, "type": "S4", "isInPath": False},
    ], "source_neurons.csv")
    _write_enrollment_csv(run, [
        {"bodyId": 20, "type": "T4", "Checked": True},
        {"bodyId": 21, "type": "T4", "Checked": False},
    ], "target_neurons.csv")
    (run / "parameters.txt").write_text(
        "requested_threshold:           3\n"
        "applied_threshold:             3\n"
        "shortest source coverage:      any\n"
        "shortest target coverage:      100%\n"
    )
    attrs = {
        "shortest_source_coverage": 0.0,
        "shortest_target_coverage": 1.0,
        "shortest_discovery_diagnostics": {"coverage_stop": {
            "stopped_at_layer": 2,
            "requirements": {"source": 0.0, "target": 1.0},
        }},
    }
    (run / "all_attributes.json").write_text(
        json.dumps(attrs), encoding="utf-8")
    from paths_pair_report import (
        generate_paths_pair_report, _coverage_stop_summary,
    )
    assert _coverage_stop_summary(run) == (
        "layer 2 (source any + target 100%)")
    report = generate_paths_pair_report(run, log=None)
    text = report.read_text(encoding="utf-8")
    assert "Coverage stop" in text
    assert "layer 2 (source any + target 100%)" in text
    assert "Shortest source coverage" in text


# ---------------------------------------------------------------------------
# Refill + adjusted ratios + Overview query coverage (2026-10-05 round)
# ---------------------------------------------------------------------------
def test_adjusted_ratio_preferred_and_labeled(tmp_path):
    """The pair-ratio loader prefers connection_ratio_adj and reports
    'adjusted'; plain runs report 'plain'."""
    from paths_pair_report import _load_pair_level_ratios
    run = tmp_path / "find-paths-complete_FAFB_S5_to_T5_L2w3_20260101_000000"
    (run / "data_details").mkdir(parents=True)
    pd.DataFrame([
        {"type_pre": "S5", "type_post": "T5",
         "connection_ratio": 0.10, "connection_ratio_adj": 0.25},
    ]).to_csv(run / "data_details" / "connection_type.csv", index=False)
    mapping, source = _load_pair_level_ratios(run)
    assert source == "adjusted"
    assert mapping[("S5", "T5")] == 0.25

    run2 = tmp_path / "find-paths-complete_FAFB_S6_to_T6_L2w3_20260101_000000"
    (run2 / "data_details").mkdir(parents=True)
    pd.DataFrame([
        {"type_pre": "S6", "type_post": "T6", "connection_ratio": 0.10},
    ]).to_csv(run2 / "data_details" / "connection_type.csv", index=False)
    mapping2, source2 = _load_pair_level_ratios(run2)
    assert source2 == "plain"
    assert mapping2[("S6", "T6")] == 0.10


def test_overview_query_coverage_card_and_refill_card(tmp_path):
    """Overview gains the overall (pair-ignoring) coverage card; refill
    results display automatically when the records exist (synapse-summed
    summary + top refilled pairs table)."""
    run = tmp_path / "find-paths-complete_FAFB_S7_to_T7_L2w3_20260101_000000"
    rows = [_row("S7->M->T7", 10)]
    rows[0]["coverage"] = "[1/2, 1/1, 1/3]"
    _write_csv(run, rows)
    _write_enrollment_csv(run, [
        {"bodyId": 1, "type": "S7", "isInPath": True},
        {"bodyId": 2, "type": "S7", "isInPath": False},
    ], "source_neurons.csv")
    _write_enrollment_csv(run, [
        {"bodyId": 20, "type": "T7", "Checked": True},
        {"bodyId": 21, "type": "T7", "Checked": False},
    ], "target_neurons.csv")
    (run / "data_details").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"type_pre": "S7", "type_post": "T7", "weight": 5,
         "nt_type": "ACH", "connection_ratio": 0.1,
         "traversal_probability": 0.3, "block_probability": 0.7,
         "conn_layer": "0->1", "connection_ratio_adj": 0.4},
    ]).to_csv(run / "data_details" / "connection_type.csv", index=False)
    refill = run / "data_details" / "type_level_refill"
    refill.mkdir(parents=True)
    json.dump({"status": "refilled", "refill_edges": 42,
               "refill_weight_total": 0.02},
              open(refill / "refill_provenance.json", "w"))
    pd.DataFrame([
        {"type_pre": "S7", "type_post": "T7", "emitted_weight": 5,
         "refill_weight": 100, "refilled_total": 105,
         "emitted_pair_count": 1, "refill_pair_count": 20,
         "refilled_connection_ratio": 0.44, "split_status": "ok"},
    ]).to_csv(refill / "refill_type_pairs.csv", index=False)
    from paths_pair_report import generate_paths_pair_report
    report = generate_paths_pair_report(run, log=None)
    text = report.read_text(encoding="utf-8")
    assert "Overall bodyId coverage (query scope)" in text
    assert "source:" in text and "target:" in text
    assert "Type-level refill (budget-cut mass" in text
    assert "42 bodyId edges" in text          # provenance edge count
    assert "100 synapses of budget-cut mass" in text  # CSV sum, not the
    # ratio-mass provenance total
    assert "Pair-ratio (adjusted)" in text


# ---------------------------------------------------------------------------
# Percent convention + COLUMN SPEC + Refill tab (plan
# plan-pair-report-refill-tab-and-per-hop-ratios, 2026-10-05)
# ---------------------------------------------------------------------------
def test_percent_formatter_pins():
    from paths_pair_report import _fmt_pct, _format_ratio_list
    assert _fmt_pct(0.677) == '67.70%'
    assert _fmt_pct(0.0263) == '2.63%'
    assert _fmt_pct(0.000602) == '0.06%'
    assert _fmt_pct(0.0000042) == '0.00%'   # rounds away; title rule
    assert _fmt_pct(None) == '—'
    assert _fmt_pct('nan') == '—'
    assert _format_ratio_list('[0.0263, 0.060]') == '[2.63%, 6.00%]'
    assert _format_ratio_list('') == ''
    assert _format_ratio_list('[1]') == '[100.00%]'


def test_per_hop_adjusted_list_and_report_percent_forms(tmp_path):
    """pradj lists carry per-hop adjusted ratios; the payload's ratios
    lists are percent-formatted (the CSV keeps raw)."""
    from paths_pair_report import (
        _per_hop_adjusted_list, generate_paths_pair_report,
    )
    run = tmp_path / "find-paths-complete_FAFB_S8_to_T8_L2w3_20260101_000000"
    rows = [_row("S8->M8a->T8", 10), _row("S8->M8b->T8", 8)]
    rows[0]["ratios"] = "[0.5, 0.25]"
    rows[1]["ratios"] = "[0.1, 0.02]"
    _write_csv(run, rows)
    (run / "data_details").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"type_pre": "S8", "type_post": "M8a", "weight": 5,
         "nt_type": "ACH", "connection_ratio": 0.1,
         "traversal_probability": 0.3, "block_probability": 0.7,
         "conn_layer": "0->1", "connection_ratio_adj": 0.4},
        {"type_pre": "M8a", "type_post": "T8", "weight": 3,
         "nt_type": "ACH", "connection_ratio": 0.2,
         "traversal_probability": 0.5, "block_probability": 0.5,
         "conn_layer": "1->2", "connection_ratio_adj": 0.8},
    ]).to_csv(run / "data_details" / "connection_type.csv", index=False)
    pair_ratio = {("S8", "M8a"): 0.4, ("M8a", "T8"): 0.8,
                  ("S8", "M8b"): 0.25, ("M8b", "T8"): 0.5}
    assert _per_hop_adjusted_list("S8->M8a->T8", pair_ratio) == \
        '[40.00%, 80.00%]'
    assert _per_hop_adjusted_list("S8->M9->T8", pair_ratio) == \
        '[—, —]'

    report = generate_paths_pair_report(run, log=None)
    payload = _payload(report.read_text(encoding="utf-8"))
    entry = payload["pairs"][0]
    trows = [r for g in entry["table"]["groups"] for r in g["rows"]]
    by_path = {r["path"]: r for r in trows}
    assert by_path["S8->M8a->T8"]["ratios"] == "[50.00%, 25.00%]"
    assert by_path["S8->M8a->T8"]["pradj"] == "[40.00%, 80.00%]"
    # the breakdown CSV keeps the RAW strings
    bd = pd.read_csv(run / "paths_pair_breakdown" / "pair_breakdown_paths.csv")
    assert "[0.5, 0.25]" in set(bd.ratios.astype(str))
    # scalar pr = min over the same adjusted map
    assert by_path["S8->M8a->T8"]["pr"] == 0.4


def test_column_spec_header_cell_alignment_source_pins():
    """The renderer builds headers AND cells from one COLUMN SPEC —
    pinned structurally (the 2026-10-05 screenshot mismatch class)."""
    import inspect
    import paths_pair_report as M
    src = inspect.getsource(M)
    js = src[src.index("REPORT_JS"):]
    assert "var COLS = [" in js
    assert "COLS.forEach(function(col) {" in js
    # headers and cells both iterate the SAME spec
    assert js.count("COLS.forEach(function(col)") == 2
    # the hand-kept parallel lists are gone
    assert "var headers = ['#'];" not in js
    assert "headers.push('Path', 'Len'" not in js
    # percent formatter + the rounding-to-zero title fallback
    assert "function fmtPct(v)" in js
    assert "toFixed(2) + '%'" in js
    assert "Math.abs(100 * v) < 0.005" in js


def test_refill_tab_present_with_hint_state(tmp_path):
    """The Refill tab is a standalone page: always present, hint state
    when no records, full table when they exist (adjusted column when
    the engine wrote it)."""
    from paths_pair_report import generate_paths_pair_report
    run = tmp_path / "find-paths-complete_FAFB_S9_to_T9_L2w3_20260101_000000"
    rows = [_row("S9->M->T9", 10)]
    _write_csv(run, rows)
    report = generate_paths_pair_report(run, log=None)
    text = report.read_text(encoding="utf-8")
    assert 'id="tpl-refill"' in text
    assert "TAB.show('refill')" in text
    assert "No refill records — the budget did not bite" in text

    # now with records (+ the adjusted column)
    refill = run / "data_details" / "type_level_refill"
    refill.mkdir(parents=True)
    json.dump({"status": "refilled", "refill_edges": 7},
              open(refill / "refill_provenance.json", "w"))
    pd.DataFrame([
        {"type_pre": "S9", "type_post": "T9", "emitted_weight": 5,
         "refill_weight": 40, "refilled_total": 45,
         "emitted_pair_count": 1, "refill_pair_count": 8,
         "refilled_connection_ratio": 0.09,
         "refilled_connection_ratio_adj": 0.45,
         "split_status": "rederived_topn"},
    ]).to_csv(refill / "refill_type_pairs.csv", index=False)
    report = generate_paths_pair_report(run, log=None)
    text = report.read_text(encoding="utf-8")
    assert "Refilled ratio (adj)" in text
    assert "45.00%" in text            # refilled_connection_ratio_adj
    assert "9.00%" in text             # plain refilled ratio


def test_pair_level_ratio_two_row_shapes(tmp_path):
    """Real-data audit (2026-10-05): connection_type.csv carries recurring
    pairs in two shapes — TOTAL-stamped rows (same weight on every
    conn_layer row: the pair value is the row value; summing overcounts)
    and SPLIT rows (per-depth weights: the pair value is the SUM of the
    row ratios — shared per-post denominator, so ratios add like weights).
    .first() understated every split pair."""
    from paths_pair_report import _load_pair_level_ratios
    run = tmp_path / "find-paths-complete_FAFB_SA_to_TA_L2w3_20260101_000000"
    (run / "data_details").mkdir(parents=True)
    pd.DataFrame([
        # total-stamped recurring pair (3 identical rows)
        {"type_pre": "SA", "type_post": "TA", "weight": 86,
         "connection_ratio": 0.035581, "connection_ratio_adj": 0.05643,
         "conn_layer": f"{i}->{i+1}"} for i in range(3)
    ] + [
        # split recurring pair (per-depth weights 23/20/23)
        {"type_pre": "SB", "type_post": "TA", "weight": w,
         "connection_ratio": r, "connection_ratio_adj": a,
         "conn_layer": f"{i}->{i+1}"}
        for i, (w, r, a) in enumerate(
            [(23, 0.002969, 0.005356), (20, 0.002582, 0.004658),
             (23, 0.002969, 0.005356)])
        ] + [
        # single-row pair
        {"type_pre": "SC", "type_post": "TA", "weight": 10,
         "connection_ratio": 0.01, "connection_ratio_adj": 0.02,
         "conn_layer": "0->1"},
    ]).to_csv(run / "data_details" / "connection_type.csv", index=False)
    mapping, source = _load_pair_level_ratios(run)
    assert source == "adjusted"
    assert abs(mapping[("SA", "TA")] - 0.05643) < 1e-9      # not 3x
    assert abs(mapping[("SB", "TA")]
               - (0.005356 + 0.004658 + 0.005356)) < 1e-9  # summed
    assert mapping[("SC", "TA")] == 0.02
