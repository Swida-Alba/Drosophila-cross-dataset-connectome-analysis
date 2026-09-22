"""Tests for the TM VEV per-run report generator
(``comparison.mapping_validation_report``, template
``_plan/tmvev-run-report-template.md``, DECIDED 2026-09-16).

Covers: self-contained HTML assembly (no external refs), the §1
coverage levels, the §6 fill table with per-row cross-bin provenance,
the hover-glossary layer, the 20-row viewport catch-all, warning-note
collection + the ``user_warning_notes.txt`` header discipline, the
mapper-gap fallback chain (in-memory → set_coverage.json → README), and
the slim README contract.
"""

import csv
import json
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.mapping_validation_report import (  # noqa: E402
    append_warning_notes,
    build_report_document,
    collect_run_data,
    collect_warnings,
    _viewport,
)


def _write_csv(path: Path, header: list, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "type-map_dsA_to_dsB_test_20260916_000000"
    rd.mkdir()
    (rd / "parameters.json").write_text(json.dumps({
        "source_dataset": "dsA", "target_dataset": "dsB",
        "query_types": ["q1"], "validation_mode": "family",
        "top_k": 25, "min_synapse_threshold": 3,
        "morph_enabled": True, "morph_auc_floor": 0.65,
        "scene_selfcheck": True, "aggressive_expansion": False,
    }))
    (rd / "pipeline_progress.jsonl").write_text("\n".join([
        json.dumps({"ts": "2026-09-16T01:00:00", "event": "run_start"}),
        json.dumps({"ts": "2026-09-16T01:00:01", "event": "stage_start",
                    "stage": "1"}),
        json.dumps({"ts": "2026-09-16T01:00:11", "event": "stage_done",
                    "stage": "1"}),
        json.dumps({"ts": "2026-09-16T01:00:12", "event": "run_done",
                    "elapsed_s": 61.0}),
    ]) + "\n")
    (rd / "set_coverage.json").write_text(json.dumps({
        "source": {"total_queried": 4, "assigned": 2,
                 "fill_proposed_only": 1, "unpaired_unproposed": 1,
                 "per_type": {"A": {"pool": 2, "assigned": 2,
                                    "fill_proposed": 0,
                                    "unpaired_unproposed": 0}}},
        "target": {"mapped_target_set": 4, "in_branch_pool": 3,
                 "reached_as_candidates_only": 1, "holes": 1,
                 "family_material": [900, 901],
                 "per_type": {
                     "X": {"mapped_population": 2, "in_pool": 2,
                           "in_pool_matched": 1, "in_pool_verified": 1,
                           "in_pool_borderline": 0,
                           "reached_as_candidates_only": 0, "holes": 0,
                           "hole_body_ids": []},
                     "Y": {"mapped_population": 2, "in_pool": 1,
                           "in_pool_matched": 0, "in_pool_verified": 1,
                           "in_pool_borderline": 0,
                           "reached_as_candidates_only": 1, "holes": 1,
                           "hole_body_ids": [901]}},
                 },
    }))
    (rd / "README.txt").write_text(
        "Type-mapping validation run\n"
        "===========================\n"
        "source: dsA  target: dsB\n"
        "queries: q1\n"
        "\nRun log:\n"
        "[stage 1] resolving type pairs dsA -> dsB\n"
        "    self-check [A]: all legend leaves match their neuron "
        "geometry\n"
        "    ! something failed loudly\n"
        "\nMapper-gap evidence (target types flagged in\n"
        " this run with NO backward mapping to the source\n"
        " dataset - candidate annotation holes; consider\n"
        " annotating or crosswalking them):\n"
        "  SMP217: 44 row(s)\n"
        "  (untyped): 2 row(s) - no type annotation at all\n"
        "\nMorphology calibration: {...}\n")
    _write_csv(rd / "pair_summary.csv",
               ["query", "source_type", "target_type", "mapping_status",
                "pool_basis", "source_pool", "target_pool",
                "source_type_total", "target_type_total", "matched",
                "verdict_verified_strong", "verdict_verified",
                "verdict_borderline", "verdict_unmatched",
                "suspicious_neurons", "suspicious_noise_filtered",
                "gap", "gap_ratio", "gap_triggered", "hemisphere"],
               [["q1", "A", "X", "mapped", "full population", 2, 2, 2, 2,
                 1, 1, 1, 0, 0, 0, 0, 1, 0.5, True,
                 "{'source_sides': {'L': 1, 'R': 1}, 'target_sides': "
                 "{'L': 1, 'R': 1}, 'hemisphere_asymmetry': False}"],
                ["q1", "B", "Y", "evidence_only", "linker rows", 1, 1, 2,
                 2, 0, 0, 0, 1, 0, 2, 1, 1, 0.5, True,
                 "{'source_sides': {'L': 1, 'R': 0}, 'target_sides': "
                 "{'L': 1, 'R': 0}, 'hemisphere_asymmetry': True}"]])
    _write_csv(rd / "validation_results.csv",
               ["query", "source_bodyId", "verdict"],
               [["q1", "101", "verified_strong"],
                ["q1", "102", "verified"],
                ["q1", "103", "borderline"],
                ["q1", "104", "unmatched"]])
    _write_csv(rd / "examinees.csv",
               ["query", "source_type", "target_type", "category",
                "in_scope", "morph_failed", "candidate_annotation",
                "ahead_target_bodyId", "bar_kind", "bar_value"],
               [["q1", "A", "X", "candidates", True, False,
                 "Z(no_source)", "700", "native", 0.61],
                ["q1", "B", "Y", "candidates", True, False,
                 "W(out-map)", "701", "track_a_backup", 0.55],
                ["q1", "B", "Y", "sibling", True, False, "", "300",
                 "", ""],
                ["q1", "B", "Y", "", False, True, "V>{src}", "702",
                 "", ""]])
    _write_csv(rd / "gap_fill_dedup.csv",
               ["target_bodyId", "target_type", "dedup_category",
                "n_branches", "dup", "counts_toward_restrictive_fill",
                "counts_toward_family_fill"],
               [["700", "Z", "candidates", 1, False, True, True],
                ["701", "W", "candidates", 2, True, True, True],
                ["705", "W", "candidates", 1, False, True, True],
                ["500", "W", "family", 1, False, False, True],
                ["600", "V", "relative", 1, False, False, True]])
    _write_csv(rd / "gap_fill_levels.csv",
               ["level", "target_bodyId", "target_type",
                "dedup_category", "evidence", "bar_value", "dup", "note"],
               [["high", "700", "Z", "candidates", "native", 0.61, False,
                 ""],
                ["medium", "701", "W", "candidates", "track_a_backup",
                 0.55, True, ""]])
    _write_csv(rd / "gap_fill_proposals.csv",
               ["query", "source_type", "target_type", "side",
                "fill_class", "proposal_bodyId", "source_verdict"],
               [["q1", "B", "W", "source", "out_of_pool", "701",
                 "borderline"],
                ["q1", "B", "W", "source", "out_of_pool", "705",
                 "borderline"]])
    _write_csv(rd / "family_candidates.csv",
               ["query", "source_bodyId"], [["q1", "500"]])
    _write_csv(rd / "relatives.csv",
               ["query", "source_bodyId"], [["q1", "600"]])
    _write_csv(rd / "out_map_expansion.csv",
               ["query", "source_type", "source_bodyId", "target_bodyId",
                "target_type", "rank_union", "rank_union_rank", "jaccard",
                "jaccard_rank", "in_map", "morph_v2_similarity",
                "morph_qualified"],
               [["q1", "C", "201", "801", "T1", -0.1, 1, 0.2, 1, False,
                 0.4, True],
                ["q1", "C", "201", "802", "T2", -0.2, 2, 0.1, 2, False,
                 -0.1, False],
                ["q1", "D", "202", "803", "T1", -0.3, 1, 0.0, 1, False,
                 0.05, False]])
    (rd / "morphology_calibration.json").write_text(json.dumps({
        "auc": 0.54, "calibrated": False, "note": "gate INACTIVE",
        "auc_floor": 0.65, "track_a_null_bar": 0.23,
        "track_a_null_n": 346,
        "track_a_null_source": "p95 of jaccard<=0.05 window rows",
        "bar_params": {"rule": "floors v3"},
        "branch_bars": {
            "A->X": {"candidate_kind": "native", "native_floor": 0.8,
                     "suspicious_kind": "track_a_suspicious"},
            "B->Y": {"candidate_kind": "null", "native_floor": None,
                     "suspicious_kind": "null_lo"},
        },
        "score_frame": {"track_a": "target render space",
                        "track_b": "target native",
                        "visualization": "source coordinates"},
        "n_verified_scored": 5, "n_suspicious_scored": 7,
    }))
    scene = rd / "visualization" / "plot-3d_dsA_branches_A_20260101_000001"
    scene.mkdir(parents=True)
    (scene / "branches_A.html").write_text("<html>scene</html>")
    (scene / "branches_A.png").write_bytes(b"\x89PNG fake")
    return rd


def test_report_is_self_contained_with_headline(run_dir: Path):
    d = collect_run_data(run_dir)
    html = build_report_document(d)
    assert "<!DOCTYPE html>" in html
    # decided D6: the count verb is map-covered; headline carries 3 of 4
    assert "map-covered" in html
    assert "<b>3</b>" in html and "75.0%" in html
    # the 8 tabs of the presentation spec
    for label in ("Coverage", "Branches", "Targets", "Fill", "Out-map",
                  "Morph", "Scenes", "Log"):
        assert f">{label}</button>" in html
    # zero external references (offline convention)
    assert not re.search(r'(?:src|href)=["\']?(?:https?:)?//', html)
    # relative scene links survive
    assert "visualization/plot-3d_dsA_branches_A_20260101_000001/" \
        in html


def test_hover_glossary_and_definitions(run_dir: Path):
    html = build_report_document(collect_run_data(run_dir))
    assert "class='term'" in html          # hover layer present
    assert "<span class='tip'>" in html
    assert "Definitions for this section" in html
    # structured candidates breakdown (in-family / no_source / backward)
    assert "Candidates breakdown" in html
    assert "no_source — orphan types" in html


def test_fill_table_cross_bin_provenance(run_dir: Path):
    d = collect_run_data(run_dir)
    html = build_report_document(d)
    # the restrictive headline equals the dedup count (3: 700, 701 +
    # the cross-bin 705), and the family fill is the dedup remainder
    assert "3 restrictive · family-fill +2 (family 1 + relative 1)" \
        in html
    # in-bin row carries its expansion provenance + bar
    assert "Z(no_source)" in html and "0.610 (native)" in html
    # cross-bin row keeps per-row provenance (decided D8): 705 is in the
    # dedup candidates but has no candidates-bin expansion row
    assert "out-of-pool same-type proposal" in html
    assert "B→W, source verdict borderline" in html


def test_warnings_and_notes_header_discipline(run_dir: Path, capsys):
    d = collect_run_data(run_dir)
    warns = collect_warnings(d)
    tags = [w.split("]")[0] + "]" for w in warns]
    assert "[self-check]" in tags
    assert "[run-warning] ! something failed loudly" in warns
    assert any(w.startswith("[null-sample]") for w in warns)
    assert any(w.startswith("[mapper-gap] SMP217: 44 row(s)")
               for w in warns)
    assert any("(untyped): 2 row(s)" in w for w in warns)

    notes = run_dir / "user_warning_notes.txt"
    append_warning_notes(run_dir, warns)
    first = notes.read_text(encoding="utf-8")
    assert first.startswith("User warning notes\n===\n") or \
        first.startswith("User warning notes\n==")
    append_warning_notes(run_dir, ["[x] second block"])
    second = notes.read_text(encoding="utf-8")
    assert "[x] second block" in second and len(second) > len(first)
    # legacy headerless file is healed
    legacy = run_dir / "legacy"
    legacy.mkdir()
    (legacy / "user_warning_notes.txt").write_text("old scratch note\n")
    append_warning_notes(legacy, ["[y] healed"])
    healed = (legacy / "user_warning_notes.txt").read_text(encoding="utf-8")
    assert healed.startswith("User warning notes")
    assert "old scratch note" in healed and "[y] healed" in healed

    # regeneration idempotency: re-appending the same blocks is a no-op
    before = notes.read_text(encoding="utf-8")
    append_warning_notes(run_dir, warns)
    assert notes.read_text(encoding="utf-8") == before


def test_mapper_gap_fallback_chain(run_dir: Path):
    # 1) README block (legacy runs)
    d = collect_run_data(run_dir)
    assert d["mapper_gap"] == {"SMP217": 44}
    assert d["mapper_gap_untyped"] == 2
    # 2) in-memory pass-through wins
    d2 = collect_run_data(run_dir, mapper_gap_types={"OTHER": 3},
                          mapper_gap_untyped=1)
    assert d2["mapper_gap"] == {"OTHER": 3}
    assert d2["mapper_gap_untyped"] == 1
    # 3) set_coverage.json payload (new runs, slim README has no block)
    sc = json.loads((run_dir / "set_coverage.json").read_text(encoding="utf-8"))
    sc["mapper_gap"] = {"types": {"FROMJSON": 9}, "untyped_rows": 0}
    (run_dir / "set_coverage.json").write_text(json.dumps(sc))
    (run_dir / "README.txt").write_text(
        (run_dir / "README.txt").read_text(encoding="utf-8").replace(
            "  SMP217: 44 row(s)\n  (untyped): 2 row(s) - no type "
            "annotation at all\n", ""))
    d3 = collect_run_data(run_dir)
    assert d3["mapper_gap"] == {"FROMJSON": 9}


def test_viewport_catch_all():
    rows = [f"<tr><td>r{i}</td></tr>" for i in range(25)]
    html = _viewport(rows, "<th>c</th>")
    assert "Show all 25 rows (first 20 above)" in html
    assert html.count("<td>r0</td>") == 1  # open table keeps first 20
    small = _viewport(rows[:5], "<th>c</th>")
    assert "Show all" not in small


def test_scenes_status_and_outmap(run_dir: Path):
    d = collect_run_data(run_dir)
    html = build_report_document(d)
    assert "photoreceptor population" not in html  # no R1-R6 in fixture
    assert "2/3 pass" in html or "1/3 pass" in html
    # out-map: per-source best candidate row (rank_union_rank == 1)
    assert "<td>801 T1</td>" in html


def test_null_advisory_rendered(run_dir: Path):
    d = collect_run_data(run_dir)
    assert d["null_used"] is True   # B->Y candidate_kind == null
    html = build_report_document(d)
    assert "Null-sample" in html and "run-sensitive" in html


# ---------------------------------------------------------------------------
# slim README contract (mapping_validation.py)
# ---------------------------------------------------------------------------

def test_out_map_expansion_keeps_morph_columns(tmp_path: Path):
    """Regression (write-path review 2026-09-16): `_write_outputs` used
    to subset out_map_expansion.csv to the 11 base columns, dropping
    `morph_v2_similarity` / `morph_qualified` — which the report's
    Out-map tab and the scene layer read back."""
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from comparison.mapping_validation import MappingValidator, \
        MappingValidationConfig, run_file_path

    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(
        source_dataset="dsA", target_dataset="dsB", query_types=["q1"],
        visualize=False)
    v.notes = []
    v.run_dir = tmp_path
    v.pairs = []
    row = {"query": "q1", "source_type": "A", "source_bodyId": 1,
           "target_bodyId": 2, "target_type": "X", "rank_union": 0.1,
           "rank_union_rank": 1, "jaccard": 0.2, "jaccard_rank": 1,
           "in_map": False, "morph_v2_similarity": 0.5,
           "morph_qualified": True}
    v._write_outputs([], [], [], [], None, None, [], [],
                     out_map_rows=[row])
    # resolved through the layout registry (expansion/ since 2026-09-19) so
    # this stays a morph-column regression test, not a path test
    text = run_file_path(tmp_path, "out_map_expansion.csv").read_text(
        encoding="utf-8")
    assert "morph_v2_similarity" in text and "morph_qualified" in text


def test_no_scene_run_display(run_dir: Path):
    """Refinement (2026-09-16): a run without scenes must not render
    '0/0' self-check counts anywhere — hero, L3, warnings, log tab."""
    import shutil
    shutil.rmtree(run_dir / "visualization")
    readme = run_dir / "README.txt"
    kept = [ln for ln in readme.read_text(encoding="utf-8").splitlines()
            if "self-check [A]" not in ln]
    readme.write_text("\n".join(kept) + "\n")

    d = collect_run_data(run_dir)
    assert d["selfcheck"] == {"pass": 0, "fail": 0}
    warns = collect_warnings(d)
    assert any(w.startswith("[scenes] none rendered") for w in warns)
    assert not any(w.startswith("[self-check]") for w in warns)

    html = build_report_document(d)
    assert "no scenes" in html
    assert "0/0" not in html


def test_slim_readme_contract(tmp_path: Path):
    from comparison.mapping_validation import MappingValidator

    class _Cfg:
        source_dataset = "dsA"
        target_dataset = "dsB"
        query_types = ["q1"]
        effective_mode = "family"
        run_label = "stub"

    v = MappingValidator.__new__(MappingValidator)
    v.cfg = _Cfg()
    v.notes = ["[stage 1] ok", "    ! a failure line"]
    v.run_dir = tmp_path
    v._write_readme([], None, None)
    text = (tmp_path / "README.txt").read_text(encoding="utf-8")
    assert "Start here:" in text
    assert "report.html" in text
    assert "Run log:" in text
    assert "! a failure line" in text       # raw log kept verbatim
    assert "Column glossary" not in text     # moved to the report
    assert "SET-LEVEL COVERAGE" not in text  # lives in report §1
    assert "Morphology calibration:" not in text  # no JSON blob
    assert "Pair summaries:" not in text     # lives in report §3


def test_readme_refresh_contract(tmp_path: Path):
    """run() refreshes the slim README after the closing log lines
    (refinement 2026-09-16): the README is a pure function of
    self.notes, so rewriting it at completion captures 'report
    written' / 'done in' — lines logged after _write_outputs."""
    from comparison.mapping_validation import MappingValidator

    class _Cfg:
        source_dataset = "dsA"
        target_dataset = "dsB"
        query_types = ["q1"]
        effective_mode = "family"
        run_label = "stub"

    v = MappingValidator.__new__(MappingValidator)
    v.cfg = _Cfg()
    v.run_dir = tmp_path
    v.notes = ["[stage 1] ok"]
    v._write_readme([], None, None)
    assert "done in" not in (tmp_path / "README.txt").read_text(encoding="utf-8")
    # the run() tail: closing lines are logged, then the refresh
    v.notes.append("[TMVEV] report written: …/report.html")
    v.notes.append("done in 3315s -> …/type-map_run")
    v._write_readme([], None, None)
    text = (tmp_path / "README.txt").read_text(encoding="utf-8")
    assert "done in 3315s" in text
    assert text.index("Start here:") < text.index("done in 3315s")


def test_branch_bar_join_accepts_query_prefixed_keys(run_dir: Path):
    """Branch bars are keyed 'query|src->tgt' in multi-query runs and
    plain 'src->tgt' in v3r7-era runs — the Branches tab must resolve
    both (found via the 2026-09-17 real-run inspection)."""
    import json
    cal = json.loads((run_dir / "morphology_calibration.json").read_text(encoding="utf-8"))
    cal["branch_bars"] = {
        f"q1|{k}": v for k, v in cal["branch_bars"].items()}
    (run_dir / "morphology_calibration.json").write_text(json.dumps(cal))
    html = build_report_document(collect_run_data(run_dir))
    assert "0.800 (native)" in html      # A->X native floor resolves


def test_branches_tab_counts_mapped_not_paired(run_dir: Path):
    """The old `matched` column showed the mutual-best PAIR count and the
    gap was measured off it, so a branch whose sources all carry a mapping
    verdict still read as a 1-neuron gap (user 2026-09-20).  Mapped sums
    the verdicts (verified_strong + verified + borderline) and the gap is
    measured against THAT number — the pair count stays visible beside it,
    as the stricter subset rather than an addend."""
    html = build_report_document(collect_run_data(run_dir))
    assert "class='term'>Mapped (M)<span class='tip'>" in html
    assert ">Matched<" not in html
    # A->X: v★ 1 + v 1 + b 0 = 2 mapped, and both pools are 2, so the
    # verdict-based gap is 0 even though the CSV pair-based gap is 1.
    row = html.split("A → X", 1)[1].split("</tr>", 1)[0]
    assert "2<span class='mv-note'> · pairs 1</span>" in row
    assert "0 (0%)" in row
    assert "1 (50%)" not in row


def test_branches_tab_hemisphere_warning_carries_its_own_evidence(
        run_dir: Path):
    """⚠ in the Gap-triggered column used to be a bare `title` attribute —
    invisible under the JS hover layer and silent about WHY it fired.  It
    now names the pools' L/R counts, and only on the branch that flagged."""
    html = build_report_document(collect_run_data(run_dir))
    assert "class='term warn-chip'" in html
    assert "Source pool L 1 / R 0, target pool L 1 / R 0" in html
    assert "Source pool L 1 / R 1" not in html


def test_track_a_unavailable_run_surfaced(run_dir: Path):
    """A transient enrichment failure silently degrades the run to
    Track-B floors (found in the 2026-09-17 reportcheck run) — the
    report must flag it loudly instead of rendering '—' bars and a
    'bar None' advisory."""
    import json
    cal = json.loads((run_dir / "morphology_calibration.json").read_text(encoding="utf-8"))
    cal.update({"n_verified_scored": 0, "n_suspicious_scored": 0,
                "track_a_null_bar": None, "track_a_null_n": 0})
    for spec in cal["branch_bars"].values():
        spec["pool_track_a_baseline"] = None
        spec["n_scored_pool"] = 0
    (run_dir / "morphology_calibration.json").write_text(json.dumps(cal))
    (run_dir / "visualization.rmdir" if False else None)
    d = collect_run_data(run_dir)
    assert d["track_a_dead"] is True
    assert "morph Track-A unavailable" in d["advisories"]
    warns = collect_warnings(d)
    assert any(w.startswith("[morph] Track-A unavailable") for w in warns)
    assert not any(w.startswith("[null-sample]") for w in warns)
    html = build_report_document(d)
    assert "Track-A morphology unavailable this run" in html
    assert "bar None" not in html


def _add_mapping_export(run_dir: Path, linker_bids, full_bids):
    """Write a selected-branch mapping_export.csv with the given source
    bodyId pools per branch basis."""
    import csv as _csv
    rows = []
    for basis, bids, branch in (("linker rows", linker_bids, "A"),
                                ("full population", full_bids, "B")):
        rows.append({
            "source_dataset": "dsA", "source_type": "A",
            "target_dataset": "dsB", "target_type": "X",
            "query": "q1", "is_selected": "True", "pool_basis": basis,
            "source_body_ids": "{" + ", ".join(str(b) for b in bids) + "}",
            "branch_index": branch, "branch_of": "A",
        })
    with open(run_dir / "mapping_export.csv", "w", newline="",
              encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_waterfall_uses_mapper_consistent_basis_split(run_dir: Path):
    """D9: §2 buckets are row-evidence backed / same-name pooled /
    out-map, derived from the SELECTED mapping_export branches; the
    decided flat split stays as the evidence overlay."""
    _add_mapping_export(run_dir, linker_bids=[101, 102], full_bids=[103])
    html = build_report_document(collect_run_data(run_dir))
    # 4 queried = 2 linker + 1 full population + 1 out-map
    assert "2 row-evidence backed" in html
    assert "1 same-name pooled" in html
    assert "<td>1 out-map</td>" in html
    assert "Consistent with the type mapper" in html
    # D-B10: the bookkeeping overlay is gone — statuses live in the
    # Backward tab
    assert "+1 fill-proposed" not in html
    assert "Backward tab" in html


def test_waterfall_outmap_derived_from_coverage(run_dir: Path):
    """BANC case (2026-09-17): when every queried source sits in a
    branch but the expansion produced 0 rows, the out-map bucket is 0
    via coverage (NOT via row presence) and the gap is stated."""
    import csv as _csv
    _add_mapping_export(run_dir, linker_bids=[101, 102],
                        full_bids=[103, 104])
    with open(run_dir / "out_map_expansion.csv", "w", newline="",
              encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["query", "source_type", "source_bodyId",
                    "target_bodyId", "target_type", "rank_union",
                    "rank_union_rank", "jaccard", "jaccard_rank", "in_map",
                    "morph_v2_similarity", "morph_qualified"])
    d = collect_run_data(run_dir)
    html = build_report_document(d)
    assert "<td>0 out-map</td>" in html
    assert "expansion produced 0 candidate rows" in html
    # buckets still account the whole query: 2 linker + 2 full + 0
    assert "2 row-evidence backed" in html
    assert "2 same-name pooled" in html


def test_family_reconciliation_line(run_dir: Path):
    """Family material is presented as 219−204-style remainder and
    reconciled against the dedup bins."""
    d = collect_run_data(run_dir)
    html = build_report_document(d)
    # fixture: family_material [900, 901]; neither appears in a dedup bin
    assert "Family material 2 (= 4 in-map − 3 map-covered)" in html
    assert "in no expansion bin" in html


def test_dedup_rank_candidates_outranks_family():
    """Restored precedence (user 2026-09-17): candidates above family in
    the dedup rollup (fill accounting); tier above both, examinees
    last."""
    from comparison.mapping_validation import DEDUP_RANK
    assert DEDUP_RANK["candidates"] > DEDUP_RANK["family"]
    assert DEDUP_RANK["matched"] > DEDUP_RANK["sibling"] > DEDUP_RANK[
        "candidates"]
    assert DEDUP_RANK["candidates"] > DEDUP_RANK["relative"] > DEDUP_RANK[
        "examinees"]


def test_basis_buckets_partition_on_evidence_kind():
    """The in-map source population was read as the 'linker rows' plus the
    'full population' keys, so a side resolved through the release relation
    counted in neither and the Coverage/Targets cells under-reported it."""
    from comparison.mapping_validation_report import split_basis_buckets
    basis = {'linker rows': {1, 2},
             'release relation participants': {3},
             'full population': {4, 5},
             'unmeasured': set()}
    row, wide = split_basis_buckets(basis)
    assert row == ['linker rows', 'release relation participants']
    assert wide == ['full population', 'unmeasured']
    assert sum(len(basis[b]) for b in row) == 3
    assert sum(len(basis[b]) for b in wide) == 2
    assert split_basis_buckets({}) == ([], [])


def test_branches_cell_labels_each_side_of_the_pool(run_dir):
    """One `pool_basis` per branch went ambiguous once each side could come
    from a different chain; the cell names both, and a run written before the
    per-side column exists still reads (source side only)."""
    wide_term = "source <span class='term'>full population"
    doc = build_report_document(collect_run_data(run_dir))
    assert wide_term in doc and ' · target ' not in doc
    header = ["query", "source_type", "target_type", "mapping_status",
              "pool_basis", "target_pool_basis", "source_pool",
              "target_pool", "source_type_total", "target_type_total",
              "matched", "verdict_verified_strong", "verdict_verified",
              "verdict_borderline", "verdict_unmatched",
              "suspicious_neurons", "suspicious_noise_filtered", "gap",
              "gap_ratio", "gap_triggered", "hemisphere"]
    rows = list(csv.reader(
        (run_dir / "pair_summary.csv").read_text().splitlines()))[1:]
    _write_csv(run_dir / "pair_summary.csv", header,
               [[*r[:5], 'linker rows', *r[5:]] for r in rows])
    doc = build_report_document(collect_run_data(run_dir))
    # row A is wide on the source, row B is row-backed on both sides
    assert wide_term in doc and ' · target linker rows' in doc
    assert 'source linker rows · target linker rows' in doc
