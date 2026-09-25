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
    _scene_failures,
    _scenes_tab,
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
                 # the set-level count spans EVERY candidate-reached bodyId
                 # whatever its type, while `per_type` exists only for mapped
                 # types — so the two scopes differ by construction (measured
                 # 27 vs 10 on the 2026-09-25 male-cns family run). The fixture
                 # keeps that shape on purpose.
                 "reached_as_candidates_only": 3, "holes": 1,
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
                "source_type_total", "target_type_total", "best",
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
    # the header mirrors the schema the writer produces, including the bar
    # columns #61 added — a fixture that keeps an old header tests a file no
    # run writes
    _write_csv(rd / "out_map_expansion.csv",
               ["query", "source_type", "source_bodyId", "target_bodyId",
                "target_type", "rank_union", "rank_union_rank", "jaccard",
                "jaccard_rank", "in_map", "morph_v2_similarity", "morph_bar",
                "morph_bar_kind", "morph_qualified"],
               [["q1", "C", "201", "801", "T1", -0.1, 1, 0.2, 1, False,
                 0.4, 0.143, "null", True],
                ["q1", "C", "201", "802", "T2", -0.2, 2, 0.1, 2, False,
                 -0.1, 0.143, "null", False],
                ["q1", "D", "202", "803", "T1", -0.3, 1, 0.0, 1, False,
                 0.05, 0.143, "null", False]])
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
    # …and the cell prints the bar its ✓ was measured against (#61), so the
    # mark is recomputable from the row instead of from prose
    assert "vs 0.143" in _plain(html)


def test_null_advisory_rendered(run_dir: Path):
    d = collect_run_data(run_dir)
    assert d["null_used"] is True   # B->Y candidate_kind == null
    html = build_report_document(d)
    assert "Null-sample" in html and "run-sensitive" in html
    # The same callout used to reassure the reader that native-floor bars were
    # "unaffected" across runs — the 2026-09-25 cold/warm BANC divergence
    # (107 rows) was exactly a native-floor bar moving. The sentence now names
    # the invariant and where to check it.
    assert "one vector-cache read returns one space" in html
    assert "morph_stores" in html


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
           "morph_bar": 0.143, "morph_bar_kind": "null",
           "morph_qualified": True}
    v._write_outputs([], [], [], [], None, None, [], [],
                     out_map_rows=[row])
    # resolved through the layout registry (expansion/ since 2026-09-19) so
    # this stays a morph-column regression test, not a path test
    text = run_file_path(tmp_path, "out_map_expansion.csv").read_text(
        encoding="utf-8")
    assert "morph_v2_similarity" in text and "morph_qualified" in text
    assert "morph_bar,morph_bar_kind" in text, text
    assert "0.143" in text and "null" in text


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
    """The `best` column (published as `matched` before 2026-09-24) shows the
    mutual-best PAIR count and the
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
              "best", "verdict_verified_strong", "verdict_verified",
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


def test_branches_tab_names_the_parents_without_a_scene(run_dir: Path):
    """One scene renders one PARENT type's whole branch group, and stage 4
    can be capped — so a bare '—' in the Scene column left 'the cap dropped
    this parent' indistinguishable from 'this branch had nothing to review'
    (user 2026-09-22, on a 43-branch run that rendered 12 of 21 parents).
    The tab now states the coverage and marks the affected rows."""
    html = build_report_document(collect_run_data(run_dir))
    assert "1 of 2 parent types have a rendered scene" in html
    assert "no scene for: B" in html
    row_b = html.split("B → Y", 1)[1].split("</tr>", 1)[0]
    assert "No scene rendered for this parent type" in row_b
    # the parent WITH a scene keeps its plain link and no warning
    row_a = html.split("A → X", 1)[1].split("</tr>", 1)[0]
    assert "branches_A.html" in row_a
    assert "No scene rendered" not in row_a


# ---------------------------------------------------------------------------
# pooling mode: the unsupervised pool and its post-hoc comparison
# ---------------------------------------------------------------------------

POOL_HEADER = ["target_bodyId", "target_type", "leaf", "best_source_bodyId",
               "best_source_type", "jaccard", "jaccard_rank", "rank_union",
               "window_size", "size_nm3", "size_universe_percentile",
               "in_scope", "n_sources", "dup", "morph_gate",
               "morph_bar_kind", "morph_similarity", "morph_pool_ref",
               "morph_bar", "morph_qualified",
               "mapper_cell", "mapper_verdict"]

XVAL = {
    "universe_scanned": 103770,
    "seed": {"queried_sources": 8, "scanned": 8,
             "sources_with_a_candidate": 5, "distinct_best_sources": 4},
    # the blocks `cross_validation` writes on a bar-driven run, in its shape:
    # the bar decides admission, the gate holds the floors it only flags.
    "bar": {"metric": "either", "top_n": 3, "row_cap_multiple": 2,
            "role": "the admission rule: each source keeps the top-N of each "
                    "chosen metric, capped at this many x N rows per source",
            "rows_cut": 1},
    "gate": {"jaccard_floor": 0.1, "rank_union_floor": 0.0,
             "window_mult": 2.0,
             "role": "advisory flags — each is evaluated and published per "
                     "row, and none of them removes a candidate. The bar "
                     "(pooling_bar_metric x pooling_bar_top_n) is what "
                     "decides admission."},
    "floor_flags": {"below_jaccard_floor": 2, "below_rank_union_floor": 3,
                    "outside_window": 1, "rows": 12},
    "tiers": {"matched": 4, "verified": 5, "nominated": 3},
    "pool_per_source": 0.75,
    "pool_size_warning": "",
    "cells": {"confirmed": 3, "pool_miss": 2, "type_miss": 1,
              "type_new": 1, "verified_only": 7},
    "body_ids": {"pool_miss": [501, 502], "verified_only": [601]},
    "pool_miss_by_type": {"DLp11": 1, "": 1},
    "morph": {"attempted": 8, "scored": 4, "qualified": 2, "capped": 0,
              "no_score": 4, "error": "", "gate_applied": True,
              "dropped_targets": 2,
              "warnings": ["BANC morphology is experimental: public "
                           "skeleton products mix L2 / full / µm sources"]},
    "input_fingerprint": {"git_rev": "abcdef1234567890",
                          "git_dirty": False,
                          "scanned_target_universe": 101995,
                          # the REAL shape `_store_identity` writes: epoch
                          # seconds under `mtime_s`. A fixture that invents a
                          # prettier key (`mtime`) lets a `@ None` ship — which
                          # is what a 2026-09-23 real-data run caught.
                          "mapper_snapshot": {"bytes": 123456,
                                              "mtime_s": 1789866123},
                          # the shape `_record_morph_stores` writes, from the
                          # 2026-09-25 FAFB->BANC run's own parameters.json
                          "morph_stores": {
                              "target": {
                                  "vector_cache": {
                                      "path": "/c/banc_v888/find_similar/"
                                              "morphology/skeleton__vectors_v2"
                                              ".parquet",
                                      "bytes": 2625339, "mtime_s": 1790199390},
                                  "vector_meta": {
                                      "path": "/c/banc_v888/find_similar/"
                                              "morphology/meta_v2.json",
                                      "bytes": 14892, "mtime_s": 1790268866},
                                  "skeleton_files": 2607,
                                  "newest_skeleton_mtime_s": 1790268617},
                              "source": {}}},
    "reading_notes": ["No cell is a recall measure."],
}


def _plain(html: str) -> str:
    """Report markup as a reader sees it: tags out, whitespace collapsed, so an
    assertion names the sentence rather than the span layout that carries it."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def _pool_row(**kw):
    r = {"target_bodyId": 500, "target_type": "DLp11",
         "leaf": "DLp11(out-map)", "best_source_bodyId": 101,
         "best_source_type": "s-LNv", "jaccard": 0.31, "jaccard_rank": 2,
         "rank_union": 0.44, "window_size": 16, "size_nm3": 120000.0,
         "size_universe_percentile": 61.0, "in_scope": True, "n_sources": 2,
         "dup": 1, "morph_gate": "scored", "morph_bar_kind": "null_bar",
         "morph_similarity": 0.66, "morph_pool_ref": None,
         "morph_bar": 0.55, "morph_qualified": True,
         "mapper_cell": "type_miss", "mapper_verdict": ""}
    r.update(kw)
    return r


def _as_pooling_run(run_dir: Path, pool=None, xval=XVAL, mode="pooling",
                    flat=False, sources=None):
    params = json.loads((run_dir / "parameters.json").read_text())
    params.update({"validation_mode": mode, "pooling_jaccard_floor": 0.1,
                   "pooling_rank_union_floor": 0.0,
                   "pooling_window_mult": 2.0, "pooling_bar_metric": "either",
                   "pooling_bar_top_n": 3,
                   "pooling_max_morph_targets": 0})
    (run_dir / "parameters.json").write_text(json.dumps(params))
    rows = pool if pool is not None else [_pool_row()]
    out = run_dir if flat else (run_dir / "pooling")
    if out != run_dir:
        out.mkdir(exist_ok=True)
    _write_csv(out / "pooling_pool.csv", POOL_HEADER,
               [[r.get(k, "") for k in POOL_HEADER] for r in rows])
    _write_csv(out / "pooling_candidates.csv",
               ["source_bodyId", "target_bodyId", "jaccard", "mapper_cell"],
               [[r["best_source_bodyId"], r["target_bodyId"], r["jaccard"],
                 r["mapper_cell"]] for r in rows])
    # the source axis, on the header `pool.apply_morph_gate`/`pool_by_source`
    # really writes — the tab reads it by name, so a fixture that renames a
    # column tests a file no run produces.
    _write_csv(out / "pooling_sources.csv", SOURCES_HEADER,
               [[r.get(k, "") for k in SOURCES_HEADER]
                for r in (sources if sources is not None
                          else _default_sources())])
    (out / "pooling_cross_validation.json").write_text(
        json.dumps(xval if isinstance(xval, dict) else {}))
    return run_dir


SOURCES_HEADER = [
    "source_bodyId", "source_type", "n_admitted", "n_in_pool", "n_refused",
    "tier", "best_target_bodyId", "best_bar_rank", "jaccard", "jaccard_rank",
    "rank_union", "rank_union_rank", "supported_by", "single_metric_support",
    "source_claimed", "morph_gate", "morph_bar_kind", "morph_similarity",
    "morph_pool_ref", "morph_bar", "morph_qualified", "verdict_for_pair",
    "no_finding"]


def _source_row(**kw):
    r = {"source_bodyId": 101, "source_type": "s-LNv", "n_admitted": 3,
         "n_in_pool": 2, "n_refused": 1, "tier": "matched",
         "best_target_bodyId": 500, "best_bar_rank": 1, "jaccard": 0.31,
         "jaccard_rank": 1, "rank_union": 0.44, "rank_union_rank": 2,
         "supported_by": "jaccard+rank_union", "single_metric_support": False,
         "source_claimed": True, "morph_gate": "scored",
         "morph_bar_kind": "null_bar", "morph_similarity": 0.66,
         "morph_pool_ref": "", "morph_bar": 0.55, "morph_qualified": True,
         "verdict_for_pair": "", "no_finding": ""}
    r.update(kw)
    return r


def _default_sources():
    return [_source_row(),
            # a source whose every finding the morphology bar refused: it
            # admitted rows and keeps none of them
            _source_row(source_bodyId=102, source_type="DN1a",
                        tier="verified", n_admitted=2, n_in_pool=0,
                        n_refused=2, source_claimed=False),
            # a source the bar admitted NOTHING for: present with a named
            # absence, never missing from the file
            _source_row(source_bodyId=103, source_type="DN1pA", tier="",
                        n_admitted=0, n_in_pool=0, n_refused=0,
                        best_target_bodyId="", best_bar_rank="", jaccard="",
                        morph_gate="", morph_qualified="",
                        source_claimed=False,
                        no_finding="no-admitted-target")]


def test_the_two_candidate_scopes_are_not_printed_as_one_number(run_dir: Path):
    """`mapped − in_pool = reached_as_candidates_only + holes` is a PER-TYPE
    identity, and the set-level field of the same name counts a wider set: every
    candidate-reached bodyId, including types the map never asserts. On the
    2026-09-25 male-cns family run the report printed 27 between "map-covered
    204" and "holes 5", so subtracting gave 15 ≠ 32 and the row looked like an
    arithmetic bug in the coverage engine. The engine is right; the row has to
    say which scope it is quoting, and show the term the identity uses.
    """
    html = build_report_document(collect_run_data(run_dir))
    tip = html.split("Reached only as candidates", 1)[1].split("</span>", 6)[0]
    assert "Two scopes" in tip
    assert "sums to 1" in tip          # Σ over per_type, the identity's term
    assert "INCLUDING types the map does not assert" in tip


def test_pooling_tab_carries_the_gate_cells_and_the_pool(run_dir: Path):
    """The Pooling tab is the only place a pooling run's result reads: the
    nested ladder's tabs are empty by construction there.  So the tab leads
    with the bar that admits each row, names the floors as the flags they
    became (no fitted provenance dressing a knob that measures pool size), and
    the cells keep their names."""
    _as_pooling_run(run_dir)
    html = build_report_document(collect_run_data(run_dir))
    assert ">Pooling</button>" in html
    assert "Pooling — the unsupervised scan" in html
    # the bar is what admits a row, so it leads the block; the floors state
    # that they flag rather than filter
    assert "either × top-3" in html
    assert "the union of BOTH metrics' own top-N" in _plain(html)
    assert "row cap 2 × N, 1 row(s) cut by it" in _plain(html)
    assert "advisory flag only" in html
    assert "configured 0.1000" in html
    assert "advisory flags" in html      # the run's own role line, quoted
    assert "dataset-fitted" not in html     # the fit is deleted, not hidden
    assert "103770 target neurons" in html
    assert "3 of the pool also sit in a branch" in html
    assert "type_miss 1" in html and "type_new 1" in html
    assert "graded matched/verified that this gate did not admit" in html
    assert "500" in html and "DLp11(out-map)" in html
    assert "0.660 vs 0.550 ✓" in html
    assert "morph_pool_ref" in html      # the hover names where the number came from
    assert "The harvest, by target type" in html
    assert "No cell is a recall measure." in html
    # what the last gate ACTUALLY measured, and what it could not
    assert "attempted 8 · scored 4 · qualified 2 · no-score 4" in html
    # and what the bar REFUSED, so a shrunken pool is not a smaller harvest
    assert "gate applied, 2 target(s) refused for scoring below the bar" \
        in html
    # …and the stores it measured against, because the cells are only
    # comparable across runs that read the same ones
    assert "git abcdef12 · target universe 101995" in html
    assert "mapper snapshot 123456 B @ 2026-09-20 01:02 UTC" in html
    # the native score's OWN store, which the profile caches do not cover
    assert ("morph vector cache 2625339 B @ 2026-09-23 21:36 UTC "
            "· 2607 skeletons") in html
    assert '@ None' not in html       # an unrendered key must never ship


def test_a_dirty_worktree_says_so_beside_the_rev(run_dir: Path):
    """`git_rev` names the COMMIT, not the code that scored when the tree carries
    uncommitted edits — the run that certified the `vectors_for` one-space fix
    printed the pre-fix rev for exactly that reason."""
    _as_pooling_run(run_dir)
    d = collect_run_data(run_dir)
    d['pooling_xval']['input_fingerprint']['git_dirty'] = True
    html = build_report_document(d)
    assert "git abcdef12-dirty" in html


def test_pooling_tab_headlines_the_source_axis(run_dir: Path):
    """The target pool answers "which neurons were found"; the mode's own
    question is per QUERIED SOURCE, and a source that found nothing must show
    up as a named absence rather than a smaller denominator.

    The tier counts here are per source (each source's chain-best finding), so
    they sum to the queried population — NOT to the row-level `tiers` the
    cross-validation record publishes, which count rows. Reading one as the
    other is the mistake this block exists to prevent.
    """
    _as_pooling_run(run_dir)
    text = _plain(build_report_document(collect_run_data(run_dir)))
    # (the block TITLE goes through `_esc`, so assert on a span of it that has
    # no apostrophe to escape)
    assert "Per source — the mode" in text and "own axis" in text
    assert "3 · across 3 source types" in text
    assert "2 of 3 · 5 admitted rows total" in text
    assert "matched 1 · verified 1 · nominated 0 · 1 found nothing" in text
    assert "1 of 3 · 1 saw every finding refused by the bar" in text
    assert "1 — post-hoc advice only" in text
    assert "0.750 targets per queried source" in text
    # the ladder's vocabulary is pooling's own: no `borderline`, no `relative`
    assert "borderline" not in text.split("Per source")[1][:400]


def test_pooling_tiers_are_glossed_as_poolings_own_ladder(run_dir: Path):
    """`matched` and `verified` mean something else in the nested modes — the
    mapper's asserted tier and its review tier, both decided by POOL
    membership. A pooling run prints the same two words over a bar, so the tab
    has to say which ladder it is quoting; without this a reader hovers
    `matched` in a pooling block and gets the supervised definition.
    """
    _as_pooling_run(run_dir)
    html = build_report_document(collect_run_data(run_dir))
    # `_term(key, label)` prints the LABEL (escaped) and puts the definition in
    # the tip, so the assertion reaches into the tip rather than the key.
    assert "Tier of each source" in html
    tip = html.split("best finding", 1)[1].split("</span>", 4)[0]
    assert "OWN ladder" in tip and "branch pool" in tip
    assert "There is no `borderline`" in tip


def test_a_pooling_run_leads_its_own_answer_not_the_ladder(run_dir: Path):
    """Issue 13: the hero quoted the supervised levels (map-covered /
    mutual-best / fill-proposed) for a run whose mode is parallel to that
    ladder, while the pool itself sat two screens down.

    The headline now leads with the source axis and names the ladder as what
    it is — the same run's supervised path, not this mode's answer.
    """
    _as_pooling_run(run_dir)
    html = build_report_document(collect_run_data(run_dir))
    hero = html.split('<p class="report-subtitle">', 1)[1].split('</p>', 1)[0]
    assert "queried sources" in hero and "reached ≥1 candidate" in hero
    assert "kept one after morphology" in hero and "distinct pooled targets" in hero
    assert "Supervised ladder, same run:" in hero
    # the ladder is still published, just not mistaken for the headline
    assert "map-covered" in hero
    assert hero.index("distinct pooled targets") < hero.index("map-covered")


def test_a_native_row_prints_the_pair_the_gate_graded(run_dir: Path):
    """The native track decides a `scored` row by its pool reference against
    the native floor, NOT by its Track-A number.

    Measured on the 2026-09-24 male-cns pooling run: 14 of 41 scored rows
    published `morph_similarity` below their published `morph_bar` while
    `morph_qualified` said yes, because the bar column carried the per-source
    NULL bar under a `native` kind label — a passing row that reads as a broken
    gate.  The cell now prints the deciding pair, and the ✓ is recomputable
    from the numbers beside it.
    """
    _as_pooling_run(run_dir, pool=[_pool_row(
        morph_bar_kind="native", morph_similarity=0.054,
        morph_pool_ref=0.42, morph_bar=0.296, morph_qualified=True)])
    text = _plain(build_report_document(collect_run_data(run_dir)))
    assert "native 0.420 vs 0.296 ✓" in text
    assert "0.054" not in text          # the Track-A number is not the grade
    # a refusal still reads as one, with the same pair named
    _as_pooling_run(run_dir, pool=[_pool_row(
        morph_bar_kind="native", morph_similarity=0.054,
        morph_pool_ref=0.12, morph_bar=0.296, morph_qualified=False)])
    text = _plain(build_report_document(collect_run_data(run_dir)))
    assert "native 0.120 vs 0.296 ✗" in text
    # and a scored row the native track has no evidence for is named as a
    # missing measurement, never as a refusal (decision 3 keeps those apart)
    _as_pooling_run(run_dir, pool=[_pool_row(
        morph_bar_kind="native", morph_similarity=0.054,
        morph_pool_ref=None, morph_bar=0.296, morph_qualified=None)])
    text = _plain(build_report_document(collect_run_data(run_dir)))
    assert "native: no evidence for this pair" in text
    # the row itself carries no refusal mark (the column hover legitimately
    # explains what a scored ✗ would have meant)
    row = text.split('500 DLp11(out-map)')[1].split('type_miss')[0]
    assert "✗" not in row


def test_the_scored_against_line_names_what_it_cannot_record(run_dir: Path):
    """A run whose code tree has no git rev, or whose snapshot mtime is gone,
    must say WHICH part is missing: a line that quietly drops the rev reads as
    "same code as the other run" to someone comparing two."""
    _as_pooling_run(run_dir)
    d = collect_run_data(run_dir)
    d['pooling_xval']['input_fingerprint'] = {
        'git_rev': None, 'scanned_target_universe': 101995,
        'mapper_snapshot': {'bytes': 123456, 'mtime_s': None}}
    text = re.sub(r'<[^>]+>', ' ', build_report_document(d))
    assert 'git rev not recorded' in text
    assert 'target universe 101995' in text
    assert 'mapper snapshot 123456 B' in text
    assert '123456 B @' not in text      # no stamp to name, no dangling '@'


def test_the_pooling_caveats_reach_the_warning_notes(run_dir: Path):
    """A pooling run whose morph gate scored 4 of 8 attempted targets must not
    read as a morphologically cleared pool: the caveat is published in the
    report AND in user_warning_notes.txt from ONE builder, so the two cannot
    drift apart."""
    _as_pooling_run(run_dir)
    d = collect_run_data(run_dir)
    line = next(w for w in collect_warnings(d) if w.startswith('[pooling] '))
    assert 'scored 4/8 attempted targets (4 `no-score`)' in line
    assert 'BANC morphology is experimental' in line
    html = build_report_document(d)
    text = re.sub(r'<[^>]+>', '', html)
    # the same sentence the warning file carries is readable in the report
    assert 'scored 4/8 attempted targets (4 `no-score`)' in text
    assert 'BANC morphology is experimental' in text
    # no caveat to state → no line (a healthy run is not noise)
    d['pooling_xval']['morph'] = {'attempted': 8, 'scored': 8,
                                  'qualified': 8, 'capped': 0,
                                  'no_score': 0, 'error': '',
                                  'warnings': []}
    assert not [w for w in collect_warnings(d) if w.startswith('[pooling] ')]


def test_the_mode_that_did_not_pool_says_it_is_parallel(run_dir: Path):
    """A family run's empty Pooling tab must not read as a lost artifact: the
    mode ladder and `pooling` are different questions."""
    html = build_report_document(collect_run_data(run_dir))
    assert ">Pooling</button>" in html
    assert "is a rung of the nested ladder; `pooling` is PARALLEL" in html
    assert "Pooling — the unsupervised scan" not in html


def test_a_failed_pooling_pass_is_not_a_null_result(run_dir: Path):
    """The writer exports an empty pool for a crashed pass too — the report
    has to keep "did not finish" apart from "nothing qualified"."""
    _as_pooling_run(run_dir, pool=[], xval={"error": "RuntimeError: boom"})
    html = build_report_document(collect_run_data(run_dir))
    assert "RuntimeError: boom" in html
    assert "not a universe with no homolog" in html
    assert "The bar admitted no candidate" in html


def test_the_pool_table_has_no_cross_dataset_agreement_column(run_dir: Path):
    """`targets_corroborated` was deleted with the sibling join (user,
    2026-09-23): pooling neglects the type name by design, so counting how
    many other target datasets neglected it the same way grades nothing — and
    a blank cell that only ever meant "you did not ask for that join" cost a
    column of reading.  The cell count is the tripwire against quietly
    re-growing it; the refused-candidate count is what replaced the column."""
    _as_pooling_run(run_dir)
    html = build_report_document(collect_run_data(run_dir))
    row = html.split(">500 <", 1)[1].split("</tr>", 1)[0]
    assert row.count("</td>") == 6        # the split drops the first opening tag
    assert "cross-target" not in html
    # the morphology cell still keeps a no-verdict row apart from a refusal
    assert "0.660 vs 0.550 ✓" in row


def test_an_untouched_morph_gate_is_a_word_not_a_cross(run_dir: Path):
    """`not-attempted-cap` means the budget never looked; rendering it as ✗
    would turn a no-sample into a rejection.  The gate itself says so: off is
    a word, and it is the only case where a below-bar row can still be
    published."""
    _as_pooling_run(run_dir, pool=[_pool_row(
        morph_gate="not-attempted-cap", morph_similarity="", morph_bar="",
        morph_qualified="")])
    html = build_report_document(collect_run_data(run_dir))
    row = html.split(">500 <", 1)[1].split("</tr>", 1)[0]
    assert "not-attempted-cap" in row and "✗" not in row and "✓" not in row


def test_pooling_artifacts_resolve_from_a_flat_run_folder(run_dir: Path):
    """The layout registry moved the exports under `pooling/`; runs written
    before it stay readable, because the report must regenerate any past
    folder from the folder alone."""
    _as_pooling_run(run_dir, flat=True)
    html = build_report_document(collect_run_data(run_dir))
    assert "Pooling — the unsupervised scan" in html
    assert "either × top-3" in html
    assert "advisory flag only" in html


def test_an_attempted_scene_that_crashed_is_named_not_counted_away(run_dir: Path):
    """A crash must not be reported as the "nothing renderable" policy.

    The Scenes tab used to say "N scenes rendered — only types with renderable
    expansion content get a scene" whatever the reason, so a male-cns family run
    that lost five parents to a ZeroDivisionError read as a design decision. Both
    records of the crash now surface it: the run log's `! scene X failed:` line
    (which is what archived runs have) and the folder's SCENE_FAILED.txt (which
    is what new runs write).
    """
    readme = run_dir / "README.txt"
    lines = readme.read_text(encoding="utf-8").splitlines()
    # the parser only reads `!` lines from inside the "Run log:" block, so the
    # failure goes there, after a whole log line (the fixture's own log lines
    # wrap, so a substring splice would cut one in half)
    at = next(i for i, ln in enumerate(lines) if 'self-check [A]' in ln)
    lines.insert(at + 1, '    ! scene s-LNv failed: float division by zero')
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    marker = run_dir / "visualization" / "plot-3d_stub_branches_q1_x"
    marker.mkdir()
    (marker / "SCENE_FAILED.txt").write_text(
        "this scene did not render\nparent type: q1\n"
        "error: ZeroDivisionError: float division by zero\n",
        encoding="utf-8")

    d = collect_run_data(run_dir)
    assert any('scene s-LNv failed' in ln
               for ln in d['readme']['bang_lines'])       # parsed as a warning
    assert _scene_failures(d) == [
        ("q1", "ZeroDivisionError: float division by zero"),
        ("s-LNv", "float division by zero")]
    html = _scenes_tab(d)
    assert "s-LNv" in html and "attempted and FAILED" in html
    # the policy sentence stays, but no longer speaks for the crashes
    assert "nothing renderable get no scene" in html
