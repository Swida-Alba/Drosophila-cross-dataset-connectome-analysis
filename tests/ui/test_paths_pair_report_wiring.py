"""Tests for the paths_pair_report UI embedding (plan-paths-pair-report
§7 round 6: the report is generated automatically after every
pathfinding run by the runner's post-run hook — no standalone button, no
TOOL_REGISTRY entry).

Pins the hook behavior (generate / skip / never-raise), the registry
absence, and the run-guide file spec that surfaces the report files.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paths_pair_report import (  # noqa: E402
    OUTPUT_DIR_NAME,
    PATHS_CSV_NAME,
    REPORT_NAME,
    has_paths_tables,
)
from ui.runner import TOOL_REGISTRY, _maybe_generate_pair_report  # noqa: E402
import ui.output_guide as guide  # noqa: E402

CSV_COLUMNS = [
    "path", "weights", "probabilities", "ratios", "min_weight",
    "path_prob", "min_ratio", "length", "nt_types",
]


def _make_run(tmp_path: Path, rows=None, name="find-paths-complete_FAFB_A_to_B_L1w3_20260101_000000"):
    run = tmp_path / name
    run.mkdir(parents=True)
    if rows is None:
        rows = [{
            "path": "A->B", "weights": "[4]", "probabilities": "[0.5]",
            "ratios": "[0.1]", "min_weight": 4, "path_prob": 0.5,
            "min_ratio": 0.1, "length": 1, "nt_types": '["ACH"]',
        }]
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(
        run / "A_to_B_allpaths_type.csv", index=False)
    return run


def test_no_registry_entry_and_no_button():
    """The report is embedded in the run flow, not a standalone tool."""
    assert "paths_pair_report" not in TOOL_REGISTRY
    common_src = (PROJECT_ROOT / "ui" / "components" / "common.py").read_text(
        encoding="utf-8")
    assert "pair_report_button" not in common_src


def test_hook_generates_into_run_folder(tmp_path):
    run = _make_run(tmp_path)
    logs, phases = [], []
    _maybe_generate_pair_report(
        run, log=lambda msg, level="info": logs.append((level, msg)),
        progress=lambda phase, label="": phases.append(phase))
    assert (run / REPORT_NAME).exists()
    assert (run / OUTPUT_DIR_NAME / PATHS_CSV_NAME).exists()
    assert any("pair report" in msg.lower() for _, msg in logs)
    assert "pair-report" in phases


def _install_generator_probe(monkeypatch):
    """Replace the hook's own `generate_paths_pair_report`.

    The hook imports it INSIDE the function (`src.paths_pair_report`, then
    `paths_pair_report`), so both module names are patched — patching neither
    (the previous state) left `generated == []` unfailable: the real generator
    ran, and only the missing file proved anything."""
    import sys

    import paths_pair_report as ppr
    called = []

    def fake_generate(run_dir, **kwargs):
        called.append(Path(run_dir))
        return None

    # patch EVERY module alias the hook may import from, and through
    # monkeypatch so each is restored (a raw assignment leaked the fake into
    # the next test, which then found no report file at all)
    monkeypatch.setattr(ppr, "generate_paths_pair_report", fake_generate)
    for mod in ("src.paths_pair_report", "paths_pair_report"):
        if mod in sys.modules and sys.modules[mod] is not ppr:
            monkeypatch.setattr(sys.modules[mod],
                                "generate_paths_pair_report", fake_generate)
    return called


def test_hook_skips_non_pathfinding_runs(tmp_path, monkeypatch):
    empty = tmp_path / "plot-3d_FAFB_neuronX_20260101_000000"
    empty.mkdir()
    assert has_paths_tables(empty) is False
    called = _install_generator_probe(monkeypatch)
    _maybe_generate_pair_report(
        empty, log=lambda msg, level="info": None,
        progress=lambda phase, label="": None)
    assert called == [], f"the guard let a non-pathfinding run through: {called}"
    assert not (empty / REPORT_NAME).exists()


def test_hook_calls_the_generator_for_a_pathfinding_run(tmp_path, monkeypatch):
    """The same probe, the other way round: a pathfinding folder that holds a
    table MUST reach the generator — otherwise the empty `called` list above
    would pass for a hook that never generates at all."""
    run = tmp_path / "find-paths-complete_MCNS_S_to_T_L1w1_20260101_000000"
    (run / "data_details").mkdir(parents=True)
    pd.DataFrame({"path": ["S->T"], "weights": ["[3]"], "min_weight": [3],
                  "length": [2]}).to_csv(
        run / "src_to_tgt_allpaths_type.csv", index=False)
    called = _install_generator_probe(monkeypatch)
    _maybe_generate_pair_report(
        run, log=lambda msg, level="info": None,
        progress=lambda phase, label="": None)
    assert [Path(c).name for c in called] == [run.name]


def test_hook_renders_the_empty_state_for_a_zero_path_run(tmp_path):
    """F-EX-010: the hook's gate was "has a paths table", so a run that
    found nothing got no report at all. A pathfinding folder without any
    table now still receives path_report.html (the empty state)."""
    run = tmp_path / "find-paths-complete_MCNS_S_to_T_L1w1_20260101_000000"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "min synapse number:            5\n"
        "weight basis:                  synapse\n"
        "max interlayer:                1\n"
        "dataset:                       male-cns:v1.0\n", encoding="utf-8")
    assert has_paths_tables(run) is False
    _maybe_generate_pair_report(
        run, log=lambda msg, level="info": None,
        progress=lambda phase, label="": None)
    text = (run / REPORT_NAME).read_text(encoding="utf-8")
    assert "This run found no paths" in text
    assert "Min Synapse Count ≥ 5" in text


def test_hook_never_raises_on_broken_inputs(tmp_path):
    run = _make_run(tmp_path, rows=[{"path": None}])
    _maybe_generate_pair_report(
        run, log=lambda msg, level="info": None,
        progress=lambda phase, label="": None)  # must not raise
    # a non-existent folder must also be a no-op
    _maybe_generate_pair_report(
        tmp_path / "missing", log=lambda *a, **k: None,
        progress=lambda *a, **k: None)


def test_pair_report_files_in_pathfinding_guide_spec():
    """The run-guide file spec surfaces the report files for the three
    pathfinding tools AND find_network (Phase B3: network runs emit the
    1-hop table, so the runner hook generates the same report)."""
    for tool in ("find_path", "find_shortest", "inter_dataset",
                 "find_network"):
        patterns = {f["pattern"] for f in guide.TOOL_GUIDE_SPECS[tool]["files"]}
        assert "path_report.html" in patterns, tool
        assert ("paths_pair_breakdown/pair_breakdown_paths.csv") in patterns, tool


def test_network_runs_feed_the_pair_report_hook(tmp_path):
    """Phase B3: a FindNetwork run emits `*_allpaths_type.csv` (the 1-hop
    table), so `has_paths_tables` matches network folders and the runner
    post-run hook gives network runs the same pair report the
    pathfinding tools get."""
    run = _make_run(
        tmp_path,
        name="find-network_FAFB_clock_21types_w3r0p0_20260101_000000")
    assert has_paths_tables(run)
    # Phase B2: network runs also carry the refill-full mirror (always
    # produced) — the pair report's full-scope refill builder reads it
    # and must survive a unit with NO intermediates rows (1-hop table):
    # the stale-`key` UnboundLocalError regression lived exactly here.
    refill_dir = run / "type_level_refill"
    refill_dir.mkdir()
    pd.DataFrame([{
        "path": "A->B", "weights": "[4]", "ratios": "[0.1]",
        "min_weight": 4, "min_ratio": 0.1, "length": 1,
        "nt_types": '["ACH"]', "coverage": "[1/1, 1/1]",
    }], columns=[
        "path", "weights", "ratios", "min_weight", "min_ratio",
        "length", "nt_types", "coverage",
    ]).to_csv(refill_dir / "refill_full_carrying_paths_type.csv",
              index=False)
    logs = []
    _maybe_generate_pair_report(
        run, log=lambda msg, level="info": logs.append(msg),
        progress=lambda phase, label="": None)
    assert (run / REPORT_NAME).exists()
