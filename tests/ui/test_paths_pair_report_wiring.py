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


def test_hook_skips_non_pathfinding_runs(tmp_path):
    empty = tmp_path / "plot-3d_FAFB_neuronX_20260101_000000"
    empty.mkdir()
    assert has_paths_tables(empty) is False
    generated = []

    def fake_generate(run_dir, **kwargs):  # pragma: no cover - guard probe
        generated.append(run_dir)

    logs = []
    _maybe_generate_pair_report(
        empty, log=lambda msg, level="info": logs.append(msg),
        progress=lambda phase, label="": None)
    assert generated == [] and not (empty / REPORT_NAME).exists()


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
    pathfinding tools (the files now exist for every UI run)."""
    for tool in ("find_path", "find_shortest", "inter_dataset"):
        patterns = {f["pattern"] for f in guide.TOOL_GUIDE_SPECS[tool]["files"]}
        assert "path_report.html" in patterns, tool
        assert ("paths_pair_breakdown/pair_breakdown_paths.csv") in patterns, tool
