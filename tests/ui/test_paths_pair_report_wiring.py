"""Tests for the paths_pair_report UI wiring (plan-paths-pair-report §7).

Pins the TOOL_REGISTRY entry, the TOOL_GUIDE_SPECS entry, and that the
registry-facing wrapper class actually generates a report.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paths_pair_report import PathsPairReportTool  # noqa: E402
from ui.runner import TOOL_REGISTRY  # noqa: E402
import ui.output_guide as guide  # noqa: E402

CSV_COLUMNS = [
    "path", "weights", "probabilities", "ratios", "min_weight",
    "path_prob", "min_ratio", "length", "nt_types",
]


def test_registry_entry_present():
    entry = TOOL_REGISTRY["paths_pair_report"]
    assert entry["class"] == "PathsPairReportTool"
    assert entry["var"] == "pair_report"
    assert entry["init_method"] is None
    assert entry["methods"]["generate"] == "pair_report.generate(**method_params)"


def test_guide_spec_matches_registry():
    # the registry-wide invariant test (test_output_guide) enforces the
    # guide spec's existence; here pin the file surface itself.
    spec = guide.TOOL_GUIDE_SPECS["paths_pair_report"]
    patterns = {f["pattern"] for f in spec["files"]}
    assert {"path_report.html",
            "paths_pair_breakdown/pair_breakdown_paths.csv",
            "paths_pair_breakdown/pair_breakdown_intermediates.csv"} <= patterns


def test_tool_wrapper_generates(tmp_path):
    run = tmp_path / "find-paths-complete_FAFB_A_to_B_L1w3_20260101_000000"
    run.mkdir()
    pd.DataFrame([{
        "path": "A->B", "weights": "[4]", "probabilities": "[0.5]",
        "ratios": "[0.1]", "min_weight": 4, "path_prob": 0.5,
        "min_ratio": 0.1, "length": 1, "nt_types": '["ACH"]',
    }], columns=CSV_COLUMNS).to_csv(
        run / "A_to_B_allpaths_type.csv", index=False)
    tool = PathsPairReportTool()
    out = tool.generate(str(run))
    assert Path(out).name == "path_report.html"
    assert (run / "paths_pair_breakdown" / "pair_breakdown_paths.csv").exists()
