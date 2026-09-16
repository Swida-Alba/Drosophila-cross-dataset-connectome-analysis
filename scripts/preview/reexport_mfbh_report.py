"""Re-export the completed 205545 MFBH run with the current code.

Resume mode (skip_existing): cached results are reloaded, nothing is
re-enumerated, and the report + exports are regenerated so the round-2
rendering fixes (intermediates styling, muted markers, etc.) show up in a
real report. report_layout='both' also produces the tabbed variant for
verification.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))

from comparison import ComparisonParameters, ComparisonAnalyzer  # noqa: E402

SRC = ("/Users/apple/Local/connection_data/DROCAT_data/"
       "cross-dataset_aMe1_etc_to_PPL101_etc_MFBH_20260915_205545")
params = ComparisonParameters.from_dict(
    json.load(open(f"{SRC}/parameters.json")))
params.report_layout = "both"  # tabbed comparison_report.html + legacy copy

analyzer = ComparisonAnalyzer(params, verbose=True)
analyzer.run_comparison(skip_existing=True)
analyzer.export_results()
print("[REEXPORT] DONE")
for ds, info in analyzer.dataset_coverage().items():
    print("[REEXPORT] coverage:", ds, "->", info)
