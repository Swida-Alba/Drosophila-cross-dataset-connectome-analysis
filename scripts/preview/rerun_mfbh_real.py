"""Real-data verification re-run of the MFBH scenario (plan round 2).

Replicates the user's cross-dataset_aMe1_etc_to_PPL101_etc_MFBH_20260915_184948
run from its saved parameters.json, but starts in threshold_mode='auto' so
the density bootstrap re-measures windows and installs fresh vertical +
horizontal rows (the original run's horizontal rows were zero because of
the hemibrain capture failure that this round fixes).
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
       "cross-dataset_aMe1_etc_to_PPL101_etc_MFBH_20260915_184948")
params_data = json.load(open(f"{SRC}/parameters.json"))

params = ComparisonParameters.from_dict(params_data)
# Fresh auto bootstrap: re-measure the density windows and install the
# vertical + horizontal alignment rows (threshold_mode='auto' is the gate
# in run_all_analyses; install_auto_combinations flips it back to
# 'combinations' afterwards).
params.threshold_mode = "auto"
params.threshold_auto = True
params.threshold_combinations = []

print("[RERUN] datasets:", params.get_dataset_names())
print("[RERUN] mode:", params.threshold_mode, "| auto:", params.threshold_auto)

analyzer = ComparisonAnalyzer(params, verbose=True)
results = analyzer.run_comparison()
analyzer.export_results()

out = params.full_output_path
print("[RERUN] DONE. output:", out)
cov = analyzer.dataset_coverage()
for ds, info in cov.items():
    print("[RERUN] coverage:", ds, "->", info)
