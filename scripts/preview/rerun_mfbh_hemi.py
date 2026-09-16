"""Real-data verification: separate_hemispheres=True MFBH run (plan R1).

Replicates the 213447 run's installed query set (5 vertical + 4 horizontal
rows) with separate_hemispheres=True, exercising the hemi-aware alignment
lanes on real suffixed data. Warm caches keep this fast.
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
       "cross-dataset_aMe1_etc_to_PPL101_etc_MFBH_20260915_213447")
params = ComparisonParameters.from_dict(
    json.load(open(f"{SRC}/parameters.json")))
params.separate_hemispheres = True
params.symmetry_analysis = True

print("[HEMI-RUN] separate_hemispheres:", params.separate_hemispheres)
print("[HEMI-RUN] queries:", len(params.threshold_combinations or []))

analyzer = ComparisonAnalyzer(params, verbose=True)
analyzer.run_comparison()
analyzer.export_results()

print("[HEMI-RUN] DONE. output:", params.full_output_path)
for ds, info in analyzer.dataset_coverage().items():
    print("[HEMI-RUN] coverage:", ds, "->", info)
