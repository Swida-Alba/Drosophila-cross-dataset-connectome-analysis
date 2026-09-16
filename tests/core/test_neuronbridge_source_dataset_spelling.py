"""Regression tests for stable source-dataset spellings in outputs.

Fresh NeuronBridge fetches used to stamp the published identity
(``flywire_fafb:v783``) while cache loads stamped the caller's spelling
(``flywire_FAFB_v783``), splitting ``datasets_labeled`` /
``matched_datasets`` counts for the same release.  Identity-equal
fetches must now adopt the caller's spelling.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from neuronbridge_finder import NeuronBridgeFinder  # noqa: E402


class TestCanonicalSourceDataset:
    def test_identity_equal_fetch_adopts_caller_spelling(self):
        assert NeuronBridgeFinder._canonical_source_dataset(
            "flywire_FAFB_v783", "flywire_fafb:v783") == "flywire_FAFB_v783"

    def test_exact_spellings_pass_through(self):
        assert NeuronBridgeFinder._canonical_source_dataset(
            "male-cns:v0.9", "male-cns:v0.9") == "male-cns:v0.9"

    def test_missing_metadata_falls_back_to_expected(self):
        assert NeuronBridgeFinder._canonical_source_dataset(
            "male-cns:v1.0", None) == "male-cns:v1.0"

    def test_unknown_expected_keeps_the_fetched_identity(self):
        assert NeuronBridgeFinder._canonical_source_dataset(
            "unknown", "male-cns:v0.9") == "male-cns:v0.9"

    def test_genuinely_different_identity_is_never_rewritten(self):
        assert NeuronBridgeFinder._canonical_source_dataset(
            "optic-lobe:v1.1", "manc:v1.2.1") == "manc:v1.2.1"

