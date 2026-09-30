"""Pins for the 2026-09-30 UI/docs/report audit fix round
(docs/audits/REPOSITORY_AUDIT_2026-09-30-ui-docs-reports.md).

The output-guide fixes are enforced by the existing consistency suites
(test_output_guide.py: every spec column must have a glossary entry);
these pins cover the payload/behavior fixes that have no cheap event
harness.
"""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
for entry in ("ui", "src"):
    if str(PROJECT / entry) not in sys.path:
        sys.path.insert(0, str(PROJECT / entry))


def test_shortest_paths_sends_the_none_keyword_default():
    """F-UI-004: an empty keyword box must send ['None'] like Complete
    Paths — the backend's own 'None' default is overridden by an explicit
    empty list, so shortest runs used to keep None-labeled paths while
    Complete Paths dropped them."""
    src = (PROJECT / "ui" / "tabs" / "find_shortest.py").read_text(
        encoding="utf-8")
    assert "keyword_filter.get_value()[1]] or ['None']" in src


def test_auto_threshold_mode_requires_two_datasets():
    """F-UI-003: the guides' contract — Auto mode derives thresholds from
    cross-dataset overlap and needs >= 2 selected datasets; the UI refuses
    a one-dataset Auto run instead of silently bootstrapping vertical
    rows."""
    src = (PROJECT / "ui" / "tabs" / "inter_dataset.py").read_text(
        encoding="utf-8")
    assert "Auto threshold mode needs at least two selected" in src


def test_tmvev_report_names_its_real_tab_count():
    """F-UI-009/F-UI-034: the output guide and OUTPUT_FILES must say the
    report's real tab count (14) and the report hero must point at the
    Coverage tab (there is no Mapping tab)."""
    from ui import output_guide as guide
    spec = guide.TOOL_GUIDE_SPECS["type_mapping_validation"]
    report_entry = next(f for f in spec["files"]
                        if f["pattern"] == "report.html")
    assert "14 tabs" in report_entry["description"]
    assert "Homolog · forward" in report_entry["description"]
    report_src = (PROJECT / "src" / "comparison" /
                  "mapping_validation_report.py").read_text(encoding="utf-8")
    assert "see the Mapping tab" not in report_src
    assert "see the Coverage tab" in report_src


def test_morph_scene_pattern_matches_scene_files():
    """F-UI-010: the morphology-comparison spec's 3D-scene pattern must
    actually match files (a trailing 'plot-3d_*/' never matches a file
    path, silently dropping the entry from every exported guide)."""
    from fnmatch import fnmatch
    from ui import output_guide as guide
    spec = guide.TOOL_GUIDE_SPECS["morphology_comparison"]
    pattern = next(f["pattern"] for f in spec["files"]
                   if f["pattern"].startswith("plot-3d_"))
    assert fnmatch("plot-3d_male-cns_v1_0/index.html", pattern)
    # the marker lives in the TM VEV spec now, not the skeleton-viewer spec
    viewer_spec = guide.TOOL_GUIDE_SPECS["plot3d_skeleton"]
    assert not any(f["pattern"] == "SCENE_FAILED.txt"
                   for f in viewer_spec["files"])
    tmvev_spec = guide.TOOL_GUIDE_SPECS["type_mapping_validation"]
    assert any("SCENE_FAILED.txt" in f["pattern"] for f in tmvev_spec["files"])


def test_similarity_specs_carry_the_v22_columns():
    """F-UI-008: the similarity-matrix spec entries and the glossary must
    carry the ten v2.2 advanced-metric columns the writer emits."""
    from ui import output_guide as guide
    spec = guide.TOOL_GUIDE_SPECS["inter_dataset"]
    threshold_entry = next(f for f in spec["files"]
                           if f["pattern"].endswith("similarity_threshold_*.csv"))
    columns = set(threshold_entry["columns"])
    for column in ("path_jaccard_similarity", "path_top20_overlap",
                   "hop_profile_w1", "netsimile_similarity", "coverage_d1",
                   "coverage_d2", "coverage_min", "top20_overlap",
                   "strength_w1_out", "strength_w1_in"):
        assert column in columns, column
        assert column in guide.COLUMN_GLOSSARY, column
