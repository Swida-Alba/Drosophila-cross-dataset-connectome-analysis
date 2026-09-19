"""Tests for the exported per-run user guide (ui/output_guide.py)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import ui.config as cfg
import ui.output_guide as guide
from ui.runner import TOOL_REGISTRY


PARAMS = {
    "dataset": "male-cns:v1.0",
    "sourceNeurons": {"type": ["aMe12"]},
    "targetNeurons": {"type": ["PPL1*"]},
    "max_interlayer": 5,
    "skip_bodyId": True,
}


def _make_pathfinding_folder(root: Path, with_warnings: bool = True) -> Path:
    """Synthetic Complete Paths run folder with representative outputs."""
    run = root / "find_path_20240101_120000"
    (run / "data_details").mkdir(parents=True)
    (run / "visualization" / "visualization_data").mkdir(parents=True)
    (run / "aMe12_to_PPL101_allpaths_type.csv").write_text(
        "path,weights,probabilities,ratios,min_weight,path_prob,min_ratio,"
        "length,nt_types\n", encoding="utf-8")
    (run / "data_details" / "connection_type.csv").write_text(
        "type_pre,type_post,weight,connection_ratio,traversal_probability,"
        "block_probability,nt_type\n", encoding="utf-8")
    (run / "visualization" / "Network_20240101.html").write_text(
        "<html></html>", encoding="utf-8")
    if with_warnings:
        (run / guide.WARNING_FILENAME).write_text(
            "Note: 3 source neurons had no downstream partners.\n",
            encoding="utf-8")
    return run


class TestSpecIntegrity:
    """Content model consistency across glossary, specs, and registry."""

    def test_all_spec_columns_exist_in_glossary(self):
        for tool_name, spec in guide.TOOL_GUIDE_SPECS.items():
            for file_spec in spec["files"]:
                for column in file_spec.get("columns", []):
                    assert column in guide.COLUMN_GLOSSARY, (
                        f"{tool_name}: column '{column}' "
                        f"(pattern {file_spec['pattern']}) missing from "
                        f"COLUMN_GLOSSARY")

    def test_every_registered_tool_has_guide_spec(self):
        missing = set(TOOL_REGISTRY) - set(guide.TOOL_GUIDE_SPECS)
        assert not missing

    def test_glossary_entries_are_description_range_pairs(self):
        for column, entry in guide.COLUMN_GLOSSARY.items():
            assert isinstance(entry, tuple) and len(entry) == 2, column
            assert entry[0], f"empty description for {column}"

    def test_type_mapping_run_folder_is_fully_described(self, tmp_path):
        """The TM VEV writer resolves every export through RUN_FILE_LAYOUT, so
        the guide must carry a folder-qualified entry for each: an uncovered
        file drops into the generic "Tabular data file." fallback."""
        from comparison.mapping_validation import RUN_FILE_LAYOUT

        run = tmp_path / "type-map-validation_dsA_to_dsB_20260919_000000"
        for name, folder in RUN_FILE_LAYOUT.items():
            path = run / folder / name if folder else run / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        content = guide.assemble_run_content(run, "type_mapping_validation",
                                             {})
        assert [x["path"] for x in content["leftovers"]] == []
        patterns = {entry["pattern"] for entry in
                    guide.TOOL_GUIDE_SPECS["type_mapping_validation"]["files"]}
        for name, folder in RUN_FILE_LAYOUT.items():
            want = f"{folder}/{name}" if folder else name
            assert want in patterns, want


class TestPreviewRegistry:
    """The preview flags feed the UI Output panel's result previews."""

    def test_every_tool_yields_valid_preview_views(self):
        for tool_name in TOOL_REGISTRY:
            for view in guide.preview_views(tool_name):
                assert view["pattern"].endswith(".csv"), (tool_name, view)
                assert view["title"], (tool_name, view)
                assert view["description"], (tool_name, view)

    def test_homolog_preview_files_are_described(self):
        """All previewed homolog views exist as guide spec entries, so
        exported run guides describe them instead of the generic fallback."""
        patterns = {entry["pattern"]
                    for entry in guide.TOOL_GUIDE_SPECS["find_homologs"]["files"]}
        assert "results/type_level_results.csv" in patterns
        previewed = {view["pattern"]
                     for view in guide.preview_views("find_homologs")}
        assert previewed == {
            "results/bodyid_results.csv",
            "results/type_summary.csv",
            "results/type_level_results.csv",
        }

    def test_similarity_profile_spec_removed_with_mode(self):
        """The connectivity-similarity mode was folded into Connectivity →
        Find Similar, so its dedicated tool spec is gone."""
        assert "find_similar_profile" not in guide.TOOL_GUIDE_SPECS
        assert "find_similar_profile" not in TOOL_REGISTRY

    def test_image_only_tool_has_no_previews(self):
        assert guide.preview_views("flylight_download") == []


class TestAssemble:
    def test_files_matched_to_spec_entries(self, tmp_path):
        run = _make_pathfinding_folder(tmp_path)
        content = guide.assemble_run_content(run, "find_path", PARAMS)
        assert content["title"] == "Complete Paths"
        matched = {m for e in content["entries"] for m in e["matched"]}
        assert "aMe12_to_PPL101_allpaths_type.csv" in matched
        assert "data_details/connection_type.csv" in matched
        assert "visualization/Network_20240101.html" in matched
        assert content["warnings"] is not None

    def test_no_file_listed_twice(self, tmp_path):
        run = _make_pathfinding_folder(tmp_path)
        content = guide.assemble_run_content(run, "find_path", PARAMS)
        listed = [m for e in content["entries"] for m in e["matched"]]
        assert len(listed) == len(set(listed))

    def test_guide_file_excluded_from_listings(self, tmp_path):
        run = _make_pathfinding_folder(tmp_path)
        guide.write_run_guide(run, "find_path", PARAMS, fmt="html")
        content = guide.assemble_run_content(run, "find_path", PARAMS)
        listed = [m for e in content["entries"] for m in e["matched"]]
        leftover_paths = [item["path"] for item in content["leftovers"]]
        assert not any(p.startswith(guide.GUIDE_BASENAME) for p in listed)
        assert not any(p.startswith(guide.GUIDE_BASENAME)
                       for p in leftover_paths)

    def test_unmatched_files_get_generic_descriptions(self, tmp_path):
        run = _make_pathfinding_folder(tmp_path)
        (run / "mystery_output.xyz").write_text("x", encoding="utf-8")
        content = guide.assemble_run_content(run, "find_path", PARAMS)
        leftovers = {i["path"]: i["description"] for i in content["leftovers"]}
        assert "mystery_output.xyz" in leftovers
        # Not listed under any spec entry either.
        listed = [m for e in content["entries"] for m in e["matched"]]
        assert "mystery_output.xyz" not in listed

    def test_unknown_tool_still_lists_files(self, tmp_path):
        run = _make_pathfinding_folder(tmp_path)
        content = guide.assemble_run_content(run, "no_such_tool", PARAMS)
        leftover_paths = [i["path"] for i in content["leftovers"]]
        assert "aMe12_to_PPL101_allpaths_type.csv" in leftover_paths

    def test_warnings_absent_returns_none(self, tmp_path):
        run = _make_pathfinding_folder(tmp_path, with_warnings=False)
        content = guide.assemble_run_content(run, "find_path", PARAMS)
        assert content["warnings"] is None

    def test_empty_warnings_file_treated_as_absent(self, tmp_path):
        run = _make_pathfinding_folder(tmp_path, with_warnings=False)
        (run / guide.WARNING_FILENAME).write_text("   \n", encoding="utf-8")
        content = guide.assemble_run_content(run, "find_path", PARAMS)
        assert content["warnings"] is None


class TestWriteRunGuide:
    def test_html_default(self, tmp_path):
        run = _make_pathfinding_folder(tmp_path)
        path = guide.write_run_guide(run, "find_path", PARAMS, fmt="html")
        assert path is not None
        assert path.name == guide.GUIDE_BASENAME + ".html"
        text = path.read_text(encoding="utf-8")
        assert text.lstrip().lower().startswith("<!doctype html>")
        assert "DROCAT Run Guide" in text
        # Glossary-driven column descriptions rendered.
        assert "connection_ratio" in text
        # Warning note rendered.
        assert "no downstream partners" in text
        # Top-level HTML visualizations become relative links.
        assert '<a href="visualization/Network_20240101.html"' not in text
        # (nested viz files are intentionally not linked)

    def test_formulas_render_as_self_contained_html_math(self, tmp_path):
        """Score formulas in the exported HTML guide must render as styled
        Unicode math (no MathJax / external script), not as raw $...$/LaTeX."""
        run = _make_pathfinding_folder(tmp_path)
        path = guide.write_run_guide(run, "find_path", PARAMS, fmt="html")
        text = path.read_text(encoding="utf-8")
        # No MathJax CDN and no raw TeX delimiters.
        assert "mathjax" not in text
        assert "\\lvert" not in text
        # Formulas are wrapped in a styled span with sub/superscript markup.
        assert '<span class="math">' in text
        assert "<sub>" in text
        # connection_ratio carries its formula (w_ij / sum_k w_kj).
        assert "w<sub>ij</sub>" in text
        assert "Σ<sub>k</sub>" in text

    def test_metrics_panel_referenced_by_output_files(self, tmp_path):
        """Descriptions live once in the Metrics & parameters panel; the
        Output Files section lists metric names and links back to the panel
        instead of repeating each description inline."""
        run = _make_pathfinding_folder(tmp_path)
        path = guide.write_run_guide(run, "find_path", PARAMS, fmt="html")
        text = path.read_text(encoding="utf-8")
        # A dedicated metrics panel exists with per-metric anchors.
        assert '<h2 id="metrics">Metrics &amp; parameters</h2>' in text
        assert 'id="metric-connection_ratio"' in text
        # The formula description appears once, in the panel.
        assert text.count("w<sub>ij</sub>") == 1
        # Output Files reference the panel through links, not descriptions.
        outp = text[text.index("<h2>Output files</h2>"):]
        assert 'href="#metric-connection_ratio"' in outp
        assert "w<sub>ij</sub>" not in outp  # description not repeated inline

    def test_markdown_keeps_inline_latex_math(self, tmp_path):
        """Markdown output keeps the standard $...$ inline-math markers."""
        run = _make_pathfinding_folder(tmp_path)
        path = guide.write_run_guide(run, "find_path", PARAMS, fmt="markdown")
        text = path.read_text(encoding="utf-8")
        assert "$w_{ij} / \\sum_k w_{kj}$" in text
        assert "$\\min(1.0,\\ connection\\_ratio/0.3)$" in text

    def test_txt_format(self, tmp_path):
        run = _make_pathfinding_folder(tmp_path)
        path = guide.write_run_guide(run, "find_path", PARAMS, fmt="txt")
        assert path is not None and path.suffix == ".txt"
        text = path.read_text(encoding="utf-8")
        assert "DROCAT RUN GUIDE" in text
        assert "RUN PARAMETERS" in text
        assert "WARNINGS & NOTES" in text
        assert "OUTPUT FILES" in text

    def test_markdown_format(self, tmp_path):
        run = _make_pathfinding_folder(tmp_path)
        path = guide.write_run_guide(run, "find_path", PARAMS, fmt="markdown")
        assert path is not None and path.suffix == ".md"
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# DROCAT Run Guide")
        assert "| Metric | Description | Range |" in text
        assert "## Metrics & parameters" in text

    def test_disabled_writes_nothing(self, tmp_path):
        run = _make_pathfinding_folder(tmp_path)
        before = sorted(p.name for p in run.iterdir())
        path = guide.write_run_guide(run, "find_path", PARAMS, fmt="disabled")
        assert path is None
        assert sorted(p.name for p in run.iterdir()) == before

    def test_missing_folder_returns_none(self, tmp_path):
        assert guide.write_run_guide(
            tmp_path / "nope", "find_path", fmt="html") is None

    def test_warnings_absent_fallback_text(self, tmp_path):
        run = _make_pathfinding_folder(tmp_path, with_warnings=False)
        path = guide.write_run_guide(run, "find_path", PARAMS, fmt="txt")
        text = path.read_text(encoding="utf-8")
        assert guide.NO_WARNINGS_TEXT in text

    def test_guide_never_lists_itself(self, tmp_path):
        run = _make_pathfinding_folder(tmp_path)
        for fmt, ext in guide.GUIDE_EXTENSIONS.items():
            path = guide.write_run_guide(run, "find_path", PARAMS, fmt=fmt)
            text = path.read_text(encoding="utf-8")
            assert guide.GUIDE_BASENAME + ext not in text
            assert guide.GUIDE_BASENAME not in text


class TestFormatResolution:
    def test_env_overrides_saved_default(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE",
                            tmp_path / "local_config.json")
        cfg.save_local_config({"user_defaults": {"run_guide_format": "html"}})
        monkeypatch.setenv(guide.GUIDE_FORMAT_ENV, "markdown")
        assert guide.resolve_guide_format() == "markdown"

    def test_explicit_arg_beats_env(self, monkeypatch):
        monkeypatch.setenv(guide.GUIDE_FORMAT_ENV, "txt")
        assert guide.resolve_guide_format("disabled") == "disabled"

    def test_invalid_env_falls_back_to_saved(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE",
                            tmp_path / "local_config.json")
        cfg.save_local_config({"user_defaults": {"run_guide_format": "txt"}})
        monkeypatch.setenv(guide.GUIDE_FORMAT_ENV, "bogus")
        assert guide.resolve_guide_format() == "txt"

    def test_no_env_uses_saved_default(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE",
                            tmp_path / "local_config.json")
        cfg.save_local_config(
            {"user_defaults": {"run_guide_format": "markdown"}})
        monkeypatch.delenv(guide.GUIDE_FORMAT_ENV, raising=False)
        assert guide.resolve_guide_format() == "markdown"

    def test_no_env_no_saved_defaults_to_html(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "nope.json")
        monkeypatch.delenv(guide.GUIDE_FORMAT_ENV, raising=False)
        assert guide.resolve_guide_format() == "html"

    def test_env_disabled_skips_write(self, monkeypatch, tmp_path):
        run = _make_pathfinding_folder(tmp_path)
        monkeypatch.setenv(guide.GUIDE_FORMAT_ENV, "disabled")
        assert guide.write_run_guide(run, "find_path", PARAMS) is None


class TestConfigIntegration:
    def test_default_value_registered(self):
        assert cfg.DEFAULTS["run_guide_format"] == "html"

    def test_setting_spec_registered(self):
        spec = cfg.DEFAULT_SETTING_SPECS["run_guide_format"]
        assert spec["options"] == cfg.RUN_GUIDE_FORMATS
        assert spec["group"] == "pathfinding_output"
        assert spec["kind"] == "select"
