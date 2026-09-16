"""Tests for the NeuronBridge output-detail prune policy.

Contract (plan `_plan/plan-nb-find-lines-output-modes.md` §3): Full keeps
everything; Compact removes bodyId-level source data only AFTER the run's
summaries/report exist, images only after the PDF/PPTX artifact exists,
is idempotent, and audits every removal in cleanup_audit.json.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import src.neuronbridge_output_policy as policy  # noqa: E402


def _make_lines_run(root: Path, *, with_summary=True, with_images=True,
                    with_pdf=False):
    root.mkdir(parents=True, exist_ok=True)
    (root / "DNp01_lines.csv").write_text("line,score\nVT000000,1\n")
    (root / "parameters.json").write_text("{}")
    if with_summary:
        (root / "line_summary.csv").write_text(
            "line,weighted_score\nVT000000,1\n")
    if with_images:
        img = root / "images" / "VT000000"
        img.mkdir(parents=True, exist_ok=True)
        (img / "img.png").write_bytes(b"x" * 32)
    if with_pdf:
        (root / "images_summary.pdf").write_bytes(b"%PDF-1.4\n")
    return root


def test_compact_run_without_payloads_still_records_audit(tmp_path):
    """An imageless, summary-less Compact run removes nothing on the day —
    but the audit must still be written, or later retention sweeps classify
    the run as unknown mode and never reclaim its match tables
    (review 2026-09-16)."""
    run = _make_lines_run(tmp_path / "run_x", with_summary=False,
                          with_images=False)
    policy.prune_find_lines_run(str(run), keep_per_match_csv=False,
                                cleanup_source_images=True)
    assert (run / policy.AUDIT_FILENAME).exists()
    assert policy._run_was_compact(run) is True


class TestPruneFindLinesRun:
    def test_full_keeps_everything(self, tmp_path):
        root = _make_lines_run(tmp_path / "run")
        audit = policy.prune_find_lines_run(str(root))
        assert (root / "DNp01_lines.csv").exists()
        assert (root / "images" / "VT000000" / "img.png").exists()
        assert audit["removed"] == []

    def test_compact_drops_match_tables_and_images_after_summaries(
        self, tmp_path
    ):
        root = _make_lines_run(tmp_path / "run", with_pdf=True)
        audit = policy.prune_find_lines_run(
            str(root), keep_per_match_csv=False, cleanup_source_images=True)
        assert not (root / "DNp01_lines.csv").exists()
        assert not (root / "images").exists()
        assert (root / "line_summary.csv").exists()
        assert (root / "parameters.json").exists()
        assert (root / "images_summary.pdf").exists()  # the deliverable stays
        assert audit["bytes_reclaimed"] > 0
        persisted = json.loads((root / policy.AUDIT_FILENAME).read_text())
        removed_names = [Path(e["path"]).name for e in persisted["removed"]]
        assert removed_names == ["DNp01_lines.csv", "images"]

    def test_compact_without_summary_keeps_match_tables(self, tmp_path):
        root = _make_lines_run(tmp_path / "run", with_summary=False,
                               with_images=False)
        policy.prune_find_lines_run(str(root), keep_per_match_csv=False)
        assert (root / "DNp01_lines.csv").exists()

    def test_compact_without_pdf_keeps_images(self, tmp_path):
        root = _make_lines_run(tmp_path / "run", with_summary=True)
        # images/ exists but no PDF/PPTX artifact -> images must survive
        policy.prune_find_lines_run(
            str(root), keep_per_match_csv=True, cleanup_source_images=True)
        assert (root / "images" / "VT000000" / "img.png").exists()

    def test_idempotent_second_pass_merges_audit(self, tmp_path):
        root = _make_lines_run(tmp_path / "run", with_pdf=True)
        policy.prune_find_lines_run(
            str(root), keep_per_match_csv=False, cleanup_source_images=True)
        audit = policy.prune_find_lines_run(
            str(root), keep_per_match_csv=False, cleanup_source_images=True)
        # Per-pass removals: the second pass finds nothing left to remove.
        assert audit["removed"] == []
        assert audit["passes"] == 2
        persisted = json.loads((root / policy.AUDIT_FILENAME).read_text())
        assert len(persisted["removed"]) == 2  # merged, not duplicated

    def test_multi_chip_layout_prunes_each_chip_independently(self, tmp_path):
        root = tmp_path / "run"
        for chip, with_summary in (("chip_DNp01", True), ("chip_Tm4", False)):
            d = root / chip
            d.mkdir(parents=True)
            (d / f"{chip.split('_')[1]}_lines.csv").write_text("line\n")
            if with_summary:
                (d / "line_summary.csv").write_text("line\n")
            img = d / "images" / "L"
            img.mkdir(parents=True)
            (img / "i.png").write_bytes(b"z" * 8)
            (d / "images_summary.pdf").write_bytes(b"%PDF-1.4\n")
        # Root-level summary must not license pruning inside chips.
        (root / "line_summary.csv").write_text("line\n")

        audit = policy.prune_find_lines_run(
            str(root), keep_per_match_csv=False, cleanup_source_images=True)

        # chip_DNp01 (summary present): pruned. chip_Tm4 (no summary): its
        # match table stays, but its images still go — image cleanup is
        # gated by the PDF artifact, which exists in both chips.
        assert not (root / "chip_DNp01" / "DNp01_lines.csv").exists()
        assert not (root / "chip_DNp01" / "images").exists()
        assert (root / "chip_Tm4" / "Tm4_lines.csv").exists()
        assert not (root / "chip_Tm4" / "images").exists()
        removed_paths = [e["path"] for e in audit["removed"]]
        assert [p for p in removed_paths if p.endswith("_lines.csv")] == \
            [str(root / "chip_DNp01" / "DNp01_lines.csv")]


def _make_find_neurons_run(root: Path, *, with_types=True):
    root.mkdir(parents=True, exist_ok=True)
    by_ds = root / "by_dataset"
    by_ds.mkdir(exist_ok=True)
    (root / "all_neurons.csv").write_text("bodyId\n1\n")
    (root / "VT019730_neurons.csv").write_text("bodyId\n1\n")
    (by_ds / "VT019730_hemibrain_v1_2_1_neurons.csv").write_text("bodyId\n1\n")
    if with_types:
        (by_ds / "VT019730_hemibrain_v1_2_1_types.csv").write_text("type\nTm4\n")
    (root / "labeling_distribution.html").write_text("<html></html>")
    return root


class TestPruneFindNeuronsRun:
    def test_compact_drops_bodyid_tables_keeps_type_summaries(self, tmp_path):
        root = _make_find_neurons_run(tmp_path / "run")
        audit = policy.prune_find_neurons_run(
            str(root), keep_per_match_csv=False)
        assert not (root / "all_neurons.csv").exists()
        assert not (root / "VT019730_neurons.csv").exists()
        assert not (root / "by_dataset"
                    / "VT019730_hemibrain_v1_2_1_neurons.csv").exists()
        assert (root / "by_dataset"
                / "VT019730_hemibrain_v1_2_1_types.csv").exists()
        assert (root / "labeling_distribution.html").exists()
        assert audit["bytes_reclaimed"] > 0

    def test_without_type_summaries_keeps_everything(self, tmp_path):
        root = _make_find_neurons_run(tmp_path / "run", with_types=False)
        policy.prune_find_neurons_run(str(root), keep_per_match_csv=False)
        assert (root / "all_neurons.csv").exists()


def _make_colabel_run(root: Path, *, with_deliverables=True):
    root.mkdir(parents=True, exist_ok=True)
    labeled = root / "line_labeled_neurons"
    labeled.mkdir(exist_ok=True)
    (labeled / "VT019730_neurons.csv").write_text("bodyId\n1\n")
    (labeled / "VT019730_types.csv").write_text("type\nTm4\n")
    (root / "distribution_data_by_neuron.csv").write_text("bodyId\n1\n")
    if with_deliverables:
        (root / "colabeling_matrix_jaccard.csv").write_text("a,b\n1,2\n")
        (root / "colabeling_report.html").write_text("<html></html>")
    return root


class TestPruneColabelRun:
    def test_compact_drops_row_level_folder_keeps_deliverables(self, tmp_path):
        root = _make_colabel_run(tmp_path / "run")
        policy.prune_colabel_run(str(root), keep_per_match_csv=False)
        assert not (root / "line_labeled_neurons").exists()
        assert not (root / "distribution_data_by_neuron.csv").exists()
        assert (root / "colabeling_matrix_jaccard.csv").exists()
        assert (root / "colabeling_report.html").exists()

    def test_without_deliverables_keeps_everything(self, tmp_path):
        root = _make_colabel_run(tmp_path / "run", with_deliverables=False)
        policy.prune_colabel_run(str(root), keep_per_match_csv=False)
        assert (root / "line_labeled_neurons"
                / "VT019730_neurons.csv").exists()


class TestColabelFlatLayout:
    def test_group_in_by_dataset_false_keeps_target_layout(self, tmp_path):
        """Co-Labeling passes group_in_by_dataset=False: per-dataset files
        land flat in the target dir (line_labeled_neurons/), preserving its
        historical layout."""
        import pandas as pd
        from neuronbridge_finder import NeuronBridgeFinder

        finder = NeuronBridgeFinder.__new__(NeuronBridgeFinder)
        finder.verbose = False
        df = pd.DataFrame({
            "bodyId": ["1", "2"],
            "type": ["Tm4", "Tm4"],
            "score": [40.0, 30.0],
            "dataset": ["hemibrain:v1.2.1", "hemibrain:v1.2.1"],
        })
        finder._save_dataset_categorized_files(
            df, "L1", str(tmp_path), verbose=False,
            group_in_by_dataset=False)
        assert (tmp_path / "L1_hemibrain_v1_2_1_neurons.csv").exists()
        assert (tmp_path / "L1_hemibrain_v1_2_1_types.csv").exists()
        assert not (tmp_path / "by_dataset").exists()


class TestMatchTableRetention:
    """Rolling window (plan §5 D2 refinement): the newest N runs keep their
    bodyId-level match tables; older runs are swept only when they opted
    into Compact.  Full and unknown-mode runs are immune."""

    def _run(self, root, name, *, compact=True):
        d = root / name
        d.mkdir(parents=True)
        (d / "X_lines.csv").write_text("line\n")
        (d / "line_summary.csv").write_text("line\n")
        (d / "parameters.json").write_text(json.dumps({
            "function_params": {"keep_per_match_csv": not compact}}))
        return d

    def test_window_keeps_newest_and_sweeps_older_compact(self, tmp_path):
        old = self._run(tmp_path, "NB-find-lines-expanded_ALL_OLD_20260914_000000")
        new = self._run(tmp_path, "NB-find-lines-expanded_ALL_NEW_20260915_000000")
        result = policy.enforce_match_table_retention(
            str(tmp_path), str(new), 1)
        assert not (old / "X_lines.csv").exists()
        assert (new / "X_lines.csv").exists()
        assert result["pruned_runs"] == [old.name]

    def test_full_runs_are_immune(self, tmp_path):
        old_full = self._run(
            tmp_path, "NB-find-lines_MCNS_FULL_20260914_000000", compact=False)
        new = self._run(tmp_path, "NB-find-lines-expanded_ALL_NEW_20260915_000000")
        policy.enforce_match_table_retention(str(tmp_path), str(new), 1)
        assert (old_full / "X_lines.csv").exists()

    def test_unknown_mode_is_immune(self, tmp_path):
        old = tmp_path / "NB-find-lines_MCNS_LEGACY_20260914_000000"
        old.mkdir(parents=True)
        (old / "X_lines.csv").write_text("line\n")
        (old / "line_summary.csv").write_text("line\n")
        new = self._run(tmp_path, "NB-find-lines-expanded_ALL_NEW_20260915_000000")
        policy.enforce_match_table_retention(str(tmp_path), str(new), 1)
        assert (old / "X_lines.csv").exists()

    def test_zero_window_is_a_noop(self, tmp_path):
        old = self._run(tmp_path, "NB-find-lines-expanded_ALL_OLD_20260914_000000")
        new = self._run(tmp_path, "NB-find-lines-expanded_ALL_NEW_20260915_000000")
        result = policy.enforce_match_table_retention(
            str(tmp_path), str(new), 0)
        assert (old / "X_lines.csv").exists()
        assert result["kept_window"] == 0


class TestNonEmptyGating:
    def test_zero_byte_summary_never_licenses_deletion(self, tmp_path):
        root = tmp_path / "run"
        root.mkdir()
        (root / "DNp01_lines.csv").write_text("line\n")
        (root / "line_summary.csv").write_text("")  # crashed writer
        policy.prune_find_lines_run(str(root), keep_per_match_csv=False)
        assert (root / "DNp01_lines.csv").exists()

    def test_zero_byte_pdf_never_licenses_image_removal(self, tmp_path):
        root = tmp_path / "run"
        root.mkdir()
        img = root / "images" / "L"
        img.mkdir(parents=True)
        (img / "i.png").write_bytes(b"z" * 8)
        (root / "images_summary.pdf").write_text("")  # crashed writer
        policy.prune_find_lines_run(
            str(root), keep_per_match_csv=True, cleanup_source_images=True)
        assert (root / "images" / "L" / "i.png").exists()
