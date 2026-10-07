"""Regression tests for the issues found by the 2026-08 E2E audit.

Covered fixes:

1. ``image_summary_skip_note``: Find Lines must loudly announce that the
   PDF/PPTX summary is skipped when image download is disabled (previously
   a silent no-op: the user asked for a PDF and nothing appeared).
2. Polars API modernization: the connection-matrix pivot (``on=`` instead of
   the deprecated ``columns=``) must emit no DeprecationWarning and produce
   matrices identical to the pandas engine path (equivalence requirement).
3. Membership filtering with ``is_in(<series>.implode())`` (the unambiguous
   polars >=1.30 form) must filter identically to the legacy call and emit
   no DeprecationWarning.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import neuronbridge_finder as nbf  # noqa: E402
from coana import FindNeuronConnection  # noqa: E402


# =============================================================================
# 1. Find Lines summary-skip note
# =============================================================================
class TestImageSummarySkipNote:
    def test_note_when_summary_requested_without_download(self):
        note = nbf.image_summary_skip_note("pdf", None)
        assert note is not None
        assert "summary" in note.lower()
        assert "image download" in note.lower()
        assert "pdf" in note.lower()

    def test_note_with_format_list(self):
        note = nbf.image_summary_skip_note(["pdf", "pptx"], None)
        assert note is not None
        assert "pptx" in note.lower()

    def test_no_note_when_download_enabled(self):
        assert nbf.image_summary_skip_note("pdf", "flylight") is None
        assert nbf.image_summary_skip_note("pdf", "neuronbridge") is None
        assert nbf.image_summary_skip_note(["pdf"], "both") is None

    def test_no_note_when_no_format_requested(self):
        # None/'' mean "no summary" — nothing to warn about.
        assert nbf.image_summary_skip_note(None, None) is None
        assert nbf.image_summary_skip_note("", None) is None
        assert nbf.image_summary_skip_note([None, ""], None) is None


# =============================================================================
# 2. Connection-matrix export: no deprecation + pandas/polars equivalence
# =============================================================================
def _matrix_df() -> pl.DataFrame:
    return pl.DataFrame({
        "type_pre": ["A", "A", "B", "B", "A"],
        "type_post": ["A", "B", "A", "B", "B"],
        "weight": [5, 3, 2, 4, 1],
        "connection_ratio": [0.5, 0.3, 0.2, 0.8, 0.1],
        "traversal_probability": [0.4, 0.2, 0.1, 0.6, 0.05],
        "nt_type": ["acetylcholine", "GABA", "acetylcholine", "glutamate", "GABA"],
    })


class TestMatrixExportModernizedPivot:
    def test_no_pivot_deprecation_warning(self, tmp_path):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            FindNeuronConnection._save_matrices_to_csv(
                None, _matrix_df(), str(tmp_path), level="type")
        dep = [str(w.message) for w in caught
               if issubclass(w.category, DeprecationWarning)]
        assert not any("pivot" in m or "`columns`" in m for m in dep), dep

    def test_matrices_match_pandas_engine(self, tmp_path):
        FindNeuronConnection._save_matrices_to_csv(
            None, _matrix_df(), str(tmp_path), level="type")
        pdf = _matrix_df().to_pandas()
        # PROBABILITY SUSPENSION (2026-10-06): the prob pivot is no longer
        # written.
        for name, col, agg in [
            ("weight", "weight", "sum"),
            ("ratio", "connection_ratio", "max"),
        ]:
            out = pd.read_csv(tmp_path / f"conn_mat_type_{name}.csv", index_col=0)
            expected = pd.pivot_table(
                pdf, index="type_pre", columns="type_post",
                values=col, aggfunc=agg)
            if agg == "sum":
                # duplicate A->B rows (3 + 1) must be summed
                assert out.loc["A", "B"] == 4
            expected = expected.fillna(0).sort_index()
            expected.columns.name = None  # CSV round-trip drops the axis name
            pd.testing.assert_frame_equal(
                out.sort_index().astype(float), expected.astype(float))

    def test_all_matrix_files_written(self, tmp_path):
        FindNeuronConnection._save_matrices_to_csv(
            None, _matrix_df(), str(tmp_path), level="type")
        for suffix in ("weight", "ratio", "nt"):
            assert (tmp_path / f"conn_mat_type_{suffix}.csv").exists(), suffix
        # PROBABILITY SUSPENSION (2026-10-06): no prob matrix file.
        assert not (tmp_path / "conn_mat_type_prob.csv").exists()


# =============================================================================
# 3. is_in with implode(): identical filtering, no deprecation
# =============================================================================
class TestIsInImplodeEquivalence:
    def test_membership_filter_identical_and_warning_free(self):
        conn = pl.DataFrame({
            "bodyId_pre": [1, 2, 3, 4, 5],
            "bodyId_post": [10, 20, 30, 40, 50],
        })
        q = pl.Series("q", [2, 4, 30], dtype=pl.Int64)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            new = conn.filter(pl.col("bodyId_pre").is_in(q.implode()))
            new_not = conn.filter(~pl.col("bodyId_post").is_in(q.implode()))
        dep = [str(w.message) for w in caught
               if issubclass(w.category, DeprecationWarning)]
        assert not any("is_in" in m for m in dep), dep
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            old = conn.filter(pl.col("bodyId_pre").is_in(q))
            old_not = conn.filter(~pl.col("bodyId_post").is_in(q))
        assert new.equals(old)
        assert new_not.equals(old_not)
        assert new["bodyId_pre"].to_list() == [2, 4]
        assert new_not["bodyId_post"].to_list() == [10, 20, 40, 50]

    def test_string_membership_with_imploded_series(self):
        df = pl.DataFrame({"roi": ["AL", "MB", "LAL", ""]})
        roi_filter = ["AL", "LAL"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            new = df.filter(
                pl.col("roi").is_in(pl.Series("roi_filter", roi_filter).implode()))
        dep = [str(w.message) for w in caught
               if issubclass(w.category, DeprecationWarning)]
        assert not any("is_in" in m for m in dep), dep
        assert new["roi"].to_list() == ["AL", "LAL"]
