"""Regression tests for analysis-tab skeleton visualization settings."""

import inspect
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.profile_comparator import HomologFinder  # noqa: E402
from neuronbridge_finder import NeuronBridgeFinder  # noqa: E402
from visualize_skeleton import VisualizeSkeleton  # noqa: E402
from visualization_options import (  # noqa: E402
    default_analysis_skeleton_mesh_simplification,
    default_skeleton_tab_simplification,
)


def test_homolog_visualization_defaults_to_native_brain():
    finder = HomologFinder(verbose=False)
    assert finder.visualization_settings["brain_mesh"] == "native"


def test_homolog_visualization_settings_override_renderer_defaults():
    finder = HomologFinder(
        verbose=False,
        visualization_settings={
            "brain_mesh": "none",
            "skeleton_mode": "line",
            "show_fig": True,
        },
    )
    options = finder._homolog_visualizer_kwargs(
        {"brain_mesh": "FAFB", "show_fig": False},
        dataset="test:v1",
        neuron_layers=["A"],
    )
    assert options["brain_mesh"] == "none"
    assert options["skeleton_mode"] == "line"
    assert options["show_fig"] is True
    assert options["skeleton_mesh_simplification"] == 0.95


def test_analysis_skeleton_simplification_defaults_follow_pipeline():
    assert default_analysis_skeleton_mesh_simplification("male-cns:v1.0") == 0.95
    assert default_analysis_skeleton_mesh_simplification("hemibrain:v1.2.1") == 0.95
    assert default_analysis_skeleton_mesh_simplification("flywire_FAFB_v783") == 0.95
    assert default_analysis_skeleton_mesh_simplification("FAFB:v783") == 0.95
    assert default_analysis_skeleton_mesh_simplification(
        "male-cns:v1.0", "fast"
    ) == 0.90
    assert default_analysis_skeleton_mesh_simplification(
        "flywire_FAFB_v783", "fast"
    ) == 0.90


def test_dedicated_skeleton_tab_defaults_follow_pipeline_for_fafb():
    assert default_skeleton_tab_simplification(
        "flywire_FAFB_v783", "fast"
    ) == 0.90
    assert default_skeleton_tab_simplification(
        "FAFB:v783", "fine"
    ) == 0.95
    assert default_skeleton_tab_simplification(
        "flywire_FAFB_v783", "artistic"
    ) == 0.95
    assert default_skeleton_tab_simplification(
        "male-cns:v1.0", "fast"
    ) == 0.90
    assert default_skeleton_tab_simplification(
        "male-cns:v1.0", "fine"
    ) == 0.95


def test_primary_roi_keyword_reads_dataset_metadata(tmp_path):
    """The programmatic primary keyword uses the local NeuPrint sidecar."""
    import json

    dataset = "test:v1.0"
    folder = dataset.replace(":", "_").replace(".", "_")
    dataset_dir = tmp_path / "datasets" / folder
    dataset_dir.mkdir(parents=True)
    (dataset_dir / f"{folder}_metadata.json").write_text(
        json.dumps({
            "roi_coverage": {
                "roi_list": ["A(L)", "A(R)", "M", "NotPrimary"],
            },
        }),
        encoding="utf-8",
    )

    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.dataset = dataset
    visualizer.script_path = str(tmp_path)
    visualizer._vprint = lambda *args, **kwargs: None
    visualizer._get_available_rois = lambda **kwargs: [
        "A(L)", "A(R)", "M", "A-sub(L)",
    ]

    assert visualizer._get_metadata_primary_rois() == ["A(L)", "A(R)", "M"]
    assert visualizer._expand_mesh_roi_patterns(["primary"]) == [
        "A(L)", "A(R)", "M",
    ]


def test_visualization_warning_notes_record_effective_simplification(tmp_path):
    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.dataset = "flywire_FAFB_v783"
    visualizer.skeleton_mesh_simplification = 0.98
    visualizer.skeleton_mode = "tube"
    visualizer.save_folder = str(tmp_path)
    visualizer._vprint = lambda *args, **kwargs: None

    visualizer._write_user_warning_notes()

    notes = (tmp_path / "user_warning_notes.txt").read_text(encoding="utf-8")
    assert "skeleton_mesh_simplification=0.98" in notes
    assert "rendering only" in notes


def test_html_simplification_warning_uses_dataset_specific_thresholds():
    # The default pipeline is now fast: NeuPrint warns above 0.90 unless
    # a fine/artistic pipeline is explicitly selected.
    assert VisualizeSkeleton._skeleton_simplification_warning(
        "male-cns:v1.0", "neuprint", "tube", 0.90
    ) is None
    assert VisualizeSkeleton._skeleton_simplification_warning(
        "male-cns:v1.0", "neuprint", "tube", 0.901
    )["threshold"] == 0.90
    assert VisualizeSkeleton._skeleton_simplification_warning(
        "male-cns:v1.0", "neuprint", "tube", 0.90, "fast"
    ) is None
    assert VisualizeSkeleton._skeleton_simplification_warning(
        "male-cns:v1.0", "neuprint", "tube", 0.901, "fast"
    )["threshold"] == 0.90
    assert VisualizeSkeleton._skeleton_simplification_warning(
        "male-cns:v1.0", "neuprint", "tube", 0.95, "fine"
    ) is None
    assert VisualizeSkeleton._skeleton_simplification_warning(
        "male-cns:v1.0", "neuprint", "tube", 0.951, "fine"
    )["threshold"] == 0.95
    assert VisualizeSkeleton._skeleton_simplification_warning(
        "male-cns:v1.0", "neuprint", "tube", 0.95, "artistic"
    ) is None
    assert VisualizeSkeleton._skeleton_simplification_warning(
        "male-cns:v1.0", "neuprint", "tube", 0.951, "artistic"
    )["threshold"] == 0.95

    assert VisualizeSkeleton._skeleton_simplification_warning(
        "flywire_FAFB_v783", "flywire", "tube", 0.95
    ) is None
    assert VisualizeSkeleton._skeleton_simplification_warning(
        "flywire_FAFB_v783", "flywire", "tube", 0.951
    )["threshold"] == 0.95

    # The warning is for the tube-surface pipeline, not line rendering or the
    # separate BANC mesh pipeline.
    assert VisualizeSkeleton._skeleton_simplification_warning(
        "male-cns:v1.0", "neuprint", "line", 0.99
    ) is None
    assert VisualizeSkeleton._skeleton_simplification_warning(
        "banc_v888", "flywire", "tube", 0.99
    ) is None


def test_html_writer_embeds_plotly_runtime_and_injects_warning(tmp_path):
    class FakeFigure:
        def __init__(self):
            self.kwargs = None

        def write_html(self, path, **kwargs):
            self.kwargs = kwargs
            Path(path).write_text(
                "<html><head></head><body><div id='figure'></div></body></html>",
                encoding="utf-8",
            )

    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.dataset = "male-cns:v1.0"
    visualizer.client_type = "neuprint"
    visualizer.skeleton_mode = "tube"
    visualizer.neuprint_skeleton_pipeline = "fast"
    visualizer.skeleton_mesh_simplification = 0.95
    visualizer._vprint = lambda *args, **kwargs: None

    figure = FakeFigure()
    path = tmp_path / "skeleton.html"
    visualizer._write_plotly_html(figure, str(path))

    assert figure.kwargs["include_plotlyjs"] is True
    html = path.read_text(encoding="utf-8")
    assert "drocat-in-page-warning" in html
    assert "skeleton_mesh_simplification=0.950" in html
    assert "shared raw level-0" in html
    assert "fixed simp90 skeleton cache" not in html
    assert html.index("drocat-in-page-warning") > html.index("<body>")


def test_line_mode_html_raises_export_pipeline_warning(tmp_path):
    """Every line-mode page carries the high-quality export-pipeline warning."""

    class FakeFigure:
        def write_html(self, path, **kwargs):
            Path(path).write_text(
                "<html><head></head><body><div id='figure'></div></body></html>",
                encoding="utf-8",
            )

    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.dataset = "male-cns:v1.0"
    visualizer.client_type = "neuprint"
    visualizer.skeleton_mode = "line"
    visualizer.neuprint_skeleton_pipeline = "fast"
    visualizer.skeleton_mesh_simplification = 0.99
    visualizer._vprint = lambda *args, **kwargs: None

    path = tmp_path / "line_mode.html"
    visualizer._write_plotly_html(FakeFigure(), str(path))

    html = path.read_text(encoding="utf-8")
    assert "drocat-in-page-warning" in html
    assert "Line-mode skeleton rendering" in html
    assert "high-quality export pipeline" in html
    assert html.index("drocat-in-page-warning") > html.index("<body>")
    # The tube-surface simplification warning does not apply to lines.
    assert "Skeleton simplification warning" not in html


def test_tube_mode_html_omits_line_mode_export_warning(tmp_path):
    class FakeFigure:
        def write_html(self, path, **kwargs):
            Path(path).write_text(
                "<html><head></head><body><div id='figure'></div></body></html>",
                encoding="utf-8",
            )

    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.dataset = "male-cns:v1.0"
    visualizer.client_type = "neuprint"
    visualizer.skeleton_mode = "tube"
    visualizer.neuprint_skeleton_pipeline = "fast"
    visualizer.skeleton_mesh_simplification = 0.90
    visualizer._vprint = lambda *args, **kwargs: None

    path = tmp_path / "tube_mode.html"
    visualizer._write_plotly_html(FakeFigure(), str(path))

    html = path.read_text(encoding="utf-8")
    assert "Line-mode skeleton rendering" not in html
    assert "drocat-in-page-warning" not in html


def test_user_warning_notes_record_line_mode_export_warning(tmp_path):
    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.dataset = "flywire_FAFB_v783"
    visualizer.skeleton_mode = "line"
    visualizer.skeleton_mesh_simplification = 0.98
    visualizer.save_folder = str(tmp_path)
    visualizer._vprint = lambda *args, **kwargs: None

    visualizer._write_user_warning_notes()

    notes = (tmp_path / "user_warning_notes.txt").read_text(encoding="utf-8")
    assert "in-page export-pipeline warning raised for line mode" in notes


def test_fine_html_warning_starts_above_simp95(tmp_path):
    class FakeFigure:
        def write_html(self, path, **kwargs):
            Path(path).write_text(
                "<html><head></head><body><div id='figure'></div></body></html>",
                encoding="utf-8",
            )

    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.dataset = "male-cns:v1.0"
    visualizer.client_type = "neuprint"
    visualizer.skeleton_mode = "tube"
    visualizer.neuprint_skeleton_pipeline = "fine"
    visualizer.skeleton_mesh_simplification = 0.95
    visualizer._vprint = lambda *args, **kwargs: None

    exact_path = tmp_path / "fine_simp95.html"
    visualizer._write_plotly_html(FakeFigure(), str(exact_path))
    assert "Skeleton simplification warning" not in exact_path.read_text(
        encoding="utf-8"
    )

    visualizer.skeleton_mesh_simplification = 0.951
    high_path = tmp_path / "fine_simp951.html"
    visualizer._write_plotly_html(FakeFigure(), str(high_path))
    html = high_path.read_text(encoding="utf-8")
    assert "Skeleton simplification warning" in html
    assert "rebuild the FAFB-style tube mesh" in html
    assert "raw .swc.gz source" in html


def test_user_warning_notes_record_fine_threshold(tmp_path):
    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.dataset = "male-cns:v1.0"
    visualizer.skeleton_mode = "tube"
    visualizer.neuprint_skeleton_pipeline = "fine"
    visualizer.skeleton_mesh_simplification = 0.95
    visualizer.save_folder = str(tmp_path)
    visualizer._vprint = lambda *args, **kwargs: None

    visualizer._write_user_warning_notes()

    notes = (tmp_path / "user_warning_notes.txt").read_text(encoding="utf-8")
    assert "neuprint_skeleton_pipeline=fine" in notes
    assert "in-page warning threshold >0.95" in notes


def test_large_html_render_warning_is_written_to_user_warning_notes(tmp_path):
    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.save_folder = str(tmp_path)
    visualizer._vprint = lambda *args, **kwargs: None

    html_path = tmp_path / "large_skeleton.html"
    html_path.write_text("<html></html>", encoding="utf-8")
    with html_path.open("ab") as handle:
        handle.truncate(50 * 1024 * 1024 + 1)

    assert visualizer._record_large_html_warning(str(html_path)) is True

    notes = (tmp_path / "user_warning_notes.txt").read_text(encoding="utf-8")
    assert "[render warning] visualization HTML is too large" in notes
    assert "Browser rendering and exporting may fail" in notes
    assert "0.98 or 0.99" in notes
    assert "skeleton mode to 'line'" in notes


def test_large_html_render_warning_skips_temporary_export_copy(tmp_path):
    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.save_folder = str(tmp_path)
    visualizer._vprint = lambda *args, **kwargs: None

    html_path = tmp_path / "_temp_export.html"
    html_path.write_text("<html></html>", encoding="utf-8")
    with html_path.open("ab") as handle:
        handle.truncate(50 * 1024 * 1024 + 1)

    assert visualizer._record_large_html_warning(str(html_path)) is False
    assert not (tmp_path / "user_warning_notes.txt").exists()


def test_large_html_render_warning_uses_strictly_greater_than_50_mb(tmp_path):
    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.save_folder = str(tmp_path)
    visualizer._vprint = lambda *args, **kwargs: None

    html_path = tmp_path / "50mb_skeleton.html"
    html_path.write_text("<html></html>", encoding="utf-8")
    with html_path.open("ab") as handle:
        handle.truncate(50 * 1024 * 1024)

    assert visualizer._record_large_html_warning(str(html_path)) is False
    assert not (tmp_path / "user_warning_notes.txt").exists()


def test_real_skeleton_html_is_portable_without_plotly_sidecar(tmp_path):
    import plotly.graph_objects as go

    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.dataset = "flywire_FAFB_v783"
    visualizer.client_type = "flywire"
    visualizer.skeleton_mode = "tube"
    visualizer.skeleton_mesh_simplification = 0.98
    visualizer._vprint = lambda *args, **kwargs: None

    path = tmp_path / "portable.html"
    visualizer._write_plotly_html(
        go.Figure(go.Scatter3d(x=[0], y=[0], z=[0])), str(path)
    )

    html = path.read_text(encoding="utf-8")
    assert 'src="plotly.min.js"' not in html
    assert "Plotly.newPlot" in html
    assert "drocat-in-page-warning" in html


def test_all_skeleton_result_paths_accept_visualization_settings():
    assert "visualization_settings" in inspect.signature(
        NeuronBridgeFinder.find_neurons_batch
    ).parameters
    assert "visualization_settings" in inspect.signature(
        NeuronBridgeFinder.analyze_colabeling
    ).parameters
