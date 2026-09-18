"""Regression coverage for query-keyed comparison exports."""

import pandas as pd
import pytest


def test_query_visualization_points_are_all_exported(tmp_path):
    from src.comparison.visualizations import ComparisonVisualizer

    datasets = ["male-cns:v1.0", "flywire_FAFB_v783", "banc_v888"]
    queries = ["combo_001", "combo_002"]
    raw = {
        dataset: {
            query: pd.DataFrame({
                "type_pre": ["source"],
                "type_post": ["target"],
                "weight": [3],
            })
            for query in queries
        }
        for dataset in datasets
    }
    aligned = pd.DataFrame(
        {
            dataset: [3, 2, 2, 0] for dataset in datasets
        },
        index=["source -> target", "source -> inter", "inter -> target",
               "source -> other"],
    )
    path_data = pd.DataFrame(
        {dataset: [3] for dataset in datasets},
        index=["source -> inter -> target"],
    )

    def metadata(query):
        return {
            "query_id": query,
            "query_label": f"Label {query}",
            "threshold_scope": "query",
            "requested_thresholds": {
                dataset: 3 for dataset in datasets
            },
            "applied_thresholds": {
                dataset: 3 for dataset in datasets
            },
        }

    visualizer = ComparisonVisualizer(verbose=False)
    visualizer.save_all_plots(
        results=raw,
        aligned_data=aligned,
        similarities=pd.DataFrame(),
        output_dir=str(tmp_path),
        thresholds=queries,
        align_func=lambda _query: aligned,
        similarity_func=lambda _query: pd.DataFrame(),
        current_threshold=queries[0],
        path_data_func=lambda _query: path_data,
        ratio_data_func=None,
        prob_data_func=None,
        nickname_map={dataset: dataset for dataset in datasets},
        point_metadata_func=metadata,
        point_label_func=lambda query: f"Label {query}",
        point_stem_func=lambda query: f"query_{query}",
        threshold_mode="combinations",
        silent=True,
    )

    vis_data = tmp_path / "visualization_data"
    assert (tmp_path / "edge_heatmap_query_combo_001.png").exists()
    assert (tmp_path / "edge_heatmap_query_combo_002.png").exists()
    assert (tmp_path / "path_heatmap_query_combo_001.png").exists()
    assert (tmp_path / "path_heatmap_query_combo_002.png").exists()
    path_counts = pd.read_csv(vis_data / "path_counts.csv")
    assert set(path_counts["query_id"]) == set(queries)
    assert set(path_counts["query_label"]) == {"Label combo_001", "Label combo_002"}
    overlap = pd.read_csv(vis_data / "overlap_matrices_per_threshold.csv")
    assert set(overlap["query_id"]) == set(queries)
    assert set(overlap["threshold_scope"]) == {"query"}
    assert overlap["threshold"].isna().all()
    assert not (tmp_path / "by_ratio").exists()
    assert not (tmp_path / "by_probability").exists()
    # Query order is not a threshold schedule: trend/combined figures that
    # imply a monotone threshold axis must not be emitted in combinations
    # mode (plan Phase D / non-goals).
    assert not (tmp_path / "jaccard_similarity_trend.png").exists()
    assert not (tmp_path / "top20_overlap_trend.png").exists()
    assert not (tmp_path / "path_rank_correlation_trend.png").exists()
    assert not (tmp_path / "cosine_similarity_trend.png").exists()
    assert not (tmp_path / "conservation_across_thresholds.png").exists()
    assert not (tmp_path / "path_heatmap_all_thresholds.png").exists()
    assert not (tmp_path / "edge_heatmap_all_thresholds.png").exists()
    # Aggregate rows must never carry a query id inside the scalar
    # threshold column (Phase G item 5).
    for csv_name in ("path_counts.csv", "threshold_comparison.csv",
                     "key_findings_per_threshold.csv"):
        frame = pd.read_csv(vis_data / csv_name)
        if "threshold" in frame.columns:
            assert frame["threshold"].isna().all(), csv_name
            assert not frame["threshold"].astype(str).str.contains(
                "combo_").any(), csv_name


def _build_query_report(tmp_path, extra_parameters=None):
    """Shared two-query HTML fixture for the report-shell tests."""
    from src.comparison.html_report_generator import generate_html_report
    from src.comparison.point_context import ComparisonPoint

    datasets = ["d1", "d2", "d3"]
    queries = [
        {"id": "combo_001", "label": "Low", "thresholds": {"d1": 3, "d2": 5, "d3": 7}},
        {"id": "combo_002", "label": "High", "thresholds": {"d1": 5, "d2": 7, "d3": 3}},
    ]

    class Parameters:
        full_output_path = str(tmp_path)
        comparison_mode = "path"
        path_mode = "all"
        max_interlayer = 2
        separate_hemispheres = False
        symmetry_analysis = False
        auto_type_mapping = False
        source_neurons = []
        target_neurons = []

        @staticmethod
        def get_dataset_nicknames():
            return datasets

    for name, value in (extra_parameters or {}).items():
        setattr(Parameters, name, value)

    class Analyzer:
        parameters = Parameters()
        label_mapper = None
        comparison_report = {"threshold_similarities": pd.DataFrame([
            {
                "query_id": "combo_001",
                "dataset_1": "d1",
                "dataset_2": "d2",
                "jaccard_similarity": 0.5,
                "ruzicka_similarity": 0.4,
                "pearson_correlation": 0.3,
                "edge_rank_correlation": 0.6,
                "cosine_similarity": 0.7,
                "spearman_rank_correlation": 0.2,
                "common_edges": 1,
            },
        ])}
        _similarity_cache = {}
        raw_results = {
            dataset: {
                threshold: pd.DataFrame({
                    "type_pre": ["source"],
                    "type_post": ["target"],
                    "weight": [threshold],
                })
                for threshold in (3, 5, 7)
            }
            for dataset in datasets
        }
        _path_run_meta = {
            (dataset, threshold): {
                "applied_threshold": threshold,
                "applied_threshold_source": "requested",
                "paths_complete": True,
                "strongest_first_budget": 100,
                "strongest_first_tau": None,
                "edge_budget": 1000,
                "edge_budget_applied": False,
            }
            for dataset in datasets for threshold in (3, 5, 7)
        }
        _neuron_counts_summary = pd.DataFrame()
        _neuron_type_counts = pd.DataFrame()
        _neuron_group_counts = pd.DataFrame()

        @staticmethod
        def get_threshold_queries():
            return queries

        @staticmethod
        def resolve_query_inputs():
            return [
                {"token": "source", "dataset": "d1", "role": "source",
                 "status": "same_name_identity", "method": "native_type",
                 "target_types": ["source"], "evidence": "confirmed",
                 "confidence": 4, "note": ""},
                {"token": "target", "dataset": "d3", "role": "target",
                 "status": "mapped", "method": "crosswalk",
                 "target_types": ["target"], "evidence": "",
                 "confidence": 3, "note": ""},
            ]

        @staticmethod
        def get_aligned_data_for_query(_query):
            return pd.DataFrame(
                {dataset: [3, 2, 2, 0] for dataset in datasets},
                index=["source -> target", "source -> inter",
                       "inter -> target", "source -> other"],
            )

        @staticmethod
        def get_aligned_data_for_network(_query):
            return pd.DataFrame(
                {dataset: [3, 2, 2, 0] for dataset in datasets},
                index=["source -> target", "source -> inter",
                       "inter -> target", "source -> other"],
            )

        @staticmethod
        def _get_path_data_for_query(_query):
            return pd.DataFrame(
                {dataset: [3] for dataset in datasets},
                index=["source -> inter -> target"],
            )

        @staticmethod
        def _get_path_hop_weights_for_threshold(threshold):
            if isinstance(threshold, dict):
                return {
                    "source -> inter -> target": {
                        dataset: [10, 7] for dataset in datasets
                    }
                }
            return {}

        @staticmethod
        def _path_provenance_row(dataset, threshold):
            return {
                "dataset": dataset,
                "threshold": threshold,
                "requested_threshold": threshold,
                "applied_threshold": threshold,
                "applied_threshold_source": "requested",
                "strongest_first_budget": 100,
                "strongest_first_budget_bitten": False,
                "tau": None,
                "edge_budget": 1000,
                "edge_budget_applied": False,
                "edge_weight_floor": None,
                "edge_budget_landing": None,
                "strongest_dropped_bottleneck": None,
                "strongest_retained_bottleneck": None,
                "paths_complete": True,
            }

    points = [
        ComparisonPoint(query["id"], query["label"], "combinations",
                        query["thresholds"], index)
        for index, query in enumerate(queries, start=1)
    ]
    # Conserved-graph exports referenced by the report links.
    (tmp_path / "conserved_paths").mkdir(exist_ok=True)
    (tmp_path / "conserved_reciprocal_graph").mkdir(exist_ok=True)
    for query_id in ("combo_001", "combo_002"):
        (tmp_path / "conserved_paths" / f"conserved_network_t{query_id}_network.html").write_text("<html></html>")
        (tmp_path / "conserved_reciprocal_graph" / f"conserved_reciprocal_t{query_id}_network.html").write_text("<html></html>")
    results_dir = tmp_path / "comparison_results"
    results_dir.mkdir(exist_ok=True)
    for query_id in ("combo_001", "combo_002"):
        (results_dir / f"edge_presence_matrix_query_{query_id}.csv").write_text("edge,d1\n")
        (results_dir / f"path_presence_matrix_query_{query_id}.csv").write_text("path,d1\n")
    similarity_dir = tmp_path / "similarity_matrices"
    similarity_dir.mkdir(exist_ok=True)
    for query_id in ("combo_001", "combo_002"):
        (similarity_dir / f"similarity_query_{query_id}.csv").write_text("dataset_1,dataset_2\n")
    report = generate_html_report(
        analyzer=Analyzer(),
        dataset_names=datasets,
        thresholds=[],
        mode_specific_note="",
        path_count_data=[],
        key_findings_per_threshold={},
        comparison_points=points,
    )
    return report


def test_combination_html_uses_full_query_keyed_report_shell(tmp_path):
    report = _build_query_report(tmp_path)

    # Standard shell contract: exactly one of each shared section, and the
    # shared section IDs match the Standard report taxonomy.
    assert report.count('id="summary"') == 1
    for section_id in ('neuron-counts', 'hemisphere-symmetry', 'similarity',
                       'networks', 'edge-matrices', 'path-matrices',
                       'conservation', 'overlap-matrices', 'statistics'):
        assert f'id="{section_id}"' in report, section_id
    assert 'id="threshold-provenance"' in report
    # Standard network controls are reused per query (Phase B item 6).
    assert report.count('id="network_combo_00') == 2
    # JSON string arguments are HTML-escaped inside inline attributes; the
    # browser decodes these entities before executing the handler.
    assert 'toggleNetworkFilter(&quot;combo_001&quot;)' in report
    assert 'toggleNetworkFilter(&quot;combo_002&quot;)' in report
    assert 'showNetworkTab(&quot;combo_001&quot;, this)' in report
    assert 'showNetworkTab(&quot;combo_002&quot;, this)' in report
    assert 'event.target' not in report
    # Per-query similarity heatmap cards use the shared four-representative
    # set (v2.2: edge/path/graph levels)
    assert 'id="jaccard_combo_001"' in report
    assert 'id="cosine_combo_001"' in report
    assert 'id="path_jaccard_combo_001"' in report
    assert 'id="netsimile_combo_001"' in report
    assert 'id="jaccard_combo_002"' in report
    # Overlap heatmaps with the count/proportion toggle per query.
    assert 'id="edge_overlap_combo_001"' in report
    assert 'id="path_overlap_combo_002"' in report
    assert 'updateOverlapMode_combo_001' in report
    # Statistics tabs and the query-axis 2x2 trends plot.
    assert 'id="stats_tab_combo_001"' in report
    assert 'id="stats_tab_combo_002"' in report
    assert 'Similarity Trends Across Query Rows' in report
    assert 'Query (display order)' in report
    # Conservation donuts per query and query-keyed conserved-graph links.
    assert 'id="cons_edge_combo_001"' in report
    assert 'id="cons_path_combo_002"' in report
    assert 'conserved_network_tcombo_001_network.html' in report
    # Edge/path matrix query tabs plus the per-dataset view.
    assert 'id="edge-matrices_query_tab_combo_001"' in report
    assert 'id="edge-matrices_dataset_tab_d1"' in report
    assert 'id="path-matrices_query_tab_combo_002"' in report
    assert 'data-matrix-mode="query"' in report
    assert 'data-matrix-query-key="combo_001"' in report
    assert 'data-matrix-dataset-key="d1"' in report
    assert 'switch_edge-matrices_mode' not in report
    assert 'show_path-matrices_query_tab' not in report
    assert 'onclick="switch_' not in report
    assert '<strong>7</strong>' in report
    assert 'Plotly.newPlot' in report
    assert 'edge_presence_matrix_query_combo_001.csv' in report
    assert 'edge_presence_matrix_combination_' not in report
    # Path length column preserved in the shared path presence table.
    assert '<th>Item</th><th>Len</th>' in report
    assert '<strong>source -> inter -> target</strong></td><td>2</td>' in report
    # Query ids must not appear in numeric-threshold chart phrasing
    # (Phase G item 5 / F-RPT-003).
    assert 'Threshold=combo' not in report
    assert 'Threshold = combo' not in report


def test_combination_toc_is_complete(tmp_path):
    """Phase G / F-RPT-006: the shared TOC lists every rendered section and
    links the Custom provenance section; the type-mapping entry appears only
    when auto type mapping is enabled."""
    report = _build_query_report(tmp_path)
    assert 'href="#threshold-provenance"' in report
    assert 'href="#overlap-matrices"' in report
    assert 'href="#type-mapping"' not in report

    (tmp_path / "with_mapping").mkdir()
    report_with_mapping = _build_query_report(
        tmp_path / "with_mapping", extra_parameters={"auto_type_mapping": True})
    assert 'href="#type-mapping"' in report_with_mapping


def test_combination_similarity_html_rows_match_used_data_csv(tmp_path):
    """F-RPT-002 (rev 2026-09-15): the pair-metrics table was dropped from
    the Similarity section — the annotated heatmaps plus the data links
    remain, and the used-data CSV stays the machine-readable source of the
    same metric values."""
    report = _build_query_report(tmp_path)

    used_csv = tmp_path / "comparison_report_used_data" / "similarity_by_query.csv"
    assert used_csv.exists()
    used = pd.read_csv(used_csv)
    assert set(used["query_id"]) == {"combo_001"}
    # The HTML pair table is gone.
    assert 'Pair metrics' not in report
    # The heatmap card carries the same jaccard value inside its JSON embed
    # (d1-d2 pair = 0.5 on a 3-dataset diagonal matrix).
    assert 'id="jaccard_combo_001"' in report
    assert ('const jaccard = [[1.0, 0.5, null], [0.5, 1.0, null], '
            '[null, null, 1.0]]') in report
    # The data links survive: used-data export + per-query export.
    assert 'similarity_by_query.csv' in report
    assert 'similarity_query_combo_001.csv' in report
    # And the CSV row count per query is unchanged.
    assert len(used[used["query_id"] == "combo_001"]) == 1


def test_combination_summary_has_ratio_and_probability_charts(tmp_path):
    """Phase B: the Custom summary carries the four Standard summary charts
    and their used-data files."""
    report = _build_query_report(tmp_path)
    assert 'queryEdgeCountChart' in report
    assert 'queryTotalWeightChart' in report
    assert 'queryAvgRatioChart' in report
    assert 'queryAvgProbChart' in report
    used_dir = tmp_path / "comparison_report_used_data"
    for name in ("edge_count_data_by_query.csv", "total_weight_data_by_query.csv",
                 "avg_ratio_data_by_query.csv", "avg_prob_data_by_query.csv",
                 "provenance_by_query.csv"):
        assert (used_dir / name).exists(), name
    # 2026-09-16: the per-query summary charts render as LINES (user
    # request — grouped bars were unreadable at 4 datasets x N queries),
    # same convention as the similarity-trends grid.
    card_start = report.index('Edges, Weight, Ratio and Probability')
    card_js = report[card_start:card_start + 6000]
    assert "type: 'scatter', mode: 'lines+markers'" in card_js
    assert "type: 'bar'" not in card_js
    assert 'barmode' not in card_js


def test_combination_parameters_validation_is_behavioral():
    """Phase G item 10: combination validation raises for a single selected
    dataset, for rows missing a dataset threshold, and for unknown datasets."""
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    for entry in (str(project_root), str(project_root / "src")):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    from comparison import ComparisonParameters

    base = dict(
        datasets=["hemibrain:v1.2.1", "male-cns:v1.0"],
        threshold_mode="combinations",
    )
    good_rows = [
        {"id": "q1", "thresholds": {"hemibrain:v1.2.1": 3, "male-cns:v1.0": 5}},
        {"id": "q2", "thresholds": {"hemibrain:v1.2.1": 4, "male-cns:v1.0": 6}},
    ]
    # A valid two-dataset configuration constructs cleanly.
    params = ComparisonParameters(**base, threshold_combinations=good_rows)
    assert len(params.get_threshold_queries()) == 2

    with pytest.raises(ValueError, match="at least two"):
        ComparisonParameters(
            datasets=["hemibrain:v1.2.1"],
            threshold_mode="combinations",
            threshold_combinations=[
                {"id": "q1", "thresholds": {"hemibrain:v1.2.1": 3}},
            ],
        )

    with pytest.raises(ValueError, match="missing thresholds"):
        ComparisonParameters(
            **base,
            threshold_combinations=[
                {"id": "q1", "thresholds": {"hemibrain:v1.2.1": 3}},
            ],
        )

    with pytest.raises(ValueError, match="unknown datasets"):
        ComparisonParameters(
            **base,
            threshold_combinations=[
                {"id": "q1",
                 "thresholds": {"hemibrain:v1.2.1": 3, "male-cns:v1.0": 5,
                                "banc:v888": 2}},
            ],
        )


def test_raw_schedule_diagnostics_are_labeled(tmp_path):
    """Phase G item 9: in combinations mode the alignment/density exports
    carry threshold_scope=raw_run_schedule_diagnostic, keeping the raw-run
    schedule off the query comparison axis."""
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    for entry in (str(project_root), str(project_root / "src")):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    from comparison.comparison_analyzer import ComparisonAnalyzer
    from comparison import ComparisonParameters

    params = ComparisonParameters(
        datasets=["hemibrain:v1.2.1", "male-cns:v1.0"],
        threshold_mode="combinations",
        threshold_combinations=[
            {"id": "q1",
             "thresholds": {"hemibrain:v1.2.1": 3, "male-cns:v1.0": 5}},
        ],
        verbose=False,
    )
    params.output_folder = str(tmp_path)
    analyzer = ComparisonAnalyzer(params)
    results_dir = tmp_path / "comparison_results"
    results_dir.mkdir(parents=True)
    analyzer._export_threshold_alignment(results_dir)

    density = pd.read_csv(results_dir / "edge_density_per_threshold.csv")
    assert set(density["threshold_scope"]) == {"raw_run_schedule_diagnostic"}
    matrix = pd.read_csv(results_dir / "threshold_alignment_matrix.csv")
    assert set(matrix["threshold_scope"]) == {"raw_run_schedule_diagnostic"}
    best_path = results_dir / "threshold_alignment_best_matches.csv"
    if best_path.exists():
        best = pd.read_csv(best_path)
        assert set(best["threshold_scope"]) == {"raw_run_schedule_diagnostic"}


def test_standard_and_custom_pathfinding_suppress_ratio_probability(tmp_path, monkeypatch):
    """Phase G item 8 / F-RPT-004: both modes must pass None ratio/probability
    callbacks to the visualizer so by_ratio/by_probability folders are never
    emitted for pathfinding comparisons."""
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    for entry in (str(project_root), str(project_root / "src")):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    import pandas as pd
    from comparison import ComparisonParameters
    from comparison import comparison_analyzer as ca

    captured = {}

    class StubVisualizer:
        def __init__(self, *args, **kwargs):
            pass

        def save_all_plots(self, **kwargs):
            captured.update(kwargs)

    import comparison.visualizations as vis_module
    monkeypatch.setattr(vis_module, 'ComparisonVisualizer', StubVisualizer)

    datasets = ["hemibrain:v1.2.1", "male-cns:v1.0"]
    raw = {
        dataset: {
            3: pd.DataFrame({
                "type_pre": ["source"],
                "type_post": ["target"],
                "weight": [3],
            })
        }
        for dataset in datasets
    }
    aligned = pd.DataFrame(
        {dataset: [3] for dataset in datasets},
        index=["source -> target"],
    )

    def run(mode):
        captured.clear()
        if mode == "standard":
            params = ComparisonParameters(datasets=datasets, thresholds=[3])
            analyzer = ca.ComparisonAnalyzer(params)
            analyzer.raw_results = raw
            analyzer.get_mapped_results = lambda: raw
            analyzer.get_aligned_data = lambda _t: aligned
            analyzer._analysis_thresholds = lambda: [3]
            analyzer.get_cached_similarities = lambda _t: pd.DataFrame()
            analyzer._get_path_data_for_threshold = lambda _t: pd.DataFrame()
            analyzer.comparison_report = {}
        else:
            params = ComparisonParameters(
                datasets=datasets,
                threshold_mode="combinations",
                threshold_combinations=[
                    {"id": "q1",
                     "thresholds": {datasets[0]: 3, datasets[1]: 3}},
                ],
            )
            analyzer = ca.ComparisonAnalyzer(params)
            analyzer.raw_results = raw
            analyzer.get_threshold_queries = lambda: [
                {"id": "q1", "thresholds": {datasets[0]: 3, datasets[1]: 3}}]
            analyzer.get_mapped_results = lambda: {
                ds: {"q1": raw[ds][3]} for ds in datasets}
            analyzer.get_aligned_data_for_query = lambda _q: aligned
            analyzer._get_path_data_for_query = lambda _q: pd.DataFrame()
            analyzer._similarity_cache = {}
            analyzer._query_record = lambda _k: {
                "id": "q1", "thresholds": {datasets[0]: 3, datasets[1]: 3}}
            analyzer.comparison_report = {}
        # This test covers the visualizer callback contract.  The conserved
        # graph orchestration has its own regression test and should not write
        # generated HTML into the repository during this callback test.
        analyzer._generate_vispath_visualizations = lambda _dir: None
        analyzer.visualize_conserved_paths_all_thresholds = lambda **kw: []
        analyzer._generate_visualizations(str(tmp_path / mode))
        return captured

    for mode in ("standard", "combinations"):
        kwargs = run(mode)
        assert kwargs.get("ratio_data_func") is None, mode
        assert kwargs.get("prob_data_func") is None, mode


def test_combination_report_renders_query_resolution_section(tmp_path):
    """The combination Quick Navigation links #query-resolution — the
    section itself must render too (2026-09-16 report inspection found the
    dead anchor: nav item present, section Standard-only)."""
    report = _build_query_report(tmp_path)
    assert 'id="query-resolution"' in report
    assert 'Query Resolution' in report
    assert '<td>source</td>' in report
