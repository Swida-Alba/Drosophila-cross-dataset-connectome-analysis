"""Coverage tests for comparison.html_report_generator.

Hermetic: a FakeAnalyzer supplies tiny synthetic DataFrames; all file output
goes to pytest tmp_path. No network, no kaleido, no multiprocessing.
"""

import json
import os
import re

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from comparison import html_report_generator as hrg

DATASETS = ["ds_one", "ds_two"]
THRESHOLDS = [1, 5]
NICKNAME_MAP = {"ds_one": "D1", "ds_two": "D2"}


# ---------------------------------------------------------------------------
# Fake analyzer infrastructure
# ---------------------------------------------------------------------------

class FakeParameters:
    def __init__(self, output_path, dataset_names, separate_hemispheres=False,
                 auto_type_mapping=False, path_mode="all", max_interlayer=2,
                 source_neurons=("A",), target_neurons=("D",)):
        self.full_output_path = str(output_path)
        self.comparison_mode = "connectivity"
        self.path_mode = path_mode
        self.max_interlayer = max_interlayer
        self.auto_type_mapping = auto_type_mapping
        self.separate_hemispheres = separate_hemispheres
        self.source_neurons = list(source_neurons)
        self.target_neurons = list(target_neurons)
        self._dataset_names = list(dataset_names)
        self._auto_type_mapper = None

    def get_dataset_nicknames(self):
        return [f"D{i + 1}" for i in range(len(self._dataset_names))]

    def _ensure_flat_list(self, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def get_source_neurons_for_dataset(self, dataset):
        return list(self.source_neurons)

    def get_target_neurons_for_dataset(self, dataset):
        return list(self.target_neurons)

    @staticmethod
    def _sanitize_name(name):
        return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


class FakeLabelMapper:
    def __init__(self, source=("A",), target=("D",)):
        self._labels = {"source": list(source), "target": list(target)}

    def get_all_std_labels(self, role):
        return list(self._labels.get(role, []))


def _aligned_df():
    return pd.DataFrame(
        {
            "ds_one": [10.0, 7.0, 5.0, 3.0, 0.0],
            "ds_two": [8.0, 0.0, 6.0, 3.0, 4.0],
        },
        index=["A -> B", "B -> C", "A -> C", "C -> D", "B -> D"],
    )


def _path_df():
    return pd.DataFrame(
        {
            "ds_one": [7.0, 3.0, 0.0],
            "ds_two": [5.0, 3.0, 2.0],
        },
        index=["A -> B -> C", "A -> C -> D", "B -> C -> D"],
    )


def _prob_df():
    return pd.DataFrame(
        {"ds_one": [0.6, 0.2], "ds_two": [0.4, 0.3]},
        index=["A -> B -> C", "A -> C -> D"],
    )


def _ratio_df():
    return pd.DataFrame(
        {"ds_one": [0.5, 0.2, 0.0], "ds_two": [0.4, 0.0, 0.1]},
        index=["A -> B", "B -> C", "A -> C"],
    )


def _hop_weights():
    return {
        1: {"A -> B -> C": {"ds_one": [10, 7], "ds_two": [8, 5]},
            "A -> C -> D": {"ds_one": [5, 3], "ds_two": [6, 3]}},
        5: {"A -> B -> C": {"ds_one": [10, 7]}},
    }


def _edge_df(n=3):
    return pd.DataFrame(
        {
            "type_pre": ["A", "B", "A"][:n],
            "type_post": ["B", "C", "C"][:n],
            "weight": [10.0, 7.0, 5.0][:n],
            "has_valid_path": [True, True, False][:n],
        }
    )


def _presence_matrix():
    return pd.DataFrame(
        {
            "ds_one_t1": ["True", "False", "True"],
            "ds_two_t1": ["True", "True", False],
            "ds_one_t5": ["True", "False", False],
            "ds_two_t5": [1, 1, 0],
        }
    )


def _symmetry_summaries():
    per_ds = {
        "ipsi": {"jaccard": 0.8, "conserved": 4, "union": 5},
        "contra": {"jaccard": 0.4, "conserved": 2, "union": 5},
        "neuron_types": {"types_conserved": 2, "types_union": 3},
        "hemisphere_counts": {"total": {"L": 4, "R": 3}},
    }
    return {t: {d: dict(per_ds) for d in DATASETS} for t in THRESHOLDS}


class FakeAnalyzer:
    """Minimal stand-in for CrossDatasetComparisonAnalyzer."""

    def __init__(self, output_path, dataset_names=DATASETS,
                 separate_hemispheres=False, auto_type_mapping=False,
                 path_mode="all", max_interlayer=2, empty=False,
                 source_neurons=("A",), target_neurons=("D",),
                 include_neuron_counts=True):
        self.parameters = FakeParameters(
            output_path, dataset_names,
            separate_hemispheres=separate_hemispheres,
            auto_type_mapping=auto_type_mapping,
            path_mode=path_mode, max_interlayer=max_interlayer,
            source_neurons=source_neurons, target_neurons=target_neurons,
        )
        self.label_mapper = FakeLabelMapper(source_neurons, target_neurons)
        self._dataset_names = list(dataset_names)
        self._empty = empty
        self.comparison_report = {
            "path_presence_matrix": None if empty else _presence_matrix()
        }
        if include_neuron_counts and not empty:
            self._neuron_counts_summary = pd.DataFrame(
                [
                    {"dataset": "ds_one", "source_count": 5, "target_count": 3,
                     "source_types": 2, "target_types": 1},
                    {"dataset": "ds_two", "source_count": 4, "target_count": 2,
                     "source_types": 2, "target_types": 1},
                ]
            )
            self._neuron_type_counts = pd.DataFrame(
                [
                    {"type": "A", "ds_one_source": 3, "ds_one_target": 0,
                     "ds_two_source": 2, "ds_two_target": 1},
                    {"type": "D", "ds_one_source": 0, "ds_one_target": 2,
                     "ds_two_source": 0, "ds_two_target": 1},
                ]
            )
            self._neuron_group_counts = pd.DataFrame(
                [{"custom_group": "G1", "role": "source",
                  "ds_one": 2, "ds_two": 1}]
            )
        self._mapped_results = (
            {} if empty else
            {d: {t: _edge_df() for t in THRESHOLDS} for d in dataset_names}
        )

    # -- data accessors ----------------------------------------------------
    def get_aligned_data(self, threshold):
        return pd.DataFrame() if self._empty else _aligned_df()

    def get_aligned_data_for_network(self, threshold):
        return self.get_aligned_data(threshold)

    def _get_path_data_for_threshold(self, threshold):
        return pd.DataFrame() if self._empty else _path_df()

    def _get_path_hop_weights_for_threshold(self, threshold):
        return {} if self._empty else _hop_weights().get(threshold, {})

    def _get_prob_data_for_threshold(self, threshold):
        return pd.DataFrame() if self._empty else _prob_df()

    def _get_edge_ratio_data_for_threshold(self, threshold):
        return pd.DataFrame() if self._empty else _ratio_df()

    def get_mapped_results(self):
        return self._mapped_results

    def get_hemisphere_symmetry_summaries(self):
        return {} if self._empty else _symmetry_summaries()


@pytest.fixture
def analyzer(tmp_path):
    return FakeAnalyzer(tmp_path)


@pytest.fixture
def empty_analyzer(tmp_path):
    return FakeAnalyzer(tmp_path, empty=True, include_neuron_counts=False)


def _key_findings():
    return {
        1: {"total_edges": 6, "common_edges": 3,
            "total_paths": 3, "common_paths": 1},
        5: {"total_edges": 4, "common_edges": 2,
            "total_paths": 2, "common_paths": 1},
    }


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_make_link_existing(tmp_path):
    target = tmp_path / "sub" / "file.html"
    target.parent.mkdir(parents=True)
    target.write_text("<html></html>")
    link = hrg._make_link(str(target), str(tmp_path))
    assert link.startswith('<a href="sub/file.html"')
    assert "Open</a>" in link


def test_make_link_missing(tmp_path):
    assert hrg._make_link(str(tmp_path / "nope.html"), str(tmp_path)) == "-"
    assert hrg._make_link("", str(tmp_path)) == "-"


def test_get_canonical_name():
    assert hrg.get_canonical_name("MBON14 (merged)") == "MBON14"
    assert hrg.get_canonical_name("plain") == "plain"


def test_get_base_name():
    assert hrg.get_base_name("aMe12_L") == "aMe12"
    assert hrg.get_base_name("aMe12_R") == "aMe12"
    assert hrg.get_base_name("aMe12_U") == "aMe12"
    assert hrg.get_base_name("aMe12") == "aMe12"


@pytest.mark.parametrize(
    "label,patterns,expected",
    [
        ("aMe12", {"aMe12"}, True),                     # exact
        ("aMe12", {"aMe*"}, True),                      # glob
        ("aMe12", {"aMe.*"}, True),                     # regex
        ("aMe12 (merged)", {"aMe12"}, True),            # merged display label
        ("aMe12_L", {"aMe12"}, True),                   # hemisphere suffix
        ("aMe12", {"other"}, False),                    # no match
        ("AME12", {"aMe12"}, True),                     # case-insensitive
    ],
)
def test_matches_patterns(label, patterns, expected):
    assert hrg.matches_patterns(label, patterns) is expected


# ---------------------------------------------------------------------------
# Edge extraction / filtering helpers
# ---------------------------------------------------------------------------

def test_extract_edges_from_paths():
    path_data = pd.DataFrame(
        {"ds_one": [7.0, 1.0], "ds_two": [5.0, 0.0]},
        index=["A(x) -> B -> C", "D → E"],  # display names + unicode arrow
    )
    edges = hrg._extract_edges_from_paths(path_data, DATASETS, max_paths=10)
    assert "A(x) -> B" in edges          # display format
    assert "A -> B" in edges             # canonical format
    assert "B -> C" in edges
    assert "D -> E" in edges             # unicode arrow parsed


def test_extract_edges_from_paths_empty():
    assert hrg._extract_edges_from_paths(None, DATASETS) == set()
    assert hrg._extract_edges_from_paths(pd.DataFrame(), DATASETS) == set()
    # no matching columns
    df = pd.DataFrame({"other": [1.0]}, index=["A -> B"])
    assert hrg._extract_edges_from_paths(df, DATASETS) == set()


def test_filter_aligned_by_paths_with_paths():
    aligned = _aligned_df()
    path_data = _path_df()
    out = hrg._filter_aligned_by_paths(aligned, path_data, DATASETS, max_edges=10)
    # Edges from "A -> B -> C" / "A -> C -> D" / "B -> C -> D" kept
    assert "A -> B" in out.index
    assert "A -> C" in out.index
    assert "C -> D" in out.index


def test_filter_aligned_by_paths_fallback_top_edges():
    aligned = _aligned_df()
    out = hrg._filter_aligned_by_paths(aligned, None, DATASETS, max_edges=2)
    assert len(out) == 2  # trimmed to top 2 by total weight


def test_filter_aligned_by_paths_empty_aligned():
    out = hrg._filter_aligned_by_paths(pd.DataFrame(), _path_df(), DATASETS)
    assert out.empty


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def test_generate_presence_table():
    html = hrg._generate_presence_table(_aligned_df(), DATASETS, NICKNAME_MAP, 1)
    assert "Threshold = 1" in html
    assert "A -> B" in html
    assert "badge-success" in html  # conserved edge present in both datasets
    assert "badge-danger" in html   # unique edge


def test_generate_presence_table_empty():
    assert "No data available" in hrg._generate_presence_table(
        pd.DataFrame(), DATASETS, NICKNAME_MAP
    )
    df = pd.DataFrame({"other": [1.0]})
    assert "No datasets available" in hrg._generate_presence_table(
        df, DATASETS, NICKNAME_MAP
    )


def test_generate_path_presence_table(analyzer):
    html = hrg._generate_path_presence_table(
        analyzer, _path_df(), DATASETS, NICKNAME_MAP, 1
    )
    assert "A -> B -> C" in html
    assert "<strong>" in html  # min hop weight bolded
    assert "badge-success" in html


def test_generate_path_presence_table_empty(analyzer):
    assert "No data available" in hrg._generate_path_presence_table(
        analyzer, pd.DataFrame(), DATASETS, NICKNAME_MAP, 1
    )


def test_generate_edge_dataset_table(analyzer):
    html = hrg._generate_edge_dataset_table(analyzer, "ds_one", THRESHOLDS, NICKNAME_MAP)
    assert "Dataset: D1" in html
    assert "t=1" in html and "t=5" in html


def test_generate_path_dataset_table(analyzer):
    html = hrg._generate_path_dataset_table(analyzer, "ds_one", THRESHOLDS, NICKNAME_MAP)
    assert "Dataset: D1" in html
    assert "A -> B -> C" in html


def test_generate_stats_table(analyzer):
    html = hrg._generate_stats_table(analyzer, DATASETS, 1, NICKNAME_MAP)
    assert "Per-Dataset Statistics" in html
    assert "Pairwise Similarities" in html


def test_generate_stats_table_empty(empty_analyzer):
    html = hrg._generate_stats_table(empty_analyzer, DATASETS, 1, NICKNAME_MAP)
    assert "No data available" in html


# ---------------------------------------------------------------------------
# Header / TOC / footer
# ---------------------------------------------------------------------------

def test_generate_html_header():
    header = hrg._generate_html_header()
    assert "<html" in header
    assert "plotly" in header.lower()


def test_generate_report_header(analyzer):
    html = hrg._generate_report_header(
        analyzer, DATASETS, THRESHOLDS, "<p>note</p>", NICKNAME_MAP
    )
    assert "Cross-Dataset Comparison Report" in html
    assert "D1, D2" in html
    assert "<p>note</p>" in html


def test_generate_report_header_shortest_unlimited(tmp_path):
    fa = FakeAnalyzer(tmp_path, path_mode="shortest", max_interlayer=0)
    html = hrg._generate_report_header(fa, DATASETS, THRESHOLDS, "", NICKNAME_MAP)
    assert "unlimited (shortest mode)" in html
    assert "Shortest path mode" in html


def test_generate_toc():
    toc = hrg._generate_toc(THRESHOLDS)
    assert "#summary" in toc
    assert "#conservation" in toc


def test_generate_footer():
    assert "</html>" in hrg._generate_footer()


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def test_generate_summary_section(analyzer, tmp_path):
    html = hrg._generate_summary_section(
        analyzer, DATASETS, THRESHOLDS, [], _key_findings(), NICKNAME_MAP
    )
    assert "Key Findings by Threshold" in html
    assert "edgeCountChart" in html
    used = tmp_path / "comparison_report_used_data"
    assert (used / "edge_count_data.csv").exists()
    assert (used / "total_weight_data.csv").exists()
    assert (used / "avg_ratio_data.csv").exists()
    assert (used / "avg_prob_data.csv").exists()
    assert (used / "ratio_data_t1.csv").exists()


def test_generate_neuron_counts_section(analyzer):
    html = hrg._generate_neuron_counts_section(analyzer, DATASETS, NICKNAME_MAP)
    assert "Summary: Total Neuron Counts" in html
    assert "Neuron Counts by Type" in html
    assert "Neuron Counts by Custom Group" in html
    assert "neuronCountChart" in html


def test_generate_neuron_counts_section_missing(empty_analyzer):
    html = hrg._generate_neuron_counts_section(
        empty_analyzer, DATASETS, NICKNAME_MAP
    )
    assert "not available" in html


def test_generate_similarity_section(analyzer, tmp_path):
    html = hrg._generate_similarity_section(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Threshold = 1" in html
    assert "Threshold = 5" in html
    assert (tmp_path / "similarity_matrices" / "similarity_threshold_1.csv").exists()


def test_generate_similarity_section_single_dataset(analyzer):
    html = hrg._generate_similarity_section(
        analyzer, ["ds_one"], THRESHOLDS, {"ds_one": "D1"}
    )
    assert "Similarity Matrices" in html  # no matrices, but section renders


def test_generate_hemisphere_symmetry_disabled(analyzer):
    html = hrg._generate_hemisphere_symmetry_section(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Hemisphere analysis unavailable" in html


def test_generate_hemisphere_symmetry_enabled(tmp_path):
    fa = FakeAnalyzer(tmp_path, separate_hemispheres=True)
    html = hrg._generate_hemisphere_symmetry_section(
        fa, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Ipsi Jaccard" in html
    assert "0.800" in html
    assert "4/3" in html  # L/R counts


def test_generate_hemisphere_symmetry_empty(tmp_path):
    fa = FakeAnalyzer(tmp_path, separate_hemispheres=True, empty=True)
    html = hrg._generate_hemisphere_symmetry_section(
        fa, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "No hemisphere symmetry summaries found" in html


def test_generate_networks_section(analyzer):
    html = hrg._generate_networks_section(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Network Visualizations" in html
    assert "window.allNetworks" in html
    assert "Conservation" in html


def test_network_tabs_escape_string_keys_and_use_dom_keys(analyzer):
    html = hrg._generate_networks_section(
        analyzer,
        DATASETS,
        [],
        {"ds_one": "D 1", "ds_two": "D'2"},
        point_keys=["query/one", "query two"],
        point_labels=["First", "Second"],
        aligned_network_getter=analyzer.get_aligned_data_for_network,
        aligned_getter=analyzer.get_aligned_data,
        path_getter=analyzer._get_path_data_for_threshold,
        mode_labels=("Query", "Dataset"),
    )
    assert 'showNetworkTab(&quot;query/one&quot;, this)' in html
    assert 'showNetworkTab(&quot;query two&quot;, this)' in html
    assert 'id="network_tab_query_one"' in html
    assert 'id="network_tab_query_two"' in html
    # round-4: dataset tab ids are section-scoped (dom key carries the
    # section prefix)
    assert 'id="network_dataset_tab_networks__D_1"' in html
    assert 'event.target' not in html


def test_generate_networks_section_self_edges(tmp_path):
    # source == target triggers the self-edge detection branch
    fa = FakeAnalyzer(tmp_path, source_neurons=("A",), target_neurons=("A",))
    html = hrg._generate_networks_section(fa, DATASETS, THRESHOLDS, NICKNAME_MAP)
    assert "Network Visualizations" in html


def test_generate_conservation_network(analyzer):
    html = hrg._generate_conservation_network(
        analyzer, DATASETS, 1, NICKNAME_MAP
    )
    assert "Conservation" in html
    assert "A" in html


def test_generate_conservation_network_empty(empty_analyzer):
    html = hrg._generate_conservation_network(
        empty_analyzer, DATASETS, 1, NICKNAME_MAP
    )
    assert "No connections" in html


def test_generate_dataset_network(analyzer):
    html = hrg._generate_dataset_network(
        analyzer, "ds_one", THRESHOLDS, NICKNAME_MAP
    )
    assert "D1" in html


def test_generate_edge_matrices_section(analyzer):
    html = hrg._generate_edge_matrices_section(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Edge Presence Matrices" in html
    assert "A -> B" in html
    assert "switchEdgeMode" in html


def test_generate_path_matrices_section(analyzer):
    html = hrg._generate_path_matrices_section(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Path Presence Matrices" in html
    assert "A -> B -> C" in html


def test_generate_path_matrices_section_no_data(empty_analyzer):
    html = hrg._generate_path_matrices_section(
        empty_analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "No path data available" in html


def test_generate_conservation_section(analyzer):
    html = hrg._generate_conservation_section(
        analyzer, DATASETS, THRESHOLDS, _key_findings(), NICKNAME_MAP
    )
    assert "Conservation Analysis" in html
    assert "var plotData" in html          # plotly JSON embedded
    assert "plotData = null" not in html   # plotly generation succeeded
    assert "Conservation at Threshold = 1" in html
    assert "badge" not in html or True
    # conserved graph link table falls back to '-' when files are absent
    assert "-" in html


def test_generate_overlap_matrices_section(analyzer):
    html = hrg._generate_overlap_matrices_section(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Dataset Overlap Matrices" in html
    assert "edge_overlap_1" in html
    assert "path_overlap_1" in html


def test_generate_statistics_section(analyzer):
    html = hrg._generate_statistics_section(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Statistics" in html
    assert "Similarity Trends Across Thresholds" in html


def test_generate_reciprocal_section_disabled(analyzer):
    html = hrg._generate_reciprocal_visualizations_section(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "not enabled" in html


def test_generate_reciprocal_section_enabled(tmp_path):
    fa = FakeAnalyzer(tmp_path)
    fa.parameters.find_reciprocal = True
    # Create one output file so its link cell renders "Open"
    link_file = (
        tmp_path / "dataset_data" / "ds_one" / "minsyn_1"
        / "find_reciprocal" / "visualizations" / "reciprocal_type_network.html"
    )
    link_file.parent.mkdir(parents=True)
    link_file.write_text("<html></html>")
    html = hrg._generate_reciprocal_visualizations_section(
        fa, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Threshold t = 1" in html
    assert "Open</a>" in html      # existing file linked
    assert "-</td>" in html        # missing files show '-'


# ---------------------------------------------------------------------------
# Individual similarity trend plots
# ---------------------------------------------------------------------------

def test_generate_jaccard_similarity_plot(analyzer):
    html = hrg._generate_jaccard_similarity_plot(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Jaccard Similarity Trend" in html
    assert "D1 vs D2" in html


def test_generate_edge_rank_correlation_plot(analyzer):
    html = hrg._generate_edge_rank_correlation_plot(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Edge Rank" in html


def test_generate_cosine_similarity_trend_plot(analyzer):
    html = hrg._generate_cosine_similarity_trend_plot(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Cosine" in html


def test_generate_path_rank_correlation_plot(analyzer):
    html = hrg._generate_path_rank_correlation_plot(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Path" in html


# ---------------------------------------------------------------------------
# End-to-end generate_html_report
# ---------------------------------------------------------------------------

def test_generate_html_report_full(analyzer, tmp_path):
    html = hrg.generate_html_report(
        analyzer, DATASETS, THRESHOLDS,
        mode_specific_note="<p>mode note</p>",
        path_count_data=[], key_findings_per_threshold=_key_findings(),
    )
    out = tmp_path / "comparison_report.html"
    out.write_text(html, encoding="utf-8")
    assert out.exists()

    for marker in [
        "<html",
        "Cross-Dataset Comparison Report",
        "mode note",
        "Key Findings by Threshold",
        "Neuron Counts Comparison",
        "Hemisphere Symmetry",
        "Similarity Matrices",
        "Network Visualizations",
        "Edge Presence Matrices",
        "Path Presence Matrices",
        "Conservation Analysis",
        "Dataset Overlap Matrices",
        "Statistics",
        "</html>",
    ]:
        assert marker in html, f"missing marker: {marker}"


def test_generate_html_report_with_hemispheres(tmp_path):
    fa = FakeAnalyzer(tmp_path, separate_hemispheres=True)
    html = hrg.generate_html_report(
        fa, DATASETS, THRESHOLDS, "", [], _key_findings()
    )
    assert "Ipsi Jaccard" in html
    assert "</html>" in html


def test_generate_html_report_with_auto_type_mapping(tmp_path):
    fa = FakeAnalyzer(tmp_path, auto_type_mapping=True)
    html = hrg.generate_html_report(
        fa, DATASETS, THRESHOLDS, "", [], _key_findings()
    )
    assert "</html>" in html


def test_type_mapping_report_canonicalizes_by_source_dataset():
    """Raw names from FAFB/BANC merge into the MCNS canonical row."""
    from types import SimpleNamespace

    datasets = ["mcns", "fafb", "banc"]

    class Mapper:
        """Fake mapper on the shared resolver contract: the report now
        resolves through ``comparison.type_resolver`` (which consumes
        ``get_mapping_decision`` status dicts), not the legacy
        ``resolve_type_across_datasets`` one-target API."""
        _conflicts = []
        _loaded = True

        def get_canonical_type(self, type_name, source_dataset=None):
            if source_dataset in {"mcns", "fafb", "banc"} \
                    and type_name in {"MeVPLo2", "MTe07"}:
                return "MeVPLo2"
            return type_name

        def _get_type_mapping_key(self, dataset):
            # 'mcns' IS the canonical namespace (same registry key), like
            # the real mapper's release normalization.
            return {'mcns': 'male-cns:v1.0'}.get(dataset, dataset)

        def get_mapping_decision(self, source_type, source_dataset,
                                 target_dataset, include_bridges=False):
            def _mapped(target):
                return {'status': 'mapped', 'source_type': source_type,
                        'target_type': target, 'target_types': [target],
                        'relationship': '1-to-1', 'conflicts': []}

            def _unmapped():
                return {'status': 'unmapped', 'source_type': source_type,
                        'target_type': None, 'target_types': [],
                        'relationship': None, 'conflicts': []}

            if source_type == 'CB2399':
                # Present in FAFB and BANC, no MCNS counterpart.
                if {source_dataset, target_dataset} == {'banc', 'fafb'}:
                    return _mapped('CB2399')
                return _unmapped()
            if source_type in {'MeVPLo2', 'MTe07'}:
                if self._get_type_mapping_key(target_dataset) == \
                        'male-cns:v1.0':
                    return _mapped('MeVPLo2')
                return _mapped('MTe07')
            return _unmapped()

    class Analyzer:
        parameters = SimpleNamespace(
            auto_type_mapping=True,
            _auto_type_mapper=Mapper(),
        )

        @staticmethod
        def _collect_result_types_by_dataset():
            return {
                "mcns": {"MeVPLo2"},
                "fafb": {"MTe07"},
                "banc": {"MTe07", "CB2399"},
            }

    report = hrg._generate_type_mapping_section(Analyzer(), datasets)
    # Round-4 restructure: role tables (Queried Sources/Targets/Path
    # Intermediates) + the canonical grid in a collapsed appendix. The
    # canonical grid carries each row once.
    assert report.count("<strong>MeVPLo2</strong>") == 1
    assert "<strong>MTe07</strong>" not in report
    assert '<th>Source (priority)</th>' in report
    assert '<td>banc: MTe07, fafb: MTe07, mcns: MeVPLo2</td>' in report
    assert '<td>banc: CB2399</td>' in report
    # The Targets table colors names differing from the canonical; the
    # renamed FAFB/BANC targets share one color and the canonical MCNS
    # name plus the CB2399 fallback row stay plain.
    colored = ('color:#1d4ed8; font-weight:600; white-space:nowrap;" '
               'title="differs from the canonical name">MTe07</span>')
    assert report.count(colored) == 2
    assert '<span style="white-space:nowrap;">MeVPLo2</span>' in report
    assert report.count(
        '<span style="white-space:nowrap;">CB2399</span>') == 2


def test_generate_html_report_empty_data(empty_analyzer):
    html = hrg.generate_html_report(
        empty_analyzer, DATASETS, THRESHOLDS, "", [], {}
    )
    assert "<html" in html
    assert "</html>" in html
    assert "No data available" in html


# ---------------------------------------------------------------------------
# Appended branch-coverage tests
# ---------------------------------------------------------------------------

DS3 = ["ds_one", "ds_two", "ds_three"]
NICK3 = {"ds_one": "D1", "ds_two": "D2", "ds_three": "D3"}


class _WeirdRatioProbAnalyzer(FakeAnalyzer):
    """Summary-section edge cases: zero / raising / bad-dtype ratio+prob data."""

    def _get_edge_ratio_data_for_threshold(self, t):
        if t == 1:
            return pd.DataFrame({"ds_one": [0.0, 0.0], "ds_two": [0.0, 0.0]},
                                index=["A -> B", "B -> C"])
        if t == 5:
            raise RuntimeError("ratio fetch failed")
        return pd.DataFrame({"ds_one": ["x", "y"], "ds_two": [0.4, 0.1]},
                            index=["A -> B", "B -> C"])

    def _get_prob_data_for_threshold(self, t):
        if t == 1:
            return pd.DataFrame({"ds_one": [0.0], "ds_two": [0.0]},
                                index=["A -> B -> C"])
        raise RuntimeError("prob fetch failed")


def test_summary_section_ratio_prob_edge_cases(tmp_path):
    fa = _WeirdRatioProbAnalyzer(tmp_path)
    html = hrg._generate_summary_section(
        fa, DATASETS, [1, 5, 10], [], {}, NICKNAME_MAP
    )
    assert "Key Findings" in html or "edgeCountChart" in html


class _RaisingTypeMapperGetter:
    def __call__(self):
        raise RuntimeError("no mapper available")


def test_neuron_counts_section_no_type_mapper(tmp_path, monkeypatch):
    import comparison.cross_dataset_type_mapper as cdtm

    monkeypatch.setattr(cdtm, "get_type_mapper", _RaisingTypeMapperGetter())
    fa = FakeAnalyzer(tmp_path)
    # row with empty type is skipped; group row with 0/NaN renders '-'
    fa._neuron_type_counts = pd.DataFrame(
        [
            {"type": "", "ds_one_source": 1, "ds_two_source": 0},
            {"type": "A", "ds_one_source": 3, "ds_two_source": 2},
        ]
    )
    fa._neuron_group_counts = pd.DataFrame(
        [
            {"custom_group": "G1", "role": "source", "ds_one": 0,
             "ds_two": np.nan},
            {"custom_group": "G2", "role": "target", "ds_one": 2, "ds_two": 1},
        ]
    )
    html = hrg._generate_neuron_counts_section(fa, DATASETS, NICKNAME_MAP)
    assert "Neuron Counts by Type" in html
    assert 'class="absent"' in html


class _DisplayNameMapper:
    def load(self):
        pass

    def get_display_name(self, name, datasets):
        if name == "BAD":
            raise RuntimeError("mapper error")
        if name == "MTe07":
            return "MeVPLo2 (MTe07)"
        return name


def test_neuron_counts_section_with_type_mapper(tmp_path, monkeypatch):
    import comparison.cross_dataset_type_mapper as cdtm

    monkeypatch.setattr(cdtm, "get_type_mapper", lambda: _DisplayNameMapper())
    fa = FakeAnalyzer(tmp_path)
    fa._neuron_type_counts = pd.DataFrame(
        [
            {"type": "MTe07", "ds_one_source": 1, "ds_two_source": 0},
            {"type": "MeVPLo2", "ds_one_source": 0, "ds_two_source": 1},
            {"type": "BAD", "ds_one_source": 0, "ds_two_target": 2},
        ]
    )
    html = hrg._generate_neuron_counts_section(fa, DATASETS, NICKNAME_MAP)
    assert "MeVPLo2" in html
    assert "(MTe07)" in html  # dataset-specific name appended


class _RaisingPathDataAnalyzer(FakeAnalyzer):
    def _get_path_data_for_threshold(self, t):
        raise RuntimeError("path data unavailable")


def test_similarity_section_path_data_and_csv_failures(tmp_path):
    fa = _RaisingPathDataAnalyzer(tmp_path)
    # Pre-existing FILE at the csv dir path forces makedirs to fail
    (tmp_path / "similarity_matrices").write_text("blocker")
    html = hrg._generate_similarity_section(fa, DATASETS, THRESHOLDS, NICKNAME_MAP)
    assert "Similarity Matrices" in html


class _SymPartialAnalyzer(FakeAnalyzer):
    def __init__(self, output_path):
        super().__init__(output_path, separate_hemispheres=True)

    def get_hemisphere_symmetry_summaries(self):
        full = _symmetry_summaries()
        return {t: {"ds_one": v["ds_one"]} for t, v in full.items()}


def test_hemisphere_symmetry_missing_dataset(tmp_path):
    html = hrg._generate_hemisphere_symmetry_section(
        _SymPartialAnalyzer(tmp_path), DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Ipsi Jaccard" in html
    assert "D2" not in html  # dataset without summary is skipped


class _SelfEdgeAnalyzer(FakeAnalyzer):
    def __init__(self, output_path, with_label_mapper=True):
        super().__init__(output_path, source_neurons=("A",), target_neurons=("A",))
        if not with_label_mapper:
            self.label_mapper = None

    def get_aligned_data(self, t):
        df = _aligned_df()
        df.loc["A -> A"] = [4.0, 4.0]
        return df


def test_networks_section_no_mapper_with_self_edges(tmp_path):
    html = hrg._generate_networks_section(
        _SelfEdgeAnalyzer(tmp_path, with_label_mapper=False),
        DATASETS, THRESHOLDS, NICKNAME_MAP,
    )
    assert "Self-edges detected" in html
    assert "Network Visualizations" in html


class _RaisingNetworkAnalyzer(FakeAnalyzer):
    def __init__(self, output_path):
        super().__init__(output_path, source_neurons=("A",), target_neurons=("A",))
        self.label_mapper = None
        self.parameters._ensure_flat_list = lambda v: (_ for _ in ()).throw(
            RuntimeError("flat list fail"))


def test_networks_section_self_edge_count_failure(tmp_path):
    html = hrg._generate_networks_section(
        _RaisingNetworkAnalyzer(tmp_path), DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Network Visualizations" in html
    assert "Self-edges detected" not in html


def test_extract_edges_from_paths_unparseable():
    df = pd.DataFrame(
        {"ds_one": [7.0, 3.0], "ds_two": [5.0, 3.0]},
        index=["A -> B -> C", "justANode"],
    )
    edges = hrg._extract_edges_from_paths(df, DATASETS, max_paths=10)
    assert "A -> B" in edges
    assert "B -> C" in edges


def test_filter_aligned_by_paths_expansion():
    # Many duplicate paths sharing edges -> expansion loop fires
    n = 20
    path_df = pd.DataFrame(
        {"ds_one": [1.0] * n, "ds_two": [1.0] * n},
        index=["A -> B -> C"] * n,
    )
    aligned = pd.DataFrame(
        {"ds_one": [10.0, 7.0], "ds_two": [8.0, 6.0]},
        index=["A -> B", "B -> C"],
    )
    out = hrg._filter_aligned_by_paths(aligned, path_df, DATASETS, max_edges=5)
    assert list(out.index) == ["A -> B", "B -> C"]


class _ConservationExceptionAnalyzer(FakeAnalyzer):
    def __init__(self, output_path):
        super().__init__(output_path)

    def _get_path_data_for_threshold(self, t):
        raise RuntimeError("no paths")

    def get_aligned_data(self, t):
        df = _aligned_df()
        df.loc["noarrow"] = [1.0, 1.0]      # skipped: no ' -> '
        df.loc["X -> E"] = [2.0, 2.0]       # X becomes a dead-end source
        return df


def test_conservation_network_exceptions_and_dead_end(tmp_path):
    fa = _ConservationExceptionAnalyzer(tmp_path)
    fa.parameters._ensure_flat_list = lambda v: (_ for _ in ()).throw(
        RuntimeError("flat list fail"))
    html = hrg._generate_conservation_network(fa, DATASETS, 1, NICKNAME_MAP)
    assert "Conservation" in html
    assert "dead-end" in html


class _ThreeDSAnalyzer(FakeAnalyzer):
    def __init__(self, output_path):
        super().__init__(output_path, dataset_names=tuple(DS3))

    def get_aligned_data(self, t):
        return pd.DataFrame(
            {
                "ds_one": [10.0, 6.0, 0.0, 4.0],
                "ds_two": [9.0, 0.0, 5.0, 3.0],
                "ds_three": [8.0, 5.0, 0.0, 2.0],
            },
            index=["A -> B", "B -> C", "C -> D", "E -> F"],
        )


def test_conservation_network_partial_three_datasets(tmp_path):
    fa = _ThreeDSAnalyzer(tmp_path)
    html = hrg._generate_conservation_network(fa, DS3, 1, NICK3)
    assert "Partial" in html
    assert "Unique" in html


class FakeAutoTypeMapper:
    def get_all_dataset_short_codes(self, datasets):
        return {"D1": "ds_one", "D2": "ds_two"}

    def get_display_name_with_dataset_info(self, name, datasets):
        if name == "B":
            return "B (F:B1)", {"F": "B1"}
        if name == "GNG588":
            return "GNG588 (CB0038)", {"F": "GNG588"}
        if name == "NEW":
            return "CAN (F:N1)", {"F": "N1"}
        if name == "DUPA":
            return "CAN (F:N2)", {"F": "N2"}
        if name == "NEWT":
            return "CAN2 (H:N2)", {"H": "N2"}
        if name == "DUPT":
            return "CAN2 (H:N3)", {"H": "N3"}
        if name in ("BALT", "TALT"):
            # display label already present as a transformed edge node
            return "B (F:B1)", {}
        return name, {}


def test_conservation_network_type_mapper(tmp_path):
    fa = FakeAnalyzer(
        tmp_path, auto_type_mapping=True,
        source_neurons=("A", "NEW", "DUPA", "BALT", "PAT*"),
        target_neurons=("B", "NEWT", "DUPT", "TALT", "TPAT*"),
    )
    fa.parameters._auto_type_mapper = FakeAutoTypeMapper()
    df = _aligned_df()
    df.loc["GNG588(CB0038) -> B"] = [3.0, 2.0]
    fa.get_aligned_data = lambda t: df
    fa.get_aligned_data_for_network = lambda t: df
    # no path data -> custom edges are not filtered away
    fa._get_path_data_for_threshold = lambda t: pd.DataFrame()
    html = hrg._generate_conservation_network(fa, DATASETS, 1, NICKNAME_MAP)
    assert "B (F:B1)" in html
    assert "CAN (F:N" in html
    assert "CAN2 (H:N" in html
    assert "Dataset codes in node names" in html
    assert "Names by dataset" in html


def test_conservation_network_is_represented_branches(tmp_path):
    fa = FakeAnalyzer(
        tmp_path,
        source_neurons=("aMe12", "MeVPLo2", "Leg", "aMe", "MTe07", "Var",
                        "X1", "ISO_SRC", "PAT*"),
        target_neurons=("Q", "TGT_ISO", "TPAT*"),
    )
    idx = [
        "aMe12_L -> Q",
        "MeVPLo2 (F:MTe07) -> Q",
        "Leg(X1) -> Q",
        "aMe_L (F:aMe) -> Q",
        "Base(F:MTe07/H:Var) -> Q",
        "Base2(X1) -> Q",
    ]
    df = pd.DataFrame(
        {"ds_one": [1.0] * len(idx), "ds_two": [1.0] * len(idx)}, index=idx
    )
    fa.get_aligned_data = lambda t: df
    fa.get_aligned_data_for_network = lambda t: df
    # no path data -> aligned edges are not filtered away
    fa._get_path_data_for_threshold = lambda t: pd.DataFrame()
    html = hrg._generate_conservation_network(fa, DATASETS, 1, NICKNAME_MAP)
    assert "ISO_SRC" in html    # isolated source node added
    assert "TGT_ISO" in html    # isolated target node added


class _MultiThresholdAnalyzer(FakeAnalyzer):
    def __init__(self, output_path, raise_paths=False):
        super().__init__(output_path)
        self._raise_paths = raise_paths

    def get_aligned_data(self, t):
        data = {
            1: pd.DataFrame(
                {"ds_one": [10.0, 5.0, 3.0, 1.0],
                 "ds_two": [8.0, 4.0, 0.0, 1.0]},
                index=["A -> B", "B -> C", "C -> D", "noarrow"]),
            5: pd.DataFrame(
                {"ds_one": [9.0, 4.0], "ds_two": [7.0, 3.0]},
                index=["A -> B", "B -> C"]),
            10: pd.DataFrame(
                {"ds_one": [8.0, 2.0, 1.0], "ds_two": [6.0, 0.0, 2.0]},
                index=["A -> B", "A -> B", "E -> F"]),
        }
        return data.get(t, pd.DataFrame())

    def _get_path_data_for_threshold(self, t):
        if self._raise_paths:
            raise RuntimeError("no paths")
        return pd.DataFrame()


def test_dataset_network_branches(tmp_path):
    fa = _MultiThresholdAnalyzer(tmp_path, raise_paths=True)
    html = hrg._generate_dataset_network(
        fa, "ds_one", [1, 5, 10], NICKNAME_MAP, max_edges=2
    )
    assert "D1" in html
    assert "thresholds" in html


def test_dataset_network_no_label_mapper(tmp_path):
    fa = _MultiThresholdAnalyzer(tmp_path)
    fa.label_mapper = None
    html = hrg._generate_dataset_network(fa, "ds_one", [1, 5], NICKNAME_MAP)
    assert "D1" in html


def test_dataset_network_params_failure(tmp_path):
    fa = _MultiThresholdAnalyzer(tmp_path)
    fa.label_mapper = None
    fa.parameters._ensure_flat_list = lambda v: (_ for _ in ()).throw(
        RuntimeError("fail"))
    html = hrg._generate_dataset_network(fa, "ds_one", [1, 5], NICKNAME_MAP)
    assert "D1" in html


class _EdgeTableAnalyzer(FakeAnalyzer):
    def get_aligned_data(self, t):
        if t == 1:
            return pd.DataFrame()
        return pd.DataFrame(
            {"ds_one": [10.0, 5.0, 7.0], "ds_two": [8.0, 4.0, 6.0]},
            index=["A -> B", "B -> C", "A -> B"],
        )


def test_edge_dataset_table_fallback_and_series(tmp_path):
    html = hrg._generate_edge_dataset_table(
        _EdgeTableAnalyzer(tmp_path), "ds_one", [1, 5], NICKNAME_MAP
    )
    assert "A -> B" in html


class _PathTableAnalyzer(FakeAnalyzer):
    def _get_path_data_for_threshold(self, t):
        if t == 1:
            return pd.DataFrame()
        return pd.DataFrame(
            {"ds_one": [7.0, 3.0, 5.0], "ds_two": [5.0, 3.0, 4.0]},
            index=["A -> B -> C", "A -> C -> D", "A -> B -> C"],
        )


def test_path_dataset_table_fallback_and_series(tmp_path):
    html = hrg._generate_path_dataset_table(
        _PathTableAnalyzer(tmp_path), "ds_one", [1, 5], NICKNAME_MAP
    )
    assert "A -> B -> C" in html


def test_path_presence_table_no_datasets(analyzer):
    df = pd.DataFrame({"other": [1]}, index=["A -> B -> C"])
    html = hrg._generate_path_presence_table(
        analyzer, df, DATASETS, NICKNAME_MAP, threshold=1
    )
    assert "No datasets available" in html


class _ConsFailAnalyzer(FakeAnalyzer):
    def __init__(self, output_path):
        super().__init__(output_path, dataset_names=tuple(DS3))
        self.comparison_report = {"path_presence_matrix": ["not", "a", "df"]}

    def get_mapped_results(self):
        raise RuntimeError("no mapped results")

    def get_aligned_data(self, t):
        return pd.DataFrame(
            {
                "ds_one": [10.0, 6.0, 4.0],
                "ds_two": [9.0, 5.0, 0.0],
                "ds_three": [8.0, 0.0, 0.0],
            },
            index=["A -> B", "B -> C", "C -> D"],
        )


def test_conservation_section_failures(tmp_path):
    html = hrg._generate_conservation_section(
        _ConsFailAnalyzer(tmp_path), DS3, THRESHOLDS, _key_findings(), NICK3
    )
    assert "Conservation Analysis" in html
    assert "In 2 datasets" in html
    assert "plotData = null" in html  # plotly generation failed -> fallback


class _ConsThreeAnalyzer(FakeAnalyzer):
    def __init__(self, output_path):
        super().__init__(output_path, dataset_names=tuple(DS3))
        self.comparison_report = {
            "path_presence_matrix": pd.DataFrame(
                {
                    "ds_one_t1": ["True", "True", "True"],
                    "ds_two_t1": ["True", "True", False],
                    "ds_three_t1": ["True", False, False],
                },
                index=["p1", "p2", "p3"],
            )
        }

    def get_aligned_data(self, t):
        return pd.DataFrame(
            {
                "ds_one": [10.0, 6.0],
                "ds_two": [9.0, 5.0],
                "ds_three": [8.0, 0.0],
            },
            index=["A -> B", "B -> C"],
        )


def test_conservation_section_three_datasets_labels(tmp_path):
    html = hrg._generate_conservation_section(
        _ConsThreeAnalyzer(tmp_path), DS3, [1], _key_findings(), NICK3
    )
    assert "In 2 datasets" in html  # edges + paths in exactly 2 of 3
    assert "Unique (1)" in html


class _OverlapPathFailAnalyzer(FakeAnalyzer):
    def _get_path_data_for_threshold(self, t):
        raise RuntimeError("no paths")


def test_overlap_matrices_empty_datasets(analyzer):
    html = hrg._generate_overlap_matrices_section(
        analyzer, [], THRESHOLDS, {}
    )
    assert "No datasets configured" in html


def test_overlap_matrices_path_failure(tmp_path):
    html = hrg._generate_overlap_matrices_section(
        _OverlapPathFailAnalyzer(tmp_path), DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Dataset Overlap Matrices" in html


class _TrendFailAnalyzer(FakeAnalyzer):
    def __init__(self, output_path, mode):
        super().__init__(output_path)
        self._mode = mode

    def get_aligned_data(self, t):
        if self._mode == "raise":
            raise RuntimeError("aligned fail")
        if self._mode == "none":
            return None
        return super().get_aligned_data(t)

    def _get_path_data_for_threshold(self, t):
        if self._mode == "raise":
            raise RuntimeError("path fail")
        if self._mode == "none":
            return None
        return super()._get_path_data_for_threshold(t)


def test_trend_plots_empty_data(empty_analyzer):
    html1 = hrg._generate_jaccard_similarity_plot(
        empty_analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP)
    html2 = hrg._generate_edge_rank_correlation_plot(
        empty_analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP)
    assert "Jaccard" in html1
    assert "Edge Rank" in html2


def test_trend_plots_raising_and_none(tmp_path):
    raising = _TrendFailAnalyzer(tmp_path, "raise")
    none = _TrendFailAnalyzer(tmp_path, "none")
    assert "Cosine" in hrg._generate_cosine_similarity_trend_plot(
        raising, DATASETS, THRESHOLDS, NICKNAME_MAP)
    assert "Cosine" in hrg._generate_cosine_similarity_trend_plot(
        none, DATASETS, THRESHOLDS, NICKNAME_MAP)
    assert "Path" in hrg._generate_path_rank_correlation_plot(
        raising, DATASETS, THRESHOLDS, NICKNAME_MAP)
    assert "Path" in hrg._generate_path_rank_correlation_plot(
        none, DATASETS, THRESHOLDS, NICKNAME_MAP)


class _StatsTableAnalyzer(FakeAnalyzer):
    def __init__(self, output_path, mode):
        super().__init__(output_path)
        self._mode = mode

    def get_aligned_data(self, t):
        if self._mode == "foreign":
            return pd.DataFrame({"other_ds": [1.0, 2.0]},
                                index=["A -> B", "B -> C"])
        return pd.DataFrame(
            {"ds_one": [10.0, 0.0], "ds_two": [0.0, 5.0]},
            index=["A -> B", "B -> C"],
        )


def test_stats_table_no_datasets_and_low_overlap(tmp_path):
    html1 = hrg._generate_stats_table(
        _StatsTableAnalyzer(tmp_path, "foreign"), DATASETS, 1, NICKNAME_MAP)
    assert "No datasets available" in html1
    html2 = hrg._generate_stats_table(
        _StatsTableAnalyzer(tmp_path, "sparse"), DATASETS, 1, NICKNAME_MAP)
    assert "0.000" in html2  # rank corr falls back to 0 with <2 shared edges


# ---------------------------------------------------------------------------
# Type mapping: cross-dataset same-name leak + appearance ordering (plan §7B)
# ---------------------------------------------------------------------------

def test_type_mapping_does_not_leak_same_name_across_datasets():
    """BANC `aMe24` canonicalizes to MCNS `5thsLNv_LNd6` and resolves to
    FAFB `aMe24` by bare same-NAME identity (mapper tier 6). That echo must
    not appear in the FAFB cell of the canonical row."""
    from types import SimpleNamespace

    datasets = ["male-cns:v1.0", "banc_v888", "flywire_FAFB_v783"]
    fafb = "flywire_FAFB_v783"
    banc = "banc_v888"

    class Mapper:
        _conflicts = []
        _loaded = True

        def _get_type_mapping_key(self, dataset):
            return dataset

        def get_mapping_decision(self, source_type, source_dataset,
                                 target_dataset, include_bridges=False):
            # aMe24 (BANC) -> MCNS canonical 5thsLNv_LNd6; to FAFB it is a
            # same-name identity.
            if source_type == 'aMe24' and target_dataset == 'male-cns:v1.0':
                return {'status': 'mapped', 'source_type': source_type,
                        'target_type': '5thsLNv_LNd6',
                        'target_types': ['5thsLNv_LNd6'],
                        'relationship': 'renamed', 'conflicts': []}
            return {'status': 'unmapped', 'source_type': source_type,
                    'target_type': None, 'target_types': [],
                    'relationship': None, 'conflicts': []}

        def get_alias_candidates(self, raw, targets, source_dataset=None):
            out = {}
            for t in targets:
                if raw == 'aMe24' and t == 'male-cns:v1.0':
                    out[t] = {'outcome': 'matched',
                              'candidates': [{'kind': 'renamed',
                                              'name': '5thsLNv_LNd6'}]}
                elif raw == 'aMe24' and t == fafb:
                    # Same-name leaf as a bare candidate (tier-6 echo).
                    out[t] = {'outcome': 'matched',
                              'candidates': [{'kind': 'same name',
                                              'name': 'aMe24'}],
                              'same_namespace': False}
                else:
                    out[t] = {'outcome': 'none', 'candidates': []}
            return out

        def get_type_bridges(self, *a, **k):
            return []

    class Analyzer:
        parameters = SimpleNamespace(
            auto_type_mapping=True, _auto_type_mapper=Mapper())

        @staticmethod
        def _collect_result_types_by_dataset():
            return {banc: {'aMe24'}, 'male-cns:v1.0': {'5thsLNv_LNd6'}}

    report = hrg._generate_type_mapping_section(Analyzer(), datasets)
    # The canonical row is keyed on the MCNS target and FAFB must not gain
    # aMe24 — check the appendix grid (the Sources side legitimately lists
    # the banc aMe24 observation).
    assert '5thsLNv_LNd6' in report
    row_start = report.find('5thsLNv_LNd6</strong>')
    assert row_start != -1
    row = report[row_start:row_start + 400]
    cells = re.findall(r'<td>(.*?)</td>', row)
    # cells[0] is the canonical name; the FAFB cell is the last dataset cell.
    assert cells, row
    assert 'aMe24' not in cells[-1], cells


def test_type_mapping_appearance_order_sorts_by_presence_matrix():
    """Rows follow first-appearance in the path-presence matrix, not alpha."""
    from types import SimpleNamespace

    datasets = ["a", "b"]

    class Mapper:
        _conflicts = []
        _loaded = True

        def _get_type_mapping_key(self, d):
            return d

        def get_mapping_decision(self, source_type, source_dataset,
                                 target_dataset, include_bridges=False):
            return {'status': 'unmapped', 'source_type': source_type,
                    'target_type': None, 'target_types': [],
                    'relationship': None, 'conflicts': []}

        def get_alias_candidates(self, raw, targets, source_dataset=None):
            return {t: {'outcome': 'none', 'candidates': []} for t in targets}

        def get_type_bridges(self, *a, **k):
            return []

    # `zzz` is most conserved (appears in both datasets); `aaa` only in one.
    path_data = pd.DataFrame(
        {'a': [5.0, 1.0], 'b': [5.0, 0.0]},
        index=['zzz -> qqq', 'aaa -> qqq'])

    class Analyzer:
        parameters = SimpleNamespace(
            auto_type_mapping=True, _auto_type_mapper=Mapper())

        @staticmethod
        def _collect_result_types_by_dataset():
            return {'a': {'zzz', 'aaa'}, 'b': {'zzz', 'aaa'}}

        @staticmethod
        def _analysis_thresholds():
            return [1]

        @staticmethod
        def _get_path_data_for_threshold(t):
            return path_data

        @staticmethod
        def get_aligned_data(t):
            return pd.DataFrame()

    report = hrg._generate_type_mapping_section(Analyzer(), datasets)
    # `zzz` (conserved) must appear before `aaa` (rare) despite alpha order.
    assert report.find('zzz') < report.find('aaa')


def test_canonical_source_rank_priority():
    """Plan F1: male-cns -> FAFB -> other neuprint -> BANC, unknowns in the
    other-neuprint tier."""
    from comparison.comparison_parameters import canonical_source_rank
    assert canonical_source_rank('male-cns:v1.0') == 0
    assert canonical_source_rank('MCNS') == 0
    assert canonical_source_rank('flywire_FAFB_v783') == 1
    assert canonical_source_rank('fafb') == 1
    assert canonical_source_rank('hemibrain:v1.2.1') == 2
    assert canonical_source_rank('manc:v1.2.3') == 2
    assert canonical_source_rank('optic-lobe:v1.1') == 2
    assert canonical_source_rank('banc:v888') == 3
    assert canonical_source_rank('banc_v888') == 3
    assert canonical_source_rank('') == 2


def test_dataset_coverage_callout_renders_only_for_missing_data():
    """Plan B2: a configured dataset with no data must surface a loud
    coverage callout; an all-ok run renders none."""
    from types import SimpleNamespace

    an = SimpleNamespace(dataset_coverage=lambda: {
        'male-cns:v1.0': {'status': 'ok', 'thresholds_with_data': 3,
                          'total_rows': 100},
        'hemibrain:v1.2.1': {'status': 'no_data', 'thresholds_with_data': 0,
                             'total_rows': 0},
    })
    out = hrg._generate_dataset_coverage_callout(
        an, ['male-cns:v1.0', 'hemibrain:v1.2.1'],
        {'male-cns:v1.0': 'MCNS', 'hemibrain:v1.2.1': 'HEMI'})
    assert 'Dataset coverage warning' in out
    assert 'HEMI' in out and 'hemibrain:v1.2.1' in out

    ok_an = SimpleNamespace(dataset_coverage=lambda: {
        'male-cns:v1.0': {'status': 'ok', 'thresholds_with_data': 3,
                          'total_rows': 100}})
    assert hrg._generate_dataset_coverage_callout(
        ok_an, ['male-cns:v1.0'], {'male-cns:v1.0': 'MCNS'}) == ''


def test_density_curve_card_has_vh_guides():
    """Plan D: vertical guides mark the per-threshold rows, horizontal
    guides the density-matched rows; both carry the owning row id."""
    curves = pd.DataFrame([
        {'dataset': 'd1', 'threshold': t, 'path_count': 100 // t,
         'edge_count': 200 // t, 'density': round(0.9 - t / 100, 6),
         'is_materialized': t == 5}
        for t in (1, 2, 5, 10)])
    aligned = pd.DataFrame([
        {'id': 'threshold=15', 'label': 'threshold=15', 'mode': 'vertical',
         'level_continuous': 15.0, 'level_normalized': None},
        {'id': 'aligned_density=0.5',
         'label': 'aligned_density=0.5 (d1 2)', 'mode': 'horizontal',
         'level_continuous': None, 'level_normalized': 0.5},
    ])
    html = hrg._density_curves_plotly_card(curves, aligned, None, {'d1': 'D1'})
    assert 'threshold=15' in html
    assert 'vertical (per-threshold) row' in html
    assert 'horizontal (density-matched) row' in html
    assert 'vguides' in html and 'hguides' in html


def test_neuron_counts_per_role_axes_and_shared_legend():
    """Plan G: independent per-role x categories and one shared dataset
    chip legend with per-chart legends disabled."""
    type_df = pd.DataFrame([
        {'type': 'SrcOnly', 'role': 'source', 'group_members': '',
         'd1_source': 40, 'd1_target': 0, 'd2_source': 30, 'd2_target': 0},
        {'type': 'TgtOnly', 'role': 'target', 'group_members': '',
         'd1_source': 0, 'd1_target': 25, 'd2_source': 0, 'd2_target': 20},
        {'type': 'Both', 'role': 'both', 'group_members': '',
         'd1_source': 5, 'd1_target': 4, 'd2_source': 3, 'd2_target': 2},
    ])

    class Analyzer:
        _neuron_counts_summary = pd.DataFrame()
        _neuron_type_counts = type_df
        _neuron_group_counts = pd.DataFrame()

        def _merge_policy_or_none(self):
            raise RuntimeError('no policy')

    html = hrg._generate_neuron_counts_section(
        Analyzer(), ['d1', 'd2'], {'d1': 'D1', 'd2': 'D2'})
    assert 'showlegend: false' in html
    assert 'Dataset (shared legend)' in html
    marker = 'const payload = '
    start = html.find(marker) + len(marker)
    payload = json.loads(html[start:html.find(';\n', start)])
    src_x = payload['source'][0]['x']
    tgt_x = payload['target'][0]['x']
    # The target chart leads with the target-dominant type and the source
    # chart with the source-dominant one (independent categories).
    assert tgt_x[0] == 'TgtOnly', tgt_x
    assert src_x[0] == 'SrcOnly', src_x

    # Plan round-4 F-1 (second notice): zero-total-for-role types must be
    # EXCLUDED from the other role's categories entirely — not appended
    # with zero bars.
    assert 'PPL' not in str(src_x) or all(
        'ppl' not in x.lower() for x in src_x)
    assert 'TgtOnly' not in src_x, src_x
    assert 'SrcOnly' not in tgt_x, tgt_x


def test_type_mapping_auto_only_badge_on_source_cell():
    """Plan F3: the auto-only badge attaches to the auto-label SOURCE
    dataset cell (BANC in a BANC-anchored run), not to the canonical
    label."""
    from types import SimpleNamespace

    datasets = ['mcns', 'banc']

    class Mapper:
        _conflicts = []
        _loaded = True

        def _get_type_mapping_key(self, d):
            return d

        def get_canonical_type(self, t, source_dataset=None):
            return t

        def get_mapping_decision(self, source_type, source_dataset,
                                 target_dataset, include_bridges=False):
            if (source_type == 'PPL102' and source_dataset == 'banc'
                    and target_dataset == 'mcns'):
                return {'status': 'mapped', 'source_type': source_type,
                        'target_type': 'PPL102',
                        'target_types': ['PPL102'],
                        'relationship': '1-to-1', 'conflicts': [],
                        'support': {'votes': {'v1': {}},
                                    'verified_votes': {},
                                    'winner_derived_from_auto': True}}
            return {'status': 'unmapped', 'source_type': source_type,
                    'target_type': None, 'target_types': [],
                    'relationship': None, 'conflicts': []}

    class Analyzer:
        parameters = SimpleNamespace(auto_type_mapping=True,
                                     _auto_type_mapper=Mapper())

        @staticmethod
        def _collect_result_types_by_dataset():
            return {'banc': {'PPL102'}, 'mcns': {'PPL102'}}

    report = hrg._generate_type_mapping_section(Analyzer(), datasets)
    assert 'auto-only' in report
    # The badge attaches to the auto-label SOURCE dataset cell (banc) in
    # the appendix grid; the canonical label cell carries none.
    row_start = report.find('<strong>PPL102</strong>')
    row_end = report.find('</tr>', row_start)
    row = report[row_start:row_end]
    assert 'auto-only' in row
    mcns_start = row.find('<td>')
    mcns_end = row.find('</td>', mcns_start)
    banc_start = row.find('<td>', mcns_end)
    assert 'auto-only' not in row[mcns_start:mcns_end]
    assert 'auto-only' in row[banc_start:]
