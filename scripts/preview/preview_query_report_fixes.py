"""One-off visual preview of the 2026-09-15 report fixes.

Builds a query-scoped cross-dataset report (auto-mode style: threshold= and
aligned_density= rows) with a merge policy carrying a split group, density
curve frames, and a non-requested applied threshold, then writes the HTML
for headless rendering. Exercises:
  1. type-mapping Source/Target columns + colored branch names + collapsed
     topology
  2. merged provenance (Key Findings req->app cells + collapsed details)
  3. similarity section without the pair-metrics table
  4. conservation donut grid + deduplicated headers
  5. interactive density curves (plotly) instead of the static PNG
"""
import os
import sys
import json
import tempfile

import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
for entry in (PROJECT, os.path.join(PROJECT, 'src')):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from comparison.html_report_generator import generate_html_report  # noqa: E402
from comparison.point_context import ComparisonPoint  # noqa: E402
from comparison.merge_policy import MergeGroup, MergePolicy  # noqa: E402

DATASETS = ["male-cns:v1.0", "flywire_FAFB_v783", "banc_v888"]
QUERIES = [
    {"id": "threshold=50", "label": "threshold=50",
     "thresholds": {"male-cns:v1.0": 50, "flywire_FAFB_v783": 50,
                    "banc_v888": 50}, "row_mode": "vertical"},
    {"id": "threshold=100", "label": "threshold=100",
     "thresholds": {"male-cns:v1.0": 100, "flywire_FAFB_v783": 100,
                    "banc_v888": 100}, "row_mode": "vertical"},
    {"id": "aligned_density=0.2464",
     "label": "aligned_density=0.2464 (male-cns 14, FAFB 13, BANC 6)",
     "thresholds": {"male-cns:v1.0": 14, "flywire_FAFB_v783": 13,
                    "banc_v888": 6}, "row_mode": "horizontal"},
    {"id": "aligned_density=0.5125",
     "label": "aligned_density=0.5125 (male-cns 10, FAFB 9, BANC 5)",
     "thresholds": {"male-cns:v1.0": 10, "flywire_FAFB_v783": 9,
                    "banc_v888": 5}, "row_mode": "horizontal"},
]

# --- merge policy: one same-name group + one split group with branches ----
split = MergeGroup(
    group_id='g002', label='5thsLNv_LNd6', kind='merged_split',
    anchor=('male-cns:v1.0', '5thsLNv_LNd6'),
    members={'male-cns:v1.0': ['5thsLNv_LNd6'],
             'flywire_FAFB_v783': ['5th-LNv', 'LNd_CRY+_ITP+'],
             'banc_v888': ['LNd_a', 'aMe24', 's-LNv_a']},
    branches=[
        MergeGroup(group_id='g002b1', label='aMe24', kind='leaf',
                   anchor=('banc_v888', 'aMe24'),
                   members={'banc_v888': ['aMe24']}),
        MergeGroup(group_id='g002b2', label='s-LNv_a', kind='leaf',
                   anchor=('banc_v888', 's-LNv_a'),
                   members={'banc_v888': ['s-LNv_a'],
                            'flywire_FAFB_v783': ['5th-LNv']}),
        MergeGroup(group_id='g002b3', label='LNd_CRY+_ITP+', kind='leaf',
                   anchor=('flywire_FAFB_v783', 'LNd_CRY+_ITP+'),
                   members={'flywire_FAFB_v783': ['LNd_CRY+_ITP+'],
                            'banc_v888': ['LNd_a']}),
    ],
)
same_name = MergeGroup(
    group_id='g001', label='SMP368', kind='same_name',
    anchor=('male-cns:v1.0', 'SMP368'),
    members={'male-cns:v1.0': ['SMP368'], 'flywire_FAFB_v783': ['SMP368'],
             'banc_v888': ['SMP368']},
)
key_map = {}
for ds, names in split.members.items():
    for nm in names:
        key_map[(ds, nm)] = '5thsLNv_LNd6'
for ds, names in same_name.members.items():
    for nm in names:
        key_map[(ds, nm)] = 'SMP368'
for br in split.branches:
    for ds, names in br.members.items():
        for nm in names:
            key_map[(ds, nm)] = '5thsLNv_LNd6'
POLICY = MergePolicy(
    anchor_ds='male-cns:v1.0', anchor_case='span',
    groups=[same_name, split], key_map=key_map,
    warnings=['type granularity is ambiguous for intermediate types '
              '(1-to-N branches keep every leaf)'])


class Parameters:
    full_output_path = OUT = os.path.join(
        tempfile.mkdtemp(prefix='drocat_report_preview_'))
    comparison_mode = 'path'
    path_mode = 'all'
    max_interlayer = 2
    separate_hemispheres = False
    symmetry_analysis = False
    auto_type_mapping = True
    threshold_auto = True
    source_neurons = []
    target_neurons = []

    @staticmethod
    def get_dataset_nicknames():
        return ['MCNS', 'FAFB', 'BANC']

    @staticmethod
    def _sanitize_name(ds):
        return ds.split(':')[0]


class Mapper:
    _conflicts = []
    _loaded = True

    def get_canonical_type(self, type_name, source_dataset=None):
        return type_name

    def get_mapping_decision(self, source_type, source_dataset,
                             target_dataset, include_bridges=False):
        return {'status': 'unmapped', 'source_type': source_type,
                'target_type': None, 'target_types': [],
                'relationship': None, 'conflicts': []}


Parameters._auto_type_mapper = Mapper()


def make_aligned():
    return pd.DataFrame(
        {ds: [3, 2, 2, 0] for ds in DATASETS},
        index=["A -> B", "A -> C", "B -> C", "A -> D"])


class Analyzer:
    parameters = Parameters()
    label_mapper = None
    _merge_policy = POLICY

    @classmethod
    def _merge_policy_or_none(cls):
        return cls._merge_policy

    comparison_report = {"threshold_similarities": pd.DataFrame([
        {
            "query_id": q["id"],
            "dataset_1": DATASETS[0],
            "dataset_2": DATASETS[1],
            "jaccard_similarity": 0.5,
            "ruzicka_similarity": 0.4,
            "pearson_correlation": 0.3,
            "edge_rank_correlation": 0.6,
            "cosine_similarity": 0.7,
            "spearman_rank_correlation": 0.2,
            "common_edges": 1,
        },
    ] for q in QUERIES)}
    _similarity_cache = {}
    raw_results = {
        ds: {t: pd.DataFrame({
            "type_pre": ["A"], "type_post": ["B"], "weight": [t]})
            for t in (6, 5, 14, 13, 10, 9, 100, 50)}
        for ds in DATASETS}
    _path_run_meta = {
        (ds, t): {
            "applied_threshold": t,
            "applied_threshold_source": "requested",
            "paths_complete": True,
            "strongest_first_budget": 100,
            "strongest_first_tau": None,
            "edge_budget": 1000,
            "edge_budget_applied": False,
        }
        for ds in DATASETS
        for t in (50, 100, 14, 13, 6, 5, 10, 9)}
    _neuron_counts_summary = pd.DataFrame()
    _neuron_type_counts = pd.DataFrame([
        {'type': 'aMe5', 'role': 'source', 'group_members': '',
         'male_cns_v1_0_source': 40, 'flywire_FAFB_v783_source': 30,
         'banc_v888_source': 20, 'hemibrain_v1_2_1_source': 10,
         'male_cns_v1_0_target': 0, 'flywire_FAFB_v783_target': 0,
         'banc_v888_target': 0, 'hemibrain_v1_2_1_target': 0},
        {'type': 'PPL101', 'role': 'target', 'group_members': '',
         'male_cns_v1_0_source': 0, 'flywire_FAFB_v783_source': 0,
         'banc_v888_source': 0, 'hemibrain_v1_2_1_source': 0,
         'male_cns_v1_0_target': 8, 'flywire_FAFB_v783_target': 6,
         'banc_v888_target': 4, 'hemibrain_v1_2_1_target': 2},
    ])
    _neuron_group_counts = pd.DataFrame()

    # Density capture (issue 5): a plausible curve per dataset.
    _density_rows = []
    for ds in DATASETS:
        for t in (2, 5, 9, 13, 25, 50, 100):
            _density_rows.append({
                "dataset": ds, "threshold": t,
                "path_count": int(4000 / t), "edge_count": int(9000 / t),
                "density": round(min(0.9, 30.0 / t), 6),
                "normalizer": "per_node", "basis": "typed",
                "w_start": 1, "w_star_measured": 40,
                "path_complete_from": 1,
                "is_materialized": t in (50, 100, 14, 13, 10, 9, 6, 5)})
    _density_curves_df = pd.DataFrame(_density_rows)
    _density_windows_df = pd.DataFrame([
        {"dataset": ds, "applied": 14, "w_start": 1, "w_star_stored": 40,
         "w_star_measured": 40, "w_star_mismatch": False,
         "budget_bitten": False, "paths_complete": True,
         "path_complete_from": 1, "n_paths": 120, "n_edges": 300,
         "n_edges_active_basis": 300, "n_nodes": 42, "n_nodes_typed": 40,
         "n_nodes_untyped": 2, "n_nodes_debris": 0}
        for ds in DATASETS])
    _density_alignment_df = pd.DataFrame([
        {"mode": "vertical", "level_continuous": 50.0,
         "level_normalized": None, "degenerate": False, "clamped": False,
         "max_abs_deviation": None, **{ds: 50 for ds in DATASETS}},
        {"mode": "horizontal", "level_continuous": None,
         "level_normalized": 0.2464, "degenerate": False, "clamped": False,
         "max_abs_deviation": 0.01, **{
             ds: t for ds, t in zip(DATASETS, (14, 13, 6))}},
        {"mode": "horizontal", "level_continuous": None,
         "level_normalized": 0.5125, "degenerate": False, "clamped": False,
         "max_abs_deviation": 0.02, **{
             ds: t for ds, t in zip(DATASETS, (10, 9, 5))}},
    ])

    @staticmethod
    def get_threshold_queries():
        return QUERIES

    @staticmethod
    def get_aligned_data_for_query(query):
        return make_aligned()

    @staticmethod
    def get_aligned_data_for_network(query):
        return make_aligned()

    @staticmethod
    def _get_path_data_for_query(query):
        return pd.DataFrame(
            {ds: [3] for ds in DATASETS},
            index=["A -> B -> C"])

    @staticmethod
    def _get_path_hop_weights_for_threshold(threshold):
        if isinstance(threshold, dict):
            return {"A -> B -> C": {ds: [10, 7] for ds in DATASETS}}
        return {}

    @staticmethod
    def _path_provenance_row(dataset, threshold):
        aliased = threshold == 6
        return {
            "dataset": dataset,
            "threshold": threshold,
            "requested_threshold": threshold,
            "applied_threshold": 5 if aliased else threshold,
            "applied_threshold_source":
                "strongest_first_budget" if aliased else "requested",
            "strongest_first_budget": 100,
            "strongest_first_budget_bitten": aliased,
            "tau": None,
            "edge_budget": 1000,
            "edge_budget_applied": False,
            "edge_weight_floor": None,
            "edge_budget_landing": None,
            "strongest_dropped_bottleneck": None,
            "strongest_retained_bottleneck": None,
            "paths_complete": True,
        }

    @staticmethod
    def _collect_result_types_by_dataset():
        return {
            "male-cns:v1.0": {"SMP368", "5thsLNv_LNd6", "MeVPLo2",
                              "VS", "CB1011"},
            "flywire_FAFB_v783": {"SMP368", "5th-LNv", "LNd_CRY+_ITP+",
                                  "MTe07"},
            "banc_v888": {"SMP368", "aMe24", "s-LNv_a", "LNd_a", "CB2399",
                          "CB1011"},
        }

    @staticmethod
    def _analysis_thresholds():
        return [50, 100, 14, 13, 10, 9]

    @staticmethod
    def get_aligned_data(threshold):
        return make_aligned()

    @staticmethod
    def _get_path_data_for_threshold(threshold):
        return pd.DataFrame(
            {ds: [3] for ds in DATASETS},
            index=["A -> B -> C"])


def main():
    out_dir = Parameters.full_output_path
    for sub in ("conserved_paths", "conserved_reciprocal_graph",
                "comparison_results", "similarity_matrices",
                "comparison_visualizations"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    points = [
        ComparisonPoint(q["id"], q["label"], "combinations", q["thresholds"],
                        i)
        for i, q in enumerate(QUERIES, start=1)
    ]
    report = generate_html_report(
        analyzer=Analyzer(),
        dataset_names=DATASETS,
        thresholds=[],
        mode_specific_note="",
        path_count_data=[],
        key_findings_per_threshold={},
        comparison_points=points,
    )
    out = os.path.join(out_dir, 'comparison_report.html')
    with open(out, 'w') as fh:
        fh.write(report)
    print('WROTE', out, len(report))
    checks = {
        'no pair metrics table': 'Pair metrics' not in report,
        'no standalone provenance section':
            'id="threshold-provenance" class="section"' not in report,
        'collapsed provenance details present':
            '<details id="threshold-provenance"' in report,
        'req->app header present': 'req&rarr;app' in report,
        'aliased req->app cell present':
            '50&rarr;5 *' in report or '6&rarr;5 *' in report,
        'no banner at top of query report':
            report.index('Applied minimal thresholds')
            > report.index('id="summary"'),
        'donut grid present': 'repeat(auto-fill, minmax(380px, 1fr))'
            in report,
        'dedup conservation header':
            'Conservation at aligned_density=0.2464 (male-cns 14'
            in report,
        'no duplicated conservation header':
            'aligned_density=0.2464 — aligned_density=0.2464' not in report,
        'similarity heading deduped':
            'threshold=50 — threshold=50' not in report,
        # Phase F: three tables + priority source + no leak
        'three type-mapping tables':
            '<h4>Sources</h4>' in report and '<h4>Targets</h4>' in report
            and '<h4>Intermediates</h4>' in report,
        'source column is priority-labeled':
            '<th>Source (priority)</th>' in report,
        'type mapping targets':
            '<th>Target in MCNS</th>' in report,
        'priority source wins (mcns over banc anchor)':
            '<strong>male-cns:v1.0: 5thsLNv_LNd6</strong>' in report,
        'no CL317 leak in this fixture': 'CL317' not in report,
        'branch colors present': 'font-weight:600' in report,
        'collapsed topology': 'Resolution topology\n'
            in report or 'Resolution topology (query-anchored '
            'merge)</summary>' in report,
        # Phase D: V/H guides
        'density plotly card': 'densityCurves_density' in report,
        'V guide labels present': 'threshold=50' in report
            and 'vertical (per-threshold) row' in report,
        'H guide labels present':
            'horizontal (density-matched) row' in report,
        'no static density png embed':
            'density_alignment_threshold_curves.png' not in report,
        # Phase B: coverage callout absent when all datasets have data
        'no coverage callout in healthy fixture':
            'Dataset coverage warning' not in report,
        # Phase G: per-role axes + shared chip legend
        'chip legend present': 'Dataset (shared legend)' in report,
        'per-chart legends disabled': 'showlegend: false' in report,
    }
    ok = True
    for name, passed in checks.items():
        print(('PASS ' if passed else 'FAIL ') + name)
        ok = ok and passed
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
