"""The cross-dataset comparison is backended by FindAllPath: ComparisonParameters
must carry the bodyId pan-graph edge limit and the visualization-only edge
limit, while keeping the separate report-row cap explicit."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison import ComparisonParameters  # noqa: E402


def test_comparison_parameters_edge_limit_defaults():
    """The cross-dataset defaults match FindAllPath: the bodyId edge limit
    is mode-resolved (None = per-mode default: 1M for FindAllPath, 0 for
    FindShortestPath) and the Visualization Edge Limit is 500."""
    p = ComparisonParameters(
        datasets=['hemibrain:v1.2.1', 'male-cns:v1.0'])
    assert p.graph_edge_limit_bodyid is None
    assert p.edgeN_limit == 500


def test_symmetry_analysis_gated_by_separate_hemispheres():
    """symmetry_analysis is forced OFF when separate_hemispheres=False
    (it would otherwise run and produce meaningless summaries) and
auto-enabled when separate_hemispheres=True."""
    p = ComparisonParameters(
        datasets=['hemibrain:v1.2.1', 'male-cns:v1.0'],
        separate_hemispheres=False, symmetry_analysis=True,
    )
    assert p.symmetry_analysis is False
    p2 = ComparisonParameters(
        datasets=['hemibrain:v1.2.1', 'male-cns:v1.0'],
        separate_hemispheres=True, symmetry_analysis=False,
    )
    assert p2.symmetry_analysis is True


def test_comparison_parameters_edge_limits_survive_dict_roundtrip():
    """Customized limits survive to_dict -> from_dict."""
    p = ComparisonParameters(
        datasets=['hemibrain:v1.2.1', 'male-cns:v1.0'],
        top_edges=250, graph_edge_limit_bodyid=500000, edgeN_limit=1500,
    )
    d = p.to_dict()
    assert d["top_edges"] == 250
    assert d["graph_edge_limit_bodyid"] == 500000
    assert d["edgeN_limit"] == 1500
    p2 = ComparisonParameters.from_dict(d)
    assert p2.top_edges == 250
    assert p2.graph_edge_limit_bodyid == 500000
    assert p2.edgeN_limit == 1500


# =============================================================================
# path_mode ('all' | 'shortest') — FindShortestPath integration
# =============================================================================

def test_path_mode_default_is_all():
    """Default stays 'all' (FindAllPath) so existing comparisons are
    byte-compatible."""
    p = ComparisonParameters(
        datasets=['hemibrain:v1.2.1', 'male-cns:v1.2.1'])
    assert p.path_mode == 'all'


def test_path_mode_invalid_value_raises():
    import pytest
    with pytest.raises(ValueError):
        ComparisonParameters(
            datasets=['hemibrain:v1.2.1', 'male-cns:v1.2.1'],
            path_mode='direct')


def test_path_mode_survives_dict_roundtrip_and_legacy_default():
    p = ComparisonParameters(
        datasets=['hemibrain:v1.2.1', 'male-cns:v1.2.1'],
        path_mode='shortest')
    d = p.to_dict()
    assert d['path_mode'] == 'shortest'
    p2 = ComparisonParameters.from_dict(d)
    assert p2.path_mode == 'shortest'
    # Legacy saved parameters without the key fall back to 'all'.
    d_legacy = p.to_dict()
    del d_legacy['path_mode']
    assert ComparisonParameters.from_dict(d_legacy).path_mode == 'all'


def test_run_path_analysis_selects_find_tool_by_path_mode(monkeypatch):
    """run_path_analysis must call FindShortestPath in shortest mode and
    FindAllPath in all mode, with identical constructor wiring otherwise."""
    import coana
    from comparison import ComparisonAnalyzer

    calls = {'init': [], 'method': None}

    class FakeFNC:
        def __init__(self, **kwargs):
            calls['init'].append(kwargs)

        def InitializeNeuronInfo(self):
            pass

        def FindAllPath(self, **kwargs):
            calls['method'] = 'FindAllPath'

        def FindShortestPath(self, **kwargs):
            calls['method'] = 'FindShortestPath'

        allpath_folder = ''

    monkeypatch.setattr(coana, 'FindNeuronConnection', FakeFNC)

    def _run(path_mode):
        calls['method'] = None
        params = ComparisonParameters(
            datasets=['hemibrain:v1.2.1'],
            source_neurons=['aMe12'], target_neurons=['PPL101'],
            thresholds=[5], max_interlayer=0, path_mode=path_mode,
            output_folder="",  # no filesystem scaffold needed for wiring check
            verbose=False,
        )
        analyzer = ComparisonAnalyzer(params, verbose=False)
        analyzer._get_dataset_config = lambda name: None
        analyzer.parameters.get_source_neurons_for_dataset = lambda name: ['aMe12']
        analyzer.parameters.get_target_neurons_for_dataset = lambda name: ['PPL101']
        analyzer.parameters.get_dataset_output_path = lambda name, thr: '/tmp/fake_out'
        return analyzer.run_path_analysis('hemibrain:v1.2.1', 5,
                                          verbose_mode='silent')

    _run('shortest')
    assert calls['method'] == 'FindShortestPath'
    shortest_kwargs = calls['init'][0]

    _run('all')
    assert calls['method'] == 'FindAllPath'
    all_kwargs = calls['init'][1]

    # Same constructor wiring in both modes (path mode is handled inside
    # the FNC method call, not by different constructor args). The label
    # mapper is a fresh instance per analyzer run, so compare it by state.
    lm_shortest = shortest_kwargs.pop('label_mapper')
    lm_all = all_kwargs.pop('label_mapper')
    assert shortest_kwargs == all_kwargs
    assert (lm_shortest is None) == (lm_all is None)


def test_single_dataset_pipeline_runs_each_threshold(monkeypatch):
    """One dataset still runs the complete threshold iteration pipeline."""
    import pandas as pd
    from comparison import ComparisonAnalyzer

    params = ComparisonParameters(
        datasets=['hemibrain:v1.2.1'],
        source_neurons=['aMe12'],
        target_neurons=['PPL101'],
        thresholds=[3, 7],
        output_folder="",
        auto_type_mapping=False,
        verbose=False,
    )
    analyzer = ComparisonAnalyzer(params, verbose=False)
    # legacy per-threshold routing under test (Feature F replay would
    # otherwise route the whole list through FindAllPathMultiThreshold)
    analyzer.parameters.replay_paths = False
    calls = []

    def fake_run_path_analysis(dataset, threshold, verbose_mode="simple"):
        calls.append((dataset, threshold, verbose_mode))
        return pd.DataFrame()

    monkeypatch.setattr(analyzer, "run_path_analysis", fake_run_path_analysis)
    results = analyzer.run_all_analyses(skip_existing=False)
    summary = analyzer.run_comparison_analysis()

    assert calls == [
        ("hemibrain:v1.2.1", 3, "simple"),
        ("hemibrain:v1.2.1", 7, "silent"),
    ]
    assert sorted(results["hemibrain:v1.2.1"]) == [3, 7]
    assert summary["datasets"] == ["hemibrain:v1.2.1"]
    assert summary["thresholds"] == [3, 7]


def test_aggregate_and_find_paths_shortest_flags_only_shortest_route():
    """In shortest mode an edge is valid only when it lies on a per-pair
    minimum-hop path; 'all' mode keeps every on-path edge. max_layers=None
    (unlimited) must also work."""
    import pandas as pd
    from comparison import ComparisonAnalyzer

    params = ComparisonParameters(
        datasets=['hemibrain:v1.2.1'],
        source_neurons=['TS'], target_neurons=['TT'],
        thresholds=[1], output_folder="", verbose=False,
    )
    analyzer = ComparisonAnalyzer(params, verbose=False)

    # One bodyId per type: S->A->T (2 hops) plus the long route S->B->C->T.
    bodyid_df = pd.DataFrame({
        'bodyId_pre': ['S', 'A', 'S', 'B', 'C'],
        'bodyId_post': ['A', 'T', 'B', 'C', 'T'],
        'weight': [10, 10, 10, 10, 10],
    })
    label_map = {'S': 'TS', 'A': 'TA', 'T': 'TT', 'B': 'TB', 'C': 'TC'}

    res_shortest = analyzer._aggregate_and_find_paths(
        bodyid_df, label_map, {'TS'}, {'TT'},
        max_layers=None, dataset_name='hemibrain:v1.2.1', threshold=1,
        path_mode='shortest')
    flags = {(r['type_pre'], r['type_post']): bool(r['has_valid_path'])
             for _, r in res_shortest.iterrows()}
    assert flags[('TS', 'TA')] is True
    assert flags[('TA', 'TT')] is True
    # The long route carries no shortest path -> its edges are invalid.
    assert flags[('TS', 'TB')] is False
    assert flags[('TB', 'TC')] is False
    assert flags[('TC', 'TT')] is False

    res_all = analyzer._aggregate_and_find_paths(
        bodyid_df, label_map, {'TS'}, {'TT'},
        max_layers=4, dataset_name='hemibrain:v1.2.1', threshold=1,
        path_mode='all')
    flags_all = {(r['type_pre'], r['type_post']): bool(r['has_valid_path'])
                 for _, r in res_all.iterrows()}
    assert all(flags_all.values()), flags_all


# ===========================================================================
# Extended coverage tests (appended)
# ===========================================================================

import os  # noqa: E402

import pytest  # noqa: E402

from comparison.label_mapper import LabelMapper  # noqa: E402

DS1 = 'hemibrain:v1.2.1'
DS2 = 'male-cns:v0.9'


def _basic(**overrides):
    defaults = dict(
        datasets=[DS1, DS2],
        source_neurons=['Src'],
        target_neurons=['Tgt'],
        thresholds=[1, 3],
        output_folder='',
        auto_type_mapping=False,
        verbose=False,
    )
    defaults.update(overrides)
    return ComparisonParameters(**defaults)


def _mapper(source_ds=DS1, target_ds=DS1):
    return LabelMapper(
        source_mapping_dict={source_ds: [['aMe1']]},
        source_labels=['SrcStd'],
        target_mapping_dict={target_ds: [['PPL1']]},
        target_labels=['TgtStd'],
    )


def test_scalar_inputs_coerced_to_lists():
    p = ComparisonParameters(
        datasets=DS1, allow_single_dataset=True,
        source_neurons='Src', target_neurons='Tgt',
        source_labels='L1', target_labels='L2',
        intermediate_labels='IL', verbose=False,
    )
    assert p.datasets == [DS1]
    assert p.source_neurons == ['Src']
    assert p.target_neurons == ['Tgt']
    assert p.source_labels == ['L1']
    assert p.target_labels == ['L2']
    assert p.intermediate_labels == ['IL']


def test_overall_mapper_conflicts_raise():
    mapper = _mapper()
    with pytest.raises(ValueError):
        ComparisonParameters(datasets=[DS1, DS2],
                             overall_label_mapper=mapper,
                             source_neurons=['X'],
                             thresholds=[1], verbose=False)
    with pytest.raises(ValueError):
        ComparisonParameters(datasets=[DS1, DS2],
                             overall_label_mapper=mapper,
                             target_neurons=['Y'],
                             thresholds=[1], verbose=False)
    with pytest.raises(ValueError):
        ComparisonParameters(datasets=[DS1, DS2],
                             overall_label_mapper=mapper,
                             source_neurons=_mapper(),
                             thresholds=[1], verbose=False)


def test_overall_mapper_populates_labels():
    mapper = _mapper()
    p = ComparisonParameters(datasets=[DS1, DS2],
                             overall_label_mapper=mapper,
                             thresholds=[1], verbose=False)
    assert p.source_labels == ['SrcStd']
    assert p.target_labels == ['TgtStd']
    assert p._source_mapper is mapper
    assert p._target_mapper is mapper


def test_source_target_mappers_merged_into_overall():
    src_mapper = LabelMapper(source_mapping_dict={DS1: [['aMe1']]},
                             source_labels=['SrcStd'])
    tgt_mapper = LabelMapper(target_mapping_dict={DS1: [['PPL1']]},
                             target_labels=['TgtStd'])
    p = ComparisonParameters(datasets=[DS1, DS2],
                             source_neurons=src_mapper,
                             target_neurons=tgt_mapper,
                             thresholds=[1], verbose=False)
    assert p.overall_label_mapper is not None
    assert p.source_neurons == []
    assert p.target_neurons == []
    assert p.source_labels == ['SrcStd']
    assert p.target_labels == ['TgtStd']


def test_keep_only_conserved_requires_separate_hemispheres():
    p = _basic(keep_only_hemisphere_conserved_connections=True,
               separate_hemispheres=False)
    assert p.keep_only_hemisphere_conserved_connections is False


def test_validation_errors():
    with pytest.raises(ValueError):
        _basic(comparison_mode='bogus')
    with pytest.raises(ValueError):
        _basic(datasets=[])
    with pytest.raises(ValueError):
        # allow_single_dataset defaults to True, so disable it explicitly
        _basic(datasets=[DS1], allow_single_dataset=False)
    with pytest.raises(ValueError):
        _basic(thresholds=[])


def test_verbose_init_prints(capsys):
    # source/target None -> all-neurons warning + verbose summary
    ComparisonParameters(datasets=[DS1, DS2], source_neurons=None,
                         target_neurons=None, thresholds=[1],
                         verbose=True)
    out = capsys.readouterr().out
    assert 'Initialization Summary' in out

    # overall mapper + intermediate labels verbose branches
    mapper = _mapper()
    ComparisonParameters(datasets=[DS1, DS2],
                         overall_label_mapper=mapper,
                         intermediate_labels=['IL'],
                         thresholds=[1], verbose=True)
    out2 = capsys.readouterr().out
    assert 'Defined by LabelMapper' in out2

    # empty list neurons branch
    ComparisonParameters(datasets=[DS1, DS2], source_neurons=[],
                         target_neurons=[], thresholds=[1], verbose=True)
    out3 = capsys.readouterr().out
    assert 'All typed neurons' in out3


def test_label_mapper_consistency_warnings(capsys):
    mapper = LabelMapper(
        source_mapping_dict={DS1: [['aMe1']]},
        source_labels=['SrcStd'],
        target_mapping_dict={DS2: [['PPL1']]},
        target_labels=['TgtStd'],
    )
    ComparisonParameters(datasets=[DS1, DS2, 'weird-ds:v1'],
                         overall_label_mapper=mapper,
                         thresholds=[1], verbose=False)
    out = capsys.readouterr().out
    assert 'missing from LabelMapper' in out
    assert 'Inconsistent datasets' in out
    assert 'Source only' in out
    assert 'Target only' in out


def test_neuron_abbreviation_branches():
    p = _basic()
    assert p._get_neuron_abbreviation(['X'], labels=['Lbl']) == 'Lbl'
    assert p._get_neuron_abbreviation(['X'], labels=['A', 'B']) == 'A_etc'

    mapper = _mapper()
    assert p._get_neuron_abbreviation(mapper) == 'SrcStd'

    assert p._get_neuron_abbreviation({'contains': 'DN.*'}) == 'DN'
    assert p._get_neuron_abbreviation(
        {'startswith': ['Abc', 'Def']}) == 'Abc_etc'
    assert p._get_neuron_abbreviation(
        {'contains': 'VeryLongPatternName'}) == 'VeryLongPa'
    assert p._get_neuron_abbreviation({'unknown_op': 1}) == 'filter'

    assert p._get_neuron_abbreviation(['X.*']) == 'X'
    assert p._get_neuron_abbreviation(['A', 'B']) == 'A_etc'
    assert p._get_neuron_abbreviation([['Nested']]) == 'Nested'
    assert p._get_neuron_abbreviation([]) == 'ALL'
    assert p._get_neuron_abbreviation(None) == 'ALL'


def test_unknown_dataset_codes_and_nicknames():
    p = ComparisonParameters(datasets=['weird-ds:v1'],
                             allow_single_dataset=True,
                             thresholds=[1], verbose=False)
    assert p._get_dataset_short_codes() == 'W'
    assert p.get_display_nickname('weird-ds:v1') == 'WEIRD'


def test_saveas_and_path_properties(tmp_path):
    p = _basic(saveas='myname', output_folder=str(tmp_path))
    assert p.output_name == 'myname'
    assert p.comparison_results_path == os.path.join(
        p.full_output_path, 'comparison_results')
    assert p.visualizations_path == os.path.join(
        p.comparison_results_path, 'visualizations')


def test_mapper_based_neuron_resolution():
    mapper = _mapper(source_ds=DS1, target_ds=DS2)
    p = ComparisonParameters(datasets=[DS1, DS2],
                             overall_label_mapper=mapper,
                             thresholds=[1], verbose=False)
    assert p.get_source_neurons_for_dataset(DS1) == ['aMe1']
    assert p.get_target_neurons_for_dataset(DS2) == ['PPL1']
    assert isinstance(p.get_source_neurons_for_dataset_filtered(DS1), list)
    assert isinstance(p.get_target_neurons_for_dataset_filtered(DS2), list)


def test_ensure_flat_list_and_grouped():
    p = _basic()
    assert p._ensure_flat_list(None) is None
    assert p._ensure_flat_list([]) == []
    assert p._ensure_flat_list(['A', ['B', ['C']]]) == ['A', 'B', 'C']
    assert p._ensure_flat_list([[], 'A']) == ['A']
    assert p._ensure_grouped(None) == [[]]
    assert p._ensure_grouped([]) == [[]]
    assert p._ensure_grouped(['A', 'B']) == [['A', 'B']]
    assert p._ensure_grouped([['A'], ['B']]) == [['A'], ['B']]


def test_create_output_directories(tmp_path):
    p = _basic(output_folder=str(tmp_path))
    p.create_output_directories()
    assert os.path.isdir(p.full_output_path)
    assert os.path.isdir(p.comparison_results_path)
    assert os.path.isdir(p.visualizations_path)
    # Per-dataset ROOT only: requested `minsyn_N` folders are created by the
    # pathfinding run under the applied grammar, never pre-created here
    # (pre-creation left empty mislabeled folders after a collapse).
    assert os.path.isdir(
        os.path.join(p.dataset_data_path, p._sanitize_name(DS1)))
    assert not os.path.isdir(p.get_dataset_output_path(DS1, 1))


def test_resolve_token(monkeypatch):
    p = _basic(token='abc')
    assert p.resolve_token() == 'abc'
    p2 = _basic()
    monkeypatch.setenv('NEUPRINT_APPLICATION_TOKEN', 'envtok')
    assert p2.resolve_token() == 'envtok'


def test_auto_mapping_workspace_missing(monkeypatch):
    real_exists = os.path.exists

    def fake_exists(path):
        if str(path).endswith('male-cns_v1_0_allneurons_neuron_df.csv'):
            return False
        return real_exists(path)

    monkeypatch.setattr(os.path, 'exists', fake_exists)
    p = _basic(auto_type_mapping=True, verbose=True)
    assert p._auto_type_mapper is None


def test_auto_mapping_with_fake_mapper(monkeypatch):
    """The CD-tab backend uses the SHARED mapper singleton
    (``get_type_mapper``), so the test injects at the singleton accessor:
    a loaded mapper becomes ``_auto_type_mapper``, an unloadable one is
    reported as None (the previous new-instance contract)."""
    import comparison.cross_dataset_type_mapper as cdtm

    class _FakeSingletonMapper:
        loaded_result = True

        def __init__(self):
            self._loaded = _FakeSingletonMapper.loaded_result

    fake = _FakeSingletonMapper()
    monkeypatch.setattr(cdtm, 'get_type_mapper',
                        lambda workspace_path=None, force_reload=False: fake)

    real_exists = os.path.exists

    def fake_exists(path):
        if str(path).endswith('male-cns_v1_0_allneurons_neuron_df.csv'):
            return True
        return real_exists(path)

    monkeypatch.setattr(os.path, 'exists', fake_exists)

    _FakeSingletonMapper.loaded_result = True
    fake._loaded = True
    p = _basic(auto_type_mapping=True)
    assert p._auto_type_mapper is fake

    _FakeSingletonMapper.loaded_result = False
    fake._loaded = False
    p2 = _basic(auto_type_mapping=True)
    assert p2._auto_type_mapper is None


class _FakeTypeMapper:
    """Fake mapper on the shared resolver contract: the parameters layer
    resolves through ``comparison.type_resolver`` (which consumes
    ``get_mapping_decision`` status dicts), not the legacy
    ``get_mapped_type`` one-target API."""

    _loaded = True

    def __init__(self):
        self.exported_paths = []
        self.mapping_calls = []

    def _detect_type_source(self, t):
        return DS1 if t in ('Src', 'NoMap') else None

    def detect_type_source(self, t):
        return self._detect_type_source(t)

    def _get_type_mapping_key(self, ds):
        return ds

    def get_mapping_decision(self, t, src_ds, dst_ds, include_bridges=False):
        self.mapping_calls.append((t, src_ds, dst_ds))
        if t == 'Src':
            return {'status': 'mapped', 'source_type': t,
                    'target_type': 'SrcMapped', 'target_types': ['SrcMapped'],
                    'relationship': '1-to-1', 'conflicts': []}
        return {'status': 'unmapped', 'source_type': t,
                'target_type': None, 'target_types': [],
                'relationship': None, 'conflicts': []}

    def get_alias_candidates(self, type_name, datasets, source_dataset=None):
        return {ds: {'outcome': 'no counterpart known', 'candidates': []}
                for ds in datasets}

    def get_type_bridges(self, type_name, source_ds, target_ds,
                         max_bridges=8):
        return []

    def export_mapping(self, path, **kwargs):
        self.exported_paths.append(path)


def test_resolve_neurons_with_auto_mapping():
    p = _basic()
    p._auto_type_mapper = _FakeTypeMapper()
    neurons = ['Src', 'NoMap', 'Regex.*', 123]
    out = p._resolve_neurons_with_auto_mapping(neurons, DS2)
    assert out == ['SrcMapped', 'NoMap', 'Regex.*', 123]
    out_rm = p._resolve_neurons_with_auto_mapping(neurons, DS2,
                                                  remove_unmapped=True)
    assert out_rm == ['SrcMapped', 'Regex.*', 123]
    # None and [] pass through
    assert p._resolve_neurons_with_auto_mapping(None, DS2) is None
    assert p._resolve_neurons_with_auto_mapping([], DS2) == []
    # no mapper -> passthrough
    p2 = _basic()
    assert p2._resolve_neurons_with_auto_mapping(['Src'], DS2) == ['Src']


def test_explicit_source_release_wins_over_auto_detection():
    p = _basic()
    p.source_dataset = 'male-cns:v0.9'
    mapper = _FakeTypeMapper()
    p._auto_type_mapper = mapper

    # The EXPLICIT source release — not the mapper's auto-detection (which
    # answers DS1 for 'Src') — is what the shared resolver receives.  The
    # target is DS1 here because the resolver's same-namespace rule (like
    # production get_mapped_type) resolves v0.9->v0.9 natively without
    # consulting the mapping tables.
    assert p._resolve_neurons_with_auto_mapping(['Src'], DS1) == ['SrcMapped']
    assert mapper.mapping_calls == [
        ('Src', 'male-cns:v0.9', DS1),
    ]


def test_print_neuron_mapping_summary(capsys):
    p = _basic()
    p._auto_type_mapper = _FakeTypeMapper()
    p._auto_type_mapper.get_source_target_mapping_summary = (
        lambda types, datasets: {
            'different_mappings': [('Src', {DS1: 'Src', DS2: 'SrcAlt'})],
            'n_to_1_warnings': [
                ('T1', DS1, DS2, {'T1', 'T2'}),
                ('T3', DS1, DS2, {'T4', 'T5'}),
            ],
            'one_to_n_warnings': [('S1', DS1, DS2, {'S1a', 'S1b'})],
        })
    p._print_neuron_mapping_summary(['Src', 'T1', 42], [DS1, DS2])
    out = capsys.readouterr().out
    assert 'Auto-mapped types' in out
    assert 'N-to-1 type mapping detected' in out
    assert '1-to-N type mapping detected' in out

    # no mapper -> early return
    p2 = _basic()
    assert p2._print_neuron_mapping_summary(['Src'], [DS1, DS2]) is None


def test_get_auto_type_mapper_and_export(tmp_path):
    p = _basic()
    assert p.get_auto_type_mapper() is None
    assert p.export_auto_mapping() is None

    fake = _FakeTypeMapper()
    p._auto_type_mapper = fake
    out_path = p.export_auto_mapping(str(tmp_path / 'map.csv'))
    assert out_path == str(tmp_path / 'map.csv')
    assert fake.exported_paths == [str(tmp_path / 'map.csv')]

    default_path = p.export_auto_mapping()
    assert default_path == os.path.join(p.full_output_path,
                                        'auto_type_mapping.csv')


# ---------------------------------------------------------------------------
# Row-wise threshold query contract
# ---------------------------------------------------------------------------

def test_standard_thresholds_are_query_rows_and_jobs_are_deduplicated():
    p = _basic(thresholds=[5, 3, 5])

    assert [q["id"] for q in p.get_threshold_queries()] == [
        "threshold_3", "threshold_5"
    ]
    assert all(q["thresholds"] == {DS1: int(q["thresholds"][DS1]),
                                    DS2: int(q["thresholds"][DS2])}
               for q in p.get_threshold_queries())
    assert p.get_unique_threshold_jobs() == [(DS1, 3), (DS1, 5),
                                             (DS2, 3), (DS2, 5)]


def test_combination_rows_preserve_identity_and_derive_raw_schedule():
    p = _basic(
        thresholds=[],
        threshold_mode="combinations",
        threshold_dataset_order=[DS2, DS1],
        threshold_combinations=[
            {"id": "q-low", "label": "Low", "thresholds": {DS2: 7, DS1: 3}},
            {"id": "q-high", "label": "High", "thresholds": {DS2: 7, DS1: 9}},
        ],
    )

    queries = p.get_threshold_queries()
    assert [(q["id"], q["label"]) for q in queries] == [
        ("q-low", "Low"), ("q-high", "High")
    ]
    assert p.thresholds == [3, 7, 9]
    assert p.get_thresholds_for_dataset(DS1) == [3, 9]
    assert p.get_thresholds_for_dataset(DS2) == [7]
    assert p.get_unique_threshold_jobs() == [(DS2, 7), (DS1, 3), (DS1, 9)]

    restored = ComparisonParameters.from_dict(p.to_dict())
    assert restored.threshold_mode == "combinations"
    assert restored.threshold_dataset_order == [DS2, DS1]
    assert [q["id"] for q in restored.get_threshold_queries()] == [
        "q-low", "q-high"
    ]


def test_combination_mode_requires_at_least_two_datasets():
    with pytest.raises(ValueError, match="at least two selected datasets"):
        _basic(
            datasets=[DS1],
            thresholds=[],
            threshold_mode="combinations",
            threshold_combinations=[
                {"id": "q", "thresholds": {DS1: 3}},
            ],
        )


@pytest.mark.parametrize("rows, message", [
    ([{"id": "q", "thresholds": {DS1: 3}}], "missing"),
    ([{"id": "q", "thresholds": {DS1: 3, DS2: 5, "other": 1}}], "unknown"),
    ([{"id": "q", "thresholds": {DS1: 3, DS2: 5}},
      {"id": "q", "thresholds": {DS1: 4, DS2: 6}}], "duplicate id"),
    ([{"id": "q1", "thresholds": {DS1: 3, DS2: 5}},
      {"id": "q2", "thresholds": {DS1: 3, DS2: 5}}], "duplicate threshold"),
])
def test_combination_rows_validate_complete_unique_queries(rows, message):
    with pytest.raises(ValueError, match=message):
        _basic(
            thresholds=[],
            threshold_mode="combinations",
            threshold_combinations=rows,
        )


def test_from_dict_migrates_equal_length_legacy_threshold_lists():
    """Old serialized per-dataset lists become explicit query rows once."""
    legacy = {
        "datasets": [DS1, DS2],
        "source_neurons": ["Src"],
        "target_neurons": ["Tgt"],
        "thresholds": [3, 5, 7, 9],
        "dataset_thresholds": {
            DS1: [3, 5],
            DS2: [7, 9],
        },
        "output_folder": "",
        "auto_type_mapping": False,
        "verbose": False,
    }

    with pytest.warns(UserWarning, match="Migrated legacy"):
        restored = ComparisonParameters.from_dict(legacy)

    assert restored.threshold_mode == "combinations"
    assert restored.dataset_thresholds is None
    assert restored.threshold_dataset_order == [DS1, DS2]
    assert restored.get_threshold_queries() == [
        {
            "id": "combo_001",
            "label": "Combination 1",
            "thresholds": {DS1: 3, DS2: 7},
            "dataset_order": [DS1, DS2],
            "row_mode": None,
        },
        {
            "id": "combo_002",
            "label": "Combination 2",
            "thresholds": {DS1: 5, DS2: 9},
            "dataset_order": [DS1, DS2],
            "row_mode": None,
        },
    ]


def test_from_dict_migrates_one_dataset_legacy_schedule_to_standard():
    legacy = {
        "datasets": [DS1],
        "dataset_thresholds": {DS1: [3, 5]},
        "verbose": False,
    }

    with pytest.warns(UserWarning, match="standard thresholds"):
        restored = ComparisonParameters.from_dict(legacy)

    assert restored.threshold_mode == "standard"
    assert restored.thresholds == [3, 5]
    assert restored.threshold_combinations is None


def test_from_dict_rejects_unequal_legacy_threshold_lists():
    legacy = {
        "datasets": [DS1, DS2],
        "dataset_thresholds": {DS1: [3, 5], DS2: [7]},
        "verbose": False,
    }
    with pytest.raises(ValueError, match="unequal lengths"):
        ComparisonParameters.from_dict(legacy)

# --- PARAMS-APPEND-DONE ---


def test_for_run_folder_pins_full_output_path(tmp_path):
    """Plan R4: for_run_folder reads a run's parameters.json and pins
    full_output_path to the EXISTING folder (from_dict alone re-timestamps
    the folder name, so re-export scripts would silently start a new
    run)."""
    import json

    p = ComparisonParameters(
        datasets=['hemibrain:v1.2.1', 'male-cns:v1.0'],
        source_neurons=['aMe1'],
        target_neurons=['KCg-d'],
        threshold_mode='combinations',
        threshold_combinations=[
            {'id': 'threshold=3', 'label': 'threshold=3',
             'thresholds': {'hemibrain:v1.2.1': 3, 'male-cns:v1.0': 3}},
            {'id': 'threshold=10', 'label': 'threshold=10',
             'thresholds': {'hemibrain:v1.2.1': 10, 'male-cns:v1.0': 10}},
        ],
    )
    run_dir = tmp_path / "cross-dataset_aMe1_to_KCg_H_20260101_000000"
    run_dir.mkdir()
    (run_dir / "parameters.json").write_text(json.dumps(p.to_dict()))

    pinned = ComparisonParameters.for_run_folder(str(run_dir))
    assert pinned.full_output_path == str(run_dir)
    assert pinned.output_folder == str(tmp_path)
    assert pinned.saveas == run_dir.name
    # the saved query set survives the round trip
    assert pinned.threshold_mode == 'combinations'
    assert [q['id'] for q in (pinned.threshold_combinations or [])] == \
        [q['id'] for q in (p.threshold_combinations or [])]


def test_for_run_folder_missing_parameters_json(tmp_path):
    import pytest

    empty = tmp_path / "not_a_run"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        ComparisonParameters.for_run_folder(str(empty))


def test_from_dict_round_trip_preserves_flags():
    """Plan round-4 fix: from_dict silently dropped five flags (skip_bodyId,
    find_reciprocal, separate_hemispheres, symmetry_analysis,
    keep_only_hemisphere_conserved_connections) — a pinned resume/re-export
    downgraded a hemispheric run to non-hemispheric."""
    p = ComparisonParameters(
        datasets=['d1', 'd2'], source_neurons=['a'], target_neurons=['b'],
        separate_hemispheres=True, symmetry_analysis=True,
        keep_only_hemisphere_conserved_connections=True,
        find_reciprocal=True, skip_bodyId=False,
    )
    d = p.to_dict()
    p2 = ComparisonParameters.from_dict(d)
    assert p2.separate_hemispheres is True
    assert p2.symmetry_analysis is True
    assert p2.keep_only_hemisphere_conserved_connections is True
    assert p2.find_reciprocal is True
    assert p2.skip_bodyId is False
