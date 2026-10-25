"""Default-on autapse exclusion — the shared Paths/Network contract.

Acceptance fixture: ``A -> A`` weight 5 (one neuron's self-edge) plus
``A -> A'`` weight 3 (two distinct neurons of the same type). The type-level
self-cell must weigh 3 with the rule on and 8 with it off, the run notes must
report 1 row / 5 synapses, and the manifest must carry the census.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))

from comparison.comparison_analyzer import ComparisonAnalyzer  # noqa: E402
from comparison.comparison_parameters import ComparisonParameters  # noqa: E402
from tests.core.test_coana_coverage import make_fc  # noqa: E402
from tests.core.test_find_network import _make_network_fc  # noqa: E402
from type_level_refill import (  # noqa: E402
    _polars_stream_passes, read_run_provenance,
)


def _network_self_cell(fc) -> pd.DataFrame:
    table = pd.read_csv(
        Path(fc.network_folder) / 'data_details' / 'connection_type.csv')
    return table[table['type_pre'] == table['type_post']]


class TestNetworkTab:
    def test_type_self_cell_weight_on_and_off(self, monkeypatch, tmp_path):
        edges = [("A", "A", 5), ("A", "A2", 3)]

        fc, *_ = _make_network_fc(
            monkeypatch, tmp_path / 'on', edges, ["A", "A2"])
        fc.FindNetwork()
        self_cell = _network_self_cell(fc)
        assert list(self_cell['weight']) == [3]

        attrs = json.loads(
            (Path(fc.network_folder) / 'all_attributes.json').read_text(encoding="utf-8"))
        assert attrs['autapses_excluded'] == {'rows': 1, 'synapses': 5}
        note = (Path(fc.network_folder) / 'user_warning_notes.txt').read_text(encoding="utf-8")
        assert '[autapses] exclude_autapses=True (default)' in note
        assert '1 row(s) / 5 synapse(s)' in note
        params_txt = (
            Path(fc.network_folder) / 'parameters.txt').read_text(encoding="utf-8")
        param_line = [line for line in params_txt.splitlines()
                      if line.startswith('exclude autapses:')]
        assert param_line and param_line[0].split(':', 1)[1].strip() == 'True'

        fc_off, *_ = _make_network_fc(
            monkeypatch, tmp_path / 'off', edges, ["A", "A2"],
            exclude_autapses=False)
        fc_off.FindNetwork()
        self_cell_off = _network_self_cell(fc_off)
        assert list(self_cell_off['weight']) == [8]

        attrs_off = json.loads(
            (Path(fc_off.network_folder) / 'all_attributes.json').read_text(encoding="utf-8"))
        assert attrs_off['autapses_excluded'] == {'rows': 0, 'synapses': 0}
        assert attrs_off['autapses_found'] == {'rows': 1, 'synapses': 5}
        note_off = (
            Path(fc_off.network_folder) / 'user_warning_notes.txt').read_text(encoding="utf-8")
        assert 'exclude_autapses=False' in note_off
        assert '1 row(s) / 5 synapse(s)' in note_off

    def test_autapse_only_network_reports_before_empty_exit(
            self, monkeypatch, tmp_path):
        fc, *_ = _make_network_fc(
            monkeypatch, tmp_path, [("A", "A", 5)], ["A"])
        fc.FindNetwork()
        attrs = json.loads(
            (Path(fc.network_folder) / 'all_attributes.json').read_text(encoding="utf-8"))
        assert attrs['autapses_excluded'] == {'rows': 1, 'synapses': 5}
        note = (Path(fc.network_folder) / 'user_warning_notes.txt').read_text(encoding="utf-8")
        assert '[autapses]' in note


class TestPathsTab:
    def test_graph_frames_drop_self_edges(self):
        frame = pl.DataFrame({
            'bodyId_pre': ['A', 'A'],
            'bodyId_post': ['A', 'A2'],
            'weight': [5, 3],
        })
        fc, _ = make_fc(exclude_autapses=True)
        fc._reset_autapse_tracking()
        out = fc._filter_autapses_in_frames([frame])[0]
        assert out['bodyId_pre'].to_list() == ['A']
        assert out['bodyId_post'].to_list() == ['A2']
        assert fc.autapses_excluded == {'rows': 1, 'synapses': 5}
        assert fc.autapses_found == {'rows': 1, 'synapses': 5}
        assert fc._autapse_pairs == {('A', 'A'): 5}

        fc_off, _ = make_fc(exclude_autapses=False)
        fc_off._reset_autapse_tracking()
        out_off = fc_off._filter_autapses_in_frames([frame])[0]
        assert out_off.equals(frame)
        assert fc_off.autapses_excluded == {'rows': 0, 'synapses': 0}
        assert fc_off.autapses_found == {'rows': 1, 'synapses': 5}

    def test_pandas_frame_census(self):
        frame = pd.DataFrame({
            'bodyId_pre': ['A', 'A', 'B'],
            'bodyId_post': ['A', 'A2', 'B'],
            'weight': [5, 3, 7],
        })
        fc, _ = make_fc(exclude_autapses=True)
        fc._reset_autapse_tracking()
        out = fc._filter_autapses_in_frames([frame])[0]
        assert len(out) == 1
        assert 5 not in out['weight'].tolist()
        assert fc.autapses_excluded == {'rows': 2, 'synapses': 12}

    def test_graph_frames_and_note(self, tmp_path):
        frame = pl.DataFrame({
            'bodyId_pre': ['A', 'A'],
            'bodyId_post': ['A', 'A2'],
            'weight': [5, 3],
        })
        fc, _ = make_fc(
            exclude_autapses=True, max_interlayer=3, weight_basis='synapse',
            threshold_combination='or', min_ratio=0.0, min_synapse_num=1,
            graph_edge_limit_bodyid=None)
        fc._reset_autapse_tracking()
        frames = fc._graph_edge_frames([frame], ['A'], ['A2'])
        assert len(frames) == 1
        assert frames[0]['bodyId_post'].to_list() == ['A2']
        assert fc.autapses_excluded == {'rows': 1, 'synapses': 5}

        run = tmp_path / 'run'
        run.mkdir()
        fc._write_user_warning_notes(str(run))
        note = (run / 'user_warning_notes.txt').read_text(encoding="utf-8")
        assert '[autapses]' in note
        assert '1 row(s) / 5 synapse(s)' in note

    def test_no_autapse_frames_are_identical(self):
        frame = pl.DataFrame({
            'bodyId_pre': ['A', 'B'],
            'bodyId_post': ['B', 'C'],
            'weight': [4, 9],
        })
        fc_on, _ = make_fc(exclude_autapses=True)
        fc_off, _ = make_fc(exclude_autapses=False)
        for fc in (fc_on, fc_off):
            fc._reset_autapse_tracking()
        on = fc_on._filter_autapses_in_frames([frame])[0]
        off = fc_off._filter_autapses_in_frames([frame])[0]
        assert on.equals(off)
        assert on.equals(frame)
        assert fc_on.autapses_excluded == {'rows': 0, 'synapses': 0}

    def test_graph_frames_census_is_endpoint_scoped(self):
        frame = pl.DataFrame({
            'bodyId_pre': ['A', 'B', 'A'],
            'bodyId_post': ['A', 'B', 'A2'],
            'weight': [5, 7, 3],
        })
        fc, _ = make_fc(exclude_autapses=True)
        fc._reset_autapse_tracking()
        out = fc._graph_edge_frames([frame], ['A'], ['A2'])
        # Both self-edges are removed, but only the queried endpoint A's
        # autapse belongs to this query's census (B is a cone intermediate).
        assert out[0]['bodyId_pre'].to_list() == ['A']
        assert out[0]['bodyId_post'].to_list() == ['A2']
        assert fc.autapses_excluded == {'rows': 1, 'synapses': 5}
        assert fc.autapses_found == {'rows': 1, 'synapses': 5}

    def test_per_tier_census_for_replay_slices(self):
        fc, _ = make_fc(exclude_autapses=True, weight_basis='synapse')
        fc._reset_autapse_tracking()
        fc._autapse_pairs = {('A', 'A'): 5, ('B', 'B'): 2}
        assert fc._autapse_stats_at_threshold(3) == {
            'rows': 1, 'synapses': 5}
        assert fc._autapse_stats_at_threshold(1) == {
            'rows': 2, 'synapses': 7}


class TestComparisonLane:
    def test_parameters_roundtrip(self):
        params = ComparisonParameters(
            datasets=['test:v1'], thresholds=[3], exclude_autapses=False)
        assert params.to_dict()['exclude_autapses'] is False
        restored = ComparisonParameters.from_dict(params.to_dict())
        assert restored.exclude_autapses is False
        # Old payloads default to the contract's True.
        payload = params.to_dict()
        payload.pop('exclude_autapses')
        assert ComparisonParameters.from_dict(payload).exclude_autapses is True

    def test_direct_edge_filter_and_manifest(self, tmp_path):
        params = ComparisonParameters(
            datasets=['test:v1'], thresholds=[3],
            output_folder=str(tmp_path))
        analyzer = ComparisonAnalyzer(params, verbose=False)
        frame = pd.DataFrame({
            'bodyId_pre': ['A', 'A'],
            'bodyId_post': ['A', 'A2'],
            'weight': [5, 3],
            'type_pre': ['TA', 'TA'],
            'type_post': ['TA', 'TA'],
        })
        out = analyzer._filter_autapses('test:v1', 3, frame)
        assert len(out) == 1
        assert int(out.iloc[0]['weight']) == 3
        assert analyzer._autapse_excluded[('test:v1', 3)] == {
            'rows': 1, 'synapses': 5}

        analyzer._write_run_manifest(str(tmp_path))
        manifest = json.loads(
            (tmp_path / 'run_manifest.json').read_text(encoding="utf-8"))
        assert manifest['autapses_excluded']['test:v1@3'] == {
            'rows': 1, 'synapses': 5}
        note = (tmp_path / 'user_warning_notes.txt').read_text(encoding="utf-8")
        assert '[autapses]' in note
        assert 'test:v1@3 1 row(s) / 5 synapse(s)' in note
        # A re-export replaces its own census bullet; it never accumulates.
        analyzer._write_run_manifest(str(tmp_path))
        note_again = (tmp_path / 'user_warning_notes.txt').read_text(encoding="utf-8")
        assert note_again.count('[autapses]') == 1

    def test_edge_analysis_notes_the_drop_without_export(self, tmp_path,
                                                         monkeypatch):
        dataset = 'banc_v888'
        safe = 'banc_v888'
        folder = tmp_path / safe
        folder.mkdir()
        pd.DataFrame({
            'bodyId_pre': ['A', 'A'],
            'bodyId_post': ['A', 'A2'],
            'weight': [5, 3],
            'type_pre': ['TA', 'TA'],
            'type_post': ['TA', 'TA'],
        }).to_parquet(folder / f'{safe}_connections.parquet', index=False)
        params = ComparisonParameters(
            datasets=[dataset], source_neurons=['TA'],
            target_neurons=['TA'], thresholds=[3],
            output_folder=str(tmp_path))
        analyzer = ComparisonAnalyzer(params, verbose=False)
        monkeypatch.setattr(analyzer, '_get_datasets_folder',
                            lambda: str(tmp_path))
        out = analyzer.run_edge_analysis(dataset, 3)
        assert len(out) == 1
        assert int(out.iloc[0]['weight']) == 3
        assert analyzer._autapse_excluded[(dataset, 3)] == {
            'rows': 1, 'synapses': 5}
        # The direct-edge query reports its own drop immediately; no
        # export_results()/run_manifest.json call is required.
        note_path = (Path(analyzer.parameters.full_output_path)
                     / 'user_warning_notes.txt')
        assert note_path.exists()
        note = note_path.read_text(encoding="utf-8")
        assert '[autapses]' in note
        assert 'banc_v888@3 1 row(s) / 5 synapse(s)' in note

    def test_off_keeps_rows_and_reports_found_mass(self, tmp_path):
        params = ComparisonParameters(
            datasets=['test:v1'], thresholds=[3],
            output_folder=str(tmp_path), exclude_autapses=False)
        analyzer = ComparisonAnalyzer(params, verbose=False)
        frame = pd.DataFrame({
            'bodyId_pre': ['A', 'A'],
            'bodyId_post': ['A', 'A2'],
            'weight': [5, 3],
        })
        out = analyzer._filter_autapses('test:v1', 3, frame)
        assert len(out) == 2
        assert analyzer._autapse_excluded[('test:v1', 3)] == {
            'rows': 0, 'synapses': 0}
        assert analyzer._autapse_found[('test:v1', 3)] == {
            'rows': 1, 'synapses': 5}


class TestRefillConsistency:
    def test_provenance_reads_the_flag(self, tmp_path):
        (tmp_path / 'parameters.txt').write_text(
            'exclude autapses: True\n'
            'requested_threshold: 3\n'
            'applied_threshold: 3\n')
        assert read_run_provenance(tmp_path)['exclude_autapses'] == 'True'

    def test_refill_admissions_skip_self_edges(self):
        edge_lf = pl.DataFrame({
            'bodyId_pre': ['A', 'A'],
            'bodyId_post': ['A', 'A2'],
            'weight': [5, 3],
        }).lazy()
        kwargs = dict(
            ratio_mode=False, asked=1, and_syn_floor=None,
            and_ratio_tier=None, min_syn_active=False, min_syn=1,
            exclude_intra=False, want_full=True)
        on = _polars_stream_passes(
            edge_lf, {'A': 'TA', 'A2': 'TA'}, {('TA', 'TA')},
            {'A', 'A2'}, exclude_autapses=True, **kwargs)
        assert on['e_u'] == [('A', 'A2', 3.0)]
        assert on['e_u_full'] == [('A', 'A2', 3.0)]
        off = _polars_stream_passes(
            edge_lf, {'A': 'TA', 'A2': 'TA'}, {('TA', 'TA')},
            {'A', 'A2'}, exclude_autapses=False, **kwargs)
        assert set(off['e_u']) == {('A', 'A', 5.0), ('A', 'A2', 3.0)}
