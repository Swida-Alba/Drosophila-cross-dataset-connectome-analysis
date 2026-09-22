"""Tests for the run manifest the skeleton viewer writes next to its HTML.

``visualization_manifest.json`` is what lets the individual and video
re-exporters work on a stored page: it names the canonical page, records the
render settings, and carries the per-trace role table so identity does not
have to be guessed from trace names again.
"""

import json
import sys
from pathlib import Path

import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visualize_skeleton import (  # noqa: E402
    VisualizeSkeleton,
    classify_traces,
    manifest_role_drift,
)


def make_vis(tmp_path, figure, **attrs):
    vis = object.__new__(VisualizeSkeleton)
    merged = dict(
        verbose=False, dataset='hemibrain:v1.2.1', client_type='neuprint',
        version=None, background_color='rgba(255, 255, 255, 1.0)',
        brain_mesh='native', vnc_mesh='false', mesh_roi=['LH'],
        legend_mode='tree', freeze_view=True, export_method='kaleido',
        export_scale=2, neuron_alpha=0.2, layer_names=['KC layer'],
        saveas='scene', save_folder=str(tmp_path), fig_3d=figure,
    )
    merged.update(attrs)
    for key, value in merged.items():
        setattr(vis, key, value)
    vis.fig_path = str(tmp_path / 'scene')
    return vis


def figure():
    vis = make_vis(Path('.'), None)
    fig = go.Figure()
    neuron = go.Scatter3d(x=[0, 10], y=[0, 10], z=[0, 10], mode='lines',
                          name='KC1a', legendgroup='KC1a', showlegend=True)
    vis._stamp_trace_identity(neuron, kind='neuron', group='KC layer',
                              type_label='KC1a', item='256_aMe4',
                              body_id='256')
    fig.add_trace(neuron)
    fig.add_trace(go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode='markers',
                              name='synapses 1 -> 2',
                              legendgroup='synapses 1 -> 2'))
    mesh = go.Mesh3d(x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0], i=[0], j=[1],
                     k=[2], name='Brain mesh')
    mesh.legendrank = 2e8
    fig.add_trace(mesh)
    return fig


class TestManifestContent:
    def test_pages_and_render_settings(self, tmp_path):
        manifest = make_vis(tmp_path, figure()).visualization_manifest()
        assert manifest['schema_version'] == 1
        assert manifest['canonical_page'] == 'scene.html'
        assert manifest['freeze_view'] is True
        assert manifest['legend_mode'] == 'tree'
        assert manifest['neuron_alpha'] == 0.2
        assert manifest['layer_names'] == ['KC layer']
        # Keys the re-exporter cannot derive from the page itself. The
        # simplified copy is recognised by its _simplified suffix, and the
        # freeze pin is baked into the page's own scene layout, so neither is
        # mirrored here.
        assert 'degraded_pages' not in manifest
        assert 'frozen_ranges' not in manifest

    def test_trace_table_records_roles_and_visibility(self, tmp_path):
        manifest = make_vis(tmp_path, figure()).visualization_manifest()
        rows = manifest['traces']
        assert [r['kind'] for r in rows] == ['neuron', 'synapse', 'mesh']
        assert rows[0]['item'] == '256_aMe4'
        assert rows[0]['body_id'] == '256'
        assert rows[0]['rule'] == 'stamped_identity'
        assert all(r['visible'] is True for r in rows)
        assert [r['index'] for r in rows] == [0, 1, 2]

    def test_hidden_trace_visibility_is_recorded(self, tmp_path):
        fig = figure()
        fig.data[0].visible = False
        rows = make_vis(tmp_path, fig).visualization_manifest()['traces']
        assert rows[0]['visible'] is False

    def test_views_carry_cameras_for_the_known_dataset(self, tmp_path):
        manifest = make_vis(tmp_path, figure()).visualization_manifest()
        assert 'Front' in manifest['views']
        assert set(manifest['views']['Front']) == {'eye', 'up', 'center'}

    def test_camera_lookup_never_breaks_the_manifest(self, tmp_path):
        # dataset_view_cameras falls back to a generic rig for unknown
        # datasets rather than raising, so the manifest always has views.
        vis = make_vis(tmp_path, figure(), dataset='not-a-real-dataset:v9')
        views = vis.visualization_manifest()['views']
        assert 'Front' in views
        assert set(views['Front']) == {'eye', 'up', 'center'}

    def test_no_figure_means_no_manifest(self, tmp_path):
        vis = make_vis(tmp_path, None)
        assert vis.visualization_manifest() is None
        assert vis.write_visualization_manifest() is None


class TestManifestFile:
    def test_written_next_to_the_page_and_json_valid(self, tmp_path):
        vis = make_vis(tmp_path, figure())
        path = vis.write_visualization_manifest()
        assert Path(path).name == 'visualization_manifest.json'
        on_disk = json.loads(Path(path).read_text(encoding='utf-8'))
        assert on_disk == vis.visualization_manifest()
        assert vis._manifest_path == path

    def test_a_manifest_failure_never_breaks_the_run(self, tmp_path, monkeypatch):
        vis = make_vis(tmp_path / 'missing-folder', figure())
        assert vis.write_visualization_manifest() is None

    def test_unwritable_manifest_is_swallowed(self, tmp_path, monkeypatch):
        vis = make_vis(tmp_path, figure())

        def boom(*args, **kwargs):
            raise OSError('disk on fire')

        monkeypatch.setattr(json, 'dump', boom)
        assert vis.write_visualization_manifest() is None


class TestRecordedRolesAreCheckedOnReexport:
    """``traces`` is the one manifest table the re-export reads back.

    A page that loses its ``meta`` stamps in the HTML round trip still opens and
    still profiles -- at ``legend`` level only, which is indistinguishable from
    an honest legacy run. Comparing the page's roles with what the run recorded
    is the only way the loss shows up.
    """

    def test_a_page_that_reads_back_as_recorded_says_nothing(self, tmp_path):
        fig = figure()
        manifest = make_vis(tmp_path, fig).visualization_manifest()
        roles = classify_traces(fig.data, mesh_roi_names=['LH'])
        assert manifest_role_drift(manifest, roles) is None

    def test_roles_that_changed_are_named_with_both_counts(self, tmp_path):
        manifest = make_vis(tmp_path, figure()).visualization_manifest()
        # What a page whose stamps vanished parses as: unnamed, ungrouped
        # geometry, i.e. one plain neuron and no synapse or mesh at all.
        read_back = classify_traces(
            [go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1])])
        drift = manifest_role_drift(manifest, read_back)
        assert '1 mesh -> 0' in drift
        assert '1 synapse -> 0' in drift
        assert ' neuron ' not in drift, \
            'a role count that survived must not be reported'
        assert 'round trip' in drift

    def test_a_run_without_a_recorded_table_stays_silent(self, tmp_path):
        fig = figure()
        roles = classify_traces(fig.data, mesh_roi_names=['LH'])
        assert manifest_role_drift({}, roles) is None
        assert manifest_role_drift({'traces': []}, roles) is None
        assert manifest_role_drift(
            {'traces': [{'index': 0, 'label': 'KC1a'}]}, roles) is None
