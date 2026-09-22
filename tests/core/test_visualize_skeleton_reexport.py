"""Tests for re-exporting individual profiles from an existing HTML page.

The re-export path is Figure-free on purpose: it reads the page's own
``Plotly.newPlot`` payload, classifies the traces the same way the live
renderer does, and hands the resulting plan to the shared WebDriver export.
No Chrome is launched here -- ``export_individuals_webdriver`` is captured.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import plotly.graph_objects as go
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import visualize_skeleton as vs  # noqa: E402
from visualize_skeleton import (  # noqa: E402
    VISUALIZATION_MANIFEST_NAME,
    VisualizeSkeleton,
    export_individuals_from_html,
    profile_plan_from_html,
    read_visualization_manifest,
    resolve_viewer_page,
)


def bare(**attrs):
    vis = object.__new__(VisualizeSkeleton)
    merged = dict(
        verbose=False, dataset='hemibrain:v1.2.1', client_type='neuprint',
        background_color='rgba(255, 255, 255, 1.0)', legend_mode='tree',
        layer_names=['KC layer'], custom_layer_names=None,
    )
    merged.update(attrs)
    for key, value in merged.items():
        setattr(vis, key, value)
    vis._in_page_warning_html = lambda: ''
    return vis


def scene(tmp_path, saveas='scene'):
    """Write a small viewer page (no plotly.js bundle) and return its path."""
    vis = bare()
    fig = go.Figure()
    for body_id, item in (('256', '256_aMe4'), ('257', '257_aMe5')):
        trace = go.Scatter3d(x=[0, 100], y=[0, 10], z=[0, 10], mode='lines',
                             name='KC1a', legendgroup='KC1a',
                             showlegend=body_id == '256')
        vis._stamp_trace_identity(trace, kind='neuron', group='KC layer',
                                  type_label='KC1a', item=item,
                                  body_id=body_id)
        fig.add_trace(trace)
    mesh = go.Mesh3d(x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0], i=[0], j=[1],
                     k=[2], name='Brain mesh', legendgroup='KC1a')
    mesh.legendrank = 2e8
    fig.add_trace(mesh)
    path = tmp_path / f'{saveas}.html'
    vis._write_plotly_html(fig, str(path), include_plotlyjs=False)
    return path


def write_manifest(run_dir, **over):
    manifest = {
        'schema_version': 1, 'dataset': 'hemibrain:v1.2.1',
        'canonical_page': 'scene.html', 'mesh_roi': ['LH'],
        'brain_mesh': 'native', 'background_color': 'white',
        'views': {'Front': {'eye': {'x': 1, 'y': 0, 'z': 0},
                            'up': {'x': 0, 'y': 0, 'z': 1},
                            'center': {'x': 0, 'y': 0, 'z': 0}}},
    }
    manifest.update(over)
    (run_dir / VISUALIZATION_MANIFEST_NAME).write_text(
        json.dumps(manifest), encoding='utf-8')
    return manifest


@pytest.fixture
def run(tmp_path):
    page = scene(tmp_path)
    (tmp_path / 'scene_simplified.html').write_text(
        page.read_text().replace('</body>', '<!-- degraded --></body>'),
        encoding='utf-8')
    write_manifest(tmp_path)
    return tmp_path, page


class TestResolveViewerPage:
    def test_a_plain_page_is_used_as_is(self, run):
        _run_dir, page = run
        assert resolve_viewer_page(str(page)) == str(page)

    def test_a_folder_resolves_through_the_manifest(self, run):
        run_dir, page = run
        assert resolve_viewer_page(str(run_dir)) == str(page)

    def test_a_folder_without_a_manifest_skips_simplified_pages(self, tmp_path):
        page = scene(tmp_path, saveas='older')
        (tmp_path / 'older_simplified.html').write_text('x', encoding='utf-8')
        (tmp_path / '_temp_export.html').write_text('x', encoding='utf-8')
        assert resolve_viewer_page(str(tmp_path)) == str(page)

    def test_a_simplified_page_is_routed_to_its_canonical_twin(self, run):
        run_dir, page = run
        degraded = run_dir / 'scene_simplified.html'
        assert resolve_viewer_page(str(degraded)) == str(page)

    def test_a_simplified_page_without_a_manifest_is_rejected(self, tmp_path):
        page = scene(tmp_path)
        degraded = tmp_path / 'scene_simplified.html'
        degraded.write_text(page.read_text(), encoding='utf-8')
        with pytest.raises(ValueError, match='simplified copy'):
            resolve_viewer_page(str(degraded))

    def test_a_folder_with_no_page_raises(self, tmp_path):
        (tmp_path / 'notes.txt').write_text('x', encoding='utf-8')
        with pytest.raises(FileNotFoundError, match='no viewer HTML'):
            resolve_viewer_page(str(tmp_path))

    def test_a_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_viewer_page(str(tmp_path / 'nope'))


class TestManifestReader:
    def test_found_from_either_a_page_or_a_folder(self, run):
        run_dir, page = run
        expected = read_visualization_manifest(str(run_dir))
        assert read_visualization_manifest(str(page)) == expected
        assert read_visualization_manifest(
            str(run_dir / VISUALIZATION_MANIFEST_NAME)) == expected
        assert expected['canonical_page'] == 'scene.html'

    def test_legacy_runs_simply_have_no_manifest(self, tmp_path):
        assert read_visualization_manifest(str(tmp_path)) == {}
        page = scene(tmp_path, saveas='legacy')
        assert read_visualization_manifest(str(page)) == {}

    def test_unreadable_manifest_is_tolerated(self, run):
        run_dir, _page = run
        (run_dir / VISUALIZATION_MANIFEST_NAME).write_text('{not json',
                                                           encoding='utf-8')
        assert read_visualization_manifest(str(run_dir)) == {}


class TestPlanFromHtml:
    def test_each_granularity_groups_the_stored_traces(self, run):
        _run_dir, page = run
        (entries, background, unlabeled), roles = profile_plan_from_html(
            str(page))
        assert unlabeled == [] and background == [2]
        assert set(entries) == {'KC1a'}
        assert sorted(entries['KC1a']) == [0, 1]
        for granularity, expected in (
                ('layer', {'KC layer'}),
                ('type', {'KC1a'}),
                ('body', {'KC layer__KC1a__256_aMe4',
                          'KC layer__KC1a__257_aMe5'})):
            (plan, _bg, _un), _roles = profile_plan_from_html(
                str(page), granularity)
            assert set(plan) == expected, granularity
        assert [r.role for r in roles] == ['neuron', 'neuron', 'mesh']

    def test_unknown_granularity_is_rejected(self, run):
        _run_dir, page = run
        with pytest.raises(ValueError, match='granularity must be one of'):
            profile_plan_from_html(str(page), 'instance')

    def test_colliding_layer_names_split_by_the_stamped_index(self):
        """Two layers whose smart names collide must not merge into one
        profile: ``layer_index`` is the only identity that tells them apart.
        """
        vis = bare()
        fig = go.Figure()
        for body_id, group, layer in (('256', 'LH', 0), ('257', 'LH', 1),
                                      ('258', 'MB', 0)):
            trace = go.Scatter3d(x=[0, 100], y=[0, 10], z=[0, 10],
                                 mode='lines', name=group, legendgroup=group)
            vis._stamp_trace_identity(trace, kind='neuron', group=group,
                                      type_label=f'T{body_id}',
                                      item=f'{group}__{body_id}',
                                      body_id=body_id, layer_index=layer)
            fig.add_trace(trace)
        roles = vs.classify_traces(list(fig.data))
        entries, _bg, _un = VisualizeSkeleton._build_profile_plan(
            roles, 'layer')
        assert entries == {'LH__L0': [0], 'LH__L1': [1], 'MB': [2]}

    def test_indexless_traces_never_join_a_layer_they_cannot_prove(self):
        """Legacy traces carry no layer_index, so a colliding group leaves
        them under the plain name instead of guessing a layer; a purely
        legacy page (no index anywhere) merges exactly as it always did.
        """
        vis = bare()
        fig = go.Figure()
        for body_id, layer in (('256', 0), ('257', 0), ('258', None)):
            trace = go.Scatter3d(x=[0, 100], y=[0, 10], z=[0, 10],
                                 mode='lines', name='LH', legendgroup='LH')
            vis._stamp_trace_identity(trace, kind='neuron', group='LH',
                                      type_label='T', item=f'LH__{body_id}',
                                      body_id=body_id, layer_index=layer)
            fig.add_trace(trace)
        roles = vs.classify_traces(list(fig.data))
        entries, _bg, _un = VisualizeSkeleton._build_profile_plan(
            roles, 'layer')
        assert entries == {'LH__L0': [0, 1], 'LH': [2]}

        plain = [go.Scatter3d(x=[0, 100], y=[0, 10], z=[0, 10], mode='lines',
                              name='LH', legendgroup='LH')]
        legacy_roles = vs.classify_traces(plain)
        entries, _bg, _un = VisualizeSkeleton._build_profile_plan(
            legacy_roles, 'layer')
        assert entries == {'LH': [0]}


class TestExportFromHtml:
    @pytest.fixture(autouse=True)
    def capture(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            vs, 'export_individuals_webdriver',
            lambda **kwargs: calls.append(kwargs) or {'png': ['shot.png']})
        self.calls = calls

    def test_plan_and_cameras_come_from_the_page_and_manifest(self, run):
        run_dir, page = run
        result = export_individuals_from_html(str(run_dir), views=['FRONT'])
        assert result == {'png': ['shot.png']}
        kwargs = self.calls[0]
        assert kwargs['html_path'] == str(page)
        assert kwargs['legend_entries'] == {'KC1a': [0, 1]}
        assert kwargs['background_indices'] == [2]
        assert kwargs['total_traces'] == 3
        assert kwargs['views'] == ['front']
        assert kwargs['view_cameras'] == {
            'front': {'eye': {'x': 1, 'y': 0, 'z': 0},
                      'up': {'x': 0, 'y': 0, 'z': 1},
                      'center': {'x': 0, 'y': 0, 'z': 0}}}
        assert kwargs['background_color'] == 'white'
        assert kwargs['auto_crop'] is True

    def test_default_output_folder_sits_beside_the_page(self, run):
        run_dir, page = run
        export_individuals_from_html(str(page))
        expected = str(run_dir / f'{page.stem}_profiles')
        assert self.calls[0]['output_dir'] == expected
        assert Path(expected).is_dir()

    def test_granularity_reaches_the_plan(self, run):
        _run_dir, page = run
        export_individuals_from_html(str(page), granularity='body',
                                     output_dir=str(page.parent / 'out'))
        assert sorted(self.calls[0]['legend_entries']) == [
            'KC layer__KC1a__256_aMe4', 'KC layer__KC1a__257_aMe5']

    def _with_recorded_roles(self, run_dir, kinds):
        """Rewrite the run manifest's role table to claim ``kinds``."""
        path = run_dir / VISUALIZATION_MANIFEST_NAME
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest['traces'] = [{'index': index, 'kind': kind}
                              for index, kind in enumerate(kinds)]
        path.write_text(json.dumps(manifest), encoding='utf-8')
        return manifest

    def test_a_page_whose_stamps_did_not_survive_says_so(self, run, capsys):
        """The recorded role table is the re-export's only round-trip check.

        A page that loses its ``meta`` stamps still opens and still profiles --
        just one level coarser -- so without this the failure is
        indistinguishable from an honest legacy run. It stays advisory: the
        export still runs on what the page can prove.
        """
        run_dir, page = run
        self._with_recorded_roles(
            run_dir, ['neuron', 'neuron', 'companion', 'site'])
        export_individuals_from_html(str(page))
        warned = capsys.readouterr().out
        assert '1 companion -> 0' in warned and '1 site -> 0' in warned
        assert '0 mesh -> 1' in warned
        assert 'round trip' in warned
        assert self.calls, 'the drift check must not block the export'

    def test_recorded_roles_that_match_keep_the_log_clean(self, run, capsys):
        run_dir, page = run
        self._with_recorded_roles(
            run_dir, ['neuron', 'neuron', 'mesh'])
        export_individuals_from_html(str(page))
        assert 'round trip' not in capsys.readouterr().out

    def test_a_manifest_less_page_still_moves_the_camera_per_view(
            self, tmp_path):
        """The preset table must arrive in the exporter's own key casing.

        ``export_individuals_webdriver`` looks a view up by its filename-style
        key, so the 'Front'-style keys the table defaults to resolved to no
        camera and a real 30-page run produced byte-identical front/top PNGs.
        """
        page = scene(tmp_path, saveas='legacy')
        export_individuals_from_html(str(page), views=['front', 'top'],
                                     output_dir=str(tmp_path / 'out'))
        cameras = self.calls[0]['view_cameras']
        assert self.calls[0]['views'] == ['front', 'top']
        assert {'front', 'top'} <= set(cameras)
        assert cameras['front']['eye'] != cameras['top']['eye']

    def test_every_canonical_view_gets_its_own_camera(self, tmp_path):
        """'left' and 'right' are separate views, and 'all' is the table's set.

        A name the camera table does not carry renders from the page's own
        framing, so an alias invented in the UI would land in the output
        folder as a second view that is actually the first one.
        """
        page = scene(tmp_path, saveas='legacy')
        export_individuals_from_html(str(page), views=['Left', 'right'],
                                     output_dir=str(tmp_path / 'a'))
        assert self.calls[0]['views'] == ['left', 'right']
        left = self.calls[0]['view_cameras']['left']['eye']
        right = self.calls[0]['view_cameras']['right']['eye']
        assert left != right
        export_individuals_from_html(str(page), views=['all'],
                                     output_dir=str(tmp_path / 'b'))
        assert self.calls[1]['views'] == list(self.calls[1]['view_cameras'])

    def test_a_view_the_page_cannot_render_is_refused_not_faked(self, tmp_path):
        page = scene(tmp_path, saveas='legacy')
        with pytest.raises(ValueError, match='no renderable view'):
            export_individuals_from_html(str(page), views=['nowhere'],
                                         output_dir=str(tmp_path / 'out'))
        assert self.calls == []

    def test_an_over_budget_profile_run_renders_what_fits_and_says_the_rest(
            self, run, monkeypatch, capsys):
        """A whole-brain export must deliver the first pages, not nothing.

        One profile is one full-scene render, so the count that matters is
        groups x views; and because a partial folder is easy to mistake for a
        complete one, the message has to name the capped count and the way out
        (split the scene, go coarser, request fewer views).
        """
        _run_dir, page = run
        monkeypatch.setattr(vs, 'MAX_INDIVIDUAL_PROFILES', 1)
        result = export_individuals_from_html(
            str(page), granularity='body', views=['front'],
            output_dir=str(page.parent / 'capped'))
        assert result == {'png': ['shot.png']}, 'the capped run still renders'
        kept = self.calls[0]['legend_entries']
        assert len(kept) == 1 and list(kept)[0].endswith('256_aMe4')
        warned = capsys.readouterr().out
        assert 'capped' in warned and 'skipped 1' in warned
        assert 'Separate the plots' in warned

    def test_the_budget_counts_groups_times_views(self, monkeypatch):
        monkeypatch.setattr(vs, 'MAX_INDIVIDUAL_PROFILES', 10)
        both = {'a': [0], 'b': [1]}
        assert vs.profile_budget(both, ['front', 'top']) == (both, None)
        six = {str(i): [i] for i in range(6)}
        kept, warning = vs.profile_budget(six, ['front', 'top'])
        assert warning and kept == {str(i): [i] for i in range(5)}
        assert vs.profile_budget({}, ['front']) == ({}, None)

    def test_legacy_page_falls_back_to_the_pages_dataset_cameras(
            self, tmp_path, monkeypatch):
        page = scene(tmp_path, saveas='legacy')
        seen = {}

        def cameras(dataset, brain_mesh=None, **kwargs):
            seen['args'] = (dataset, brain_mesh)
            seen['kwargs'] = kwargs
            return {'front': {'eye': {'x': 0, 'y': 1, 'z': 0}}}

        monkeypatch.setattr(vs, 'dataset_view_cameras', cameras)
        export_individuals_from_html(str(page), views=['front'])
        assert seen['args'] == ('hemibrain:v1.2.1', None)
        assert seen['kwargs'] == {'lowercase': True}
        assert 'front' in self.calls[0]['view_cameras']

    def test_a_page_of_only_background_profiles_nothing_but_does_not_fail(
            self, tmp_path):
        """A pure ROI/mesh run is valid input, and its video still exports.

        Raising here used to abort a combined re-export before its video phase,
        which threw away a rotating mesh video the page could have rendered.
        """
        fig = go.Figure()
        mesh = go.Mesh3d(x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0], i=[0], j=[1],
                         k=[2], name='Brain mesh')
        mesh.legendrank = 2e8
        fig.add_trace(mesh)
        path = tmp_path / 'empty.html'
        bare()._write_plotly_html(fig, str(path), include_plotlyjs=False)
        result = export_individuals_from_html(str(path))
        assert result['success'] is True
        assert result['files'] == {}
        assert self.calls == [], 'no session for a scene with nothing to profile'
        assert not (tmp_path / 'empty_profiles').exists()

    def test_invalid_granularity_is_rejected_before_any_io(self, run, tmp_path):
        with pytest.raises(ValueError, match='granularity must be one of'):
            export_individuals_from_html(
                str(tmp_path), granularity='nope')
        assert self.calls == []


class TestPythonAndJsClassifiersAgree:
    """``classify_traces`` reads ``meta.drocatTrace`` while the page's tree
    (``buildModel``) reads ``meta.drocatLegend`` -- two stamps written by
    different code paths, so only running both over the same traces catches
    a silent drift. The test executes the panel's own emitted buildModel
    under node; machines without node skip it.
    """

    @pytest.fixture(autouse=True)
    def _require_node(self):
        if shutil.which('node') is None:
            pytest.skip('node is not installed')

    @staticmethod
    def _harness(panel):
        """The emitted buildModel plus a small reporter, as one JS program."""
        start = panel.index('function buildModel(data) {')
        end = panel.index('\n  }', start) + len('\n  }')
        build = panel[start:end]
        rank = re.search(r'meshRankBase"?\s*:\s*([0-9.eE+-]+)', panel)
        assert rank, 'meshRankBase missing from the tree panel config'
        assert float(rank.group(1)) == float(vs.ROI_MESH_LEGEND_RANK_BASE), \
            'the JS mesh rank band and the Python one have drifted'
        return f"""
const CONFIG = {{ meshRankBase: {rank.group(1)} }};
function traceColor() {{ return null; }}
{build}
const model = buildModel(JSON.parse(require('fs').readFileSync(0, 'utf8')));
const leaves = {{}};
for (const [g, obj] of Object.entries(model.groups)) {{
  for (const t of obj.typeOrder) {{
    for (const [item, idx] of Object.entries(obj.types[t].items)) {{
      for (const i of idx) {{ leaves[i] = ['custom', g, t, item]; }}
    }}
  }}
  for (const [item, idx] of Object.entries(obj.direct)) {{
    for (const i of idx) {{ leaves[i] = ['flat', g, null, item]; }}
  }}
  for (const s of obj.sites) {{
    leaves[s.index] = ['site', g, s.type, null];
  }}
}}
const meshes = [];
for (const m of model.meshOrder) meshes.push(...model.meshes[m].indices);
const synapses = [];
for (const s of model.synOrder) synapses.push(...model.synapses[s].indices);
const grouped = [];
for (const g of Object.values(model.groups)) grouped.push(...g.indices);
console.log(JSON.stringify({{leaves, meshes, synapses, grouped}}));
"""

    @staticmethod
    def _stamped(vis, *, kind, group, type_label, item, body_id, layer_index,
                 custom):
        """One neuron-family trace carrying both stamps the renderer writes."""
        trace = go.Scatter3d(x=[0, 100], y=[0, 10], z=[0, 10], mode='lines',
                             name=item, legendgroup=group)
        vis._stamp_trace_identity(trace, kind=kind, group=group,
                                  type_label=type_label, item=item,
                                  body_id=body_id, layer_index=layer_index)
        legend = {'kind': 'neuron', 'group': group, 'item': item,
                  'color': '#4c78a8'}
        if custom:
            legend.update({'type': type_label, 'customGroup': True})
        trace.meta = {**trace.meta, 'drocatLegend': legend}
        return trace

    def test_tree_rows_and_profiles_group_the_same_traces(self):
        vis = bare()
        traces = [
            # custom-group layer: two types, one with a soma companion
            self._stamped(vis, kind='neuron', group='KC', type_label='KC1a',
                          item='KC1a_256', body_id='256', layer_index=0,
                          custom=True),
            self._stamped(vis, kind='companion', group='KC',
                          type_label='KC1a', item='KC1a_256', body_id='256',
                          layer_index=0, custom=True),
            self._stamped(vis, kind='neuron', group='KC', type_label='KC1b',
                          item='KC1b_257', body_id='257', layer_index=0,
                          custom=True),
            # flat layer (no custom groups): the group doubles as the type
            self._stamped(vis, kind='neuron', group='LH', type_label='LH',
                          item='LHi_258', body_id='258', layer_index=1,
                          custom=False),
        ]
        site = go.Scatter3d(x=[1, 2], y=[1, 2], z=[1, 2], mode='markers',
                            name='site_256',
                            legendgroup='pre_post:syn:KC1a_256')
        vis._stamp_site_identity(site, group_layer=0, item='site_256',
                                 owner='KC1a_256', owner_body_id='256')
        site.meta = {**site.meta,
                     'drocatLegend': {'kind': 'site', 'group': 'KC1a_256',
                                      'item': 'site_256',
                                      'color': '#4c78a8'}}
        traces.append(site)
        traces.append(go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1],
                                   mode='markers', name='synapses KC',
                                   legendgroup='synapses KC'))
        mesh = go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode='lines',
                            name='Brain mesh', legendgroup='Brain mesh',
                            legendrank=vs.ROI_MESH_LEGEND_RANK_BASE)
        traces.append(mesh)
        traces.append(go.Scatter3d(x=[None], y=[None], z=[None],
                                   mode='markers', name='KC swatch',
                                   legendgroup='KC'))

        roles = vs.classify_traces(traces)
        by_index = {r.index: r for r in roles}
        payload = json.dumps([t.to_plotly_json() for t in traces])
        harness = self._harness(bare()._legend_tree_html())
        proc = subprocess.run(['node', '-e', harness], input=payload,
                              capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr
        js = json.loads(proc.stdout)

        assert sorted(js['grouped']) == sorted(
            r.index for r in roles
            if r.role in ('neuron', 'companion', 'site'))
        for index, (shape, group, type_label, item) in js['leaves'].items():
            role = by_index[int(index)]
            if shape == 'site':
                # the site rides under its owner's row, not its layer's
                assert role.role == 'site', index
                continue
            assert role.group == group, index
            assert role.item == item, index
            if shape == 'custom':
                assert role.type == type_label, index
        assert sorted(js['meshes']) == sorted(
            r.index for r in roles if r.rule == 'mesh_rank_band')
        assert sorted(js['synapses']) == sorted(
            r.index for r in roles if r.role == 'synapse')
        # the swatch belongs to neither side
        swatch = by_index[len(traces) - 1]
        assert swatch.role == 'legend_swatch'
        assert str(len(traces) - 1) not in js['leaves']
