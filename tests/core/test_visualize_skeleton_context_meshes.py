"""Context meshes are always in the page; the checkbox only sets visibility.

The brain/VNC selection used to decide whether the geometry existed at all, so
a viewer handed a brain-only male-cns export could never reveal the nerve cord.
These tests pin the new contract and, just as much, the three places that had
to learn about a mesh that is present but not shown: the frozen scene box, the
individual-profile background list, and the legend tree's restore.

Run:
    conda run -n drocat-4.5.0 python -m pytest \
        tests/core/test_visualize_skeleton_context_meshes.py -q
"""

import json
import re
import sys
from pathlib import Path

import plotly.graph_objects as go
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visualize_skeleton import (  # noqa: E402
    BRAIN_MESH_LEGEND_RANK,
    CONTEXT_MESH_EMBEDDED_TARGET_FACES,
    ROI_MESH_LEGEND_RANK_BASE,
    VNC_MESH_LEGEND_RANK,
    VisualizeSkeleton,
    _is_unshown_context_mesh,
    _unshown_context_mesh_indices,
    classify_traces,
)

# The renderer's own range helper, reached as the tests reach it.
_scene_data_ranges_for_test = VisualizeSkeleton._scene_data_ranges


def scatter(x, y, z, **kwargs):
    return go.Scatter3d(x=x, y=y, z=z, mode='lines', **kwargs)


def mesh(x, y, z, **kwargs):
    return go.Mesh3d(x=x, y=y, z=z, i=[0, 0, 0], j=[1, 1, 1], k=[2, 2, 2],
                     **kwargs)


def figure(*traces):
    return go.Figure(data=list(traces))


def make_vis(**attrs):
    """A VisualizeSkeleton carrying only what the context-mesh path reads."""
    vis = object.__new__(VisualizeSkeleton)
    vis.backend = 'plotly'
    vis.dataset = 'male-cns:v1.0'
    vis.brain_mesh = 'native'
    vis.vnc_mesh = False
    vis.fig_3d = go.Figure()
    vis.exportable_meshes = []
    vis._ctx_halves = None
    vis._outline_vnc_volume = None
    vis._vprint_calls = []

    def _vprint(*args, **kwargs):
        vis._vprint_calls.append(' '.join(str(a) for a in args))

    vis._vprint = _vprint
    vis._suppress_output = lambda: _NullCtx()
    vis._get_effective_mesh_color = lambda role: 'rgba(1, 2, 3, 0.5)'
    vis._apply_plotly_trace_color = lambda trace, color: None
    for key, value in attrs.items():
        setattr(vis, key, value)
    return vis


def _writer_vis(**attrs):
    """A vis that can run ``_write_plotly_html`` end to end.

    The writer is the only place that decides whether a page gets a second
    box, so those tests go through it rather than reimplementing its rule.
    """
    merged = dict(verbose=False, html_theme_toggle=False, legend_mode='tree',
                  freeze_view=True, save_folder='')
    merged.update(attrs)
    vis = make_vis(**merged)
    vis._in_page_warning_html = lambda: ''
    return vis


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeVolume:
    """Stands in for a navis.Volume; only its identity is asserted on."""

    def __init__(self, name='vol'):
        self.name = name


@pytest.fixture
def fake_plot3d(monkeypatch):
    """Replace navis.plot3d so the test is about our fields, not navis.

    The trace is a real Mesh3d: the figure itself rejects anything else, and
    pretending otherwise would test the stub rather than the emit path.
    """
    import visualize_skeleton as vs

    made = []

    def _plot3d(volume, backend='plotly', **kwargs):
        trace = mesh([0, 1, 2], [0, 1, 2], [0, 1, 2])
        made.append((volume, trace))
        return type('F', (), {'data': [trace]})()

    monkeypatch.setattr(vs.navis, 'plot3d', _plot3d)
    return made


class TestUnshownContextMeshPredicate:
    def test_a_hidden_mesh_band_trace_is_the_only_match(self):
        assert _is_unshown_context_mesh(
            mesh([0, 1, 2], [0, 1, 2], [0, 1, 2],
                 legendrank=VNC_MESH_LEGEND_RANK, visible=False))
        assert _is_unshown_context_mesh(
            mesh([0, 1, 2], [0, 1, 2], [0, 1, 2],
                 legendrank=BRAIN_MESH_LEGEND_RANK, visible=False))

    def test_shown_and_legendonly_meshes_still_count(self):
        # legendonly is not "embedded for later": the requested path never
        # emits it, and treating it as hidden would shrink the box mid-session.
        for visible in (True, 'legendonly', None):
            assert not _is_unshown_context_mesh(
                mesh([0, 1, 2], [0, 1, 2], [0, 1, 2],
                     legendrank=VNC_MESH_LEGEND_RANK, visible=visible))

    def test_hidden_neurons_and_roi_meshes_keep_their_old_treatment(self):
        assert not _is_unshown_context_mesh(
            scatter([0, 1], [0, 1], [0, 1], visible=False))
        assert not _is_unshown_context_mesh(
            mesh([0, 1, 2], [0, 1, 2], [0, 1, 2],
                 legendrank=ROI_MESH_LEGEND_RANK_BASE, visible=False))


class TestFrozenBoxSpansTheEmbeddedHalf:
    """The page has one box, and it already fits the half held hidden.

    This inverted on the user's round-4 call. Growing the box when a viewer
    reveals the envelope moves the view (in a 3D scene the ranges *are* the
    camera normalization), and holding the tight box instead draws nothing --
    0.3% of a real nerve cord falls inside a brain-only box. One box that fits
    everything makes both problems disappear, at a measured 1.46x of zoom-out
    on a brain-only page.
    """

    def test_a_hidden_vnc_still_sizes_the_box(self):
        brain_only = figure(mesh([0, 10], [0, 10], [0, 10],
                                 legendrank=BRAIN_MESH_LEGEND_RANK))
        with_embedded = figure(
            mesh([0, 10], [0, 10], [0, 10],
                 legendrank=BRAIN_MESH_LEGEND_RANK),
            mesh([0, 10], [0, 10], [900, 1200],
                 legendrank=VNC_MESH_LEGEND_RANK, visible=False))
        assert (_scene_data_ranges_for_test(with_embedded)['z'][1]
                > _scene_data_ranges_for_test(brain_only)['z'][1])

    def test_a_hidden_neuron_widens_it_the_same_way(self):
        # One rule for every hidden trace: revealing anything must not clip it.
        without = figure(scatter([0, 10], [0, 10], [0, 10]))
        with_hidden = figure(
            scatter([0, 10], [0, 10], [0, 10]),
            scatter([0, 10], [0, 10], [900, 1200], visible=False))
        assert (_scene_data_ranges_for_test(without)['z']
                != _scene_data_ranges_for_test(with_hidden)['z'])


class TestEmbedContextMesh:
    def test_it_emits_hidden_traces_and_skips_the_glb_registry(
            self, fake_plot3d):
        vis = make_vis()
        volume = _FakeVolume()
        assert vis._embed_context_mesh(
            volume, name='JRCFIB2022M (VNC)', rank=VNC_MESH_LEGEND_RANK,
            role='vnc') is True

        trace = fake_plot3d[0][1]
        assert trace.visible is False
        assert trace.name == 'JRCFIB2022M (VNC)'
        assert trace.legendrank == VNC_MESH_LEGEND_RANK
        assert trace.showlegend is True
        assert trace.hoverinfo == 'none'
        assert len(vis.fig_3d.data) == 1
        # export_3d_model writes out every registered mesh, so an unrequested
        # envelope must never reach that list.
        assert vis.exportable_meshes == []

    def test_none_volume_is_a_no_op(self, fake_plot3d):
        assert make_vis()._embed_context_mesh(
            None, name='x', rank=VNC_MESH_LEGEND_RANK, role='vnc') is False
        assert fake_plot3d == []

    def test_k3d_backend_is_left_alone(self, fake_plot3d):
        vis = make_vis(backend='k3d')
        assert vis._embed_context_mesh(
            _FakeVolume(), name='x', rank=VNC_MESH_LEGEND_RANK,
            role='vnc') is False

    def test_decimation_is_applied_and_a_failure_keeps_full_detail(
            self, monkeypatch):
        import numpy as np
        import visualize_skeleton as vs

        vis = make_vis()
        seen = {}

        class _Big:
            faces = np.zeros((CONTEXT_MESH_EMBEDDED_TARGET_FACES + 10, 3),
                             dtype=int)

        class _Reduced:
            faces = np.zeros((1000, 3), dtype=int)
            vertices = np.zeros((3000, 3), dtype=int)

        monkeypatch.setattr(vis, '_extract_trimesh', lambda v: _Big())

        def fake_simplify(tm, target_faces):
            seen['target'] = target_faces
            return _Reduced()

        monkeypatch.setattr(vis, '_simplify_mesh_open3d', fake_simplify)
        # navis.Volume would try to interpret the stub mesh for real.
        monkeypatch.setattr(vs.navis, 'Volume',
                            lambda m, name=None: f'volume:{name}')
        volume = _FakeVolume()
        out = vis._decimate_context_mesh(volume, label='JRCFIB2022M (VNC)')
        assert seen['target'] == CONTEXT_MESH_EMBEDDED_TARGET_FACES
        assert out == 'volume:JRCFIB2022M (VNC)_embedded'

        # A decimator that cannot deliver must not cost us the mesh.
        monkeypatch.setattr(vis, '_simplify_mesh_open3d',
                            lambda tm, target: None)
        assert vis._decimate_context_mesh(volume, label='x') is volume

    def test_a_small_mesh_is_embedded_untouched(self, monkeypatch):
        import numpy as np
        vis = make_vis()
        called = []
        monkeypatch.setattr(vis, '_extract_trimesh', lambda v: type(
            'M', (), {'faces': np.zeros((10, 3), dtype=int)})())
        monkeypatch.setattr(vis, '_simplify_mesh_open3d',
                            lambda *a: called.append(a) or None)
        volume = _FakeVolume()
        assert vis._decimate_context_mesh(volume, label='x') is volume
        assert called == []


class TestUnshownHalfSelection:
    def test_brain_only_run_embeds_the_vnc_half(self):
        vis = make_vis()
        brain, vnc = _FakeVolume('brain'), _FakeVolume('vnc')
        vis._ctx_halves = (brain, vnc, 'JRCFIB2022M (brain)',
                           'JRCFIB2022M (VNC)')
        vis.fig_3d = figure(mesh([0, 1], [0, 1], [0, 1],
                                 legendrank=BRAIN_MESH_LEGEND_RANK))
        embedded = []
        vis._embed_context_mesh = (
            lambda volume, *, name, rank, role: embedded.append((name, volume))
            or True)
        vis._embed_unshown_context_mesh()
        assert embedded == [('JRCFIB2022M (VNC)', vnc)]

    def test_a_run_that_shown_both_halves_embeds_nothing(self):
        vis = make_vis(vnc_mesh=True)
        vis._ctx_halves = (_FakeVolume(), _FakeVolume(), 'b', 'v')
        vis.fig_3d = figure(
            mesh([0, 1], [0, 1], [0, 1], legendrank=BRAIN_MESH_LEGEND_RANK),
            mesh([0, 1], [0, 1], [0, 1], legendrank=VNC_MESH_LEGEND_RANK,
                 visible=False),
        )
        # The visible=False above is only the test's way of marking the band as
        # covered; _context_meshes_shown reads ranks, not intent.
        called = []
        vis._embed_context_mesh = lambda *a, **k: called.append(a)
        vis._embed_unshown_context_mesh()
        assert called == []

    def test_mesh_free_page_embeds_both_halves_for_mcns(self, fake_plot3d):
        vis = make_vis(brain_mesh='none', vnc_mesh=False)
        halves = (_FakeVolume('b'), _FakeVolume('v'), 'JRCFIB2022M (brain)',
                  'JRCFIB2022M (VNC)')
        vis._native_context_halves = lambda: halves
        embedded = []
        vis._embed_context_mesh = (
            lambda volume, *, name, rank, role: embedded.append(name) or True)
        vis._embed_unshown_context_mesh()
        # Brain first, so the VNC stays the last trace as everywhere else.
        assert embedded == ['JRCFIB2022M (brain)', 'JRCFIB2022M (VNC)']

    def test_a_vnc_only_run_still_owes_the_viewer_a_brain_envelope(self):
        """'none' + VNC checked shows the cord; the brain half is still owed.

        The reachability test is about the scene's space, not about both halves
        being absent, so the half the checkbox did cover must not veto the
        other one.
        """
        vis = make_vis(brain_mesh='none', vnc_mesh=True)
        vis.fig_3d = figure(
            mesh([0, 1], [0, 1], [900, 901, 902], name='JRCFIB2022M (VNC)',
                 legendrank=VNC_MESH_LEGEND_RANK),
        )
        halves = (_FakeVolume('b'), _FakeVolume('v'), 'JRCFIB2022M (brain)',
                  'JRCFIB2022M (VNC)')
        vis._native_context_halves = lambda: halves
        embedded = []
        vis._embed_context_mesh = (
            lambda volume, *, name, rank, role: embedded.append(name) or True)
        vis._embed_unshown_context_mesh()
        assert embedded == ['JRCFIB2022M (brain)']

    def test_a_cross_template_scene_never_reaches_for_native_halves(self):
        """The native split lives in the dataset's own coordinates.

        A scene drawn in BANC/FAFB/male-cns space has its halves from the
        outline spec; falling back to the native template here would embed
        geometry in the wrong space, which is worse than embedding nothing.
        """
        vis = make_vis(brain_mesh='BANC', vnc_mesh=True)
        vis.fig_3d = figure(
            mesh([0, 1], [0, 1], [900, 901, 902], name='BANC (VNC)',
                 legendrank=VNC_MESH_LEGEND_RANK),
        )

        def _boom():
            raise AssertionError('must not load the dataset template')

        vis._native_context_halves = _boom
        vis._embed_unshown_context_mesh()

    def test_a_cross_template_scene_carries_its_own_split(self):
        """'BANC'/'male-cns' outlines bring their VNC half with them.

        The brain block stashes it on ``_outline_vnc_volume`` instead of
        ``_ctx_halves``, and it is already transformed into the scene's space --
        the only geometry such a page may embed.
        """
        vis = make_vis(brain_mesh='BANC', vnc_mesh=False)
        vis.fig_3d = figure(
            mesh([0, 1], [0, 1], [0, 1], name='BANC (brain)',
                 legendrank=BRAIN_MESH_LEGEND_RANK),
        )
        outline = _FakeVolume('outline-vnc')
        vis._outline_vnc_volume = outline
        vis._outline_vnc_name = 'BANC (VNC)'
        embedded = []
        vis._embed_context_mesh = (
            lambda volume, *, name, rank, role: embedded.append(
                (name, rank, role, volume)) or True)
        vis._embed_unshown_context_mesh()
        assert embedded == [('BANC (VNC)', VNC_MESH_LEGEND_RANK, 'vnc', outline)]

    def test_only_mcns_and_banc_have_two_halves_to_embed(self):
        assert make_vis(dataset='male-cns:v1.0')._context_halves_kind() \
            == 'mcns'
        assert make_vis(dataset='BANC:v1.1')._context_halves_kind() == 'banc'
        for dataset in ('fafb:v5.1', 'hemibrain:v1.2.1', 'optic-lobe:v1.0',
                        'manc:v0.0'):
            assert make_vis(dataset=dataset)._context_halves_kind() is None

    def test_k3d_never_embeds(self):
        vis = make_vis(backend='k3d')
        assert vis._context_halves_kind() is None


class TestProfilePlanExcludesTheHiddenHalf:
    def test_hidden_mesh_is_not_background(self):
        traces = [
            scatter([0, 1], [0, 1], [0, 1], name='12345_aMe12_L',
                    legendgroup='aMe12'),
            mesh([0, 1, 2], [0, 1, 2], [0, 1, 2], name='JRCFIB2022M (brain)',
                 legendrank=BRAIN_MESH_LEGEND_RANK),
            mesh([0, 1, 2], [0, 1, 2], [900, 901, 902],
                 name='JRCFIB2022M (VNC)', legendrank=VNC_MESH_LEGEND_RANK,
                 visible=False),
        ]
        roles = classify_traces(traces)
        entries, background, unlabeled = VisualizeSkeleton._build_profile_plan(
            roles)
        assert background == [1]                 # the shown brain mesh
        assert 2 not in background + unlabeled   # the embedded half is absent
        assert list(entries) == ['aMe12']

    def test_a_requested_vnc_is_still_background(self):
        traces = [
            scatter([0, 1], [0, 1], [0, 1], name='12345_aMe12_L',
                    legendgroup='aMe12'),
            mesh([0, 1, 2], [0, 1, 2], [0, 1, 2], name='JRCFIB2022M (VNC)',
                 legendrank=VNC_MESH_LEGEND_RANK),
        ]
        roles = classify_traces(traces)
        _entries, background, _unlabeled = \
            VisualizeSkeleton._build_profile_plan(roles)
        assert background == [1]


class TestTreeRestoreBaseline:
    """The tree's restore must return to the page's start, not to all-on."""

    def _tree_js(self):
        vis = make_vis(legend_mode='tree')
        vis.fig_3d = figure(
            scatter([0, 1], [0, 1], [0, 1], name='aMe12',
                    legendgroup='aMe12', legendrank=1),
            mesh([0, 1, 2], [0, 1, 2], [900, 901, 902],
                 name='JRCFIB2022M (VNC)', legendrank=VNC_MESH_LEGEND_RANK,
                 visible=False),
        )
        return vis._legend_tree_html()

    def test_snapshot_and_baseline_restore_are_emitted(self):
        js = self._tree_js()
        assert 'function snapshotInitialVisibility()' in js
        assert 'initialVisible[i] = isVisible(gd.data[i]);' in js
        # sync() is the boot hook: it runs before any row can be clicked.
        assert js.index('snapshotInitialVisibility();') > \
            js.index('function sync() {')
        assert 'if (initialVisible && initialVisible[i] === false)' in js

    def test_showall_no_longer_blanks_everything_visible(self):
        js = self._tree_js()
        body = js[js.index('function showAll() {'):js.index('function isolate(')]
        assert 'Plotly.restyle(gd, {visible: true}, managedIndices());' \
            not in body

    def test_the_two_all_buttons_stay_deliberately_different(self):
        """The header eye reveals what the restore puts back.

        ``toggleAll`` is every managed trace -- which on this page is also the
        context mesh the run embedded hidden -- while ``showAll`` is the
        opening baseline. Both are one-eyed buttons in the same panel, so the
        tip has to carry the difference rather than the code being flattened.
        """
        js = self._tree_js()
        assert 'including an envelope this page opens without' in js
        assert 'returns to how the page opened' in js

        toggle = js[js.index('function toggleAll() {'):
                    js.index('function addToggleRow')]
        assert '{visible: !on}, all' in toggle
        assert 'initialVisible' not in toggle
        show_all = js[js.index('function showAll() {'):
                      js.index('function isolate(')]
        assert 'initialVisible && initialVisible[i] === false' in show_all


class TestOneBoxBaking:
    """The page bakes one box, and it spans the half the run kept hidden.

    Two boxes meant a choice at click time, and either choice broke a promise:
    hold the tight one and the reveal draws nothing, switch to the wide one and
    the freeze moves the view. One box that fits both removes the choice. The
    writer therefore bakes the union, and the script gets no box machinery at
    all -- which is what these tests pin, because a leftover selector would
    silently reintroduce the switch.
    """

    @staticmethod
    def _figure(vnc_z):
        return figure(
            scatter([0, 10], [0, 10], [0, 10], name='aMe12'),
            mesh([0, 5, 10], [0, 5, 10], vnc_z, name='JRCFIB2022M (VNC)',
                 legendrank=VNC_MESH_LEGEND_RANK, visible=False),
        )

    def test_indices_point_at_the_hidden_half_in_trace_order(self):
        fig = figure(
            scatter([0, 1], [0, 1], [0, 1]),
            mesh([0, 1, 2], [0, 1, 2], [0, 1, 2],
                 name='JRCFIB2022M (brain)',
                 legendrank=BRAIN_MESH_LEGEND_RANK),
            mesh([0, 1, 2], [0, 1, 2], [900, 901, 902],
                 name='JRCFIB2022M (VNC)',
                 legendrank=VNC_MESH_LEGEND_RANK, visible=False),
            mesh([0, 1, 2], [0, 1, 2], [0, 1, 2], name='some ROI',
                 legendrank=ROI_MESH_LEGEND_RANK_BASE, visible=False),
        )
        assert _unshown_context_mesh_indices(fig) == [2]

    def test_the_baked_box_is_the_union(self):
        ranges = VisualizeSkeleton._scene_data_ranges(self._figure([900, 901, 902]))
        assert ranges['z'][1] > 902
        # The union is only in the axis the half actually extends; x and y are
        # the neurons' own, padded, so a cord below the brain cannot stretch
        # the scene sideways.
        assert ranges['x'] == pytest.approx(
            VisualizeSkeleton._scene_data_ranges(
                figure(scatter([0, 10], [0, 10], [0, 10], name='aMe12')))['x'])

    def test_the_script_carries_no_box_machinery(self):
        script = make_vis()._freeze_view_html(
            VisualizeSkeleton._scene_data_ranges(self._figure([900, 901])))
        baked = script.split('var CONFIG = ')[1].split('};')[0] + '}'
        assert list(json.loads(baked)) == ['ranges', 'hiddenHalf']
        for gone in ('syncBox', 'contextIsShown', 'activeRanges',
                     'CONFIG.reveal', '"reveal"'):
            assert gone not in script

    def test_a_written_page_bakes_the_spanning_box(self, tmp_path):
        vis = _writer_vis()
        path = tmp_path / 'plot.html'
        vis._write_plotly_html(self._figure([900, 901, 902]), str(path),
                               include_plotlyjs=False, freeze_view=True)
        baked = self._config(path.read_text())
        assert baked['ranges']['z'][1] > 902
        # The box spans a half the page never draws, so its centre is empty
        # space: the same write has to tell the script to open pivoted on the
        # half it shows.
        assert baked['hiddenHalf'] is True

    def test_a_page_that_draws_everything_it_bakes_is_not_re_pivoted(self, tmp_path):
        """No hidden half means the pivot is already the anatomy's midpoint.

        Moving it there anyway would be a framing change on every ordinary
        page, so the flag has to be read off the figure rather than assumed.
        """
        vis = _writer_vis()
        path = tmp_path / 'plot.html'
        shown = figure(
            scatter([0, 10], [0, 10], [0, 10], name='aMe12'),
            mesh([0, 5, 10], [0, 5, 10], [900, 901, 902],
                 name='JRCFIB2022M (VNC)', legendrank=VNC_MESH_LEGEND_RANK),
        )
        vis._write_plotly_html(shown, str(path), include_plotlyjs=False,
                               freeze_view=True)
        assert self._config(path.read_text())['hiddenHalf'] is False

    @staticmethod
    def _config(html):
        return json.loads(re.search(
            r'var CONFIG = (\{[^\n]*"ranges"[^\n]*\});', html).group(1))
        assert 'syncBox' not in html


class TestParametersTxtRecordsTheHiddenHalf:
    """``parameters.txt`` must not read as if the page held only what it shows.

    The ``Brain Mesh:`` / ``VNC Mesh:`` lines are composed during
    initialization, before any trace exists, so they cannot name the envelope
    the page went on to embed. The note is appended from the same hook that
    writes the manifest -- and writes nothing at all when there is nothing
    hidden, which is every FAFB / hemibrain / optic-lobe / MANC run.
    """

    PARAMS = ('[Visualization]\n'
              '  Brain Mesh:       native\n'
              '  VNC Mesh:         False\n'
              + '=' * 70 + '\n')

    def _vis(self, tmp_path, *traces):
        vis = make_vis()
        vis.save_folder = str(tmp_path)
        vis.fig_3d = figure(*traces)
        (tmp_path / 'parameters.txt').write_text(self.PARAMS,
                                                 encoding='utf-8')
        return vis

    def _text(self, tmp_path):
        return (tmp_path / 'parameters.txt').read_text(encoding='utf-8')

    def test_the_embedded_half_is_named_below_the_parameter_block(
            self, tmp_path):
        vis = self._vis(
            tmp_path,
            mesh([0, 1, 2], [0, 1, 2], [0, 1, 2], name='JRCFIB2022M (brain)'),
            mesh([0, 1, 2], [0, 1, 2], [900, 901, 902],
                 name='JRCFIB2022M (VNC)', legendrank=VNC_MESH_LEGEND_RANK,
                 visible=False),
        )
        vis._note_hidden_context_meshes()
        text = self._text(tmp_path)
        assert '[Context Meshes Embedded But Hidden]' in text
        assert 'JRCFIB2022M (VNC)' in text
        assert f'{CONTEXT_MESH_EMBEDDED_TARGET_FACES:,} faces' in text
        # Appended, not rewritten: what the run asked for is still there.
        assert 'Brain Mesh:       native' in text
        assert text.index('[Context Meshes') > text.index('Brain Mesh:')

    def test_a_page_with_nothing_hidden_is_left_byte_identical(self, tmp_path):
        vis = self._vis(tmp_path,
                        mesh([0, 1, 2], [0, 1, 2], [0, 1, 2],
                             name='JRCFIB2022M (brain)'))
        vis._note_hidden_context_meshes()
        assert self._text(tmp_path) == self.PARAMS

    def test_an_unwritable_note_never_costs_the_run_its_page(self, tmp_path):
        # save_folder pointing at a file, not a directory, is the cheap way to
        # make the append fail; the manifest hook swallows failures the same
        # way because a note about the page is not worth losing the page over.
        vis = self._vis(tmp_path,
                        mesh([0, 1, 2], [0, 1, 2], [900, 901, 902],
                             name='JRCFIB2022M (VNC)',
                             legendrank=VNC_MESH_LEGEND_RANK, visible=False))
        vis.save_folder = str(tmp_path / 'parameters.txt')
        vis._note_hidden_context_meshes()
        assert any('hidden context meshes' in c
                   for c in vis._vprint_calls)
