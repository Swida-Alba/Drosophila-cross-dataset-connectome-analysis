"""Offline tests for the isolated BANC render path.

Covers the render-space mapping, the BANC template branch, the source
resolver, and the BANC node-reduction rules — all with stubbed fetchers
and geometry helpers (no network, no CloudVolume).
"""
from pathlib import Path
import sys

import numpy as np
import navis
import pytest
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import visualize_skeleton  # noqa: E402
from visualize_skeleton import (  # noqa: E402
    _orbit_frame_camera,
    _rotation_camera,
    dataset_native_space,
    dataset_render_space,
    VisualizeSkeleton,
)

SWC = """1 0 0 0 0 2 -1
2 0 4 0 0 2 1
3 0 8 0 0 2 2
4 0 12 0 0 2 3
"""


class _nullcontext:
    """Minimal context manager for hermetic tests (no contextlib import
    churn in the assertion helpers)."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _make_visualizer(dataset='banc_v888', skeleton_mode='tube'):
    vs = object.__new__(VisualizeSkeleton)
    vs.dataset = dataset
    vs.skeleton_mode = skeleton_mode
    vs.skeleton_mesh_simplification = 0.95
    vs.soma_mesh_simplification = None
    vs.soma_region_radius = 20000
    vs.cache_neurons = True
    vs.banc_skeleton_resolution = 'l2'
    vs.script_path = PROJECT_ROOT
    vs.FAKE_CACHE = {}
    vs._vprint = lambda *a, **k: None
    # Geometry helpers become pass-through recorders.
    vs._simplify_mesh_fafb_fine = lambda mesh, target: mesh
    vs._simplify_mesh_vertex_clustering = lambda mesh, target: mesh
    vs._simplify_mesh_with_soma_awareness = lambda *a, **k: a[0]
    return vs


def _tree(resolution=None):
    neuron = navis.read_swc(__import__('io').StringIO(SWC))
    if resolution:
        neuron._drocat_banc_resolution = resolution
    return neuron


class TestRenderSpace:
    def test_banc_renders_in_native_space(self):
        assert dataset_render_space('banc_v888') == 'BANC'
        assert dataset_native_space('banc_v626') == 'BANC'

    def test_fafb_unchanged(self):
        assert dataset_render_space('flywire_FAFB_v783') == 'FLYWIRE'


class TestViewCameras:
    """Plan item C: the shared per-dataset camera table."""

    def test_banc_front_matches_plan_table(self):
        from visualize_skeleton import dataset_view_cameras

        cams = dataset_view_cameras('banc_v888')
        assert cams['Front']['eye'] == dict(x=0, y=-2.5, z=0)
        assert cams['Front']['up'] == dict(x=0, y=0, z=1)
        assert cams['Top']['eye'] == dict(x=0, y=0, z=2.5)
        # Posterior (+Y) renders upward, so the top view reads anterior
        # downward like every other family (BANC's anterior is -Y).
        assert cams['Top']['up'] == dict(x=0, y=1, z=0)
        assert cams['Bottom']['up'] == dict(x=0, y=-1, z=0)
        assert cams['Left']['eye'] == dict(x=2.5, y=0, z=0)
        assert cams['Right']['eye'] == dict(x=-2.5, y=0, z=0)

    def test_banc_lowercase_keys_for_png_export(self):
        from visualize_skeleton import dataset_view_cameras

        upper = dataset_view_cameras('banc_v626')
        lower = dataset_view_cameras('banc_v626', lowercase=True)
        assert set(lower) == {'front', 'back', 'top', 'bottom',
                              'left', 'right'}
        assert lower['front'] == upper['Front']

    def test_distance_scaling(self):
        from visualize_skeleton import dataset_view_cameras

        cams = dataset_view_cameras('banc_v888', distance=1.5)
        assert cams['Front']['eye'] == dict(x=0, y=-1.5, z=0)

    def test_other_families_unchanged(self):
        from visualize_skeleton import dataset_view_cameras

        assert (dataset_view_cameras('manc:v1.2.1')['Front']['eye']
                == dict(x=0, y=0, z=2.5))
        assert (dataset_view_cameras(
            'hemibrain:v1.2.1', brain_mesh='native')['Front']['eye']
            == dict(x=0, y=2.5, z=0))
        assert (dataset_view_cameras('male-cns:v1.0')['Front']['eye']
                == dict(x=0, y=0, z=-2.5))

    def test_every_family_tops_anterior_downward(self):
        """One orientation rule across all tables: Top reads posterior-up.

        The Front eye sits on the anterior side, so Top's up must point
        away from it (anterior downward) and Bottom's must point along it
        (the flip you get looking from the ventral side). BANC, MANC and
        native-template hemibrain each historically pointed Top's up along
        the anterior axis and read upside down next to the default family.
        """
        from visualize_skeleton import dataset_view_cameras

        cases = [
            ('hemibrain:v1.2.1', None),          # default table
            ('flywire_FAFB_v783', None),
            ('flywire_FAFB_v783', 'FAFB'),       # default-table selection
            ('banc_v888', None),
            ('banc_v888', 'BANC'),
            ('manc:v1.2.1', None),
            ('hemibrain:v1.2.1', 'native'),
            ('male-cns:v1.0', None),
        ]
        for dataset, brain_mesh in cases:
            cams = dataset_view_cameras(dataset, brain_mesh)
            front_eye = cams['Front']['eye']
            top_up = cams['Top']['up']
            bottom_up = cams['Bottom']['up']
            top_dot = sum(top_up[k] * front_eye[k] for k in 'xyz')
            bottom_dot = sum(bottom_up[k] * front_eye[k] for k in 'xyz')
            assert top_dot < 0, (dataset, brain_mesh, 'Top not anterior-down')
            assert bottom_dot > 0, (dataset, brain_mesh,
                                    'Bottom not anterior-up')
            assert all(top_up[k] == -bottom_up[k] for k in 'xyz'), \
                (dataset, brain_mesh, 'Top/Bottom up vectors not opposed')


class TestVideoOrbitCameras:
    """Video turntables derive every frame from the family's Front camera.

    The historical orbits hardcoded the default family's XZ plane with
    up (0,-1,0) and a MANC-only handedness flip, so BANC and native-template
    hemibrain videos rolled around the anterior-posterior axis from the
    wrong side.
    """

    def test_default_family_frames_match_the_historical_orbit(self):
        from visualize_skeleton import _orbit_frame_camera

        d = 2.5
        off = d * 0.01
        front_eye = {'x': 0, 'y': 0, 'z': -d}
        front_up = {'x': 0, 'y': -1, 'z': 0}
        for deg, c, s in ((0, 1.0, 0.0), (90, 0.0, 1.0), (180, -1.0, 0.0)):
            eye, up = _orbit_frame_camera(front_eye, front_up, 'xz', deg, d)
            assert eye == pytest.approx({'x': d * s, 'y': off, 'z': -d * c})
            assert up == pytest.approx({'x': 0, 'y': -1, 'z': 0})
            eye, up = _orbit_frame_camera(front_eye, front_up, 'yz', deg, d)
            assert eye == pytest.approx({'x': off, 'y': d * s, 'z': -d * c})
            assert up == pytest.approx({'x': 0, 'y': -c, 'z': -s})
            eye, up = _orbit_frame_camera(front_eye, front_up, 'xy', deg, d)
            assert eye == pytest.approx({'x': d * s, 'y': d * c, 'z': off})
            assert up == pytest.approx({'x': 0, 'y': 0, 'z': 1})

    def test_orbits_start_on_each_family_front_and_keep_its_up(self):
        from visualize_skeleton import _orbit_frame_camera, dataset_view_cameras

        d = 2.5
        for dataset, brain_mesh in (('banc_v888', None), ('manc:v1.2.1', None),
                                    ('hemibrain:v1.2.1', 'native'),
                                    ('male-cns:v1.0', None)):
            front = dataset_view_cameras(dataset, brain_mesh)['Front']
            eye, up = _orbit_frame_camera(front['eye'], front['up'],
                                          'xz', 0, d)
            # The 1% gimbal guard rides the up axis away from the front.
            assert all(abs(eye[k] - front['eye'][k]) <= d * 0.011
                       for k in 'xyz'), (dataset, eye)
            assert up == front['up'], dataset
            bottom = dataset_view_cameras(dataset, brain_mesh)['Bottom']
            eye, up = _orbit_frame_camera(front['eye'], front['up'],
                                          'yz', 90, d)
            # The vertical flip's 90-degree frame is the family's own bottom
            # view (the 1% gimbal guard rides the lateral axis).
            assert all(abs(eye[k] - bottom['eye'][k]) <= d * 0.011
                       for k in 'xyz'), (dataset, eye)
            assert up == pytest.approx(bottom['up']), (dataset, up)

    def test_rotation_camera_shim_defaults_to_the_default_family(self):
        from visualize_skeleton import _rotation_camera

        d = 2.5
        eye, up = _rotation_camera(0, 'xz', d)
        assert eye == {'x': 0, 'y': d * 0.01, 'z': -d}
        assert up == {'x': 0, 'y': -1, 'z': 0}


class TestSelectionCameraConsistency:
    """The camera gate and the scene render target share one predicate.

    Both used to decide "will this brain-mesh selection move the scene?"
    independently -- the scene against the live transform registry, the
    cameras against a hardcoded dataset-substring list -- so a registry
    change could leave the scene native while the cameras served the
    selection family: sideways views on every export of that run.
    """

    def test_cameras_follow_the_scene_fallback_when_the_bridge_is_gone(
            self, monkeypatch):
        import visualize_skeleton as vs

        monkeypatch.setattr(vs, '_selection_bridge_state',
                            lambda s, t: (False, 'mocked: bridge gone'))
        from visualize_skeleton import dataset_view_cameras
        cams = dataset_view_cameras('banc_v888', 'FAFB')
        native = dataset_view_cameras('banc_v888')
        assert cams['Front'] == native['Front']
        assert cams['Top'] == native['Top']

    def test_cameras_follow_a_reachable_selection(self, monkeypatch):
        import visualize_skeleton as vs

        monkeypatch.setattr(vs, '_selection_bridge_state',
                            lambda s, t: (True, ''))
        from visualize_skeleton import dataset_view_cameras
        cams = dataset_view_cameras('banc_v888', 'FAFB')
        default = dataset_view_cameras('flywire_FAFB_v783')
        assert cams['Front'] == default['Front']
        assert cams['Top'] == default['Top']

    def test_same_space_selection_never_queries_the_registry(
            self, monkeypatch):
        import visualize_skeleton as vs

        def boom(source, target):
            raise AssertionError('registry queried for a same-space selection')

        monkeypatch.setattr(vs, '_selection_bridge_state', boom)
        from visualize_skeleton import dataset_view_cameras
        cams = dataset_view_cameras('flywire_FAFB_v783', 'FAFB')
        assert cams['Front']['eye'] == {'x': 0, 'y': 0, 'z': -2.5}

    def test_unknown_native_space_keeps_the_substring_gate(self, monkeypatch):
        import visualize_skeleton as vs

        def raise_value_error(dataset):
            raise ValueError(f'no known native space for {dataset}')

        monkeypatch.setattr(vs, 'dataset_native_space', raise_value_error)
        from visualize_skeleton import dataset_view_cameras
        cams = dataset_view_cameras('fafb_curious:v1', 'FAFB')
        default = dataset_view_cameras('flywire_FAFB_v783')
        assert cams['Front'] == default['Front']

    def test_scene_render_target_shares_the_predicate(self, monkeypatch):
        import visualize_skeleton as vs

        monkeypatch.setattr(vs, '_selection_bridge_state',
                            lambda s, t: (False, 'mocked: bridge gone'))
        vis = object.__new__(VisualizeSkeleton)
        vis.brain_mesh = 'FAFB'
        vis._vprint = lambda *a, **k: None
        # The scene falls back to native, and the cameras follow it there.
        assert vis._scene_render_target('BANC', 'BANC') == 'BANC'
        from visualize_skeleton import dataset_view_cameras
        cams = dataset_view_cameras('banc_v888', 'FAFB')
        assert cams['Front'] == dataset_view_cameras('banc_v888')['Front']


class TestTemplateInfo:
    def test_banc_branch_skips_transform(self):
        vs = _make_visualizer()
        vs.brain_mesh = 'native'
        sentinel = object()
        vs._get_banc_template_volume = lambda: sentinel
        info = vs._get_template_info()
        assert info['source'] == 'BANC'
        assert info['target'] == 'BANC'
        assert info['skip_transform'] is True
        assert info['template_obj'] is sentinel

    def test_fafb_selection_moves_scene_to_flywire(self):
        vs = _make_visualizer()
        vs.brain_mesh = 'FAFB'
        info = vs._get_template_info()
        # An explicit cross-template selection moves the whole scene into
        # the selected template's space.
        assert info['source'] == 'BANC'
        assert info['target'] == 'FLYWIRE'
        assert info.get('skip_transform') is not True
        assert vs._needs_skeleton_transform() is True

    def test_transform_not_needed_for_banc(self):
        vs = _make_visualizer()
        vs.brain_mesh = 'native'
        vs._get_banc_template_volume = lambda: object()
        assert vs._needs_skeleton_transform() is False


class TestBancProcessor:
    def _run_tube(self, monkeypatch, pipeline='fast', resolutions=('l2', 'full')):
        calls = []
        monkeypatch.setattr(
            visualize_skeleton, 'simplify_skeleton_nodes',
            lambda n, factor: calls.append((getattr(n, 'id'), factor))
            or (_tree(), {'raw_nodes': 4, 'achieved_nodes': 1}))
        # A tiny but valid mesh stands in for the tube product.
        box = trimesh.creation.box(extents=(1, 1, 1))
        monkeypatch.setattr(
            navis.conversion, 'tree2meshneuron',
            lambda work, tube_points=6: navis.MeshNeuron(box))

        vs = _make_visualizer(skeleton_mode='tube')
        vs._vprint = lambda *a, **k: None
        neurons = [_tree(r) for r in resolutions]
        for i, n in enumerate(neurons):
            n.id = 1000 + i
        out, already = vs._process_banc_layer(
            navis.NeuronList(neurons), pipeline)
        return out, already, calls

    def test_fast_reduces_only_full_resolution(self, monkeypatch):
        out, already, calls = self._run_tube(
            monkeypatch, pipeline='fast', resolutions=('l2', 'full'))
        reduced = [c[0] for c in calls]
        # The L2 source keeps every node; only the full-res one is reduced.
        assert reduced == [1001]
        assert already is True
        assert all(isinstance(n, navis.MeshNeuron) for n in out)

    def test_fine_never_reduces(self, monkeypatch):
        out, already, calls = self._run_tube(
            monkeypatch, pipeline='fine', resolutions=('l2', 'full'))
        assert calls == []

    def test_line_reduces_only_full_resolution(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            visualize_skeleton, 'simplify_skeleton_nodes',
            lambda n, factor: calls.append((getattr(n, 'id'), factor))
            or (n, {'raw_nodes': 4, 'achieved_nodes': 4}))
        vs = _make_visualizer(skeleton_mode='line')
        l2, full = _tree('l2'), _tree('full')
        l2.id, full.id = 1, 2
        out, already = vs._process_banc_layer(
            navis.NeuronList([l2, full]), 'fast')
        assert already is False
        # Reduction is attempted for the full-res source only; the L2
        # source keeps every node untouched.
        assert calls == [(2, visualize_skeleton.BANC_LINE_FULL_NODE_REDUCTION)]
        assert {n.id: len(n.nodes) for n in out} == {1: 4, 2: 4}

    def test_faces_l2_excess_decimation(self, monkeypatch):
        """L2 tubes are the cache-level product, already ~90% simplified:
        the asked simplification only touches them through the EXCESS above
        the 0.90 baseline, scaled onto the L2 density (0.95 -> 50% of L2
        faces removed).  Full-resolution sources are decimated to the
        slider target with the 4,000-face safety floor."""
        dense = trimesh.creation.icosphere(subdivisions=4)  # 5120 faces
        assert len(dense.faces) == 5120
        decimate_calls = []
        monkeypatch.setattr(
            visualize_skeleton, 'simplify_skeleton_nodes',
            lambda n, factor: (n, {'raw_nodes': 4, 'achieved_nodes': 4}))
        monkeypatch.setattr(
            navis.conversion, 'tree2meshneuron',
            lambda work, tube_points=6: navis.MeshNeuron(dense))

        l2, full = _tree('l2'), _tree('full')
        l2.id, full.id = 1, 2

        vs = _make_visualizer(skeleton_mode='tube')
        vs._vprint = lambda *a, **k: None
        vs._simplify_mesh_fafb_fine = (
            lambda mesh, target: decimate_calls.append(target) or mesh)
        out, already = vs._process_banc_layer(
            navis.NeuronList([l2, full]), 'fast')
        assert already is True
        # asked 0.95 -> L2 excess = (0.95-0.90)/0.10 = 50% removed:
        # max(100, int(5120 * 0.50)) = 2560.  Full: floored 4000 target.
        by_id = {n.id: n for n in out}
        assert decimate_calls == [2560, 4000]
        assert len(by_id[1].faces) == 5120  # pass-through decimator
        assert len(by_id[2].faces) == 5120

    def test_faces_l2_untouched_at_baseline(self, monkeypatch):
        """Asked simplification at or below the 0.90 L2 baseline leaves
        L2 tubes as-is."""
        dense = trimesh.creation.icosphere(subdivisions=4)  # 5120 faces
        decimate_calls = []
        monkeypatch.setattr(
            visualize_skeleton, 'simplify_skeleton_nodes',
            lambda n, factor: (n, {'raw_nodes': 4, 'achieved_nodes': 4}))
        monkeypatch.setattr(
            navis.conversion, 'tree2meshneuron',
            lambda work, tube_points=6: navis.MeshNeuron(dense))

        l2, full = _tree('l2'), _tree('full')
        l2.id, full.id = 1, 2

        vs = _make_visualizer(skeleton_mode='tube')
        vs.skeleton_mesh_simplification = 0.90
        vs._vprint = lambda *a, **k: None
        vs._simplify_mesh_fafb_fine = (
            lambda mesh, target: decimate_calls.append(target) or mesh)
        out, already = vs._process_banc_layer(
            navis.NeuronList([l2, full]), 'fast')
        by_id = {n.id: n for n in out}
        # L2: no decimation call, 5120 faces kept.  Full: 0.90 -> int(5120
        # * 0.10) = 512, lifted to the 4,000-face full-res safety floor.
        assert decimate_calls == [4000]
        assert len(by_id[1].faces) == 5120
        assert len(by_id[2].faces) == 5120

    def test_faces_l2_excess_scales_to_ninety_nine(self, monkeypatch):
        """asked 0.99 -> L2 excess 90% removed (10% of faces kept)."""
        dense = trimesh.creation.icosphere(subdivisions=4)  # 5120 faces
        decimate_calls = []
        monkeypatch.setattr(
            visualize_skeleton, 'simplify_skeleton_nodes',
            lambda n, factor: (n, {'raw_nodes': 4, 'achieved_nodes': 4}))
        monkeypatch.setattr(
            navis.conversion, 'tree2meshneuron',
            lambda work, tube_points=6: navis.MeshNeuron(dense))

        l2 = _tree('l2')
        l2.id = 1

        vs = _make_visualizer(skeleton_mode='tube')
        vs.skeleton_mesh_simplification = 0.99
        vs._vprint = lambda *a, **k: None
        vs._simplify_mesh_fafb_fine = (
            lambda mesh, target: decimate_calls.append(target) or mesh)
        vs._process_banc_layer(navis.NeuronList([l2]), 'fast')
        assert decimate_calls == [max(100, int(5120 * 0.10))]


class TestBancRoiFromMaleCns:
    """BANC scenes have no named ROI product of their own: named ROIs are
    fetched from male-cns (JRCFIB2022Mraw) and bridged into BANC space,
    cached under meshes_transformed/BANC/ — mirroring the FAFB handling."""

    def _run_plot_mesh(self, tmp_path, monkeypatch, roi='AL(R)',
                       segid_map=None):
        import trimesh as _tm
        import navis.interfaces.neuprint as neu
        from utils import token_manager

        vs = _make_visualizer()
        vs.script_path = str(tmp_path)
        vs.brain_mesh = 'native'
        vs.mesh_roi = [roi]
        vs.mesh_color = '#94a3b8'
        vs.mesh_alpha = 0.1
        vs.background_color = 'rgba(255, 255, 255, 1.0)'
        vs.FAFB_template_correction = True
        vs.roi_mesh_simplification = 0  # keep the 1-face test mesh intact
        vs.verbose = False
        vs.backend = 'plotly'
        import plotly.graph_objects as _go
        vs.fig_3d = _go.Figure()
        vs.token = 'tok'
        vs._vprint = lambda *a, **k: None
        vs._suppress_output = lambda: _nullcontext()
        vs._apply_plotly_trace_color = lambda trace, color: None

        # The male-cns ROI mesh sits at a JRCFIB2022Mraw-voxel-style
        # location (Z near zero) so the bridging into BANC space moves it.
        verts = np.array([[12000.0, 13000.0, 5000.0],
                          [12100.0, 13100.0, 5100.0],
                          [11900.0, 13200.0, 5200.0]])
        mesh_vol = navis.Volume(
            _tm.Trimesh(vertices=verts,
                        faces=np.array([[0, 1, 2]], dtype=np.int64)),
            name=roi)
        monkeypatch.setattr(neu, 'fetch_roi', lambda *a, **k: mesh_vol)
        monkeypatch.setattr(
            'neuprint.Client', lambda *a, **k: object(), raising=False)
        monkeypatch.setattr(
            token_manager.TokenManager, 'get_neuprint_token',
            lambda self, *a, **k: 'tok')
        segids = segid_map or {}
        vs._get_banc_segid = lambda roi_name: segids.get(roi_name)

        rc = vs.plot_mesh()
        return rc, vs

    def test_named_roi_fetched_and_bridged(self, tmp_path, monkeypatch):
        rc, vs = self._run_plot_mesh(tmp_path, monkeypatch)
        assert rc == 0
        # The male-cns mesh was bridged into BANC space (nm) and cached
        # under meshes_transformed/BANC/.
        cache_dir = (Path(tmp_path) / 'cache' / 'banc_v888'
                     / 'meshes_transformed' / 'BANC')
        files = list(cache_dir.glob('*.json'))
        assert files, 'transformed ROI not cached'
        vol = navis.Volume.from_json(str(files[0]))
        zc = np.asarray(vol.vertices)[:, 2].mean()
        # JRCFIB2022Mraw Z ~5k nm -> BANC Z is hundreds of thousands of nm
        assert zc > 50_000

    def test_banc_aggregates_skip_transform(self, tmp_path, monkeypatch):
        import trimesh as _tm
        verts = np.array([[500000.0, 100000.0, 100000.0],
                          [501000.0, 101000.0, 100000.0],
                          [499000.0, 102000.0, 100000.0]])
        agg = navis.Volume(
            _tm.Trimesh(vertices=verts,
                        faces=np.array([[0, 1, 2]], dtype=np.int64)),
            name='BANC_neuropil')
        # Pre-stage the aggregate mesh the way _get_banc_region_volume
        # does (it caches the product as a side effect).
        mesh_dir = Path(tmp_path) / 'cache' / 'banc_v888' / 'meshes'
        mesh_dir.mkdir(parents=True, exist_ok=True)
        agg.to_json(str(mesh_dir / 'BANC__n_e_u_r_o_p_i_l.json'))
        rc, vs = self._run_plot_mesh(tmp_path, monkeypatch,
                                     roi='BANC_neuropil',
                                     segid_map={'BANC_neuropil': 2})
        assert rc == 0
        # No transformed cache entry: native aggregates need no bridging
        # (the directory itself may be created by cache-path helpers).
        tx_dir = (Path(tmp_path) / 'cache' / 'banc_v888' /
                  'meshes_transformed')
        assert not tx_dir.exists() or not any(tx_dir.rglob('*.json'))


class TestBancResolver:
    def test_resolver_uses_bucket_and_raw_cache(self, tmp_path, monkeypatch):
        import banc_public_data

        fetched = []

        def fake_fetch(dataset, body_id, resolution='l2', project_root=None,
                       use_cache=True):
            fetched.append((dataset, str(body_id), resolution, use_cache))
            return _tree('l2')

        monkeypatch.setattr(banc_public_data, "fetch_banc_swc", fake_fetch)

        vs = _make_visualizer()
        vs.script_path = str(tmp_path)
        vs._load_api_cached_skeletons = (
            lambda ids: ({}, list(ids)))

        sources, skeleton_cache = vs._resolve_banc_sources(
            [12345], use_cache=True)
        assert sources == {'12345': 'banc_gcs'}
        assert '12345' in skeleton_cache
        assert fetched == [('banc_v888', '12345', 'l2', True)]

    def test_resolver_raw_cache_wins(self, tmp_path, monkeypatch):
        import banc_public_data

        def fail_fetch(*a, **k):
            raise AssertionError("bucket fetch should not run on cache hit")

        monkeypatch.setattr(banc_public_data, "fetch_banc_swc", fail_fetch)

        vs = _make_visualizer()
        vs.script_path = str(tmp_path)
        cached = _tree()
        vs._load_api_cached_skeletons = (
            lambda ids: ({'12345': cached}, []))

        sources, skeleton_cache = vs._resolve_banc_sources([12345])
        assert sources == {'12345': 'raw_cache'}
        assert skeleton_cache['12345'] is cached

    def test_resolver_restores_resolution_provenance(self, tmp_path,
                                                     monkeypatch):
        """Plan item R2: cache reloads must recover the resolution.

        The shared raw cache strips in-memory attributes; the resolver
        restores them from the provenance header ('banc_gcs_full') so
        full-resolution neurons keep their pipeline stages.
        """
        import banc_public_data

        monkeypatch.setattr(
            banc_public_data, "fetch_banc_swc",
            lambda *a, **k: pytest.fail(
                "cache hits must not reach the bucket"))

        vs = _make_visualizer()
        vs.script_path = str(tmp_path)
        cached_full = _tree()   # no in-memory resolution attribute
        cached_full._drocat_source = 'banc_gcs_full'
        cached_l2 = _tree()
        cached_l2._drocat_source = 'banc_gcs_l2'
        cached_legacy = _tree()  # headerless legacy entry
        vs._load_api_cached_skeletons = lambda ids: (
            {'11': cached_full, '22': cached_l2, '33': cached_legacy}, [])

        _, skeleton_cache = vs._resolve_banc_sources(
            [11, 22, 33], use_cache=True)
        assert skeleton_cache['11']._drocat_banc_resolution == 'full'
        assert skeleton_cache['22']._drocat_banc_resolution == 'l2'
        assert skeleton_cache['33']._drocat_banc_resolution == 'l2'

    def test_missing_skeleton_is_skipped(self, tmp_path, monkeypatch):
        import banc_public_data

        monkeypatch.setattr(
            banc_public_data, "fetch_banc_swc",
            lambda *a, **k: None)
        vs = _make_visualizer()
        vs.script_path = str(tmp_path)
        vs._load_api_cached_skeletons = lambda ids: ({}, list(ids))
        sources, skeleton_cache = vs._resolve_banc_sources([999])
        assert sources == {} and skeleton_cache == {}


class TestBancOutlineProducts:
    """E4: the whole-CNS outline is cached as render (~100k faces) + full."""

    def _make_vs(self):
        vs = _make_visualizer()
        vs.script_path = str(PROJECT_ROOT)
        return vs

    def test_render_full_product_split(self, tmp_path, monkeypatch):
        import types

        vs = self._make_vs()
        # Synthetic 300k-face "outline" on a small vertex ring.
        n_faces = 300_000
        verts = np.array([[0, 0, 0], [1000, 0, 0], [0, 1000, 0]], dtype=float)
        rng = np.random.default_rng(0)
        faces = rng.integers(0, 3, size=(n_faces, 3))
        mesh = types.SimpleNamespace(vertices=verts, faces=faces)

        fake_cv_mod = types.ModuleType("cloudvolume")

        def _fake_get(segid, _mesh=mesh):
            return _mesh

        class FakeMesh:
            get = staticmethod(_fake_get)

        class FakeCV:
            def __init__(self, url, **kwargs):
                pass

            mesh = FakeMesh()

        fake_cv_mod.CloudVolume = FakeCV
        monkeypatch.setitem(sys.modules, "cloudvolume", fake_cv_mod)
        import trimesh
        small = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
            faces=[[0, 1, 2], [1, 3, 2], [0, 2, 1], [1, 2, 3]])
        monkeypatch.setattr(
            vs, "_simplify_mesh_open3d", lambda tm, target: small)
        # script_path must point at tmp so the cache writes stay isolated.
        vs.script_path = str(tmp_path)

        volume = vs._get_banc_region_volume("BANC_outline", segid=1)
        assert volume is not None
        assert len(volume.faces) == 4
        # Pair written: render product under the standard name, full next to
        # it — both under their PLAIN names (not the per-lowercase encoding).
        import json as _json
        import os
        mesh_dir = os.path.join(str(tmp_path), "cache", "banc_v888", "meshes")
        std = vs._banc_mesh_file_path(mesh_dir, "BANC_outline")
        full = vs._banc_mesh_file_path(mesh_dir, "BANC_outline_full")
        assert std.endswith("BANC_outline.json")
        assert full.endswith("BANC_outline_full.json")
        assert os.path.exists(std)
        assert os.path.exists(full)
        std_faces = len(_json.load(open(std))["faces"])
        full_faces = len(_json.load(open(full))["faces"])
        assert std_faces == 4 and full_faces == n_faces

    def test_self_heal_oversized_cached_outline(self, tmp_path, monkeypatch):
        import json as _json
        import os
        import types

        vs = self._make_vs()
        vs.script_path = str(tmp_path)
        # Pre-seed an oversized standard product (pre-E4 cache), under the
        # old per-lowercase-encoded file name.
        mesh_dir = vs._get_dataset_mesh_dir()
        os.makedirs(mesh_dir, exist_ok=True)
        rng = np.random.default_rng(1)
        verts = np.array([[0, 0, 0], [1000, 0, 0], [0, 1000, 0]], dtype=float)
        faces = rng.integers(0, 3, size=(200_000, 3))
        std = vs._banc_mesh_file_path(mesh_dir, "BANC_outline")
        with open(std, "w") as handle:
            _json.dump({"vertices": verts.tolist(), "faces": faces.tolist()}, handle)

        mesh = types.SimpleNamespace(vertices=verts, faces=faces)
        fake_cv_mod = types.ModuleType("cloudvolume")

        import trimesh
        small = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
            faces=[[0, 1, 2], [1, 3, 2], [0, 2, 1], [1, 2, 3]])

        def _fake_get(segid, _mesh=mesh):
            return _mesh

        class FakeMesh:
            get = staticmethod(_fake_get)

        class FakeCV:
            def __init__(self, url, **kwargs):
                pass

            mesh = FakeMesh()

        fake_cv_mod.CloudVolume = FakeCV
        monkeypatch.setitem(sys.modules, "cloudvolume", fake_cv_mod)
        monkeypatch.setattr(
            vs, "_simplify_mesh_open3d", lambda tm, target: small)

        volume = vs._get_banc_region_volume("BANC_outline", segid=1)
        # Self-healed: standard product decimated, full copy retained.
        assert len(volume.faces) == 4
        full = vs._banc_mesh_file_path(mesh_dir, "BANC_outline_full")
        assert os.path.exists(full)

    def test_clean_names_and_legacy_migration(self, tmp_path):
        """Aggregate products cache under their plain names (not the
        per-lowercase encoding), and cache files written with the old
        encoding are renamed in place on first load."""
        import json as _json
        import os

        vs = self._make_vs()
        vs.script_path = str(tmp_path)
        mesh_dir = vs._get_dataset_mesh_dir()
        os.makedirs(mesh_dir, exist_ok=True)
        payload = {"vertices": [[0, 0, 0]], "faces": [[0, 0, 0]]}

        legacy = os.path.join(
            mesh_dir, vs._roi_to_filename("BANC_neuropil"))
        with open(legacy, "w") as handle:
            _json.dump(payload, handle)
        clean = os.path.join(mesh_dir, "BANC_neuropil.json")
        assert not os.path.exists(clean)

        loaded = vs._load_banc_region_mesh_json("BANC_neuropil")
        assert loaded is not None
        assert os.path.exists(clean) and not os.path.exists(legacy)

        # Non-aggregate regions keep the generic case-safe encoding
        # (lowercase letters stay prefixed, e.g. for 'aL(L)' vs 'AL(L)').
        fine = vs._banc_mesh_file_path(mesh_dir, "aL(L)")
        assert os.path.basename(fine) == "_aL(L).json"


class TestSplitBancCns:
    def test_split_partitions_faces_at_neck(self, tmp_path):
        import navis
        import numpy as np
        from banc_public_data import BANC_BRAIN_VNC_Y_CUTOFF

        vs = _make_visualizer()
        vs._vprint = lambda *a, **k: None
        verts = np.array([
            [0.0, 0.0, 0.0],        # brain-side vertex
            [10.0, float(BANC_BRAIN_VNC_Y_CUTOFF - 1), 0.0],
            [20.0, float(BANC_BRAIN_VNC_Y_CUTOFF + 1), 0.0],
            [30.0, float(BANC_BRAIN_VNC_Y_CUTOFF + 2), 5.0],
        ])
        faces = np.array([
            [0, 1, 0],  # brain face (all y < cutoff)
            [2, 3, 2],  # VNC face (all y >= cutoff)
            [1, 2, 1],  # crossing face -> assigned by centroid
        ], dtype=np.int64)
        import trimesh
        volume = navis.Volume(trimesh.Trimesh(vertices=verts, faces=faces))
        brain, vnc = vs._split_banc_cns_volume(volume)
        # Centroid assignment: nothing is dropped (no crack), and the
        # crossing face lands on the brain side (its centroid is anterior).
        assert len(brain.faces) == 2 and len(vnc.faces) == 1
        assert len(brain.faces) + len(vnc.faces) == len(faces)

    def _two_sided_volume(self):
        """Small outline with faces strictly on both sides of the neck."""
        import navis
        import numpy as np
        from banc_public_data import BANC_BRAIN_VNC_Y_CUTOFF

        verts = np.array([
            [0.0, float(BANC_BRAIN_VNC_Y_CUTOFF - 10), 0.0],
            [5.0, float(BANC_BRAIN_VNC_Y_CUTOFF - 10), 5.0],
            [10.0, float(BANC_BRAIN_VNC_Y_CUTOFF + 10), 0.0],
            [15.0, float(BANC_BRAIN_VNC_Y_CUTOFF + 10), 5.0],
        ])
        faces = np.array([[0, 1, 0], [2, 3, 2]], dtype=np.int64)
        return navis.Volume(trimesh.Trimesh(vertices=verts, faces=faces))

    def test_template_info_names_brain_portion(self):
        """Plan item E2: template mode is labelled '(brain)'."""
        vs = _make_visualizer()
        vs.brain_mesh = 'native'
        vs._get_banc_template_volume = self._two_sided_volume
        info = vs._get_template_info()
        assert info['mesh_name'] == 'BANC (brain)'

    def test_banc_outline_spec(self):
        vs = _make_visualizer()
        vs._vprint = lambda *a, **k: None
        vs.brain_mesh = 'BANC'
        vs._get_banc_template_volume = self._two_sided_volume
        spec = vs._get_outline_template_info()
        assert spec['space'] == 'BANC'
        assert spec['split'] == 'banc'
        brain, vnc = vs._split_banc_cns_volume(spec['build']())
        assert brain is not None and vnc is not None

    def test_vnc_template_info_names_vnc_portion(self):
        """Plan item E3: the VNC branch is labelled '(VNC)' and carries
        only the posterior faces."""
        import numpy as np

        vs = _make_visualizer()
        vs._vprint = lambda *a, **k: None
        vs._get_banc_template_volume = self._two_sided_volume
        info = vs._get_vnc_template_info()
        assert info['mesh_name'] == 'BANC (VNC)'
        assert len(info['mesh'].faces) == 1


class TestOverlayDropWarning:
    """Plan item R6: overlays between spaces without a bridge are dropped
    with an explicit warning naming the dropped bodies."""

    def test_no_bridge_warning_lists_body_ids(self, capsys):
        from types import SimpleNamespace
        import flybrains  # noqa: F401 - the production transform registry

        flybrains.register_transforms()
        # 'NOT_A_TEMPLATE_SPACE' can never be registered, so the lookup
        # deterministically raises and takes the drop-with-warning path.
        neurons = [SimpleNamespace(id=111), SimpleNamespace(id=222)]
        out = visualize_skeleton.transform_neurons_to_space(
            neurons, 'FLYWIRE', 'NOT_A_TEMPLATE_SPACE')
        assert out == []
        output = capsys.readouterr().out
        assert '2 overlay neuron(s) dropped' in output
        assert '111' in output and '222' in output


class TestResolutionKnob:
    def test_banc_notice_is_emitted_by_runtime_source_selector(self):
        """The runtime selector identifies BANC's public SWC source."""
        emitted = []
        vs = _make_visualizer(dataset='flywire_BANC_v888')
        vs._vprint = lambda message, **kwargs: emitted.append(
            (message, kwargs))

        vs._log_local_release_source_selection(use_api_fetching=False)

        assert len(emitted) == 1
        message, kwargs = emitted[0]
        assert message == (
            '  ℹ️  BANC public release source selected '
            '(SWC-first; public bucket; no CAVE token).')
        assert kwargs['level'] == 'simple'

    def test_resolver_uses_unified_chain(self, tmp_path, monkeypatch):
        """Source selection is removed: the resolver never forwards a
        resolution preference to the fetcher (unified L2->full->pcg)."""
        import banc_public_data

        seen = {}

        def fake_fetch(dataset, body_id, project_root=None,
                       use_cache=True, *args, **kwargs):
            seen['args'] = (dataset, str(body_id))
            seen['resolution_forwarded'] = (
                len(args) >= 1 or 'resolution' in kwargs)
            return _tree()

        monkeypatch.setattr(banc_public_data, "fetch_banc_swc", fake_fetch)
        vs = _make_visualizer()
        vs.banc_skeleton_resolution = 'full'  # deprecated/ignored
        vs.script_path = str(tmp_path)
        vs._load_api_cached_skeletons = lambda ids: ({}, list(ids))
        vs._resolve_banc_sources([1], resolution='full')
        assert seen['args'] == ('banc_v888', '1')
        assert seen['resolution_forwarded'] is False

    def test_fetch_rejects_invalid_resolution(self, tmp_path):
        import banc_public_data

        with pytest.raises(ValueError):
            banc_public_data.fetch_banc_swc(
                "banc_v888", "1", resolution="coarse", project_root=tmp_path)
