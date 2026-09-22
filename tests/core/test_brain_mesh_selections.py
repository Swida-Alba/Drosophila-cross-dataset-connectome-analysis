"""Brain-mesh selection vocabulary + cross-template outline specs.

The Skeleton tab's Brain Mesh selection is ``native``/``BANC``/``FAFB``/
``male-cns``/``none`` and nothing else: 'native' keeps the per-dataset
outline, the explicit selections draw that template's outline bridged into
the scene's render space (outline only -- the scene's skeletons never move).
"""
import trimesh
import navis
import numpy as np
import pytest

from visualize_skeleton import (
    BRAIN_MESH_SELECTIONS,
    VisualizeSkeleton,
    normalize_brain_mesh,
)


def _make_visualizer(dataset='male-cns:v1.0'):
    vs = object.__new__(VisualizeSkeleton)
    vs.dataset = dataset
    vs.brain_mesh = 'native'
    vs.vnc_mesh = False
    vs._vprint = lambda *a, **k: None
    return vs


class TestNormalizeBrainMesh:
    """Normalization only restores capitals; it does not translate tokens.

    The pre-rename spellings ('template', 'whole', 'mcns') were folded onto
    the current options for a while, and are now retired: each has to fail
    validation loudly instead of silently drawing a different outline than
    the caller named.
    """

    def test_capitalized_options_are_restored_from_lowercase(self):
        assert normalize_brain_mesh('fafb') == 'FAFB'
        assert normalize_brain_mesh('banc') == 'BANC'

    def test_retired_tokens_are_returned_unchanged_so_validation_rejects(
            self):
        # ``brain_mesh not in BRAIN_MESH_SELECTIONS`` is the validator's own
        # predicate, so this is the rejection -- not a reimplemented one.
        for token in ('template', 'whole', 'mcns', 'WHOLE'):
            folded = normalize_brain_mesh(token)
            assert folded == token.strip().lower()
            assert folded not in BRAIN_MESH_SELECTIONS

    def test_current_tokens_pass_through(self):
        for token in BRAIN_MESH_SELECTIONS:
            assert normalize_brain_mesh(token) == token

    def test_case_and_whitespace(self):
        assert normalize_brain_mesh('  Native ') == 'native'
        assert normalize_brain_mesh('FaFb') == 'FAFB'
        assert normalize_brain_mesh('BANC') == 'BANC'

    def test_unknown_passes_through_for_validation(self):
        assert normalize_brain_mesh('bogus') == 'bogus'


class TestOutlineSpecs:
    def test_native_and_none_have_no_outline_spec(self):
        vs = _make_visualizer()
        assert vs._get_outline_template_info() is None
        vs.brain_mesh = 'none'
        assert vs._get_outline_template_info() is None

    def test_fafb_spec(self):
        vs = _make_visualizer()
        vs.brain_mesh = 'FAFB'
        spec = vs._get_outline_template_info()
        assert spec['space'] == 'FLYWIRE'
        assert spec['split'] is None  # FAFB has no VNC
        assert callable(spec['build'])

    def test_banc_spec(self):
        vs = _make_visualizer('banc_v888')
        vs.brain_mesh = 'BANC'
        spec = vs._get_outline_template_info()
        assert spec['space'] == 'BANC'
        assert spec['split'] == 'banc'

    def test_mcns_spec(self):
        vs = _make_visualizer('flywire_FAFB_v783')
        vs.brain_mesh = 'male-cns'
        spec = vs._get_outline_template_info()
        assert spec['space'] == 'JRCFIB2022M'
        assert spec['split'] == 'mcns'


class TestSelectionMovesScene:
    def test_banc_scene_with_fafb_moves_to_flywire(self):
        """BANC scene + FAFB outline: the whole scene renders in FLYWIRE."""
        vs = _make_visualizer('banc_v888')
        vs.brain_mesh = 'FAFB'
        info = vs._get_template_info()
        assert info['source'] == 'BANC'
        assert info['target'] == 'FLYWIRE'
        assert vs._needs_skeleton_transform() is True
        assert vs._get_template_info().get('skip_transform') is not True

    def test_fafb_scene_with_malecns_moves_to_jrcfib2022m(self):
        vs = _make_visualizer('flywire_FAFB_v783')
        vs.brain_mesh = 'male-cns'
        info = vs._get_template_info()
        assert info['source'] == 'FLYWIRE'
        assert info['target'] == 'JRCFIB2022M'
        assert vs._needs_skeleton_transform() is True

    def test_malecns_scene_with_banc_moves_to_banc(self):
        vs = _make_visualizer('male-cns:v1.0')
        vs.brain_mesh = 'BANC'
        info = vs._get_template_info()
        assert info['source'] == 'JRCFIB2022Mraw'
        assert info['target'] == 'BANC'

    def test_identity_selection_stays_native(self):
        vs = _make_visualizer('banc_v888')
        vs.brain_mesh = 'BANC'
        info = vs._get_template_info()
        assert info['target'] == 'BANC'
        assert info.get('skip_transform') is True

    def test_unreachable_selection_falls_back_native(self):
        """hemibrain cannot reach FLYWIRE: keep the native scene."""
        vs = _make_visualizer('hemibrain:v1.2.1')
        vs.brain_mesh = 'FAFB'
        info = vs._get_template_info()
        assert info['target'] == 'JRCFIB2018F'

    def test_too_indirect_selection_falls_back_native(self):
        """MANC -> FLYWIRE needs >2 hops: keep the native MANC scene."""
        vs = _make_visualizer('manc:v1.2.1')
        vs.brain_mesh = 'FAFB'
        info = vs._get_template_info()
        assert info['target'] == 'MANC'


class TestMcnsSplit:
    def _whole_cns(self):
        verts = np.array([
            # two faces well below the Z=340000 neck cutoff (brain side)
            [0.0, 0.0, 100.0], [10.0, 0.0, 100.0], [0.0, 10.0, 100.0],
            # two faces well above it (VNC side)
            [0.0, 0.0, 500000.0], [10.0, 0.0, 500000.0], [0.0, 10.0, 500000.0],
        ])
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        return navis.Volume(trimesh.Trimesh(vertices=verts, faces=faces))

    def test_split_assigns_by_centroid(self):
        vs = _make_visualizer()
        vs._vprint = lambda *a, **k: None
        brain, vnc = vs._split_mcns_cns_volume(self._whole_cns())
        assert brain is not None and vnc is not None
        assert len(brain.faces) == 1
        assert len(vnc.faces) == 1

    def test_split_none_volume(self):
        vs = _make_visualizer()
        brain, vnc = vs._split_mcns_cns_volume(None)
        assert brain is None and vnc is None


class TestFafbTiltScope:
    """The FAFB tilt correction applies whenever the FAFB mesh is used."""

    def _vis(self, dataset, mesh):
        vs = object.__new__(VisualizeSkeleton)
        vs.dataset = dataset
        vs.brain_mesh = mesh
        vs.FAFB_template_correction = True
        vs._vprint = lambda *a, **k: None
        return vs

    def test_applies_on_native_fafb_scene(self):
        import numpy as np
        m = self._vis('flywire_FAFB_v783', 'native') \
            ._get_fafb_tilt_correction_matrix()
        assert not np.allclose(m, np.eye(4))

    def test_applies_when_fafb_mesh_used_elsewhere(self):
        import numpy as np
        for dataset in ('banc_v888', 'male-cns:v1.0'):
            m = self._vis(dataset, 'FAFB') \
                ._get_fafb_tilt_correction_matrix()
            assert not np.allclose(m, np.eye(4)), dataset

    def test_not_applied_outside_flywire_scenes(self):
        import numpy as np
        for dataset, mesh in (('flywire_FAFB_v783', 'BANC'),
                              ('banc_v888', 'native'),
                              ('manc:v1.2.1', 'FAFB'),
                              ('hemibrain:v1.2.1', 'FAFB'),
                              ('flywire_FAFB_v783', 'none')):
            m = self._vis(dataset, mesh) \
                ._get_fafb_tilt_correction_matrix()
            assert np.allclose(m, np.eye(4)), (dataset, mesh)

    def test_flag_disabled_returns_identity(self):
        import numpy as np
        vs = self._vis('flywire_FAFB_v783', 'FAFB')
        vs.FAFB_template_correction = False
        assert np.allclose(vs._get_fafb_tilt_correction_matrix(), np.eye(4))


class TestMeshTransform:
    def test_identity_when_same_space(self):
        vs = _make_visualizer()
        vol = navis.Volume(trimesh.Trimesh())
        assert vs._transform_template_mesh(vol, 'BANC', 'BANC') is vol
        assert vs._transform_template_mesh(None, 'BANC', 'FLYWIRE') is None

    def test_no_path_returns_none(self):
        """Spaces with no bridging degrade to None instead of raising."""
        vs = _make_visualizer()
        vol = navis.Volume(trimesh.Trimesh())
        # FLYWIRE -> MANC has no registered bridging path.
        assert vs._transform_template_mesh(vol, 'FLYWIRE', 'MANC') is None

    def test_bridges_between_connected_spaces(self):
        vs = _make_visualizer()
        verts = np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0],
                          [0.0, 1000.0, 0.0]])
        vol = navis.Volume(
            trimesh.Trimesh(
                vertices=verts, faces=np.array([[0, 1, 2]], dtype=np.int64)))
        out = vs._transform_template_mesh(vol, 'BANC', 'JRCFIB2022M')
        assert out is not None
