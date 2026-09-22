"""Coverage tests for src/visualize_skeleton.py helper surface.

Targets the large uncovered helper surface of ``visualize_skeleton``:
module-level crop utilities, color standardization/expansion helpers,
ROI expansion, radius/pipeline resolvers, graph-distance math on tiny
synthetic neurons, cache-path helpers, HTML warning banners, and the
GLB export mesh plumbing.

All tests are hermetic: no network, no cache reads from the repo
(cache paths are redirected to ``tmp_path``), no ``fig.show()``, and
plotly is only used offline.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import navis  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

import visualize_skeleton as vs_module  # noqa: E402
from visualize_skeleton import (  # noqa: E402
    TRANSFORMED_ROI_CACHE_MARKER,
    VisualizeSkeleton,
    _apply_consistent_crop,
    _apply_consistent_crop_standalone,
    _compute_unified_crop_bounds,
    _configure_roi_mesh_traces,
    _detect_content_bounds,
    _detect_content_bounds_standalone,
)

DEFAULTS = dict(
    verbose=False,
    dataset='hemibrain:v1.2.1',
    client_type='neuprint',
    background_color='rgba(255, 255, 255, 1.0)',
)


def make_vis(**attrs):
    """Build a VisualizeSkeleton without running __post_init__."""
    vis = object.__new__(VisualizeSkeleton)
    merged = dict(DEFAULTS)
    merged.update(attrs)
    for key, value in merged.items():
        setattr(vis, key, value)
    return vis


def make_chain_neuron(n_nodes=8, body_id='42', radius=10.0):
    """Tiny straight-chain TreeNeuron (hermetic synthetic skeleton)."""
    types = ['root'] + ['slab'] * max(0, n_nodes - 2) + ['end']
    neuron = navis.TreeNeuron(pd.DataFrame({
        'node_id': np.arange(n_nodes, dtype=np.int64),
        'parent_id': np.array([-1] + list(range(n_nodes - 1)), dtype=np.int64),
        'x': np.arange(n_nodes, dtype=float) * 100.0,
        'y': np.zeros(n_nodes),
        'z': np.zeros(n_nodes),
        'radius': np.full(n_nodes, radius, dtype=float),
        'type': types[:n_nodes],
    }))
    neuron.soma = None
    neuron.id = body_id
    return neuron


def make_branched_neuron():
    """Root with two branches of different lengths (graph-distance tests)."""
    neuron = navis.TreeNeuron(pd.DataFrame({
        'node_id': np.array([0, 1, 2, 3, 4], dtype=np.int64),
        'parent_id': np.array([-1, 0, 1, 1, 3], dtype=np.int64),
        'x': np.array([0.0, 100.0, 200.0, 100.0, 100.0]),
        'y': np.array([0.0, 0.0, 0.0, 100.0, 200.0]),
        'z': np.zeros(5),
        'radius': np.full(5, 5.0),
        'type': ['root', 'slab', 'end', 'slab', 'end'],
    }))
    neuron.soma = None
    neuron.id = '7'
    return neuron


def make_white_image_with_block(size=(100, 120), block=((30, 40), (60, 80))):
    """RGB image: white background with a dark rectangle."""
    from PIL import Image
    img = Image.new('RGB', size, (255, 255, 255))
    px = img.load()
    (r0, c0), (r1, c1) = block
    for r in range(r0, r1):
        for c in range(c0, c1):
            px[c, r] = (10, 10, 10)
    return img


# ---------------------------------------------------------------------------
# module-level crop helpers
# ---------------------------------------------------------------------------
class TestModuleCropHelpers:
    def test_configure_roi_mesh_traces(self):
        traces = [go.Mesh3d(x=[0], y=[0], z=[0]) for _ in range(3)]
        out = _configure_roi_mesh_traces(traces, 'AL(L)')
        assert out[0].legendgroup == 'roi_mesh:AL(L)'
        assert out[0].showlegend is True
        assert out[1].showlegend is False
        assert all(t.name == 'brain region [AL(L)]' for t in out)
        assert all(t.hoverinfo == 'name' for t in out)

    def test_detect_content_bounds_rgb(self):
        img = make_white_image_with_block()
        assert _detect_content_bounds(img) == (30, 59, 40, 79)

    def test_detect_content_bounds_rgba(self):
        img = make_white_image_with_block().convert('RGBA')
        assert _detect_content_bounds(img) == (30, 59, 40, 79)

    def test_detect_content_bounds_none_for_blank(self):
        from PIL import Image
        blank = Image.new('RGB', (20, 20), (255, 255, 255))
        assert _detect_content_bounds(blank) is None

    def test_standalone_bounds_matches(self):
        img = make_white_image_with_block(block=((10, 10), (15, 15)))
        assert _detect_content_bounds_standalone(img) == (10, 14, 10, 14)
        from PIL import Image
        assert _detect_content_bounds_standalone(
            Image.new('RGB', (10, 10), (0, 0, 0)), (0, 0, 0)) is None

    def test_compute_unified_crop_bounds(self, tmp_path):
        paths = []
        for i, block in enumerate([((10, 10), (20, 20)), ((40, 60), (55, 90))]):
            p = tmp_path / f'frame_{i}.png'
            make_white_image_with_block(block=block).save(p)
            paths.append(str(p))
        bounds = _compute_unified_crop_bounds(paths)
        assert bounds == (10, 54, 10, 89)

    def test_compute_unified_crop_bounds_sampling_and_errors(self, tmp_path):
        paths = []
        for i in range(8):
            p = tmp_path / f'frame_{i}.png'
            make_white_image_with_block(block=((i, i), (i + 5, i + 5))).save(p)
            paths.append(str(p))
        paths.append(str(tmp_path / 'broken.png'))  # unreadable -> skipped
        bounds = _compute_unified_crop_bounds(paths, sample_count=3)
        assert bounds is not None
        assert _compute_unified_crop_bounds([]) is None
        # only-blank frames -> None
        blank = tmp_path / 'blank.png'
        from PIL import Image
        Image.new('RGB', (10, 10), (255, 255, 255)).save(blank)
        assert _compute_unified_crop_bounds([str(blank)]) is None

    def test_apply_consistent_crop(self, tmp_path):
        for i in range(3):
            img = make_white_image_with_block(block=((20 + i, 30), (40, 60)))
            img.save(tmp_path / f'deg_{i:03d}.jpeg', 'JPEG')
        size = _apply_consistent_crop(str(tmp_path), margin=5)
        # JPEG compression can bleed a few pixels past the drawn block.
        assert 38 <= size[0] <= 52 and 28 <= size[1] <= 40
        from PIL import Image
        with Image.open(tmp_path / 'deg_000.jpeg') as cropped:
            assert cropped.size == (size[0], size[1])

    def test_apply_consistent_crop_empty_and_blank(self, tmp_path):
        assert _apply_consistent_crop(str(tmp_path)) is None
        from PIL import Image
        Image.new('RGB', (20, 20), (255, 255, 255)).save(
            tmp_path / 'deg_000.jpeg', 'JPEG')
        assert _apply_consistent_crop(str(tmp_path)) is None

    def test_standalone_consistent_crop(self, tmp_path):
        for i in range(2):
            img = make_white_image_with_block(block=((10, 10), (30 + i * 10, 30)))
            img.save(tmp_path / f'deg_{i:03d}.jpeg', 'JPEG')
        size = _apply_consistent_crop_standalone(str(tmp_path), margin=0)
        assert 18 <= size[0] <= 30 and 28 <= size[1] <= 42
        assert _apply_consistent_crop_standalone(str(tmp_path.parent / 'nope')) is None

    def test_instance_crop_delegates(self):
        vis = make_vis()
        img = make_white_image_with_block()
        assert vis._detect_content_bounds(img) == (30, 59, 40, 79)
        assert vis._compute_unified_crop_bounds([]) is None


# ---------------------------------------------------------------------------
# verbosity / progress / background helpers
# ---------------------------------------------------------------------------
class TestVerbosityHelpers:
    def test_vprint_levels(self, capsys):
        assert make_vis(verbose=False)._vprint('hidden') is None
        make_vis(verbose='simple')._vprint('shown', level='simple')
        make_vis(verbose='simple')._vprint('hidden', level='full')
        make_vis(verbose='full')._vprint('full-shown', level='full')
        out = capsys.readouterr().out
        assert 'shown' in out and 'full-shown' in out and 'hidden' not in out

    def test_vprint_tqdm_paths(self, capsys):
        make_vis(verbose='full')._vprint('via-tqdm', use_tqdm=True)
        make_vis(verbose='full')._vprint('partial', use_tqdm=True, end='|')
        out = capsys.readouterr().out
        assert 'via-tqdm' in out and 'partial|' in out

    def test_progress(self, capsys):
        make_vis(verbose=False)._progress(1, 3, 'step')
        assert capsys.readouterr().out == ''
        make_vis(verbose='full')._progress(2, 5, 'loading')
        assert '[DROCAT][progress] 2/5 loading' in capsys.readouterr().out

    def test_is_dark_background(self):
        assert make_vis(background_color='black')._is_dark_background() is True
        assert make_vis(background_color='white')._is_dark_background() is False
        assert make_vis()._is_dark_background(color='#ff0000') is True
        assert make_vis()._is_dark_background(color='yellow') is False
        assert make_vis()._is_dark_background(color=object()) is False

    def test_get_effective_mesh_color(self):
        vis = make_vis(brain_mesh_color='auto', vnc_mesh_color='auto')
        assert vis._get_effective_mesh_color('brain') == 'rgba(116, 168, 214, 0.04)'
        vis.background_color = 'black'
        assert vis._get_effective_mesh_color('brain') == 'rgba(116, 168, 214, 0.04)'
        assert vis._get_effective_mesh_color('vnc') == 'rgba(116, 168, 214, 0.04)'
        vis.brain_mesh_color = 'rgba(1, 2, 3, 0.5)'
        vis.vnc_mesh_color = 'rgba(4, 5, 6, 0.5)'
        assert vis._get_effective_mesh_color('brain') == 'rgba(1, 2, 3, 0.5)'
        assert vis._get_effective_mesh_color('vnc') == 'rgba(4, 5, 6, 0.5)'
        vis.background_color = 'white'
        vis.vnc_mesh_color = 'auto'
        assert vis._get_effective_mesh_color('vnc') == 'rgba(116, 168, 214, 0.04)'

    def test_mesh_opacity_defaults(self):
        assert vs_module.DEFAULT_BRAIN_VNC_MESH_ALPHA == 0.04
        assert vs_module.DEFAULT_ROI_MESH_ALPHA == 0.1
        assert VisualizeSkeleton.mesh_alpha == 0.1

    def test_get_html_size_cap(self):
        assert make_vis(html_size_cap=42, export_method='kaleido')._get_html_size_cap() == 42
        assert make_vis(html_size_cap=None, export_method='webdriver')._get_html_size_cap() == 200
        assert make_vis(html_size_cap=None, export_method='webdriver-fast')._get_html_size_cap() == 200
        assert make_vis(html_size_cap=None, export_method='kaleido')._get_html_size_cap() == 100

    def test_add_view_selection_menu_variants(self):
        for dataset, brain_mesh in (
                ('hemibrain:v1.2.1', 'native'),
                ('manc:v1.0', None),
                ('male-cns:v0.9', None)):
            vis = make_vis(dataset=dataset, brain_mesh=brain_mesh, fig_3d=go.Figure())
            vis._add_view_selection_menu()
            menus = vis.fig_3d.layout.updatemenus
            assert len(menus) == 1 and len(menus[0].buttons) == 6
            assert len(vis.fig_3d.layout.annotations) == 2


# ---------------------------------------------------------------------------
# simplification / line-mode warning banners + HTML writing
# ---------------------------------------------------------------------------
class TestWarningBanners:
    def test_simplification_warning_matrix(self):
        warn = VisualizeSkeleton._skeleton_simplification_warning
        assert warn('hemibrain:v1.2.1', 'neuprint', 'line', 0.99) is None
        assert warn('hemibrain:v1.2.1', 'neuprint', 'tube', 'oops') is None
        assert warn('hemibrain:v1.2.1', 'neuprint', 'tube', 0.90) is None
        got = warn('hemibrain:v1.2.1', 'neuprint', 'tube', 0.91)
        assert got['family'] == 'NeuPrint' and got['threshold'] == 0.90
        got = warn('hemibrain:v1.2.1', 'neuprint', 'tube', 0.96, 'fine')
        assert got['threshold'] == 0.95
        assert warn('hemibrain:v1.2.1', 'neuprint', 'tube', 0.94, 'fine') is None
        got = warn('flywire_FAFB_v783', 'flywire', 'tube', 0.96)
        assert got['family'] == 'FlyWire FAFB' and got['threshold'] == 0.95
        assert warn('flywire_FAFB_v783', 'flywire', 'tube', 0.95) is None
        assert warn('banc_v626', 'flywire', 'tube', 0.99) is None

    def test_warning_html_variants(self):
        vis = make_vis(skeleton_mode='tube', skeleton_mesh_simplification=0.99,
                       neuprint_skeleton_pipeline='fast')
        html = vis._skeleton_simplification_warning_html()
        assert 'Skeleton simplification warning.' in html
        assert '0.90' in html
        vis.neuprint_skeleton_pipeline = 'fine'
        assert 'raw .swc.gz source' in vis._skeleton_simplification_warning_html()
        vis.dataset = 'flywire_FAFB_v783'
        assert 'level-0 tree' in vis._skeleton_simplification_warning_html()
        vis.skeleton_mesh_simplification = 0.90
        assert vis._skeleton_simplification_warning_html() == ''
        assert vis._in_page_warning_html() == ''

    def test_line_mode_warning_html(self):
        vis = make_vis(skeleton_mode='tube')
        assert vis._line_mode_export_warning_html() == ''
        vis.skeleton_mode = 'line'
        html = vis._line_mode_export_warning_html()
        assert 'Line-mode skeleton rendering.' in html
        assert 'hemibrain' in html

    def test_combined_in_page_warning_is_single_banner(self):
        vis = make_vis(
            skeleton_mode='line',
            skeleton_mesh_simplification=0.99,
            synapse_mode='pre_post',
            layer_sample_notes=["r1_T1: showing 20 of 25 members"])
        html = vis._in_page_warning_html()
        # exactly one styled banner wrapping every applicable section
        assert html.count('drocat-in-page-warning') == 1
        assert html.startswith('<div id="drocat-in-page-warning"')
        assert html.endswith('</div>')
        assert 'Line-mode skeleton rendering.' in html
        assert 'Pre/post-synaptic site mode.' in html
        assert 'Truncated type layers.' in html
        # line mode suppresses the tube-mesh simplification section
        assert 'Skeleton simplification warning.' not in html
        assert html.count('<strong>') == 3
        # sections render as plain paragraphs inside the one banner
        assert '<li>' not in html

    def test_inject_warning_into_html(self, tmp_path):
        vis = make_vis(skeleton_mode='line', save_folder=str(tmp_path))
        page = tmp_path / 'page.html'
        page.write_text('<html><body><div>plot</div></body></html>')
        vis._inject_page_extras(str(page))
        content = page.read_text(encoding='utf-8')
        assert 'drocat-in-page-warning' in content
        # second injection must be a no-op
        vis._inject_page_extras(str(page))
        assert content == page.read_text(encoding='utf-8')
        # no body tag -> untouched; missing file -> untouched
        page2 = tmp_path / 'nobody.html'
        page2.write_text('<html></html>')
        vis._inject_page_extras(str(page2))
        assert page2.read_text(encoding='utf-8') == '<html></html>'
        vis._inject_page_extras(str(tmp_path / 'absent.html'))

    def test_layer_sampling_warning_html(self):
        assert make_vis()._layer_sampling_warning_html() == ''
        vis = make_vis(layer_sample_notes=[
            "r1_aMe10: showing 5 of 12 members of type 'aMe10' "
            "(per-layer render cap 20)",
        ])
        html = vis._layer_sampling_warning_html()
        # one-row summary: counts the capped layers, points at the notes file
        assert 'Truncated type layers.' in html
        assert '1 type layer shows' in html
        assert 'user_warning_notes.txt' in html
        # per-layer details stay out of the page
        assert 'r1_aMe10' not in html
        assert 'showing 5 of 12' not in html
        vis2 = make_vis(layer_sample_notes=[
            "r1_a: showing 1 of 2", "r2_b: showing 2 of 3"])
        assert '2 type layers show' in vis2._layer_sampling_warning_html()

    def test_inject_layer_sampling_warning_into_html(self, tmp_path):
        vis = make_vis(
            layer_sample_notes=["r1_T1: showing 20 of 25 members"])
        page = tmp_path / 'page.html'
        page.write_text('<html><body><div>plot</div></body></html>')
        vis._inject_page_extras(str(page))
        content = page.read_text(encoding='utf-8')
        assert 'drocat-in-page-warning' in content
        assert 'Truncated type layers.' in content
        # second injection must be a no-op
        vis._inject_page_extras(str(page))
        assert content == page.read_text(encoding='utf-8')

    def test_write_plotly_html(self, tmp_path):
        vis = make_vis(skeleton_mode='line', save_folder=str(tmp_path))
        fig = go.Figure(go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1]))
        out = tmp_path / 'fig.html'
        vis._write_plotly_html(fig, str(out))
        content = out.read_text(encoding="utf-8")
        assert 'plotly' in content.lower()
        assert 'drocat-in-page-warning' in content

    def test_record_large_html_warning(self, tmp_path):
        vis = make_vis(save_folder=str(tmp_path))
        small = tmp_path / 'small.html'
        small.write_text('x')
        assert vis._record_large_html_warning(str(small)) is False
        temp = tmp_path / '_temp_export.html'
        temp.write_text('x')
        assert vis._record_large_html_warning(str(temp)) is False
        assert vis._record_large_html_warning(str(tmp_path / 'missing.html')) is False
        big = tmp_path / 'big.html'
        with open(big, 'wb') as handle:
            handle.truncate(51 * 1024 * 1024)  # sparse 51 MB file
        assert vis._record_large_html_warning(str(big)) is True
        note = (tmp_path / 'user_warning_notes.txt').read_text(encoding='utf-8')
        assert 'render warning' in note and 'big.html' in note
        # duplicate marker is not appended twice
        assert vis._record_large_html_warning(str(big)) is True
        assert note == (tmp_path / 'user_warning_notes.txt').read_text(encoding='utf-8')

    def test_write_user_warning_notes(self, tmp_path):
        vis = make_vis(save_folder=str(tmp_path), skeleton_mode='tube',
                       skeleton_mesh_simplification=0.99,
                       neuprint_skeleton_pipeline='fine_opt')
        vis._write_user_warning_notes()
        text = (tmp_path / 'user_warning_notes.txt').read_text(encoding='utf-8')
        assert 'skeleton_mesh_simplification=0.99' in text
        assert 'neuprint_skeleton_pipeline=fine' in text
        assert '>0.95' in text
        vis.skeleton_mode = 'line'
        vis.dataset = 'flywire_FAFB_v783'
        vis.skeleton_mesh_simplification = None
        vis._write_user_warning_notes()
        text = (tmp_path / 'user_warning_notes.txt').read_text(encoding='utf-8')
        assert "not set" in text and 'not applied' in text

    def test_write_user_warning_notes_truncation_details(self, tmp_path):
        # capped layers are listed per-layer in the notes file
        vis = make_vis(save_folder=str(tmp_path), layer_sample_notes=[
            "r1_T1: showing 20 of 25 members of type 'T1'",
            "r2_T2: showing 20 of 80 members of type 'T2'",
        ])
        vis._write_user_warning_notes()
        text = (tmp_path / 'user_warning_notes.txt').read_text(encoding='utf-8')
        assert '[truncated type layers]' in text
        assert '2 type layers show' in text
        assert "r1_T1: showing 20 of 25 members of type 'T1'" in text
        assert "r2_T2: showing 20 of 80 members of type 'T2'" in text
        # no capped layers -> no truncation block
        vis2 = make_vis(save_folder=str(tmp_path))
        vis2._write_user_warning_notes()
        assert '[truncated type layers]' not in (
            tmp_path / 'user_warning_notes.txt').read_text(encoding='utf-8')


# ---------------------------------------------------------------------------
# small static parsers
# ---------------------------------------------------------------------------
class TestStaticParsers:
    def test_parse_synapse_size(self):
        parse = VisualizeSkeleton._parse_synapse_size
        assert parse('real') == 'real'
        assert parse('  REAL ') == 'real'
        assert parse('2') == 2.0
        assert parse('2.5x') == 2.5
        assert parse('2 x real') == 2.0
        assert parse('1.25\u00d7') == 1.25
        assert parse('banana') is None

    def test_is_valid_color(self):
        vis = make_vis()
        assert vis._is_valid_color('red') is True
        assert vis._is_valid_color((10, 20, 30)) is True
        assert vis._is_valid_color('not-a-color') is False

    def test_unwrap_palette_value(self):
        unwrap = VisualizeSkeleton._unwrap_palette_value
        colors, continuous = unwrap({'colors': ['red'], 'continuous': True})
        assert colors == ['red'] and continuous is True
        colors, continuous = unwrap(['red', 'blue'])
        assert colors == ['red', 'blue'] and continuous is False

        class Palette(list):
            is_continuous_palette = True
        colors, continuous = unwrap(Palette(['red']))
        assert continuous is True


# ---------------------------------------------------------------------------
# color input normalization (instance methods)
# ---------------------------------------------------------------------------
class TestColorNormalization:
    def _color_vis(self, **attrs):
        base = dict(
            neuron_colors=None,
            synapse_colors=[],
            mesh_color=None,
            neuron_layers=['a'],
            brain_mesh_color='auto',
            vnc_mesh_color='auto',
        )
        base.update(attrs)
        return make_vis(**base)

    def test_standardize_color_input(self):
        vis = make_vis()
        assert vis._standardize_color_input('red') == ['rgba(255, 0, 0, 1.0)']
        assert vis._standardize_color_input('bogus') == ['rgba(128, 128, 128, 1.0)']
        assert vis._standardize_color_input((0, 0, 255)) == ['rgba(0, 0, 255, 1.0)']
        assert vis._standardize_color_input(['red', 'zz-nope-123']) == [
            'rgba(255, 0, 0, 1.0)', 'rgba(128, 128, 128, 1.0)']
        assert vis._standardize_color_input([]) == ['rgba(128, 128, 128, 1.0)']
        assert vis._standardize_color_input(np.array(['red'])) == [
            'rgba(255, 0, 0, 1.0)']

    def test_standardize_mesh_color_input(self):
        vis = make_vis(mesh_roi=['A', 'B'])
        assert vis._standardize_mesh_color_input('red') == 'rgba(255, 0, 0, 0.1)'
        assert vis._standardize_mesh_color_input('nope') == 'rgba(100, 100, 100, 0.1)'
        assert vis._standardize_mesh_color_input((0, 255, 0)) == 'rgba(0, 255, 0, 0.1)'
        got = vis._standardize_mesh_color_input(['red', 'blue'])
        assert got == ['rgba(255, 0, 0, 0.1)', 'rgba(0, 0, 255, 0.1)']
        seq = ['#000000', '#444444', '#888888', '#cccccc']
        cont = vis._standardize_mesh_color_input(seq, continuous=True)
        assert len(cont) == 2  # sampled to mesh_roi count
        assert vis._standardize_mesh_color_input(12345) == 'rgba(100, 100, 100, 0.1)'

    def test_is_custom_mesh_color_specified(self):
        assert make_vis(mesh_color=(100, 100, 100))._is_custom_mesh_color_specified() is False
        assert make_vis(mesh_color='rgba(100, 100, 100, 0.1)')._is_custom_mesh_color_specified() is False
        assert make_vis(mesh_color='red')._is_custom_mesh_color_specified() is True
        vis = make_vis(mesh_color='red', _original_mesh_color=(100, 100, 100))
        assert vis._is_custom_mesh_color_specified() is False

    def test_normalize_color_inputs_defaults_and_fallbacks(self):
        vis = self._color_vis(neuron_colors=[], synapse_colors=[], mesh_color=None,
                              brain_mesh_color='not-a-color', vnc_mesh_color=(1, 2),
                              background_color='auto')
        vis._normalize_color_inputs()
        assert vis.neuron_colors is None  # resolved later against background
        assert len(vis.synapse_colors) == 10  # Category10 fallback
        assert vis.mesh_color is None or vis.mesh_color == (100, 100, 100)
        assert vis.brain_mesh_color == 'auto'
        assert vis.vnc_mesh_color == 'auto'
        assert vis.background_color.startswith('rgba(255, 255, 255')

    def test_normalize_color_inputs_valid_and_dicts(self):
        vis = self._color_vis(
            neuron_colors={'colors': ['red', 'bogus'], 'continuous': False},
            synapse_colors={'colors': [], 'continuous': True},
            mesh_color=['red', 'blue'],
            brain_mesh_color='#123456',
            vnc_mesh_color='rgba(1, 2, 3, 0.5)',
            background_color='black',
            neuron_layers=['a', 'b'],
        )
        vis._normalize_color_inputs()
        assert vis.neuron_colors == ['red']
        assert vis._neuron_colors_continuous is False
        assert vis.synapse_colors == list(__import__('bokeh').palettes.Category10[10])
        assert vis.mesh_color == ['red', 'blue']
        assert vis.brain_mesh_color == '#123456'
        assert vis.vnc_mesh_color == 'rgba(1, 2, 3, 0.5)'
        assert vis.background_color.startswith('rgba(0, 0, 0')

    def test_normalize_color_inputs_invalid_sequences(self):
        vis = self._color_vis(neuron_colors=['only-zzz'], synapse_colors='zz-nope-123',
                              mesh_color=123, background_color='navy')
        vis._normalize_color_inputs()
        assert vis.neuron_colors is None
        assert len(vis.synapse_colors) == 10
        assert vis.mesh_color == (100, 100, 100)
        assert vis.background_color.startswith('rgba(0, 0, 128')


# ---------------------------------------------------------------------------
# color string helpers
# ---------------------------------------------------------------------------
class TestColorStringHelpers:
    def test_alpha_and_hex_helpers(self):
        vis = make_vis()
        assert vis._colors_have_explicit_alpha('rgba(1,2,3,0.5)') is True
        assert vis._colors_have_explicit_alpha('red') is False
        assert vis._extract_alpha_from_color('rgba(1, 2, 3, 0.25)') == 0.25
        assert vis._extract_alpha_from_color('garbage') == 1.0
        assert vis._rgba_to_hex('rgba(255, 0, 0, 0.5)') == '#ff0000'
        assert vis._rgba_to_hex('garbage') == '#808080'
        assert vis._get_opaque_color('rgba(255, 0, 0, 0.2)') == 'rgba(255, 0, 0, 1.0)'
        assert vis._get_opaque_color('garbage') == 'garbage'

    def test_darken_color(self):
        vis = make_vis()
        assert vis._darken_color('#ff0000', 0.5) == 'rgba(127, 0, 0, 1.0)'
        assert vis._darken_color((200, 100, 50), 1.0) == 'rgba(200, 100, 50, 1.0)'
        assert vis._darken_color('red', 0.0) == 'rgba(0, 0, 0, 1.0)'

    def test_interpolate_colors(self):
        vis = make_vis()
        base = ['red', 'blue']
        out = vis._interpolate_colors(base, 4)
        assert len(out) == 4
        assert out[0] == 'rgba(255, 0, 0, 1.0)' and out[1] == 'rgba(0, 0, 255, 1.0)'
        assert vis._interpolate_colors(['red'], 3) == ['rgba(255, 0, 0, 1.0)'] * 3
        assert vis._interpolate_colors([], 2) == ['rgba(128, 128, 128, 1.0)'] * 2
        assert vis._interpolate_colors(base, 1) == ['rgba(255, 0, 0, 1.0)']
        out = vis._interpolate_colors(['red', 'zz-nope-123', 'blue'], 6)
        assert len(out) == 6 and out[1] == 'rgba(128, 128, 128, 1.0)'

    def test_sample_continuous_color_sequence(self):
        vis = make_vis()
        assert vis._sample_continuous_color_sequence(['a', 'b'], 0) == []
        assert vis._sample_continuous_color_sequence([], 3) == []
        assert vis._sample_continuous_color_sequence(['a', 'b', 'c'], 1) == ['a']
        assert vis._sample_continuous_color_sequence(['a'], 3) == ['a', 'a', 'a']
        assert vis._sample_continuous_color_sequence(
            ['a', 'b', 'c', 'd', 'e'], 3) == ['a', 'c', 'e']

    def test_expand_color_sequence_policies(self):
        colors = ('red', 'blue')
        vis = make_vis(expand_colors='interpolation')
        out = vis._expand_color_sequence(colors, 4, warn=False)
        assert len(out) == 4
        vis = make_vis(expand_colors='darken')
        out = vis._expand_color_sequence(colors, 5, warn=False)
        assert len(out) == 5 and out[2].startswith('rgba(')
        vis = make_vis(expand_colors='cycle')
        assert vis._expand_color_sequence(colors, 5, warn=False) == (
            'red', 'blue', 'red', 'blue', 'red')
        assert vis._expand_color_sequence(colors, 0, warn=False) == tuple()
        assert vis._expand_color_sequence(colors, 1, warn=False) == ('red',)
        cont = vis._expand_color_sequence(('a', 'b', 'c'), 2, warn=False,
                                          continuous=True)
        assert cont == ('a', 'c')


# ---------------------------------------------------------------------------
# neuron lookup / per-neuron colors
# ---------------------------------------------------------------------------
class TestNeuronColors:
    def test_normalize_neuron_lookup_keys(self):
        vis = make_vis()
        assert vis._normalize_neuron_lookup_keys(None) == []
        assert vis._normalize_neuron_lookup_keys(42) == [42, '42']
        assert vis._normalize_neuron_lookup_keys(np.int64(7)) == [7, '7']
        assert vis._normalize_neuron_lookup_keys(float('nan')) == []
        assert vis._normalize_neuron_lookup_keys('  ') == []
        assert vis._normalize_neuron_lookup_keys('nan') == []
        assert vis._normalize_neuron_lookup_keys('123') == [123, '123']
        assert vis._normalize_neuron_lookup_keys(' DA1_lPN ') == ['DA1_lPN']

    def test_build_per_neuron_color_map(self):
        vis = make_vis(
            neuron_dfs=[pd.DataFrame({'bodyId': [1, 2], 'instance': ['a_L', 'b_R']}),
                        pd.DataFrame()],
            _base_neuron_colors=('red', 'blue'),
            _neuron_colors_continuous=False,
            expand_colors='cycle',
        )
        cmap = vis._build_per_neuron_color_map()
        assert cmap[1] == 'red'
        assert cmap['2'] == 'blue'
        assert cmap['a_L'] == 'red'
        vis.neuron_dfs = [pd.DataFrame()]
        assert vis._build_per_neuron_color_map() == {}

    def test_resolve_neuron_color(self):
        vis = make_vis(neuron_colors=['rgba(1, 1, 1, 1.0)'], color_mode='per_layer')
        assert vis._resolve_neuron_color(5, 0) == 'rgba(1, 1, 1, 1.0)'
        vis.color_mode = 'per_neuron'
        vis._per_neuron_colors = {5: 'rgba(9, 9, 9, 1.0)'}
        assert vis._resolve_neuron_color(5, 0) == 'rgba(9, 9, 9, 1.0)'
        vis._neuron_color_overrides = {'5': 'rgba(7, 7, 7, 1.0)'}
        assert vis._resolve_neuron_color(5, 0) == 'rgba(7, 7, 7, 1.0)'

    def test_apply_plotly_trace_color(self):
        vis = make_vis()
        trace = go.Scatter3d(x=[0], y=[0], z=[0], line=dict(color='blue'),
                             marker=dict(color='blue'))
        vis._apply_plotly_trace_color(trace, 'rgba(255, 0, 0, 0.5)')
        assert trace.line.color == '#ff0000'
        assert trace.marker.color == '#ff0000'
        assert trace.opacity == 0.5

    def test_apply_plotly_trace_opacity_preserves_subdecimal_alpha(self):
        vis = make_vis()
        trace = go.Mesh3d(
            x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0],
            i=[0], j=[1], k=[2], color='rgba(100,100,100,0.0)',
        )

        vis._apply_plotly_trace_opacity(trace, 0.03)

        # The RGB stays intact while the exact opacity is carried by Plotly's
        # trace property instead of navis' one-decimal embedded alpha.
        assert trace.color == '#646464'
        assert trace.opacity == pytest.approx(0.03)

        line_trace = go.Scatter3d(
            x=[0, 1], y=[0, 1], z=[0, 1],
            mode='lines+markers',
            line=dict(color='rgba(10,20,30,0.0)'),
            marker=dict(color='rgba(40,50,60,0.0)'),
        )
        vis._apply_plotly_trace_opacity(line_trace, 0.03)
        assert line_trace.line.color == '#0a141e'
        assert line_trace.marker.color == '#28323c'
        assert line_trace.opacity == pytest.approx(0.03)

    def test_apply_k3d_object_color(self):
        class Obj:
            color = None
            opacity = None
        vis = make_vis()
        obj = Obj()
        vis._apply_k3d_object_color(obj, 'rgba(255, 0, 0, 0.5)')
        assert obj.color == (255 << 16)
        assert obj.opacity == 0.5
        vis._apply_k3d_object_color(obj, 'garbage')  # swallowed


# ---------------------------------------------------------------------------
# ROI expansion helpers
# ---------------------------------------------------------------------------
class TestRoiExpansion:
    def test_flatten_nested_roi_groups(self):
        vis = make_vis()
        flat, colors, groups = vis._flatten_nested_roi_groups(
            ['AME', ['aL', 'bL'], 'EB'], ['red', 'green', 'blue'])
        assert flat == ['AME', 'aL', 'bL', 'EB']
        assert colors == ['red', 'green', 'green', 'blue']
        assert groups['aL'] == 'aL+bL'
        flat, colors, groups = vis._flatten_nested_roi_groups(
            ['A', ['w', 'x', 'y', 'z']], 'red')
        assert groups['z'] == 'w+x+y+1more'
        flat, colors, _ = vis._flatten_nested_roi_groups(
            ['A', 'B'], (255, 0, 0))
        assert colors == [(255, 0, 0), (255, 0, 0)]
        flat, colors, _ = vis._flatten_nested_roi_groups(['A'], ['red'])
        assert colors == ['red']
        flat, colors, _ = vis._flatten_nested_roi_groups(['A', 'B'], [object()])
        assert colors[-1] == 'gray' or colors[0] is not None
        assert vis._flatten_nested_roi_groups([], 'red') == ([], 'red', {})

    def test_expand_roi_names_with_colors(self):
        vis = make_vis()
        available = ['LH(L)', 'LH(R)', 'EB', 'AL(L)']
        rois, colors = vis._expand_roi_names_with_colors(
            ['LH', 'EB', 'AL(L)', 'GONE'], ['red', 'blue'], available)
        assert rois == ['LH(L)', 'LH(R)', 'EB', 'AL(L)', 'GONE']
        assert colors == ['red', 'red', 'blue', 'blue', 'blue']
        rois, colors = vis._expand_roi_names_with_colors(
            ['AL'], 'green', ['AL(R)'])
        assert rois == ['AL(R)'] and colors == ['green']
        assert vis._expand_roi_names_with_colors([], 'red', []) == ([], 'red')


# ---------------------------------------------------------------------------
# hemisphere filter + layer map CSV
# ---------------------------------------------------------------------------
class TestDataFrameHelpers:
    def test_filter_neuron_df_by_hemisphere(self):
        ndf = pd.DataFrame({
            'bodyId': [1, 2, 3, 4],
            'Soma side': ['L', 'R', None, None],
            'instance': ['x', 'y', 'z_R', 'plain'],
        })
        vis = make_vis(hemisphere='left')
        out, rdf = vis._filter_neuron_df_by_hemisphere(ndf)
        assert out['bodyId'].tolist() == [1, 4]
        vis = make_vis(hemisphere='right')
        out, _ = vis._filter_neuron_df_by_hemisphere(ndf)
        assert out['bodyId'].tolist() == [2, 3, 4]
        vis = make_vis(hemisphere='both')
        out, _ = vis._filter_neuron_df_by_hemisphere(ndf)
        assert len(out) == 4
        # rdf filtering by bodyId index
        rdf = pd.DataFrame({'w': [1, 2, 3]}, index=pd.Index(['1', '2', '5'],
                                                             name='bodyId'))
        vis = make_vis(hemisphere='right')
        _, rdf_out = vis._filter_neuron_df_by_hemisphere(ndf, rdf)
        assert list(rdf_out.index) == ['2']

    def test_parse_layer_map_csv(self, tmp_path):
        csv = tmp_path / 'layers.csv'
        csv.write_text(
            'layer,id_type_instance,color\n'
            'L1,101,red\n'
            'L1,102,blue\n'
            'L2,DA1_lPN,\n'
        )
        vis = make_vis(layer_map_csv=str(csv), neuron_alpha=0.8)
        vis._parse_layer_map_csv()
        assert vis.neuron_layers == [[101, 102], 'DA1_lPN']
        assert vis.custom_layer_names == ['L1', 'L2']
        assert vis._neuron_color_overrides[101].startswith('rgba(255, 0, 0')
        assert vis._neuron_color_overrides['102'].startswith('rgba(0, 0, 255')

    def test_parse_layer_map_csv_errors(self, tmp_path):
        vis = make_vis(layer_map_csv=str(tmp_path / 'absent.csv'))
        with pytest.raises(FileNotFoundError):
            vis._parse_layer_map_csv()
        bad = tmp_path / 'bad.csv'
        bad.write_text('wrong,columns\n1,2\n')
        vis.layer_map_csv = str(bad)
        with pytest.raises(ValueError):
            vis._parse_layer_map_csv()

    def test_parse_layer_map_csv_flywire_ids(self, tmp_path):
        csv = tmp_path / 'fw.csv'
        csv.write_text('layer,id_type_instance\nL1,720575940614131061\n')
        vis = make_vis(layer_map_csv=str(csv), dataset='flywire_FAFB_v783',
                       neuron_alpha=1.0)
        vis._parse_layer_map_csv()
        assert len(vis.neuron_layers) == 1

    def test_apply_soma_radius_cap(self):
        neuron = make_chain_neuron(n_nodes=6, radius=100.0)
        vis = make_vis(soma_radius_cap=50.0, smooth_skeleton=False)
        vis._apply_soma_radius_cap(navis.NeuronList([neuron]))
        assert (neuron.nodes['radius'] <= 50.0).all()
        neuron2 = make_chain_neuron(n_nodes=6, radius=100.0)
        neuron2.nodes.loc[3, 'radius'] = 500.0
        vis = make_vis(soma_radius_cap=50.0, smooth_skeleton=True)
        vis._apply_soma_radius_cap(navis.NeuronList([neuron2]))
        assert (neuron2.nodes['radius'] <= 50.0).all()

    def test_generate_smart_layer_names(self):
        vis = make_vis(
            neuron_dfs=[
                pd.DataFrame({'type': ['aMe12', 'aMe12'], 'bodyId': [1, 2]}),
                pd.DataFrame({'type': ['x', 'y'], 'bodyId': [3, 4]}),
                pd.DataFrame({'bodyId': [5]}),
                pd.DataFrame({'bodyId': [6, 7]}),
                pd.DataFrame(),
            ],
            layer_names=['l0', 'l1', 'l2', 'l3', 'l4'],
        )
        names = vis._generate_smart_layer_names()
        assert names[0] == 'aMe12'
        assert names[1].endswith('_etc')
        assert names[2:] == ['5', '6_etc', 'l4']

    def test_save_synapse_data(self, tmp_path):
        frames = [
            pd.DataFrame({'viz_layer': ['0->1'], 'pre': [1], 'post': [2]}),
            pd.DataFrame({'viz_layer': ['1->2'], 'pre': [2], 'post': [3]}),
        ]
        vis = make_vis(save_folder=str(tmp_path), saveas='run', output_format='csv')
        out = vis._save_synapse_data(frames)
        assert out == str(tmp_path / 'run_synapses.csv')
        saved = pd.read_csv(out)
        assert list(saved.columns)[0] == 'viz_layer' and len(saved) == 2
        vis.output_format = 'xlsx'
        out = vis._save_synapse_data(frames)
        assert out.endswith('.xlsx') and os.path.exists(out)
        vis.output_format = 'tsv'
        with pytest.raises(ValueError):
            vis._save_synapse_data(frames)
        assert vis._save_synapse_data([]) is None


# ---------------------------------------------------------------------------
# cache paths / dataset helpers
# ---------------------------------------------------------------------------
class TestCachePaths:
    def test_get_cache_path_and_synapse_path(self, tmp_path):
        vis = make_vis(script_path=str(tmp_path))
        cache_dir = vis._get_cache_path('skeletons')
        assert cache_dir == str(tmp_path / 'cache' / 'hemibrain_v1_2_1' / 'skeletons')
        assert os.path.isdir(cache_dir)
        syn_path = vis._get_synapse_cache_path(11, 22)
        assert syn_path.endswith('11_22.parquet')

    def test_get_dataset_mesh_dir_and_files(self, tmp_path):
        vis = make_vis(script_path=str(tmp_path))
        mesh_dir = vis._get_dataset_mesh_dir()
        assert os.path.isdir(mesh_dir)
        assert vis._roi_to_filename('AL(L)') == 'AL(L).json'
        assert vis._roi_to_filename('aL(L)') == '_aL(L).json'
        # encoded file exists -> returned
        encoded = Path(mesh_dir) / '_aL(L).json'
        encoded.write_text('{}')
        assert vis._get_mesh_file_path(mesh_dir, 'aL(L)') == str(encoded)
        # legacy fallback for uppercase
        legacy = Path(mesh_dir) / 'EB.json'
        legacy.write_text('{}')
        assert vis._get_mesh_file_path(mesh_dir, 'EB') == str(legacy)
        # nothing exists -> encoded path returned
        assert vis._get_mesh_file_path(mesh_dir, 'PB').endswith('PB.json')

    def test_get_dataset_abbreviation(self):
        cases = {
            'hemibrain:v1.2.1': 'HEMI',
            'male-cns:v0.9': 'MCNS',
            'flywire_FAFB_v783': 'FAFB',
            'banc_v626': 'BANC',
            'optic-lobe:v1.1': 'OL',
            'manc:v1.0': 'MANC',
            'customset': 'CUST',
        }
        for dataset, expected in cases.items():
            assert make_vis(dataset=dataset)._get_dataset_abbreviation() == expected
        assert make_vis(dataset=None)._get_dataset_abbreviation() == 'UNKN'

    def test_get_synapse_table_path(self, tmp_path):
        vis = make_vis(script_path=str(tmp_path))
        assert vis._get_synapse_table_path() is None
        ds_dir = tmp_path / 'datasets' / 'hemibrain_v1_2_1'
        ds_dir.mkdir(parents=True)
        table = ds_dir / 'hemibrain_v1_2_1_synapse_table.parquet'
        table.write_bytes(b'x')
        assert vis._get_synapse_table_path() == str(table)
        fw = make_vis(script_path=str(tmp_path), dataset='flywire_FAFB_v783')
        assert fw._get_synapse_table_path() is None  # no datasets dir present


# ---------------------------------------------------------------------------
# pipeline / radius-style resolvers + skeleton graph math
# ---------------------------------------------------------------------------
class TestPipelineResolvers:
    def test_resolved_neuprint_pipeline(self):
        vis = make_vis(neuprint_skeleton_pipeline='fast', skeleton_mode='tube')
        assert vis._resolved_neuprint_skeleton_pipeline() == 'direct'
        vis.neuprint_skeleton_pipeline = 'fine_opt'
        assert vis._resolved_neuprint_skeleton_pipeline() == 'fine'
        vis.neuprint_skeleton_pipeline = 'fine_opt1'
        assert vis._resolved_neuprint_skeleton_pipeline() == 'artistic'
        vis.neuprint_skeleton_pipeline = 'fafb'
        assert vis._resolved_neuprint_skeleton_pipeline() == 'fine'
        vis.skeleton_mode = 'line'
        assert vis._resolved_neuprint_skeleton_pipeline() == 'direct'
        assert vis._uses_neuprint_fine_pipeline() is False
        vis.skeleton_mode = 'tube'
        vis.neuprint_skeleton_pipeline = 'fine'
        assert vis._uses_neuprint_fine_pipeline() is True
        assert vis._uses_neuprint_fine_optimized_pipeline() is True

    def test_resolved_fafb_pipeline(self):
        vis = make_vis(neuprint_skeleton_pipeline='direct')
        assert vis._resolved_fafb_pipeline() == 'fast'
        vis.neuprint_skeleton_pipeline = 'artistic'
        assert vis._resolved_fafb_pipeline() == 'artistic'
        vis.neuprint_skeleton_pipeline = None
        assert vis._resolved_fafb_pipeline() == 'fast'

    def test_resolved_skeleton_radius_style(self):
        assert make_vis(skeleton_radius_style='auto',
                        dataset='hemibrain:v1.2.1')._resolved_skeleton_radius_style() == 'fafb'
        assert make_vis(skeleton_radius_style='auto',
                        dataset='flywire_FAFB_v783')._resolved_skeleton_radius_style() == 'source'
        assert make_vis(skeleton_radius_style='default')._resolved_skeleton_radius_style() == 'constant'
        assert make_vis(skeleton_radius_style='source')._resolved_skeleton_radius_style() == 'source'
        assert make_vis(skeleton_radius_style=None)._resolved_skeleton_radius_style() == 'fafb'

    def test_cache_keys(self):
        vis = make_vis(neuprint_skeleton_pipeline='fast', skeleton_mode='tube',
                       skeleton_radius_style='auto')
        assert vis._get_neuprint_mesh_cache_key() == 'NEUPRINT_simp95'
        vis.neuprint_skeleton_pipeline = 'artistic'
        assert vis._get_neuprint_mesh_cache_key().endswith('_vertexcluster')
        vis.skeleton_radius_style = 'constant'
        assert '_radiusconstant' in vis._get_neuprint_mesh_cache_key()
        assert make_vis()._skeleton_cache_is_simplified() is False
        assert make_vis(skeleton_mesh_simplification=0.93)._effective_render_simplification(True) == 0.93


class TestSkeletonGraphMath:
    def test_edge_lengths_slow_and_fast(self):
        neuron = make_chain_neuron(n_nodes=5)
        vis = make_vis()
        slow = vis._neuron_edge_lengths(neuron)
        fast = vis._neuron_edge_lengths_fast(neuron)
        assert len(slow) == 4 and len(fast) == 4
        np.testing.assert_allclose(slow, fast)
        np.testing.assert_allclose(fast, np.full(4, 100.0))

    def test_edge_lengths_single_node(self):
        single = navis.TreeNeuron(pd.DataFrame({
            'node_id': np.array([0], dtype=np.int64),
            'parent_id': np.array([-1], dtype=np.int64),
            'x': [0.0], 'y': [0.0], 'z': [0.0], 'radius': [1.0],
            'type': ['root'],
        }))
        vis = make_vis()
        assert len(vis._neuron_edge_lengths(single)) == 0
        assert len(vis._neuron_edge_lengths_fast(single)) == 0

    def test_dist_to_tip_slow_and_fast(self):
        neuron = make_branched_neuron()
        vis = make_vis()
        slow = vis._neuron_dist_to_tip(neuron)
        fast = vis._neuron_dist_to_tip_fast(neuron)
        assert set(slow) == set(fast)
        for key in slow:
            assert slow[key] == pytest.approx(fast[key])
        assert fast[2] == 0.0 and fast[4] == 0.0
        assert fast[1] == pytest.approx(100.0)
        # single-node tree -> all zeros
        single = navis.TreeNeuron(pd.DataFrame({
            'node_id': np.array([0], dtype=np.int64),
            'parent_id': np.array([-1], dtype=np.int64),
            'x': [0.0], 'y': [0.0], 'z': [0.0], 'radius': [1.0],
            'type': ['root'],
        }))
        assert vis._neuron_dist_to_tip(single) == {0: 0.0}
        assert vis._neuron_dist_to_tip_fast(single) == {0: 0.0}

    def test_fafb_style_radius(self):
        vis = make_vis()
        neuron = make_branched_neuron()
        for optimized in (False, True):
            out = vis._fafb_style_radius(neuron, optimized=optimized)
            radii = out.nodes['radius'].to_numpy()
            assert (radii <= vis.NEUPRINT_RADIUS_SOMA_CAP).all()
            assert (radii > 0).all()

    def test_constant_neuprint_radius(self):
        vis = make_vis()
        out = vis._constant_neuprint_radius(make_chain_neuron())
        assert (out.nodes['radius'] == vis.NEUPRINT_DEFAULT_RADIUS).all()

    def test_fafb_style_neuprint_skeleton(self):
        vis = make_vis(dataset='hemibrain:v1.2.1', skeleton_radius_style='auto')
        for optimized in (False, True):
            out = vis._fafb_style_neuprint_skeleton(
                make_chain_neuron(n_nodes=10), optimized=optimized)
            assert 'radius' in out.nodes.columns
        out = vis._fafb_style_neuprint_skeleton(
            make_chain_neuron(n_nodes=6), radius_style='constant', optimized=True)
        assert (out.nodes['radius'] == vis.NEUPRINT_DEFAULT_RADIUS).all()
        out = vis._fafb_style_neuprint_skeleton(
            make_chain_neuron(n_nodes=6), radius_style='source')
        assert len(out.nodes) > 0

    def test_downsample_neuprint_skeleton_for_cache(self):
        vis = make_vis()
        obj = object()
        assert vis._downsample_neuprint_skeleton_for_cache(obj) is obj
        out = vis._downsample_neuprint_skeleton_for_cache(make_chain_neuron(n_nodes=30))
        assert hasattr(out, 'nodes')


# ---------------------------------------------------------------------------
# cached-neuron plumbing (monkeypatched cache backend)
# ---------------------------------------------------------------------------
class FakeRawCache:
    def __init__(self, store):
        self.store = store
        self.persisted = None

    def load_skeleton(self, bid):
        return self.store.get(bid)

    def persist_skeletons(self, items):
        self.persisted = items
        return len(items)


class TestCachedNeuronPlumbing:
    def test_load_cached_neurons_short_circuits(self):
        df = pd.DataFrame({'bodyId': [1, 2]})
        vis = make_vis(dataset='flywire_FAFB_v783')
        neurons, missing = vis._load_cached_neurons(df)
        assert neurons is None and missing == [1, 2]
        vis = make_vis()
        neurons, missing = vis._load_cached_neurons(df, ignore_cache=True)
        assert neurons is None and missing == [1, 2]

    def test_load_cached_neurons_from_raw_cache(self, monkeypatch):
        store = {1: make_chain_neuron(body_id='1')}
        fake = FakeRawCache(store)
        import morphology
        monkeypatch.setattr(morphology, 'find_similar_raw_cache',
                            lambda *a, **k: fake)
        vis = make_vis(script_path='/tmp')
        df = pd.DataFrame({'bodyId': [1, 2]})
        neurons, missing = vis._load_cached_neurons(df)
        assert len(neurons) == 1 and missing == [2]
        # failing backend -> graceful None
        def boom(*a, **k):
            raise RuntimeError('no cache')
        monkeypatch.setattr(morphology, 'find_similar_raw_cache', boom)
        neurons, missing = vis._load_cached_neurons(df)
        assert neurons is None and missing == [1, 2]

    def test_save_cached_neurons(self, monkeypatch):
        import morphology
        fake = FakeRawCache({})
        monkeypatch.setattr(morphology, 'find_similar_raw_cache',
                            lambda *a, **k: fake)
        vis = make_vis(script_path='/tmp')
        df = pd.DataFrame({'bodyId': [1]})
        vis._save_cached_neurons(df, navis.NeuronList(
            [make_chain_neuron(body_id='1')]))
        assert fake.persisted and 1 in fake.persisted
        # flywire datasets skip persistence entirely
        fake2 = FakeRawCache({})
        monkeypatch.setattr(morphology, 'find_similar_raw_cache',
                            lambda *a, **k: fake2)
        make_vis(dataset='flywire_FAFB_v783')._save_cached_neurons(
            df, navis.NeuronList([make_chain_neuron(body_id='1')]))
        assert fake2.persisted is None
        vis._save_cached_neurons(df, None)

    def test_cave_store_probe_reads_dedicated_store(self, tmp_path):
        """`_load_cave_cached_skeleton` reads the cave_skeletons store
        (network-free) and misses cleanly when it is empty."""
        from cave_data_fetcher import _write_compressed_swc_zst

        vis = make_vis(cache_neurons=True, dataset='flywire_FAFB_v783',
                       script_path=str(tmp_path))
        assert vis._load_cave_cached_skeleton(FAFB_ID) is None

        store_dir = (tmp_path / 'cache' / 'flywire_FAFB_v783'
                     / 'skeletons' / 'cave_skeletons')
        store_dir.mkdir(parents=True)
        _write_compressed_swc_zst(
            store_dir / f'{FAFB_ID_STR}.swc.zst',
            make_fafb_swc_string().encode('utf-8'), simplification=0)
        loaded = vis._load_cave_cached_skeleton(FAFB_ID)
        assert isinstance(loaded, navis.TreeNeuron)
        assert loaded.name == FAFB_ID_STR

    def test_extrusion_cache_delegation(self, monkeypatch):
        import fafb_utils
        calls = {}
        monkeypatch.setattr(fafb_utils, 'load_extrusion_check_cache',
                            lambda root, ds: (calls.update(load=(root, ds)), {1: True})[1])
        monkeypatch.setattr(fafb_utils, 'save_extrusion_check_cache',
                            lambda root, ds, res: calls.update(save=(root, ds, res)))
        vis = make_vis(script_path='/proj', dataset='flywire_FAFB_v783')
        assert vis._load_extrusion_check_cache() == {1: True}
        vis._save_extrusion_check_cache({2: False})
        assert calls['save'][1] == 'flywire_FAFB_v783'
        flagged = []
        monkeypatch.setattr(fafb_utils, 'flag_extrusions',
                            lambda *a, **k: flagged.append(k) or [42])
        assert vis._detect_extrusions_in_skeletons({42: object()},
                                                   use_cache=False) == [42]
        assert flagged[0]['use_cache'] is False


# ---------------------------------------------------------------------------
# extrusion detection on synthetic meshes
# ---------------------------------------------------------------------------
class TestExtrusionDetection:
    def _clean_mesh(self):
        import trimesh
        return trimesh.creation.box(extents=(100.0, 100.0, 100.0))

    def _spiky_mesh(self):
        import trimesh
        box = trimesh.creation.box(extents=(100.0, 100.0, 100.0))
        verts = list(box.vertices)
        faces = list(box.faces)
        verts.append((0.0, 0.0, 50000.0))  # long spike
        spike = len(verts) - 1
        faces.append((0, 1, spike))
        faces.append((1, 2, spike))
        return trimesh.Trimesh(vertices=np.array(verts),
                               faces=np.array(faces), process=False)

    def test_clean_mesh(self):
        result = VisualizeSkeleton.detect_mesh_extrusions(self._clean_mesh())
        assert result['has_extrusions'] is False
        assert result['severity'] == 'none'
        assert 'clean' in result['recommendation']

    def test_spiky_mesh_with_soma(self, capsys):
        result = VisualizeSkeleton.detect_mesh_extrusions(
            self._spiky_mesh(), soma_pos=[0, 0, 0], verbose=True)
        assert result['has_extrusions'] == True
        assert result['extrusion_count'] > 0
        assert result['edge_length_ratio'] > 3.0
        assert result['soma_region_issues'] == True
        assert 'extrusion' in result['recommendation'].lower()
        assert 'Mesh Analysis Results' in capsys.readouterr().out

    def test_invalid_mesh(self):
        result = VisualizeSkeleton.detect_mesh_extrusions(object())
        assert result['severity'] == 'unknown'
        assert result['has_extrusions'] is False

    def test_meshneuron_like_wrapper(self):
        class Wrapper:
            def __init__(self, tm):
                self.trimesh = tm
        result = VisualizeSkeleton.detect_mesh_extrusions(
            Wrapper(self._clean_mesh()))
        assert result['severity'] == 'none'


# ---------------------------------------------------------------------------
# kaleido simplification + GLB export plumbing
# ---------------------------------------------------------------------------
class TestFigureSimplificationAndExport:
    def test_simplify_figure_for_kaleido(self):
        import trimesh
        vis = make_vis(verbose=False)
        sphere = trimesh.creation.icosphere(subdivisions=2)
        mesh_trace = go.Mesh3d(
            x=sphere.vertices[:, 0], y=sphere.vertices[:, 1],
            z=sphere.vertices[:, 2],
            i=sphere.faces[:, 0], j=sphere.faces[:, 1], k=sphere.faces[:, 2],
        )
        long_line = go.Scatter3d(x=np.arange(2000.0), y=np.arange(2000.0),
                                 z=np.arange(2000.0),
                                 marker=dict(color=list(range(2000))))
        short_line = go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1])
        fig = go.Figure(data=[mesh_trace, long_line, short_line])
        out = vis._simplify_figure_for_kaleido(fig, 0.25)
        assert len(out.data) == 3
        assert len(out.data[1].x) < 2000
        assert len(out.data[2].x) == 2

    def test_to_rgba(self):
        vis = make_vis()
        rgba = vis._to_rgba('red')
        assert tuple(rgba) == (255, 0, 0, 255)
        rgba = vis._to_rgba('red', alpha=0.5)
        assert rgba[3] == 128 or rgba[3] == 127
        rgba = vis._to_rgba(object())
        assert tuple(rgba)[:3] == (128, 128, 128)

    def test_extract_trimesh_variants(self):
        import trimesh
        vis = make_vis()
        assert vis._extract_trimesh(None) is None
        box = trimesh.creation.box()
        assert vis._extract_trimesh(box) is box
        mesh3d = go.Mesh3d(x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0],
                           i=[0], j=[1], k=[2])
        tm = vis._extract_trimesh(mesh3d)
        assert tm is not None and len(tm.faces) == 1
        empty = go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[])
        assert vis._extract_trimesh(empty) is None

        class WithMesh:
            def __init__(self, tm):
                self.mesh = tm
        assert vis._extract_trimesh(WithMesh(box)) is box

        class VertsFaces:
            vertices = box.vertices
            faces = box.faces
        assert vis._extract_trimesh(VertsFaces()) is not None
        assert vis._extract_trimesh(object()) is None

    def test_append_exportable_mesh_and_glb_export(self, tmp_path):
        import trimesh
        vis = make_vis(save_folder=str(tmp_path), saveas='scene',
                       exportable_meshes=[])
        box = trimesh.creation.box()
        assert vis._append_exportable_mesh(box, color='red', name='roi',
                                           role='roi') is True
        assert vis._append_exportable_mesh(object()) is False
        assert len(vis.exportable_meshes) == 1
        stored = vis.exportable_meshes[0]
        assert stored.metadata['export_role'] == 'roi'
        rgba = vis._get_glb_export_rgba(stored)
        assert tuple(rgba)[:3] == (255, 0, 0)
        prepared = vis._prepare_glb_mesh(stored, 'roi')
        assert prepared is not stored
        out = vis.export_3d_model()
        assert out == str(tmp_path / 'scene.glb')
        assert os.path.getsize(out) > 0
        # no meshes at all -> None
        empty_vis = make_vis(save_folder=str(tmp_path), saveas='empty',
                             exportable_meshes=[])
        assert empty_vis.export_3d_model() is None

    def test_get_glb_export_rgba_fallbacks(self):
        import trimesh
        vis = make_vis()
        box = trimesh.creation.box()
        box.visual.face_colors = [10, 20, 30, 255]
        rgba = vis._get_glb_export_rgba(box)
        assert tuple(rgba)[:3] == (10, 20, 30)


# ---------------------------------------------------------------------------
# transform helpers + available-ROI discovery
# ---------------------------------------------------------------------------
class TestTransformHelpers:
    def test_dataset_needs_transform(self):
        # The JRC2018F scene-transform mode ('whole') was retired: no
        # selection downloads H5 transforms anymore.
        assert make_vis(dataset='hemibrain:v1.2.1',
                        brain_mesh='native')._dataset_needs_transform() is False
        assert make_vis(dataset='flywire_FAFB_v783',
                        brain_mesh='native')._dataset_needs_transform() is False
        assert make_vis(dataset='flywire_FAFB_v783',
                        brain_mesh='FAFB')._dataset_needs_transform() is False
        assert make_vis(dataset='male-cns:v0.9',
                        brain_mesh='male-cns')._dataset_needs_transform() is False

    def test_fafb_tilt_correction_matrix(self):
        vis = make_vis(dataset='hemibrain:v1.2.1', brain_mesh='native')
        np.testing.assert_array_equal(vis._get_fafb_tilt_correction_matrix(),
                                      np.eye(4))
        # 'FAFB' renders the scene in FLYWIRE coordinates: the template
        # tilt correction applies there too (mesh + neurons together).
        vis = make_vis(dataset='flywire_FAFB_v783', brain_mesh='FAFB')
        matrix = vis._get_fafb_tilt_correction_matrix()
        assert matrix.shape == (4, 4)
        assert not np.allclose(matrix, np.eye(4))
        vis = make_vis(dataset='flywire_FAFB_v783', brain_mesh='native')
        matrix = vis._get_fafb_tilt_correction_matrix()
        assert matrix.shape == (4, 4)
        assert not np.allclose(matrix, np.eye(4))
        # rotation keeps the brain center fixed
        center = np.array([527652.0, 240039.0, 148110.0, 1.0])
        np.testing.assert_allclose(matrix @ center, center, atol=1e-6)

    def test_apply_fafb_tilt_correction(self):
        # short-circuit paths return the object unchanged
        vis = make_vis(dataset='hemibrain:v1.2.1', brain_mesh='native')
        df = pd.DataFrame({'x': [1.0], 'y': [2.0], 'z': [3.0]})
        assert vis._apply_fafb_tilt_correction(df) is df
        vis = make_vis(dataset='flywire_FAFB_v783', brain_mesh='native',
                       FAFB_template_correction=False)
        assert vis._apply_fafb_tilt_correction(df) is df

        vis = make_vis(dataset='flywire_FAFB_v783', brain_mesh='native',
                       FAFB_template_correction=True)
        out = vis._apply_fafb_tilt_correction(df)
        assert out is not df
        assert np.isfinite(out[['x', 'y', 'z']].to_numpy()).all()
        # the brain-center point itself stays fixed under the transform
        center_df = pd.DataFrame({'x': [527652.0], 'y': [240039.0],
                                  'z': [148110.0]})
        center_out = vis._apply_fafb_tilt_correction(center_df)
        np.testing.assert_allclose(
            center_out[['x', 'y', 'z']].to_numpy(),
            center_df[['x', 'y', 'z']].to_numpy(), atol=1e-4)
        # DataFrame without xyz columns passes through
        plain = pd.DataFrame({'a': [1]})
        assert vis._apply_fafb_tilt_correction(plain) is plain

        neuron = make_chain_neuron(n_nodes=5)
        xformed = vis._apply_fafb_tilt_correction(neuron)
        assert hasattr(xformed, 'nodes')
        nlist = navis.NeuronList([make_chain_neuron(body_id='1'),
                                  make_chain_neuron(body_id='2')])
        xformed_list = vis._apply_fafb_tilt_correction(nlist)
        assert len(xformed_list) == 2


class TestAvailableRois:
    def test_reads_cached_roi_list(self, tmp_path):
        cache_dir = tmp_path / 'cache' / 'hemibrain_v1_2_1'
        cache_dir.mkdir(parents=True)
        (cache_dir / 'available_rois.json').write_text('["AL(L)", "EB"]')
        vis = make_vis(script_path=str(tmp_path))
        assert vis._get_available_rois() == ['AL(L)', 'EB']

    def test_flywire_local_scan(self, tmp_path):
        primary = tmp_path / 'navis_roi_meshes_json' / 'primary_rois'
        primary.mkdir(parents=True)
        (primary / 'LH.json').write_text('{}')
        hb_meshes = tmp_path / 'cache' / 'hemibrain_v1_2_1' / 'meshes'
        hb_meshes.mkdir(parents=True)
        (hb_meshes / 'EB.json').write_text('{}')
        vis = make_vis(script_path=str(tmp_path), dataset='flywire_FAFB_v783')
        rois = vis._get_available_rois(use_cache=False, fetch_online=True)
        assert rois == ['EB', 'LH']
        cached = tmp_path / 'cache' / 'flywire_FAFB_v783' / 'available_rois.json'
        assert cached.exists()

    def test_local_mesh_dir_fallback(self, tmp_path):
        vis = make_vis(script_path=str(tmp_path))
        mesh_dir = Path(vis._get_dataset_mesh_dir())
        (mesh_dir / 'AL(L).json').write_text('{}')
        (mesh_dir / 'FB.json').write_text('{}')
        rois = vis._get_available_rois(use_cache=False, fetch_online=False)
        assert rois == sorted(['AL(L)', 'FB'])
        # corrupt cache file falls back to local scan as well
        cache_file = (tmp_path / 'cache' / 'hemibrain_v1_2_1'
                      / 'available_rois.json')
        cache_file.write_text('{not json')
        rois = vis._get_available_rois(use_cache=True, fetch_online=False)
        assert rois == sorted(['AL(L)', 'FB'])


# ---------------------------------------------------------------------------
# individual PDF/PPTX profile exports (synthetic PNGs)
# ---------------------------------------------------------------------------
class TestIndividualExports:
    def _images_dict(self, tmp_path):
        from PIL import Image
        images = {}
        for rank in (1, 2, 10):
            entries = []
            for view in ('front', 'top'):
                p = tmp_path / f'r{rank}_{view}.png'
                Image.new('RGB', (60, 40), (rank * 20 % 256, 100, 50)).save(p)
                entries.append((str(p), view))
            images[f'r{rank}'] = entries
        images['ghost'] = [(str(tmp_path / 'missing.png'), 'front')]
        return images

    def test_create_individual_pdf_by_name(self, tmp_path):
        vis = make_vis()
        out = vis._create_individual_pdf(
            str(tmp_path), self._images_dict(tmp_path),
            images_per_page=(2, 2), title='Profiles',
            organize_by='name', pdf_suffix='_by_name')
        assert out is not None and os.path.exists(out)
        assert out.endswith('_by_name.pdf')
        assert os.path.getsize(out) > 0

    def test_create_individual_pdf_by_view_and_dark_bg(self, tmp_path):
        vis = make_vis()
        out = vis._create_individual_pdf(
            str(tmp_path), self._images_dict(tmp_path),
            organize_by='view', views=['front', 'top'],
            background_color='black')
        assert out is not None and os.path.exists(out)

    def test_create_individual_pdf_empty(self, tmp_path):
        vis = make_vis()
        assert vis._create_individual_pdf(str(tmp_path), {}) is None
        assert vis._create_individual_pdf(
            str(tmp_path), {'a': [(str(tmp_path / 'nope.png'), 'front')]}) is None

    def test_create_individual_pptx_by_name(self, tmp_path):
        vis = make_vis()
        out = vis._create_individual_pptx(
            str(tmp_path), self._images_dict(tmp_path),
            images_per_page=(2, 2), title='Profiles',
            organize_by='name', pptx_suffix='_by_name')
        assert out is not None and os.path.exists(out)

    def test_create_individual_pptx_by_view_dark(self, tmp_path):
        vis = make_vis()
        out = vis._create_individual_pptx(
            str(tmp_path), self._images_dict(tmp_path),
            organize_by='view', views=['front', 'top'],
            background_color='#101010')
        assert out is not None and os.path.exists(out)

    def test_create_individual_pptx_empty(self, tmp_path):
        vis = make_vis()
        assert vis._create_individual_pptx(str(tmp_path), {}) is None


# ---------------------------------------------------------------------------
# backward-compat module functions
# ---------------------------------------------------------------------------
class TestCompatWrappers:
    def test_img2pptx_wrapper(self, tmp_path):
        from PIL import Image
        folder = tmp_path / 'imgs'
        folder.mkdir()
        for i in range(2):
            Image.new('RGB', (40, 40), (255 * i % 256, 100, 200)).save(
                folder / f'img_{i}.png')
        out = vs_module.img2pptx(str(folder), output_pptx=str(tmp_path / 'out.pptx'))
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# real constructor (__post_init__) branch coverage — fully hermetic:
# getNeurons / file converters / token manager are monkeypatched and all
# cache + output paths are redirected to tmp_path.
# ---------------------------------------------------------------------------
class TestRealConstructor:
    @pytest.fixture
    def hermetic_ctor(self, monkeypatch):
        """Patch every remote/cache touch-point used by __post_init__."""
        import utils.token_manager as tm_module

        calls = []

        def fake_get_neurons(layer_input, dataset='hemibrain:v1.2.1',
                             client=None, verbose=True,
                             search_columns='auto', **kwargs):
            calls.append((list(layer_input), dataset))
            ndf = pd.DataFrame({
                'bodyId': [101, 202],
                'type': ['DA1_lPN', 'DA1_lPN'],
                'instance': ['DA1_lPN_L', 'DA1_lPN_R'],
            })
            rdf = pd.DataFrame({'ROI': ['AL(L)', 'AL(R)'],
                                'volume': [1000.0, 1200.0]})
            return ndf, rdf, 'DA1_lPN', {'type': 'DA1_lPN'}

        monkeypatch.setattr(vs_module.sv, 'getNeurons', fake_get_neurons)
        monkeypatch.setattr(
            vs_module.FAFB_file_converter, 'ensure_flywire_data',
            lambda dataset, dataset_dir: True)
        monkeypatch.setattr(
            vs_module.BANC_file_converter, 'ensure_banc_data',
            lambda dataset, dataset_dir: True)
        monkeypatch.setattr(tm_module.token_manager, 'get_token',
                            lambda name, direct=None: None)
        monkeypatch.delenv('NEUPRINT_APPLICATION_CREDENTIALS', raising=False)
        monkeypatch.delenv('CAVE_TOKEN', raising=False)
        return calls

    @staticmethod
    def ctor_kwargs(tmp_path, **extra):
        kw = dict(
            script_path=str(tmp_path),
            output_dir=str(tmp_path / 'out'),
            data_folder=str(tmp_path / 'data'),
            include_timestamp=False,
            verbose=False,
        )
        kw.update(extra)
        return kw

    def test_hemibrain_defaults(self, tmp_path, hermetic_ctor):
        vis = VisualizeSkeleton(
            dataset='hemibrain:v1.2.1', neuron_layers=['DA1_lPN'],
            **self.ctor_kwargs(tmp_path))
        assert vis.client_type == 'neuprint'
        assert vis.skeleton_mesh_simplification == 0.90
        assert vis.exportable_meshes == []
        # light background -> Category10 palette, first entry #1f77b4
        assert len(vis.neuron_colors) == 1
        assert vis.neuron_colors[0].startswith('rgba(31, 119, 180')
        # base palette keeps the full 10-entry Category10 palette; the
        # neuron_colors slice used for layers is a prefix of it
        assert len(vis._base_neuron_colors) == 10
        assert tuple(vis._base_neuron_colors[:len(vis.neuron_colors)]) == tuple(vis.neuron_colors)
        assert len(hermetic_ctor) == 1
        assert isinstance(vis.fig_3d, go.Figure)
        assert os.path.isdir(vis.save_folder)
        params = Path(vis.save_folder) / 'parameters.txt'
        assert params.exists()
        text = params.read_text(encoding='utf-8')
        assert 'Dataset:          hemibrain:v1.2.1' in text
        assert vis.saveas == 'DA1_lPN'
        assert vis.layer_names == ['DA1_lPN']

    def test_string_layer_spec_parsing(self, tmp_path, hermetic_ctor):
        # 'A -> B' string splits into two layers; numeric token -> int
        vis = VisualizeSkeleton(
            dataset='hemibrain:v1.2.1', neuron_layers='DA1_lPN -> 202',
            **self.ctor_kwargs(tmp_path))
        assert vis.neuron_layers == ['DA1_lPN', 202]
        assert len(hermetic_ctor) == 2
        assert len(vis.layer_names) == 2

    def test_fafb_flywire_autodetect(self, tmp_path, hermetic_ctor):
        vis = VisualizeSkeleton(
            dataset='flywire_FAFB_v783',
            neuron_layers=['720575028614304'],
            cache_synapses=True,
            **self.ctor_kwargs(tmp_path, verbose=True))
        assert vis.client_type == 'flywire'          # auto-detected
        assert vis.version == 783                    # parsed from dataset
        # FAFB uses its dataset-native coordinate-bearing synapse
        # table as the shared connector/site cache source.
        assert vis.cache_synapses is True
        assert vis.verbose == 'full'                 # True normalized
        pipeline = vis._resolved_fafb_pipeline()
        expected = 0.90 if pipeline in {'fast', 'direct'} else 0.95
        assert vis.skeleton_mesh_simplification == expected
        assert vis._flywire_skeleton_access['is_fafb'] is True
        assert vis._flywire_skeleton_access['ready'] is False

    def test_dark_background_palette(self, tmp_path, hermetic_ctor):
        vis = VisualizeSkeleton(
            dataset='hemibrain:v1.2.1', neuron_layers=['DA1_lPN'],
            background_color='black',
            **self.ctor_kwargs(tmp_path))
        # dark background -> Set3 palette, first entry #8dd3c7
        assert vis.neuron_colors[0].startswith('rgba(141, 211, 199')

    def test_empty_neuron_layers_mesh_only(self, tmp_path, hermetic_ctor):
        vis = VisualizeSkeleton(
            dataset='hemibrain:v1.2.1', neuron_layers='',
            brain_mesh='native',
            **self.ctor_kwargs(tmp_path))
        assert vis.neuron_layers == []
        assert hermetic_ctor == []           # no layer fetches at all
        assert vis.saveas == 'brain_mesh'
        assert 'brain_mesh' in os.path.basename(vis.save_folder)

    def test_color_expansion_for_many_layers(self, tmp_path, hermetic_ctor):
        layers = [['DA1_lPN'] if i % 2 else 'DA1_lPN' for i in range(12)]
        vis = VisualizeSkeleton(
            dataset='hemibrain:v1.2.1', neuron_layers=layers,
            **self.ctor_kwargs(tmp_path))
        assert len(vis.neuron_colors) == 12
        assert len(vis.layer_names) == 12
        assert len(hermetic_ctor) == 12
        assert len(vis.synapse_colors) == 11

    def test_banc_dataset_accepted_with_resolution_default(self, tmp_path, hermetic_ctor):
        # BANC renders from the public bucket: the constructor no longer
        # rejects it, and the resolution knob defaults to the coarse L2.
        visualizer = VisualizeSkeleton(
            dataset='banc_v888', neuron_layers=['A00c'],
            **self.ctor_kwargs(tmp_path))
        assert visualizer.banc_skeleton_resolution == 'l2'
        assert visualizer._flywire_skeleton_access['ready'] is True

    def test_banc_local_data_missing_raises_instead_of_exiting(
            self, tmp_path, hermetic_ctor, monkeypatch):
        # A library path must not terminate the host process: missing local
        # data surfaces as an error the caller can render.
        monkeypatch.setattr(
            vs_module.BANC_file_converter, 'ensure_banc_data',
            lambda dataset, dataset_dir: False)
        with pytest.raises(RuntimeError, match='banc_v888'):
            VisualizeSkeleton(
                dataset='banc_v888', neuron_layers=['A00c'],
                **self.ctor_kwargs(tmp_path))

    def test_invalid_synapse_mode_raises(self, tmp_path, hermetic_ctor):
        with pytest.raises(ValueError, match='synapse_mode'):
            VisualizeSkeleton(
                dataset='hemibrain:v1.2.1', neuron_layers=['DA1_lPN'],
                synapse_mode='bogus', **self.ctor_kwargs(tmp_path))

    def test_invalid_legend_and_color_modes(self, tmp_path, hermetic_ctor):
        with pytest.raises(ValueError, match='legend_mode'):
            VisualizeSkeleton(
                dataset='hemibrain:v1.2.1', neuron_layers=['DA1_lPN'],
                legend_mode='huge', **self.ctor_kwargs(tmp_path))
        with pytest.raises(ValueError, match='color_mode'):
            VisualizeSkeleton(
                dataset='hemibrain:v1.2.1', neuron_layers=['DA1_lPN'],
                color_mode='rainbow', **self.ctor_kwargs(tmp_path))

    def test_manc_vnc_mesh_defaults(self, tmp_path, hermetic_ctor):
        # MANC auto-enables the VNC mesh unless brain_mesh='none'
        vis = VisualizeSkeleton(
            dataset='manc:v1.0', neuron_layers=['A00c'],
            brain_mesh='native', **self.ctor_kwargs(tmp_path))
        assert vis.vnc_mesh is True
        vis2 = VisualizeSkeleton(
            dataset='manc:v1.0', neuron_layers=['A00c'],
            brain_mesh='none', **self.ctor_kwargs(tmp_path / 'none'))
        assert vis2.vnc_mesh is False


# ---------------------------------------------------------------------------
# pre/post-mode warning banner
# ---------------------------------------------------------------------------
class TestPrePostWarningHtml:
    def test_banner_only_in_pre_post_mode(self):
        vis = make_vis(synapse_mode='pre_post', dataset='flywire_FAFB_v783')
        html = vis._pre_post_mode_warning_html()
        assert 'Pre/post-synaptic site mode.' in html
        assert 'flywire_FAFB_v783' in html
        assert 'synapse_mode=pre_post' in html

    def test_banner_empty_for_other_modes(self):
        assert make_vis(synapse_mode='connectors')._pre_post_mode_warning_html() == ''
        assert make_vis()._pre_post_mode_warning_html() == ''


# ---------------------------------------------------------------------------
# synapse size parsing / px / fold helpers + slider
# ---------------------------------------------------------------------------
class TestSynapseSizeHelpers:
    def test_parse_synapse_size_variants(self):
        parse = VisualizeSkeleton._parse_synapse_size
        assert parse('real') == 'real'
        assert parse('3') == 3.0
        assert parse('2.5x') == 2.5
        assert parse('2 x real') == 2.0
        assert parse(' 4 ') == 4.0
        assert parse('garbage') is None
        assert parse('') is None

    def test_synapse_size_px_clamped(self):
        assert make_vis(synapse_size='3')._synapse_size_px() == 3.0
        assert make_vis(synapse_size='25')._synapse_size_px() == 12.0
        assert make_vis(synapse_size='0.2')._synapse_size_px() == 1.0
        assert make_vis(synapse_size='real')._synapse_size_px() == 1.0
        assert make_vis(synapse_size='bogus')._synapse_size_px() == 1.0

    def test_synapse_size_fold(self):
        assert make_vis(synapse_size='3')._synapse_size_fold() == 3.0
        assert make_vis(synapse_size='0')._synapse_size_fold() == 0.0
        assert make_vis(synapse_size='real')._synapse_size_fold() == 1.0
        assert make_vis(synapse_size='junk')._synapse_size_fold() == 1.0

    def test_scatter_marker_size(self):
        assert make_vis(synapse_size='5')._scatter_synapse_marker_size() == 5.0
        assert make_vis(synapse_size='real')._scatter_synapse_marker_size() == 1.0

    def test_pre_post_site_scale(self, monkeypatch):
        vis = make_vis(synapse_size='2')
        monkeypatch.setattr(
            vis, '_estimate_pre_post_real_synapse_distance', lambda: 40.0)
        assert vis._pre_post_site_scale() == 80.0

    def test_sample_size_estimation_frame(self):
        sample = VisualizeSkeleton._sample_size_estimation_frame
        assert sample(None) is None
        assert sample(pd.DataFrame()) is None
        assert sample('not-a-frame') is None
        frame = pd.DataFrame({
            'bodyId_pre': [1, 2], 'bodyId_post': [3, 4],
            'x_pre': [0.0, np.nan], 'y_pre': [0.0, 1.0], 'z_pre': [0.0, 2.0],
            'x_post': [1.0, 2.0], 'y_post': [1.0, 2.0], 'z_post': [1.0, 2.0],
        })
        out = sample(frame)
        assert len(out) == 1  # NaN row dropped
        assert sample(frame.drop(columns=['x_pre'])) is None
        big = pd.DataFrame({
            'bodyId_pre': range(30), 'bodyId_post': range(30),
            'x_pre': np.arange(30.0), 'y_pre': np.arange(30.0),
            'z_pre': np.arange(30.0), 'x_post': np.arange(30.0),
            'y_post': np.arange(30.0), 'z_post': np.arange(30.0),
        })
        assert len(sample(big, max_rows=10)) == 10

    def test_connection_distance_values(self, monkeypatch):
        vis = make_vis()
        monkeypatch.setattr(vis, '_transform_site_df', lambda df: df)
        frame = pd.DataFrame({
            'bodyId_pre': [1, 1], 'bodyId_post': [2, 2],
            'x_pre': [0.0, 0.0], 'y_pre': [0.0, 0.0], 'z_pre': [0.0, 0.0],
            'x_post': [3.0, 0.0], 'y_post': [4.0, 0.0], 'z_post': [0.0, 0.0],
        })
        distances = vis._connection_distance_values(frame)
        assert distances.tolist() == [5.0]  # zero-distance row filtered
        assert vis._connection_distance_values(None).size == 0

    def test_transform_site_df_identity_and_paths(self, monkeypatch):
        site_df = pd.DataFrame({'x': [1.0], 'y': [2.0], 'z': [3.0]})
        # FlyWire template mode: no skeleton transform, tilt correction only
        vis = make_vis(dataset='flywire_FAFB_v783', brain_mesh='native')
        out = vis._transform_site_df(site_df)
        assert list(out.columns) == ['x', 'y', 'z']
        assert out is not None
        # hemibrain: monkeypatch navis.xform_brain
        vis2 = make_vis(dataset='hemibrain:v1.2.1', brain_mesh='native')
        calls = []

        def fake_xform(coords, source=None, target=None, **kw):
            calls.append((source, target))
            return coords + 1.0

        monkeypatch.setattr(navis, 'xform_brain', fake_xform)
        out2 = vis2._transform_site_df(site_df)
        assert calls == [('JRCFIB2018Fraw', 'JRCFIB2018F')]
        assert out2.loc[0, 'x'] == 2.0
        # empty / None passthrough
        assert vis2._transform_site_df(None) is None
        empty = pd.DataFrame(columns=['x', 'y', 'z'])
        assert vis2._transform_site_df(empty).empty

    def test_size_slider_scatter_mode(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode='markers',
            meta={'drocat_scatter_size_role': 'synapse',
                  'drocat_scatter_size_factor': 2.0}))
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode='markers',
            meta={'drocat_scatter_size_role': 'pre_post_site'}))
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0]))  # no meta
        vis = make_vis(backend='plotly', synapse_mode='scatter',
                       synapse_size='3', fig_3d=fig)
        vis._add_plotly_synapse_size_slider()
        assert len(fig.layout.sliders) == 1
        slider = vis._plotly_sliders[0]
        assert len(slider['steps']) == 12
        assert slider['active'] == 2  # 3px -> index 2
        step3 = slider['steps'][2]
        assert step3['args'][0]['marker.size'] == [6.0, 3.0]
        assert step3['args'][1] == [0, 1]

    def test_size_slider_skips_inapplicable_configs(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            meta={'drocat_scatter_size_role': 'synapse'}))
        # non-plotly backend
        vis = make_vis(backend='k3d', synapse_mode='scatter', fig_3d=fig)
        vis._add_plotly_synapse_size_slider()
        assert vis._plotly_sliders == []
        # pre_post without scatter
        vis2 = make_vis(backend='plotly', synapse_mode='pre_post',
                        pre_post_scatter=False, fig_3d=fig)
        vis2._add_plotly_synapse_size_slider()
        assert vis2._plotly_sliders == []
        # no eligible traces
        fig_empty = go.Figure()
        vis3 = make_vis(backend='plotly', synapse_mode='scatter',
                        fig_3d=fig_empty)
        vis3._add_plotly_synapse_size_slider()
        assert vis3._plotly_sliders == []
        # pre_post + scatter enabled builds a slider
        vis4 = make_vis(backend='plotly', synapse_mode='pre_post',
                        pre_post_scatter=True, synapse_size='12', fig_3d=fig)
        vis4._add_plotly_synapse_size_slider()
        assert vis4._plotly_sliders[0]['active'] == 11


# ---------------------------------------------------------------------------
# neuron info / viz-layer-info CSV exports
# ---------------------------------------------------------------------------
class TestNeuronInfoCsvExports:
    def test_save_neuron_info_csv_none_without_layers(self, tmp_path):
        vis = make_vis(neuron_dfs=[], save_folder=str(tmp_path), saveas='x')
        assert vis._save_neuron_info_csv() is None

    def test_save_neuron_info_csv_merges_layers(self, tmp_path):
        df1 = pd.DataFrame({'bodyId': [1, 2], 'type': ['a', 'b']})
        df1.insert(0, 'Unnamed: 0', [0, 1])  # serialized index column
        vis = make_vis(neuron_dfs=[df1, None,
                                   pd.DataFrame({'bodyId': [3]})],
                       save_folder=str(tmp_path), saveas='demo')
        path = vis._save_neuron_info_csv()
        assert path == os.path.join(str(tmp_path), 'demo_neuron_info.csv')
        out = pd.read_csv(path)
        assert 'Unnamed: 0' not in out.columns
        assert list(out['viz_layer']) == [0, 0, 2]
        assert list(out['bodyId']) == [1, 2, 3]

    def test_save_viz_layer_info_csv_source_rows_exclusivity(self, tmp_path):
        source = [{
            'layer': 'L1', 'neuron': 'DA1_lPN', 'color': '#ff0000',
            'synapse_color': '#00ff00',
            'pre_synaptic_color': '#0000ff',
            'post_synaptic_color': '#ffff00',
        }]
        # connector mode: blanks pre/post columns
        vis = make_vis(save_folder=str(tmp_path), synapse_mode='connectors',
                       skip_synapse=False, _viz_layer_info_input_rows=source)
        out = pd.read_csv(vis._save_viz_layer_info_csv())
        assert out.loc[0, 'synapse_color'] == '#00ff00'
        assert pd.isna(out.loc[0, 'pre_synaptic_color'])
        # pre_post mode: blanks synapse_color
        vis2 = make_vis(save_folder=str(tmp_path), synapse_mode='pre_post',
                        skip_synapse=False, _viz_layer_info_input_rows=source)
        out2 = pd.read_csv(vis2._save_viz_layer_info_csv())
        assert pd.isna(out2.loc[0, 'synapse_color'])
        assert out2.loc[0, 'pre_synaptic_color'] == '#0000ff'
        # skip_synapse: blanks all three
        vis3 = make_vis(save_folder=str(tmp_path), synapse_mode='connectors',
                        skip_synapse=True, _viz_layer_info_input_rows=source)
        out3 = pd.read_csv(vis3._save_viz_layer_info_csv())
        for col in ('synapse_color', 'pre_synaptic_color',
                    'post_synaptic_color'):
            assert pd.isna(out3.loc[0, col])

    def test_save_viz_layer_info_csv_direct_inputs(self, tmp_path):
        vis = make_vis(
            save_folder=str(tmp_path), synapse_mode='connectors',
            skip_synapse=False, color_mode='per_layer',
            neuron_colors=['rgba(10, 20, 30, 1.0)'],
            synapse_colors=['rgba(40, 50, 60, 1.0)'],
            _viz_input_neuron_layers=[['DA1_lPN', 202], 'VA1d'],
            _viz_input_custom_layer_names=['first', None],
            _viz_input_neuron_colors=None,
            _viz_input_synapse_colors=None,
        )
        out = pd.read_csv(vis._save_viz_layer_info_csv())
        assert len(out) == 3
        # both neurons of layer 0 share the layer label
        assert list(out['layer']) == ['first', 'first', '2']
        assert list(out['neuron']) == ['DA1_lPN', '202', 'VA1d']
        assert out.loc[0, 'color'] == 'rgba(10, 20, 30, 1.0)'
        assert out.loc[0, 'synapse_color'] == 'rgba(40, 50, 60, 1.0)'

    def test_save_viz_layer_info_csv_string_layers_and_palettes(self, tmp_path):
        vis = make_vis(
            save_folder=str(tmp_path), synapse_mode='connectors',
            skip_synapse=False, color_mode='per_neuron',
            neuron_colors=[], synapse_colors=[],
            _viz_input_neuron_layers='A -> B',
            _viz_input_neuron_colors=['red', 'blue'],
            _viz_input_synapse_colors=(255, 0, 0),  # one RGB tuple color
        )
        out = pd.read_csv(vis._save_viz_layer_info_csv())
        assert list(out['neuron']) == ['A', 'B']
        assert out.loc[0, 'color'] == 'red'
        assert out.loc[1, 'color'] == 'blue'
        assert out.loc[0, 'synapse_color'] == 'rgba(255, 0, 0, 0.6)'
        # empty string layer input -> header-only artifact
        vis2 = make_vis(
            save_folder=str(tmp_path), synapse_mode='connectors',
            skip_synapse=False, color_mode='per_layer',
            neuron_colors=['rgba(1, 2, 3, 1.0)'],
            synapse_colors=['rgba(4, 5, 6, 1.0)'],
            _viz_input_neuron_layers='   ',
        )
        out2 = pd.read_csv(vis2._save_viz_layer_info_csv())
        assert out2.empty
        assert list(out2.columns)[0] == 'layer'


# ---------------------------------------------------------------------------
# color standardization / darkening extras
# ---------------------------------------------------------------------------
class TestColorStandardizationExtras:
    def test_standardize_color_input_numpy_and_fallbacks(self):
        vis = make_vis()
        arr = np.array(['red', 'blue'])
        assert vis._standardize_color_input(arr) == [
            'rgba(255, 0, 0, 1.0)', 'rgba(0, 0, 255, 1.0)']
        assert vis._standardize_color_input('not-a-color') == [
            'rgba(128, 128, 128, 1.0)']
        assert vis._standardize_color_input((255, 0, 0)) == [
            'rgba(255, 0, 0, 1.0)']
        # 3-tuple of invalid colors: first treated as one RGB(A) attempt,
        # then falls through to per-element handling
        assert vis._standardize_color_input(('nope1', 'nope2', 'nope3')) == [
            'rgba(128, 128, 128, 1.0)'] * 3
        # non-string/list input -> final fallback
        assert vis._standardize_color_input(12345) == [
            'rgba(128, 128, 128, 1.0)']
        assert vis._standardize_color_input([]) == [
            'rgba(128, 128, 128, 1.0)']

    def test_standardize_mesh_color_input_variants(self):
        vis = make_vis(mesh_roi=['A', 'B', 'C'])
        assert vis._standardize_mesh_color_input('red') == 'rgba(255, 0, 0, 0.1)'
        assert vis._standardize_mesh_color_input('nope') == 'rgba(100, 100, 100, 0.1)'
        assert vis._standardize_mesh_color_input(np.array([10, 20, 30])) == \
            'rgba(10, 20, 30, 0.1)'
        # continuous sampling over mesh_roi count
        out = vis._standardize_mesh_color_input(
            ['rgba(255, 0, 0, 0.1)', 'rgba(0, 0, 255, 0.1)'],
            continuous=True)
        assert len(out) == 3
        # invalid 3-tuple falls through to per-element handling
        assert vis._standardize_mesh_color_input(
            ('nope1', 'nope2', 'nope3')) == [
            'rgba(100, 100, 100, 0.1)'] * 3
        # list of colors
        assert vis._standardize_mesh_color_input(['red', 'blue']) == [
            'rgba(255, 0, 0, 0.1)', 'rgba(0, 0, 255, 0.1)']
        # final fallback
        assert vis._standardize_mesh_color_input(99) == 'rgba(100, 100, 100, 0.1)'

    def test_is_custom_mesh_color_specified(self):
        vis = make_vis(mesh_color=(100, 100, 100))
        assert vis._is_custom_mesh_color_specified() is False
        vis.mesh_color = 'rgba(100, 100, 100, 0.1)'
        assert vis._is_custom_mesh_color_specified() is False
        vis.mesh_color = 'red'
        assert vis._is_custom_mesh_color_specified() is True
        vis._original_mesh_color = (100, 100, 100, 0.1)
        assert vis._is_custom_mesh_color_specified() is False

    def test_darken_color_manual_fallbacks(self, monkeypatch):
        def raiser(color):
            raise ValueError('unparseable')
        monkeypatch.setattr(vs_module, 'extract_rgba_tuple', raiser)
        vis = make_vis()
        assert vis._darken_color('#ff8040', 0.5) == 'rgba(127, 64, 32, 1.0)'
        assert vis._darken_color('#f84', 0.5) == 'rgba(128, 128, 128, 1.0)'
        assert vis._darken_color('red', 0.5) == 'rgba(128, 128, 128, 1.0)'
        assert vis._darken_color((200, 100, 50, 0.5), 0.5) == \
            'rgba(100, 50, 25, 0.5)'
        assert vis._darken_color([300, 0, 0], 1.0) == 'rgba(255, 0, 0, 1.0)'
        assert vis._darken_color(5, 0.5) == 'rgba(128, 128, 128, 1.0)'


# ---------------------------------------------------------------------------
# neuron lookup keys + layer-map override linking/resolution
# ---------------------------------------------------------------------------
class TestNeuronLookupAndOverrides:
    def test_normalize_neuron_lookup_keys(self):
        vis = make_vis()
        assert vis._normalize_neuron_lookup_keys(None) == []
        assert vis._normalize_neuron_lookup_keys(5) == [5, '5']
        assert vis._normalize_neuron_lookup_keys(np.int64(7)) == [7, '7']
        assert vis._normalize_neuron_lookup_keys(float('nan')) == []
        assert vis._normalize_neuron_lookup_keys('') == []
        assert vis._normalize_neuron_lookup_keys('nan') == []
        assert vis._normalize_neuron_lookup_keys('15832.0') == \
            [15832, '15832.0', '15832']
        big = vis._normalize_neuron_lookup_keys('720575028614304.0')
        assert big[0] == 720575028614304
        assert vis._normalize_neuron_lookup_keys('DA1_lPN') == ['DA1_lPN']
        assert vis._normalize_neuron_lookup_keys('-42.00') == \
            [-42, '-42.00', '-42']

    def test_link_layer_map_color_overrides(self):
        ndf = pd.DataFrame({
            'bodyId': [101, 102], 'type': ['T1', 'T2'],
            'name': ['n1', 'n2'],
        })
        vis = make_vis(
            neuron_dfs=[ndf, None, pd.DataFrame()],
            _neuron_color_overrides={'T1': 'red'},
            _neuron_pre_color_overrides={102: 'blue'},
            _neuron_synapse_color_overrides={'T2': 'green'},
            _neuron_synapse_color_overrides_by_layer={0: {102: 'green2'}},
            _neuron_post_color_overrides={},
        )
        vis._link_layer_map_color_overrides_to_neuron_dfs()
        overrides = vis._neuron_color_overrides
        assert overrides[101] == 'red' and overrides['101'] == 'red'
        assert overrides['n1'] == 'red'
        pre = vis._neuron_pre_color_overrides
        assert pre['102'] == 'blue' and pre['T2'] == 'blue'
        syn_layered = vis._neuron_synapse_color_overrides_by_layer
        # layered value wins over the global one during alias expansion
        assert syn_layered[0][102] == 'green2'
        assert syn_layered[0]['T2'] == 'green2'
        assert vis._neuron_synapse_color_overrides[102] == 'green2'

    def test_resolve_neuron_override_layer_priority(self):
        vis = make_vis(
            _neuron_color_overrides={'42': 'globalcolor'},
            _neuron_color_overrides_by_layer={0: {'42': 'layercolor'}},
        )
        assert vis._resolve_neuron_override(
            '_neuron_color_overrides', '42', 0) == 'layercolor'
        assert vis._resolve_neuron_override(
            '_neuron_color_overrides', 42, 1) == 'globalcolor'
        assert vis._resolve_neuron_override(
            '_neuron_color_overrides', None, 0) is None
        # non-numeric layer index does not crash
        assert vis._resolve_neuron_override(
            '_neuron_color_overrides', '42', 'weird') == 'globalcolor'

    def test_resolve_neuron_color_paths(self):
        vis = make_vis(
            neuron_colors=['rgba(1, 2, 3, 1.0)'], color_mode='per_layer')
        assert vis._resolve_neuron_color('x', 0) == 'rgba(1, 2, 3, 1.0)'
        assert vis._resolve_neuron_color('x', 5) == 'rgba(1, 2, 3, 1.0)'
        vis2 = make_vis(neuron_colors=[], color_mode='per_layer')
        assert vis2._resolve_neuron_color('x', 0) == 'rgba(0, 0, 0, 0.2)'
        vis3 = make_vis(
            neuron_colors=['rgba(1, 2, 3, 1.0)'], color_mode='per_neuron',
            _per_neuron_colors={'42': 'rgba(9, 9, 9, 1.0)'})
        assert vis3._resolve_neuron_color(42, 0) == 'rgba(9, 9, 9, 1.0)'
        vis3._neuron_color_overrides = {'42': 'rgba(8, 8, 8, 1.0)'}
        assert vis3._resolve_neuron_color(42, 0) == 'rgba(8, 8, 8, 1.0)'

    def test_resolve_synapse_pre_post_colors(self):
        vis = make_vis(
            neuron_colors=['rgba(1, 1, 1, 1.0)'],
            synapse_colors=['rgba(5, 5, 5, 1.0)'],
            color_mode='per_layer',
        )
        assert vis._resolve_synapse_color('a', 0) == 'rgba(5, 5, 5, 1.0)'
        assert vis._resolve_synapse_color('a', 3) == 'rgba(5, 5, 5, 1.0)'
        vis.synapse_colors = []
        assert vis._resolve_synapse_color('a', 0) == 'rgba(0, 0, 0, 0.6)'
        vis._neuron_synapse_color_overrides = {'a': 'rgba(7, 7, 7, 1.0)'}
        assert vis._resolve_synapse_color('a', 0) == 'rgba(7, 7, 7, 1.0)'
        assert vis._resolve_pre_color('a', 0) == 'rgba(1, 1, 1, 1.0)'
        assert vis._resolve_post_color('a', 0) == 'rgba(1, 1, 1, 1.0)'
        vis._neuron_pre_color_overrides = {'a': 'rgba(2, 2, 2, 1.0)'}
        vis._neuron_post_color_overrides = {'a': 'rgba(3, 3, 3, 1.0)'}
        assert vis._resolve_pre_color('a', 0) == 'rgba(2, 2, 2, 1.0)'
        assert vis._resolve_post_color('a', 0) == 'rgba(3, 3, 3, 1.0)'


# ---------------------------------------------------------------------------
# dataset metadata primary ROIs + mesh_roi pattern expansion
# ---------------------------------------------------------------------------
class TestMetadataRoisAndPatterns:
    def _write_metadata(self, tmp_path, folder, payload, name=None):
        folder_path = tmp_path / 'datasets' / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        fname = name or f'{folder}_metadata.json'
        import json as _json
        if isinstance(payload, str):
            (folder_path / fname).write_text(payload)
        else:
            (folder_path / fname).write_text(_json.dumps(payload))

    def test_primary_rois_from_preferred_file(self, tmp_path):
        self._write_metadata(tmp_path, 'hemibrain_v1_2_1', {
            'primaryRois': ['AL(L)', 'NotPrimary', 'AL(L)', 'EB', '  '],
        })
        vis = make_vis(dataset='hemibrain:v1.2.1', script_path=str(tmp_path))
        assert vis._get_metadata_primary_rois() == ['AL(L)', 'EB']

    def test_primary_rois_alternate_keys_and_files(self, tmp_path):
        self._write_metadata(tmp_path, 'hemibrain_v1_2_1',
                             '{not valid json',
                             name=f'hemibrain_v1_2_1_metadata.json')
        self._write_metadata(tmp_path, 'hemibrain_v1_2_1',
                             {'roi_coverage': {'roi_list': ['PB', 'FB']}},
                             name='other_metadata.json')
        vis = make_vis(dataset='hemibrain:v1.2.1', script_path=str(tmp_path))
        assert vis._get_metadata_primary_rois() == ['PB', 'FB']

    def test_primary_rois_flywire_prefers_male_cns(self, tmp_path):
        self._write_metadata(tmp_path, 'male-cns_v0_9',
                             {'primaryRois': ['CRE', 'GNG']})
        vis = make_vis(dataset='flywire_FAFB_v783', script_path=str(tmp_path))
        assert vis._get_metadata_primary_rois() == ['CRE', 'GNG']

    def test_primary_rois_missing_returns_empty(self, tmp_path):
        vis = make_vis(dataset='hemibrain:v1.2.1', script_path=str(tmp_path))
        assert vis._get_metadata_primary_rois() == []
        # non-dict payloads are skipped
        self._write_metadata(tmp_path, 'hemibrain_v1_2_1', [1, 2, 3])
        assert vis._get_metadata_primary_rois() == []

    def _pattern_vis(self, tmp_path, monkeypatch, available, primary):
        vis = make_vis(dataset='hemibrain:v1.2.1', script_path=str(tmp_path))
        monkeypatch.setattr(
            vis, '_get_available_rois',
            lambda use_cache=True, fetch_online=True: list(available))
        monkeypatch.setattr(vis, '_get_metadata_primary_rois', lambda: primary)
        return vis

    def test_expand_patterns_all_primary_regex_literal(self, tmp_path, monkeypatch):
        available = ['AL(L)', 'EB', 'ME(L)', 'ME(R)', 'LH']
        vis = self._pattern_vis(tmp_path, monkeypatch, available, ['EB', 'FB'])
        assert vis._expand_mesh_roi_patterns([]) == []
        assert vis._expand_mesh_roi_patterns('LH') == ['LH']
        assert vis._expand_mesh_roi_patterns(['all']) == sorted(available)
        assert vis._expand_mesh_roi_patterns(['primary']) == ['EB', 'FB']
        vis2 = self._pattern_vis(tmp_path, monkeypatch, available, [])
        # metadata-less fallback uses the hard-coded primary patterns
        assert vis2._expand_mesh_roi_patterns(['primary']) == \
            ['EB', 'AL(L)', 'ME(L)', 'ME(R)']
        assert vis2._expand_mesh_roi_patterns(['ME.*']) == ['ME(L)', 'ME(R)']
        assert vis2._expand_mesh_roi_patterns(['ZZ.*']) == []
        # invalid regex is kept as a literal
        assert vis2._expand_mesh_roi_patterns(['[']) == ['[']
        # dedupe across mixed inputs
        assert vis2._expand_mesh_roi_patterns(['LH', 'EB', 'LH']) == \
            ['LH', 'EB']

    def test_list_available_rois(self, monkeypatch, capsys):
        vis = make_vis(dataset='hemibrain:v1.2.1')
        seen = {}

        def fake_get(use_cache=True, fetch_online=True):
            seen['use_cache'] = use_cache
            seen['fetch_online'] = fetch_online
            return ['EB', 'AL(L)']

        monkeypatch.setattr(vis, '_get_available_rois', fake_get)
        assert vis.list_available_rois(refresh=True, fetch_online=False) == \
            ['EB', 'AL(L)']
        assert seen == {'use_cache': False, 'fetch_online': False}
        monkeypatch.setattr(vis, '_get_available_rois',
                            lambda use_cache=True, fetch_online=True: [])
        assert vis.list_available_rois() == []


# ---------------------------------------------------------------------------
# template / transform info
# ---------------------------------------------------------------------------
class TestTemplateInfo:
    def test_get_template_info_all_datasets(self):
        import flybrains
        cases = [
            (dict(dataset='hemibrain:v1.2.1', brain_mesh='native'),
             'JRCFIB2018Fraw', 'JRCFIB2018F', flybrains.JRCFIB2018F),
            (dict(dataset='optic-lobe:v1.1', brain_mesh='native'),
             'JRCFIB2022Mraw', 'JRCFIB2022M', flybrains.JRCFIB2022M),
            (dict(dataset='manc:v1.0', brain_mesh='native'),
             'MANC', 'MANC', flybrains.MANC),
            (dict(dataset='male-cns:v0.9', brain_mesh='native'),
             'JRCFIB2022Mraw', 'JRCFIB2022M', flybrains.JRCFIB2022M),
        ]
        for attrs, source, target, template in cases:
            info = make_vis(**attrs)._get_template_info()
            assert info['source'] == source
            assert info['target'] == target
            assert info['template_obj'] is template
            assert info['mesh_name']
        flywire_native = make_vis(
            dataset='flywire_FAFB_v783', brain_mesh='native'
        )._get_template_info()
        assert flywire_native['skip_transform'] is True
        # Outline selections never change the skeleton transform path.
        fafb_outline = make_vis(
            dataset='flywire_FAFB_v783', brain_mesh='FAFB')._get_template_info()
        assert fafb_outline['source'] == 'FLYWIRE'
        assert fafb_outline['target'] == 'FLYWIRE'
        assert fafb_outline['skip_transform'] is True
        fallback = make_vis(dataset='weird:v9', brain_mesh='native')
        info = fallback._get_template_info()
        assert info['target'] == 'JRCFIB2018F'

    def test_needs_skeleton_transform(self):
        assert make_vis(dataset='hemibrain:v1.2.1', brain_mesh='none') \
            ._needs_skeleton_transform() is False
        assert make_vis(dataset='hemibrain:v1.2.1', brain_mesh='native') \
            ._needs_skeleton_transform() is True
        assert make_vis(dataset='flywire_FAFB_v783', brain_mesh='native') \
            ._needs_skeleton_transform() is False
        # MANC native space: source == target identity
        assert make_vis(dataset='manc:v1.0', brain_mesh='native') \
            ._needs_skeleton_transform() is False

    def test_get_vnc_template_info(self):
        vnc = make_vis(dataset='manc:v1.0')._get_vnc_template_info()
        assert vnc is not None and 'mesh' in vnc and vnc['mesh_name']
        vnc2 = make_vis(dataset='male-cns:v0.9')._get_vnc_template_info()
        assert vnc2 is not None
        assert make_vis(dataset='hemibrain:v1.2.1')._get_vnc_template_info() is None


# ---------------------------------------------------------------------------
# WebDriver session helpers (fake driver, no browser)
# ---------------------------------------------------------------------------
class FakeWebDriver:
    def __init__(self):
        self.scripts = []

    def execute_script(self, code):
        self.scripts.append(code)


class TestWebDriverSessionHelpers:
    def _session(self):
        sess = object.__new__(vs_module.WebDriverExportSession)
        sess.driver = FakeWebDriver()
        return sess

    def test_set_trace_visibility(self):
        sess = self._session()
        sess.set_trace_visibility([0, 2, 99], total_traces=3)
        js = sess.driver.scripts[0]
        assert '[true, false, true]' in js
        assert 'Plotly.restyle' in js

    def test_update_layout(self):
        sess = self._session()
        sess.update_layout({'scene': {'camera': None}})
        js = sess.driver.scripts[0]
        assert 'Plotly.relayout' in js and 'scene' in js

    def test_auto_crop_image(self):
        sess = self._session()
        img = make_white_image_with_block()
        cropped = sess._auto_crop_image(img, margin=2)
        assert cropped.size == (44, 34)  # 40x30 block + 2px margin each side
        from PIL import Image
        blank = Image.new('RGB', (20, 20), (255, 255, 255))
        assert sess._auto_crop_image(blank) is blank


# ---------------------------------------------------------------------------
# _xform_neurons_safe (per-neuron transform with fallback)
# ---------------------------------------------------------------------------
class TestXformNeuronsSafe:
    def _patch_registry(self, monkeypatch):
        import flybrains  # noqa: F401  (pre-import: avoids re-registration)
        from navis.transforms import registry as _registry

        def fake_seq(source, target):
            return ([source, target], ['fake-xf'])

        fake_seq.cache_clear = lambda: None
        monkeypatch.setattr(_registry, 'shortest_bridging_seq', fake_seq)

    def test_transform_success_and_passthrough(self, monkeypatch):
        self._patch_registry(monkeypatch)
        monkeypatch.setattr(navis, 'xform', lambda n, transform=None: n)
        vis = make_vis(verbose=False)
        n1 = make_chain_neuron(body_id='1')
        n2 = make_chain_neuron(body_id='2')
        out = vis._xform_neurons_safe(
            navis.NeuronList([n1, n2]), source='SRC', target='TGT',
            layer_label='L0')
        assert len(out) == 2
        # single neuron (not NeuronList) + private progress bar
        out2 = vis._xform_neurons_safe(n1, source='SRC', target='TGT')
        assert len(out2) == 1
        # non-neuron objects (volumes) pass through untouched
        import trimesh
        vol = navis.Volume(trimesh.creation.box(), name='v')
        out3 = vis._xform_neurons_safe([vol], source='SRC', target='TGT')
        # volumes bypass navis.xform entirely (pass-through branch)
        assert len(out3) == 1

    def test_transform_failure_keeps_original(self, monkeypatch):
        self._patch_registry(monkeypatch)

        def bad_xform(n, transform=None):
            raise RuntimeError('boom')

        monkeypatch.setattr(navis, 'xform', bad_xform)
        vis = make_vis(verbose='full')
        neuron = make_chain_neuron(body_id='9')
        out = vis._xform_neurons_safe(
            navis.NeuronList([neuron]), source='SRC', target='TGT',
            layer_label='L1', compact_progress=True)
        assert len(out) == 1 and out[0] is neuron

    def test_plain_list_unwrap(self, monkeypatch):
        self._patch_registry(monkeypatch)
        monkeypatch.setattr(navis, 'xform', lambda n, transform=None: n)
        vis = make_vis()
        neurons = [make_chain_neuron(body_id='1'), make_chain_neuron(body_id='2')]
        out = vis._xform_neurons_safe(neurons, source='A', target='B')
        assert len(out) == 2


# ---------------------------------------------------------------------------
# plot_skeleton end-to-end (hermetic: prepared synthetic skeletons)
# ---------------------------------------------------------------------------
class TestPlotSkeletonEndToEnd:
    def _make_vis(self, tmp_path, monkeypatch, prepared, **extra):
        ndf = pd.DataFrame({
            'bodyId': list(prepared.keys()),
            'type': ['DA1_lPN'] * len(prepared),
            'name': [f"n{bid}" for bid in prepared.keys()],
        })
        attrs = dict(
            dataset='hemibrain:v1.2.1', client_type='neuprint',
            neuron_layers=['DA1_lPN'], layer_names=['DA1_lPN'],
            neuron_dfs=[ndf],
            skeleton_mode='tube', skeleton_mesh_simplification=0.9,
            neuprint_skeleton_pipeline='fast',
            cache_neurons=False, force_API_fetching=False,
            soma_radius_cap=None, smooth_skeleton=False,
            mirror_on_contralateral=False,
            show_soma=False, show_skeleton_radius=True,
            show_connectors=False,
            backend='plotly', fig_3d=go.Figure(),
            neuron_colors=['rgba(31, 119, 180, 1.0)'] * len(prepared),
            _neuron_colors_have_explicit_alpha=False,
            legend_mode='layer', color_mode='per_layer',
            brain_mesh='native', script_path=str(tmp_path),
            exportable_meshes=[], synapse_mode='connectors',
            skip_synapse=True,
        )
        attrs.update(extra)
        vis = make_vis(**attrs)
        monkeypatch.setattr(
            vis, '_prepare_neuprint_skeletons_for_render',
            lambda body_ids, fine, use_cache, cache: (prepared, False))
        # Hermetic identity transforms (no H5 downloads).
        import flybrains  # noqa: F401  (pre-import: avoids re-registration)
        from navis.transforms import registry as _registry

        def fake_seq(source, target):
            return ([source, target], ['fake-xf'])

        fake_seq.cache_clear = lambda: None
        monkeypatch.setattr(_registry, 'shortest_bridging_seq', fake_seq)
        monkeypatch.setattr(navis, 'xform', lambda n, transform=None: n)
        return vis

    def test_tube_plotly_layer_legend(self, tmp_path, monkeypatch):
        prepared = {42: make_chain_neuron(body_id='42')}
        vis = self._make_vis(tmp_path, monkeypatch, prepared,
                             soma_radius_cap=1000.0)
        assert vis.plot_skeleton() == 0
        assert len(vis.fig_3d.data) >= 1
        trace = vis.fig_3d.data[0]
        assert trace.legendgroup == 'DA1_lPN'
        assert trace.showlegend is True
        assert len(vis.exportable_meshes) == 1
        assert vis._neuron_legend_labels_by_layer[0]['42'] == 'DA1_lPN'

    def test_line_mode_single_legend(self, tmp_path, monkeypatch):
        prepared = {
            42: make_chain_neuron(body_id='42'),
            43: make_chain_neuron(body_id='43'),
        }
        vis = self._make_vis(tmp_path, monkeypatch, prepared,
                             skeleton_mode='line', legend_mode='single',
                             skeleton_mesh_simplification=0.0)
        assert vis.plot_skeleton() == 0
        names = [t.name for t in vis.fig_3d.data]
        assert any('(42)_DA1_lPN' in name for name in names)
        assert all(t.showlegend for t in vis.fig_3d.data)

    def test_type_legend_and_per_neuron_colors(self, tmp_path, monkeypatch):
        prepared = {42: make_chain_neuron(body_id='42')}
        vis = self._make_vis(
            tmp_path, monkeypatch, prepared,
            legend_mode='type', color_mode='per_neuron',
            neuron_colors=['rgba(200, 10, 10, 0.5)'],
        )
        assert vis.plot_skeleton() == 0
        # type legend uses the cleaned type label
        assert any(t.name == 'DA1_lPN' for t in vis.fig_3d.data)
        # alpha < 1.0 triggers the opaque legend-patch trace
        assert any(getattr(t, 'hoverinfo', None) == 'skip'
                   for t in vis.fig_3d.data)

    def test_explicit_alpha_legend_fix(self, tmp_path, monkeypatch):
        prepared = {42: make_chain_neuron(body_id='42')}
        vis = self._make_vis(
            tmp_path, monkeypatch, prepared,
            _neuron_colors_have_explicit_alpha=True,
            neuron_colors=['rgba(31, 119, 180, 0.4)'],
        )
        assert vis.plot_skeleton() == 0
        assert any(getattr(t, 'hoverinfo', None) == 'skip'
                   for t in vis.fig_3d.data)

    def test_missing_prepared_skeleton_skips_layer(self, tmp_path, monkeypatch):
        vis = self._make_vis(tmp_path, monkeypatch, {})
        # empty prepared map -> aggregate preprocessing yields nothing;
        # the layer reports a failed fetch and continues
        monkeypatch.setattr(
            vis, '_prepare_neuprint_skeletons_for_render',
            lambda body_ids, fine, use_cache, cache: ({}, False))
        assert vis.plot_skeleton() == 0
        assert len(vis.fig_3d.data) == 0

    def test_invalid_legend_mode_raises(self, tmp_path, monkeypatch):
        prepared = {42: make_chain_neuron(body_id='42')}
        vis = self._make_vis(tmp_path, monkeypatch, prepared,
                             legend_mode='bogus')
        # object.__new__ bypassed __post_init__ validation
        with pytest.raises(ValueError, match='legend_mode'):
            vis.plot_skeleton()


# ---------------------------------------------------------------------------
# Batch 2: synapse plotting pipeline (plot_synapses, flywire reader,
# synapse cache plumbing, pre/post site mode)
# ---------------------------------------------------------------------------
def make_conn_frame():
    """Synthetic paired synapse frame: 101<->202 plus one out-of-set row."""
    return pd.DataFrame({
        'bodyId_pre': [101, 999, 202],
        'bodyId_post': [202, 202, 101],
        'x_pre': [0.0, 10.0, 100.0],
        'y_pre': [0.0, 10.0, 100.0],
        'z_pre': [0.0, 10.0, 100.0],
        'x_post': [30.0, 50.0, 0.0],
        'y_post': [40.0, 60.0, 0.0],
        'z_post': [0.0, 10.0, 0.0],
        'weight': [1, 1, 1],
    })


def write_flywire_parquet(tmp_path, df, dataset='flywire'):
    """Write a synthetic FlyWire synapse table at the datasets-dir layout."""
    d = tmp_path / 'datasets' / dataset
    d.mkdir(parents=True, exist_ok=True)
    path = d / f'{dataset}_synapse_table.parquet'
    df.to_parquet(path)
    return path


class TestPlotSynapsesNeuprint:
    def _make(self, tmp_path, monkeypatch, fetcher=None, **extra):
        ndf0 = pd.DataFrame({'bodyId': [101], 'type': ['A0'],
                             'name': ['n101']})
        ndf1 = pd.DataFrame({'bodyId': [202], 'type': ['B0'],
                             'name': ['n202']})
        attrs = dict(
            dataset='hemibrain:v1.2.1', client_type='neuprint',
            neuron_layers=['L0', 'L1'], layer_names=['L0', 'L1'],
            layer_criteria=[{'type': 'A0'}, {'type': 'B0'}],
            neuron_dfs=[ndf0, ndf1],
            synapse_colors=['rgba(255, 0, 0, 0.6)'],
            synapse_mode='scatter', synapse_size='1',
            uniform_synapse_size=False, min_synapse_num=0,
            cache_synapses=False, script_path=str(tmp_path),
            save_folder=str(tmp_path), saveas='t', output_format='csv',
            backend='plotly', fig_3d=go.Figure(), brain_mesh='none',
            synapse_alpha=0.6, exportable_meshes=[], client=None,
            synapse_criteria=None, server=None, version=None,
            skip_synapse=False, pre_post_scatter=False,
            legend_mode='layer', color_mode='per_layer',
            neuron_colors=['rgba(31, 119, 180, 1.0)',
                           'rgba(44, 160, 44, 1.0)'],
            FAFB_template_correction=False,
        )
        attrs.update(extra)
        vis = make_vis(**attrs)
        if fetcher is None:
            def fetcher(**kwargs):
                fetcher.calls.append(kwargs)
                return make_conn_frame()
            fetcher.calls = []
        monkeypatch.setattr(vs_module, 'fetch_synapse_connections', fetcher)
        return vis, fetcher

    def test_scatter_mode_fetch_filter_export_slider(self, tmp_path,
                                                     monkeypatch):
        vis, fetcher = self._make(tmp_path, monkeypatch)
        assert vis.plot_synapses() == 0
        assert len(fetcher.calls) == 1
        # one scatter trace; out-of-set row 999 filtered by body-ID sets
        assert len(vis.fig_3d.data) == 1
        trace = vis.fig_3d.data[0]
        assert trace.meta['drocat_scatter_size_role'] == 'synapse'
        assert len(trace.x) == 1
        assert trace.name == 'synapses 0 -> 1 (1)'
        # pixel-size slider attached for scatter mode
        assert len(vis._plotly_sliders) == 1
        assert vis.fig_3d.layout.sliders
        # merged export written next to save_folder
        out = tmp_path / 't_synapses.csv'
        assert out.exists()
        saved = pd.read_csv(out)
        assert list(saved['viz_layer']) == ['0->1']
        assert len(saved) == 1

    def test_scatter_transform_branch(self, tmp_path, monkeypatch):
        calls = []

        def identity_xform(df, source=None, target=None):
            calls.append((source, target))
            return df

        vis, _ = self._make(tmp_path, monkeypatch, brain_mesh='native')
        monkeypatch.setattr(navis, 'xform_brain', identity_xform)
        assert vis.plot_synapses() == 0
        assert calls == [('JRCFIB2018Fraw', 'JRCFIB2018F')]

    def test_scatter_per_point_color_array(self, tmp_path, monkeypatch):
        # a per-point color list of matching length becomes a color array
        vis, _ = self._make(
            tmp_path, monkeypatch,
            synapse_colors=[['rgba(255, 0, 0, 0.6)']])
        assert vis.plot_synapses() == 0
        trace = vis.fig_3d.data[0]
        assert list(trace.marker.color) == ['rgba(255, 0, 0, 1.0)']

    def test_sphere_mode_uniform_real_size(self, tmp_path, monkeypatch):
        vis, _ = self._make(tmp_path, monkeypatch, synapse_mode='sphere',
                            uniform_synapse_size=True, synapse_size='real')
        assert vis.plot_synapses() == 0
        # one Mesh3d + one dummy legend scatter
        assert len(vis.fig_3d.data) == 2
        dummy = vis.fig_3d.data[-1]
        assert dummy.showlegend is True
        assert dummy.x == (None,)
        assert len(vis.exportable_meshes) == 1
        assert vis.exportable_meshes[0].metadata['export_role'] == 'synapse'
        # solid modes carry no pixel slider
        assert vis._plotly_sliders == []

    def test_cone_mode_per_neuron_overrides_split_groups(self, tmp_path,
                                                         monkeypatch):
        base = make_conn_frame()
        extra = base.iloc[[0]].copy()
        extra['bodyId_pre'] = 103
        frame = pd.concat([base, extra], ignore_index=True)

        def fetcher(**kwargs):
            return frame.copy()

        ndf0 = pd.DataFrame({'bodyId': [101, 103],
                             'type': ['A0', 'A0'],
                             'name': ['n101', 'n103']})
        vis, _ = self._make(
            tmp_path, monkeypatch, fetcher=fetcher,
            neuron_dfs=[ndf0, pd.DataFrame({'bodyId': [202],
                                            'type': ['B0'],
                                            'name': ['n202']})],
            synapse_mode='cone', synapse_size='2',
            _neuron_synapse_color_overrides={
                '101': 'rgba(0, 255, 0, 0.8)'},
        )
        assert vis.plot_synapses() == 0
        # two color groups (override vs layer color) + dummy legend trace
        meshes = [t for t in vis.fig_3d.data
                  if getattr(t, 'type', '') == 'mesh3d']
        assert len(meshes) == 2
        assert len(vis.exportable_meshes) == 2

    def test_tetrahedron_mode(self, tmp_path, monkeypatch):
        vis, _ = self._make(tmp_path, monkeypatch,
                            synapse_mode='tetrahedron')
        assert vis.plot_synapses() == 0
        assert len(vis.exportable_meshes) == 1

    def test_skip_synapse_short_circuit(self, tmp_path, monkeypatch):
        vis, fetcher = self._make(tmp_path, monkeypatch, skip_synapse=True)
        assert vis.plot_synapses() is None
        assert fetcher.calls == []
        assert len(vis.fig_3d.data) == 0

    def test_empty_frame_skips_layer(self, tmp_path, monkeypatch):
        def empty_fetcher(**kwargs):
            return make_conn_frame().iloc[0:0]

        vis, _ = self._make(tmp_path, monkeypatch, fetcher=empty_fetcher)
        assert vis.plot_synapses() == 0
        assert len(vis.fig_3d.data) == 0
        assert not (tmp_path / 't_synapses.csv').exists()

    def test_fetch_failure_returns_no_synapses(self, tmp_path, monkeypatch):
        def bad_fetcher(**kwargs):
            raise RuntimeError('offline')

        vis, _ = self._make(tmp_path, monkeypatch, fetcher=bad_fetcher)
        assert vis.plot_synapses() == 0
        assert len(vis.fig_3d.data) == 0

    def test_cache_enabled_query_roundtrip_memo(self, tmp_path, monkeypatch):
        vis, fetcher = self._make(tmp_path, monkeypatch,
                                  cache_synapses=True)
        assert vis.plot_synapses() == 0
        # second run on the same vis hits the in-memory query memo
        vis.fig_3d = go.Figure()
        assert vis.plot_synapses() == 0
        assert len(fetcher.calls) == 1
        # the broad query was persisted under the tmp cache root
        cache_root = tmp_path / 'cache' / 'hemibrain_v1_2_1' / 'synapses'
        assert any(cache_root.rglob('*.parquet'))

    def test_partial_then_full_pair_cache(self, tmp_path, monkeypatch):
        base = make_conn_frame()
        extra = base.iloc[[0]].copy()
        extra['bodyId_pre'] = 103
        broad = pd.concat([base, extra], ignore_index=True)

        def fetcher(**kwargs):
            fetcher.calls.append(kwargs)
            return broad.copy()
        fetcher.calls = []

        ndf0 = pd.DataFrame({'bodyId': [101, 103],
                             'type': ['A0', 'A0'],
                             'name': ['n101', 'n103']})
        ndf1 = pd.DataFrame({'bodyId': [202], 'type': ['B0'],
                             'name': ['n202']})
        vis, _ = self._make(tmp_path, monkeypatch, fetcher=fetcher,
                            cache_synapses=True,
                            neuron_dfs=[ndf0, ndf1])
        # seed only one of the two pairs into the pair cache
        seed = make_conn_frame()[make_conn_frame()['bodyId_pre'] == 101]
        vis._save_cached_synapses(seed, attempted_pairs=[('101', '202')],
                                  persist_pairs=True)
        assert vis.plot_synapses() == 0
        # partial cache -> one fetch, merged + deduped to 2 connectors
        assert len(fetcher.calls) == 1
        assert vis.fig_3d.data[0].name == 'synapses 0 -> 1 (2)'
        # now seed the second pair: a full cache hit needs no fetch
        rest = broad[broad['bodyId_pre'] == 103]
        vis._save_cached_synapses(rest, attempted_pairs=[('103', '202')],
                                  persist_pairs=True)
        vis.fig_3d = go.Figure()
        vis._synapse_query_memory = {}
        assert vis.plot_synapses() == 0
        assert len(fetcher.calls) == 1
        assert vis.fig_3d.data[0].name == 'synapses 0 -> 1 (2)'

    def test_save_cached_synapses_early_returns(self, tmp_path, monkeypatch):
        vis, _ = self._make(tmp_path, monkeypatch)
        # disabled cache -> no-op
        vis._save_cached_synapses(make_conn_frame())
        vis.dataset = 'flywire'
        vis.cache_synapses = True
        # flywire master table is canonical -> no pair files
        vis._save_cached_synapses(make_conn_frame(),
                                  attempted_pairs=[('101', '202')])
        assert not (tmp_path / 'cache').exists()
        vis.dataset = 'hemibrain:v1.2.1'
        # persist_pairs=False -> broad query path, still no pair files
        vis._save_cached_synapses(make_conn_frame(),
                                  attempted_pairs=[('101', '202')],
                                  persist_pairs=False)
        assert not (tmp_path / 'cache').exists()

    def test_load_cached_synapses_disabled_lists_all_pairs(self, tmp_path,
                                                           monkeypatch):
        vis, _ = self._make(tmp_path, monkeypatch)
        cached, missing = vis._load_cached_synapses(['101'], ['202'])
        assert cached is None
        assert missing == [('101', '202')]

    def test_synapse_cache_path_and_spec(self, tmp_path, monkeypatch):
        vis, _ = self._make(tmp_path, monkeypatch, cache_synapses=True)
        path = vis._get_synapse_cache_path('101', '202')
        assert path.endswith('101_202.parquet')
        spec = vis._synapse_query_spec(source_ids=['101'],
                                       target_ids=['202'])
        assert spec['dataset'] == 'hemibrain:v1.2.1'
        assert spec['source_ids'] == ['101']
        assert spec['min_total_weight'] == 0

    def test_xlsx_export(self, tmp_path, monkeypatch):
        pytest.importorskip('openpyxl')
        vis, _ = self._make(tmp_path, monkeypatch, output_format='xlsx')
        assert vis.plot_synapses() == 0
        assert (tmp_path / 't_synapses.xlsx').exists()

    def test_invalid_output_format_raises(self, tmp_path, monkeypatch):
        vis, _ = self._make(tmp_path, monkeypatch, output_format='hdf5')
        with pytest.raises(ValueError, match='output_format'):
            vis.plot_synapses()


class TestFlywireConnectionFrame:
    def _make(self, tmp_path, **extra):
        attrs = dict(
            dataset='flywire', client_type='flywire',
            script_path=str(tmp_path), min_synapse_num=0,
            brain_mesh='native', FAFB_template_correction=True,
        )
        attrs.update(extra)
        return make_vis(**attrs)

    def test_nm_scale_filters_and_rename(self, tmp_path):
        df = pd.DataFrame({
            'pre_root_id': np.array([101, 101], dtype=np.int64),
            'post_root_id': np.array([202, 202], dtype=np.int64),
            'x_pre': [0.0, 5.0], 'y_pre': [0.0, 5.0],
            'z_pre': [26000.0, 26100.0],
            'x_post': [30.0, 35.0], 'y_post': [40.0, 45.0],
            'z_post': [26000.0, 26100.0],
            'weight': [5, 1],
        })
        write_flywire_parquet(tmp_path, df)
        vis = self._make(tmp_path)
        assert str(vis._get_synapse_table_path()).endswith(
            'flywire_synapse_table.parquet')
        frame = vis._read_flywire_connection_frame(
            source_ids={'101'}, target_ids={'202'}, min_synapse_num=2)
        assert frame is not None and len(frame) == 1
        assert frame['bodyId_pre'].iloc[0] == '101'
        assert frame['bodyId_post'].iloc[0] == '202'
        # z max > 10000 -> nanometre scale, coordinates untouched
        assert frame['x_pre'].iloc[0] == 0.0
        assert frame['z_pre'].iloc[0] == 26000.0

    def test_voxel_scale_alt_columns(self, tmp_path):
        df = pd.DataFrame({
            'pre_pt_root_id': np.array([101], dtype=np.int64),
            'post_pt_root_id': np.array([202], dtype=np.int64),
            'pre_x': [10.0], 'pre_y': [20.0], 'pre_z': [300.0],
            'post_x': [11.0], 'post_y': [21.0], 'post_z': [301.0],
            'syn_count': [3],
        })
        write_flywire_parquet(tmp_path, df)
        vis = self._make(tmp_path)
        frame = vis._read_flywire_connection_frame()
        assert frame is not None and len(frame) == 1
        # z max < 10000 -> voxel units scaled by (4, 4, 40)
        assert frame['x_pre'].iloc[0] == pytest.approx(40.0)
        assert frame['z_pre'].iloc[0] == pytest.approx(12000.0)
        assert frame['x_post'].iloc[0] == pytest.approx(44.0)

    def test_touching_ids_filter(self, tmp_path):
        df = pd.DataFrame({
            'pre_root_id': np.array([101, 303, 444], dtype=np.int64),
            'post_root_id': np.array([202, 101, 555], dtype=np.int64),
            'x_pre': [0.0, 1.0, 2.0], 'y_pre': [0.0, 1.0, 2.0],
            'z_pre': [26000.0, 26000.0, 26000.0],
            'x_post': [3.0, 4.0, 5.0], 'y_post': [3.0, 4.0, 5.0],
            'z_post': [26000.0, 26000.0, 26000.0],
            'weight': [1, 1, 1],
        })
        write_flywire_parquet(tmp_path, df)
        vis = self._make(tmp_path)
        frame = vis._read_flywire_connection_frame(touching_ids={'101'})
        assert frame is not None and len(frame) == 2

    def test_missing_file_and_bad_schemas(self, tmp_path):
        vis = self._make(tmp_path)
        assert vis._read_flywire_connection_frame() is None
        # no id columns
        write_flywire_parquet(tmp_path, pd.DataFrame({
            'x_pre': [1.0], 'y_pre': [1.0], 'z_pre': [26000.0],
            'x_post': [1.0], 'y_post': [1.0], 'z_post': [26000.0],
        }))
        assert vis._read_flywire_connection_frame() is None
        # id columns but missing coordinates
        write_flywire_parquet(tmp_path, pd.DataFrame({
            'pre_root_id': np.array([1], dtype=np.int64),
            'post_root_id': np.array([2], dtype=np.int64),
            'x_pre': [1.0],
        }))
        vis2 = self._make(tmp_path)
        assert vis2._read_flywire_connection_frame() is None

    def test_memo_reuse(self, tmp_path):
        df = pd.DataFrame({
            'pre_root_id': np.array([101], dtype=np.int64),
            'post_root_id': np.array([202], dtype=np.int64),
            'x_pre': [0.0], 'y_pre': [0.0], 'z_pre': [26000.0],
            'x_post': [30.0], 'y_post': [40.0], 'z_post': [26000.0],
            'weight': [1],
        })
        write_flywire_parquet(tmp_path, df)
        vis = self._make(tmp_path)
        first = vis._read_flywire_connection_frame()
        assert first is not None and len(first) == 1
        # identical arguments hit the run-scoped memory memo
        second = vis._read_flywire_connection_frame()
        assert second is not None
        pd.testing.assert_frame_equal(first, second)
        assert len(vis._flywire_synapse_frame_memory) == 1

    def test_non_numeric_filter_literals_fall_back(self, tmp_path):
        df = pd.DataFrame({
            'pre_root_id': np.array([101], dtype=np.int64),
            'post_root_id': np.array([202], dtype=np.int64),
            'x_pre': [0.0], 'y_pre': [0.0], 'z_pre': [26000.0],
            'x_post': [30.0], 'y_post': [40.0], 'z_post': [26000.0],
            'weight': [1],
        })
        write_flywire_parquet(tmp_path, df)
        vis = self._make(tmp_path)
        frame = vis._read_flywire_connection_frame(source_ids={'abc'})
        assert frame is not None and frame.empty

    def test_path_helper_missing_dir(self, tmp_path):
        vis = self._make(tmp_path, dataset='flywire:v9')
        assert vis._get_synapse_table_path() is None


class TestPlotSynapsesFlywire:
    def _flywire_parquet(self, tmp_path):
        df = pd.DataFrame({
            'pre_root_id': np.array([101, 999], dtype=np.int64),
            'post_root_id': np.array([202, 202], dtype=np.int64),
            'x_pre': [0.0, 10.0], 'y_pre': [0.0, 10.0],
            'z_pre': [26000.0, 26100.0],
            'x_post': [30.0, 50.0], 'y_post': [40.0, 60.0],
            'z_post': [26000.0, 26100.0],
            'weight': [1, 1],
        })
        write_flywire_parquet(tmp_path, df)

    def _make(self, tmp_path, monkeypatch, **extra):
        ndf0 = pd.DataFrame({'bodyId': [101], 'type': ['A0'],
                             'name': ['n101']})
        ndf1 = pd.DataFrame({'bodyId': [202], 'type': ['B0'],
                             'name': ['n202']})
        attrs = dict(
            dataset='flywire', client_type='flywire',
            neuron_layers=['L0', 'L1'], layer_names=['L0', 'L1'],
            layer_criteria=[None, None], neuron_dfs=[ndf0, ndf1],
            synapse_colors=['rgba(255, 0, 0, 0.6)'],
            synapse_mode='scatter', synapse_size='1',
            uniform_synapse_size=False, min_synapse_num=0,
            cache_synapses=False, script_path=str(tmp_path),
            save_folder=str(tmp_path), saveas='t', output_format='csv',
            backend='plotly', fig_3d=go.Figure(), brain_mesh='native',
            synapse_alpha=0.6, exportable_meshes=[], client=None,
            synapse_criteria=None, server=None, version=None,
            skip_synapse=False, pre_post_scatter=False,
            legend_mode='layer', color_mode='per_layer',
            neuron_colors=['rgba(31, 119, 180, 1.0)',
                           'rgba(44, 160, 44, 1.0)'],
            FAFB_template_correction=True,
        )
        attrs.update(extra)
        return make_vis(**attrs)

    def test_scatter_with_tilt_correction(self, tmp_path, monkeypatch):
        self._flywire_parquet(tmp_path)
        vis = self._make(tmp_path, monkeypatch)
        assert vis.plot_synapses() == 0
        assert len(vis.fig_3d.data) == 1
        assert len(vis.fig_3d.data[0].x) == 1
        saved = pd.read_csv(tmp_path / 't_synapses.csv')
        assert len(saved) == 1
        # tilt correction rotated the midpoint away from the raw average
        assert saved['viz_layer'].iloc[0] == '0->1'

    def test_sphere_mode_flywire(self, tmp_path, monkeypatch):
        self._flywire_parquet(tmp_path)
        vis = self._make(tmp_path, monkeypatch, synapse_mode='sphere',
                         synapse_size='real')
        assert vis.plot_synapses() == 0
        assert len(vis.exportable_meshes) == 1

    def test_missing_table_skips_layer(self, tmp_path, monkeypatch):
        vis = self._make(tmp_path, monkeypatch, verbose='full')
        assert vis.plot_synapses() == 0
        assert len(vis.fig_3d.data) == 0


class TestPrePostSites:
    def _make(self, tmp_path, monkeypatch, fetcher=None, **extra):
        ndf0 = pd.DataFrame({'bodyId': [101], 'type': ['A0'],
                             'name': ['n101']})
        ndf1 = pd.DataFrame({'bodyId': [202], 'type': ['B0'],
                             'name': ['n202']})
        attrs = dict(
            dataset='hemibrain:v1.2.1', client_type='neuprint',
            neuron_layers=['L0', 'L1'], layer_names=['L0', 'L1'],
            layer_criteria=[{'type': 'A0'}, {'type': 'B0'}],
            neuron_dfs=[ndf0, ndf1],
            synapse_colors=['rgba(255, 0, 0, 0.6)'],
            synapse_mode='pre_post', synapse_size='1',
            uniform_synapse_size=False, min_synapse_num=0,
            cache_synapses=False, script_path=str(tmp_path),
            save_folder=str(tmp_path), saveas='t', output_format='csv',
            backend='plotly', fig_3d=go.Figure(), brain_mesh='none',
            synapse_alpha=0.6, exportable_meshes=[], client=None,
            synapse_criteria=None, server=None, version=None,
            skip_synapse=False, pre_post_scatter=False,
            legend_mode='layer', color_mode='per_layer',
            neuron_colors=['rgba(31, 119, 180, 1.0)',
                           'rgba(44, 160, 44, 1.0)'],
            FAFB_template_correction=False,
        )
        attrs.update(extra)
        vis = make_vis(**attrs)
        if fetcher is None:
            def fetcher(**kwargs):
                fetcher.calls.append(kwargs)
                return make_conn_frame()
            fetcher.calls = []
        monkeypatch.setattr(vs_module, 'fetch_synapse_connections', fetcher)
        return vis, fetcher

    def test_layer_legend_solid_sites(self, tmp_path, monkeypatch):
        vis, _ = self._make(tmp_path, monkeypatch)
        assert vis.plot_synapses() == 0
        names = [t.name for t in vis.fig_3d.data]
        assert set(names) == {'L0_pre', 'L0_post', 'L1_pre', 'L1_post'}
        # pre sites are cones, post sites are spheres (exportable meshes)
        assert len(vis.exportable_meshes) == 4
        assert all(m.metadata['export_role'] == 'synapse'
                   for m in vis.exportable_meshes)
        assert vis._pre_post_seen_legend_groups
        # real-distance sizing from the synthetic pair (distance 50.0)
        assert vis._pre_post_real_synapse_distance == pytest.approx(50.0)
        saved = pd.read_csv(tmp_path / 't_synapses.csv')
        assert set(saved['viz_layer']) == {'0:pre', '0:post',
                                           '1:pre', '1:post'}
        # solid mode: no pixel slider
        assert vis._plotly_sliders == []

    def test_single_legend_and_type_labels(self, tmp_path, monkeypatch):
        vis, _ = self._make(tmp_path, monkeypatch, legend_mode='single')
        assert vis.plot_synapses() == 0
        names = {t.name for t in vis.fig_3d.data}
        assert 'A0_L0_pre' in names
        assert 'B0_L1_post' in names

    def test_scatter_mode_symbols_and_slider(self, tmp_path, monkeypatch):
        vis, _ = self._make(tmp_path, monkeypatch, pre_post_scatter=True)
        assert vis.plot_synapses() == 0
        traces = list(vis.fig_3d.data)
        assert len(traces) == 4
        symbols = {t.marker.symbol for t in traces}
        assert symbols == {'diamond', 'circle'}
        assert all(t.meta.get('drocat_scatter_size_role') == 'pre_post_site'
                   for t in traces)
        assert len(vis._plotly_sliders) == 1

    def test_fallback_distance_when_no_connectors(self, tmp_path,
                                                  monkeypatch):
        def bad_fetcher(**kwargs):
            raise RuntimeError('offline')

        vis, _ = self._make(tmp_path, monkeypatch, fetcher=bad_fetcher)
        assert vis.plot_synapses() == 0
        assert len(vis.fig_3d.data) == 0
        assert vis._pre_post_real_synapse_distance == 40.0
        assert vis._pre_post_real_synapse_distance_sample_count == 0

    def test_single_layer_neuprint_bidirectional_estimate(
            self, tmp_path, monkeypatch):
        ndf0 = pd.DataFrame({'bodyId': [101], 'type': ['A0'],
                             'name': ['n101']})
        vis, _ = self._make(
            tmp_path, monkeypatch,
            neuron_layers=['L0'], layer_names=['L0'],
            layer_criteria=[{'type': 'A0'}], neuron_dfs=[ndf0],
            neuron_colors=['rgba(31, 119, 180, 1.0)'],
            synapse_colors=['rgba(255, 0, 0, 0.6)'],
        )
        assert vis.plot_synapses() == 0
        # single-layer estimate queries both directions: 101->202 (dist 50)
        # and 202->101 (dist sqrt(30000)); the baseline is their mean
        expected = (50.0 + np.sqrt(30000.0)) / 2.0
        assert vis._pre_post_real_synapse_distance == pytest.approx(expected)
        names = {t.name for t in vis.fig_3d.data}
        assert names == {'L0_pre', 'L0_post'}

    def test_flywire_sites_single_layer(self, tmp_path, monkeypatch):
        df = pd.DataFrame({
            'pre_root_id': np.array([101, 303], dtype=np.int64),
            'post_root_id': np.array([202, 101], dtype=np.int64),
            'x_pre': [0.0, 5.0], 'y_pre': [0.0, 5.0],
            'z_pre': [26000.0, 26100.0],
            'x_post': [30.0, 0.0], 'y_post': [40.0, 0.0],
            'z_post': [26000.0, 26000.0],
            'weight': [1, 1],
        })
        write_flywire_parquet(tmp_path, df)
        ndf0 = pd.DataFrame({'bodyId': [101], 'type': ['A0'],
                             'name': ['n101']})
        vis = make_vis(
            dataset='flywire', client_type='flywire',
            neuron_layers=['L0'], layer_names=['L0'],
            layer_criteria=[None], neuron_dfs=[ndf0],
            synapse_colors=['rgba(255, 0, 0, 0.6)'],
            synapse_mode='pre_post', synapse_size='1',
            uniform_synapse_size=False, min_synapse_num=0,
            cache_synapses=False, script_path=str(tmp_path),
            save_folder=str(tmp_path), saveas='t', output_format='csv',
            backend='plotly', fig_3d=go.Figure(), brain_mesh='native',
            synapse_alpha=0.6, exportable_meshes=[], client=None,
            synapse_criteria=None, server=None, version=None,
            skip_synapse=False, pre_post_scatter=False,
            legend_mode='layer', color_mode='per_layer',
            neuron_colors=['rgba(31, 119, 180, 1.0)'],
            FAFB_template_correction=False,
        )
        assert vis.plot_synapses() == 0
        names = {t.name for t in vis.fig_3d.data}
        assert names == {'L0_pre', 'L0_post'}
        assert vis._pre_post_real_synapse_distance_sample_count >= 1

    def test_pre_post_size_criteria_variants(self, tmp_path, monkeypatch):
        vis, _ = self._make(tmp_path, monkeypatch)
        assert vis._pre_post_size_criteria(0) == {'type': 'A0'}
        # out-of-range layer with no metadata frame
        assert vis._pre_post_size_criteria(5) is None
        # None criteria falls back to a body-id NeuronCriteria; without a
        # default neuprint Client the fallback degrades gracefully to None.
        # Clear any process-global default client so this assertion is
        # deterministic regardless of test ordering.
        try:
            import neuprint
            neuprint.clear_default_client()
        except Exception:
            pass
        vis.layer_criteria = [None]
        criteria = vis._pre_post_size_criteria(0)
        assert criteria is None
        # empty metadata frame -> no criteria
        vis.neuron_dfs = [pd.DataFrame(columns=['bodyId'])]
        assert vis._pre_post_size_criteria(0) is None

    def test_site_label_and_owner_helpers(self, tmp_path, monkeypatch):
        vis, _ = self._make(tmp_path, monkeypatch)
        clean = VisualizeSkeleton._clean_pre_post_type_label
        assert clean('s-LNv_15832.0') == 's-LNv'
        assert clean('DA2_mPN') == 'DA2_mPN'
        assert clean(float('nan')) == ''
        assert clean(None) == ''
        # owner/type labels resolve through the metadata frame
        assert vis._pre_post_site_owner_label(101, 0) == 'A0'
        assert vis._pre_post_site_type_label(101, 0) == 'A0'
        assert vis._pre_post_site_owner_label(77777, 0) == '77777'
        # identity honors an existing skeleton legend label map
        vis._neuron_legend_labels_by_layer = {0: {'101': 'custom'}}
        assert vis._pre_post_site_owner_identity(101, 0, 'L0') == 'custom'
        # site rows filtered by owner
        site_df = pd.DataFrame({
            'x': [0.0, 1.0], 'y': [0.0, 1.0], 'z': [0.0, 1.0],
            'neuron_id': [101, 202],
        })
        owned = vis._site_rows_for_owner(site_df, 101)
        assert len(owned) == 1
        assert vis._site_rows_for_owner(None, 101) is None
        # legend tuple contract: group carries the role, rank is offset
        group, label, rank = vis._pre_post_site_legend('pre', 0, 'L0', 101)
        assert group.startswith('pre_post:pre:')
        assert label.endswith('_pre')
        assert rank >= 0

    def test_connection_distance_values_empty(self, tmp_path, monkeypatch):
        vis, _ = self._make(tmp_path, monkeypatch)
        distances = vis._connection_distance_values(
            pd.DataFrame({'foo': [1]}))
        assert distances.size == 0


# ---------------------------------------------------------------------------
# Batch 3: plot_mesh, save_figure, PNG timeout export, simplified figures,
# WebDriver multi-view export session — all hermetic (fake sessions,
# monkeypatched write_image, synthetic meshes redirected to tmp_path).
# ---------------------------------------------------------------------------
import flybrains  # noqa: E402
from types import SimpleNamespace  # noqa: E402


def make_noise_png(path, size=(120, 120), seed=0):
    """Deterministic noise PNG (~45KB), large enough to pass >10KB checks."""
    from PIL import Image
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr, 'RGB').save(str(path), 'PNG')


def make_small_trimesh(extent=200.0):
    import trimesh
    return trimesh.creation.box(extents=(extent, extent, extent))


def write_roi_mesh_json(tmp_path, dataset, roi='AL(R)', subdir='meshes'):
    """Write a synthetic ROI mesh JSON into the dataset mesh cache dir."""
    norm = dataset.replace(':', '_').replace('.', '_')
    d = tmp_path / 'cache' / norm / subdir
    d.mkdir(parents=True, exist_ok=True)
    path = d / (roi + '.json')
    navis.Volume(make_small_trimesh(), name=roi).to_json(str(path))
    return path


def fake_template_info(mesh_name='JRCFIB2018F (hemibrain)',
                       source='JRCFIB2018Fraw', target='JRCFIB2018F'):
    """Tiny synthetic _get_template_info() replacement (no big flybrains mesh)."""
    obj = SimpleNamespace(
        mesh=navis.Volume(make_small_trimesh(100.0), name='template'))
    return {'source': source, 'target': target, 'template_obj': obj,
            'mesh_name': mesh_name}


def make_mesh_vis(tmp_path, **over):
    """VisualizeSkeleton configured for plot_mesh() tests."""
    attrs = dict(
        backend='plotly',
        script_path=str(tmp_path),
        fig_3d=go.Figure(),
        brain_mesh='none',
        vnc_mesh=False,
        mesh_color='rgba(200, 200, 200, 0.3)',
        mesh_alpha=0.3,
        roi_mesh_simplification=0.0,
        mirror_on_contralateral=False,
        brain_mesh_color='rgba(200, 230, 240, 0.1)',
        vnc_mesh_color='rgba(200, 230, 240, 0.1)',
        token=None,
        client=None,
        exportable_meshes=[],
        FAFB_template_correction=False,
        mesh_roi=[],
    )
    attrs.update(over)
    vis = make_vis(**attrs)
    vis._get_available_rois = lambda use_cache=True, fetch_online=True: ['AL(R)']
    return vis


def patch_write_image(monkeypatch, mode='ok'):
    """Monkeypatch go.Figure.write_image (kaleido never invoked)."""
    calls = []

    def fake_write_image(self, file=None, **kwargs):
        calls.append(file)
        if mode == 'raise':
            raise RuntimeError('kaleido exploded')
        if mode == 'nofile':
            return
        if mode == 'blank':
            from PIL import Image
            Image.new('RGB', (20, 20), (255, 255, 255)).save(str(file), 'PNG')
            return
        make_noise_png(file)

    monkeypatch.setattr(go.Figure, 'write_image', fake_write_image)
    return calls


def make_fake_session_class(monkeypatch, fail_attempts=0,
                            error_msg='generic driver error'):
    """Replace vs_module.WebDriverExportSession with a deterministic fake."""
    state = {'attempts': 0, 'loaded': []}

    class FakeSession:
        def __init__(self, width=1200, height=900, scale=2, timeout=300,
                     render_wait=None):
            self._render_wait = 0.05
            self.scale = scale

        def __enter__(self):
            state['attempts'] += 1
            if state['attempts'] <= fail_attempts:
                raise RuntimeError(error_msg)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def load_html(self, path, wait_for_render=True, render_wait=3,
                      background_color=None):
            state['loaded'].append(path)

        def set_camera(self, eye=None, up=None, center=None):
            self.camera = (eye, up, center)

        def screenshot(self, path):
            make_noise_png(path)

    monkeypatch.setattr(vs_module, 'WebDriverExportSession', FakeSession)
    return state


class TestPlotMesh:
    def test_returns_early_when_mesh_roi_none(self, tmp_path):
        vis = make_mesh_vis(tmp_path, mesh_roi=None)
        assert vis.plot_mesh() is None
        assert len(vis.fig_3d.data) == 0

    def test_returns_early_when_nothing_to_plot(self, tmp_path):
        vis = make_mesh_vis(tmp_path, mesh_roi=[])
        assert vis.plot_mesh() is None
        assert len(vis.fig_3d.data) == 0

    def test_roi_mesh_only_hemibrain(self, tmp_path):
        write_roi_mesh_json(tmp_path, 'hemibrain:v1.2.1')
        vis = make_mesh_vis(tmp_path, mesh_roi=['AL(R)'])
        assert vis.plot_mesh() == 0
        assert len(vis.fig_3d.data) >= 1
        assert vis.fig_3d.data[0].legendgroup == 'roi_mesh:AL(R)'
        assert len(vis.exportable_meshes) == 1
        assert vis.exportable_meshes[0].metadata['export_role'] == 'roi'

    def test_roi_color_list_short_fallback(self, tmp_path):
        write_roi_mesh_json(tmp_path, 'hemibrain:v1.2.1', roi='AL(R)')
        write_roi_mesh_json(tmp_path, 'hemibrain:v1.2.1', roi='AL(L)')
        vis = make_mesh_vis(tmp_path, mesh_roi=['AL(R)', 'AL(L)'],
                            mesh_color=['rgba(255, 0, 0, 0.4)'])
        assert vis.plot_mesh() == 0
        assert len(vis.exportable_meshes) == 2

    def test_roi_simplification_attempted(self, tmp_path):
        write_roi_mesh_json(tmp_path, 'hemibrain:v1.2.1')
        vis = make_mesh_vis(tmp_path, mesh_roi=['AL(R)'],
                            roi_mesh_simplification=0.5)
        assert vis.plot_mesh() == 0
        assert len(vis.fig_3d.data) >= 1

    def test_hemibrain_native_transform_and_brain(
            self, tmp_path, monkeypatch):
        write_roi_mesh_json(tmp_path, 'hemibrain:v1.2.1')
        monkeypatch.setattr(
            navis, 'xform_brain',
            lambda m, source=None, target=None, **kw: m)
        vis = make_mesh_vis(tmp_path, mesh_roi=['AL(R)'],
                            brain_mesh='native')
        vis._get_template_info = lambda: fake_template_info()
        assert vis.plot_mesh() == 0
        # ROI trace + brain mesh trace at minimum (contralateral mirroring
        # is disabled: no mirrored '(L)' twin is appended).
        names = [getattr(t, 'name', '') for t in vis.fig_3d.data]
        assert any('JRCFIB2018F' in n for n in names)
        roles = [m.metadata.get('export_role') for m in vis.exportable_meshes]
        assert 'roi' in roles and 'brain' in roles

    def test_roi_missing_legacy_fallback_and_not_found(self, tmp_path):
        # legacy fallback dir used when dataset-cache file missing
        legacy = tmp_path / 'navis_roi_meshes_json' / 'primary_rois'
        legacy.mkdir(parents=True)
        navis.Volume(make_small_trimesh(), name='LO(R)').to_json(
            str(legacy / 'LO(R).json'))
        vis = make_mesh_vis(tmp_path, mesh_roi=['LO(R)', 'NOPE(R)'])
        assert vis.plot_mesh() == 0
        assert len(vis.exportable_meshes) == 1  # only legacy ROI loaded

    def test_flywire_fafb_outline_plots_flywire_mesh(self, tmp_path):
        # 'fafb' draws the FLYWIRE outline directly (no JRC2018F scene
        # transform, ROIs untouched).
        vis = make_mesh_vis(tmp_path, dataset='flywire', mesh_roi=['AL(R)'],
                            brain_mesh='FAFB')
        vis._get_template_info = lambda: fake_template_info(
            'FLYWIRE (native FAFB coordinates)', 'FLYWIRE', 'FLYWIRE')
        assert vis.plot_mesh() == 0
        assert vis.mesh_roi == ['AL(R)']
        names = [getattr(t, 'name', '') for t in vis.fig_3d.data]
        assert any('FLYWIRE' in n for n in names)

    def test_flywire_transformed_cache_hit_with_tilt(self, tmp_path):
        write_roi_mesh_json(tmp_path, 'flywire', roi='AL(R)',
                            subdir='meshes_transformed/FLYWIRE')
        # Current-pipeline marker: without it the entry would be treated
        # as a pre-tilt cache and ignored.
        (tmp_path / 'cache' / 'flywire' / 'meshes_transformed' / 'FLYWIRE' /
         TRANSFORMED_ROI_CACHE_MARKER).write_text('v2\n')
        vis = make_mesh_vis(tmp_path, dataset='flywire', mesh_roi=['AL(R)'],
                            brain_mesh='native',
                            FAFB_template_correction=True)
        vis._get_template_info = lambda: fake_template_info(
            'FLYWIRE (native FAFB coordinates)', 'FLYWIRE', 'FLYWIRE')
        assert vis.plot_mesh() == 0
        roles = [m.metadata.get('export_role') for m in vis.exportable_meshes]
        assert roles.count('roi') == 1 and 'brain' in roles

    def test_flywire_stale_transformed_cache_ignored(
            self, tmp_path, monkeypatch):
        # A markerless transformed-cache dir predates the tilt baking: the
        # entry is ignored, and the raw cache entry is re-transformed and
        # rewritten with the marker in place.
        write_roi_mesh_json(tmp_path, 'flywire', roi='AL(R)')
        write_roi_mesh_json(tmp_path, 'flywire', roi='AL(R)',
                            subdir='meshes_transformed/FLYWIRE')
        calls = []
        monkeypatch.setattr(
            navis, 'xform_brain',
            lambda m, source=None, target=None, **kw:
                calls.append((source, target)) or m)
        vis = make_mesh_vis(tmp_path, dataset='flywire', mesh_roi=['AL(R)'],
                            brain_mesh='native')
        vis._get_template_info = lambda: fake_template_info(
            'FLYWIRE (native FAFB coordinates)', 'FLYWIRE', 'FLYWIRE')
        assert vis.plot_mesh() == 0
        roles = [m.metadata.get('export_role') for m in vis.exportable_meshes]
        assert roles.count('roi') == 1
        assert calls == [('JRCFIB2022Mraw', 'FLYWIRE')]
        cache_dir = (tmp_path / 'cache' / 'flywire' /
                     'meshes_transformed' / 'FLYWIRE')
        assert (cache_dir / TRANSFORMED_ROI_CACHE_MARKER).exists()

    def test_flywire_raw_cache_roi_gets_transformed_and_cached(
            self, tmp_path, monkeypatch):
        write_roi_mesh_json(tmp_path, 'flywire', roi='AL(R)')
        monkeypatch.setattr(
            navis, 'xform_brain',
            lambda m, source=None, target=None, **kw: m)
        vis = make_mesh_vis(tmp_path, dataset='flywire', mesh_roi=['AL(R)'],
                            brain_mesh='native')
        vis._get_template_info = lambda: fake_template_info(
            'FLYWIRE (native FAFB coordinates)', 'FLYWIRE', 'FLYWIRE')
        assert vis.plot_mesh() == 0
        cached = tmp_path / 'cache' / 'flywire' / 'meshes_transformed' / \
            'FLYWIRE' / 'AL(R).json'
        assert cached.exists()

    def test_male_cns_brain_only_extraction_and_vnc(self, tmp_path,
                                                    monkeypatch):
        # synthetic CNS mesh: one face below the Z cutoff (brain), one above
        import trimesh
        verts = np.array([
            [0.0, 0.0, 100000.0], [1000.0, 0.0, 100000.0],
            [0.0, 1000.0, 100000.0],
            [0.0, 0.0, 400000.0], [1000.0, 0.0, 400000.0],
            [0.0, 1000.0, 400000.0],
        ])
        split = trimesh.Trimesh(vertices=verts, faces=[[0, 1, 2], [3, 4, 5]],
                                process=False)
        monkeypatch.setattr(flybrains, 'JRCFIB2022M',
                            SimpleNamespace(mesh=split))
        vis = make_mesh_vis(tmp_path, dataset='male-cns:v0.9',
                            brain_mesh='native', vnc_mesh=True,
                            brain_mesh_color='auto', vnc_mesh_color='auto',
                            background_color='rgba(0, 0, 0, 1.0)')
        vis._get_vnc_template_info = lambda: {
            'mesh': navis.Volume(make_small_trimesh(80.0), name='vnc'),
            'mesh_name': 'JRCFIB2022M (VNC)'}
        assert vis.plot_mesh() == 0
        names = [getattr(t, 'name', '') for t in vis.fig_3d.data]
        assert any('brain' in n for n in names)
        assert any('VNC' in n for n in names)

    def test_male_cns_brain_extraction_fallback_full_mesh(
            self, tmp_path, monkeypatch):
        import trimesh
        # all vertices above the cutoff -> extraction fails -> full mesh
        verts = np.array([
            [0.0, 0.0, 400000.0], [1000.0, 0.0, 400000.0],
            [0.0, 1000.0, 400000.0], [500.0, 500.0, 500000.0],
        ])
        all_vnc = trimesh.Trimesh(
            vertices=verts, faces=[[0, 1, 2], [0, 1, 3]], process=False)
        monkeypatch.setattr(flybrains, 'JRCFIB2022M',
                            SimpleNamespace(mesh=all_vnc))
        vis = make_mesh_vis(tmp_path, dataset='male-cns:v0.9',
                            brain_mesh='native', vnc_mesh=True)
        vis._get_vnc_template_info = lambda: {
            'mesh': navis.Volume(make_small_trimesh(80.0), name='vnc'),
            'mesh_name': 'JRCFIB2022M (VNC)'}
        assert vis.plot_mesh() == 0

    def test_manc_vnc_mesh_only(self, tmp_path):
        vis = make_mesh_vis(tmp_path, dataset='manc:v1.0',
                            brain_mesh='none', vnc_mesh=True)
        vis._get_vnc_template_info = lambda: {
            'mesh': navis.Volume(make_small_trimesh(80.0), name='vnc'),
            'mesh_name': 'MANC (VNC)'}
        assert vis.plot_mesh() == 0
        names = [getattr(t, 'name', '') for t in vis.fig_3d.data]
        assert any('MANC' in n for n in names)
        roles = [m.metadata.get('export_role') for m in vis.exportable_meshes]
        assert 'vnc' in roles

    def test_manc_vnc_already_template(self, tmp_path):
        vis = make_mesh_vis(tmp_path, dataset='manc:v1.0',
                            brain_mesh='native', vnc_mesh=True, mesh_roi=[])
        vis._get_template_info = lambda: fake_template_info(
            'MANC (VNC envelope)', 'MANC', 'MANC')
        assert vis.plot_mesh() == 0

    def test_vnc_unavailable_for_hemibrain(self, tmp_path):
        vis = make_mesh_vis(tmp_path, brain_mesh='none', vnc_mesh=True)
        vis._get_vnc_template_info = lambda: None
        assert vis.plot_mesh() == 0
        assert len(vis.fig_3d.data) == 0

    def test_brain_mesh_failure_retry_path(self, tmp_path):
        vis = make_mesh_vis(tmp_path, brain_mesh='native', mesh_roi=[])

        class BadTemplate:
            @property
            def mesh(self):
                raise RuntimeError('mesh missing')

        vis._get_template_info = lambda: {
            'source': 'JRCFIB2018Fraw', 'target': 'JRCFIB2018F',
            'template_obj': BadTemplate(),
            'mesh_name': 'JRCFIB2018F (hemibrain)'}
        vis._dataset_needs_transform = lambda: False
        assert vis.plot_mesh() == 0  # failure caught, retry also fails
        assert len(vis.fig_3d.data) == 0


class TestExportPngWithTimeout:
    def _write_block_png(self, path):
        """White background with a dark noise block (>10KB overall)."""
        from PIL import Image
        img = Image.new('RGB', (300, 300), (255, 255, 255))
        rng = np.random.RandomState(7)
        block = rng.randint(0, 80, (90, 90, 3), dtype=np.uint8)
        arr = np.array(img)
        arr[100:190, 100:190] = block
        Image.fromarray(arr, 'RGB').save(str(path), 'PNG')

    def test_success_with_auto_crop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            go.Figure, 'write_image',
            lambda fig, file=None, **kw: self._write_block_png(file))
        vis = make_vis()
        out = tmp_path / 'out.png'
        ok, msg, scale = vis._export_png_with_timeout(
            go.Figure(), str(out), auto_crop=True, crop_margin=5, timeout=10)
        assert ok is True
        assert out.exists()
        from PIL import Image
        cropped = Image.open(str(out))
        assert cropped.size[0] < 300 and cropped.size[1] < 300

    def test_write_image_raises(self, tmp_path, monkeypatch):
        patch_write_image(monkeypatch, mode='raise')
        vis = make_vis()
        ok, msg, scale = vis._export_png_with_timeout(
            go.Figure(), str(tmp_path / 'x.png'))
        assert ok is False and 'kaleido exploded' in msg

    def test_no_file_created(self, tmp_path, monkeypatch):
        patch_write_image(monkeypatch, mode='nofile')
        vis = make_vis()
        ok, msg, scale = vis._export_png_with_timeout(
            go.Figure(), str(tmp_path / 'x.png'))
        assert ok is False and 'no file created' in msg

    def test_blank_image(self, tmp_path, monkeypatch):
        patch_write_image(monkeypatch, mode='blank')
        vis = make_vis()
        ok, msg, scale = vis._export_png_with_timeout(
            go.Figure(), str(tmp_path / 'x.png'))
        assert ok is False and 'blank' in msg

    def test_timeout_exception(self, tmp_path, monkeypatch):
        def writer(self, file=None, **kwargs):
            raise vs_module.PNGExportTimeout('too slow')
        monkeypatch.setattr(go.Figure, 'write_image', writer)
        vis = make_vis(export_timeout=3)
        ok, msg, scale = vis._export_png_with_timeout(
            go.Figure(), str(tmp_path / 'x.png'), timeout=99)
        assert ok is False and 'timed out after 3s' in msg



SAVE_ATTRS = dict(
    backend='plotly',
    brain_mesh='none',
    synapse_colors=['rgba(255, 0, 0, 1.0)'],
    saveas='fig',
    interactive_html=False,
    show_fig=False,
    export_views=False,
    export_simplified_png=False,
    export_method='kaleido',
    export_scale=2,
    html_size_cap=None,
    export_timeout=300,
    skeleton_mode='tube',
    skeleton_mesh_simplification=0.0,
    neuprint_skeleton_pipeline='fast',
    synapse_mode='scatter',
)


def make_save_vis(tmp_path, fig=None, **over):
    attrs = dict(SAVE_ATTRS)
    attrs['save_folder'] = str(tmp_path)
    attrs['fig_3d'] = fig if fig is not None else go.Figure()
    attrs.update(over)
    return make_vis(**attrs)


def make_big_figure():
    """Figure with a >5000-face mesh, a long scatter3d, and a 2d trace."""
    import trimesh
    sph = trimesh.creation.icosphere(subdivisions=4)  # 5120 faces
    n = 2000
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=sph.vertices[:, 0].tolist(), y=sph.vertices[:, 1].tolist(),
        z=sph.vertices[:, 2].tolist(), i=sph.faces[:, 0].tolist(),
        j=sph.faces[:, 1].tolist(), k=sph.faces[:, 2].tolist(),
        color='rgba(200, 200, 200, 0.3)'))
    fig.add_trace(go.Scatter3d(
        x=np.arange(n, dtype=float), y=np.arange(n, dtype=float),
        z=np.zeros(n), mode='lines',
        line=dict(color='rgba(255, 0, 0, 1.0)', width=4)))
    fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]))
    return fig


class TestSaveFigure:
    def test_html_only_default_camera(self, tmp_path):
        vis = make_save_vis(tmp_path)
        vis.save_figure()
        assert (tmp_path / 'fig.html').exists()
        assert vis.fig_path == str(tmp_path / 'fig')
        cam = vis.fig_3d.layout.scene.camera
        assert cam.eye.z == -2.5  # default frontal view

    def test_html_manc_camera(self, tmp_path):
        vis = make_save_vis(tmp_path, dataset='manc:v1.0')
        vis.save_figure()
        assert vis.fig_3d.layout.scene.camera.eye.z == 2.5

    def test_html_hemibrain_template_camera(self, tmp_path):
        vis = make_save_vis(tmp_path, brain_mesh='native')
        vis.save_figure()
        # Normalized onto the shared camera table (2.5), matching the
        # interactive view menu; the layout used to use a one-off 2.0.
        assert vis.fig_3d.layout.scene.camera.eye.y == 2.5

    def test_html_banc_camera_non_interactive(self, tmp_path):
        """Plan item C, non-interactive path: without the view menu the
        layout camera is the only one applied, so it must still be the
        BANC frontal view (anterior at -Y) — not the default Z-frontal,
        which shows BANC's ventral side."""
        vis = make_save_vis(tmp_path, dataset='banc_v888')
        vis.save_figure()
        cam = vis.fig_3d.layout.scene.camera
        assert cam.eye.y == -2.5
        assert cam.up.z == 1

    def test_interactive_html_view_menu(self, tmp_path):
        vis = make_save_vis(tmp_path, interactive_html=True)
        vis.save_figure()
        assert len(vis.fig_3d.layout.updatemenus) == 1
        assert len(vis.fig_3d.layout.annotations) == 2
        # dark background variant flips menu colors
        vis2 = make_save_vis(tmp_path, interactive_html=True, saveas='fig2',
                             background_color='rgba(0, 0, 0, 1.0)')
        vis2.save_figure()
        assert len(vis2.fig_3d.layout.updatemenus) == 1

    def test_interactive_html_manc_and_hemibrain_menus(self, tmp_path):
        vis = make_save_vis(tmp_path, interactive_html=True,
                            dataset='manc:v1.0', saveas='m')
        vis.save_figure()
        vis2 = make_save_vis(tmp_path, interactive_html=True,
                             brain_mesh='native', saveas='h')
        vis2.save_figure()

    def test_show_fig_opens_browser(self, tmp_path, monkeypatch):
        opened = []
        import webbrowser
        monkeypatch.setattr(webbrowser, 'open',
                            lambda url: opened.append(url))
        vis = make_save_vis(tmp_path, show_fig=True)
        vis.save_figure()
        assert len(opened) == 1 and opened[0].startswith('file://')

    def test_export_views_kaleido_success(self, tmp_path, monkeypatch):
        patch_write_image(monkeypatch, mode='ok')
        vis = make_save_vis(tmp_path, export_views=['front', 'back'])
        vis.save_figure()
        views = tmp_path / 'exported_views'
        assert (views / 'fig_front.png').exists()
        assert (views / 'fig_back.png').exists()
        assert (tmp_path / 'fig.png').exists()  # front copied to root

    def test_export_views_kaleido_all_views_true(self, tmp_path, monkeypatch):
        patch_write_image(monkeypatch, mode='ok')
        vis = make_save_vis(tmp_path, export_views=True)
        vis.save_figure()
        assert len(list((tmp_path / 'exported_views').glob('*.png'))) == 6

    def test_export_views_kaleido_str_variant(self, tmp_path, monkeypatch):
        patch_write_image(monkeypatch, mode='ok')
        vis = make_save_vis(tmp_path, export_views='top',
                            dataset='manc:v1.0')  # manc camera variant
        vis.save_figure()
        assert (tmp_path / 'exported_views' / 'fig_top.png').exists()

    def test_export_views_kaleido_fail_skips_rest(self, tmp_path, monkeypatch):
        patch_write_image(monkeypatch, mode='raise')
        vis = make_save_vis(tmp_path, export_views=['front', 'back'])
        vis.save_figure()  # outer except must not trigger
        assert not (tmp_path / 'exported_views' / 'fig_back.png').exists()
        assert not (tmp_path / 'fig.png').exists()

    def test_export_views_kaleido_blank(self, tmp_path, monkeypatch):
        patch_write_image(monkeypatch, mode='blank')
        vis = make_save_vis(tmp_path, export_views=['front'])
        vis.save_figure()
        assert not (tmp_path / 'fig.png').exists()

    def test_export_views_invalid_name(self, tmp_path, monkeypatch):
        patch_write_image(monkeypatch, mode='ok')
        vis = make_save_vis(tmp_path, export_views=['sideways', 'front'])
        vis.save_figure()
        assert (tmp_path / 'exported_views' / 'fig_front.png').exists()

    def test_export_views_webdriver_success(self, tmp_path, monkeypatch):
        make_fake_session_class(monkeypatch)
        vis = make_save_vis(tmp_path, export_method='webdriver',
                            export_views=['front', 'back'])
        vis.save_figure()
        assert (tmp_path / 'exported_views' / 'fig_front.png').exists()
        assert (tmp_path / 'fig.png').exists()

    def test_export_views_webdriver_fail_falls_back_to_kaleido(
            self, tmp_path, monkeypatch):
        make_fake_session_class(monkeypatch, fail_attempts=99,
                                error_msg='boom')  # non-chrome-crash: re-raise
        patch_write_image(monkeypatch, mode='ok')
        vis = make_save_vis(tmp_path, export_method='webdriver-fast',
                            export_views=['front'])
        vis.save_figure()
        assert (tmp_path / 'exported_views' / 'fig_front.png').exists()

    def test_auto_simplify_for_kaleido(self, tmp_path, monkeypatch):
        patch_write_image(monkeypatch, mode='ok')
        vis = make_save_vis(tmp_path, fig=make_big_figure(),
                            export_views=['front'])
        vis._get_html_size_cap = lambda: 0  # force auto-simplify branch
        vis.save_figure()
        assert getattr(vis, '_simplified_html_path', None)
        assert os.path.exists(vis._simplified_html_path)
        assert getattr(vis, '_simplified_export_fig', None) is not None
        assert (tmp_path / 'exported_views' / 'fig_front.png').exists()

    def test_export_simplified_png_threshold(self, tmp_path, monkeypatch):
        patch_write_image(monkeypatch, mode='ok')
        vis = make_save_vis(tmp_path, fig=make_big_figure(),
                            export_views=['front'],
                            export_simplified_png=1)  # 1 MB threshold
        vis.save_figure()
        assert (tmp_path / 'exported_views' / 'fig_front.png').exists()

    def test_export_simplified_png_true_variant(self, tmp_path, monkeypatch):
        # True -> 50MB threshold; tiny HTML stays below it (info branch only)
        patch_write_image(monkeypatch, mode='ok')
        vis = make_save_vis(tmp_path, export_views=['front'],
                            export_simplified_png=True)
        vis.save_figure()
        assert (tmp_path / 'exported_views' / 'fig_front.png').exists()

    def test_k3d_backend_branch(self, tmp_path):
        vis = make_save_vis(tmp_path, backend='k3d',
                            fig=SimpleNamespace())
        vis.save_figure()  # embed fails gracefully
        assert vis.fig_path == str(tmp_path / 'fig')


class TestWebDriverViewExportSession:
    """Direct coverage of _export_views_with_webdriver_session."""

    def _vis(self, tmp_path, **over):
        vis = make_save_vis(tmp_path, **over)
        vis.webdriver_render_wait = 0.01
        return vis

    CAMERAS = {'front': dict(eye=dict(x=0, y=0, z=-2.5),
                             up=dict(x=0, y=-1, z=0),
                             center=dict(x=0, y=0, z=0)),
               'back': dict(eye=dict(x=0, y=0, z=2.5),
                            up=dict(x=0, y=-1, z=0),
                            center=dict(x=0, y=0, z=0))}

    def test_success_skips_invalid_view_and_cleans_temp(
            self, tmp_path, monkeypatch):
        state = make_fake_session_class(monkeypatch)
        vis = self._vis(tmp_path)
        out = tmp_path / 'views'
        out.mkdir()
        exported = vis._export_views_with_webdriver_session(
            go.Figure(), ['front', 'bogus'], self.CAMERAS, str(out))
        assert exported == ['front']
        assert (out / 'fig_front.png').exists()
        assert not (out / '_temp_export.html').exists()
        assert len(state['loaded']) == 1

    def test_blank_screenshot_rejected(self, tmp_path, monkeypatch):
        def make_small_png_session(monkeypatch_):
            class TinySession:
                def __init__(self, **kw):
                    self._render_wait = 0.01

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def load_html(self, path, **kw):
                    pass

                def set_camera(self, **kw):
                    pass

                def screenshot(self, path):
                    from PIL import Image
                    Image.new('RGB', (10, 10), (0, 0, 0)).save(path, 'PNG')
            monkeypatch_.setattr(vs_module, 'WebDriverExportSession',
                                 TinySession)
        make_small_png_session(monkeypatch)
        vis = self._vis(tmp_path)
        out = tmp_path / 'views'
        out.mkdir()
        exported = vis._export_views_with_webdriver_session(
            go.Figure(), ['front'], self.CAMERAS, str(out))
        assert exported == []

    def test_chrome_crash_retries_then_succeeds(self, tmp_path, monkeypatch):
        state = make_fake_session_class(monkeypatch, fail_attempts=1,
                                        error_msg='chrome not reachable')
        vis = self._vis(tmp_path)
        out = tmp_path / 'views'
        out.mkdir()
        exported = vis._export_views_with_webdriver_session(
            go.Figure(), ['front'], self.CAMERAS, str(out))
        assert exported == ['front']
        assert state['attempts'] == 2

    def test_non_crash_failure_returns_empty(self, tmp_path, monkeypatch):
        make_fake_session_class(monkeypatch, fail_attempts=99,
                                error_msg='generic failure')
        vis = self._vis(tmp_path)
        out = tmp_path / 'views'
        out.mkdir()
        exported = vis._export_views_with_webdriver_session(
            go.Figure(), ['front'], self.CAMERAS, str(out))
        assert exported == []

    def test_existing_simplified_figure_reused(self, tmp_path, monkeypatch):
        # chrome-crash all 3 primary attempts -> outer handler -> reuse
        state = make_fake_session_class(monkeypatch, fail_attempts=3,
                                        error_msg='session deleted')
        vis = self._vis(tmp_path)
        vis._get_html_size_cap = lambda: 0
        simplified_html = tmp_path / 'prev_simplified.html'
        go.Figure().write_html(str(simplified_html), include_plotlyjs=False)
        vis._simplified_export_fig = go.Figure()
        vis._simplified_html_path = str(simplified_html)
        out = tmp_path / 'views'
        out.mkdir()
        exported = vis._export_views_with_webdriver_session(
            go.Figure(), ['front'], self.CAMERAS, str(out))
        assert exported == ['front']
        assert state['attempts'] == 4
        assert state['loaded'][-1] == str(simplified_html)

    def test_new_simplified_figure_created(self, tmp_path, monkeypatch):
        state = make_fake_session_class(monkeypatch, fail_attempts=3,
                                        error_msg='no such window')
        vis = self._vis(tmp_path, fig=make_big_figure())
        vis._get_html_size_cap = lambda: 0
        out = tmp_path / 'views'
        out.mkdir()
        exported = vis._export_views_with_webdriver_session(
            vis.fig_3d, ['front'], self.CAMERAS, str(out))
        assert exported == ['front']
        assert os.path.exists(str(tmp_path / 'views' / 'fig_simplified.html'))
        assert getattr(vis, '_simplified_export_fig', None) is not None


class TestSimplifiedFigures:
    def test_create_simplified_export_figure(self):
        vis = make_vis(fig_3d=make_big_figure())
        out = vis._create_simplified_export_figure()
        assert len(out.data) == 3
        mesh_trace = out.data[0]
        assert len(mesh_trace.i) <= 5120  # decimated (or kept on open3d miss)
        scatter = out.data[1]
        assert len(scatter.x) < 2000  # subsampled to ~500 points
        assert isinstance(out.data[2], go.Scatter)  # other types kept as-is

    def test_create_simplified_export_figure_empty_mesh(self):
        fig = go.Figure()
        fig.add_trace(go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[]))
        fig.add_trace(go.Scatter3d(x=[1.0], y=[1.0], z=[1.0]))
        vis = make_vis(fig_3d=fig)
        out = vis._create_simplified_export_figure()
        assert len(out.data) == 2

    def test_simplify_figure_for_kaleido(self):
        vis = make_vis()
        big = make_big_figure()
        out = vis._simplify_figure_for_kaleido(big, 0.25)
        assert len(out.data) == 3
        assert len(out.data[0].i) <= 5120
        assert len(out.data[1].x) < 2000

    def test_simplify_figure_for_kaleido_small_traces_kept(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=[1.0, 2.0], y=[1.0, 2.0], z=[0.0, 0.0]))
        fig.add_trace(go.Mesh3d(x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0],
                                i=[0], j=[1], k=[2]))
        vis = make_vis()
        out = vis._simplify_figure_for_kaleido(fig, 0.5)
        assert len(out.data) == 2



# ---------------------------------------------------------------------------
# Batch 4: _validate_inputs sweep + _prepare_neuprint_skeletons_for_render
# ---------------------------------------------------------------------------
VALIDATE_ATTRS = dict(
    backend='plotly',
    dataset='hemibrain:v1.2.1',
    client_type='neuprint',
    output_format='csv',
    search_columns='auto',
    hemisphere='both',
    server='https://neuprint.janelia.org',
    token=None,
    version=None,
    data_folder='.',
    output_dir=None,
    saveas='out',
    layer_map_csv=None,
    neuron_layers=['L0'],
    custom_layer_names=[],
    cache_neurons=False,
    cache_synapses=False,
    skip_synapse=False,
    include_timestamp=False,
    auto_fix_extrusions=False,
    smooth_skeleton=False,
    min_synapse_num=0,
    export_scale=2,
    export_timeout=120,
    neuron_alpha=0.8,
    synapse_alpha=0.6,
    skeleton_mesh_simplification=0.0,
    soma_mesh_simplification=None,
    soma_radius_cap=None,
    webdriver_render_wait=None,
    synapse_size=3,
    skeleton_mode='tube',
    neuprint_skeleton_pipeline='fast',
    synapse_mode='scatter',
    brain_mesh='none',
    legend_mode='layer',
    color_mode='per_layer',
    expand_colors='none',
    export_method='kaleido',
    export_views=False,
    neuron_colors='red',
    synapse_colors=['rgba(255, 0, 0, 1.0)'],
    mesh_roi=None,
    html_size_cap=None,
    export_simplified_png=False,
    force_API_fetching=False,
    vnc_mesh=False,
    mirror_on_contralateral=False,
    show_soma=True,
    show_fig=False,
    show_connectors=False,
    FAFB_template_correction=False,
    uniform_synapse_size=False,
    pre_post_scatter=False,
    soma_region_radius=None,
    roi_mesh_simplification=0.0,
    transforms_dir='transforms',
    brain_mesh_color='auto',
    vnc_mesh_color='auto',
    mesh_color=None,
    mesh_alpha=0.3,
)


class TestValidateInputs:
    def _vis(self, **over):
        attrs = dict(VALIDATE_ATTRS)
        attrs.update(over)
        return make_vis(**attrs)

    def test_valid_baseline_passes(self):
        vis = self._vis()
        assert vis._validate_inputs() is None
        # mesh_color None coerces to default gray tuple
        assert vis.mesh_color == (100, 100, 100)

    @pytest.mark.parametrize('attr,bad,fragment', [
        ('backend', 123, 'backend'),
        ('backend', 'mayavi', 'backend'),
        ('dataset', '', 'dataset'),
        ('dataset', 5, 'dataset'),
        ('client_type', 'cave', 'client_type'),
        ('output_format', 'parquet', 'output_format'),
        ('search_columns', 'wrong', 'search_columns'),
        ('hemisphere', 'middle', 'hemisphere'),
        ('server', 5, 'server'),
        ('token', 5, 'token'),
        ('version', 'x', 'version'),
        ('version', -1, 'version'),
        ('data_folder', 5, 'data_folder'),
        ('output_dir', 5, 'output_dir'),
        ('saveas', 5, 'saveas'),
        ('layer_map_csv', '/nonexistent/layer_map.csv', 'layer_map_csv'),
        ('neuron_layers', 5, 'neuron_layers'),
        ('neuron_layers', [], 'neuron_layers'),
        ('custom_layer_names', 'x', 'custom_layer_names'),
        ('cache_neurons', 'yes', 'cache_neurons'),
        ('cache_synapses', 1, 'cache_synapses'),
        ('skip_synapse', 'no', 'skip_synapse'),
        ('include_timestamp', 'no', 'include_timestamp'),
        ('auto_fix_extrusions', 0, 'auto_fix_extrusions'),
        ('smooth_skeleton', 'maybe', 'smooth_skeleton'),
        ('min_synapse_num', -1, 'min_synapse_num'),
        ('min_synapse_num', 'x', 'min_synapse_num'),
        ('export_scale', 0, 'export_scale'),
        ('export_scale', 'x', 'export_scale'),
        ('export_timeout', 0, 'export_timeout'),
        ('export_timeout', 'x', 'export_timeout'),
        ('neuron_alpha', 2.0, 'neuron_alpha'),
        ('neuron_alpha', 'x', 'neuron_alpha'),
        ('synapse_alpha', -0.1, 'synapse_alpha'),
        ('skeleton_mesh_simplification', 1.5, 'skeleton_mesh_simplification'),
        ('skeleton_mesh_simplification', 'x', 'skeleton_mesh_simplification'),
        ('soma_mesh_simplification', 2.0, 'soma_mesh_simplification'),
        ('soma_radius_cap', -1, 'soma_radius_cap'),
        ('webdriver_render_wait', -1, 'webdriver_render_wait'),
        ('synapse_size', 'garbage', 'synapse_size'),
        ('synapse_size', -1, 'synapse_size'),
        ('skeleton_mode', 5, 'skeleton_mode'),
        ('neuprint_skeleton_pipeline', 'turbo', 'neuprint_skeleton_pipeline'),
        ('synapse_mode', 5, 'synapse_mode'),
        ('brain_mesh', 5, 'brain_mesh'),
        ('legend_mode', 5, 'legend_mode'),
        ('color_mode', 5, 'color_mode'),
        ('expand_colors', 5, 'expand_colors'),
        ('export_method', 'svg', 'export_method'),
        ('export_views', 'front', 'export_views'),
        ('export_views', ['sideways'], 'export_views'),
        ('export_views', [1], 'export_views'),
        ('verbose', 'lots', 'verbose'),
        ('mesh_roi', 5, 'mesh_roi'),
        ('html_size_cap', 'big', 'html_size_cap'),
        ('html_size_cap', 0, 'html_size_cap'),
        ('export_simplified_png', 'yes', 'export_simplified_png'),
        ('export_simplified_png', -5, 'export_simplified_png'),
        ('force_API_fetching', 'x', 'force_API_fetching'),
        ('vnc_mesh', 'x', 'vnc_mesh'),
        ('mirror_on_contralateral', 'x', 'mirror_on_contralateral'),
        ('show_soma', 'x', 'show_soma'),
        ('show_fig', 'x', 'show_fig'),
        ('show_connectors', 'x', 'show_connectors'),
        ('FAFB_template_correction', 'x', 'FAFB_template_correction'),
        ('uniform_synapse_size', 'x', 'uniform_synapse_size'),
        ('pre_post_scatter', 'x', 'pre_post_scatter'),
        ('soma_region_radius', -1, 'soma_region_radius'),
        ('roi_mesh_simplification', 3.0, 'roi_mesh_simplification'),
        ('transforms_dir', 5, 'transforms_dir'),
    ])
    def test_invalid_value_raises(self, attr, bad, fragment):
        vis = self._vis(**{attr: bad})
        with pytest.raises(ValueError) as excinfo:
            vis._validate_inputs()
        assert fragment in str(excinfo.value)

    def test_empty_layers_allowed_with_mesh(self):
        vis = self._vis(neuron_layers=[], brain_mesh='native')
        assert vis._validate_inputs() is None
        vis2 = self._vis(neuron_layers='', mesh_roi=['AL(R)'])
        assert vis2._validate_inputs() is None
        vis3 = self._vis(neuron_layers=[], vnc_mesh=True)
        assert vis3._validate_inputs() is None

    def test_aggregated_errors(self):
        vis = self._vis(backend='bad', dataset='', min_synapse_num=-1)
        with pytest.raises(ValueError) as excinfo:
            vis._validate_inputs()
        msg = str(excinfo.value)
        assert 'backend' in msg and 'dataset' in msg and 'min_synapse_num' in msg

    def test_color_fallbacks_warn_not_fail(self):
        vis = self._vis(neuron_colors=123, synapse_colors=[],
                        brain_mesh_color=(1, 2), vnc_mesh_color=5,
                        mesh_color='red')
        assert vis._validate_inputs() is None
        assert vis.neuron_colors is None
        assert isinstance(vis.synapse_colors, list) and vis.synapse_colors
        assert vis.brain_mesh_color == 'auto'
        assert vis.vnc_mesh_color == 'auto'
        assert vis.mesh_color == 'red'

    def test_mesh_color_alpha_sanitized(self):
        vis = self._vis(mesh_color=(10, 20, 30, 999))
        assert vis._validate_inputs() is None
        assert vis.mesh_color == (10, 20, 30)
        # valid alpha retained
        vis2 = self._vis(mesh_color=(10, 20, 30, 0.5))
        vis2._validate_inputs()
        assert vis2.mesh_color == (10, 20, 30, 0.5)

    def test_synapse_size_strings_parsed(self):
        vis = self._vis(synapse_size='real')
        vis._validate_inputs()
        assert vis.synapse_size == 'real'
        vis2 = self._vis(synapse_size='2.5x real')
        vis2._validate_inputs()
        assert vis2.synapse_size == 2.5

    def test_parse_synapse_size_variants(self):
        parse = VisualizeSkeleton._parse_synapse_size
        assert parse('real') == 'real'
        assert parse('2') == 2.0
        assert parse('2.5x') == 2.5
        assert parse('2 x real') == 2.0
        assert parse('nonsense') is None


class TestPrepareNeuprintSkeletons:
    def _make(self, tmp_path, monkeypatch, neurons=None, **over):
        attrs = dict(
            script_path=str(tmp_path),
            cache_neurons=False,
            show_connectors=False,
            client=None,
            skeleton_mode='line',
            skeleton_mesh_simplification=0.0,
            neuprint_skeleton_pipeline='fast',
            skeleton_radius_style='auto',
        )
        attrs.update(over)
        vis = make_vis(**attrs)
        fetched = neurons if neurons is not None else [
            make_chain_neuron(body_id='101')]

        def fake_fetch(fetch_df, fetch_kwargs, persist=False):
            fake_fetch.calls.append((len(fetch_df), dict(fetch_kwargs)))
            return list(fetched)
        fake_fetch.calls = []
        vis._fetch_neuprint_skeletons_batched = fake_fetch
        return vis, fake_fetch

    def test_empty_body_ids(self, tmp_path, monkeypatch):
        vis, fetch = self._make(tmp_path, monkeypatch)
        prepared, is_mesh = vis._prepare_neuprint_skeletons_for_render(
            [], False, False, None)
        assert prepared == {} and is_mesh is False
        assert fetch.calls == []

    def test_line_mode_simplifies_tree(self, tmp_path, monkeypatch):
        vis, fetch = self._make(tmp_path, monkeypatch,
                                neurons=[make_chain_neuron(n_nodes=16,
                                                           body_id='101')])
        prepared, is_mesh = vis._prepare_neuprint_skeletons_for_render(
            [101], False, False, None)
        assert 101 in prepared
        assert isinstance(prepared[101], navis.TreeNeuron)
        assert is_mesh is False
        assert len(fetch.calls) == 1

    def test_tube_mode_mesh_and_decimate(self, tmp_path, monkeypatch):
        vis, fetch = self._make(
            tmp_path, monkeypatch,
            neurons=[make_chain_neuron(n_nodes=40, body_id='7')],
            skeleton_mode='tube', skeleton_mesh_simplification=0.5)
        prepared, is_mesh = vis._prepare_neuprint_skeletons_for_render(
            [7], False, False, None)
        assert 7 in prepared
        assert isinstance(prepared[7], navis.MeshNeuron)
        assert is_mesh is True

    def test_invalid_radii_fixed_before_meshing(self, tmp_path, monkeypatch):
        neuron = make_chain_neuron(n_nodes=24)
        neuron.nodes.loc[neuron.nodes.index[:5], 'radius'] = 0.0
        neuron.nodes.loc[neuron.nodes.index[5:8], 'radius'] = np.nan
        vis, _ = self._make(tmp_path, monkeypatch, neurons=[neuron],
                            skeleton_mode='tube',
                            skeleton_mesh_simplification=0.5)
        prepared, _ = vis._prepare_neuprint_skeletons_for_render(
            [42], False, False, None)
        assert 42 in prepared

    def test_missing_body_id_marked(self, tmp_path, monkeypatch):
        vis, _ = self._make(tmp_path, monkeypatch)
        prepared, _ = vis._prepare_neuprint_skeletons_for_render(
            [101, 999], False, False, None)
        assert set(prepared.keys()) == {101}

    def test_retryable_fetch_error_retries_then_succeeds(
            self, tmp_path, monkeypatch):
        vis = make_vis(script_path=str(tmp_path), cache_neurons=False,
                       show_connectors=False, client=None,
                       skeleton_mode='line',
                       skeleton_mesh_simplification=0.0,
                       neuprint_skeleton_pipeline='fast',
                       skeleton_radius_style='auto')
        attempts = {'n': 0}

        def flaky(fetch_df, fetch_kwargs, persist=False):
            attempts['n'] += 1
            if attempts['n'] == 1:
                raise RuntimeError('connection reset by peer')
            return [make_chain_neuron(body_id='101')]
        vis._fetch_neuprint_skeletons_batched = flaky
        prepared, _ = vis._prepare_neuprint_skeletons_for_render(
            [101], False, False, None)
        assert 101 in prepared and attempts['n'] == 2

    def test_non_retryable_fetch_error_breaks(self, tmp_path, monkeypatch):
        vis = make_vis(script_path=str(tmp_path), cache_neurons=False,
                       show_connectors=False, client=None,
                       skeleton_mode='line',
                       skeleton_mesh_simplification=0.0,
                       neuprint_skeleton_pipeline='fast',
                       skeleton_radius_style='auto')
        attempts = {'n': 0}

        def hard_fail(fetch_df, fetch_kwargs, persist=False):
            attempts['n'] += 1
            raise RuntimeError('Query returned error 400')
        vis._fetch_neuprint_skeletons_batched = hard_fail
        # A fully failed fetch must abort loudly instead of silently
        # rendering an empty figure.
        with pytest.raises(RuntimeError, match='fetch failed for all 1'):
            vis._prepare_neuprint_skeletons_for_render(
                [101], False, False, None)
        assert attempts['n'] == 1

    def test_raw_cache_check_failure_falls_through(self, tmp_path, monkeypatch):
        import morphology
        def boom(*a, **kw):
            raise RuntimeError('cache scanner offline')
        monkeypatch.setattr(morphology, 'find_similar_raw_cache', boom)
        vis, _ = self._make(tmp_path, monkeypatch, cache_neurons=True)
        prepared, _ = vis._prepare_neuprint_skeletons_for_render(
            [101], False, False, None)
        assert 101 in prepared

    def test_manc_voxel_scaling(self, tmp_path, monkeypatch):
        vis, _ = self._make(tmp_path, monkeypatch,
                            neurons=[make_chain_neuron(n_nodes=8)],
                            dataset='manc:v1.0')
        prepared, _ = vis._prepare_neuprint_skeletons_for_render(
            [42], False, False, None)
        assert prepared[42].nodes['x'].max() == pytest.approx(700.0 * 8)

    def test_fine_pipeline_constant_radius(self, tmp_path, monkeypatch):
        vis, _ = self._make(tmp_path, monkeypatch,
                            neurons=[make_chain_neuron(n_nodes=20,
                                                       body_id='101')],
                            skeleton_mode='tube',
                            neuprint_skeleton_pipeline='fine',
                            skeleton_radius_style='constant',
                            show_connectors=True)
        prepared, is_mesh = vis._prepare_neuprint_skeletons_for_render(
            [101], True, False, None)
        assert isinstance(prepared[101], navis.TreeNeuron)
        assert is_mesh is False

    def test_fine_pipeline_mesh_output(self, tmp_path, monkeypatch):
        vis, _ = self._make(tmp_path, monkeypatch,
                            neurons=[make_chain_neuron(n_nodes=30,
                                                       body_id='101')],
                            skeleton_mode='tube',
                            neuprint_skeleton_pipeline='fine',
                            skeleton_radius_style='constant',
                            skeleton_mesh_simplification=0.5)
        prepared, is_mesh = vis._prepare_neuprint_skeletons_for_render(
            [101], True, False, None)
        assert isinstance(prepared[101], navis.MeshNeuron)
        assert is_mesh is True


class TestNeuprintPipelineHelpers:
    def test_fafb_style_skeleton_radius_styles(self):
        vis = make_vis(skeleton_radius_style='auto')
        out = vis._fafb_style_neuprint_skeleton(
            make_chain_neuron(n_nodes=16), radius_style='constant')
        assert isinstance(out, navis.TreeNeuron)
        # constant style assigns the default NeuPrint radius everywhere
        assert out.nodes['radius'].iloc[0] == pytest.approx(
            vis.NEUPRINT_DEFAULT_RADIUS)

    def test_fafb_style_skeleton_fafb_radius(self):
        vis = make_vis(skeleton_radius_style='auto')
        out = vis._fafb_style_neuprint_skeleton(
            make_chain_neuron(n_nodes=16), radius_style='fafb')
        assert isinstance(out, navis.TreeNeuron)

    def test_fafb_style_skeleton_optimized(self):
        vis = make_vis(skeleton_radius_style='constant')
        out = vis._fafb_style_neuprint_skeleton(
            make_chain_neuron(n_nodes=16), optimized=True)
        assert isinstance(out, navis.TreeNeuron)

    def test_resolved_radius_style(self):
        assert make_vis(skeleton_radius_style='auto') \
            ._resolved_skeleton_radius_style() == 'fafb'
        assert make_vis(skeleton_radius_style='auto',
                        dataset='flywire')._resolved_skeleton_radius_style() \
            == 'source'
        assert make_vis(skeleton_radius_style='default') \
            ._resolved_skeleton_radius_style() == 'constant'

    def test_mesh_cache_compat_seams(self):
        vis = make_vis(skeleton_radius_style='auto',
                       neuprint_skeleton_pipeline='fast')
        key = vis._get_neuprint_mesh_cache_key()
        assert key.startswith('NEUPRINT_simp')
        assert vis._load_cached_neuprint_meshes([1, 2]) == ({}, [1, 2])
        assert vis._save_cached_neuprint_meshes({'a': 1}) == 0
        vis2 = make_vis(skeleton_radius_style='constant',
                        neuprint_skeleton_pipeline='artistic')
        key2 = vis2._get_neuprint_mesh_cache_key()
        assert 'vertexcluster' in key2 and '_radiusconstant' in key2



# ---------------------------------------------------------------------------
# Batch 5: FAFB source resolution, skeleton preload, extrusion check/fix
# ---------------------------------------------------------------------------
import fafb_utils  # noqa: E402

FAFB_ID = 720575940614131061
FAFB_ID_STR = str(FAFB_ID)


def make_fafb_swc_string():
    # Simple 8-node chain in canonical SWC text form
    # (columns: id type x y z radius parent).
    lines = []
    for i in range(1, 9):
        parent = -1 if i == 1 else i - 1
        lines.append(
            f"{i} 3 {i * 500.0:.1f} 100.0 100.0 5.0 {parent}")
    return "\n".join(lines)


class FakeFafbBundle:
    def __init__(self, contents):
        self.contents = contents
        self.closed = False

    def get(self, body_id):
        return self.contents.get(body_id)

    def close(self):
        self.closed = True


class TestPreloadFafbSkeletons:
    def _vis(self, **over):
        attrs = dict(dataset='flywire_FAFB_v783', neuron_dfs=[],
                     verbose=False)
        attrs.update(over)
        return make_vis(**attrs)

    def test_bundle_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            fafb_utils, 'get_fafb_skeleton_bundle',
            lambda data_dir: FakeFafbBundle({FAFB_ID: make_fafb_swc_string()}))
        monkeypatch.setattr(vs_module, 'resolve_flywire_dataset_dir',
                            lambda pr, ds: str(tmp_path))
        vis = self._vis()
        loaded = vis._preload_fafb_skeletons(body_ids_filter=[FAFB_ID])
        assert FAFB_ID_STR in loaded
        assert isinstance(loaded[FAFB_ID_STR], navis.TreeNeuron)

    def test_zip_fallback_path(self, tmp_path, monkeypatch):
        import zipfile
        zip_path = tmp_path / 'sk_lod1_783_healed.zip'
        with zipfile.ZipFile(str(zip_path), 'w') as zf:
            zf.writestr(f'{FAFB_ID_STR}.swc', make_fafb_swc_string())
        monkeypatch.setattr(fafb_utils, 'get_fafb_skeleton_bundle',
                            lambda data_dir: None)
        monkeypatch.setattr(fafb_utils, 'get_fafb_skeleton_zip',
                            lambda data_dir: str(zip_path))
        monkeypatch.setattr(vs_module, 'resolve_flywire_dataset_dir',
                            lambda pr, ds: str(tmp_path))
        vis = self._vis()
        loaded = vis._preload_fafb_skeletons(body_ids_filter=[FAFB_ID])
        assert FAFB_ID_STR in loaded

    def test_empty_request_returns_empty(self):
        vis = self._vis()
        assert vis._preload_fafb_skeletons() == {}

    def test_non_flywire_dataset_ids(self, tmp_path, monkeypatch):
        import zipfile
        zip_path = tmp_path / 'sk.zip'
        swc = make_fafb_swc_string()
        with zipfile.ZipFile(str(zip_path), 'w') as zf:
            zf.writestr('101.swc', swc)
        monkeypatch.setattr(fafb_utils, 'get_fafb_skeleton_bundle',
                            lambda data_dir: None)
        monkeypatch.setattr(fafb_utils, 'get_fafb_skeleton_zip',
                            lambda data_dir: str(zip_path))
        monkeypatch.setattr(vs_module, 'resolve_flywire_dataset_dir',
                            lambda pr, ds: str(tmp_path))
        vis = self._vis(dataset='hemibrain:v1.2.1')
        loaded = vis._preload_fafb_skeletons(body_ids_filter=[101])
        assert 101 in loaded

    def test_no_dataset_dir_returns_empty(self, monkeypatch):
        monkeypatch.setattr(vs_module, 'resolve_flywire_dataset_dir',
                            lambda pr, ds: None)
        vis = self._vis()
        assert vis._preload_fafb_skeletons(body_ids_filter=[FAFB_ID]) == {}


class TestResolveFafbSources:
    @pytest.fixture(autouse=True)
    def _sandbox_script_path(self, tmp_path):
        # The resolver consults the real cache/ tree when script_path
        # points at the repository root (e.g. _take_repaired serving a
        # previously repaired body).  Sandbox it so these tests stay
        # hermetic regardless of local cache state.
        self._script_path = str(tmp_path)

    def _vis(self, **over):
        attrs = dict(dataset='flywire_FAFB_v783', cache_neurons=False,
                     auto_fix_extrusions=False, neuron_dfs=[], verbose=False,
                     script_path=self._script_path)
        attrs.update(over)
        vis = make_vis(**attrs)
        vis.api_calls = []
        vis._preload_fafb_skeletons = lambda body_ids_filter=None: {}
        vis._load_api_cached_skeletons = lambda ids: ({}, list(ids))
        vis._detect_extrusions_in_skeletons = \
            lambda skeletons, use_cache=False: []

        def fake_cave(ids):
            vis.api_calls.append(list(ids))
            return {}
        vis._fetch_fafb_skeletons_via_cave = fake_cave
        return vis

    def _tree(self):
        return make_chain_neuron(body_id=FAFB_ID_STR)

    def test_api_only(self):
        vis = self._vis()
        cave_tree = self._tree()
        vis._fetch_fafb_skeletons_via_cave = lambda ids: {FAFB_ID: cave_tree}
        sources, skel = vis._resolve_fafb_sources([FAFB_ID], api_only=True)
        assert sources[FAFB_ID_STR] == 'cave'
        assert skel[FAFB_ID_STR] is cave_tree

    def test_zip_priority(self):
        vis = self._vis()
        tree = make_chain_neuron(body_id=FAFB_ID_STR)
        vis._preload_fafb_skeletons = \
            lambda body_ids_filter=None: {FAFB_ID: tree}
        sources, skel = vis._resolve_fafb_sources([FAFB_ID])
        assert sources[FAFB_ID_STR] == 'zip'
        assert skel[FAFB_ID_STR] is tree
        assert vis.api_calls == []

    def test_raw_cache_source(self):
        vis = self._vis(cache_neurons=True)
        tree = make_chain_neuron(body_id=FAFB_ID_STR)
        vis._load_api_cached_skeletons = lambda ids: ({FAFB_ID: tree}, [])
        sources, _ = vis._resolve_fafb_sources([FAFB_ID])
        assert sources[FAFB_ID_STR] == 'raw_cache'

    def test_cave_fallback_for_remaining(self):
        vis = self._vis()
        cave_tree = self._tree()
        vis._fetch_fafb_skeletons_via_cave = lambda ids: {FAFB_ID: cave_tree}
        sources, skel = vis._resolve_fafb_sources([FAFB_ID])
        assert sources[FAFB_ID_STR] == 'cave'
        assert skel[FAFB_ID_STR] is cave_tree

    def test_extrusion_repair_via_cave(self):
        vis = self._vis(auto_fix_extrusions=True)
        tree = make_chain_neuron(body_id=FAFB_ID_STR)
        vis._preload_fafb_skeletons = \
            lambda body_ids_filter=None: {FAFB_ID: tree}
        vis._detect_extrusions_in_skeletons = \
            lambda skeletons, use_cache=False: [FAFB_ID]
        cave_tree = self._tree()
        vis._fetch_fafb_skeletons_via_cave = lambda ids: {FAFB_ID: cave_tree}
        sources, skel = vis._resolve_fafb_sources([FAFB_ID])
        assert sources[FAFB_ID_STR] == 'cave'
        assert skel[FAFB_ID_STR] is cave_tree

    def test_extrusion_repair_served_from_cave_store(self, monkeypatch):
        """An api_repaired body with a cached CAVE tree skips the network:
        it is served by the leading repair-cache stage and never loads the
        zip tree nor runs the online repair."""
        vis = self._vis(auto_fix_extrusions=True, cache_neurons=True)
        tree = make_chain_neuron(body_id=FAFB_ID_STR)
        vis._preload_fafb_skeletons = \
            lambda body_ids_filter=None: {FAFB_ID: tree}
        cave_tree = self._tree()
        monkeypatch.setattr(
            'cave_data_fetcher.load_repaired_skeletons',
            lambda dataset, ids, project_root=None, log=None:
                {FAFB_ID: cave_tree})
        detect_calls = []
        vis._detect_extrusions_in_skeletons = \
            lambda skeletons, use_cache=False: detect_calls.append(
                list(skeletons)) or []
        sources, skel = vis._resolve_fafb_sources([FAFB_ID])
        assert sources[FAFB_ID_STR] == 'cave'
        assert skel[FAFB_ID_STR] is cave_tree
        assert vis.api_calls == []
        # the repaired body is final: no re-check, no repair stage work
        assert detect_calls == []

    def test_extrusion_repair_strict_no_cache_skips_cave_store(self, monkeypatch):
        """cache_neurons=False never reads a previously repaired CAVE tree."""
        vis = self._vis(auto_fix_extrusions=True, cache_neurons=False)
        tree = make_chain_neuron(body_id=FAFB_ID_STR)
        vis._preload_fafb_skeletons = \
            lambda body_ids_filter=None: {FAFB_ID: tree}
        vis._detect_extrusions_in_skeletons = \
            lambda skeletons, use_cache=False: [FAFB_ID]
        monkeypatch.setattr(
            fafb_utils, 'load_extrusion_repair_status',
            lambda root, dataset: pytest.fail(
                'strict no-cache mode read repair-status cache'))
        vis._load_cave_cached_skeleton = lambda bid: pytest.fail(
            'strict no-cache mode read CAVE skeleton cache')
        cave_tree = self._tree()
        vis._fetch_fafb_skeletons_via_cave = lambda ids: {FAFB_ID: cave_tree}

        sources, skel = vis._resolve_fafb_sources([FAFB_ID])

        assert sources[FAFB_ID_STR] == 'cave'
        assert skel[FAFB_ID_STR] is cave_tree

    def test_cave_fetch_uses_local_store_before_token_gate(self, monkeypatch):
        """A cached CAVE tree remains usable when the token is absent."""
        vis = self._vis(auto_fix_extrusions=False, cache_neurons=True)
        vis._fetch_fafb_skeletons_via_cave = (
            VisualizeSkeleton._fetch_fafb_skeletons_via_cave.__get__(
                vis, VisualizeSkeleton)
        )
        vis._flywire_skeleton_access = {'cave_token': False}
        cave_tree = self._tree()
        vis._load_cave_cached_skeleton = lambda bid: cave_tree

        trees = vis._fetch_fafb_skeletons_via_cave([FAFB_ID])

        assert trees[FAFB_ID_STR] is cave_tree

    def test_extrusion_repair_local_fallback(self, monkeypatch):
        vis = self._vis(auto_fix_extrusions=True)
        tree = make_chain_neuron(body_id=FAFB_ID_STR)
        vis._preload_fafb_skeletons = \
            lambda body_ids_filter=None: {FAFB_ID: tree}
        vis._detect_extrusions_in_skeletons = \
            lambda skeletons, use_cache=False: [FAFB_ID]
        repaired = make_chain_neuron(body_id=FAFB_ID_STR)
        monkeypatch.setattr(
            fafb_utils, 'repair_extruded_skeleton',
            lambda n: (repaired, {'repaired': True, 'removed_nodes': 2}))
        sources, skel = vis._resolve_fafb_sources([FAFB_ID])
        assert sources[FAFB_ID_STR] == 'local_repaired'
        assert skel[FAFB_ID_STR] is repaired

    def test_extrusion_repair_not_possible(self, monkeypatch):
        vis = self._vis(auto_fix_extrusions=True)
        tree = make_chain_neuron(body_id=FAFB_ID_STR)
        vis._preload_fafb_skeletons = \
            lambda body_ids_filter=None: {FAFB_ID: tree}
        vis._detect_extrusions_in_skeletons = \
            lambda skeletons, use_cache=False: [FAFB_ID]
        monkeypatch.setattr(
            fafb_utils, 'repair_extruded_skeleton',
            lambda n: (n, {'repaired': False, 'removed_nodes': 0}))
        sources, _ = vis._resolve_fafb_sources([FAFB_ID])
        assert sources[FAFB_ID_STR] == 'zip'  # unchanged, warned

    def test_repair_statuses_persisted_when_caching(self, monkeypatch):
        recorded = {}
        monkeypatch.setattr(
            fafb_utils, 'set_extrusion_repair_status',
            lambda root, dataset, statuses: recorded.update(statuses))
        vis = self._vis(auto_fix_extrusions=True, cache_neurons=True)
        tree = make_chain_neuron(body_id=FAFB_ID_STR)
        vis._preload_fafb_skeletons = \
            lambda body_ids_filter=None: {FAFB_ID: tree}
        vis._load_api_cached_skeletons = lambda ids: ({}, list(ids))
        vis._detect_extrusions_in_skeletons = \
            lambda skeletons, use_cache=False: [FAFB_ID]
        cave_tree = self._tree()
        vis._fetch_fafb_skeletons_via_cave = lambda ids: {FAFB_ID: cave_tree}
        vis._resolve_fafb_sources([FAFB_ID])
        assert recorded.get(FAFB_ID_STR) == 'api_repaired'


class TestFafbExtrusionChecks:
    def _zip_with_swc(self, tmp_path, name=f'{FAFB_ID_STR}.healed.swc'):
        import zipfile
        d = tmp_path / 'datasets' / 'flywire_FAFB_v783'
        d.mkdir(parents=True, exist_ok=True)
        zip_path = d / 'sk_lod1_783_healed.zip'
        with zipfile.ZipFile(str(zip_path), 'w') as zf:
            zf.writestr(name, make_fafb_swc_string())
        return zip_path

    def test_missing_zip_returns_unknown(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(fafb_utils, 'get_fafb_skeleton_bundle',
                            lambda p: (_ for _ in ()).throw(
                                RuntimeError('no bundle')))
        result = VisualizeSkeleton.check_fafb_skeleton_for_extrusions(
            FAFB_ID, verbose=False)
        assert result['has_extrusions'] is False
        assert 'ZIP file not found' in result['recommendation']

    def test_zip_load_and_detect(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._zip_with_swc(tmp_path)
        monkeypatch.setattr(fafb_utils, 'get_fafb_skeleton_bundle',
                            lambda p: None)
        result = VisualizeSkeleton.check_fafb_skeleton_for_extrusions(
            FAFB_ID, simplification=0.5, verbose=False)
        assert 'has_extrusions' in result
        assert result['skeleton'] is not None
        assert result['auto_fixed'] is False

    def test_bundle_load_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            fafb_utils, 'get_fafb_skeleton_bundle',
            lambda p: FakeFafbBundle({FAFB_ID: make_fafb_swc_string()}))
        result = VisualizeSkeleton.check_fafb_skeleton_for_extrusions(
            FAFB_ID, simplification=0.0, verbose=False)
        assert result['skeleton'] is not None

    def test_body_not_in_zip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._zip_with_swc(tmp_path, name='999.healed.swc')
        monkeypatch.setattr(fafb_utils, 'get_fafb_skeleton_bundle',
                            lambda p: None)
        result = VisualizeSkeleton.check_fafb_skeleton_for_extrusions(
            FAFB_ID, verbose=False)
        assert 'not found in ZIP' in result['recommendation']

    def test_auto_fix_applies(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._zip_with_swc(tmp_path)
        monkeypatch.setattr(fafb_utils, 'get_fafb_skeleton_bundle',
                            lambda p: None)
        mesh = navis.MeshNeuron(make_small_trimesh())
        mesh.id = FAFB_ID
        monkeypatch.setattr(
            VisualizeSkeleton, 'detect_mesh_extrusions',
            staticmethod(lambda mesh_neuron, soma_pos=None, soma_radius=None,
                         verbose=False: {
                             'has_extrusions': True, 'severity': 'high',
                             'extrusion_count': 1,
                             'extrusion_vertices': np.array([]),
                             'max_edge_length': 1, 'median_edge_length': 1,
                             'edge_length_ratio': 1,
                             'soma_region_issues': False,
                             'recommendation': 'fix'}))
        monkeypatch.setattr(
            VisualizeSkeleton, 'fix_fafb_extrusions',
            staticmethod(lambda ids, dataset='flywire_FAFB_v783',
                         verbose=False: {FAFB_ID: mesh}))
        result = VisualizeSkeleton.check_fafb_skeleton_for_extrusions(
            FAFB_ID, simplification=0.5, verbose=False, auto_fix=True)
        assert result['auto_fixed'] is True
        assert result['skeleton'] is mesh

    def test_fix_fafb_extrusions_via_fake_cave_fetcher(
            self, tmp_path, monkeypatch):
        import types as _types
        fake_mod = _types.ModuleType('cave_data_fetcher')
        captured = {}

        class FakeFetcher:
            def __init__(self, dataset, project_root, verbose):
                captured['dataset'] = dataset

            def fetch_skeleton(self, body_id, use_cache=True):
                captured['ids'] = captured.get('ids', []) + [body_id]
                tree = make_chain_neuron(body_id=FAFB_ID_STR)
                return tree

        fake_mod.CAVEDataFetcher = FakeFetcher
        monkeypatch.setitem(sys.modules, 'cave_data_fetcher', fake_mod)
        monkeypatch.setattr(vs_module, 'require_flywire_skeleton_access',
                            lambda *a, **kw: None)
        result = VisualizeSkeleton.fix_fafb_extrusions(
            [FAFB_ID], dataset='flywire_FAFB_v783', verbose=True)
        assert FAFB_ID_STR in result
        from flywire_ids import body_id_to_api_int
        assert captured['ids'] == [body_id_to_api_int(FAFB_ID)]

# ---------------------------------------------------------------------------
# Batch 6: plot_individuals, PDF/PPTX summaries, export_individuals_webdriver,
#          export_video (+html/webdriver variants), video2gif, crop helpers
# ---------------------------------------------------------------------------

def make_block_jpeg(path, size=(200, 150), block=((30, 40), (110, 130))):
    """White JPEG with a dark block (auto-crop-friendly video frame)."""
    make_white_image_with_block(size=size, block=block).save(
        str(path), 'JPEG', quality=90)


def make_real_mp4(path, n_frames=5, size=(32, 32), fps=10):
    """Tiny real MP4 (mp4v codec) so cv2.VideoCapture can open it."""
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'),
                        fps, size)
    assert w.isOpened()
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), (i * 40) % 255, np.uint8)
        w.write(frame)
    w.release()
    return str(path)


def make_plotly_html(path, traces=None):
    """Small plotly HTML that ``figure_from_plotly_html`` can parse."""
    if traces is None:
        traces = [go.Scatter3d(x=[0, 1, 2], y=[0, 1, 0], z=[0, 0, 1],
                               mode='lines', name='t1')]
    fig = go.Figure(data=traces)
    fig.write_html(str(path), include_plotlyjs=False, auto_open=False)
    return str(path)


def make_individuals_vis(tmp_path, fig=None, **over):
    """VisualizeSkeleton ready for plot_individuals()."""
    if fig is None:
        xyz = dict(x=[0, 1, 2], y=[0, 1, 0], z=[0, 0, 1])
        fig = go.Figure(data=[
            go.Scatter3d(mode='lines', name='neuron A',
                         legendgroup='neuron A', showlegend=True,
                         line=dict(color='rgba(255, 0, 0, 1.0)'), **xyz),
            go.Scatter3d(mode='lines', name='brain mesh', showlegend=True,
                         line=dict(color='rgba(0, 0, 255, 0.3)'), **xyz),
            go.Scatter3d(mode='markers', name='Synapses (pre)',
                         showlegend=True, marker=dict(size=3), **xyz),
            go.Scatter3d(mode='markers', name='pre sites',
                         legendgroup='pre_post:pre:neuron A',
                         showlegend=True,
                         marker=dict(color='rgb(0, 255, 0)', size=2), **xyz),
            go.Scatter3d(mode='lines', name='merged part',
                         legendgroup='grp2', showlegend=False,
                         line=dict(color='rgba(0, 128, 128, 1.0)'), **xyz),
        ])
    attrs = dict(
        backend='plotly',
        export_method='kaleido',
        export_scale=2,
        save_folder=str(tmp_path),
        dataset='hemibrain:v1.2.1',
        brain_mesh='none',
        mesh_roi=[],
        saveas='ind_fig',
        fig_3d=fig,
        verbose=False,
    )
    attrs.update(over)
    return make_vis(**attrs)


class TestPlotIndividuals:
    def test_no_figure_returns_none(self, tmp_path):
        vis = make_individuals_vis(tmp_path, fig_3d=None)
        assert vis.plot_individuals() is None

    def test_non_plotly_backend_returns_none(self, tmp_path):
        vis = make_individuals_vis(tmp_path, backend='k3d')
        assert vis.plot_individuals() is None

    def test_invalid_output_format_returns_none(self, tmp_path):
        vis = make_individuals_vis(tmp_path)
        assert vis.plot_individuals(output_format='svg') is None

    def test_invalid_view_returns_none(self, tmp_path):
        vis = make_individuals_vis(tmp_path)
        assert vis.plot_individuals(views='diagonal') is None

    def test_no_legend_entries_returns_none(self, tmp_path):
        fig = go.Figure(data=[go.Scatter3d(
            x=[0, 1], y=[0, 1], z=[0, 1], name='only mesh',
            showlegend=True)])
        vis = make_individuals_vis(tmp_path, fig=fig)
        assert vis.plot_individuals() is None

    def test_kaleido_png_html_single_view_pdf(self, tmp_path, monkeypatch):
        patch_write_image(monkeypatch, mode='ok')
        vis = make_individuals_vis(tmp_path)
        out = vis.plot_individuals(output_format=['png', 'html'],
                                   views='front', summary_format='pdf')
        assert out == os.path.join(str(tmp_path), 'individual_profiles')
        pngs = [f for f in os.listdir(out) if f.endswith('.png')]
        htmls = [f for f in os.listdir(out) if f.endswith('.html')
                 and not f.startswith('_')]
        assert pngs and htmls
        # neuron A owns the pre_post site trace; grp2 merged part too
        assert any('neuron_A' in f for f in pngs)
        pdfs = [f for f in os.listdir(str(tmp_path)) if f.endswith('.pdf')]
        assert pdfs

    def test_multi_view_pdf_and_pptx(self, tmp_path, monkeypatch):
        patch_write_image(monkeypatch, mode='ok')
        vis = make_individuals_vis(tmp_path)
        out = vis.plot_individuals(output_format='png',
                                   views=['front', 'top'],
                                   summary_format=['pdf', 'pptx'])
        assert out is not None
        parent = str(tmp_path)
        files = os.listdir(parent)
        assert 'individual_profiles_summary_by_view.pdf' in files
        assert 'individual_profiles_summary_by_name.pdf' in files
        assert 'individual_profiles_summary_by_view.pptx' in files
        assert 'individual_profiles_summary_by_name.pptx' in files

    def test_summary_true_defaults_to_pdf(self, tmp_path, monkeypatch):
        patch_write_image(monkeypatch, mode='ok')
        vis = make_individuals_vis(tmp_path)
        vis.plot_individuals(output_format='png', views='front',
                             summary_format=True, pdf_title='custom title')
        assert any(f.endswith('.pdf') for f in os.listdir(str(tmp_path)))

    def test_png_failure_marks_failed(self, tmp_path, monkeypatch):
        patch_write_image(monkeypatch, mode='raise')
        vis = make_individuals_vis(tmp_path)
        out = vis.plot_individuals(output_format='png', views='front',
                                   summary_format=False)
        assert out is not None
        assert not [f for f in os.listdir(out) if f.endswith('.png')]

    def test_webdriver_export_success_with_html(
            self, tmp_path, monkeypatch):
        captured = {}

        def fake_wd_export(**kwargs):
            captured.update(kwargs)
            out_dir = kwargs['output_dir']
            files = {}
            for name in kwargs['legend_entries']:
                safe = name.replace(' ', '_')
                entry = []
                for view in kwargs['views']:
                    p = os.path.join(out_dir, f'{view}_{safe}.png')
                    make_noise_png(p)
                    entry.append((p, view))
                files[safe] = entry
            return {'success': True, 'files': files, 'failed': [],
                    'error': None}

        monkeypatch.setattr(vs_module, 'export_individuals_webdriver',
                            fake_wd_export)
        vis = make_individuals_vis(tmp_path, export_method='webdriver')
        out = vis.plot_individuals(output_format=['png', 'html'],
                                   views='front', summary_format=False,
                                   scale=3)
        assert out is not None
        assert captured['scale'] == 3
        # temp HTML written for the session was cleaned up afterwards
        assert not os.path.exists(
            os.path.join(out, '_temp_main_figure.html'))
        assert any(f.endswith('.html') for f in os.listdir(out))

    def test_webdriver_scale_capped_at_five(self, tmp_path, monkeypatch):
        captured = {}

        def fake_wd_export(**kwargs):
            captured.update(kwargs)
            return {'success': True, 'files': {}, 'failed': [],
                    'error': None}

        monkeypatch.setattr(vs_module, 'export_individuals_webdriver',
                            fake_wd_export)
        vis = make_individuals_vis(tmp_path, export_method='webdriver',
                                   export_scale=10)
        vis.plot_individuals(output_format='png', views='front',
                             summary_format=False)
        assert captured['scale'] == 5

    def test_webdriver_failure_keeps_html_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            vs_module, 'export_individuals_webdriver',
            lambda **kw: {'success': False, 'files': {}, 'failed': [],
                          'error': 'chrome exploded'})
        vis = make_individuals_vis(tmp_path, export_method='webdriver')
        out = vis.plot_individuals(output_format='png', views='front',
                                   summary_format=False)
        assert out is not None
        assert not [f for f in os.listdir(out) if f.endswith('.png')]

    @pytest.mark.parametrize('dataset,brain_mesh,axis,value', [
        ('hemibrain:v1.2.1', 'none', 'z', -2.5),
        ('manc:v1.0', 'none', 'z', 2.5),
        ('hemibrain:v1.2.1', 'native', 'y', 2.5),
    ])
    def test_view_cameras_per_dataset(self, tmp_path, monkeypatch,
                                      dataset, brain_mesh, axis, value):
        captured = {}

        def fake_write_image(self, file=None, **kwargs):
            cam = self.layout.scene.camera
            captured['eye'] = dict(
                x=cam.eye.x, y=cam.eye.y, z=cam.eye.z)
            make_noise_png(file)

        monkeypatch.setattr(go.Figure, 'write_image', fake_write_image)
        vis = make_individuals_vis(tmp_path, dataset=dataset,
                                   brain_mesh=brain_mesh)
        vis.plot_individuals(output_format='png', views='front',
                             summary_format=False)
        assert captured['eye'][axis] == pytest.approx(value)

    def test_figure_state_restored(self, tmp_path, monkeypatch):
        patch_write_image(monkeypatch, mode='ok')
        vis = make_individuals_vis(tmp_path)
        before = [getattr(t, 'visible', None) for t in vis.fig_3d.data]
        vis.plot_individuals(output_format='png', views='front',
                             summary_format=False)
        after = [getattr(t, 'visible', None) for t in vis.fig_3d.data]
        assert before == after


class TestStructuralRolesEndToEnd:
    """Plan §5a FIX-1 (F3): the roles a profile export may never mis-file.

    These run through ``plot_individuals()`` rather than the classifier, because
    the bug was silent: a neuron whose *name* contained an ROI or dataset
    acronym was demoted to background and simply never produced a PNG.
    """

    def _figure(self, tmp_path):
        """ROI mesh + acronym-collision neurons + a legend-only swatch."""
        stamper = make_individuals_vis(tmp_path, fig_3d=None)
        fig = go.Figure()
        fig.add_trace(go.Mesh3d(x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0],
                                i=[0], j=[1], k=[2],
                                name='brain region [LH]',
                                legendgroup='roi_mesh:LH'))
        for name, group, type_label, item, body_id in (
                ('LHPD1L', 'LH', 'LHPD1L', '512_aMe1', '512'),
                ('LHPD1L', 'LH', 'LHPD1L', '513_aMe2', '513'),
                ('query_transformed_SMP227_fafb',
                 'query_transformed_SMP227_fafb', 'SMP227',
                 '99_aMe1', '99')):
            trace = go.Scatter3d(x=[0, 100], y=[0, 10], z=[0, 10],
                                 mode='lines', name=name, legendgroup=name,
                                 showlegend=True)
            stamper._stamp_trace_identity(
                trace, kind='neuron', group=group, type_label=type_label,
                item=item, body_id=body_id)
            fig.add_trace(trace)
        # The legend's colour swatch: same group, no geometry of its own.
        fig.add_trace(go.Scatter3d(x=[None, None], y=[None, None],
                                   z=[None, None], mode='lines',
                                   name='LHPD1L', legendgroup='LHPD1L',
                                   showlegend=True))
        return fig

    def _vis(self, tmp_path, **over):
        return make_individuals_vis(tmp_path, fig=self._figure(tmp_path),
                                    mesh_roi=['LH', 'LHPR'], **over)

    def _pngs(self, out):
        return sorted(f for f in os.listdir(out) if f.endswith('.png'))

    def test_profile_roi_name_substring_collision(self, tmp_path, monkeypatch):
        """ROI 'LH' stays background while type 'LHPD1L' keeps its profile."""
        patch_write_image(monkeypatch, mode='ok')
        vis = self._vis(tmp_path)
        out = vis.plot_individuals(output_format='png', views='front',
                                   summary_format=False)
        assert list(vis._profile_entries) == [
            'LHPD1L', 'query_transformed_SMP227_fafb']
        assert vis._profile_entries['LHPD1L'] == [1, 2]
        assert vis._trace_roles[0].role == 'mesh'
        assert self._pngs(out) == ['front_LHPD1L.png',
                                   'front_query_transformed_SMP227_fafb.png']

    def test_profile_cross_dataset_overlay_not_background(
            self, tmp_path, monkeypatch):
        """A '..._fafb' overlay layer is a neuron, not a dataset template.

        The name tier survives only for structurally bare traces, and only on a
        whole-word match, so neither the ``_fafb`` suffix of a cross-dataset
        overlay nor the ``LH`` inside ``LHPD1L`` can fire.
        """
        patch_write_image(monkeypatch, mode='ok')
        vis = self._vis(tmp_path)
        roles = vs_module.classify_traces(
            [go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode='lines',
                          name='query_transformed_SMP227_fafb')],
            mesh_roi_names=['LH', 'LHPR'])
        assert [role.role for role in roles] == ['neuron']
        vis.plot_individuals(output_format='png', views='front',
                             summary_format=False)
        overlay = vis._trace_roles[3]
        assert (overlay.role, overlay.item) == ('neuron', '99_aMe1')
        assert 'query_transformed_SMP227_fafb' in vis._profile_entries

    def test_profile_legend_swatch_excluded(self, tmp_path, monkeypatch):
        """The swatch carries the group's colour, never a profile of its own."""
        patch_write_image(monkeypatch, mode='ok')
        vis = self._vis(tmp_path)
        vis.plot_individuals(output_format='png', views='front',
                             summary_format=False)
        assert vis._trace_roles[4].role == 'legend_swatch'
        assert 5 not in vis._profile_entries['LHPD1L']
        assert vis._profile_entries['LHPD1L'] == [1, 2]

    @pytest.mark.parametrize('granularity, expected', [
        ('legend', {'LHPD1L', 'query_transformed_SMP227_fafb'}),
        ('layer', {'LH', 'query_transformed_SMP227_fafb'}),
        ('type', {'LHPD1L', 'SMP227'}),
        ('body', {'LH__LHPD1L__512_aMe1', 'LH__LHPD1L__513_aMe2',
                  'query_transformed_SMP227_fafb__SMP227__99_aMe1'}),
    ])
    def test_granularity_levels_change_the_profiles_end_to_end(
            self, tmp_path, monkeypatch, granularity, expected):
        patch_write_image(monkeypatch, mode='ok')
        vis = self._vis(tmp_path)
        out = vis.plot_individuals(output_format='png', views='front',
                                   summary_format=False,
                                   granularity=granularity)
        assert set(vis._profile_entries) == expected
        assert len(self._pngs(out)) == len(expected)

    def test_a_trace_with_no_identity_is_named_in_the_log(
            self, tmp_path, monkeypatch, capsys):
        """Silent loss was the bug; the log has to carry the count and index."""
        patch_write_image(monkeypatch, mode='ok')
        fig = self._figure(tmp_path)
        orphan = go.Scatter3d(x=[5, 6], y=[0, 1], z=[0, 1], mode='lines')
        fig.add_trace(orphan)
        vis = make_individuals_vis(tmp_path, fig=fig, mesh_roi=['LH'],
                                   verbose=True)
        vis.plot_individuals(output_format='png', views='front',
                             summary_format=False)
        printed = capsys.readouterr().out
        assert 'Trace roles:' in printed
        assert '1 trace(s) carry no legend identity and got no profile' in printed
        assert '[5]' in printed


class TestIndividualSummaries:
    def _vis(self, tmp_path):
        return make_individuals_vis(tmp_path)

    def _image_dict(self, tmp_path, names=('r1', 'r2')):
        images = {}
        for i, name in enumerate(names):
            p = tmp_path / f'{name}_front.png'
            make_noise_png(str(p), seed=i)
            images[name] = [(str(p), 'front')]
        return images

    def test_pdf_organize_by_name(self, tmp_path):
        vis = self._vis(tmp_path)
        path = vis._create_individual_pdf(
            output_dir=str(tmp_path), images_dict=self._image_dict(tmp_path),
            images_per_page=(2, 2), organize_by='name',
            background_color='black')
        assert path and os.path.exists(path)

    def test_pdf_organize_by_view_plain_entries(self, tmp_path):
        vis = self._vis(tmp_path)
        images = self._image_dict(tmp_path)
        # add a plain (non-tuple) entry and a missing file entry
        extra = tmp_path / 'extra.png'
        make_noise_png(str(extra), seed=9)
        images['r1'].append(str(extra))
        images['r1'].append((str(tmp_path / 'missing.png'), 'front'))
        path = vis._create_individual_pdf(
            output_dir=str(tmp_path), images_dict=images,
            organize_by='view', views=['front'], pdf_suffix='_v')
        assert path and os.path.exists(path)

    def test_pdf_no_images_returns_none(self, tmp_path):
        vis = self._vis(tmp_path)
        assert vis._create_individual_pdf(
            output_dir=str(tmp_path), images_dict={}) is None

    def test_pptx_organize_by_view(self, tmp_path):
        vis = self._vis(tmp_path)
        path = vis._create_individual_pptx(
            output_dir=str(tmp_path), images_dict=self._image_dict(tmp_path),
            organize_by='view', views=['front'], pptx_suffix='_v',
            background_color='rgba(0, 0, 0, 1.0)')
        assert path and os.path.exists(path)

    def test_pptx_organize_by_name_truncated_label(self, tmp_path):
        vis = self._vis(tmp_path)
        long_name = 'x' * 400
        p = tmp_path / 'long.png'
        make_noise_png(str(p), seed=3)
        path = vis._create_individual_pptx(
            output_dir=str(tmp_path),
            images_dict={long_name: [(str(p), 'front')]},
            organize_by='name')
        assert path and os.path.exists(path)

    def test_pptx_no_images_returns_none(self, tmp_path):
        vis = self._vis(tmp_path)
        assert vis._create_individual_pptx(
            output_dir=str(tmp_path), images_dict={}) is None


class IndividualsFakeSession:
    fail_opens = 0
    error_msg = 'chrome not reachable'
    attempts = 0
    screenshot_bytes = 'noise'

    def __init__(self, width=900, height=900, scale=2, timeout=60):
        self.scale = scale

    def __enter__(self):
        IndividualsFakeSession.attempts += 1
        if IndividualsFakeSession.attempts <= IndividualsFakeSession.fail_opens:
            raise RuntimeError(IndividualsFakeSession.error_msg)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def load_html(self, path, wait_for_render=True, render_wait=2,
                  background_color=None):
        self.loaded = path

    def update_layout(self, updates):
        self.layout_updates = updates

    def hide_modebar(self):
        self.modebar_hidden = True

    def set_trace_visibility(self, visible_indices, total_traces):
        self.last_visibility = (list(visible_indices), total_traces)

    def set_camera(self, eye=None, up=None, center=None):
        self.camera = (eye, up, center)

    def screenshot(self, path, auto_crop=True, margin=30,
                   background_color=(255, 255, 255)):
        if IndividualsFakeSession.screenshot_bytes == 'tiny':
            with open(path, 'wb') as fh:
                fh.write(b'\x89PNG\r\n\x1a\n')
        else:
            make_noise_png(path)


class TestExportIndividualsWebdriver:
    def _reset(self, monkeypatch, **cls_attrs):
        IndividualsFakeSession.attempts = 0
        IndividualsFakeSession.fail_opens = 0
        IndividualsFakeSession.error_msg = 'chrome not reachable'
        IndividualsFakeSession.screenshot_bytes = 'noise'
        for key, value in cls_attrs.items():
            monkeypatch.setattr(IndividualsFakeSession, key, value)
        monkeypatch.setattr(vs_module, 'WebDriverExportSession',
                            IndividualsFakeSession)

    def _args(self, tmp_path):
        cameras = {
            'front': dict(eye=dict(x=0, y=0, z=-2.5),
                          center=dict(x=0, y=0, z=0),
                          up=dict(x=0, y=-1, z=0)),
            'top': dict(eye=dict(x=0, y=-2.5, z=0),
                        center=dict(x=0, y=0, z=0),
                        up=dict(x=0, y=0, z=1)),
        }
        return dict(
            html_path=str(tmp_path / 'main.html'),
            output_dir=str(tmp_path),
            legend_entries={'neuron A': [0], 'neuron B': [1]},
            background_indices=[2],
            total_traces=3,
            views=['front', 'top'],
            view_cameras=cameras,
        )

    def test_success_exports_all_views(self, tmp_path, monkeypatch):
        self._reset(monkeypatch)
        make_plotly_html(tmp_path / 'main.html')
        result = vs_module.export_individuals_webdriver(
            **self._args(tmp_path), verbose=True)
        assert result['success']
        assert not result['failed']
        assert len(result['files']['neuron_A']) == 2
        assert len(result['files']['neuron_B']) == 2

    def test_small_screenshot_marked_failed(self, tmp_path, monkeypatch):
        self._reset(monkeypatch, screenshot_bytes='tiny')
        make_plotly_html(tmp_path / 'main.html')
        result = vs_module.export_individuals_webdriver(
            **self._args(tmp_path), verbose=False)
        assert result['success']
        assert result['failed'] == ['neuron A', 'neuron B']

    def test_chrome_crash_retries_then_succeeds(self, tmp_path, monkeypatch):
        self._reset(monkeypatch, fail_opens=1)
        make_plotly_html(tmp_path / 'main.html')
        result = vs_module.export_individuals_webdriver(
            **self._args(tmp_path), verbose=True)
        assert result['success']
        assert IndividualsFakeSession.attempts == 2

    def test_non_crash_error_fails_immediately(self, tmp_path, monkeypatch):
        self._reset(monkeypatch, fail_opens=99,
                    error_msg='something exploded')
        make_plotly_html(tmp_path / 'main.html')
        result = vs_module.export_individuals_webdriver(
            **self._args(tmp_path), verbose=True)
        assert not result['success']
        assert 'something exploded' in result['error']
        assert IndividualsFakeSession.attempts == 1


def make_video_vis(tmp_path, **over):
    """VisualizeSkeleton ready for export_video() (kaleido default)."""
    fig = go.Figure(data=[go.Scatter3d(
        x=[0, 1, 2], y=[0, 1, 0], z=[0, 0, 1], mode='markers',
        marker=dict(size=3))])
    attrs = dict(
        export_method='kaleido',
        dataset='hemibrain:v1.2.1',
        brain_mesh='none',
        background_color='rgba(255, 255, 255, 1.0)',
        save_folder=str(tmp_path),
        saveas='vid',
        export_scale=2,
        export_timeout=60,
        fig_3d=fig,
        fig_path=str(tmp_path / 'vid'),
        verbose=False,
    )
    attrs.update(over)
    vis = make_vis(**attrs)
    vis._get_html_size_cap = lambda: 1000
    make_plotly_html(str(tmp_path / 'vid.html'))
    return vis


class VideoFakeSession:
    fail_opens = 0
    error_msg = 'chrome not reachable'
    attempts = 0
    camera_return = {'eye': {'x': 0, 'y': 0, 'z': -2.5},
                     'up': {'x': 0, 'y': -1, 'z': 0}}

    def __init__(self, width=1200, height=900, scale=2, timeout=300,
                 render_wait=None):
        self._render_wait = 0.05

    def __enter__(self):
        VideoFakeSession.attempts += 1
        if VideoFakeSession.attempts <= VideoFakeSession.fail_opens:
            raise RuntimeError(VideoFakeSession.error_msg)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def load_html(self, path, wait_for_render=True, render_wait=3,
                  background_color=None):
        self.loaded = path

    def get_current_camera(self):
        return VideoFakeSession.camera_return

    def set_camera(self, eye=None, up=None, center=None):
        self.camera = (eye, up)

    def screenshot(self, path, convert_to_jpeg=True, jpeg_quality=95,
                   auto_crop=False, background_color=(255, 255, 255)):
        make_noise_png(path)


class TestExportVideo:
    def _reset_session(self, monkeypatch, **cls_attrs):
        VideoFakeSession.attempts = 0
        VideoFakeSession.fail_opens = 0
        VideoFakeSession.error_msg = 'chrome not reachable'
        VideoFakeSession.camera_return = {
            'eye': {'x': 0, 'y': 0, 'z': -2.5},
            'up': {'x': 0, 'y': -1, 'z': 0}}
        for key, value in cls_attrs.items():
            monkeypatch.setattr(VideoFakeSession, key, value)
        monkeypatch.setattr(vs_module, 'WebDriverExportSession',
                            VideoFakeSession)

    def _patch_frames(self, monkeypatch, fail_indices=(), timeout_indices=()):
        state = {'calls': 0}

        def fake_write_image(self, file=None, **kwargs):
            idx = state['calls']
            state['calls'] += 1
            if idx in timeout_indices:
                raise vs_module.PNGExportTimeout('timed out')
            if idx in fail_indices:
                raise RuntimeError('kaleido exploded')
            make_noise_png(file)

        monkeypatch.setattr(go.Figure, 'write_image', fake_write_image)
        return state

    def test_kaleido_horizontal_success(self, tmp_path, monkeypatch):
        self._patch_frames(monkeypatch)
        vis = make_video_vis(tmp_path)
        rc = vis.export_video(degree_per_frame=120, export_gif=False,
                              auto_crop=False, crop_margin=50)
        assert rc == 0
        pic_folder = tmp_path / 'pics_30fps_xz'
        frames = sorted(f for f in os.listdir(str(pic_folder))
                        if f.startswith('deg_'))
        assert len(frames) == 3

    def test_kaleido_vertical_with_autocrop_and_scale_warn(
            self, tmp_path, monkeypatch):
        self._patch_frames(monkeypatch)
        vis = make_video_vis(tmp_path)
        rc = vis.export_video(degree_per_frame=120, rotate='vertical',
                              export_gif=False, auto_crop=True, scale=6)
        assert rc == 0
        assert (tmp_path / 'pics_30fps_yz').exists()

    def test_html_file_mode_with_output_dir(self, tmp_path, monkeypatch):
        self._patch_frames(monkeypatch)
        vis = make_video_vis(tmp_path)
        html = make_plotly_html(tmp_path / 'src_plot.html')
        out_dir = tmp_path / 'videos'
        rc = vis.export_video(html_file=html, output_dir=str(out_dir),
                              degree_per_frame=120, export_gif=False)
        assert rc == 0
        assert (out_dir / 'pics_30fps_xz').exists()

    def test_html_file_without_payload_raises(self, tmp_path):
        # The page parser is real now, so only a payload-free page fails,
        # and it still surfaces as a RuntimeError from export_video.
        vis = make_video_vis(tmp_path)
        html = tmp_path / 'empty_plot.html'
        html.write_text('<html><body>no figure here</body></html>')
        with pytest.raises(RuntimeError):
            vis.export_video(html_file=str(html),
                             output_dir=str(tmp_path / 'videos'))

    def test_error_paths(self, tmp_path):
        vis = make_video_vis(tmp_path)
        # no save_folder / output_dir / html_file
        del vis.save_folder
        with pytest.raises(ValueError):
            vis.export_video()
        # save_folder present but no figure html
        vis2 = make_video_vis(tmp_path)
        os.remove(str(tmp_path / 'vid.html'))
        with pytest.raises(RuntimeError):
            vis2.export_video()
        # html_file does not exist
        vis3 = make_video_vis(tmp_path)
        with pytest.raises(FileNotFoundError):
            vis3.export_video(html_file=str(tmp_path / 'missing.html'),
                              output_dir=str(tmp_path))

    def test_use_existing_images_skips_render(self, tmp_path, monkeypatch):
        state = self._patch_frames(monkeypatch, fail_indices=set(range(99)))
        vis = make_video_vis(tmp_path)
        pic_folder = tmp_path / 'pics_30fps_xz'
        os.makedirs(str(pic_folder))
        for deg in (0.0, 120.0, 240.0):
            make_noise_png(str(pic_folder / f'deg_{deg}.jpeg'))
        rc = vis.export_video(degree_per_frame=120, export_gif=False,
                              use_existing_images=True)
        assert rc == 0
        assert state['calls'] == 0

    def test_first_frame_failure_returns_one(self, tmp_path, monkeypatch):
        self._patch_frames(monkeypatch, fail_indices={0, 1})
        vis = make_video_vis(tmp_path)
        rc = vis.export_video(degree_per_frame=120, export_gif=False)
        assert rc == 1

    def test_later_frame_failure_returns_one(self, tmp_path, monkeypatch):
        self._patch_frames(monkeypatch, fail_indices={1})
        vis = make_video_vis(tmp_path)
        rc = vis.export_video(degree_per_frame=120, export_gif=False)
        assert rc == 1

    def test_timeout_retry_at_scale_one(self, tmp_path, monkeypatch):
        self._patch_frames(monkeypatch, timeout_indices={0})
        vis = make_video_vis(tmp_path)
        rc = vis.export_video(degree_per_frame=120, export_gif=False)
        assert rc == 0

    def test_webdriver_success_and_simplified_copy(
            self, tmp_path, monkeypatch):
        self._reset_session(monkeypatch)
        vis = make_video_vis(tmp_path, export_method='webdriver',
                             webdriver_render_wait=0)
        vis._simplified_export_fig = vis.fig_3d
        rc = vis.export_video(degree_per_frame=120, export_gif=False,
                              auto_crop=False)
        assert rc == 0
        # simplified HTML copy persisted next to the save folder
        assert (tmp_path / 'vid_simplified.html').exists()
        frames = [f for f in os.listdir(str(tmp_path / 'pics_30fps_xz_webdriver'))
                  if f.startswith('deg_')]
        assert len(frames) == 3

    def test_webdriver_no_camera_uses_defaults(self, tmp_path, monkeypatch):
        self._reset_session(monkeypatch, camera_return=None)
        vis = make_video_vis(tmp_path, export_method='webdriver',
                             dataset='manc:v1.0',
                             webdriver_render_wait=None)
        rc = vis.export_video(degree_per_frame=120, export_gif=False)
        assert rc == 0

    def test_webdriver_crashes_fall_back_to_kaleido(
            self, tmp_path, monkeypatch):
        self._reset_session(monkeypatch, fail_opens=999)
        self._patch_frames(monkeypatch)
        vis = make_video_vis(tmp_path, export_method='webdriver')
        rc = vis.export_video(degree_per_frame=120, export_gif=False)
        assert rc == 0
        frames = [f for f in os.listdir(str(tmp_path / 'pics_30fps_xz'))
                  if f.startswith('deg_')]
        assert len(frames) == 3

    def test_kaleido_auto_simplifies_large_html(self, tmp_path, monkeypatch):
        self._patch_frames(monkeypatch)
        vis = make_video_vis(tmp_path)
        vis._get_html_size_cap = lambda: 0
        rc = vis.export_video(degree_per_frame=120, export_gif=False)
        assert rc == 0
        assert getattr(vis, '_simplified_export_fig', None) is not None
        assert (tmp_path / 'vid_simplified.html').exists()

    def test_kaleido_reuses_existing_simplified_figure(
            self, tmp_path, monkeypatch):
        self._patch_frames(monkeypatch)
        vis = make_video_vis(tmp_path)
        vis._get_html_size_cap = lambda: 0
        vis._simplified_export_fig = vis.fig_3d
        rc = vis.export_video(degree_per_frame=120, export_gif=False)
        assert rc == 0

    def test_gif_conversion_failure_warned(self, tmp_path, monkeypatch):
        self._patch_frames(monkeypatch)
        vis = make_video_vis(tmp_path)
        # avc1 writer unavailable or fake mp4 -> video2gif raises, caught
        rc = vis.export_video(degree_per_frame=120, export_gif=True)
        assert rc == 0


class TestExportVideoFromHtml:
    def test_success_with_autocrop(self, tmp_path, monkeypatch):
        patch_write_image(monkeypatch, mode='ok')
        html = make_plotly_html(tmp_path / 'plot.html')
        rc = vs_module.export_video_from_html(
            html, degree_per_frame=120, export_gif=False, auto_crop=True,
            output_dir=str(tmp_path / 'out'))
        assert rc == 0
        assert (tmp_path / 'out' / 'pics_30fps_xz').exists()

    def test_payload_free_html_raises(self, tmp_path):
        html = tmp_path / 'plot.html'
        html.write_text('<html><body>no figure here</body></html>')
        with pytest.raises(RuntimeError):
            vs_module.export_video_from_html(str(html),
                                             degree_per_frame=120)

    def test_missing_html_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            vs_module.export_video_from_html(str(tmp_path / 'nope.html'))

    def test_first_frame_failure_returns_one(self, tmp_path, monkeypatch):
        patch_write_image(monkeypatch, mode='raise')
        html = make_plotly_html(tmp_path / 'plot.html')
        rc = vs_module.export_video_from_html(
            html, degree_per_frame=120, export_gif=False)
        assert rc == 1

    def test_reuses_existing_images(self, tmp_path, monkeypatch):
        calls = patch_write_image(monkeypatch, mode='raise')
        html = make_plotly_html(tmp_path / 'plot.html')
        pic_folder = tmp_path / 'pics_30fps_xz'
        os.makedirs(str(pic_folder))
        for deg in (0.0, 120.0, 240.0):
            make_noise_png(str(pic_folder / f'deg_{deg}.jpeg'))
        rc = vs_module.export_video_from_html(
            html, degree_per_frame=120, export_gif=False,
            use_existing_images=True)
        assert rc == 0
        assert calls == []


class FakeChromeDriver:
    def __init__(self, service=None, options=None):
        self.urls = []
        self.quit_called = False

    def get(self, url):
        self.urls.append(url)

    def execute_script(self, script, *args):
        return None

    def find_element(self, by, value):
        return object()

    def save_screenshot(self, path):
        make_noise_png(path)
        return True

    def quit(self):
        self.quit_called = True


def patch_selenium(monkeypatch, chrome_raises=False):
    import selenium.webdriver as swd
    import webdriver_manager.chrome as wmc

    class FakeInstaller:
        def install(self):
            return '/fake/chromedriver'

    def chrome_factory(*args, **kwargs):
        if chrome_raises:
            raise RuntimeError('no chrome here')
        return FakeChromeDriver()

    monkeypatch.setattr(wmc, 'ChromeDriverManager', FakeInstaller)
    monkeypatch.setattr(swd, 'Chrome', chrome_factory)


class TestExportVideoWebdriverEngine:
    """``export_method='webdriver'`` renders the frames through one session.

    This engine replaces the retired ``export_video_webdriver()``: the page
    loads once, only the camera moves per frame, kaleido is never called, and
    a machine without Chrome falls back to the kaleido loop instead of losing
    the export.
    """

    def _patch_session(self, monkeypatch, cls_attrs=None, session_class=None):
        VideoFakeSession.attempts = 0
        VideoFakeSession.fail_opens = 0
        VideoFakeSession.error_msg = 'chrome not reachable'
        VideoFakeSession.camera_return = {
            'eye': {'x': 0, 'y': 0, 'z': -2.5},
            'up': {'x': 0, 'y': -1, 'z': 0}}
        for key, value in (cls_attrs or {}).items():
            monkeypatch.setattr(VideoFakeSession, key, value)
        monkeypatch.setattr(vs_module, 'WebDriverExportSession',
                            session_class or VideoFakeSession)

    @staticmethod
    def _frames(pic_folder):
        return [f for f in os.listdir(str(pic_folder))
                if f.endswith('.jpeg')]

    @staticmethod
    def _write_manifest(run_dir, dataset):
        """Record which dataset the page came from, as Phase 4 does."""
        import json
        (run_dir / vs_module.VISUALIZATION_MANIFEST_NAME).write_text(
            json.dumps({'dataset': dataset}), encoding='utf-8')

    def test_success_uses_one_session_and_no_kaleido(self, tmp_path,
                                                    monkeypatch):
        self._patch_session(monkeypatch)
        kaleido_calls = patch_write_image(monkeypatch, mode='raise')
        html = make_plotly_html(tmp_path / 'plot.html')
        out = tmp_path / 'out'
        rc = vs_module.export_video_from_html(
            html, degree_per_frame=90, export_gif=False, auto_crop=True,
            output_dir=str(out), export_method='webdriver')
        assert rc == 0
        assert VideoFakeSession.attempts == 1, 'the browser must open once'
        assert kaleido_calls == [], 'kaleido must stay out of the webdriver path'
        pic_folder = out / 'pics_30fps_xz_webdriver'
        assert len(self._frames(pic_folder)) == 4
        # The engine joins the folder name, so a kaleido cache can never be
        # resumed by the webdriver engine (or vice versa).
        assert not (out / 'pics_30fps_xz').exists()
        # The page the session drove is not a frame; leaving it behind would
        # make the video assembly pick it up as imglist[0].
        assert '_temp_video.html' not in os.listdir(str(pic_folder))

    def test_rotation_matches_the_live_pipeline_orbit(self, tmp_path,
                                                     monkeypatch):
        """Eye/up per frame come from the page's own camera distance."""
        seen = []

        class RecordingSession(VideoFakeSession):
            def set_camera(self, eye=None, up=None, center=None):
                seen.append((dict(eye), dict(up)))

        self._patch_session(monkeypatch, session_class=RecordingSession)
        html = make_plotly_html(tmp_path / 'plot.html')
        rc = vs_module.export_video_from_html(
            html, degree_per_frame=180, export_gif=False,
            output_dir=str(tmp_path / 'out'), export_method='webdriver')
        assert rc == 0
        # eye z=-2.5 => orbit radius 2.5, with the 1% off-axis gimbal offset.
        assert len(seen) == 2
        eye_front, up_front = seen[0]
        assert eye_front == pytest.approx({'x': 0.0, 'y': 0.025, 'z': -2.5})
        assert up_front == {'x': 0, 'y': -1, 'z': 0}
        eye_back, _up_back = seen[1]
        assert eye_back == pytest.approx({'x': 0.0, 'y': 0.025, 'z': 2.5})

    def test_manifest_dataset_selects_the_manc_orbit_handedness(self, tmp_path,
                                                               monkeypatch):
        """``manc`` pages rotate the other way, exactly like export_video."""
        seen = []

        class RecordingSession(VideoFakeSession):
            def set_camera(self, eye=None, up=None, center=None):
                seen.append(dict(eye))

        self._patch_session(monkeypatch, session_class=RecordingSession)
        html = make_plotly_html(tmp_path / 'plot.html')
        self._write_manifest(tmp_path, dataset='manc:v1.0')
        rc = vs_module.export_video_from_html(
            html, degree_per_frame=360, export_gif=False,
            output_dir=str(tmp_path / 'out'), export_method='webdriver')
        assert rc == 0
        assert seen[0] == pytest.approx({'x': 0.0, 'y': 0.025, 'z': 2.5})

    def test_browser_failure_falls_back_to_kaleido(self, tmp_path, monkeypatch):
        self._patch_session(monkeypatch, {'fail_opens': 99})
        kaleido_calls = patch_write_image(monkeypatch, mode='ok')
        html = make_plotly_html(tmp_path / 'plot.html')
        rc = vs_module.export_video_from_html(
            html, degree_per_frame=120, export_gif=False,
            output_dir=str(tmp_path / 'out'), export_method='webdriver')
        assert rc == 0
        assert len(kaleido_calls) == 3

    def test_missing_html_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            vs_module.export_video_from_html(
                str(tmp_path / 'nope.html'), export_method='webdriver')


class TestExportPngWebdriver:
    def test_success_default_output(self, tmp_path, monkeypatch):
        patch_selenium(monkeypatch)
        html = make_plotly_html(tmp_path / 'plot.html')
        result = vs_module.export_png_webdriver(html)
        assert result == str(tmp_path / 'plot_webdriver.png')
        assert os.path.exists(result)

    def test_custom_output(self, tmp_path, monkeypatch):
        patch_selenium(monkeypatch)
        html = make_plotly_html(tmp_path / 'plot.html')
        out = str(tmp_path / 'custom.png')
        assert vs_module.export_png_webdriver(html, output_path=out) == out

    def test_driver_init_failure_returns_none(self, tmp_path, monkeypatch):
        patch_selenium(monkeypatch, chrome_raises=True)
        html = make_plotly_html(tmp_path / 'plot.html')
        assert vs_module.export_png_webdriver(html) is None

    def test_missing_html_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            vs_module.export_png_webdriver(str(tmp_path / 'nope.html'))


class TestVideo2Gif:
    def test_pil_conversion_with_scale_and_fps(self, tmp_path):
        mp4 = make_real_mp4(tmp_path / 'in.mp4')
        gif = str(tmp_path / 'out.gif')
        result = vs_module.video2gif(mp4, gif, fps=5, scale=0.5,
                                     optimize=False)
        assert result == gif and os.path.exists(gif)

    def test_default_output_and_original_fps(self, tmp_path):
        mp4 = make_real_mp4(tmp_path / 'in.mp4')
        result = vs_module.video2gif(mp4, optimize=False)
        assert result == str(tmp_path / 'in.gif')
        assert os.path.exists(result)

    def test_ffmpeg_path_used_when_available(self, tmp_path, monkeypatch):
        import shutil as _shutil
        monkeypatch.setattr(_shutil, 'which', lambda name: '/fake/ffmpeg')
        captured = {}

        def fake_ffmpeg(input_video, output_gif, fps=None, scale=1.0,
                        loop=0):
            captured['args'] = (input_video, output_gif, fps, scale, loop)
            from PIL import Image
            Image.new('RGB', (4, 4), (1, 2, 3)).save(output_gif, 'GIF')

        monkeypatch.setattr(vs_module, '_video2gif_ffmpeg', fake_ffmpeg)
        mp4 = make_real_mp4(tmp_path / 'in.mp4')
        gif = str(tmp_path / 'out.gif')
        result = vs_module.video2gif(mp4, gif, fps=5, scale=0.5)
        assert result == gif and os.path.exists(gif)
        assert captured['args'][3] == 0.5

    def test_ffmpeg_failure_falls_back_to_pil(self, tmp_path, monkeypatch):
        import shutil as _shutil
        monkeypatch.setattr(_shutil, 'which', lambda name: '/fake/ffmpeg')

        def boom(*args, **kwargs):
            raise RuntimeError('ffmpeg exploded')

        monkeypatch.setattr(vs_module, '_video2gif_ffmpeg', boom)
        mp4 = make_real_mp4(tmp_path / 'in.mp4')
        gif = str(tmp_path / 'out.gif')
        result = vs_module.video2gif(mp4, gif, optimize=True)
        assert result == gif and os.path.exists(gif)

    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            vs_module.video2gif(str(tmp_path / 'nope.mp4'))

    def test_unopenable_video_raises(self, tmp_path):
        fake = tmp_path / 'empty.mp4'
        fake.write_bytes(b'')
        with pytest.raises(ValueError):
            vs_module.video2gif(str(fake))


class TestVideo2GifFfmpeg:
    def test_two_pass_success(self, tmp_path, monkeypatch):
        import subprocess
        commands = []

        def fake_run(cmd, **kwargs):
            commands.append(cmd)
            return SimpleNamespace(returncode=0, stderr='')

        monkeypatch.setattr(subprocess, 'run', fake_run)
        vs_module._video2gif_ffmpeg(str(tmp_path / 'in.mp4'),
                                    str(tmp_path / 'out.gif'),
                                    fps=15, scale=0.5, loop=1)
        assert len(commands) == 2
        assert any('palettegen=stats_mode=diff' in str(p)
                   for p in commands[0])
        assert any('paletteuse=dither=sierra2_4a' in str(p)
                   for p in commands[1])

    def test_palettegen_failure_raises(self, tmp_path, monkeypatch):
        import subprocess
        monkeypatch.setattr(
            subprocess, 'run',
            lambda cmd, **kw: SimpleNamespace(returncode=1, stderr='boom'))
        with pytest.raises(RuntimeError):
            vs_module._video2gif_ffmpeg(str(tmp_path / 'in.mp4'),
                                        str(tmp_path / 'out.gif'))

    def test_paletteuse_failure_raises(self, tmp_path, monkeypatch):
        import subprocess
        results = iter([
            SimpleNamespace(returncode=0, stderr=''),
            SimpleNamespace(returncode=1, stderr='pass2 bad'),
        ])
        monkeypatch.setattr(subprocess, 'run',
                            lambda cmd, **kw: next(results))
        with pytest.raises(RuntimeError):
            vs_module._video2gif_ffmpeg(str(tmp_path / 'in.mp4'),
                                        str(tmp_path / 'out.gif'))


class TestCropStandalone:
    def test_detect_bounds_block(self):
        img = make_white_image_with_block()
        bounds = vs_module._detect_content_bounds_standalone(img)
        # default helper image: block at rows 30-59, cols 40-79
        assert bounds == (30, 59, 40, 79)

    def test_detect_bounds_no_content(self):
        from PIL import Image
        img = Image.new('RGB', (50, 50), (255, 255, 255))
        assert vs_module._detect_content_bounds_standalone(img) is None

    def test_detect_bounds_rgba_composited(self):
        from PIL import Image
        img = Image.new('RGBA', (60, 60), (255, 255, 255, 0))
        px = img.load()
        for r in range(10, 20):
            for c in range(15, 25):
                px[c, r] = (0, 0, 0, 255)
        bounds = vs_module._detect_content_bounds_standalone(img)
        assert bounds == (10, 19, 15, 24)

    def test_consistent_crop_empty_folder(self, tmp_path):
        assert vs_module._apply_consistent_crop_standalone(
            str(tmp_path)) is None

    def test_consistent_crop_applies(self, tmp_path):
        for i in range(3):
            make_block_jpeg(str(tmp_path / f'deg_{i * 90}.0.jpeg'))
        result = vs_module._apply_consistent_crop_standalone(
            str(tmp_path), margin=5)
        # block rows 30-109, cols 40-129 plus 5px margin; JPEG edge
        # bleed can add a couple of pixels of detected content
        assert result is not None
        assert result[0] == 100
        assert 88 <= result[1] <= 96
        from PIL import Image
        with Image.open(str(tmp_path / 'deg_0.0.jpeg')) as img:
            assert img.size == result

    def test_consistent_crop_all_background(self, tmp_path):
        from PIL import Image
        for i in range(2):
            Image.new('RGB', (80, 60), (255, 255, 255)).save(
                str(tmp_path / f'deg_{i}.0.jpeg'), 'JPEG')
        assert vs_module._apply_consistent_crop_standalone(
            str(tmp_path)) is None


class TestImg2PptxWrapper:
    def test_wrapper_delegates(self, tmp_path, monkeypatch):
        import types
        fake_module = types.ModuleType('utils.report_utils')
        captured = {}

        def fake_img2pptx(**kwargs):
            captured.update(kwargs)
            return kwargs['output_pptx']

        fake_module.img2pptx = fake_img2pptx
        monkeypatch.setitem(sys.modules, 'utils.report_utils', fake_module)
        png = tmp_path / 'img.png'
        make_noise_png(str(png))
        out = vs_module.img2pptx(input_path=[str(png)],
                                 output_pptx=str(tmp_path / 'out.pptx'))
        assert out == str(tmp_path / 'out.pptx')
        assert captured['input_path'] == [str(png)]



# ============================================================
# Batch 7: WebDriverExportSession internals + video gif branches
# ============================================================

def _png_bytes_from_image(img):
    import io
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def _noise_png_bytes(size=(60, 60), seed=0):
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    return _png_bytes_from_image(Image.fromarray(arr))


def _black_png_bytes(size=(40, 40)):
    from PIL import Image
    return _png_bytes_from_image(Image.new('RGB', size, (0, 0, 0)))


def _content_png_bytes():
    return _png_bytes_from_image(make_white_image_with_block())


class SessionFakeChromeDriver:
    """Configurable fake Chrome driver for WebDriverExportSession internals."""

    def __init__(self, service=None, options=None):
        self.urls = []
        self.script_calls = []
        self.cdp_calls = []
        self.quit_called = False
        self.load_timeout = None
        self.script_timeout = None
        self.camera_return = None
        self.canvas_data = None
        self.async_data = None
        self.cdp_result = None
        self.cdp_raises = False
        self.emulation_raises = False
        self.script_raises = False

    def get(self, url):
        self.urls.append(url)

    def set_page_load_timeout(self, t):
        self.load_timeout = t

    def set_script_timeout(self, t):
        self.script_timeout = t

    def find_element(self, by, value):
        return object()

    def quit(self):
        self.quit_called = True

    def save_screenshot(self, path):
        make_noise_png(path)
        return True

    def execute_cdp_cmd(self, name, params):
        self.cdp_calls.append((name, params))
        if name == 'Emulation.setDeviceMetricsOverride':
            if self.emulation_raises:
                raise RuntimeError('cdp emulation broken')
            return {}
        if name == 'Page.captureScreenshot':
            if self.cdp_raises:
                raise RuntimeError('cdp screenshot broken')
            if self.cdp_result is not None:
                return self.cdp_result
            import base64
            return {'data': base64.b64encode(_noise_png_bytes()).decode()}

    def execute_script(self, script, *args):
        self.script_calls.append((script, args))
        if self.script_raises:
            raise RuntimeError('script engine broken')
        if 'return gd.layout.scene.camera' in script:
            return self.camera_return
        if "toDataURL('image/png')" in script:
            return self.canvas_data
        return None

    def execute_async_script(self, script, *args):
        self.script_calls.append((script, args))
        return self.async_data


def patch_selenium_session(monkeypatch, chrome_raises=False, installer_raises=False,
                           emulation_raises=False, camera_return='UNSET',
                           cleanup_interval=None):
    import selenium.webdriver as swd
    import webdriver_manager.chrome as wmc
    created = []

    class FakeInstaller:
        def install(self):
            if installer_raises:
                raise RuntimeError('wdm download failed')
            return '/fake/chromedriver'

    def chrome_factory(*args, **kwargs):
        if chrome_raises:
            raise RuntimeError('no chrome here')
        d = SessionFakeChromeDriver()
        d.emulation_raises = emulation_raises
        if camera_return != 'UNSET':
            d.camera_return = camera_return
        created.append(d)
        return d

    monkeypatch.setattr(wmc, 'ChromeDriverManager', FakeInstaller)
    monkeypatch.setattr(swd, 'Chrome', chrome_factory)
    return created


class TestWebDriverExportSessionLifecycle:
    def test_full_lifecycle_fixed_render_wait(self, tmp_path, monkeypatch):
        import json
        created = patch_selenium_session(monkeypatch)
        html = make_plotly_html(tmp_path / 'plot.html')
        with vs_module.WebDriverExportSession(
                width=300, height=250, scale=2, timeout=10,
                render_wait=0.05) as session:
            driver = created[0]
            assert session._viewport_configured is True
            session.load_html(str(html), wait_for_render=True, render_wait=0)
            assert driver.load_timeout == 10
            assert driver.script_timeout == 10
            assert session._loaded_url == os.path.abspath(str(html))
            assert session._cleanup_interval == 15
            assert session._render_wait_fixed is True
            assert session._render_wait == 0.05
            assert session._initial_camera == {}
            assert driver.urls and driver.urls[0].startswith('http://127.0.0.1:')

            session.set_trace_visibility([0, 5], 3)
            script, _ = driver.script_calls[-1]
            assert json.dumps([True, False, False]) in script

            session.update_layout({'showlegend': False})
            script, _ = driver.script_calls[-1]
            assert '"showlegend": false' in script

            session.set_camera({'x': 0, 'y': 2.5, 'z': 0}, up={'x': 0, 'y': 0, 'z': -1})
            script, _ = driver.script_calls[-1]
            assert '"y": 2.5' in script

            session.set_camera_for_rotation(1.0, 0.0, -2.5)

            png_out = tmp_path / 'shot.png'
            session.screenshot(str(png_out))
            assert png_out.exists()
            assert not (tmp_path / 'shot_temp.png').exists()

            jpeg_out = tmp_path / 'shot.jpeg'
            session.screenshot(str(jpeg_out), convert_to_jpeg=True, jpeg_quality=80)
            from PIL import Image
            with Image.open(str(jpeg_out)) as im:
                assert im.format == 'JPEG'
        assert driver.quit_called is True
        assert session._http_server is None
        assert ('Emulation.setDeviceMetricsOverride',
                {'width': 600, 'height': 500, 'deviceScaleFactor': 1,
                 'mobile': False}) in driver.cdp_calls

    def test_calibrated_render_wait_with_camera(self, tmp_path, monkeypatch):
        cam = {'eye': {'x': 1.0, 'y': 0.0, 'z': -2.5}}
        created = patch_selenium_session(monkeypatch, camera_return=cam)
        html = make_plotly_html(tmp_path / 'plot.html')
        with vs_module.WebDriverExportSession(
                width=300, height=250, scale=1, timeout=10) as session:
            session.load_html(str(html), wait_for_render=True, render_wait=0)
            assert session._initial_camera == cam
            assert 0.2 <= session._render_wait <= 0.6

    def test_calibrated_render_wait_no_camera(self, tmp_path, monkeypatch):
        created = patch_selenium_session(monkeypatch, camera_return=None)
        html = make_plotly_html(tmp_path / 'plot.html')
        with vs_module.WebDriverExportSession(
                width=300, height=250, scale=1, timeout=10) as session:
            session.load_html(str(html), wait_for_render=True, render_wait=0)
            # get_current_camera() coerces falsy driver results to {}
            assert session._initial_camera == {}
            assert 0.2 <= session._render_wait <= 0.6
            # Direct exercise of the no-camera calibration branch
            session._initial_camera = None
            session._calibrate_render_wait()
            assert session._render_wait == 0.4

    def test_load_html_no_render_wait(self, tmp_path, monkeypatch):
        created = patch_selenium_session(monkeypatch)
        html = make_plotly_html(tmp_path / 'plot.html')
        with vs_module.WebDriverExportSession(
                width=300, height=250, scale=1, timeout=10) as session:
            session.load_html(str(html), wait_for_render=False)
            assert session._initial_camera is None
            png_out = tmp_path / 'shot.png'
            session.screenshot(str(png_out))
            assert png_out.exists()

    def test_viewport_cdp_failure_flag(self, tmp_path, monkeypatch):
        created = patch_selenium_session(monkeypatch, emulation_raises=True)
        with vs_module.WebDriverExportSession(
                width=300, height=250, scale=1, timeout=5) as session:
            assert session._viewport_configured is False

    def test_http_server_port_exhaustion(self, tmp_path):
        import socket
        html = make_plotly_html(tmp_path / 'plot.html')
        session = vs_module.WebDriverExportSession(
            width=100, height=100, scale=1, timeout=5)
        socks = []
        try:
            for port in range(8765, 8865):
                so = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                so.bind(('127.0.0.1', port))
                socks.append(so)
            with pytest.raises(RuntimeError, match='available port'):
                session._start_http_server_for_file(str(html))
        finally:
            for so in socks:
                so.close()

    def test_large_html_cleanup_interval_and_memory_cleanup(self, tmp_path, monkeypatch):
        import gc
        created = patch_selenium_session(monkeypatch)
        html_path = tmp_path / 'big.html'
        html_path.write_text('<html><body>' + 'x' * (11 * 1024 * 1024) +
                             '</body></html>')
        with vs_module.WebDriverExportSession(
                width=300, height=250, scale=1, timeout=10,
                render_wait=0.05) as session:
            session.load_html(str(html_path), wait_for_render=True, render_wait=0)
            assert session._cleanup_interval == 10
            session._cleanup_interval = 1  # force cleanup on next screenshot
            n_scripts_before = len(created[0].script_calls)
            session.screenshot(str(tmp_path / 'shot.jpeg'), convert_to_jpeg=True)
            gc.collect()
            cleanup_scripts = [s for s, _ in created[0].script_calls[n_scripts_before:]
                               if 'window._lastImageData' in s]
            assert cleanup_scripts

    def test_screenshot_canvas_dataurl_method(self, tmp_path, monkeypatch):
        import base64
        created = patch_selenium_session(monkeypatch)
        html = make_plotly_html(tmp_path / 'plot.html')
        with vs_module.WebDriverExportSession(
                width=300, height=250, scale=1, timeout=10,
                render_wait=0.05) as session:
            session.load_html(str(html), wait_for_render=True, render_wait=0)
            created[0].canvas_data = (
                'data:image/png;base64,' +
                base64.b64encode(_noise_png_bytes()).decode())
            out = tmp_path / 'canvas.png'
            session.screenshot(str(out))
            assert out.exists()
            assert not (tmp_path / 'canvas_temp.png').exists()
            assert not any(name == 'Page.captureScreenshot'
                           for name, _ in created[0].cdp_calls)

    def test_screenshot_all_black_canvas_falls_back_to_cdp(self, tmp_path, monkeypatch):
        import base64
        created = patch_selenium_session(monkeypatch)
        html = make_plotly_html(tmp_path / 'plot.html')
        with vs_module.WebDriverExportSession(
                width=300, height=250, scale=1, timeout=10,
                render_wait=0.05) as session:
            session.load_html(str(html), wait_for_render=True, render_wait=0)
            created[0].canvas_data = (
                'data:image/png;base64,' +
                base64.b64encode(_black_png_bytes()).decode())
            out = tmp_path / 'black.png'
            session.screenshot(str(out))
            assert out.exists()
            assert any(name == 'Page.captureScreenshot'
                       for name, _ in created[0].cdp_calls)

    def test_screenshot_cdp_failure_final_fallback(self, tmp_path, monkeypatch):
        created = patch_selenium_session(monkeypatch)
        html = make_plotly_html(tmp_path / 'plot.html')
        with vs_module.WebDriverExportSession(
                width=300, height=250, scale=1, timeout=10,
                render_wait=0.05) as session:
            session.load_html(str(html), wait_for_render=True, render_wait=0)
            created[0].cdp_raises = True
            out = tmp_path / 'fallback.png'
            session.screenshot(str(out))
            assert out.exists()

    def test_screenshot_auto_crop_with_content(self, tmp_path, monkeypatch):
        import base64
        created = patch_selenium_session(monkeypatch)
        html = make_plotly_html(tmp_path / 'plot.html')
        with vs_module.WebDriverExportSession(
                width=300, height=250, scale=1, timeout=10,
                render_wait=0.05) as session:
            session.load_html(str(html), wait_for_render=True, render_wait=0)
            created[0].cdp_result = {
                'data': base64.b64encode(_content_png_bytes()).decode()}
            out = tmp_path / 'cropped.png'
            session.screenshot(str(out), auto_crop=True, margin=5,
                               background_color=(255, 255, 255))
            assert out.exists()
            from PIL import Image
            with Image.open(str(out)) as im:
                assert im.size == (50, 40)

    def test_get_current_camera_exception_returns_empty(self, tmp_path, monkeypatch):
        created = patch_selenium_session(monkeypatch)
        with vs_module.WebDriverExportSession(
                width=100, height=100, scale=1, timeout=5) as session:
            created[0].script_raises = True
            assert session.get_current_camera() == {}


class TestWebDriverExportSessionEnterFallbacks:
    def test_enter_selenium_missing(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, 'selenium', None)
        session = vs_module.WebDriverExportSession(100, 100, 1, 5)
        with pytest.raises(ImportError, match='selenium'):
            session.__enter__()

    def test_enter_wdm_missing_system_chrome_ok(self, tmp_path, monkeypatch):
        import sys
        created = patch_selenium_session(monkeypatch)
        monkeypatch.setitem(sys.modules, 'webdriver_manager.chrome', None)
        with vs_module.WebDriverExportSession(
                width=100, height=100, scale=1, timeout=5) as session:
            assert session._viewport_configured is True
        assert created[0].quit_called is True

    def test_enter_wdm_missing_chrome_fails(self, monkeypatch):
        import sys
        patch_selenium_session(monkeypatch, chrome_raises=True)
        monkeypatch.setitem(sys.modules, 'webdriver_manager.chrome', None)
        session = vs_module.WebDriverExportSession(100, 100, 1, 5)
        with pytest.raises(RuntimeError, match='Could not initialize'):
            session.__enter__()

    def test_enter_wdm_error_chrome_fails(self, monkeypatch):
        patch_selenium_session(monkeypatch, chrome_raises=True,
                               installer_raises=True)
        session = vs_module.WebDriverExportSession(100, 100, 1, 5)
        with pytest.raises(RuntimeError, match='Could not initialize'):
            session.__enter__()


class TestVideoGifBranches:
    def test_from_html_export_gif(self, tmp_path, monkeypatch):
        patch_write_image(monkeypatch, mode='ok')
        calls = []

        def fake_video2gif(video, gif, fps=30, scale=0.2, optimize=True, **kw):
            calls.append((video, gif))
            with open(gif, 'wb') as f:
                f.write(b'GIF89a')

        monkeypatch.setattr(vs_module, 'video2gif', fake_video2gif)
        html = make_plotly_html(tmp_path / 'plot.html')
        rc = vs_module.export_video_from_html(
            html, degree_per_frame=120, export_gif=True, gif_scale=0.5,
            gif_optimize=False, output_dir=str(tmp_path / 'out'))
        assert rc == 0
        assert len(calls) == 2
        for video, gif in calls:
            assert video.endswith('.mp4')
            assert os.path.exists(gif)

    def test_webdriver_export_gif(self, tmp_path, monkeypatch):
        calls = []

        def fake_video2gif(video, gif, fps=30, scale=0.2, optimize=True, **kw):
            calls.append((video, gif))
            with open(gif, 'wb') as f:
                f.write(b'GIF89a')

        monkeypatch.setattr(vs_module, 'video2gif', fake_video2gif)
        monkeypatch.setattr(vs_module, 'WebDriverExportSession', VideoFakeSession)
        html = make_plotly_html(tmp_path / 'plot.html')
        rc = vs_module.export_video_from_html(
            html, degree_per_frame=90, export_gif=True, gif_optimize=False,
            output_dir=str(tmp_path / 'out'), export_method='webdriver')
        assert rc == 0
        assert len(calls) == 2
        for video, gif in calls:
            assert os.path.exists(gif)
