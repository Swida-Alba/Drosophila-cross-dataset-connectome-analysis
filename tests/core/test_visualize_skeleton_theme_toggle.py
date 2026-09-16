"""Tests for the light/dark theme switch embedded in exported HTML viewers."""

from pathlib import Path
import sys

import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visualize_skeleton import VisualizeSkeleton  # noqa: E402


def _make_visualizer(background_color='white'):
    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.background_color = background_color
    return visualizer


def _mesh_figure():
    fig = go.Figure()
    # Auto brain/VNC mesh color; the hex form is what
    # _apply_plotly_trace_color stores on the trace.
    fig.add_trace(go.Mesh3d(
        x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0], i=[0], j=[1], k=[2],
        color='#74a8d6', opacity=0.04,
    ))
    # An explicit user mesh color must never be flipped by the switch.
    fig.add_trace(go.Mesh3d(
        x=[0, 1, 0], y=[0, 0, 1], z=[1, 1, 1], i=[0], j=[1], k=[2],
        color='#ff0000', opacity=0.5,
    ))
    fig.add_trace(go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode='lines'))
    return fig


def test_auto_mesh_theme_colors_pair():
    colors = VisualizeSkeleton._auto_mesh_theme_colors()
    assert colors['light'] == 'rgba(116, 168, 214, 0.04)'
    assert colors['dark'] == 'rgba(116, 168, 214, 0.04)'


def test_effective_mesh_color_follows_background():
    white = _make_visualizer('white')
    black = _make_visualizer('black')
    assert white._get_effective_mesh_color('brain') == 'rgba(116, 168, 214, 0.04)'
    assert black._get_effective_mesh_color('brain') == 'rgba(116, 168, 214, 0.04)'
    assert black._get_effective_mesh_color('vnc') == 'rgba(116, 168, 214, 0.04)'


def test_collect_adaptive_mesh_indices_matches_only_auto_colors():
    visualizer = _make_visualizer()
    indices = visualizer._collect_adaptive_mesh_trace_indices(_mesh_figure())
    assert indices == [0]


def test_theme_toggle_html_starts_at_background_theme():
    light = _make_visualizer('white')
    html = light._theme_toggle_html(mesh_indices=[0])
    assert 'drocat-theme-toggle' in html
    assert '"initial": "light"' in html
    # Both adaptive mesh colors are baked in so the switch can flip either
    # way regardless of the theme the page was generated with; both themes
    # now share the #74a8d6 tone at 4% opacity.
    assert html.count('#74a8d6') >= 2
    assert '"meshTraces": [0]' in html
    # Inert under automation so webdriver exports never capture it.
    assert 'navigator.webdriver' in html

    dark = _make_visualizer('black')
    assert '"initial": "dark"' in dark._theme_toggle_html()


def test_theme_toggle_html_repositions_below_warning_banner():
    # The switch is fixed to the viewport's top-right corner and would
    # otherwise paint over the full-width warning banner; the injected
    # script must measure the banner and drop the switch below it.
    html = _make_visualizer('white')._theme_toggle_html(mesh_indices=[0])
    assert 'drocat-warning-container' in html
    assert 'positionBelowBanner' in html
    assert "addEventListener('resize', positionBelowBanner)" in html


def test_write_plotly_html_embeds_toggle_only_when_requested(tmp_path):
    visualizer = _make_visualizer()

    on_path = tmp_path / 'on.html'
    visualizer._write_plotly_html(_mesh_figure(), str(on_path), theme_toggle=True)
    on_html = on_path.read_text(encoding='utf-8')
    assert 'drocat-theme-toggle' in on_html
    assert '"meshTraces": [0]' in on_html

    off_path = tmp_path / 'off.html'
    visualizer._write_plotly_html(_mesh_figure(), str(off_path))
    assert 'drocat-theme-toggle' not in off_path.read_text(encoding='utf-8')


def test_toggle_injection_is_idempotent(tmp_path):
    visualizer = _make_visualizer()
    html_path = tmp_path / 'idempotent.html'
    visualizer._write_plotly_html(
        _mesh_figure(), str(html_path), theme_toggle=True
    )
    html_before = html_path.read_text(encoding='utf-8')

    visualizer._inject_page_extras(
        str(html_path), theme_toggle=True, mesh_indices=[0]
    )
    html_after = html_path.read_text(encoding='utf-8')
    assert html_after == html_before
    assert html_after.count('<button id="drocat-theme-toggle"') == 1


def test_warning_banner_and_toggle_coexist(tmp_path):
    visualizer = _make_visualizer()
    visualizer._in_page_warning_html = (
        lambda: '<div id="drocat-in-page-warning">demo</div>'
    )
    html_path = tmp_path / 'both.html'
    visualizer._write_plotly_html(
        _mesh_figure(), str(html_path), theme_toggle=True
    )
    html = html_path.read_text(encoding='utf-8')
    assert 'drocat-in-page-warning' in html
    assert 'drocat-theme-toggle' in html
    assert html.count('<button id="drocat-theme-toggle"') == 1


def test_banner_only_page_never_gets_toggle(tmp_path):
    visualizer = _make_visualizer()
    visualizer._in_page_warning_html = (
        lambda: '<div id="drocat-in-page-warning">demo</div>'
    )
    html_path = tmp_path / 'banner.html'
    visualizer._write_plotly_html(_mesh_figure(), str(html_path))
    html = html_path.read_text(encoding='utf-8')
    assert 'drocat-in-page-warning' in html
    assert 'drocat-theme-toggle' not in html
