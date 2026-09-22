"""The 3D Skeleton tab's re-export entrance (page picker, params, script).

``ui/components/skeleton_reexport.py`` offers stored ``plot-3d_*`` viewer pages
and turns the selection into a ``plot3d_reexport`` run. These tests cover the
picker scoping (a run folder's canonical page only, newest first), the card's
controls and defaults, the parameter dictionaries the panel hands the runner,
and the generated script's phase/progress contract.

Hermetic: every filesystem scan is pointed at ``tmp_path`` and the subprocess is
stubbed at ``OutputPanel.run``.
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import pytest

from nicegui import Client, ui
from nicegui.page import page

from ui.components.output_panel import OutputPanel

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import ui.components.skeleton_reexport as reexport  # noqa: E402
from ui.components.skeleton_reexport import (  # noqa: E402
    PAGE_LIMIT,
    PROFILE_VIEWS,
    candidate_pages,
)
from ui.runner import ScriptRunner  # noqa: E402


RUN_STAMP = '20260919_101112'


def _make_run(root: Path, name: str, pages=('scene',), stamp=RUN_STAMP):
    """A per-run output folder with viewer pages (plus its simplified copy)."""
    run_dir = root / f'{name}_{stamp}'
    run_dir.mkdir(parents=True)
    for page_stem in pages:
        (run_dir / f'{page_stem}.html').write_text('<html></html>')
        (run_dir / f'{page_stem}_simplified.html').write_text('<html></html>')
    (run_dir / 'visualization_manifest.json').write_text(
        json.dumps({'canonical_page': 'scene.html'}))
    return run_dir


@pytest.fixture
def scan_root(tmp_path, monkeypatch):
    """Point every search path at *tmp_path* and freeze the recent-runs list."""
    root = tmp_path / 'outputs'
    root.mkdir()
    monkeypatch.setattr(reexport, 'get_default_output_dir', lambda: str(root))
    monkeypatch.setattr(reexport, 'get_tab_output_dir', lambda scope: str(root))
    monkeypatch.setattr(reexport, 'get_storage_scan_roots', lambda: [])
    monkeypatch.setattr(reexport.RUN_MANAGER, 'recent_runs',
                        lambda limit=25: [])
    return root


# ---------------------------------------------------------------------------
# page discovery
# ---------------------------------------------------------------------------

def test_only_skeleton_run_pages_are_offered(scan_root):
    run = _make_run(scan_root, 'plot-3d_MCNS_aMe12')
    _make_run(scan_root, 'plot-network_MCNS', pages=('graph',))
    _make_run(scan_root, 'homologs')
    # A shared root that merely carries the prefix is not a run folder.
    (scan_root / 'plot-3d').mkdir()
    (scan_root / 'plot-3d' / 'scene.html').write_text('<html></html>')

    pages = candidate_pages()

    assert list(pages) == [str(run / 'scene.html')]
    # The degraded ``_simplified`` copy is resolved away from, never offered.
    assert not [p for p in pages if p.endswith('_simplified.html')]
    assert pages[str(run / 'scene.html')] == (
        f'{run.name} · scene.html')


def test_pages_are_newest_first_and_capped(scan_root):
    older = _make_run(scan_root, 'plot-3d_A', stamp='20260101_000000')
    newer = _make_run(scan_root, 'plot-3d_B', stamp='20260919_235959')
    now = time.time()
    os.utime(older / 'scene.html', (now - 10_000, now - 10_000))
    os.utime(newer / 'scene.html', (now, now))

    assert list(candidate_pages())[:2] == [
        str(newer / 'scene.html'), str(older / 'scene.html')]
    assert len(candidate_pages(limit=1)) == 1
    assert len(candidate_pages()) == 2
    assert PAGE_LIMIT > 1


def test_recent_run_folders_and_roots_are_included(tmp_path, monkeypatch,
                                                   scan_root):
    """Runs recorded by the run manager, and roots added in Settings, count."""
    recorded = _make_run(tmp_path / 'recorded', 'plot3d_MCNS_aMe12')
    extra = _make_run(tmp_path / 'extra_root', 'plot-3d_HM_aMe12')
    monkeypatch.setattr(reexport.RUN_MANAGER, 'recent_runs',
                        lambda limit=25: [{'output_folder': str(recorded)}])
    monkeypatch.setattr(reexport, 'get_storage_scan_roots',
                        lambda: [str(extra)])

    assert {str(recorded / 'scene.html'), str(extra / 'scene.html')} <= set(
        candidate_pages())


def test_missing_roots_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(reexport, 'get_default_output_dir',
                        lambda: str(tmp_path / 'never_created'))
    monkeypatch.setattr(reexport, 'get_tab_output_dir', lambda scope: '')
    monkeypatch.setattr(reexport, 'get_storage_scan_roots', lambda: [])
    assert candidate_pages() == {}


# ---------------------------------------------------------------------------
# the card, inside the Skeleton tab
# ---------------------------------------------------------------------------

def _build_skeleton_tab(monkeypatch, pages=None):
    """Mount the Skeleton tab with a *pages*-shaped scan instead of the real one.

    ``pages=None`` means "no stored runs at all", which is what a defaults
    test needs: the card pre-selects the newest page it can find, so scanning
    the developer's own output directory would put a real page in that slot.
    """
    from ui.tabs.visualization import create_skeleton_tab
    import ui.layer_style_store as layer_style_store

    monkeypatch.setattr(layer_style_store, '_store_dir', Path('/tmp') /
                        f'tab_drafts_{uuid.uuid4().hex}')
    monkeypatch.setattr(reexport, 'candidate_pages',
                        lambda limit=PAGE_LIMIT: dict(pages or {}))
    client = Client(page(f'/skeleton-reexport-{uuid.uuid4().hex}'))
    with client:
        create_skeleton_tab()
    return client


def _descendants(element):
    """Every element below *element* (the re-export card shares label texts
    with the render controls above it, so lookups must be scoped to the card)."""
    found = []
    stack = [element]
    while stack:
        current = stack.pop()
        for child in current.default_slot.children:
            found.append(child)
            stack.append(child)
    return found


def _card(client):
    card = _element(
        list(client.elements.values()),
        lambda el: (getattr(el, '_props', None) or {}).get('id')
        == 'card-skeleton-reexport',
        'the re-export card')
    descendants = _descendants(card)
    assert descendants, 'the re-export card holds no controls'
    return descendants


def _element(elements, predicate, what):
    matches = [el for el in elements if predicate(el)]
    assert len(matches) == 1, f'{what}: {len(matches)} matches'
    return matches[0]


def _labelled(client, label):
    return _element(
        _card(client),
        lambda el: (getattr(el, '_props', None) or {}).get('label') == label,
        f'label {label!r}')


def _checkbox(client, label):
    return _element(
        _card(client),
        lambda el: type(el).__name__ == 'Checkbox'
        and getattr(el, 'text', '') == label,
        f'checkbox {label!r}')


def _button(client, text):
    # Buttons live outside the card (the run button belongs to the output
    # panel), and their captions are unique on the page.
    return _element(
        list(client.elements.values()),
        lambda el: type(el).__name__ == 'Button'
        and getattr(el, 'text', '') == text,
        f'button {text!r}')


def _captions(elements):
    """Every control caption in *elements* (select labels + checkbox texts)."""
    captions = []
    for element in elements:
        label = (getattr(element, '_props', None) or {}).get('label')
        if label:
            captions.append(str(label))
        if type(element).__name__ == 'Checkbox' and getattr(element, 'text', ''):
            captions.append(str(element.text))
    return captions


def _click(element):
    listener = next(listener for listener in element._event_listeners.values()
                    if listener.type == 'click')
    result = listener.handler.__closure__[0].cell_contents()
    if asyncio.iscoroutine(result):
        asyncio.run(result)


def test_card_shows_the_reexport_controls_with_render_parity_defaults(
        monkeypatch):
    """The knobs mirror the live export block, so a re-export can match a run."""
    client = _build_skeleton_tab(monkeypatch)

    assert _labelled(client, 'Stored 3D Skeleton Page').value is None
    granularity = _labelled(client, 'Re-export Granularity')
    assert granularity.value == 'legend'
    assert set(granularity.options) == {'legend', 'layer', 'type', 'body'}
    assert _labelled(client, 'Frame Method').value == 'webdriver'
    assert _checkbox(client, 'Re-export Profiles').value is True
    assert _checkbox(client, 'Re-export Video').value is False
    # A full turn at 2 degrees/frame is the 180-frame default the tab uses.
    assert _labelled(client, 'Video Degrees / Frame').value == 2.0
    assert _button(client, 'Refresh list') is not None
    assert _button(client, 'Re-export from Page') is not None

    captions = [getattr(el, 'text', '') for el in client.elements.values()]
    assert any('_reexport/' in str(text) for text in captions), \
        'the panel must say where a re-export writes'

    # No caption may repeat one the render controls already use: the Skeleton
    # tab is a single page, and a duplicated label makes both controls
    # ambiguous (test_skeleton_profile_options hit exactly that).
    card = _card(client)
    mine = set(_captions(card))
    theirs = set(_captions(list(client.elements.values()))) - mine
    assert not mine & theirs, f'duplicated captions: {sorted(mine & theirs)}'


def test_picker_selects_the_newest_page_and_refresh_repicks(monkeypatch,
                                                            tmp_path):
    page_a = tmp_path / 'a.html'
    page_a.write_text('<html></html>')
    page_b = tmp_path / 'b.html'
    page_b.write_text('<html></html>')
    client = _build_skeleton_tab(
        monkeypatch, pages={str(page_a): 'a', str(page_b): 'b'})

    selector = _labelled(client, 'Stored 3D Skeleton Page')
    assert selector.value == str(page_a)

    _click(_button(client, 'Refresh list'))
    assert selector.value == str(page_a)

    # A page that vanished from disk is replaced on the next refresh rather
    # than being re-exported as a missing file.
    selector.value = str(tmp_path / 'gone.html')
    _click(_button(client, 'Refresh list'))
    assert selector.value == str(page_a)


def _capture_run(monkeypatch):
    captured = []

    async def fake_run(self, runner, tool_name, constructor_params,
                       method_name, method_params=None, output_dir=None):
        captured.append((tool_name, constructor_params, method_name,
                         dict(method_params or {}), output_dir))
        return {'returncode': 0, 'files': [], 'duration': 0,
                'cancelled': False, 'output_folder': output_dir,
                'neuron_match': None}

    monkeypatch.setattr(OutputPanel, 'run', fake_run)
    for name in ('set_running', 'set_status', 'clear', 'log'):
        monkeypatch.setattr(OutputPanel, name, lambda self, *a, **k: None)
    monkeypatch.setattr(OutputPanel, 'show_files',
                        lambda self, files, output_dir=None: None)
    return captured


def test_run_sends_the_page_and_the_two_phase_settings(monkeypatch, tmp_path):
    captured = _capture_run(monkeypatch)
    page = tmp_path / 'plot-3d_run' / 'scene.html'
    page.parent.mkdir()
    page.write_text('<html></html>')
    client = _build_skeleton_tab(monkeypatch, pages={str(page): 'scene'})

    _labelled(client, 'Re-export Granularity').value = 'type'
    _labelled(client, 'Frame Method').value = 'kaleido'
    _checkbox(client, 'Re-export Video').value = True
    _click(_button(client, 'Re-export from Page'))

    tool_name, constructor, method_name, method_params, output_dir = captured[0]
    assert tool_name == 'plot3d_reexport'
    assert method_name == 'reexport'
    assert constructor == {'html_file': str(page)}
    # The scan root is the run folder: the script announces the real
    # <stem>_reexport/ target through the standard output marker.
    assert output_dir == str(page.parent)
    assert method_params['granularity'] == 'type'
    assert method_params['export_method'] == 'kaleido'
    assert method_params['export_individual_profiles'] is True
    assert method_params['export_video'] is True
    assert method_params['views'] == ['front']
    # Viewer-only knobs of the render pipeline never leak into a re-export.
    assert 'legend_mode' not in method_params
    assert 'freeze_view' not in method_params


def test_run_refuses_without_a_page_or_an_enabled_phase(monkeypatch, tmp_path):
    captured = _capture_run(monkeypatch)
    notes = []
    monkeypatch.setattr(ui, 'notify', lambda message, **kw: notes.append(message))

    page = tmp_path / 'plot-3d_run' / 'scene.html'
    page.parent.mkdir()
    page.write_text('<html></html>')
    client = _build_skeleton_tab(monkeypatch, pages={str(page): 'scene'})

    _labelled(client, 'Stored 3D Skeleton Page').value = ''
    _click(_button(client, 'Re-export from Page'))
    assert captured == [] and any('page' in str(note).lower() for note in notes)

    _labelled(client, 'Stored 3D Skeleton Page').value = str(page)
    _checkbox(client, 'Re-export Profiles').value = False
    _click(_button(client, 'Re-export from Page'))
    assert captured == [] and any('export' in str(note).lower() for note in notes)

    _checkbox(client, 'Re-export Video').value = True
    _click(_button(client, 'Re-export from Page'))
    assert len(captured) == 1


# ---------------------------------------------------------------------------
# the generated script
# ---------------------------------------------------------------------------

BOTH = {
    'export_individual_profiles': True, 'granularity': 'body',
    'views': ['front', 'left'], 'export_video': True, 'fps': 30,
    'degree_per_frame': 2.0, 'rotate': 'horizontal', 'export_gif': True,
    'gif_scale': 0.2, 'export_method': 'webdriver', 'export_scale': 3,
    'auto_crop': True,
}


def _script(method_params, html_file='/runs/plot-3d_x/scene.html'):
    return ScriptRunner()._generate_script(
        'plot3d_reexport', {'html_file': html_file}, 'reexport', method_params)


def test_script_reexports_without_touching_a_dataset():
    script = _script(BOTH)
    compile(script, 'reexport.py', 'exec')

    assert 'VisualizeSkeleton(' not in script, 'no class is constructed'
    assert 'export_individuals_from_html(' in script
    assert 'export_video_from_html(' in script
    # One output folder for both phases, beside the page, announced so the
    # results panel links the real run folder.
    assert 'output_dir = reexport_output_dir(html_file)' in script
    assert 'output_dir=os.path.join(output_dir, "individual_profiles")' in script
    assert '[DROCAT] Output will be saved to: {output_dir}' in script
    assert "granularity='body'" in script and "views=['front', 'left']" in script
    assert "export_method='webdriver'" in script
    assert '[DROCAT][progress] 1/2' in script
    assert '[DROCAT][progress] 2/2' in script
    # A failed video render must not report success.
    assert '_video_rc = export_video_from_html(' in script
    assert 'sys.exit(_video_rc)' in script


def test_script_emits_only_the_enabled_phases():
    profiles = _script({'export_individual_profiles': True})
    video = _script({'export_video': True, 'export_method': 'kaleido'})
    compile(profiles, 'p.py', 'exec')
    compile(video, 'v.py', 'exec')

    assert profiles.count('[DROCAT][progress]') == 1
    assert '[DROCAT][progress] 1/1 Export individual profiles' in profiles
    assert 'export_video_from_html(' not in profiles
    assert video.count('[DROCAT][progress]') == 1
    assert '[DROCAT][progress] 1/1 Export rotating video' in video
    assert 'export_individuals_from_html(' not in video
    assert "export_method='kaleido'" in video
    # Omitted knobs keep the backend's own defaults rather than guessing.
    assert "granularity='legend'" in profiles
    assert 'auto_crop=True' in profiles


def test_progress_bar_steps_match_the_generated_script():
    """The panel's step labels come from the same flags as the script's."""
    from ui.components.page_progress import progress_steps_for

    for context in ({'export_individual_profiles': True,
                     'export_video': True},
                    {'export_individual_profiles': True},
                    {'export_video': True}):
        steps = progress_steps_for('plot3d_reexport', context=context)
        script = _script(context)
        assert len(steps) == script.count('[DROCAT][progress]'), context
        assert [step.lower() in script.lower() for step in steps] == \
            [True] * len(steps), steps


def test_reexport_tool_has_an_output_guide_spec():
    from ui.output_guide import TOOL_GUIDE_SPECS

    spec = TOOL_GUIDE_SPECS['plot3d_reexport']
    patterns = {entry['pattern'] for entry in spec['files']}
    assert {'individual_profiles/*', '*_reexport/*'} <= patterns


def test_the_view_options_are_exactly_the_cameras_that_exist():
    """Both profile-view dropdowns list camera keys, never friendly names.

    A view the table lacks resolves to no camera, so the export silently
    re-renders the page's own framing and names the file after the view the
    user asked for. Listing the table's own keys makes that unrepresentable.
    """
    from visualize_skeleton import dataset_view_cameras

    datasets = ['hemibrain:v1.2.1', 'fafb:v5.0', 'BANC', 'male-cns',
                'manc:v1']
    for dataset in datasets:
        for brain_mesh in (None, 'native', 'FAFB', 'BANC', 'male-cns'):
            offered = set(dataset_view_cameras(
                dataset, brain_mesh, lowercase=True))
            assert set(PROFILE_VIEWS) <= offered, (dataset, brain_mesh)
