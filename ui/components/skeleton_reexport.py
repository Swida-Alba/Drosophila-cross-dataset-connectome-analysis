"""Re-export individual profiles and rotating videos from a stored 3D skeleton page.

A ``plot-3d_*`` run folder keeps everything a re-render needs: the viewer page
carries the traces, and ``visualization_manifest.json`` (when the run wrote one)
carries the cameras. Both exports therefore run offline -- no NeuPrint query, no
skeleton cache, no original script -- which is what lets a scene be re-profiled
at another granularity, or turned into a video, long after it was rendered.

Results are written beside the source page in ``<page stem>_reexport/``, so the
run itself is never rewritten and re-running the same settings reuses the frames
already on disk.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from nicegui import ui

from ..config import (
    PROFILE_GRANULARITY_CHOICES,
    get_default_output_dir,
    get_storage_scan_roots,
    get_tab_output_dir,
    get_user_default,
)
from ..run_state import RUN_MANAGER
from ..runner import ScriptRunner
from .common import (
    checkbox_input,
    multi_select_input,
    number_input,
    param_grid,
    section_header,
    select_input,
)
from .output_panel import OutputPanel

#: Prefixes of the 3D skeleton run folders (``folder_prefix`` in
#: ``src/visualize_skeleton.py`` plus the legacy one-word spelling); these are
#: the only folders the page picker offers.
SKELETON_FOLDER_PREFIXES = ('plot-3d', 'plot3d')

#: The camera keys ``dataset_view_cameras()`` actually serves. Anything here
#: that the table lacks renders from the page's own framing and looks like a
#: real second view, so the two lists must stay in step.
PROFILE_VIEWS = ['front', 'back', 'top', 'bottom', 'left', 'right']

#: Upper bound on picker entries; pages are collected newest-first.
PAGE_LIMIT = 40


def _is_skeleton_run_folder(path: Path) -> bool:
    """A per-run ``plot-3d_*_<timestamp>`` folder, not a shared root."""
    if not path.name.startswith(SKELETON_FOLDER_PREFIXES):
        return False
    try:
        from src.utils.naming_utils import is_run_folder_name
    except ImportError:
        # Same guard as ui.runner's: src may not be importable as a package.
        return bool(re.search(r'_\d{8}_\d{6}$', path.name))
    return is_run_folder_name(path.name)


def _search_dirs() -> list[str]:
    """Output roots and recent run folders that can hold stored pages."""
    candidates = [
        get_default_output_dir(),
        get_tab_output_dir('visualization_skeleton'),
        get_tab_output_dir('visualization_skeleton_reexport'),
        *get_storage_scan_roots(),
    ]
    for record in RUN_MANAGER.recent_runs(200):
        for key in ('output_folder', 'output_dir'):
            value = record.get(key)
            if value:
                candidates.append(str(value))

    dirs: list[str] = []
    for value in candidates:
        text = os.path.normpath(str(Path(str(value)).expanduser()))
        if text not in dirs and os.path.isdir(text):
            dirs.append(text)
    return dirs


def candidate_pages(limit: int = PAGE_LIMIT) -> dict[str, str]:
    """``{page path: label}`` of stored skeleton viewer pages, newest first.

    Only the recorded output roots and recent run folders are scanned, and only
    ``plot-3d_*`` run folders count, so the picker can neither wander the
    filesystem nor offer a page another tool wrote.
    """
    found: list[tuple[int, str, str]] = []
    for root in _search_dirs():
        root_path = Path(root)
        if _is_skeleton_run_folder(root_path):
            run_dirs = [root_path]
        else:
            try:
                run_dirs = sorted(
                    (child for child in root_path.iterdir()
                     if child.is_dir() and _is_skeleton_run_folder(child)),
                    key=lambda child: child.name, reverse=True,
                )
            except OSError:
                run_dirs = []
        for run_dir in run_dirs:
            for page in sorted(run_dir.glob('*.html')):
                # A ``_simplified`` copy is the degraded duplicate the viewer
                # already resolves away from; offering both is duplicate noise.
                if page.name.endswith('_simplified.html'):
                    continue
                try:
                    stamp = int(page.stat().st_mtime)
                except OSError:
                    continue
                found.append((stamp, str(page),
                              f'{run_dir.name} · {page.name}'))
    found.sort(key=lambda item: item[0], reverse=True)
    return {page: label for _stamp, page, label in found[:limit]}


def create_skeleton_reexport(form_col, results_col):
    """Mount the re-export controls in *form_col* and its panel in *results_col*."""
    runner = ScriptRunner()
    output = OutputPanel(
        'Skeleton Re-export Output', state_key='visualization_skeleton_reexport'
    )

    with form_col:
        with ui.card().classes('w-full drocat-card').props(
            'id="card-skeleton-reexport"'
        ):
            section_header('Re-export from a Stored Page', 'file_download')
            ui.label(
                'Re-render the individual profiles and the rotating video of an '
                'existing 3D skeleton page, without querying the dataset again.'
            ).classes('text-caption drocat-muted')

            page_select = ui.select(
                {}, label='Stored 3D Skeleton Page', with_input=True,
            ).props('outlined options-dense').classes('w-full')
            page_select.tooltip(
                'Every plot-3d run folder under the configured output '
                'directories. A run from before the manifest existed still '
                're-exports: its profiles are then grouped from the page alone.'
            )

            def refresh_pages():
                options = candidate_pages()
                page_select.options = options
                if page_select.value not in options:
                    page_select.value = next(iter(options), None)
                page_select.update()

            with ui.row().classes('w-full items-center gap-3'):
                ui.button('Refresh list', icon='refresh', on_click=refresh_pages).props(
                    'outline no-caps'
                )
                empty_notice = ui.label('').classes('text-caption text-amber-8')
            refresh_pages()
            if not page_select.options:
                empty_notice.text = (
                    'No stored 3D skeleton pages in the output folders yet - '
                    'render one above, or point Settings → Storage at the '
                    'folder that holds your runs.'
                )

            ui.label('Individual profiles').classes('drocat-mini-label')
            with param_grid(3):
                export_profiles = checkbox_input(
                    'Re-export Profiles', True,
                    hint='Render one PNG per profile group into the _reexport '
                         'folder. Capped at 300 renders per run (groups x '
                         'views): the groups that fit still render and the '
                         'rest are skipped and named in the log, so separate '
                         'the plots into smaller runs if a page asks for more.',
                )
                granularity = select_input(
                    'Re-export Granularity', PROFILE_GRANULARITY_CHOICES,
                    get_user_default('profile_granularity'),
                    hint='How far the tree is walked to form one profile group. '
                         "What the page can be split into is decided by its own "
                         "legend: a page rendered in 'layer' legend mode has no "
                         "finer types to separate, and a page saved before the "
                         "per-trace identity stamp was added can only be split "
                         "by legend entry (newer renders record the layer, type "
                         "and bodyId of every trace).",
                )
                views = multi_select_input(
                    'Re-export Views', PROFILE_VIEWS, ['front'],
                    hint='Camera views rendered for each profile group.',
                )

            ui.label('Rotating video').classes('drocat-mini-label')
            with param_grid(3):
                export_video = checkbox_input(
                    'Re-export Video', False,
                    hint='Render a full turn and assemble forward/backward MP4s '
                         '(plus GIFs). Frames already in the output folder are '
                         'reused when the settings match.',
                )
                fps = number_input('Video FPS', 30, 5, 60, 5)
                degree_per_frame = number_input(
                    'Video Degrees / Frame', 2.0, 0.1, 5.0, 0.1,
                    hint='Rotation step per frame: 2 degrees is 180 frames for a '
                         'full turn.',
                )
                rotate = select_input(
                    'Video Rotate', ['horizontal', 'vertical'], 'horizontal',
                )
                export_gif = checkbox_input(
                    'Video Also GIF', True,
                    hint='Convert the videos to small GIFs as well.',
                )
                gif_scale = number_input('Video GIF Scale', 0.2, 0.05, 1.0, 0.05)

            ui.label('Rendering').classes('drocat-mini-label')
            with param_grid(3):
                export_method = select_input(
                    'Frame Method', ['webdriver', 'kaleido'], 'webdriver',
                    hint="How the video frames are rendered (profiles always "
                         "use one browser session). "
                         "'webdriver': one Chrome session loads the page once and "
                         'only the camera moves per frame, which is also the only '
                         'engine that handles scenes too large for kaleido. '
                         "'kaleido': renders every frame in-process, and is the "
                         'automatic fallback when no browser is available.',
                )
                export_scale = number_input(
                    'Frame Scale', 3, 1, 5,
                    hint='Resolution multiplier for the exported PNGs and frames.',
                )
                auto_crop = checkbox_input(
                    'Frame Auto-crop', True,
                    hint='Trim the empty background off each profile and video '
                         'frame, consistently across frames.',
                )

            ui.label(
                'Outputs are written beside the source page, in its own '
                '<name>_reexport/ folder; the run itself is never modified.'
            ).classes('text-caption drocat-muted')

    with results_col:
        output.create(
            run_label='Re-export from Page', run_icon='file_download'
        )

        async def run_reexport():
            page = str(page_select.value or '').strip()
            if not page or not os.path.isfile(page):
                ui.notify('Choose a stored 3D skeleton page to re-export',
                          type='warning')
                return
            if not export_profiles.value and not export_video.value:
                ui.notify('Enable at least one of the two exports',
                          type='warning')
                return

            output.clear()
            output.set_running(True)
            result = await output.run(
                runner,
                'plot3d_reexport',
                {'html_file': page},
                'reexport',
                method_params={
                    'export_individual_profiles': bool(export_profiles.value),
                    'granularity': granularity.value,
                    'views': views.value or ['front'],
                    'export_video': bool(export_video.value),
                    'fps': int(fps.value),
                    'degree_per_frame': float(degree_per_frame.value),
                    'rotate': rotate.value,
                    'export_gif': bool(export_gif.value),
                    'gif_scale': float(gif_scale.value),
                    'export_method': export_method.value,
                    'export_scale': int(export_scale.value),
                    'auto_crop': bool(auto_crop.value),
                },
                output_dir=os.path.dirname(page),
            )
            output.set_running(False)
            output.set_status(
                'Completed' if result['returncode'] == 0 else 'Failed',
                'green' if result['returncode'] == 0 else 'red',
            )
            output.show_files(
                result['files'],
                result.get('output_folder') or os.path.dirname(page),
            )

        output.run_button.on_click(run_reexport)
        output.cancel_button.on_click(runner.cancel)
