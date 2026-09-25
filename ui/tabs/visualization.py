"""Visualization tabs - 3D Skeleton and Net-Viz (path network visualization).

Note: the interactive path-building Network tab lives in ui/tabs/network.py;
this module's pathway-graph tab is named Net-Viz (net_viz) to avoid any
name collision with it.
"""

import tempfile
from datetime import datetime
from pathlib import Path

from nicegui import ui

from ..config import (
    SKELETON_MODES,
    BRAIN_MESH_OPTIONS,
    LEGEND_MODES,
    NETWORK_LAYOUTS,
    PROFILE_GRANULARITY_CHOICES,
    SEARCH_COLUMNS,
    SYNAPSE_SIZE_OPTIONS,
    SYNAPSE_MODE_OPTIONS,
    SYNAPSE_VIEW_MODES,
    PRE_POST_SHAPES,
    LAYER_EDITOR_MODES,
    NET_VIZ_SOURCE_MODES,
    get_user_default,
    has_user_default,
    is_valid_synapse_size,
)
from ..components.banner import push_banner
from ..components.common import (
    combo_input, dataset_selector, multi_select_input, number_input, select_input,
    checkbox_input, dir_input, read_upload_event, section_header, param_grid, tool_page,
    apply_filter_mode,
)
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner
from ..components.layer_tree_editor import layer_tree_editor
from ..components.layer_style_editor import layer_style_editor
from ..components.skeleton_reexport import (
    PROFILE_VIEWS, create_skeleton_reexport)
from ..components.palette_picker import (
    palette_picker,
    palette_editor,
    assign_palette_colors,
    notify_empty_custom_palettes,
)
from ..components.edge_list_editor import edge_list_editor
from ..dataset_service import is_banc_dataset
from .. import layer_style_store
from visualization_options import default_skeleton_tab_simplification
from ..roi_options import (
    load_roi_catalog,
    roi_options_from_catalog,
)


def _offer_layer_draft_recovery(layer_style, *, on_advanced=None):
    """Build (but do not open) the draft-recovery dialog and return ``offer``.

    The draft store persists every edited table to ``local_data/layer_style_drafts/``
    with a ``dirty`` flag cleared only on export/visualization. On a fresh load any
    draft still marked dirty (unexported and unvisualized) is offered for recovery, so
    a crash or quit does not silently lose edits. ``Continue`` loads it back into the
    editor and switches to the Advanced panel (via ``on_advanced``); ``Discard`` deletes it.

    ``offer()`` presents the dialog only when the Skeleton tab is actually shown,
    so it does not pop up on app load or while the user is on another tab.
    """
    state = {"dialog": None, "message": None}

    def _discard(name):
        try:
            layer_style_store.delete_draft(name)
        finally:
            if state["dialog"] is not None:
                state["dialog"].close()

    def _continue(name):
        try:
            rows = layer_style_store.load_draft(name)
            if rows:
                layer_style.set_rows(rows, name=name)
            if on_advanced is not None:
                on_advanced()
            elif layer_style.expansion is not None:
                layer_style.expansion.set_value(True)
        finally:
            if state["dialog"] is not None:
                state["dialog"].close()

    def offer() -> None:
        if state["dialog"] is None:
            return
        try:
            pending = layer_style_store.pending_drafts()
        except Exception:
            return
        if not pending:
            return
        draft = pending[0]
        if state["message"] is not None:
            state["message"].text = (
                f"Draft '{draft.get('name')}' (updated {draft.get('updated_at')}) has "
                "unsaved changes that were not exported or visualized."
            )
        state["dialog"].open()

    try:
        pending = layer_style_store.pending_drafts()
    except Exception:
        pending = []
    if pending and not state["dialog"]:
        draft = pending[0]
        with ui.dialog() as dlg, ui.card().classes("w-96"):
            ui.label("Continue auto-saved layer style?").classes("text-subtitle1")
            _message = ui.label(
                f"Draft '{draft.get('name')}' (updated {draft.get('updated_at')}) has "
                "unsaved changes that were not exported or visualized."
            ).classes("text-caption drocat-muted")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button(
                    "Discard",
                    on_click=lambda d=draft["name"]: _discard(d),
                ).props("flat outline dense")
                ui.button(
                    "Continue",
                    on_click=lambda d=draft["name"]: _continue(d),
                ).props("color=primary dense")
        state["dialog"] = dlg
        state["message"] = _message

    return offer


def _flatten_neuron_layers(neuron_layers) -> list:
    """Flatten the nested layer model into one neuron per entry for the
    per-neuron palette counts (single-neuron layers are plain values)."""
    return [
        n for layer in neuron_layers
        for n in (layer if isinstance(layer, list) else [layer])
    ]


def _palette_colors_for_count(palette, count: int) -> list:
    """Resolve a palette for a known render count.

    Continuous preset selections must span the selected range; discrete and
    custom palettes retain their displayed order/cycling behavior.
    """
    getter = getattr(palette, "get_colors_for_count", None)
    if callable(getter):
        return getter(count)
    return assign_palette_colors(palette.get_colors(), count)


COLOR_PRESETS = {
    "Cool": {"source": "#4A90E2", "intermediate": "#50E3C2", "target": "#B8E986", "link": "rgba(74,144,226,0.3)"},
    "Default": {"source": "#1f77b4", "intermediate": "rgba(44,160,44,1.0)", "target": "#d62728", "link": "rgba(100,100,100,0.6)"},
    "Warm": {"source": "#FF6B6B", "intermediate": "#FFA500", "target": "#FFD700", "link": "rgba(255,107,107,0.3)"},
    "Purple": {"source": "#9C27B0", "intermediate": "#BA68C8", "target": "#FF1493", "link": "rgba(156,39,176,0.3)"},
}

# Neuron color palettes (per layer), matched to utils.color_utils.generate_color_palette
# Path-network color schemes (source / intermediate / target / link)
PATH_SCHEMES = [
    (name, [scheme["source"], scheme["intermediate"], scheme["target"], scheme["link"]])
    for name, scheme in COLOR_PRESETS.items()
]

COMMON_ROIS = [
    "AL", "LH", "EB", "FB", "PB", "NO", "CA", "PED", "aL", "bL", "gL",
    "AB", "BU", "LAL", "AOTU", "AVLP", "PVLP", "PLP", "WED", "SLP",
    "SIP", "SMP", "CRE", "SCL", "ICL", "IB", "ATL", "VES", "EPA",
    "GOR", "SPS", "IPS", "SAD", "FLA", "CAN", "PRW", "GNG",
    "ME", "LO", "LOP", "AME",
]


def banc_synapse_view_default(dataset: str) -> str:
    """Default synapse view mode for *dataset*.

    BANC defaults to 'skip': synapse markers need a one-time ~3.9 GB
    per-synapse table download, and the release publishes only pre-synaptic
    site coordinates (markers are drawn at pre-sites).
    """
    from ..dataset_service import is_banc_dataset

    return "skip" if is_banc_dataset(dataset) else "synapse"


def banc_synapse_warning(dataset: str, view: str):
    """Explicit warning text when BANC synapse rendering is opted into."""
    from ..dataset_service import is_banc_dataset

    if not is_banc_dataset(dataset) or view == "skip":
        return None
    return (
        "BANC synapse rendering is enabled: the per-synapse table (~3.9 GB) "
        "downloads once from the public BANC bucket into "
        "datasets/<dataset>/downloads/ (resumable). The BANC release provides "
        "only PRE-synaptic site coordinates, so markers are drawn at "
        "pre-sites (scatter) regardless of the selected shape."
    )


def mesh_color_backend_value(auto, color, opacity) -> str:
    """Compose a backend mesh color from the tab's trio of widgets.

    'auto' while Auto is on, otherwise the picked color with the chosen opacity
    baked in as rgba() -- the backend reads one string per mesh, and the two
    template halves of a male-cns/banc scene are coloured independently (the
    nerve cord whether the checkbox shows it or the page only embeds it hidden).
    """
    if auto.value:
        return "auto"
    hex_color = str(color.value or "#74A8D6").strip()
    alpha = float(opacity.value or 1.0)
    if alpha >= 1.0:
        return hex_color
    digits = hex_color[1:]
    if len(digits) == 3:
        digits = "".join(ch * 2 for ch in digits)
    try:
        r, g, b = (int(digits[i:i + 2], 16) for i in (0, 2, 4))
    except (ValueError, IndexError):
        return hex_color
    return f"rgba({r}, {g}, {b}, {alpha:g})"


def create_skeleton_tab():
    skeleton_runner = ScriptRunner()
    skeleton_output = OutputPanel(
        "3D Skeleton Output", state_key="visualization_skeleton"
    )

    form_col, results_col = tool_page(
        "3D Skeleton",
        "Interactive neuron morphology, synapse, and brain-region rendering.",
        icon="view_in_ar",
        doc="skeleton.md",
    )

    with form_col:
        ui.label(
            "Configure and render neuron morphology independently from network drawings."
        ).classes("text-caption drocat-muted")

        with ui.card().classes("w-full drocat-card").props('id="card-skeleton-dataset"'):
            section_header("Dataset", "storage")
            dataset = dataset_selector()
            output_dir = dir_input(scope="visualization_skeleton")
            # BANC skeleton source selection removed: the chain is unified
            # (888 L2 -> 888 full -> v626 pcg), no per-run choice. The BANC
            # tube-radius controls live in General Appearance.

        # ================= 3D Skeleton panel =================
        with ui.card().classes("w-full drocat-card").props('id="card-3d"'):
            section_header("Neuron Selection (3D)", "hub")
            # The three layer/color input modes are mutually exclusive and are
            # chosen with segmented buttons.
            layer_editor_mode = {"value": "Standard"}
            # Neuron/Synapse palette editors are hidden outside the Standard
            # mode; the ROI palette is mode-independent (the layer editor
            # defines neurons and synapses, never ROI meshes).
            color_editor_refs = {"neuron": None, "synapse": None}
            file_upload_path = {"path": None}
            # The Advanced editor is constructed before the shared Search
            # Columns control below; initialize the closure so suggestion
            # requests during page construction safely use the auto scope.
            search_columns = None
            with ui.row().classes(
                "w-full items-center justify-between gap-4 px-2 flex-nowrap"
            ):
                mode_buttons = {}
                for _mode in LAYER_EDITOR_MODES:
                    _btn = ui.button(_mode).props("outline no-caps").classes("w-1/3")
                    _btn.style(
                        "min-height: 3rem; font-size: 1.05rem; font-weight: 700;"
                    )
                    mode_buttons[_mode] = _btn

            # Synapse mode + threshold sit directly below the layer-input
            # selection buttons (synapse mode on the left, threshold on the
            # right). The marker-shape sub-selectors live in the Synapse Colors
            # panel further down.
            with ui.row().classes("w-full items-center gap-4 flex-nowrap") as synapse_mode_row:
                synapse_view_mode = select_input(
                    "Synapse Mode", SYNAPSE_VIEW_MODES, "synapse",
                    hint="'synapse': paired synapses between adjacent layers. "
                         "'pre-post sites': the pre- and post-synaptic SITES of the "
                         "queried neurons (post/input as spheres, pre/output as cones, "
                         "or circle/diamond scatter markers). 'skip': hide synapse markers.",
                ).props("outlined").classes("flex-1 min-w-[240px]")
                synapse_threshold = number_input(
                    "Synapse Threshold", get_user_default("min_synapse_num"), 1, 100,
                    hint="Minimum synapse weight for a marker to be shown. In pre-post site "
                         "mode, only sites whose connection with a neighbour meets this "
                         "threshold are shown.",
                ).classes("w-44")

            # Keep the interpretation warning attached to the mode selector:
            # it describes what "pre-post sites" means, not how those sites
            # are shaped. This also keeps it visible above the mode-specific
            # controls when the Synapse Colors card is scrolled.
            pre_post_warning = ui.label(
                "⚠️ Pre/post site mode: this shows the pre- and post-synaptic SITES "
                "of the queried neurons, not the paired synapses between adjacent "
                "layers. Every queried neuron's sites are shown regardless of which "
                "other neuron they connect to, with a separate legend for pre and "
                "post sites per layer. This affects visualization only."
            ).classes("w-full text-amber-8").style("font-size:14px").props(
                'id="card-skeleton-pre-post-warning"'
            ).set_visibility(False)

            standard_editor_panel = ui.column().classes("w-full").props(
                'id="card-skeleton-layers"'
            )
            advanced_editor_panel = ui.column().classes("w-full").props(
                'id="card-skeleton-layer-style"'
            ).set_visibility(False)
            file_upload_panel = ui.column().classes("w-full").props(
                'id="card-skeleton-layer-upload"'
            ).set_visibility(False)
            with standard_editor_panel:
                layer_tree = layer_tree_editor(
                    dataset_provider=lambda: dataset.value if dataset is not None else ""
                )
            with advanced_editor_panel:
                layer_style = layer_style_editor(
                    export_dir_provider=lambda: output_dir.value if output_dir.value else None,
                    dataset_provider=lambda: dataset.value if dataset is not None else "",
                    search_columns_provider=lambda: (
                        search_columns.value if search_columns is not None else "auto"
                    ),
                )

            def _recovery_to_advanced() -> None:
                # 'Continue' loads the draft into the Advanced Layer Editor and
                # switches the layer-input mode to 'Advanced' so it is visible.
                layer_editor_mode["value"] = "Advanced"
                _sync_layer_editor_mode()

            # On reopen of the Skeleton tab, offer to continue any auto-saved
            # (dirty, not-yet-exported or visualized) draft so a crash does not
            # silently lose edits. The dialog is opened only when the tab is shown.
            recovery_offer = _offer_layer_draft_recovery(
                layer_style, on_advanced=_recovery_to_advanced
            )
            with file_upload_panel:
                with ui.card().classes("w-full drocat-card").props(
                    'id="card-skeleton-layer-upload-card"'
                ):
                    section_header("Custom-layer CSV (file upload)", "upload_file")
                    ui.label(
                        "Upload a CSV of layer assignments and per-neuron colors. "
                        "Colors are read from the file (no separate color editors). "
                        "Columns: layer, neuron, color, synapse_color, "
                        "pre_synaptic_color, post_synaptic_color."
                    ).classes("text-caption drocat-muted")
                    with ui.row().classes("w-full items-center gap-2") as upload_row:
                        with ui.button(icon="upload_file").props("flat dense round").classes(
                            "drocat-upload-trigger"
                        ).tooltip("Upload a custom-layer CSV"):
                            with ui.menu() as layer_upload_menu:
                                ui.label("Custom-layer CSV").classes(
                                    "text-caption drocat-muted px-3 pt-2"
                                )
                                ui.label(
                                    "layer, neuron, color, synapse_color, "
                                    "pre_synaptic_color, post_synaptic_color; "
                                    "layer may start at 0 or 1."
                                ).classes("text-caption drocat-muted px-3 pb-1")
                                ui.upload(
                                    label="Choose CSV",
                                    on_upload=lambda e: _handle_layer_upload(
                                        e, layer_upload_menu
                                    ),
                                    auto_upload=True,
                                ).props('accept=".csv" flat dense').classes("w-72")
                        ui.link(
                            "File format",
                            "docs/ui_guides/skeleton.html#layer-editor-modes",
                        ).classes("drocat-doc-link")
                    file_upload_label = ui.label("No CSV loaded.").classes(
                        "text-caption drocat-muted drocat-truncate"
                    )

            def _sync_layer_editor_mode():
                active = layer_editor_mode["value"]
                standard_editor_panel.set_visibility(active == "Standard")
                advanced_editor_panel.set_visibility(active == "Advanced")
                file_upload_panel.set_visibility(active == "File upload")
                for _mode, _btn in mode_buttons.items():
                    _btn.props("color=primary" if _mode == active else "color=grey-7")
                # Show the Advanced Layer Editor table directly (do not leave it
                # collapsed behind its expansion header) when Advanced is active.
                if active == "Advanced" and layer_style.expansion is not None:
                    layer_style.expansion.set_value(True)
                # The Neuron/Synapse color palette editors are only used by
                # the Standard mode; the Advanced table / uploaded CSV supply
                # their own color columns. The ROI palette is independent of
                # the layer editor (it defines neurons and synapses, never
                # ROI meshes) and stays visible in every mode.
                show_colors = active == "Standard"
                for _name in ("neuron", "synapse"):
                    _handle = color_editor_refs[_name]
                    if _handle is not None:
                        _handle.set_visibility(show_colors)

            def _set_layer_editor_mode(value: str):
                layer_editor_mode["value"] = value
                _sync_layer_editor_mode()

            for _mode, _btn in mode_buttons.items():
                _btn.on_click(lambda _e, m=_mode: _set_layer_editor_mode(m))

            async def _handle_layer_upload(e, _menu):
                """Read an uploaded custom-layer CSV and remember its path."""
                temporary_path = None
                previous_path = file_upload_path.get("path")
                try:
                    filename, data = await read_upload_event(e)
                    with tempfile.NamedTemporaryFile(
                        suffix=".csv", prefix="drocat_layers_", delete=False
                    ) as temporary:
                        temporary.write(data)
                        temporary_path = temporary.name
                    file_upload_path["path"] = temporary_path
                    if previous_path and previous_path != temporary_path:
                        Path(previous_path).unlink(missing_ok=True)
                    file_upload_label.text = f"Loaded: {filename} ({len(data) / 1024:.1f} KB)"
                    file_upload_label.classes(replace="text-caption drocat-ok")
                except Exception as ex:
                    if temporary_path:
                        Path(temporary_path).unlink(missing_ok=True)
                    if previous_path:
                        Path(previous_path).unlink(missing_ok=True)
                    file_upload_path["path"] = None
                    file_upload_label.text = f"Error: {ex}"
                    file_upload_label.classes(replace="text-caption drocat-err")
                _menu.close()

            _sync_layer_editor_mode()
            with param_grid(3):
                filter_mode = select_input(
                    "Match by",
                    ["exact", "startswith", "contains", "endswith", "regex"],
                    "exact",
                    hint="The match mode (exact / starts with / contains / ends with / "
                         "regex) applies to every neuron in the layer tree, matching "
                         "the pathfinding search backend.",
                )
                search_columns = select_input(
                    "Search Columns", SEARCH_COLUMNS, get_user_default("search_columns"),
                    hint="Which columns to search when resolving neuron names (same as "
                         "pathfinding). 'auto': all columns (bodyId -> type -> instance -> "
                         "flywireType/others). Use 'type'/'instance'/'bodyId' to restrict.",
                )
                hemisphere = select_input(
                    "Hemisphere", ["both", "left", "right"], "both",
                    hint="'both': plot all neurons. 'left'/'right': plot only that hemisphere. "
                         "Neurons WITHOUT an explicit hemisphere (no _L/_R instance suffix "
                         "or Soma side) are always included in every option.",
                )

            # ------------------------------------------------------------------
            # General Appearance (independent block)
            # ------------------------------------------------------------------
            with ui.card().classes("w-full drocat-card").props('id="card-skeleton-appearance"'):
                section_header("General Appearance", "palette")
                with param_grid(2):
                    skeleton_mode = select_input(
                        "Skeleton Mode", SKELETON_MODES, get_user_default("skeleton_mode"),
                        hint="'tube': 3D tube rendering (detailed). 'line': thin line (fast, for many neurons).",
                    )
                    legend_mode = select_input(
                        "Neuron Legend Mode", LEGEND_MODES, get_user_default("legend_mode"),
                        hint="'layer': one neuron legend entry per layer (or per custom group). "
                             "'type': per neuron type. 'tree': per type, plus a "
                             "collapsible type -> bodyId/instance panel in the "
                             "exported HTML, or group -> type -> bodyId/instance "
                             "with custom groups. "
                             "'single': every neuron. "
                             "This only sets the LEGEND levels; the individual-profile "
                             "levels are chosen by Profile Granularity in the export "
                             "block. ROI meshes always remain separate.",
                    )
                    bg_color = select_input(
                        "Background", ["white", "black"], get_user_default("background"),
                        hint="Background color for the 3D scene and exports. "
                             "The default neuron palette follows it: Category10 on "
                             "white, Set3 on black (until a palette is picked manually).",
                    )
                    brain_mesh = select_input(
                        "Brain Mesh", BRAIN_MESH_OPTIONS, get_user_default("brain_mesh"),
                        hint="'native': this dataset's template outline "
                             "and coordinates. 'BANC'/'FAFB'/'male-cns': "
                             "move the whole scene (neurons included) into "
                             "that template's coordinates with its "
                             "outline. 'none': no mesh.",
                    )
                    # BANC tube-radius normalization: appearance knobs that
                    # only surface when a BANC dataset is selected (the
                    # rescale is a no-op elsewhere).
                    banc_normalize_radius = checkbox_input(
                        "Normalized Tube Radius", True,
                        hint="Rescale each BANC neuron's radii so its median maps onto "
                             "a shared target: the release products carry inconsistent "
                             "radius calibers per neuron, which otherwise shows up as "
                             "mismatched tube thicknesses in one scene.",
                    ).set_visibility(False)
                    banc_radius_target_nm = number_input(
                        "Radius Target (nm)", 240, 10, 1000, 10,
                        hint="Median tube radius (nm) all BANC neurons are normalized "
                             "to when Normalized Tube Radius is checked.",
                    ).set_visibility(False)

                def _set_banc_controls_visible(visible: bool) -> None:
                    """Toggle the BANC controls (and their wrapper columns)."""
                    for el in (banc_normalize_radius,
                               banc_radius_target_nm):
                        try:
                            el.parent.set_visibility(visible)
                        except Exception:
                            el.set_visibility(visible)

                # Mesh appearance reads as a table, one row per half. The flat
                # flex row this replaces wrapped the cord's own trio across two
                # lines, so the two halves -- which are independent knobs, and
                # the VNC one also paints the cord a male-cns/banc page embeds
                # without showing -- did not line up as a pair. The left cell
                # names the half: for the cord that cell *is* its show/hide
                # checkbox, while the brain's is decided by the Brain Mesh menu
                # two fields above. Per-ROI colors live in the ROI panel's
                # palette editor.
                def _sync_mesh_color(color, opacity, auto):
                    enabled = not auto.value
                    color.set_enabled(enabled)
                    opacity.set_enabled(enabled)

                with ui.grid(columns=4).classes(
                        "w-full drocat-param-grid items-start").style(
                        'grid-template-columns: minmax(0, 1.15fr) '
                        'minmax(0, 1.35fr) minmax(0, 0.85fr) '
                        'minmax(0, 0.95fr)'):
                    # Every field is fluid inside its track: a fixed 11rem
                    # colour input is what pushed this block past the card's
                    # ~590 px on a narrow window.
                    with ui.column().classes("gap-0 pt-1"):
                        ui.label("Brain outline").classes("font-semibold")
                        ui.label("drawn by the Brain Mesh menu").classes(
                            "text-caption drocat-muted")
                    brain_mesh_color = ui.color_input(
                        "Brain Mesh Color", value="#74A8D6",
                    ).props("dense").classes("drocat-input w-full")
                    brain_mesh_color_opacity = number_input(
                        "Brain Opacity", 0.04, 0, 1, 0.01,
                        hint=(
                            "Outline opacity for the picked color. Defaults "
                            "to 0.04 (4%); only used when Auto is off."
                        ),
                    ).props("dense").classes("drocat-input w-full")
                    brain_mesh_color_auto = checkbox_input(
                        "Auto", True,
                        hint=(
                            "Adaptive outline color: light blue #74A8D6 at "
                            "4% opacity on any background. Uncheck to use "
                            "the picked color and opacity for the brain "
                            "outline mesh."
                        ),
                    )
                    with ui.column().classes("gap-0"):
                        vnc_mesh = checkbox_input(
                            "VNC Mesh", False,
                            hint="Show the ventral nerve cord mesh (male-cns / manc datasets).",
                        )
                        ui.label("male-cns / banc datasets").classes(
                            "text-caption drocat-muted")
                    vnc_mesh_color = ui.color_input(
                        "VNC Mesh Color", value="#74A8D6",
                    ).props("dense").classes("drocat-input w-full")
                    vnc_mesh_color_opacity = number_input(
                        "VNC Opacity", 0.04, 0, 1, 0.01,
                        hint=(
                            "Outline opacity for the VNC color. It paints the "
                            "nerve cord whether the mesh checkbox shows it or "
                            "the page only embeds it hidden."
                        ),
                    ).props("dense").classes("drocat-input w-full")
                    vnc_mesh_color_auto = checkbox_input(
                        "VNC Auto", True,
                        hint=(
                            "Adaptive outline color for the nerve cord, the "
                            "same light blue #74A8D6 at 4%. Uncheck to pin the "
                            "picked color and opacity to that half alone."
                        ),
                    )
                    brain_mesh_color.disable()
                    brain_mesh_color_opacity.disable()
                    vnc_mesh_color.disable()
                    vnc_mesh_color_opacity.disable()

                    brain_mesh_color_auto.on_value_change(
                        lambda _e: _sync_mesh_color(
                            brain_mesh_color, brain_mesh_color_opacity,
                            brain_mesh_color_auto)
                    )
                    vnc_mesh_color_auto.on_value_change(
                        lambda _e: _sync_mesh_color(
                            vnc_mesh_color, vnc_mesh_color_opacity,
                            vnc_mesh_color_auto)
                    )

            def _default_synapse_shapes_for_skeleton_mode(mode=None):
                """Return marker defaults that match the morphology renderer.

                The pre/post shape defaults to scatter markers (not solid
                meshes) so the exported HTML stays small; scatter is the
                recommended shape for pre-post sites.
                """
                selected_mode = str(
                    skeleton_mode.value if mode is None else mode
                ).strip().lower()
                if selected_mode == "line":
                    return "scatter", "scatter (circles + diamonds)"
                return "cone", "scatter (circles + diamonds)"

            default_synapse_shape, default_pre_post_shape = (
                _default_synapse_shapes_for_skeleton_mode()
            )

            # ------------------------------------------------------------------
            # Neuron Skeleton Appearance (collapsible block)
            # ------------------------------------------------------------------
            # Default neuron palette follows the background color: Category10
            # on white, Set3 on black.  Once the user picks a palette card
            # manually (on_change), the background stops switching it.
            palette_locked = {"locked": False}

            def _sync_palette_to_background():
                if palette_locked["locked"]:
                    return
                neuron_palette.set_palette(
                    "Set3" if bg_color.value == "black" else "Category10"
                )

            bg_color.on_value_change(lambda _e: _sync_palette_to_background())

            with ui.card().classes("w-full drocat-card").props('id="card-skeleton-neuron-colors"'):
                with ui.expansion(
                    "Neuron Skeleton Appearance", icon="palette",
                ).classes("w-full drocat-section-expansion"):
                    neuron_palette = palette_editor(
                        "Neuron Colors",
                        value="Category10",
                        include_auto=False,
                        on_change=lambda: palette_locked.__setitem__("locked", True),
                    )
                    neuron_palette.props('id="card-skeleton-neuron-palette"')
                    color_editor_refs["neuron"] = neuron_palette
                    neuron_alpha = number_input(
                        "Neuron Opacity", 0.2, 0, 1, 0.1,
                        hint=(
                            "Global fallback opacity for skeletons (0=invisible, 1=solid). "
                            "A color with an explicit opacity channel overrides it; colors "
                            "without opacity inherit this value."
                        ),
                    ).classes("w-48")
                    ui.label(
                        "The default follows the background (Category10 on white, "
                        "Set3 on black) until a palette is picked manually. Custom colors "
                        "may include per-layer opacity; colors without an explicit "
                        "opacity channel use Neuron Opacity."
                    ).classes("text-caption drocat-muted")

            # ------------------------------------------------------------------
            # Synapse Appearance + synapse options (collapsible block)
            # ------------------------------------------------------------------
            with ui.card().classes("w-full drocat-card").props('id="card-skeleton-synapse-colors"'):
                with ui.expansion(
                    "Synapse Appearance", icon="bubble_chart",
                ).classes("w-full drocat-section-expansion"):
                    synapse_palette = palette_editor(
                        "Synapse Colors",
                        value="Dark2",
                        include_auto=False,
                    )
                    synapse_palette.props('id="card-skeleton-synapse-palette"')
                    color_editor_refs["synapse"] = synapse_palette
                    ui.label(
                        "Colors are assigned per connection between consecutive layers "
                        "(one fewer than the number of neuron layers). Custom opacity "
                        "overrides Synapse Opacity per connection layer."
                    ).classes("text-caption drocat-muted")
                    ui.label("Synapse options").classes("drocat-mini-label")
                    # Marker-shape sub-selectors (shown for the chosen mode).
                    with ui.row().classes("w-full items-center gap-4 flex-wrap"):
                        synapse_shape = select_input(
                            "Synapse Shape",
                            [m for m in SYNAPSE_MODE_OPTIONS if m != "pre_post"],
                            default_synapse_shape,
                            hint="'cone'/'sphere'/'tetrahedron': solid markers. 'scatter': "
                                 "simple points.",
                        ).props("outlined")
                        pre_post_shape = select_input(
                            "Pre/post shape", PRE_POST_SHAPES, default_pre_post_shape,
                            hint="'solid (spheres + cones)': post/input sites as solid "
                                 "spheres, pre/output sites as cones. 'scatter (circles + "
                                 "diamonds)': post/input as circles, pre/output as diamonds "
                                 "(lighter HTML; size adjustable via the figure slider).",
                        ).props("outlined")
                    with param_grid(3):
                        synapse_size = combo_input(
                            "Synapse Size", SYNAPSE_SIZE_OPTIONS,
                            get_user_default("synapse_size") or "1",
                            hint=(
                                "Marker size (1-12). Defaults per mode: 1 px for "
                                "scatter, 3x real for mesh (sphere/cone/tetrahedron "
                                "and solid pre/post). Scatter uses it as a pixel size; "
                                "mesh modes use it as a multiplier over the real "
                                "pre→post distance. Type any integer 1-12."
                            ),
                        )
                        uniform_synapse_size = checkbox_input(
                            "Uniform Synapse Size",
                            get_user_default("uniform_synapse_size"),
                            hint=(
                                "Use the median pre→post distance for every "
                                "synapse marker so all paired markers share one size. "
                                "Pre/post site mode is always uniform and uses its "
                                "mean real-distance estimate."
                            ),
                        )
                        synapse_alpha = number_input(
                            "Synapse Opacity", 0.6, 0, 1, 0.1,
                            hint=(
                                "Global fallback opacity for synapse markers. A color with an "
                                "explicit opacity channel overrides it; colors without "
                                "opacity inherit it."
                            ),
                        )

            # Shape sub-selectors + warning follow the selected synapse mode.
            def _synapse_mode_is_scatter() -> bool:
                """True when the current synapse rendering uses px scatter markers."""
                view = synapse_view_mode.value
                if view == "pre-post sites":
                    return str(pre_post_shape.value or "").startswith("scatter")
                if view == "synapse":
                    return str(synapse_shape.value or "").strip() == "scatter"
                return False  # 'skip' -> cone mesh default

            def _sync_synapse_size_default() -> None:
                """Follow the mode's size default (scatter 1 px, mesh 3x real).

                Only resets the value while it is still at a known mode default
                (1 or 3); a value the user typed (e.g. 5) is left untouched so the
                display reflects the mode switch without clobbering custom sizes.
                """
                current = str(synapse_size.value or "").strip()
                if current in {"1", "3"}:
                    synapse_size.set_value("1" if _synapse_mode_is_scatter() else "3")

            def _sync_synapse_view_mode():
                view = synapse_view_mode.value
                synapse_shape.set_visibility(view == "synapse")
                pre_post_shape.set_visibility(view == "pre-post sites")
                pre_post_warning.set_visibility(view == "pre-post sites")
                _sync_synapse_size_default()

            def _sync_shape_defaults():
                """Apply the mode defaults whenever the skeleton mode changes."""
                synapse_default, pre_post_default = (
                    _default_synapse_shapes_for_skeleton_mode()
                )
                synapse_shape.set_value(synapse_default)
                pre_post_shape.set_value(pre_post_default)
                _sync_synapse_size_default()

            synapse_view_mode.on_value_change(
                lambda _e: (_sync_synapse_view_mode(),
                            _refresh_banc_note_from_view()))
            skeleton_mode.on_value_change(lambda _e: _sync_shape_defaults())
            synapse_shape.on_value_change(lambda _e: _sync_synapse_size_default())
            pre_post_shape.on_value_change(lambda _e: _sync_synapse_size_default())
            # The Advanced Layer Editor's columns + CSV format follow the mode.
            synapse_view_mode.on_value_change(
                lambda _e: layer_style.set_synapse_mode(synapse_view_mode.value)
            )
            _sync_synapse_view_mode()
            _sync_synapse_size_default()
            layer_style.set_synapse_mode(synapse_view_mode.value)

            # ------------------------------------------------------------------
            # Brain Region ROIs + ROI Colors (collapsible block)
            # ------------------------------------------------------------------
            with ui.card().classes("w-full drocat-card").props('id="card-skeleton-roi-colors"'):
                with ui.expansion(
                    "Brain Region ROIs (independent)", icon="view_in_ar",
                ).classes("w-full drocat-section-expansion"):
                    roi_select = multi_select_input(
                        "Mesh ROIs",
                        COMMON_ROIS,
                        default=[],
                        hint=(
                            "Select brain regions to show as meshes. No ROI meshes are "
                            "selected by default. Type any ROI name or regex (e.g. "
                            "ME.*, all, primary) and press Enter to add it."
                        ),
                    ).props("outlined").props('new-value-mode="add-unique"')
                    roi_detail_hint = ui.label(
                        "Primary ROI suggestions load the selected dataset's metadata."
                    ).classes("text-caption drocat-muted")
                    with ui.row().classes("w-full items-center gap-6"):
                        include_lr = checkbox_input(
                            "Include L/R variants",
                            False,
                            hint=(
                                "Keep explicit (L)/(R) entries in the primary list. "
                                "When off, bilateral pairs are shown once without "
                                "the suffix."
                            ),
                        )
                        include_subprimary = checkbox_input(
                            "Include sub-primary ROIs",
                            False,
                            hint=(
                                "Append non-primary ROI names from the local "
                                "available-ROI inventory to the primary list."
                            ),
                        )
                    with param_grid(1):
                        mesh_alpha = number_input(
                            "ROI Mesh Opacity", 0.1, 0, 1, 0.05,
                            hint=(
                                "Global fallback opacity for ROI meshes. A color with an "
                                "explicit opacity channel overrides it; colors without "
                                "opacity inherit it."
                            ),
                        )
                    roi_palette = palette_editor(
                        "ROI Colors",
                        value="Cool",
                        include_auto=True,
                    )
                    roi_palette.props('id="card-skeleton-roi-palette"')
                    ui.label(
                        "Colors are assigned in the displayed order; every resolved ROI mesh "
                        "has its own legend entry. Use Custom colors for per-ROI opacity "
                        "overrides. The brain outline mesh color lives in General Appearance."
                    ).classes("text-caption drocat-muted")

            # ------------------------------------------------------------------
            # Export Video / GIF + Individual Profiles (collapsible blocks;
            # the Advanced Settings expansion follows below them)
            # ------------------------------------------------------------------
            with ui.card().classes("w-full drocat-card").props('id="card-skeleton-export-video"'):
                with ui.expansion(
                    "Export Video / GIF", icon="videocam",
                ).classes("w-full drocat-section-expansion"):
                    with ui.row().classes("gap-4"):
                        export_video = checkbox_input(
                            "Export Video", False,
                            hint="Render a rotating video of the 3D scene (needs Chrome/WebDriver).",
                        )
                        rotate = select_input(
                            "Rotate", ["horizontal", "vertical"], "horizontal",
                        )
                        export_gif = checkbox_input(
                            "Also Export GIF", True,
                            hint="Convert the video to a small GIF as well.",
                        )
                    with param_grid(3):
                        fps = number_input("FPS", 30, 5, 60, 5)
                        degree_per_frame = number_input("Degrees / Frame", 1.0, 0.1, 5.0, 0.1)
                        gif_scale = number_input("GIF Scale", 0.2, 0.05, 1.0, 0.05)

                with ui.expansion(
                    "Individual Profiles (PDF / PPTX)", icon="photo_library",
                ).classes("w-full drocat-section-expansion"):
                    with ui.row().classes("gap-4"):
                        export_individual_profiles = checkbox_input(
                            "Export Individual Profiles", False,
                            hint="After rendering, generate a PDF/PPTX with one "
                                 "profile plot per group (see Profile Granularity).",
                        )
                        summary_format = multi_select_input(
                            "Summary Format", ["pdf", "pptx"], ["pdf"],
                            hint="Output formats for the individual-profile summary.",
                        )
                    with param_grid(3):
                        profile_cols = number_input("Images Per Page (cols)", 3, 1, 6)
                        profile_rows = number_input("Images Per Page (rows)", 2, 1, 6)
                        profile_views = multi_select_input(
                            "Profile Views", PROFILE_VIEWS, ["front"],
                            hint="Camera views included in each individual profile.",
                        )
                        profile_granularity = select_input(
                            "Profile Granularity", PROFILE_GRANULARITY_CHOICES,
                            get_user_default("profile_granularity"),
                            hint="How far the tree is walked to form one profile group. "
                                 "'Per legend entry' keeps the historical behaviour; the "
                                 "other levels group independently of the Neuron Legend "
                                 "Mode. 'Per bodyId' produces one profile per neuron "
                                 "leaf, which can be many files: past 300 renders "
                                 "(groups x views) only the first ones are exported.",
                        )
                    ui.label(
                        "Each individual profile follows the Neuron Legend Mode only "
                        "while Profile Granularity is 'Per legend entry': then "
                        "'single' = one profile per neuron, 'type' = one profile per type "
                        "(all layers combined), 'layer' = one profile per layer / custom group. "
                        "The other granularities ('Per layer', 'Per neuron type', "
                        "'Per bodyId') group by that tree level whatever the legend mode is."
                    ).classes("text-caption drocat-muted")

            # ------------------------------------------------------------------
            # Advanced Settings (kept at the bottom, below the export blocks)
            # ------------------------------------------------------------------
            with ui.card().classes("w-full drocat-card").props('id="card-skeleton-advanced"'):
                with ui.expansion(
                    "Advanced Settings", icon="settings_suggest",
                ).classes("w-full drocat-section-expansion"):
                    ui.label("Data & Rendering").classes("drocat-mini-label")
                    with ui.row().classes("gap-4"):
                        cache_neurons = checkbox_input(
                            "Cache Neurons", get_user_default("cache_neurons"),
                            hint="Cache fetched skeletons as portable .swc.zst "
                                 "files in the shared cache for faster repeat "
                                 "renders.",
                        )
                        cache_default_state = {"user_changed": False, "updating": False}

                        def on_cache_neurons_change(_event):
                            if not cache_default_state["updating"]:
                                cache_default_state["user_changed"] = True

                        cache_neurons.on_value_change(on_cache_neurons_change)
                        cache_synapses = checkbox_input(
                            "Cache Synapses", get_user_default("cache_synapses"),
                            hint="Cache fetched synapse data locally.",
                        )
                    with ui.row().classes("gap-4"):
                        simplification_method = select_input(
                            "Simplification Method",
                            ["fast", "fine", "artistic"],
                            get_user_default("simplification_method"),
                            hint=(
                                "NeuPrint tube rendering: 'fast' (default) reads "
                                "the shared raw level-0 skeleton source, then "
                                "applies direct mesh decimation in memory plus "
                                "the FAFB fast node-reduction stage; 'fine' "
                                "smooths/resamples with the accelerated FAFB "
                                "radius profile; 'artistic' uses vertex-cluster "
                                "mesh decimation. All methods use batched "
                                "parallel online fetching and are available for "
                                "NeuPrint and FAFB/BANC tube renders; line "
                                "mode bypasses the method."
                            ),
                        )
                        default_simplification = checkbox_input(
                            "Use Default Mesh Simplification", True,
                            hint="Use the method default: fast removes 0.90 of faces; "
                                 "fine/artistic remove 0.95 for NeuPrint and "
                                 "FAFB. For BANC the value applies to "
                                 "full-resolution sources; L2 tubes are already "
                                 "cache-level and are never decimated. Uncheck to "
                                 "set the value below.",
                        )
                        mesh_simplification = number_input(
                            "Mesh Simplification (faces removed)",
                            default_skeleton_tab_simplification(
                                dataset.value, simplification_method.value,
                            ),
                            0.0, 0.99, 0.05,
                            hint="Fraction of tube-mesh faces REMOVED for rendering: "
                                 "0.95 = keep 5%. Higher = faster/coarser, lower = "
                                 "more detailed but slower. For BANC the value "
                                 "applies to full-resolution sources; L2 tubes are "
                                 "already cache-level and are never decimated.",
                        )
                        mesh_simplification.set_enabled(False)

                    ui.label("Export").classes("drocat-mini-label")
                    with param_grid(3):
                        export_method = select_input(
                            "Export Method", ["webdriver", "kaleido"], "webdriver",
                            hint="'webdriver': fast, needs Chrome 109+. 'kaleido': slower but stable fallback.",
                        )
                        export_scale = number_input(
                            "Export Scale", 3, 1, 5,
                            hint="Resolution multiplier for PNG exports (higher = sharper).",
                        )
                    with ui.row().classes("gap-4"):
                        show_fig = checkbox_input(
                            "Show Figure",
                            # The Skeleton tab is the one place that still opens
                            # the figure by default; the global default (and all
                            # analysis tabs) start unchecked unless the user saved
                            # an explicit override here in Settings.
                            get_user_default("show_fig_skeleton")
                            if has_user_default("show_fig_skeleton") else True,
                            hint="Open the 3D HTML visualization after rendering.",
                        )
                        export_views = checkbox_input(
                            "Export Views", get_user_default("export_views"),
                            hint="Export PNG screenshots from 6 angles.",
                        )
                        # HTML-viewer knob (the viewer itself is always the
                        # interactive HTML the figure opens as): pinned axes
                        # keep the scene framed while traces are toggled.
                        freeze_view = checkbox_input(
                            "Freeze 3D view when toggling the legend",
                            get_user_default("freeze_view"),
                            hint="Pin the exported HTML viewer's scene axes so "
                                 "showing or hiding traces in the legend cannot "
                                 "rescale the scene. A Freeze/Fit button in the "
                                 "page's top-left corner (also the 'F' key) "
                                 "restores Plotly's autoscaling.",
                        )

        def _sync_simplification_controls():
            is_line = skeleton_mode.value == "line"
            pipeline = str(simplification_method.value or "fine").strip().lower()
            if default_simplification.value:
                mesh_simplification.set_value(
                    default_skeleton_tab_simplification(
                        dataset.value, pipeline,
                    )
                )
            mesh_simplification.set_enabled(
                not is_line and not default_simplification.value
            )
            default_simplification.set_enabled(not is_line)
            simplification_method.set_enabled(
                not is_line
            )
            if (
                not cache_default_state["user_changed"]
                and not has_user_default("cache_neurons")
            ):
                # Fetched skeletons persist as portable .swc.zst files in
                # the shared cache, so caching is the default source policy
                # for every dataset and render pipeline.
                default_cache = True
                if cache_neurons.value != default_cache:
                    cache_default_state["updating"] = True
                    try:
                        cache_neurons.set_value(default_cache)
                    finally:
                        cache_default_state["updating"] = False

        default_simplification.on_value_change(
            lambda _e: _sync_simplification_controls()
        )
        skeleton_mode.on_value_change(lambda _e: _sync_simplification_controls())
        simplification_method.on_value_change(
            lambda _e: _sync_simplification_controls()
        )
        dataset.on_value_change(lambda _e: _sync_simplification_controls())
        _sync_simplification_controls()

        def _sync_roi_options():
            catalog = load_roi_catalog(dataset.value)
            options = roi_options_from_catalog(
                catalog,
                include_lr=bool(include_lr.value),
                include_subprimary=bool(include_subprimary.value),
                fallback=COMMON_ROIS,
            )
            primary_count = len(catalog["primary"])
            if catalog["primary"]:
                suffix = ""
                if include_subprimary.value:
                    suffix = " plus sub-primary entries"
                roi_detail_hint.set_text(
                    f"Primary list: {primary_count} metadata ROIs{suffix} "
                    f"for {dataset.value}."
                )
            else:
                roi_detail_hint.set_text(
                    "No local metadata primary list was found; using the "
                    "common fallback list."
                )
            # QSelect only renders chips whose values exist in its options,
            # so keep the currently selected ROIs in the new option list.
            for value in roi_select.value or []:
                if value not in options:
                    options.append(value)
            roi_select.set_options(options)

        _banc_synapse_note = ui.label(
            "⚠️ BANC synapse rendering: first use downloads the per-synapse "
            "table (~3.9 GB, one-time, cached). The release provides only "
            "pre-synaptic site coordinates — markers are drawn at pre-sites."
        ).classes("text-caption text-amber-8").set_visibility(False)

        def _sync_banc_synapse_default():
            # Keep the BANC synapse default (skip) in sync with the dataset.
            if is_banc_dataset(dataset.value):
                default_view = banc_synapse_view_default(dataset.value)
                if synapse_view_mode.value != default_view:
                    synapse_view_mode.set_value(default_view)
                _banc_synapse_note.set_visibility(
                    synapse_view_mode.value != "skip")
            else:
                _banc_synapse_note.set_visibility(False)

        def _refresh_banc_note_from_view():
            _banc_synapse_note.set_visibility(
                is_banc_dataset(dataset.value)
                and synapse_view_mode.value != "skip")

        def _sync_banc_controls_visibility():
            visible = is_banc_dataset(dataset.value)
            _set_banc_controls_visible(visible)
            banc_radius_target_nm.set_enabled(
                visible and banc_normalize_radius.value)

        include_lr.on_value_change(lambda _e: _sync_roi_options())
        include_subprimary.on_value_change(lambda _e: _sync_roi_options())
        dataset.on_value_change(lambda _e: _sync_roi_options())
        dataset.on_value_change(lambda _e: _sync_banc_controls_visibility())
        dataset.on_value_change(lambda _e: (
            _sync_banc_synapse_default(),
            _refresh_banc_note_from_view(),
        ))
        banc_normalize_radius.on_value_change(
            lambda _e: banc_radius_target_nm.set_enabled(
                is_banc_dataset(dataset.value)
                and banc_normalize_radius.value))
        _sync_roi_options()
        _sync_banc_controls_visibility()
        _sync_banc_synapse_default()

    with results_col:
        skeleton_output.create(run_label="Generate 3D Skeleton", run_icon="view_in_ar")

    # The re-export section gets its own workspace band below the main one so
    # its output panel sits beside the control card it belongs to (the pages
    # it offers are exactly the ones this tab produced), not at the top of the
    # results column next to the render panel.
    with form_col._drocat_page:
        with ui.row().classes("w-full drocat-workspace items-start").style(
            "margin-top: 24px"
        ):
            reexport_form_col = ui.column().classes("drocat-form gap-3")
            reexport_results_col = ui.column().classes("drocat-results gap-3")
    create_skeleton_reexport(reexport_form_col, reexport_results_col)

    async def run_panel(
        output_panel,
        runner,
        tool_name,
        constructor_params,
        output_dir_for_scan,
        method_params=None,
    ):
        output_panel.clear()
        output_panel.set_running(True)
        result = await output_panel.run(
            runner,
            tool_name,
            constructor_params,
            "plot",
            method_params=method_params,
            output_dir=output_dir_for_scan,
        )
        output_panel.set_running(False)
        output_panel.set_status(
            "Completed" if result["returncode"] == 0 else "Failed",
            "green" if result["returncode"] == 0 else "red",
        )
        output_panel.show_files(
            result["files"],
            result.get("output_folder") or output_dir_for_scan,
        )
        return result

    async def run_skeleton():
        editor_mode = layer_editor_mode["value"]
        rois = roi_select.value or []

        def _distinct_layer_count(rows) -> int:
            complete = layer_style_store.complete_rows(rows)
            layers = sorted(
                {str(r["layer"]).strip() for r in complete if str(r["layer"]).strip()},
                key=lambda v: float(v),
            )
            return len(layers)

        if editor_mode == "Advanced":
            # The Advanced Layer Editor supplies the layer list + per-neuron
            # colors as a layer_map CSV; the backend parses it into layers and
            # color overrides (so the tree-derived lists are unused here).
            layer_map_csv = layer_style.runnable_csv_path()
            if not layer_map_csv:
                ui.notify(
                    "Add at least one row with a layer and a neuron in the "
                    "Advanced Layer Editor",
                    type="warning",
                )
                return
            neuron_layers = []
            custom_names = []
            raw_neurons = [
                str(row["neuron"]).strip()
                for row in layer_style_store.complete_rows(layer_style.rows)
            ]
            # Distinct layer count drives the fallback palette sizing.
            n_layer_colors = _distinct_layer_count(layer_style.rows)
        elif editor_mode == "File upload":
            # The uploaded CSV supplies the layer list + per-neuron colors
            # directly (in-file colors; no palette editors are used).
            layer_map_csv = file_upload_path.get("path")
            if not layer_map_csv:
                ui.notify("Upload a custom-layer CSV first", type="warning")
                return
            neuron_layers = []
            custom_names = []
            rows = layer_style_store.load_rows_from_csv_text(
                Path(layer_map_csv).read_text(encoding="utf-8")
            )
            raw_neurons = [
                str(r["neuron"]).strip()
                for r in layer_style_store.complete_rows(rows)
            ]
            n_layer_colors = _distinct_layer_count(rows)
        else:
            # The layer tree maps 1:1 to the backend's nested-list model:
            # neuron_layers[i] = layer i's neurons, custom_layer_names partial.
            layer_map_csv = None
            neuron_layers = layer_tree.get_neuron_layers()
            if not neuron_layers:
                ui.notify("Add at least one neuron to a layer", type="warning")
                return
            # Raw chips (pre-pattern) for the query history: the filter-mode
            # conversion below rewrites them into regex patterns, which are not
            # useful history entries.
            raw_neurons = _flatten_neuron_layers(neuron_layers)
            # Same search semantics as pathfinding: the match mode converts
            # every neuron into the regex pattern resolved by statvis.getNeurons.
            mode = filter_mode.value
            if mode and mode != "exact":
                converted = []
                for layer in neuron_layers:
                    as_list = layer if isinstance(layer, list) else [layer]
                    filtered = apply_filter_mode(as_list, mode)
                    converted.append(
                        filtered if isinstance(layer, list) else filtered[0]
                    )
                neuron_layers = converted
            custom_names = layer_tree.get_custom_layer_names()
            n_layer_colors = len(_flatten_neuron_layers(neuron_layers))

        # Assign the exact displayed palette order (including custom reordering).
        # In the advanced editor the count is the distinct layer count (fallback
        # layer colors; per-neuron CSV colors override them); otherwise it is the
        # flattened neuron count used before. Neuron colors in the backend are
        # applied per layer, so an over-long list is harmless.
        neuron_colors = _palette_colors_for_count(neuron_palette, max(1, n_layer_colors))
        # One color per connection between consecutive layers (n_layers - 1).
        synapse_colors = _palette_colors_for_count(
            synapse_palette, max(0, n_layer_colors - 1)
        )

        # Auto is a one-color gray palette; custom mode must not be gated by
        # the last selected preset name.
        mesh_color = _palette_colors_for_count(
            roi_palette, len(rois)
        ) if rois else (100, 100, 100)

        # An empty custom palette renders with the backend's default palette
        # (which logs a warning internally); surface it in the UI too so the
        # substitution is not silent. The ROI palette is only consulted when
        # at least one ROI mesh is selected.
        palettes = [
            (neuron_palette, "Neuron Colors"),
            (synapse_palette, "Synapse Colors"),
        ]
        if rois:
            palettes.append((roi_palette, "ROI Colors"))
        notify_empty_custom_palettes(*palettes)

        # Combo box free text; an emptied or unparseable value falls back to the
        # default (1 px). Only a bare integer 1-12 is accepted.
        synapse_size_value = str(synapse_size.value or "").strip()
        if not synapse_size_value:
            synapse_size_value = "1"
        elif not is_valid_synapse_size(synapse_size_value):
            ui.notify(
                f"Invalid Synapse Size '{synapse_size_value}' — using '1' "
                "pixels instead.",
                type="warning",
            )
            synapse_size_value = "1"

        # Map the top-level synapse view mode to backend fields.
        view = synapse_view_mode.value
        if view == "skip":
            skip_synapses = True
            synapse_mode_value = "cone"
            pre_post_scatter_on = False
        elif view == "pre-post sites":
            skip_synapses = False
            synapse_mode_value = "pre_post"
            pre_post_scatter_on = pre_post_shape.value.startswith("scatter")
        else:  # 'synapse'
            skip_synapses = False
            # Keep the renderer contract explicit: the visible shape selector
            # is the source of truth for paired-synapse geometry, and only the
            # four backend-supported values may reach the generated script.
            synapse_mode_value = str(synapse_shape.value or "cone").strip().lower()
            if synapse_mode_value not in {"cone", "scatter", "sphere", "tetrahedron"}:
                synapse_mode_value = "cone"
            pre_post_scatter_on = False

        # BANC + synapses: explicit warning before the run (the per-synapse
        # table downloads once; only pre-site markers are drawn).
        warning = banc_synapse_warning(dataset.value, view)
        if warning:
            push_banner(warning, icon="warning")

        constructor_params = {
            "dataset": dataset.value,
            "banc_normalize_radius": banc_normalize_radius.value,
            "banc_radius_target_nm": float(banc_radius_target_nm.value),
            "neuron_layers": neuron_layers,
            "search_columns": search_columns.value,
            "hemisphere": hemisphere.value,
            "custom_layer_names": custom_names,
            "layer_map_csv": layer_map_csv,
            "output_dir": output_dir.value,
            "output_format": get_user_default("output_format"),
            "skeleton_mode": skeleton_mode.value,
            "brain_mesh": brain_mesh.value,
            "vnc_mesh": vnc_mesh.value,
            "legend_mode": legend_mode.value,
            "freeze_view": bool(freeze_view.value),
            "neuron_alpha": float(neuron_alpha.value),
            "neuron_colors": neuron_colors,
            "synapse_colors": synapse_colors,
            "background_color": bg_color.value,
            "skip_synapse": skip_synapses,
            "min_synapse_num": int(synapse_threshold.value),
            "synapse_size": synapse_size_value,
            "uniform_synapse_size": uniform_synapse_size.value,
            "synapse_alpha": float(synapse_alpha.value),
            "synapse_mode": synapse_mode_value,
            "pre_post_scatter": pre_post_scatter_on,
            "mesh_roi": rois,
            "mesh_color": mesh_color,
            "mesh_alpha": float(mesh_alpha.value),
            "cache_neurons": cache_neurons.value,
            "cache_synapses": cache_synapses.value,
            "show_soma": True,
            "export_method": export_method.value,
            "export_scale": int(export_scale.value),
            "export_views": export_views.value,
            "show_fig": show_fig.value,
            "brain_mesh_color": mesh_color_backend_value(
                brain_mesh_color_auto, brain_mesh_color,
                brain_mesh_color_opacity),
            "vnc_mesh_color": mesh_color_backend_value(
                vnc_mesh_color_auto, vnc_mesh_color, vnc_mesh_color_opacity),
            "neuprint_skeleton_pipeline": simplification_method.value,
            "skeleton_mesh_simplification": (
                default_skeleton_tab_simplification(
                    dataset.value, simplification_method.value,
                ) if default_simplification.value
                else float(mesh_simplification.value)
            ),
        }
        method_params = {
            "export_individual_profiles": export_individual_profiles.value,
            "pdf_images_per_page": (
                int(profile_cols.value),
                int(profile_rows.value),
            ),
            "views": profile_views.value or ["front"],
            "summary_format": summary_format.value or ["pdf"],
            "granularity": profile_granularity.value,
            "export_video": export_video.value,
            "fps": int(fps.value),
            "degree_per_frame": float(degree_per_frame.value),
            "rotate": rotate.value,
            "export_gif": export_gif.value,
            "gif_scale": float(gif_scale.value),
        }
        transient_layer_csv = (
            layer_map_csv if layer_map_csv == layer_style.transient_csv_path else None
        )
        try:
            result = await run_panel(
                skeleton_output,
                skeleton_runner,
                "plot3d_skeleton",
                constructor_params,
                output_dir.value,
                method_params,
            )
            # A completed render means the layer neurons resolved in the dataset;
            # keep the raw chips in the query history like the pathfinding tabs.
            if result["returncode"] == 0:
                from ..history_store import record as _record_history
                _record_history(
                    [str(n) for n in dict.fromkeys(raw_neurons)],
                    datasets=[dataset.value] if dataset.value else [],
                )
                # A completed visualization consumes the draft, so clear its dirty
                # flag (a crash-recovery prompt should not re-offer a used draft).
                if layer_style.current_name:
                    layer_style_store.mark_exported(layer_style.current_name)
        finally:
            if transient_layer_csv:
                layer_style.cleanup_transient_csv()

    skeleton_output.run_button.on_click(run_skeleton)
    skeleton_output.cancel_button.on_click(skeleton_runner.cancel)
    return recovery_offer


def create_net_viz_tab():
    """Create the standalone PlotPath pathway-graph tab (Net-Viz)."""
    net_viz_runner = ScriptRunner()
    net_viz_output = OutputPanel(
        "Net-Viz Output", state_key="visualization_net_viz"
    )
    path_file_path = {"path": None}

    form_col, results_col = tool_page(
        "Net-Viz",
        "Interactive pathway graphs and editable empty drawing canvases.",
        icon="account_tree",
        doc="network.md",
    )

    with form_col:
        ui.label(
            "Load a path result, build a custom edge list, or create an empty "
            "HTML canvas for direct interactive drawing."
        ).classes("text-caption drocat-muted")

        with ui.card().classes("w-full drocat-card").props('id="card-net-viz-output"'):
            section_header("Output Directory", "folder")
            path_output_dir = dir_input(
                label="Path Output Directory",
                scope="visualization_path",
            )

        with ui.card().classes("w-full drocat-card").props('id="card-net-viz-source"'):
            section_header("Net-Viz Source", "source")
            # The three canvas sources are mutually exclusive and are chosen
            # with segmented buttons (mirroring the Skeleton tab).
            net_viz_mode = {"value": "Edge list editor"}
            with ui.row().classes(
                "w-full items-center justify-between gap-4 px-2 flex-nowrap"
            ):
                mode_buttons = {}
                for _mode in NET_VIZ_SOURCE_MODES:
                    _btn = ui.button(_mode).props("outline no-caps").classes("w-1/3")
                    _btn.style(
                        "min-height: 3rem; font-size: 1.05rem; font-weight: 700;"
                    )
                    mode_buttons[_mode] = _btn
            ui.label(
                "Build an edge list, create an empty HTML canvas, or load a "
                "path result (CSV or Excel with path_type / path_bodyId sheets)."
            ).classes("text-caption drocat-muted")

            async def handle_path_upload(e):
                temporary_path = None
                previous_path = path_file_path.get("path")
                try:
                    filename, data = await read_upload_event(e)
                    suffix = Path(filename).suffix.lower()
                    if suffix not in {".csv", ".xlsx", ".xls"}:
                        raise ValueError("Use a CSV, XLSX, or XLS path file")
                    with tempfile.NamedTemporaryFile(
                        suffix=suffix, prefix="drocat_paths_", delete=False
                    ) as temporary:
                        temporary.write(data)
                        temporary_path = temporary.name

                    path_file_path["path"] = temporary_path
                    if previous_path and previous_path != temporary_path:
                        Path(previous_path).unlink(missing_ok=True)
                    path_upload_label.text = (
                        f"Loaded: {filename} ({len(data) / 1024:.1f} KB)"
                    )
                    path_upload_label.classes(replace="text-caption drocat-ok")
                except Exception as ex:
                    if temporary_path:
                        Path(temporary_path).unlink(missing_ok=True)
                    if previous_path:
                        Path(previous_path).unlink(missing_ok=True)
                    path_file_path["path"] = None
                    path_upload_label.text = f"Error: {ex}"
                    path_upload_label.classes(replace="text-caption drocat-err")
                path_upload_menu.close()

            editor_mode_panel = ui.column().classes("w-full").props(
                'id="card-net-viz-editor-panel"'
            )
            empty_canvas_panel = ui.column().classes("w-full").props(
                'id="card-net-viz-empty-canvas"'
            ).set_visibility(False)
            path_input_panel = ui.column().classes("w-full").props(
                'id="net-viz-path-input"'
            ).set_visibility(False)

            # Edge-list CSV exports are browser downloads. The configured
            # output directory is reserved for artifacts generated by the
            # Net-Viz run.
            with editor_mode_panel:
                editor = edge_list_editor(
                    on_expand=lambda: _set_net_viz_mode("Edge list editor"),
                )

            with empty_canvas_panel:
                with ui.card().classes("w-full drocat-card").props(
                    'id="card-net-viz-empty-canvas-card"'
                ):
                    section_header("Empty drawing canvas", "open_in_new")
                    ui.label(
                        "Create an empty interactive HTML canvas for direct "
                        "drawing; no path data is needed."
                    ).classes("text-caption drocat-muted")
                    empty_canvas_button = ui.button(
                        "Create Empty Canvas",
                        icon="open_in_new",
                    ).props("color=secondary outline").classes("drocat-empty-canvas-btn")

            with path_input_panel:
                with ui.card().classes("w-full drocat-card").props(
                    'id="card-net-viz-path-upload-card"'
                ):
                    section_header("Path file (file upload)", "upload_file")
                    ui.label(
                        "Upload a Complete Paths result: CSV / XLSX / XLS with "
                        "path_type or path_bodyId sheets."
                    ).classes("text-caption drocat-muted")
                    with ui.row().classes("w-full items-center gap-2"):
                        with ui.button(icon="upload_file").props(
                            "flat dense round"
                        ).classes(
                            "drocat-upload-trigger"
                        ).tooltip("Upload a *_allpaths_info file (CSV/Excel)"):
                            with ui.menu() as path_upload_menu:
                                ui.label("Load path data from Complete Paths").classes(
                                    "text-caption drocat-muted px-3 pt-2"
                                )
                                ui.label(
                                    "CSV / XLSX / XLS with path_type or path_bodyId sheets"
                                ).classes("text-caption drocat-muted px-3 pb-1")
                                ui.upload(
                                    label="Choose paths file",
                                    on_upload=handle_path_upload,
                                    auto_upload=True,
                                ).props('accept=".csv,.xlsx,.xls" flat dense').classes(
                                    "w-72"
                                )
                                ui.link(
                                    "File format instructions",
                                    "docs/ui_guides/network.html#input-files",
                                ).classes("drocat-doc-link px-3 pb-2")
                        path_upload_label = ui.label("No path file selected.").classes(
                            "text-caption drocat-muted drocat-truncate"
                        )

        def _sync_net_viz_mode():
            active = net_viz_mode["value"]
            editor_mode_panel.set_visibility(active == "Edge list editor")
            empty_canvas_panel.set_visibility(active == "Empty canvas")
            path_input_panel.set_visibility(active == "File upload")
            for _mode, _btn in mode_buttons.items():
                _btn.props("color=primary" if _mode == active else "color=grey-7")
            # Show the edge table directly (do not leave it collapsed behind
            # its expansion header) when the editor mode is active. The guard
            # avoids re-triggering the expansion's on_expand callback.
            if active == "Edge list editor" and editor.expansion is not None:
                if not editor.expansion.value:
                    editor.expansion.set_value(True)
            if active != "File upload":
                path_upload_menu.close()

        def _set_net_viz_mode(value: str):
            net_viz_mode["value"] = value
            _sync_net_viz_mode()

        # The expansion callback handles the reverse direction: opening the
        # editor immediately selects it as the run source.
        for _mode, _btn in mode_buttons.items():
            _btn.on_click(lambda _e, m=_mode: _set_net_viz_mode(m))
        _sync_net_viz_mode()

        with ui.card().classes("w-full drocat-card").props('id="card-net-viz-rendering"'):
            section_header("Rendering Options", "palette")
            path_layout = select_input(
                "Network Layout",
                NETWORK_LAYOUTS + ["hierarchical"],
                "hierarchical",
                hint="Layout algorithm for the HTML network visualization.",
            )
            color_preset = palette_picker(
                "Color Scheme",
                value="Cool",
                catalog=PATH_SCHEMES,
            )
            with ui.row().classes("gap-4"):
                show_path_fig = checkbox_input(
                    "Open in Browser",
                    True,
                    hint="Open the interactive HTML network after rendering.",
                )
            ui.label(
                "Path mode exports HTML and connection tables; empty mode exports an HTML canvas."
            ).classes("text-caption drocat-muted")
            ui.label(
                "Edge list files with a 'color' column keep their per-edge colors; "
                "the Color Scheme applies to the remaining edges and node roles."
            ).classes("text-caption drocat-muted")

    with results_col:
        net_viz_output.create(run_label="Generate Network", run_icon="account_tree")

    async def run_panel(constructor_params):
        net_viz_output.clear()
        net_viz_output.set_running(True)
        result = await net_viz_output.run(
            net_viz_runner,
            "plot_path",
            constructor_params,
            "plot",
            output_dir=path_file_path.get("output_folder") or path_output_dir.value,
        )
        net_viz_output.set_running(False)
        net_viz_output.set_status(
            "Completed" if result["returncode"] == 0 else "Failed",
            "green" if result["returncode"] == 0 else "red",
        )
        net_viz_output.show_files(
            result["files"],
            result.get("output_folder") or path_file_path.get("output_folder") or path_output_dir.value,
        )

    def make_plotpath_folder(empty_canvas=False):
        """Create the per-run output subfolder (plot-network_{name}_{timestamp}).

        Every run gets its own timestamped folder inside the user-chosen
        output directory, matching the naming of the other tools.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if empty_canvas:
            name = "empty_network"
        else:
            name = Path(path_file_path["path"]).stem or "paths"
            name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            if len(name) > 60:
                name = name[:60]
        run_folder = Path(path_output_dir.value) / f"plot-network_{name}_{timestamp}"
        run_folder.mkdir(parents=True, exist_ok=True)
        path_file_path["output_folder"] = str(run_folder)
        return str(run_folder)

    async def execute_net_viz(empty_canvas=False):
        editor_mode = (
            (not empty_canvas)
            and net_viz_mode["value"] == "Edge list editor"
        )
        transient_editor_path = None
        if editor_mode:
            # Flush the auto-save when named; otherwise use a temporary CSV.
            # A draft name is optional for running the edited edge list.
            csv_path = editor.runnable_path_file()
            if not csv_path:
                ui.notify(
                    "Add at least one complete edge (source, target, weight) first",
                    type="warning",
                )
                return
            path_file_path["path"] = csv_path
            transient_editor_path = csv_path if editor.transient_csv_path == csv_path else None
        elif not empty_canvas and not path_file_path["path"]:
            ui.notify(
                "Please upload a path file first (Complete Paths output)",
                type="warning",
            )
            return

        try:
            colors = COLOR_PRESETS.get(color_preset.get_value(), COLOR_PRESETS["Cool"])
            constructor_params = {
                "path_file": None if empty_canvas else path_file_path["path"],
                "output_folder": make_plotpath_folder(empty_canvas),
                "source_color": colors["source"],
                "intermediate_color": colors["intermediate"],
                "target_color": colors["target"],
                "link_color": colors["link"],
                "network_layout": path_layout.value,
                "showfig": True if empty_canvas else show_path_fig.value,
                "generate_empty_network": empty_canvas,
            }
            await run_panel(constructor_params)
        finally:
            if transient_editor_path:
                editor.cleanup_transient_csv()
                if path_file_path.get("path") == transient_editor_path:
                    path_file_path["path"] = None

    async def run_net_viz():
        # In Empty canvas mode the main run button generates the canvas; the
        # panel's dedicated button runs the same code path.
        if net_viz_mode["value"] == "Empty canvas":
            await execute_net_viz(empty_canvas=True)
        else:
            await execute_net_viz(empty_canvas=False)


    async def create_empty_canvas():
        await execute_net_viz(empty_canvas=True)

    net_viz_output.run_button.on_click(run_net_viz)
    net_viz_output.cancel_button.on_click(net_viz_runner.cancel)
    empty_canvas_button.on_click(create_empty_canvas)


def create_visualization_tab():
    """Backward-compatible combined view for callers outside the main app."""
    create_skeleton_tab()
    create_net_viz_tab()
