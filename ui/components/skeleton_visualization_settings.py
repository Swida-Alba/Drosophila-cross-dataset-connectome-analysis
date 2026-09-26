"""Shared controls for background skeleton visualizations.

The standalone 3D Skeleton tab exposes the complete editor because it is a
dedicated visualization workflow.  The analysis tabs need the same rendering
choices without duplicating a large collection of inputs in every tab, so
they use :func:`skeleton_visualization_settings` in a collapsed expansion.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

from nicegui import ui

from ..config import (
    BRAIN_MESH_OPTIONS,
    LEGEND_MODES,
    SKELETON_MODES,
    SYNAPSE_SIZE_OPTIONS,
    get_user_default,
    has_user_default,
    is_valid_synapse_size,
)
from .common import (
    checkbox_input,
    combo_input,
    multi_select_input,
    number_input,
    param_grid,
    select_input,
)
from .palette_picker import (
    color_swatch_picker,
    category_color_editor,
    palette_editor,
)
from visualization_options import default_analysis_skeleton_mesh_simplification


def _default_analysis_simplification(
        dataset_value: Any, pipeline: str = "fine") -> float:
    """Return the visible default for analysis-generated visualizations."""
    return default_analysis_skeleton_mesh_simplification(
        str(dataset_value or ""), pipeline,
    )


#: The panel's three captioned sections, by field name. A section caption is
#: hidden with its whole group, so a caller that scopes the panel down (the
#: type-mapping validation scenes) never sees an empty heading.
APPEARANCE_FIELDS = ("skeleton_mode", "legend_mode", "background_color",
                     "brain_mesh", "vnc_mesh")
SYNAPSE_FIELDS = ("skip_synapse", "min_synapse_num", "synapse_mode",
                  "synapse_size", "uniform_synapse_size", "synapse_alpha",
                  "mesh_alpha", "synapse_colors", "mesh_roi", "roi_colors")
EXPORT_FIELDS = ("cache_neurons", "cache_synapses", "use_default_simplification",
                 "neuprint_skeleton_pipeline", "skeleton_mesh_simplification",
                 "export_method", "export_scale", "show_fig", "export_views")



@dataclass
class SkeletonVisualizationSettings:
    """Live NiceGUI controls plus a normalized value snapshot method."""

    fields: Dict[str, Any]
    panel: Any = None
    #: The wrapping drocat-card, so a caller can show or hide the whole section
    #: (hiding only ``panel`` would leave an empty card behind).
    card: Any = None
    #: Controls the caller chose not to show. They stay out of ``values()`` so
    #: the backend's own pinned value stands — a hidden control must never
    #: reach the render as its widget default.
    hidden: frozenset = frozenset()

    def values(self) -> Dict[str, Any]:
        """Return values using the keyword names accepted by VisualizeSkeleton."""
        values: Dict[str, Any] = {}

        def palette_value(field):
            """Return a subprocess-safe value while retaining continuous metadata."""
            colors = field.get_colors()
            if getattr(colors, "is_continuous_palette", False):
                return {
                    "colors": list(colors),
                    "continuous": True,
                }
            return colors

        for name, field in self.fields.items():
            if name in self.hidden:
                continue
            if name == "category_colors":
                # A name -> color map, not a layer-ordered palette.
                values[name] = field.get_colors()
            elif name in {"neuron_colors", "synapse_colors", "roi_colors"}:
                values[name] = palette_value(field)
            elif name in {"brain_mesh_color", "vnc_mesh_color"}:
                values[name] = field.get_value()
            else:
                values[name] = field.value

        if "visualize_top_n" in values:
            values["visualize_top_n"] = int(values["visualize_top_n"] or 0)
        if "export_scale" in values:
            values["export_scale"] = int(values["export_scale"] or 1)
        if "min_synapse_num" in values:
            values["min_synapse_num"] = int(values["min_synapse_num"] or 1)
        if "synapse_size" in values:
            # The combo box accepts free text; an emptied field falls back to
            # the default (3 px). Only a bare integer 1-12 is accepted.
            size = str(values["synapse_size"] or "").strip()
            if not size:
                size = "3"
            elif not is_valid_synapse_size(size):
                ui.notify(
                    f"Invalid Synapse Size '{size}' — using '3' pixels instead.",
                    type="warning",
                )
                size = "3"
            values["synapse_size"] = size
        if "neuron_alpha" in values:
            values["neuron_alpha"] = float(values["neuron_alpha"] or 0)
        if "synapse_alpha" in values:
            values["synapse_alpha"] = float(values["synapse_alpha"] or 0)
        if "mesh_alpha" in values:
            values["mesh_alpha"] = float(values["mesh_alpha"] or 0)
        if "skeleton_mesh_simplification" in values:
            # A hidden "use the method default" checkbox must not override a
            # visible numeric control: the caller that hides it owns the default
            # resolution, so the typed fraction is what gets sent.
            use_default = (
                "use_default_simplification" not in self.hidden
                and bool(values.get("use_default_simplification", True))
            )
            if use_default:
                values["skeleton_mesh_simplification"] = None
            else:
                values["skeleton_mesh_simplification"] = float(
                    values["skeleton_mesh_simplification"] or 0
                )
        # The "use the method default" checkbox is a UI helper, never a renderer
        # keyword — dropped unconditionally so a caller that hides the
        # simplification pair cannot leak it into the render.
        values.pop("use_default_simplification", None)
        if "mesh_roi" in values:
            values["mesh_roi"] = list(values["mesh_roi"] or [])
        if "roi_colors" in values:
            values["mesh_color"] = values.pop("roi_colors")
        # Soma spheres are always rendered; the removed checkbox used to
        # expose this, and the constructor default (True) is pinned here so
        # programmatic callers of values() keep the same behavior.
        values["show_soma"] = True
        # Keep all skeleton visualizations aligned with the application's
        # global data-output preference while still allowing programmatic
        # callers to override it in their settings dictionary.
        values.setdefault("output_format", get_user_default("output_format"))
        return values

    def warn_empty_custom_palettes(self) -> None:
        """Warn when a custom palette is empty (backend default fallback).

        The ROI palette is only consulted when at least one ROI mesh is
        selected, so it is skipped otherwise. Hidden controls are skipped too:
        a caller that never shows a palette must not be warned about it, and a
        category-color editor is never empty (it is seeded from the pipeline's
        own map).
        """
        from .palette_picker import notify_empty_custom_palettes

        def _palette(name, caption):
            field = self.fields.get(name)
            if field is None or name in self.hidden:
                return None
            return (field, caption)

        palettes = [p for p in (
            _palette("neuron_colors", "Neuron Colors"),
            _palette("synapse_colors", "Synapse Colors"),
        ) if p is not None]
        if (self.fields.get("roi_colors") is not None
                and "roi_colors" not in self.hidden
                and self.fields["mesh_roi"].value):
            palettes.append((self.fields["roi_colors"], "ROI Colors"))
        if palettes:
            notify_empty_custom_palettes(*palettes)


def skeleton_visualization_settings(
    *,
    default_top_n: int = 5,
    top_n_label: str = "Visualize Top N",
    top_n_hint: str = "Number of ranked results to render as 3D skeletons.",
    default_visualize_by: str = "type",
    include_ranking: bool = True,
    default_brain_mesh: Optional[str] = None,
    default_skeleton_mode: Optional[str] = None,
    show_high_quality_warning: bool = False,
    default_neuron_alpha: float = 0.3,
    default_show_fig: Optional[bool] = None,
    default_export_views: Optional[bool] = None,
    default_export_method: str = "webdriver",
    dataset_provider: Optional[Callable[[], Any]] = None,
    dataset_watchers: Optional[Iterable[Any]] = None,
    hidden_fields: Optional[Iterable[str]] = None,
    category_colors: Optional[Dict[str, str]] = None,
    category_color_hints: Optional[Dict[str, str]] = None,
    card_id: str = "card-advanced-viz",
) -> SkeletonVisualizationSettings:
    """Create the collapsed advanced visualization editor used by analysis tabs.

    The returned ``values()`` dictionary contains the common, user-facing
    ``VisualizeSkeleton`` settings: grouping, mesh selection, appearance,
    synapses, ROI mesh selection, caching, simplification, and export behavior.
    ``include_ranking`` adds the tab-specific top-N and type/bodyId controls.
    The editor renders as a drocat-card section (same pattern as the
    Skeleton tab's appearance blocks); ``card_id`` must be unique per call
    site because every tab builder runs on the same page.

    A pipeline whose renderer pins some of these knobs (the type-mapping
    validation scenes fix the legend tree, the coordinate template, and the
    synapse skip) passes ``hidden_fields``: those controls are built but not
    shown, and stay out of ``values()`` so the backend's pinned value stands
    instead of the widget's default. ``category_colors`` replaces the ordered
    Neuron Colors palette with a per-category editor seeded from the caller's
    own map, returned under the ``category_colors`` key.
    """
    fields: Dict[str, Any] = {}
    hidden = frozenset(str(name) for name in (hidden_fields or ()))
    if category_colors:
        # The two color editors are mutually exclusive; the ordered palette must
        # not reach the render as an unused layer list.
        hidden = hidden | {"neuron_colors"}

    def _fallback(param, key):
        """Use the caller override when given, else the saved user default."""
        return param if param is not None else get_user_default(key)

    def _fallback_opt_in(param, key):
        """Opt-in controls keep their historical off state unless the user
        explicitly saved an override; the built-in skeleton-tab defaults
        (True) must not leak into analysis tabs.
        """
        if param is not None:
            return param
        if has_user_default(key):
            return get_user_default(key)
        return False

    skeleton_mode_default = _fallback(default_skeleton_mode, "analysis_skeleton_mode")
    brain_mesh_default = _fallback(default_brain_mesh, "brain_mesh")
    show_fig_default = _fallback_opt_in(default_show_fig, "show_fig_skeleton")
    export_views_default = _fallback_opt_in(default_export_views, "export_views")

    with ui.card().classes("w-full drocat-card").props(f'id="{card_id}"') as settings_card:
        with ui.expansion(
            "Advanced Visualization",
            icon="view_in_ar",
        ).classes("w-full drocat-section-expansion") as panel:
            ui.label(
                "These settings apply only to the optional skeleton visualizations "
                "generated by this analysis."
            ).classes("text-caption drocat-muted")

            if include_ranking:
                with param_grid(2):
                    fields["visualize_top_n"] = number_input(
                        top_n_label,
                        default_top_n,
                        1,
                        100,
                        hint=top_n_hint,
                    )
                    fields["visualize_by"] = select_input(
                        "Visualize By",
                        ["type", "bodyId"],
                        default_visualize_by,
                        hint=(
                            "'type': group member neurons by type. "
                            "'bodyId': show individual neurons."
                        ),
                    )

            appearance_label = ui.label("Appearance").classes("drocat-mini-label")
            with param_grid(3):
                fields["skeleton_mode"] = select_input(
                    "Skeleton Mode",
                    SKELETON_MODES,
                    skeleton_mode_default,
                    hint="'tube' is detailed; 'line' is faster for many neurons.",
                )
                fields["legend_mode"] = select_input(
                    "Legend Mode",
                    LEGEND_MODES,
                    get_user_default("legend_mode"),
                    hint="One legend entry per layer, type, or individual neuron. "
                         "'tree' adds a collapsible type -> bodyId/instance panel "
                         "to the exported HTML, or group -> type -> "
                         "bodyId/instance with custom groups.",
                )
                fields["background_color"] = select_input(
                    "Background",
                    ["white", "black"],
                    get_user_default("background"),
                    hint="Background color for the interactive scene and exported views.",
                )
                fields["brain_mesh"] = select_input(
                    "Brain Mesh",
                    BRAIN_MESH_OPTIONS,
                    brain_mesh_default,
                    hint=(
                        "Template for the scene: 'native' uses the dataset's "
                        "own outline and coordinates; 'BANC'/'FAFB'/'male-cns' "
                        "move the whole scene (neurons included) into that "
                        "template's coordinates with its outline. 'none' hides "
                        "the outline."
                    ),
                )
                fields["vnc_mesh"] = checkbox_input(
                    "VNC Mesh",
                    False,
                    hint="Show the ventral nerve cord mesh when the dataset supports it.",
                )

            if show_high_quality_warning:
                ui.label(
                    "⚠️ Analysis visualizations default to line mode for speed. "
                    "For high-quality morphology, open Visualization → Skeleton, "
                    "or change Skeleton Mode to tube and choose fine with 95% "
                    "mesh simplification."
                ).classes("text-caption text-amber-8")

            with ui.row().classes("w-full items-start gap-4"):
                with ui.column().classes("flex-grow"):
                    if category_colors:
                        # The caller's render colors neurons by CATEGORY, so an
                        # ordered layer palette is the wrong shape: the map is
                        # seeded from the pipeline's own defaults.
                        fields["category_colors"] = category_color_editor(
                            "Category Colors",
                            category_colors,
                            hints=category_color_hints,
                        )
                    else:
                        fields["neuron_colors"] = palette_editor(
                            "Neuron Colors",
                            value="Category10",
                            include_auto=False,
                        )
                    fields["neuron_alpha"] = number_input(
                        "Neuron Opacity",
                        default_neuron_alpha,
                        0,
                        1,
                        0.1,
                        hint=(
                            "Global fallback opacity for skeletons (0=invisible, 1=solid). "
                            "A color with an explicit opacity channel (#RGBA/#RRGGBBAA, "
                            "rgba(), or an RGBA tuple) overrides this value for that "
                            "layer; colors without opacity inherit it."
                            if not category_colors else
                            "Global opacity for every skeleton in the scene "
                            "(0=invisible, 1=solid). The category colors above "
                            "carry no alpha of their own, so this is the only "
                            "opacity control."
                        ),
                    ).classes("w-48")
                fields["brain_mesh_color"] = color_swatch_picker(
                    "Brain Mesh Color",
                    value="auto",
                ).classes("flex-grow")
                fields["vnc_mesh_color"] = color_swatch_picker(
                    "VNC Mesh Color",
                    value="auto",
                ).classes("flex-grow")

            synapse_label = ui.label("Synapses and regions").classes("drocat-mini-label")
            with param_grid(3):
                fields["skip_synapse"] = checkbox_input(
                    "Skip Synapses",
                    True,
                    hint="Hide synapse markers for a cleaner skeleton view.",
                )
                fields["min_synapse_num"] = number_input(
                    "Min Synapse Count",
                    3,
                    1,
                    1000,
                    hint="Minimum synapses required for a connection marker.",
                )
                fields["synapse_mode"] = select_input(
                    "Synapse Mode",
                    ["cone", "scatter"],
                    "cone",
                    hint="Directional cones or simple point markers.",
                )
                fields["synapse_size"] = combo_input(
                    "Synapse Size",
                    SYNAPSE_SIZE_OPTIONS,
                    get_user_default("synapse_size") or "1",
                    hint=(
                        "Marker size (1-12). Defaults per mode: 1 px for scatter, "
                        "3x real for mesh modes. Scatter uses it as a pixel size; "
                        "mesh modes use it as a multiplier over the real pre→post "
                        "distance. Type any integer 1-12."
                    ),
                )
                # Follow the mode's size default (scatter 1 px, mesh 3x real) while the
                # value is still at a known default; keep any custom value the user typed.
                def _sync_synapse_size_default():
                    current = str(fields["synapse_size"].value or "").strip()
                    if current in {"1", "3"}:
                        scatter = str(fields["synapse_mode"].value or "").strip().lower() == "scatter"
                        fields["synapse_size"].set_value("1" if scatter else "3")
                fields["synapse_mode"].on_value_change(lambda _e: _sync_synapse_size_default())
                _sync_synapse_size_default()
                fields["uniform_synapse_size"] = checkbox_input(
                    "Uniform Synapse Size",
                    get_user_default("uniform_synapse_size"),
                    hint=(
                        "Use the median pre→post distance for every synapse "
                        "marker so all markers share one size."
                    ),
                )
                fields["synapse_alpha"] = number_input(
                    "Synapse Opacity",
                    0.6,
                    0,
                    1,
                    0.1,
                    hint=(
                        "Global fallback opacity for synapse markers. A color with an "
                        "explicit opacity channel overrides this value; colors "
                        "without opacity inherit it."
                    ),
                )
                fields["mesh_alpha"] = number_input(
                    "ROI Mesh Opacity",
                    0.1,
                    0,
                    1,
                    0.05,
                    hint=(
                        "Global fallback opacity for ROI meshes. A color with an "
                        "explicit opacity channel overrides this value; colors "
                        "without opacity inherit it."
                    ),
                )
            fields["synapse_colors"] = palette_editor(
                "Synapse Colors",
                value="Dark2",
                include_auto=False,
            )
            fields["mesh_roi"] = multi_select_input(
                "Mesh ROIs (optional)",
                [],
                default=[],
                hint="Type ROI names and press Enter to add optional region meshes.",
            ).props("outlined").props('new-value-mode="add-unique"')
            fields["roi_colors"] = palette_editor(
                "ROI Colors",
                value="Cool",
                include_auto=True,
            )

            export_label = ui.label("Data and export").classes("drocat-mini-label")
            with param_grid(3):
                fields["cache_neurons"] = checkbox_input(
                    "Cache Neurons",
                    get_user_default("cache_neurons"),
                    hint="Cache fetched skeletons as portable .swc.zst files in "
                         "the shared cache for faster repeat renders.",
                )
                cache_default_state = {"user_changed": False, "updating": False}

                def on_cache_neurons_change(_event):
                    if not cache_default_state["updating"]:
                        cache_default_state["user_changed"] = True

                fields["cache_neurons"].on_value_change(on_cache_neurons_change)
                fields["cache_synapses"] = checkbox_input(
                    "Cache Synapses",
                    get_user_default("cache_synapses"),
                    hint="Cache fetched synapse data for faster repeat renders.",
                )
                fields["use_default_simplification"] = checkbox_input(
                    "Default Simplification",
                    True,
                    hint=(
                        "Use the method default: fast removes 0.90 of faces; "
                        "fine/artistic remove 0.95 in analysis and dedicated "
                        "Skeleton renders."
                    ),
                )
                fields["neuprint_skeleton_pipeline"] = select_input(
                    "Simplification Method",
                    ["fast", "fine", "artistic"],
                    get_user_default("simplification_method"),
                    hint=(
                        "NeuPrint tube rendering: 'fast' (default) reads the "
                        "shared raw level-0 skeleton source, then applies direct "
                        "mesh decimation in memory plus the FAFB fast "
                        "node-reduction stage; 'fine' smooths/resamples and uses "
                        "the accelerated FAFB radius profile; 'artistic' uses "
                        "vertex-cluster mesh decimation. All methods use "
                        "batched parallel online fetching and are available for "
                        "NeuPrint and FAFB/BANC tube renders; line mode "
                        "bypasses the method."
                    ),
                )
                fields["skeleton_mesh_simplification"] = number_input(
                    "Mesh Simplification",
                    _default_analysis_simplification(
                        dataset_provider() if dataset_provider else None,
                        fields["neuprint_skeleton_pipeline"].value,
                    ),
                    0,
                    0.99,
                    0.05,
                    hint=(
                        "Fraction of skeleton-mesh faces removed (higher is coarser). "
                        "Defaults are 0.90 for fast and 0.95 for fine/artistic."
                    ),
                )
                fields["export_method"] = select_input(
                    "Export Method",
                    ["webdriver", "kaleido"],
                    default_export_method,
                    hint="Renderer used for PNG exports.",
                )
                fields["export_scale"] = number_input(
                    "Export Scale",
                    3,
                    1,
                    10,
                    hint="Resolution multiplier for exported images.",
                )
                fields["show_fig"] = checkbox_input(
                    "Show Figure",
                    show_fig_default,
                    hint="Open the interactive figure after rendering.",
                )
                fields["export_views"] = checkbox_input(
                    "Export Views",
                    export_views_default,
                    hint="Export the configured view images after rendering.",
                )

            # The simplification input is only meaningful when the default is
            # disabled.  Keep the value in the returned dictionary regardless so
            # callers can take one consistent snapshot at run time.
            def refresh_default_simplification(_event=None):
                if "use_default_simplification" in hidden:
                    # The caller scoped the panel down: the visible fraction is
                    # user-owned, so a hidden checkbox must not rewrite it when
                    # the dataset changes underneath.
                    return
                if bool(fields["use_default_simplification"].value):
                    fields["skeleton_mesh_simplification"].set_value(
                        _default_analysis_simplification(
                            dataset_provider() if dataset_provider else None,
                            fields["neuprint_skeleton_pipeline"].value,
                        )
                    )

            def refresh_simplification_controls(_event=None):
                is_line = fields["skeleton_mode"].value == "line"
                follows_default = ("use_default_simplification" not in hidden
                                   and bool(fields["use_default_simplification"].value))
                fields["neuprint_skeleton_pipeline"].set_enabled(not is_line)
                fields["use_default_simplification"].set_enabled(not is_line)
                fields["skeleton_mesh_simplification"].set_enabled(
                    not is_line and not follows_default
                )
                if (
                    not cache_default_state["user_changed"]
                    and not has_user_default("cache_neurons")
                ):
                    # Fetched skeletons persist as portable .swc.zst files in
                    # the shared cache, so caching is the default source policy
                    # for every dataset and render pipeline (FlyWire's prepared
                    # mesh cache remains the default source policy there).
                    default_cache = True
                    if fields["cache_neurons"].value != default_cache:
                        cache_default_state["updating"] = True
                        try:
                            fields["cache_neurons"].set_value(default_cache)
                        finally:
                            cache_default_state["updating"] = False

            def on_default_simplification_change(event):
                # Re-selecting the default should immediately show the current
                # dataset-aware value instead of leaving the previous custom value.
                if bool(event.value):
                    refresh_default_simplification()
                refresh_simplification_controls()

            fields["use_default_simplification"].on_value_change(
                on_default_simplification_change
            )
            fields["skeleton_mode"].on_value_change(refresh_simplification_controls)
            def on_pipeline_change(_event):
                # Keep the displayed value synchronized with the selected method
                # while the user is using the method default.  A custom value is
                # deliberately preserved until the user re-enables the default.
                refresh_default_simplification()
                refresh_simplification_controls()

            fields["neuprint_skeleton_pipeline"].on_value_change(on_pipeline_change)
            for watcher in dataset_watchers or ():
                watcher.on_value_change(refresh_default_simplification)
                watcher.on_value_change(refresh_simplification_controls)
            refresh_simplification_controls()

            # Scope the panel down for this caller. The controls are still
            # built (their linkage handlers keep running against them); they are
            # simply not shown, and `values()` omits every hidden name so the
            # backend's pinned value stands instead of the widget's default.
            for _name in sorted(hidden):
                _field = fields.get(_name)
                if _field is not None:
                    _field.set_visibility(False)
            for _caption, _group in ((appearance_label, APPEARANCE_FIELDS),
                                     (synapse_label, SYNAPSE_FIELDS),
                                     (export_label, EXPORT_FIELDS)):
                if all(_n in hidden for _n in _group):
                    _caption.set_visibility(False)

    # Keep the returned object useful in tests and for callers that want to
    # toggle or restyle the section programmatically.
    result = SkeletonVisualizationSettings(fields, hidden=hidden)
    result.panel = panel
    result.card = settings_card
    return result
