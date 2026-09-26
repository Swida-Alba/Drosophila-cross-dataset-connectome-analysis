"""
Visual color palette tools for DROCAT.

- ``palette_picker``: preset palette cards with full-gradient previews.
- ``palette_editor``: palette editor with ONE interactive preview row -
  drag-and-drop reordering of discrete colors, a live range slider, and a
  reset button beside the preview (the current state is the only preview).
  Preset palettes are picked from a name dropdown with the full-palette
  preview strip beside it - the same selector layout as the custom mode's
  Bokeh palette picker. Custom colors (mode toggle) are added via the shared
  single-color popup, a Bokeh-palette swatch picker, or a free-form color
  string. Custom entries are reordered and removed directly in the horizontal
  preview row.
- ``color_swatch_picker``: single-color swatches with a custom color input.
- ``category_color_editor``: one labelled color row per named category, for
  renders whose color is a property of the category rather than of a layer
  position. Each row is the Skeleton layer editor's color cell — a swatch button
  that opens the shared ``color_picker_popup`` (palettes + preview + opt-in
  alpha) beside an editable value.
"""

import re
from typing import Callable, Dict, List, Optional, Tuple

from nicegui import ui

try:
    from utils.color_utils import color_to_rgba_string, standardize_color
except ModuleNotFoundError:  # Allow importing the UI package directly from the repo.
    from src.utils.color_utils import color_to_rgba_string, standardize_color


COLOR_FORMAT_HINT = (
    "Accepted: named colors (red, royalblue, transparent), "
    "#RGB/#RGBA/#RRGGBB/#RRGGBBAA, RGB(A) tuples/lists with 0-255 "
    "channels or normalized 0-1 floats, CSS rgb()/rgba() (comma or "
    "space/slash syntax), and hsl()/hsla(). RGB channels may be percentages; "
    "the opacity channel may be 0-1, a percentage, or 0-255. An explicit "
    "opacity channel overrides the global opacity for that entry; omit it "
    "to inherit the default."
)


def _prepare_custom_color(
    value: str,
    opacity_override: bool = False,
    opacity_value: float = 1.0,
) -> str:
    """Validate a custom color and optionally attach an explicit opacity."""
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Enter a color value first")
    if raw.lower() == "auto":
        raise ValueError("Use a concrete color for custom entries; 'auto' is a preset")
    # Parse once for an actionable error before storing the user's spelling.
    standardize_color(raw)
    if opacity_override:
        return color_to_rgba_string(raw, alpha=float(opacity_value))
    return raw


_PALETTE_CATALOG: Optional[List[Tuple[str, List[str]]]] = None

# Keep the display name and its source attribute together.  Bokeh exposes
# both size-specific lists (for example ``Blues256``) and size-indexed mapping
# objects (``Blues``).  Looking up names by string concatenation makes it easy
# for a later refactor to attach the wrong source to a card; this table makes
# every name-to-palette mapping explicit and deterministic.
_PALETTE_SOURCE_SPECS = (
    # Categorical palettes: use the largest size available from the mapping.
    ("Category10", ("Category10",)),
    ("Category20", ("Category20",)),
    ("Category20b", ("Category20b",)),
    ("Category20c", ("Category20c",)),
    ("Accent", ("Accent",)),
    ("Dark2", ("Dark2",)),
    ("Paired", ("Paired",)),
    ("Pastel1", ("Pastel1",)),
    ("Pastel2", ("Pastel2",)),
    ("Set1", ("Set1",)),
    ("Set2", ("Set2",)),
    ("Set3", ("Set3",)),
    ("Colorblind", ("Colorblind",)),
    # Sequential palettes: prefer the 256-level source, then the canonical
    # 9-level source, then the Bokeh mapping for older Bokeh releases.
    ("Blues", ("Blues256", "Blues9", "Blues")),
    ("Greens", ("Greens256", "Greens9", "Greens")),
    ("Greys", ("Greys256", "Greys9", "Greys")),
    ("Oranges", ("Oranges256", "Oranges9", "Oranges")),
    ("Purples", ("Purples256", "Purples9", "Purples")),
    ("Reds", ("Reds256", "Reds9", "Reds")),
    ("BuGn", ("BuGn256", "BuGn9", "BuGn")),
    ("BuPu", ("BuPu256", "BuPu9", "BuPu")),
    ("GnBu", ("GnBu256", "GnBu9", "GnBu")),
    ("OrRd", ("OrRd256", "OrRd9", "OrRd")),
    ("PuBu", ("PuBu256", "PuBu9", "PuBu")),
    ("PuBuGn", ("PuBuGn256", "PuBuGn9", "PuBuGn")),
    ("PuRd", ("PuRd256", "PuRd9", "PuRd")),
    ("RdPu", ("RdPu256", "RdPu9", "RdPu")),
    ("YlGn", ("YlGn256", "YlGn9", "YlGn")),
    ("YlGnBu", ("YlGnBu256", "YlGnBu9", "YlGnBu")),
    ("YlOrBr", ("YlOrBr256", "YlOrBr9", "YlOrBr")),
    ("YlOrRd", ("YlOrRd256", "YlOrRd9", "YlOrRd")),
    ("Viridis", ("Viridis256", "Viridis9", "Viridis")),
    ("Plasma", ("Plasma256", "Plasma9", "Plasma")),
    ("Inferno", ("Inferno256", "Inferno9", "Inferno")),
    ("Magma", ("Magma256", "Magma9", "Magma")),
    ("Cividis", ("Cividis256", "Cividis9", "Cividis")),
    ("Turbo", ("Turbo256", "Turbo9", "Turbo")),
    # Diverging palettes: prefer the canonical 11-level source.
    ("BrBG", ("BrBG256", "BrBG11", "BrBG9", "BrBG")),
    ("PiYG", ("PiYG256", "PiYG11", "PiYG9", "PiYG")),
    ("PRGn", ("PRGn256", "PRGn11", "PRGn9", "PRGn")),
    ("PuOr", ("PuOr256", "PuOr11", "PuOr9", "PuOr")),
    ("RdBu", ("RdBu256", "RdBu11", "RdBu9", "RdBu")),
    ("RdGy", ("RdGy256", "RdGy11", "RdGy9", "RdGy")),
    ("RdYlBu", ("RdYlBu256", "RdYlBu11", "RdYlBu9", "RdYlBu")),
    ("RdYlGn", ("RdYlGn256", "RdYlGn11", "RdYlGn9", "RdYlGn")),
    ("Spectral", ("Spectral256", "Spectral11", "Spectral9", "Spectral")),
)


def _resolve_palette_source(module, candidates: Tuple[str, ...]) -> List[str]:
    """Resolve one explicit Bokeh source, including older-version fallbacks."""
    for attribute in candidates:
        source = getattr(module, attribute, None)
        if source is None:
            continue
        if isinstance(source, dict):
            if not source:
                continue
            source = source[max(source.keys())]
        colors = list(source)
        if colors:
            return colors
    return []


def get_palette_catalog() -> List[Tuple[str, List[str]]]:
    """Return the curated catalog with an explicit source for every name."""
    global _PALETTE_CATALOG
    if _PALETTE_CATALOG is not None:
        return _PALETTE_CATALOG

    import bokeh.palettes as bp

    catalog: List[Tuple[str, List[str]]] = []
    seen = set()

    for name, candidates in _PALETTE_SOURCE_SPECS:
        colors = _resolve_palette_source(bp, candidates)
        if not colors or name in seen:
            continue
        seen.add(name)
        catalog.append((name, colors))

    _PALETTE_CATALOG = catalog
    return catalog


def sample_palette(colors: List[str], n: int) -> List[str]:
    """Evenly sample n colors from a palette (repeats when n > palette size)."""
    if n <= 0 or not colors:
        return []
    if n == 1:
        return [colors[0]]
    indices = [round(i * (len(colors) - 1) / (n - 1)) for i in range(n)]
    return [colors[i] for i in indices]


def assign_palette_colors(
    colors: List[str], n: int, *, continuous: bool = False
) -> List[str]:
    """Assign colors, sampling continuous palettes across their full range.

    Discrete palettes retain their exact displayed order and cycle only when
    more colors are needed. Continuous palettes are sampled evenly so a small
    number of layers still spans the selected gradient range.
    """
    if n <= 0 or not colors:
        return []
    if continuous:
        return sample_palette(colors, n)
    return [colors[index % len(colors)] for index in range(n)]


def empty_custom_palette_names(*palettes: tuple) -> List[str]:
    """Names of the palettes in custom mode with no colors yet.

    An empty custom palette falls back to the backend's default palette;
    callers warn about it instead of substituting silently.
    """
    return [
        name
        for palette, name in palettes
        if palette.get_mode() == "custom" and not palette.get_custom_colors()
    ]


def notify_empty_custom_palettes(*palettes: tuple) -> None:
    """Warn when any custom palette is empty (backend default fallback)."""
    names = empty_custom_palette_names(*palettes)
    if names:
        ui.notify(
            f"{' and '.join(names)} custom palette"
            f"{'s are' if len(names) > 1 else ' is'} empty; "
            "the render falls back to the default palette.",
            type="warning",
        )


class _PaletteSelection(list):
    """List-compatible palette value carrying continuous-preset metadata."""

    def __init__(self, colors=(), *, continuous: bool = False):
        super().__init__(colors)
        self.is_continuous_palette = continuous


# Option-slot template for palette dropdowns: renders the name on the left
# and the palette strip on the right, filling the expanded dropdown width,
# so the palette is visible before it is selected. Discrete palettes use a
# precomputed hard-stop gradient (distinct segments, no blurring).
# NiceGUI compiles slot templates as an inline component exposing the slot
# props under the variable ``props``; ``props.itemProps`` carries the
# option's click handler (toggleOption), so the item stays selectable.
_PALETTE_OPTION_SLOT = """
<q-item v-bind="props.itemProps">
    <q-item-section>
        <q-item-label>{{ props.opt.label }}</q-item-label>
    </q-item-section>
    <div class="drocat-select-palette-strip"
         :style="'background: ' + (props.opt.strip || '#94a3b8')">
    </div>
</q-item>
"""

_SELECT_PALETTE_STRIP_SIZE = 12


def _option_strip_background(colors: List[str], discrete: bool) -> str:
    """CSS background for one dropdown strip preview.

    Discrete palettes render their colors as distinct hard-stop segments
    (the palette as it is, no blurring); continuous palettes render as a
    smooth gradient over their sampled colors.
    """
    if discrete:
        count = len(colors)
        stops = ", ".join(
            f"{color} {index * 100 / count}% {(index + 1) * 100 / count}%"
            for index, color in enumerate(colors)
        )
        return f"linear-gradient(90deg, {stops})"
    return f"linear-gradient(90deg, {', '.join(colors)})"


def _embed_palette_strips(select, names: List[str], color_lookup: dict) -> None:
    """Embed a palette strip beside each name in a dropdown's options.

    NiceGUI rebuilds ``_props['options']`` from the plain string options on
    every update (e.g. when the value is set programmatically), so the
    enriched options are re-applied inside an update hook. The ``option``
    slot renders the strip next to the name; option labels stay plain
    strings so the selected display and the value round-trip are unchanged.
    """
    strips = []
    for index, name in enumerate(names):
        source = color_lookup[name]
        # Discrete palettes keep every color (segmented strip); continuous
        # palettes are sampled for a smooth gradient.
        discrete = len(source) <= 20
        display = (
            list(source)
            if discrete
            else sample_palette(source, _SELECT_PALETTE_STRIP_SIZE)
        )
        strips.append(
            {
                "value": index,
                "label": name,
                "colors": display,
                "strip": _option_strip_background(display, discrete=discrete),
            }
        )
    original_update = select.update

    def update_with_strips():
        original_update()
        with select._props.suspend_updates():
            select._props["options"] = strips

    select.update = update_with_strips
    select.add_slot("option", _PALETTE_OPTION_SLOT)
    update_with_strips()


def normalize_palette_range(start_pct: int, end_pct: int) -> Tuple[int, int]:
    """Clamp a palette range and guarantee at least one percentage point."""
    start = max(0, min(99, int(start_pct)))
    end = max(1, min(100, int(end_pct)))
    if end <= start:
        end = start + 1
    return start, end


def palette_slice(colors: List[str], start_pct: int, end_pct: int) -> List[str]:
    """Select a sub-range of a palette by percentage (0-100)."""
    if not colors:
        return []
    start_pct, end_pct = normalize_palette_range(start_pct, end_pct)
    start_idx = round(start_pct / 100 * (len(colors) - 1))
    end_idx = round(end_pct / 100 * (len(colors) - 1))
    return colors[start_idx:end_idx + 1]


def move_color(colors: List[str], index: int, delta: int) -> List[str]:
    """Return a copy with one color moved left/right by ``delta`` places."""
    reordered = list(colors)
    target = index + delta
    if index < 0 or index >= len(reordered) or target < 0 or target >= len(reordered):
        return reordered
    color = reordered.pop(index)
    reordered.insert(target, color)
    return reordered


def _gradient_style(colors: List[str], height: int = 18) -> str:
    """CSS linear-gradient over the whole palette (full preview for long palettes)."""
    if not colors:
        return ""
    stops = sample_palette(colors, min(len(colors), 32))
    return f"background:linear-gradient(90deg,{','.join(stops)});height:{height}px;"


def _render_color_strip(
    colors: List[str],
    height: int = 18,
    click=None,
    classes: str = "",
    selected: Optional[str] = None,
) -> None:
    """Render a color strip in the current context (gradient for long palettes).

    With a ``click`` callback the swatches become individually selectable
    (long palettes are sampled to 20); ``selected`` highlights the swatch
    whose color matches.
    """
    if len(colors) > 20 and click is None:
        strip = ui.element("div").classes(f"drocat-palette-strip {classes}").style(
            _gradient_style(colors, height)
        )
        return
    display_colors = sample_palette(colors, 20) if len(colors) > 20 else colors
    with ui.row().classes(f"gap-0 w-full drocat-palette-swatches {classes}"):
        for color in display_colors:
            sw = ui.element("div").style(
                f"background:{color}; flex:1; min-width:5px; height:{height}px;"
            )
            if click is not None:
                sw.classes("drocat-palette-pick-swatch")
                if selected is not None and color == selected:
                    sw.classes("selected")
                # The event object is the first positional argument; the
                # captured color rides as the default of the second.
                sw.on("click", lambda _e, c=color: click(c))


# Drag-and-drop reordering of discrete palette swatches. The drop handler
# computes the insertion index from the cursor position over the swatch row
# and emits {from, to} to the server (which reorders and re-renders).
_DRAG_OVER_JS = (
    "(event) => { event.preventDefault(); event.dataTransfer.dropEffect = 'move'; }"
)
_DRAG_START_JS = (
    "(event) => { event.dataTransfer.setData('text/plain', "
    "String(event.currentTarget.dataset.index)); "
    "event.dataTransfer.effectAllowed = 'move'; }"
)


def _drop_js(axis: str = "x") -> str:
    """Drop-handler JS computing the insertion index from the cursor
    position along an axis ('x' for horizontal swatch rows, 'y' for
    vertical lists). Emits {from, to} to the server."""
    coord = "clientX" if axis == "x" else "clientY"
    pos = "left" if axis == "x" else "top"
    size = "width" if axis == "x" else "height"
    return (
        "(event) => { event.preventDefault(); "
        "const from = Number(event.dataTransfer.getData('text/plain')); "
        "const rect = event.currentTarget.getBoundingClientRect(); "
        f"const {axis} = event.{coord} - rect.{pos}; "
        "let to = 0; "
        "for (const child of event.currentTarget.querySelectorAll('[draggable=\"true\"]')) { "
        f"  const c = child.getBoundingClientRect(); "
        f"  if ({axis} > c.{pos} + c.{size} / 2 - rect.{pos}) to++; "
        "} "
        "emit({from, to}); }"
    )


def _render_draggable_swatches(
    colors: List[str],
    height: int = 20,
    on_drop: Optional[Callable] = None,
    on_remove: Optional[Callable[[int], None]] = None,
) -> None:
    """Render individual draggable swatches for drag-and-drop reordering.

    ``on_remove`` is used by the custom palette preview.  Keeping the remove
    action on the swatch itself makes the preview the complete custom-palette
    editor; preset palette previews remain plain, non-removable swatches.
    """
    with ui.row().classes(
        "gap-0 w-full drocat-palette-swatches"
    ).on("dragover", None, js_handler=_DRAG_OVER_JS) as row:
        for index, color in enumerate(colors):
            sw = ui.element("div").style(
                f"background:{color}; position:relative; flex:1; "
                f"min-width:5px; height:{height}px;"
            ).tooltip("Drag to reorder")
            sw._props["draggable"] = "true"
            sw._props["data-index"] = str(index)
            sw.on("dragstart", None, js_handler=_DRAG_START_JS)
            if on_remove is not None:
                with sw:
                    ui.button(
                        icon="close",
                        on_click=lambda _e, i=index: on_remove(i),
                    ).props(
                        'flat dense round size="xs" color="negative" '
                        'aria-label="Remove custom color"'
                    ).classes("drocat-palette-remove").tooltip(
                        "Remove color"
                    )
    row.on("drop", on_drop, js_handler=_drop_js("x"))


def palette_picker(
    label: str,
    value: Optional[str] = None,
    catalog: Optional[List[Tuple[str, List[str]]]] = None,
    include_auto: bool = False,
    max_height: int = 220,
) -> ui.element:
    """Preset palette selector with full-color previews (used for small schemes)."""
    catalog = list(catalog) if catalog is not None else list(get_palette_catalog())
    if include_auto:
        catalog = [("Auto (single gray)", ["#94a3b8"])] + catalog
    names = [name for name, _ in catalog]
    if value not in names:
        value = names[0]

    state = {"value": value}
    cards: List[Tuple[ui.element, str]] = []

    with ui.column().classes("w-full gap-1") as container:
        ui.label(label).classes("drocat-mini-label")
        preview = ui.row().classes("items-center gap-2 w-full drocat-palette-preview")

        def render_preview():
            preview.clear()
            with preview:
                _render_color_strip(
                    dict(catalog)[state["value"]],
                    height=20,
                    classes="flex-grow",
                )
                ui.label(state["value"]).classes("text-caption drocat-muted")

        with ui.expansion("Choose palette", icon="palette").classes(
            "w-full drocat-palette-expansion"
        ):
            with ui.element("div").classes("drocat-palette-grid").style(
                f"max-height:{max_height}px"
            ):
                for name, colors in catalog:
                    card = ui.element("div").classes(
                        "drocat-palette-card"
                        + (" selected" if name == state["value"] else "")
                    ).on("click", lambda n=name: select(n))
                    with card:
                        _render_color_strip(colors, height=18)
                        ui.label(name).classes("drocat-palette-name")
                    cards.append((card, name))

        def select(name: str):
            state["value"] = name
            for element, card_name in cards:
                element.classes(
                    replace="drocat-palette-card"
                    + (" selected" if card_name == name else "")
                )
            container.value = name
            render_preview()

        render_preview()

    container.value = state["value"]
    container.get_value = lambda: state["value"]
    container.get_colors = lambda: dict(catalog)[state["value"]]
    return container


def palette_editor(
    label: str,
    value: Optional[str] = None,
    include_auto: bool = False,
    on_change: Optional[Callable] = None,
) -> ui.element:
    """
    Palette editor with ONE interactive preview row.

    Features:
    - Preset palettes from bokeh.palettes, picked from a full-width name
      dropdown. Every option shows the name with a strip preview filling
      the row to its right, so palettes are visible before selection; the
      selected palette is shown in the reorderable preview row.
    - The preview row shows the current state only: drag swatches to
      reorder discrete palettes, use the range slider to select part of
      the palette live, and hit reset (beside the preview) to restore the
      original order and the full range.
    - Custom colors: the palette starts empty (no preset fallback) and
      gains single colors via the shared popup, a Bokeh palette swatch grid,
      or a free-form color string. Entries without an explicit opacity
      channel inherit the renderer's global opacity. Custom swatches are
      reordered and removed directly in the horizontal preview row.

    ``on_change`` fires when the user manually edits the palette state
    (picking a palette in the dropdown, drag-reordering, adjusting the
    range, resetting); callers use it to stop auto-following e.g. the
    background color. The returned container also gains
    ``set_palette(name)`` for programmatic palette switching, which does
    NOT fire ``on_change``.
    """
    catalog = list(get_palette_catalog())
    if include_auto:
        catalog = [("Auto (single gray)", ["#94a3b8"])] + catalog
    names = [name for name, _ in catalog]
    if value not in names:
        value = names[0]

    state = {
        "mode": "preset",
        "palette": value,
        "start": 0,
        "end": 100,
        "custom": [],
        "preset_orders": {},
        "custom_input_source": "picker",
        "syncing_custom_input": False,
        "syncing_preset_select": False,
        "picked_color": None,
        "picker_palette": "Category10",
    }

    def palette_colors(name: Optional[str] = None) -> List[str]:
        palette_name = name or state["palette"]
        original = list(dict(catalog)[palette_name])
        return list(state["preset_orders"].get(palette_name, original))

    def is_discrete() -> bool:
        return len(palette_colors()) <= 20

    def effective_slice() -> List[str]:
        return palette_slice(
            palette_colors(), state["start"], state["end"]
        )

    def effective_colors() -> List[str]:
        if state["mode"] == "custom":
            # No preset fallback: the custom palette stays empty until the
            # user adds colors.
            return list(state["custom"])
        return _PaletteSelection(
            effective_slice(),
            continuous=(state["mode"] == "preset" and not is_discrete()),
        )

    def colors_for_count(n: int) -> List[str]:
        """Return colors ready for a known number of rendered items."""
        colors = effective_colors()
        return assign_palette_colors(
            colors,
            max(0, int(n or 0)),
            continuous=(
                state["mode"] == "preset"
                and not is_discrete()
            ),
        )

    def slice_start_index(colors: List[str]) -> int:
        """Index in ``colors`` where the displayed range slice starts."""
        return round(state["start"] / 100 * (len(colors) - 1))

    with ui.column().classes("w-full gap-1") as container:
        ui.label(label).classes("drocat-mini-label")

        mode_toggle = ui.toggle(
            ["Preset palette", "Custom colors"], value="Preset palette"
        ).props("dense outlined")

        # ---- Single interactive preview row (current state only) ----
        preview = ui.row().classes("items-center gap-2 w-full drocat-palette-preview")
        with preview:
            swatch_area = ui.row().classes(
                "gap-0 flex-grow drocat-palette-swatches"
            )
            reset_button = ui.button(
                icon="restart_alt", on_click=lambda: reset_all()
            ).props(
                'flat dense round aria-label="Reset palette"'
            ).tooltip("Reset order and range")

        status_label = ui.label("").classes("text-caption drocat-muted")

        def render_preview():
            colors = effective_colors()
            # Drag-and-drop only makes sense for discrete color lists: preset
            # palettes with <= 20 colors and any non-empty custom list.
            draggable = (
                (state["mode"] == "preset" and is_discrete())
                or (state["mode"] == "custom" and bool(state["custom"]))
            )
            swatch_area.clear()
            with swatch_area:
                if draggable:
                    _render_draggable_swatches(
                        colors,
                        height=20,
                        on_drop=handle_drop,
                        on_remove=(
                            remove_custom
                            if state["mode"] == "custom"
                            else None
                        ),
                    )
                elif colors:
                    _render_color_strip(
                        colors, height=20, classes="w-full"
                    )
                else:
                    # Custom mode starts empty: show a placeholder instead
                    # of the preset palette until colors are added.
                    ui.label("No custom colors yet — use Add color below.").classes(
                        "text-caption drocat-muted"
                    )
            if state["mode"] == "custom" and state["custom"]:
                status = "custom colors"
            elif state["mode"] == "custom":
                status = "no custom colors yet"
            else:
                status = f"{state['palette']}  {state['start']}–{state['end']}%"
            if draggable:
                status += " · drag swatches to reorder"
            status_label.text = status

        def handle_drop(e):
            """Reorder the displayed colors after a drag-and-drop."""
            args = e.args or {}
            from_idx = int(args.get("from", 0) or 0)
            to_idx = int(args.get("to", 0) or 0)
            if state["mode"] == "custom" and state["custom"]:
                colors = list(state["custom"])
                if not (0 <= from_idx < len(colors)):
                    return
                to_idx = max(0, min(to_idx, len(colors)))
                item = colors.pop(from_idx)
                colors.insert(
                    to_idx - 1 if from_idx < to_idx else to_idx, item
                )
                state["custom"] = colors
            else:
                colors = palette_colors()
                offset = slice_start_index(colors)
                displayed = effective_slice()
                if not (0 <= from_idx < len(displayed)):
                    return
                to_idx = max(0, min(to_idx, len(displayed)))
                item = colors.pop(offset + from_idx)
                insert_at = offset + to_idx
                if from_idx < to_idx:
                    insert_at -= 1
                colors.insert(insert_at, item)
                state["preset_orders"][state["palette"]] = colors
            render_preview()
            if on_change:
                on_change()

        # ---------------- Preset panel ----------------
        with ui.column().classes("w-full gap-1") as preset_panel:
            # Full-width palette dropdown: every option shows the name with a
            # strip preview filling the row to its right. The selected
            # palette itself is always visible in the reorderable preview
            # row above, so no preview strip sits beside the selection box.
            preset_select = ui.select(
                options=names, value=state["palette"], label="Palette"
            ).props("dense outlined").classes("w-full").tooltip(
                "Pick a preset palette from the Bokeh catalog; every option "
                "previews the palette as a strip beside its name."
            )
            _embed_palette_strips(preset_select, names, dict(catalog))

            def on_preset_select_change(event):
                if state["syncing_preset_select"]:
                    return
                apply_palette(event.value or state["palette"], notify=True)

            preset_select.on_value_change(on_preset_select_change)

            with ui.row().classes("w-full items-center gap-2"):
                ui.label("Palette range").classes("text-caption drocat-muted")
                # Thumb-position bubbles would overlap neighboring text, so
                # the start/end values are shown as lateral labels aligned
                # with the track ends instead (they update live on drag).
                range_start_label = ui.label("0").classes(
                    "text-caption font-mono drocat-muted"
                ).style("width: 26px; text-align: right; flex: none;")
                range_slider = ui.range(
                    min=0, max=100, step=1, value={"min": 0, "max": 100}
                ).props("dense color=primary").style(
                    # q-range renders at full width; a zero flex-basis lets it
                    # share the row with the lateral labels and grow into the
                    # leftover space (single line, labels at the track ends).
                    "flex: 1 1 0%; min-width: 80px;"
                )
                range_end_label = ui.label("100").classes(
                    "text-caption font-mono drocat-muted"
                ).style("width: 30px; flex: none;")

            def apply_range(e):
                value = e.args if isinstance(e.args, dict) else {}
                start, end = normalize_palette_range(
                    value.get("min", state["start"]),
                    value.get("max", state["end"]),
                )
                state["start"], state["end"] = start, end
                range_slider.value = {"min": start, "max": end}
                range_start_label.text = str(start)
                range_end_label.text = str(end)
                render_preview()
                if on_change:
                    on_change()

            range_slider.on(
                "update:model-value", apply_range,
                throttle=0.1, leading_events=True, trailing_events=True,
            )

        # ---------------- Custom panel ----------------
        with ui.column().classes("w-full gap-1") as custom_panel:
            with ui.row().classes(
                "w-full items-center gap-2 flex-wrap drocat-custom-color-input-row"
            ):
                # The shared popup is created just after these controls so the
                # palette's own Color format field remains the first editable
                # format field in the page.
                pick_button = ui.button(
                    "Pick color",
                    icon="palette",
                    on_click=lambda: pick_popup.open(current_raw_color()),
                ).props("outline dense").classes("shrink-0").tooltip(
                    "Open the shared color picker with alpha and Bokeh palettes."
                )
                format_input = ui.input(
                    label="Color format",
                    value="#145cff",
                    placeholder="rgba(255, 0, 0, 0.5)",
                ).props("dense outlined").classes("flex-grow min-w-[180px]").tooltip(
                    COLOR_FORMAT_HINT
                )
                color_preview = ui.element("div").classes(
                    "drocat-custom-color-preview"
                ).style(
                    "width:24px; height:24px; border-radius:6px; "
                    "border:1px solid rgba(11,31,58,.15); flex:none;"
                ).tooltip("Live preview of the color that will be added.")
                ui.button(
                    "Add color", icon="add", on_click=lambda: add_custom_color()
                ).props("flat dense color=primary").classes("shrink-0")

            ui.label(
                "Pick a color with the popup, click a Bokeh-palette swatch, or "
                "type a format and press Add. Supported: named colors; "
                "#RGB/#RGBA/#RRGGBB/#RRGGBBAA; RGB(A) 0–255 or normalized 0–1; "
                "rgb()/rgba(); hsl()/hsla(). Explicit alpha is kept only when "
                "it is part of the color value or Override alpha is checked."
            ).classes("text-caption drocat-muted")

            # Bokeh-palette source: pick individual colors from any catalog
            # palette. Clicking a swatch selects it (highlighted) and syncs
            # the Color format input; the Add button commits it.
            ui.label("From Bokeh palette").classes("drocat-mini-label")
            palette_select = ui.select(
                options=names, value=state["picker_palette"], label="Bokeh palette"
            ).props("dense outlined").classes("w-full").tooltip(
                "Pick individual colors from any Bokeh palette in the "
                "catalog: click a swatch below to select it, then press Add."
            )
            _embed_palette_strips(palette_select, names, dict(catalog))
            swatch_grid = ui.element("div").classes("w-full")

            def render_picker_grid():
                swatch_grid.clear()
                with swatch_grid:
                    _render_color_strip(
                        list(dict(catalog)[state["picker_palette"]]),
                        height=20,
                        click=on_swatch_click,
                        selected=state["picked_color"],
                    )

            def on_swatch_click(color: str):
                state["custom_input_source"] = "palette"
                state["picked_color"] = color
                state["syncing_custom_input"] = True
                format_input.set_value(color)
                state["syncing_custom_input"] = False
                render_picker_grid()
                render_color_preview()

            def on_palette_select_change(event):
                state["picker_palette"] = event.value or state["picker_palette"]
                state["picked_color"] = None
                render_picker_grid()

            palette_select.on_value_change(on_palette_select_change)
            render_picker_grid()

            # Build the shared popup after the custom panel's Bokeh selector so
            # the selector used to add colors remains the first matching field
            # in the page and the two palettes do not intercept each other's
            # changes.
            from .color_picker_popup import color_picker_popup

            picker_id = "card-palette-editor-picker-" + re.sub(
                r"[^a-z0-9]+", "-", str(label).lower()
            ).strip("-")
            pick_popup = color_picker_popup(
                card_id=picker_id or "card-palette-editor-picker"
            )
            container.custom_color_popup = pick_popup
            container.custom_color_picker_button = pick_button

            def current_raw_color():
                """The color value of the active input source."""
                source = state["custom_input_source"]
                if source == "palette":
                    return state["picked_color"] or format_input.value or "#145cff"
                return format_input.value or "#145cff"

            def render_color_preview():
                """Show the current color in the square preview."""
                raw = current_raw_color()
                display = "transparent"
                try:
                    standardize_color(raw)
                    display = raw
                except (TypeError, ValueError):
                    pass
                color_preview.style(f"background:{display}")

            def on_popup_color(value: str):
                state["custom_input_source"] = "picker"
                state["picked_color"] = value
                state["syncing_custom_input"] = True
                format_input.set_value(value)
                state["syncing_custom_input"] = False
                render_color_preview()

            def on_format_change(_event):
                if not state["syncing_custom_input"]:
                    state["custom_input_source"] = "text"
                render_color_preview()

            format_input.on_value_change(on_format_change)
            pick_popup.on_submit(on_popup_color)
            render_color_preview()

            def add_single_color(color: str):
                if color and color not in state["custom"]:
                    state["custom"].append(color)
                    render_preview()
                    if on_change:
                        on_change()

            def add_custom_color():
                raw = current_raw_color()
                try:
                    color = _prepare_custom_color(
                        raw,
                        opacity_override=False,
                    )
                except (TypeError, ValueError) as exc:
                    ui.notify(str(exc), type="warning")
                    return
                add_single_color(color)

            def remove_custom(index: int):
                if not 0 <= index < len(state["custom"]):
                    return
                del state["custom"][index]
                render_preview()
                if on_change:
                    on_change()

        def apply_palette(name: str, notify: bool = False):
            state["palette"] = name
            container.value = name
            state["syncing_preset_select"] = True
            preset_select.value = name
            state["syncing_preset_select"] = False
            range_slider.value = {"min": state["start"], "max": state["end"]}
            range_start_label.text = str(state["start"])
            range_end_label.text = str(state["end"])
            render_preview()
            if notify and on_change:
                on_change()

        def reset_all():
            """Restore the original palette order/range (or clear custom colors)."""
            if state["mode"] == "custom":
                state["custom"] = []
            else:
                state["preset_orders"].pop(state["palette"], None)
                state["start"], state["end"] = 0, 100
                range_slider.value = {"min": 0, "max": 100}
                range_start_label.text = "0"
                range_end_label.text = "100"
            render_preview()
            if on_change:
                on_change()

        def on_mode_change():
            state["mode"] = "custom" if mode_toggle.value == "Custom colors" else "preset"
            preset_panel.set_visibility(state["mode"] == "preset")
            custom_panel.set_visibility(state["mode"] == "custom")
            render_preview()

        mode_toggle.on_value_change(lambda _e: on_mode_change())
        on_mode_change()
        render_preview()

    container.value = state["palette"]
    container.get_value = lambda: state["palette"]
    container.get_mode = lambda: state["mode"]
    container.get_custom_colors = lambda: list(state["custom"])
    container.get_palette_order = lambda: list(palette_colors())
    container.get_range = lambda: (state["start"], state["end"])
    container.get_colors = effective_colors
    container.get_colors_for_count = colors_for_count
    container.set_palette = lambda name: apply_palette(name, notify=False)
    return container


def color_swatch_picker(
    label: str,
    value: str = "auto",
    options: Optional[List[Tuple[str, str]]] = None,
) -> ui.element:
    """Single-color swatches with free-form color and opacity support."""
    options = options or [
        ("auto", "Auto"),
        ("#94a3b8", "Gray"),
        ("#145cff", "Blue"),
        ("#22c55e", "Green"),
        ("#eab308", "Yellow"),
        ("#ef4444", "Red"),
        ("#8b5cf6", "Purple"),
        ("#000000", "Black"),
        ("#ffffff", "White"),
    ]
    state = {
        "value": value if any(v == value for v, _ in options) else "auto",
        "input_source": "picker",
        "syncing_input": False,
    }

    with ui.column().classes("w-full gap-1") as container:
        ui.label(label).classes("drocat-mini-label")
        with ui.row().classes("items-center gap-2 w-full drocat-swatch-row"):
            swatches = []
            for option_value, display in options:
                swatch = ui.element("div").classes(
                    "drocat-swatch"
                    + (" selected" if option_value == state["value"] else "")
                ).tooltip(display)
                with swatch:
                    ui.element("div").style(
                        f"background:{option_value if option_value != 'auto' else '#74A8D6'};"
                        "width:22px; height:22px; border-radius:50%;"
                        "border:2px solid rgba(11,31,58,.15);"
                    )
                swatches.append((swatch, option_value))

        with ui.row().classes("w-full items-center gap-2"):
            custom_input = ui.color_input(value="#3b82f6").props("dense").tooltip(
                "Quick opaque color picker; use Color format for opacity or other syntax."
            )
            format_input = ui.input(
                label="Color format",
                value="#3b82f6",
                placeholder="rgba(116, 168, 214, 0.04)",
            ).props("dense outlined").classes("flex-grow").tooltip(
                COLOR_FORMAT_HINT
            )
            opacity_override = ui.checkbox(
                "Opacity override", value=False
            ).tooltip(
                "Embed a per-mesh opacity that overrides any global opacity."
            )
            opacity_value = ui.number(
                label="Opacity (0–1)",
                value=1.0,
                min=0,
                max=1,
                step=0.05,
            ).classes("w-28").tooltip(
                "Per-color opacity override. Leave it off to inherit the global opacity."
            )
            opacity_value.disable()
            ui.button(
                "Use custom color", on_click=lambda: select_custom_color()
            ).props("flat dense color=primary")
            # Advanced single-color picker popup (alpha + Bokeh palette).
            from .color_picker_popup import color_picker_popup

            pick_popup = color_picker_popup(card_id="card-color-swatch-picker-popup")
            ui.button(
                "Pick with picker",
                on_click=lambda: pick_popup.open(
                    format_input.value if state["input_source"] == "text" else custom_input.value
                ),
            ).props("flat dense outline").tooltip(
                "Open the single-color picker (hex/rgb + alpha + Bokeh palette)."
            )

        ui.label(
            "Accepted: named colors; #RGB/#RGBA/#RRGGBB/#RRGGBBAA; RGB(A) "
            "0–255 or normalized 0–1; rgb()/rgba(); hsl()/hsla(). An explicit "
            "opacity channel wins for this mesh; omit it to inherit the "
            "global opacity."
        ).classes("text-caption drocat-muted")

        def on_picker_change(event):
            state["input_source"] = "picker"
            state["syncing_input"] = True
            format_input.set_value(event.value or custom_input.value)
            state["syncing_input"] = False

        def on_format_change(_event):
            if not state["syncing_input"]:
                state["input_source"] = "text"

        custom_input.on_value_change(on_picker_change)
        format_input.on_value_change(on_format_change)
        opacity_override.on_value_change(
            lambda event: opacity_value.set_enabled(bool(event.value))
        )

        def select(option_value: str):
            state["value"] = option_value
            container.value = option_value
            for element, swatch_value in swatches:
                element.classes(
                    replace="drocat-swatch"
                    + (" selected" if swatch_value == option_value else "")
                )

        def select_custom_color():
            raw = (
                format_input.value
                if state["input_source"] == "text"
                else custom_input.value
            ) or "#3b82f6"
            try:
                color = _prepare_custom_color(
                    raw,
                    opacity_override=bool(opacity_override.value),
                    opacity_value=float(opacity_value.value or 1.0),
                )
            except (TypeError, ValueError) as exc:
                ui.notify(str(exc), type="warning")
                return
            select(color)

        for element, option_value in swatches:
            element.on("click", lambda v=option_value: select(v))

        # Wire the advanced single-color picker popup (defined above) once
        # ``select`` exists, and expose the popup for tests.
        pick_popup.on_submit(select)
        container.pick_popup = pick_popup

    container.value = state["value"]
    container.get_value = lambda: state["value"]
    return container


def category_color_editor(
    label: str,
    defaults: Dict[str, str],
    *,
    hints: Optional[Dict[str, str]] = None,
    columns: int = 2,
) -> ui.element:
    """One labelled color row per category, seeded from the pipeline's own map.

    This is the shape needed when a color is a property of a CATEGORY rather
    than of a layer position: the caller wants a ``name -> color`` dict back and
    every row has to say what it paints, which an ordered ``palette_editor``
    (position-indexed, unnamed) cannot express.

    ``defaults`` must come from the backend that consumes the result, so the UI
    never owns a second copy of the palette.

    Each row is the same color cell the Skeleton tab's advanced layer editor
    uses: a swatch button that PREVIEWS the current color and opens the shared
    ``color_picker_popup`` (Bokeh palettes with strip previews, an inline color
    grid, and opt-in alpha), beside an editable value that commits on blur or
    Enter. One popup serves every row through a pending-category pointer, as the
    layer editor does, so a dozen rows do not build a dozen dialogs.

    A color stays alpha-free unless the picker's Override alpha is used; an
    explicit alpha then overrides the global opacity for that category alone.

    The returned container gains ``get_colors()`` (the live map),
    ``get_changed()`` (only the categories the user moved off the default),
    ``get_default_colors()``, ``reset_to_defaults()`` and ``pick_popup``.
    """
    from .color_picker_popup import color_picker_popup

    hints = dict(hints or {})
    original = {str(k): str(v) for k, v in defaults.items()}
    state = {"colors": dict(original)}
    rows: List[Dict[str, object]] = []
    by_key: Dict[str, Dict[str, object]] = {}

    with ui.column().classes("w-full gap-1") as container:
        with ui.row().classes("w-full items-center gap-2"):
            ui.label(label).classes("drocat-mini-label")
            ui.button(
                icon="restart_alt",
                on_click=lambda: reset_all(),
            ).props("flat dense size=sm").tooltip(
                "Restore every category to the pipeline's own color")

        def _apply(key, raw, *, from_picker=False):
            """Validate and store one category color, refreshing its row."""
            row = by_key[key]
            raw = str(raw or "").strip()
            if not raw or raw.lower() == original[key].lower():
                state["colors"][key] = original[key]
            else:
                try:
                    # Store the user's spelling only if the renderer's own
                    # parser accepts it; otherwise the row reverts.
                    standardize_color(raw)
                except (TypeError, ValueError) as exc:
                    ui.notify(f"{key}: {exc}", type="warning")
                    state["colors"][key] = original[key]
                else:
                    state["colors"][key] = raw
            current = state["colors"][key]
            row["swatch"].style(f"background:{current}")
            if from_picker:
                # Only the picker writes back into the field: doing it from the
                # field's own commit handler would re-enter it.
                row["value"].set_value(current)
            moved = current.lower() != original[key].lower()
            row["was"].text = f"was {original[key]}" if moved else ""
            row["was"].set_visibility(moved)
            container.value = dict(state["colors"])

        # One shared picker for every row (the layer editor's pattern), keyed by
        # the label so two editors on one page cannot collide on the DOM id.
        pending = {"key": None}
        picker_id = "card-category-color-editor-picker-" + re.sub(
            r"[^a-z0-9]+", "-", str(label).lower()).strip("-")
        popup = color_picker_popup(
            value=next(iter(original.values()), "#145cff"), card_id=picker_id)

        def _submit(color):
            key = pending["key"]
            if key is not None:
                _apply(key, color, from_picker=True)

        popup.on_submit(_submit)

        def _open(key):
            pending["key"] = key
            popup.open(state["colors"][key])

        with ui.grid(columns=columns).classes("w-full items-center gap-x-4"):
            for name in original:
                with ui.row().classes("w-full items-center no-wrap gap-2"):
                    with ui.column().classes("gap-0 w-40 shrink-0"):
                        ui.label(name).classes("text-caption")
                        was = ui.label("").classes(
                            "text-caption opacity-70").set_visibility(False)
                    # The layer editor's own cell markup and CSS: the button's
                    # background IS the swatch (its content is hidden once a
                    # value is set) and the field flexes beside it.
                    with ui.row().classes(
                            "items-center no-wrap drocat-color-cell flex-grow"):
                        swatch = ui.button().props(
                            "flat dense round size=xs").classes(
                            "drocat-color-cell-picker "
                            "drocat-color-cell-picker-set")
                        swatch.style(f"background:{original[name]}")
                        swatch.tooltip(
                            "Pick a color — palettes, alpha, preview")
                        swatch.on("click", lambda _e, k=name: _open(k))
                        value = ui.input(value=original[name]).props(
                            "dense borderless hide-bottom-space")
                row = {"key": name, "swatch": swatch, "value": value,
                       "was": was}
                rows.append(row)
                by_key[name] = row
                if name in hints:
                    value.tooltip(hints[name])
                # Commit on blur / Enter exactly like the layer editor's cell,
                # not on every keystroke.
                value.on("blur", lambda _e, k=name: _apply(
                    k, by_key[k]["value"].value))
                value.on("keydown.enter", lambda _e, k=name: _apply(
                    k, by_key[k]["value"].value))

        ui.label(
            "Colors stay alpha-free unless you opt into Override alpha in the "
            "picker; an explicit alpha then overrides the global opacity for "
            "that category alone."
        ).classes("text-caption drocat-muted")

        def reset_all():
            for row in rows:
                key = str(row["key"])
                state["colors"][key] = original[key]
                row["swatch"].style(f"background:{original[key]}")
                row["value"].set_value(original[key])
                row["was"].set_visibility(False)
            container.value = dict(state["colors"])

    container.pick_popup = popup
    container.rows = by_key
    container.get_colors = lambda: dict(state["colors"])
    container.get_changed = lambda: {
        k: v for k, v in state["colors"].items() if v != original[k]}
    container.get_default_colors = lambda: dict(original)
    container.reset_to_defaults = reset_all
    return container
