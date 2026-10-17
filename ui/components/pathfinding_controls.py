"""Pathfinding controls shared by the Connection tab group (plan 2026-10-16).

Complete Paths, Shortest Paths, Network and Cross-Dataset Comparison drive the
same engine fields, so their controls must mean the same thing. Every helper
here exists because the per-tab copies had already drifted in a way a reader
could be hurt by:

* the ratio threshold was entered in PERCENT on two tabs and as a RAW 0-1
  fraction on a third, so the same typed "1" meant a 1 % floor on one page and
  a 100 % floor (an empty graph) on another;
* the Run Mode selector carried two different labels and only one tab said
  whether the Fast lane would actually fire at the current depth;
* the hemisphere card was triplicated verbatim, including its enable/disable
  law.

These are the single definitions. A tab that needs one of them calls here
instead of re-typing it, and
``tests/ui/test_pathfinding_controls_parity.py`` holds the tabs to that.
"""

from nicegui import ui

from ..config import (
    CORRIDOR_MODE_CHOICES, THRESHOLD_BASIS_CHOICES,
    THRESHOLD_COMBINATION_CHOICES,
)
from .common import checkbox_input, number_input, select_input

# --- Threshold regime -------------------------------------------------------

# The tabs select on the human labels; `ratio_payloads` is the only place that
# turns a label back into the engine value the payload sends.
THRESHOLD_BASIS_OPTIONS = list(THRESHOLD_BASIS_CHOICES.values())
THRESHOLD_COMBINATION_OPTIONS = list(THRESHOLD_COMBINATION_CHOICES.values())
_BASIS_KEYS = {label: key for key, label in THRESHOLD_BASIS_CHOICES.items()}
_COMBINATION_KEYS = {label: key
                     for key, label in THRESHOLD_COMBINATION_CHOICES.items()}

BASIS_HINT = (
    "Which edge-weight basis thresholds the graph: raw synapse counts, or the "
    "connection ratio (synapses / all-post incoming weight of the post "
    "neuron — comparable across datasets). Ratio mode thresholds edges by the "
    "fraction of the post neuron's total input and ranks paths by their "
    "weakest fractional input."
)

RATIO_PERCENT_HINT = (
    "Ratio-mode threshold in PERCENT: keep edges carrying at least this share "
    "of the post neuron's TOTAL input (F9 all-post denominator). Default 1% "
    "(= 0.01); finer tiers (e.g. 0.05%) stay typeable as decimals. "
    "data_details/ratio_synapse_map.csv shows the per-neuron synapse cutoffs "
    "the value implies."
)

COMBINATION_HINT = (
    "How Min Synapse Count and Min Connection Ratio combine when both are "
    "set. Default: they are alternatives — one threshold governs the graph. "
    "Both (AND): an edge enters the graph only when it clears BOTH thresholds "
    "(each on its own scale); both inputs stay visible and the folder token "
    "carries both tiers."
)

DORMANT_RATIO_HINT = (
    "Dormant entrance, read by NO payload: this is the old raw 0-1 knob. The "
    "live ratio threshold is the PERCENT field in Core, and the ratio filters "
    "the graph whenever Threshold Basis is Connection ratio (or Combine "
    "Thresholds is Both (AND) with a ratio above 0). In 'Any one' synapse "
    "mode connection_ratio is a readout column (weight / all-post incoming "
    "mass) and does not filter."
)

INERT_AND_HINT = (
    "Both (AND) is selected but the ratio is 0, so NOTHING is co-thresholded "
    "— enter a ratio above 0 or this choice changes nothing."
)

_INERT_CLASS = "text-red-8"


def threshold_basis_select(hint: str = BASIS_HINT) -> ui.select:
    """The Threshold Basis dropdown, defaulted from Settings → Defaults."""
    from ..config import get_user_default

    return select_input(
        "Threshold Basis", THRESHOLD_BASIS_OPTIONS,
        default=THRESHOLD_BASIS_CHOICES.get(get_user_default("threshold_basis"),
                                           "Synapse count"),
        hint=hint,
    )


def threshold_combination_select(hint: str = COMBINATION_HINT) -> ui.select:
    """The Combine Thresholds dropdown, defaulted from Settings → Defaults."""
    from ..config import get_user_default

    return select_input(
        "Combine Thresholds", THRESHOLD_COMBINATION_OPTIONS,
        default=THRESHOLD_COMBINATION_CHOICES.get(
            get_user_default("threshold_combination"), "Any one (default)"),
        hint=hint,
    )


def percent_to_fraction(value) -> float:
    """THE percent -> engine conversion, and the only place the group divides
    a typed threshold by a hundred."""
    return float(value or 0.0) / 100.0


def fraction_to_percent(value) -> float:
    """THE engine -> percent display conversion: what a run stores (0.001) as
    the unit every threshold field in the group is typed in (0.1).

    12 SIGNIFICANT digits, not a fixed decimal count: a density-matched tier
    can arrive as an arbitrary float (1.2345678e-05 -> 0.0012345678%), and
    rounding to 8 decimals there lost 1.8e-6 of the tier the reader had just
    been handed, while every hand-typed tier still prints clean (0.1 -> 10.0,
    never 10.000000000000002).
    """
    return float(f'{float(value or 0.0) * 100.0:.12g}')


def min_ratio_percent_input() -> ui.number:
    """The PERCENT ratio threshold — hidden until a regime needs it, and the
    only field a payload may read for ``min_ratio`` (see `ratio_payloads`)."""
    field = number_input("Min Connection Ratio (%)", 1.0, 0.0, 100.0, 0.01)
    field.set_visibility(False)
    # ONE tooltip for the field's whole life, nested so it targets the field.
    # `Element.tooltip()` CONSTRUCTS a new q-tooltip on every call
    # (nicegui/element.py:347-354), so the sync law below mutates this object
    # instead — re-calling it grew a tooltip per user click.
    with field:
        field.inert_note = ui.tooltip(RATIO_PERCENT_HINT)
    return field


def sync_threshold_visibility(basis: ui.select, min_synapse: ui.number,
                              min_ratio_percent: ui.number,
                              combination: ui.select) -> None:
    """THE visibility law (single definition).

    'or' regimes show exactly one knob — the one the basis selects; AND shows
    both whichever basis is active (round-13).

    An AND selection with an empty ratio co-thresholds at 0: a control that
    looks active and is inert (what the Shortest tab shipped), so the field
    marks itself instead of the engine quietly ignoring the choice.

    Marking uses `classes(add=/remove=)`, NEVER `classes(replace=…)`:
    `set_visibility` hides an element by appending the ``hidden`` CLASS
    (nicegui/elements/mixins/visibility.py:105-122), so a replace erased it and
    the field the law had just hidden stayed visible in the browser while
    `.visible` read False — measured on the rendered tabs."""
    ratio_mode = _is_ratio(basis)
    and_mode = _is_and(combination)
    min_synapse.set_visibility(not ratio_mode or and_mode)
    min_ratio_percent.set_visibility(ratio_mode or and_mode)
    inert = and_mode and float(min_ratio_percent.value or 0.0) <= 0.0
    min_ratio_percent.inert_note.set_text(
        INERT_AND_HINT if inert else RATIO_PERCENT_HINT)
    min_ratio_percent.classes(add=(_INERT_CLASS if inert else None),
                              remove=(None if inert else _INERT_CLASS))


def weight_basis_of(basis: ui.select) -> str:
    """The engine ``weight_basis`` value for this selection."""
    return _BASIS_KEYS[basis.value]


def _is_ratio(basis: ui.select) -> bool:
    return weight_basis_of(basis) == "connection_ratio"


def _is_and(combination: ui.select) -> bool:
    return _COMBINATION_KEYS[combination.value] == "and"


def ratio_payloads(basis: ui.select, combination: ui.select,
                   min_synapse: ui.number,
                   min_ratio_percent: ui.number) -> dict:
    """The threshold payload keys, derived in ONE place.

    The percent conversion happens here and nowhere else (law L1). 'or' +
    ratio mode floors the synapse count at 1 (the ratio IS the threshold);
    'or' + synapse mode sends ratio 0.0 (a readout column); AND sends both,
    each on its own scale."""
    ratio_mode = _is_ratio(basis)
    and_mode = _is_and(combination)
    return {
        "weight_basis": weight_basis_of(basis),
        "threshold_combination": "and" if and_mode else "or",
        "min_synapse_num": (int(min_synapse.value)
                            if (and_mode or not ratio_mode) else 1),
        "min_ratio": (percent_to_fraction(min_ratio_percent.value)
                      if (and_mode or ratio_mode) else 0.0),
    }


# --- Run mode (the corridor lane) -------------------------------------------

RUN_MODE_LABEL = "Run Mode (Complete / Fast)"
#: The one value dict (config owns it; the Settings card and every tab read it).
RUN_MODE_OPTIONS = CORRIDOR_MODE_CHOICES
#: The depth at which 'auto' hands the query to the corridor lane — the
#: engine's own gate (`coana._maybe_generate_corridor_deep_dive`).
AUTO_DEPTH = 4

RUN_MODE_HINT = (
    "What runs on this query — complete: the COMPLETE mode only (all intact "
    "paths >= the applied threshold; the run itself). auto: COMPLETE, then "
    f"FAST on deep searches (max intermediate layers >= {AUTO_DEPTH}). both: "
    "ALWAYS run FAST after the run. fast: STANDALONE FAST — only the corridor "
    "lane runs on this query (no Complete run; the folder carries "
    "route_scope_paths/ only). FAST mode is upper-bound-guided representative "
    "exploration under per-class route-share quotas — not a guarantee of the "
    "strongest actual neuron routes. Fast never modifies other outputs. Full "
    "guide via the book icon."
)


def run_mode_selector(hint: str = RUN_MODE_HINT) -> ui.select:
    """The Run Mode selector — one label, one value set, on every surface."""
    from ..config import get_user_default

    return select_input(
        RUN_MODE_LABEL, RUN_MODE_OPTIONS,
        default=get_user_default("corridor_mode"),
        hint=hint,
        help_doc="pathfinding_algorithms",
    )


def run_mode_note_text(mode: str, depth: int, extra: str = "") -> str:
    """What the selected mode will ACTUALLY do at this depth.

    The engine's gate is ``mode in ('both', 'fast') or (mode == 'auto' and
    depth >= 4)``; a reader deserves to see it before clicking — especially on
    a tab whose own default depth already clears the gate (Shortest Paths
    starts at 5, Complete at 2, so the same saved 'auto' does different
    things there)."""
    depth = int(depth or 0)
    if mode == "fast":
        return ("Fast only: the corridor lane runs standalone on this query "
                "— no Complete run (the folder carries route_scope_paths/ "
                "only)." + extra)
    if mode == "complete":
        return "Fast mode: off (Complete only)." + extra
    will = (mode == "both" or (mode == "auto" and depth >= AUTO_DEPTH))
    return (f"Fast mode (corridor lane): "
            f"{'WILL' if will else 'will NOT'} "
            f"run after this search (mode {mode}, "
            f"depth {depth}; auto needs >= {AUTO_DEPTH})."
            # An 'all_neurons' side is not in the field: the payload forces
            # depth 0 for it, so a line that only quotes the field would
            # promise a lane to a run that searches direct connections only.
            + (" An 'all_neurons' side overrides this to depth 0, where auto "
               "does NOT run the lane." if mode == "auto" else "")
            + extra)


class RunModeControl:
    """The selector plus its live status line, wired to a depth source."""

    def __init__(self, depth_getter, extra=''):
        self.depth_getter = depth_getter
        self.extra = extra
        self.control = run_mode_selector()
        self.note = ui.label("").classes("text-caption text-blue-8")
        self.control.on_value_change(lambda _e: self.refresh())
        self.refresh()

    def note_text(self) -> str:
        return run_mode_note_text(self.control.value,
                                  self.depth_getter(), self.extra)

    def refresh(self) -> None:
        # Label.set_text (NOT set_value — house rule)
        self.note.set_text(self.note_text())

    @property
    def value(self) -> str:
        return self.control.value


def run_mode_block(depth_getter, extra: str = '') -> RunModeControl:
    """Build the selector + live line in the caller's current container."""
    return RunModeControl(depth_getter, extra)


# --- Hemisphere card --------------------------------------------------------

class HemisphereControls:
    """The four hemisphere controls plus the enable/disable law.

    When Hemisphere-aware is off the dependents are UNCHECKED and disabled, so
    a greyed-out True never reaches the backend — which applies
    ``hemisphere_filter`` independently of ``separate_hemispheres``, so a stale
    'left' would silently restrict the run."""

    def __init__(self):
        with ui.row().classes("items-center gap-4 flex-wrap"):
            self.separate_hemispheres = checkbox_input(
                "Hemisphere-aware", False,
                hint="Split type/group aggregation into _L/_R/_U hemisphere "
                     "labels.",
            )
            self.hemisphere_filter = select_input(
                "Hemisphere", ["both", "left", "right"], "both",
                hint="'both': all neurons. 'left'/'right': restrict to that "
                     "hemisphere. Neurons WITHOUT an explicit hemisphere (no "
                     "_L/_R instance suffix or Soma side) are always included "
                     "in every option.",
                inline=True,
            )
        with ui.row().classes("items-center gap-4 flex-wrap"):
            self.symmetry_analysis = checkbox_input(
                "Symmetry Analysis", False,
                hint="Generate ipsilateral vs contralateral symmetry outputs.",
            )
        with ui.row().classes("items-center gap-4 flex-wrap"):
            self.keep_only_conserved = checkbox_input(
                "Keep Only Hemisphere-Conserved Edges", False,
                hint="Keep only edges conserved between hemispheres "
                     "(requires Hemisphere-aware).",
            )
        self.separate_hemispheres.on_value_change(lambda _e: self._sync())
        self._sync()

    def _sync(self) -> None:
        if self.separate_hemispheres.value:
            self.keep_only_conserved.enable()
            self.symmetry_analysis.enable()
            self.hemisphere_filter.set_enabled(True)
        else:
            self.keep_only_conserved.value = False
            self.symmetry_analysis.value = False
            self.hemisphere_filter.value = 'both'
            self.keep_only_conserved.disable()
            self.symmetry_analysis.disable()
            self.hemisphere_filter.set_enabled(False)

    def payload(self) -> dict:
        return {
            "separate_hemispheres": self.separate_hemispheres.value,
            "hemisphere_filter": self.hemisphere_filter.value,
            "keep_only_hemisphere_conserved_connections":
                self.keep_only_conserved.value,
            "symmetry_analysis": self.symmetry_analysis.value,
        }


def hemisphere_block() -> HemisphereControls:
    """Build the hemisphere card body in the caller's current container."""
    return HemisphereControls()


# --- Query boxes -----------------------------------------------------------

ALL_NEURONS_REFUSAL = (
    "'all_neurons' is not available on this tab: Network's single query set IS "
    "both sides, so the token asks for the whole dataset against itself. For "
    "every neuron adjacent to a queried side, use Complete Paths with Max "
    "Intermediate Layers = 0."
)


def refuses_all_neurons(values) -> bool:
    """True (and notify) when a single-box query carries the all_neurons chip.

    Complete/Shortest special-case the token because it means 'this side =
    everything'; a one-box tool has no such side, and the engine rejects the
    token on both sides — which a single query always is (source == target)."""
    from .common import uses_all_neurons_token

    if not uses_all_neurons_token(values):
        return False
    ui.notify(ALL_NEURONS_REFUSAL, type="warning")
    return True
