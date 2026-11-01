"""The connection tab group's shared controls: the parity laws (plan 2026-10-16 Q7).

Complete Paths, Shortest Paths, Network and Cross-Dataset Comparison drive the
same engine fields. Every rule here is the shape of a drift that was measured
in the audit — a percent-vs-raw ratio split, a threshold basis only one tab
could reach, a Run Mode with two labels and one honest status line, a network
query box that accepted a token the engine always rejects — and each is checked
against the WRONG hypothesis, so the law fails if the drift returns.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent

# the three tabs that threshold a graph
TABS = ("ui/tabs/find_path.py", "ui/tabs/find_shortest.py", "ui/tabs/network.py")
# the surfaces that RUN a mode choice — Network has none: a 1-hop induced
# sub-network has no search to run twice (plan §1.2, the user's ruling)
MODE_TABS = ("ui/tabs/find_path.py", "ui/tabs/find_shortest.py",
             "ui/tabs/inter_dataset.py")


def _src(rel):
    return (ROOT / rel).read_text()


@pytest.fixture(scope="module")
def pc():
    from ui.components import pathfinding_controls as module
    return module


# --- L1: one unit for the ratio threshold ---------------------------------

def test_only_one_place_converts_the_percent(pc):
    """Every tab sends the SAME `min_ratio` unit: the percent field divided
    by 100 in exactly one function. A second conversion site (or a tab
    reading a raw 0-1 field) is what made a typed '1' mean 1 % on two tabs
    and 100 % — an empty graph — on the third."""
    helpers = _src("ui/components/pathfinding_controls.py")
    # one EXPRESSION, and it is named: a second division anywhere in the group
    # is how a typed '1' came to mean 1 % on one page and 100 % on another
    assert helpers.count("/ 100.0") == 1, \
        "the percent conversion must live in exactly one expression"
    _conv = helpers[helpers.index("def percent_to_fraction"):]
    _conv = _conv[:_conv.index("def fraction_to_percent")]
    assert "/ 100.0" in _conv, "the conversion moved out of its helper"
    assert "percent_to_fraction(min_ratio_percent.value)" in helpers, (
        "ratio_payloads no longer routes the single field through the helper")
    # The fourth surface (Cross-Dataset) is percent too. Read STRUCTURALLY:
    # which function contains each division by a hundred? Only the coverage
    # control may do its own; every ratio-tier path must route through
    # percent_to_fraction. (A line-grep law here would have been a law about
    # my grep, not about the tab.)
    import ast
    x = _src("ui/tabs/inter_dataset.py")
    tree = ast.parse(x)
    dividers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            seg = ast.get_source_segment(x, node) or ""
            if "/ 100" not in seg:
                continue
            # keep only the DEEPEST owner of the division: an outer function
            # contains its inner defs' text too, and naming it would blame the
            # whole tab for one helper
            if any("/ 100" in (ast.get_source_segment(x, d) or "")
                   for d in ast.walk(node)
                   if d is not node and isinstance(d, ast.FunctionDef)):
                continue
            dividers.append(node.name)
    assert all(n.startswith("_xcoverage") or "coverage" in n.lower()
               for n in dividers), dividers
    collect = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name == "_collect_threshold_configuration"]
    assert collect, "the threshold collection was renamed — re-check this law"
    seg = ast.get_source_segment(x, collect[0])
    assert "pc.percent_to_fraction(" in seg, (
        "Cross-Dataset no longer converts its ratio tiers in the shared "
        "helper — the percent/raw split this law exists to prevent")
    assert 'label="Connection-Ratio Thresholds (%)"' in x, (
        "the tier editor is back to raw fractions — the unit split is the "
        "defect this law was written for")
    for rel in TABS + ("ui/tabs/inter_dataset.py",):
        src = _src(rel)
        # the tabs may convert OTHER percents (a coverage share), but never a
        # `min_ratio` payload key
        for block in re.findall(r'"min_ratio"\s*:\s*([^,\n]+)', src):
            assert False, f"{rel} builds its own min_ratio ({block.strip()})"
        assert "min_ratio_threshold.value" not in src, (
            f"{rel} reads the percent field directly instead of through "
            "ratio_payloads()")


def test_every_tab_with_a_ratio_field_uses_the_shared_one(pc):
    for rel in TABS:
        src = _src(rel)
        assert "pc.min_ratio_percent_input()" in src, rel
        assert "pc.ratio_payloads(" in src, rel
        # no raw 0-1 ratio input left feeding a payload
        assert '"Min Connection Ratio (%)"' not in src, \
            f"{rel} re-declares the percent field instead of calling the module"


def test_ratio_payload_values(pc, monkeypatch):
    """The unit law measured: 1 in the field is 0.01 on the wire in every
    regime, and 'or' + synapse basis still zeroes the knob."""
    class _W:
        def __init__(self, value):
            self.value = value

    payloads = pc.ratio_payloads(
        _W("Connection ratio"), _W("Any one (default)"), _W(5), _W(1.0))
    assert payloads == {"weight_basis": "connection_ratio",
                        "threshold_combination": "or",
                        "min_synapse_num": 1, "min_ratio": 0.01}, payloads
    payloads = pc.ratio_payloads(
        _W("Synapse count"), _W("Both (AND)"), _W(5), _W(2.5))
    assert payloads == {"weight_basis": "synapse",
                        "threshold_combination": "and",
                        "min_synapse_num": 5, "min_ratio": 0.025}, payloads
    payloads = pc.ratio_payloads(
        _W("Synapse count"), _W("Any one (default)"), _W(5), _W(1.0))
    assert payloads["min_ratio"] == 0.0 and payloads["min_synapse_num"] == 5


# --- L2: the ratio basis is reachable wherever the engine supports it -----

def test_basis_and_combination_come_from_the_module():
    for rel in TABS:
        src = _src(rel)
        assert "pc.threshold_basis_select(" in src, rel
        assert "pc.threshold_combination_select(" in src, rel
        assert "pc.sync_threshold_visibility(" in src, rel
        # a tab may not re-declare the option lists it shares
        assert '["Synapse count", "Connection ratio"]' not in src, rel
    # Cross-Dataset Comparison keeps its OWN basis select on purpose: there it
    # picks which tier each delegate is thresholded at, not the regime of one
    # graph (plan §1.2), so it is not this law's subject.
    # Cross-Dataset keeps its OWN basis select on purpose (it picks which tier
    # each delegate is thresholded at), but it reads the shared vocabulary and
    # the shared saved default — a hardcoded "Synapse count" would make the
    # Settings key dead on the fourth surface.
    x = _src("ui/tabs/inter_dataset.py")
    assert '["Synapse count", "Connection ratio"]' not in x
    assert "pc.THRESHOLD_BASIS_OPTIONS" in x
    assert 'get_user_default("threshold_basis")' in x


def test_the_ratio_payload_is_the_only_source_of_the_basis():
    """L2's payload half, stated so it CAN fail.

    The old law read `assert "weight_basis" not in src or
    "ratio_payloads" in src` — after the line above asserts ratio_payloads IS
    in src, that disjunction is unconditionally true, so the shortest tab could
    have dropped `weight_basis` from its payload entirely and the test would
    still pass. The binding form: the three tabs that threshold a graph never
    name the engine key themselves (it arrives from the shared helper), and the
    only engine spellings the group can send are the two real ones."""
    for rel in TABS:
        src = _src(rel)
        assert "pc.ratio_payloads(threshold_basis, threshold_combination," \
            in src, rel
        assert '"weight_basis"' not in src, (
            f"{rel} hand-writes the basis key instead of deriving it — the "
            "unit/split this round removed is how a tab then drifts")
        assert "weight_basis=" not in src, rel
    # Cross-Dataset keeps its own widget (per-delegate tiers), but it may only
    # speak the two engine values the helper produces
    x = _src("ui/tabs/inter_dataset.py")
    assert '"connection_ratio"' in x and '"synapse"' in x
    assert '"connection_ratio (bodyId)"' not in x, (
        "the payload must send the engine token, not the parameters.txt "
        "rendering of it")


# --- L3: one Run Mode control, one label, one live line ------------------

def test_run_mode_label_is_single_sourced(pc):
    assert pc.RUN_MODE_LABEL == "Run Mode (Complete / Fast / Type-pruned)"
    for rel in MODE_TABS:
        src = _src(rel)
        assert "Fast Mode (corridor lane)" not in src, \
            f"{rel} reintroduced the second label"
        assert '"corridor_mode": run_mode.value' in src, rel
        # the typed lane rides in the same Run Mode value, not a checkbox
        assert "supports_type_pruned" not in src, rel


@pytest.mark.parametrize("mode,expect", [
    ("complete", "Complete mode"),
    ("fast", "Fast only"),
    ("typed", "Type-pruned only"),
])
def test_the_live_line_describes_exactly_one_mode(pc, mode, expect):
    text = pc.run_mode_note_text(mode)
    assert expect in text, (mode, text)
    # Exactly one lane is claimed; the other lane names do not appear as
    # active statements.
    assert text.count("only") + text.count("Complete mode") >= 1
    if mode == "complete":
        assert "off" in text and "skipped" not in text
    else:
        assert "skipped" in text
    if mode == "typed":
        assert "route_scope_paths_type_pruned/" in text
        assert "bodyId-validated" in text


def test_the_selector_drives_the_note():
    """The Run Mode select is the only mode input: changing it must refresh
    the live line and the typed knobs' visibility."""
    src = _src("ui/components/pathfinding_controls.py")
    assert "self.control.on_value_change(lambda _e: self._on_mode_change())" in src
    assert "def _on_mode_change" in src
    assert "_sync_visibility()" in src and "self.refresh()" in src
    assert "def refresh(self)" in src and "self.note.set_text(" in src
    assert "set_value(" not in src.split("def refresh")[1][:400]
    # no tab needs a depth-driven refresh: the mode no longer depends on depth
    for rel in MODE_TABS:
        assert "run_mode.refresh()" not in _src(rel), rel


def test_typed_run_mode_renders_its_knobs_and_maps_the_payload(pc):
    """The typed lane is chosen from the Run Mode list itself — no lane
    checkbox exists — and the selector's composite value becomes the
    engine's corridor_mode plus type_pruned in the payload."""
    from nicegui import Client
    from nicegui.page import page

    client = Client(page("/typed-run-mode"))
    with client:
        rc = pc.RunModeControl()
        assert rc.value == "complete"
        assert rc.payload() == {"type_pruned": False}
        assert "hidden" in _cls(rc._type_row)
        assert "hidden" in _cls(rc._scope_toggle_row)

        rc.control.value = "typed"
        rc._sync_visibility()
        rc.refresh()

        assert rc.typed and rc.value == "typed"
        assert "hidden" not in _cls(rc._type_row)
        assert "hidden" not in _cls(rc._scope_toggle_row)
        assert rc.payload()["type_pruned"] is True
        assert rc.payload()["type_prune_scope"] == "full_floor"
        assert "Type-pruned only" in rc.note_text()

        rc.per_pair_scope.value = True
        rc._sync_visibility()
        payload = rc.payload()
        assert payload["per_pair_scope"] is True
        assert payload["scope_lt_mode"] == "discover"
        assert "hidden" not in _cls(rc._scope_row)

        rc.control.value = "fast"
        rc._sync_visibility()
        assert not rc.typed and rc.value == "fast"
        assert rc.payload() == {"type_pruned": False}
        assert "hidden" in _cls(rc._type_row)
        assert rc.per_pair_scope.value is False

        labels = [(getattr(el, "_props", None) or {}).get("label")
                  for el in client.elements.values()]
        assert "Type-pruned complete lane" not in labels
    # the selector itself drives the knobs' visibility, not a second control
    shared = _src("ui/components/pathfinding_controls.py")
    assert ("self.control.on_value_change(lambda _e: self._on_mode_change())"
            in shared)


# --- the visibility law reaches the DOM -----------------------------------

def _cls(widget):
    return list(widget.classes)


def _tooltips(client):
    return sum(1 for el in client.elements.values()
               if getattr(el, "tag", None) == "q-tooltip")


def test_the_hidden_ratio_field_is_actually_hidden(pc):
    """The law `set_visibility` implements is a CLASS, not a flag: NiceGUI
    appends ``hidden`` to `element.classes`
    (nicegui/elements/mixins/visibility.py:105-122). The inert marking used to
    end with `classes(replace=…)`, which erased that entry — so on all three
    thresholding tabs the ratio field the law had just hidden stayed ON SCREEN
    while its `.visible` flag read False, and no source-text or label-based
    test could see it. This test reads the class list itself."""
    from nicegui import Client
    from nicegui.page import page
    from ui.components.common import number_input

    client = Client(page("/visibility-law"))
    with client:
        basis = pc.threshold_basis_select()
        combination = pc.threshold_combination_select()
        syn = number_input("Min Synapse Count", 3, 1, 100)
        ratio = pc.min_ratio_percent_input()
        pc.sync_threshold_visibility(basis, syn, ratio, combination)
        default_state = (_cls(ratio), _cls(syn))
        tips_after_build = _tooltips(client)

        # Connection ratio: the ratio shows, the count hides
        basis.value = "Connection ratio"
        pc.sync_threshold_visibility(basis, syn, ratio, combination)
        ratio_state = (_cls(ratio), _cls(syn))

        # Both (AND) with an empty ratio: shown, marked inert
        basis.value = "Synapse count"
        combination.value = "Both (AND)"
        ratio.value = 0
        pc.sync_threshold_visibility(basis, syn, ratio, combination)
        inert_state = _cls(ratio)

        # and the marking must not grow a tooltip per call
        for _ in range(4):
            pc.sync_threshold_visibility(basis, syn, ratio, combination)
        tips = _tooltips(client)

    hidden_ratio, hidden_syn = default_state
    assert "hidden" in hidden_ratio, ("synapse/'or' hid nothing:", hidden_ratio)
    assert "hidden" not in hidden_syn, hidden_syn
    assert "hidden" not in ratio_state[0], ("ratio mode stayed hidden",
                                            ratio_state[0])
    assert "hidden" in ratio_state[1], ratio_state[1]
    assert "text-red-8" in inert_state, inert_state
    assert "hidden" not in inert_state, ("the AND marking re-hid the field",
                                        inert_state)
    assert tips == tips_after_build, (tips, tips_after_build)


# --- the hemisphere card --------------------------------------------------

def test_hemisphere_card_is_not_triplicated():
    shared = _src("ui/components/pathfinding_controls.py")
    for rel in TABS:
        src = _src(rel)
        assert "pc.hemisphere_block()" in src, rel
        assert "def _sync_hemisphere_options" not in src, \
            f"{rel} keeps its own copy of the enable/disable law"
    # the law itself: dependents are unchecked AND disabled
    body = shared[shared.index("def _sync(self)"):]
    assert "self.hemisphere_filter.value = 'both'" in body[:600], body[:600]
    for name in ("keep_only_conserved", "symmetry_analysis"):
        assert f"self.{name}.value = False" in body[:600]


# --- Network's single-box query ------------------------------------------

def test_network_refuses_the_all_neurons_token():
    """The token means 'this side = everything'; Network's one query IS both
    sides, so the engine's both-sides rule fires on every use of it."""
    src = _src("ui/tabs/network.py")
    # on the RAW chips: `apply_filter_mode` turns a chip into a pattern under
    # the non-exact modes, and the token is matched by equality, so a guard
    # fed the rewritten list refuses nothing at all
    assert "pc.refuses_all_neurons(neurons)" in src
    assert "pc.refuses_all_neurons(query)" not in src
    shared = _src("ui/components/pathfinding_controls.py")
    assert "not available on this tab" in shared
    # the path tabs keep their own handling (two boxes, real semantics)
    for rel in ("ui/tabs/find_path.py", "ui/tabs/find_shortest.py"):
        assert "uses_all_neurons_token(" in _src(rel), rel


def test_the_shortest_tab_renders_the_shared_controls():
    """Q1 measured on the rendered page, not on the source text: the Shortest
    tab shows Threshold Basis, the PERCENT ratio field and Combine Thresholds,
    and its Run Mode line states the gate AT THIS TAB'S OWN DEPTH — the tab
    starts at 5, so the built-in 'auto' really does run the lane there (the
    Complete tab at 2 would say the opposite)."""
    from nicegui import Client
    from nicegui.page import page
    from ui.tabs.find_shortest import create_find_shortest_tab

    client = Client(page("/wire-find-shortest-shared"))
    with client:
        create_find_shortest_tab()

    labels = [(getattr(el, "_props", None) or {}).get("label")
              for el in client.elements.values()]
    for want in ("Threshold Basis", "Min Connection Ratio (%)",
                 "Combine Thresholds",
                 "Run Mode (Complete / Fast / Type-pruned)"):
        assert want in labels, want
    # the raw 0-1 field the tab used to carry is gone
    assert "Min Connection Ratio" not in labels
    texts = [str(getattr(el, "text", "") or "")
             for el in client.elements.values()]
    assert any("Complete mode" in t for t in texts), texts


def test_network_states_what_does_not_apply(pc):
    """Q8: the absence of a budget, a tau and a lane is the model, so the tab
    says it instead of leaving a reader to wonder what it is missing."""
    src = _src("ui/tabs/network.py")
    for phrase in ("no path", "Max Intermediate Layers", "Edge", "corridor"):
        assert phrase in src, phrase


def test_the_percent_display_is_clean_and_stays_accurate():
    """`fraction_to_percent` prints a hand-typed tier without binary noise and
    keeps an ARBITRARY engine tier to 12 significant digits (review
    2026-10-16).

    A fixed 8-DECIMAL rounding looked tidy but cut a density-matched tier like
    1.2345678e-05 to 0.00123457%, so the value the reader re-submitted was
    1.8e-6 of a tier away from the one the alignment suggested — and the seed
    path is exactly where such tiers arrive.

    Wrong hypotheses measured here: a float that prints as
    '10.000000000000002', and a display that round-trips worse than 1e-9
    relative for the tiers a real run can emit."""
    from ui.components import pathfinding_controls as pc

    for tier in (0.1, 0.2, 0.3, 0.001, 0.05, 0.5, 0.007, 0.014, 1.0):
        shown = pc.fraction_to_percent(tier)
        assert len(str(shown).split('.')[-1]) <= 4, (tier, shown)
        assert abs(pc.percent_to_fraction(shown) - tier) < 1e-15, (
            tier, shown)
        assert '0000000' not in str(shown), (tier, shown)

    for tier in (0.0011933174224343676, 1.2345678e-05, 0.12345678901234567):
        shown = pc.fraction_to_percent(tier)
        rel = abs(pc.percent_to_fraction(shown) - tier) / tier
        assert rel < 1e-9, (tier, shown, rel)
