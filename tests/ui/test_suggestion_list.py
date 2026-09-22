"""Tests for the suggestion-list contract shared by both neuron query surfaces."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ui.components.suggestion_list import (  # noqa: E402
    ACTIVE_CLASS,
    ADDED_CLASS,
    CHECK_CLASS,
    ITEM_CLASS,
    MARKED_ROW_TITLE,
    REMOVE_CLASS,
    SUGGESTION_LIMIT,
    chip_is_marked,
    marked_rows,
    without_chip,
)


class TestMarkedRows:
    def test_marks_chipped_values_without_dropping_them(self):
        rows = marked_rows(
            [("PPL101", "type"), ("PPL102", "type")], ["PPL101"])
        assert rows == [["PPL101", "type", True], ["PPL102", "type", False]]

    def test_comparison_ignores_surrounding_whitespace(self):
        rows = marked_rows([("PPL101", "type")], ["  PPL101  "])
        assert rows[0][2] is True

    def test_blank_chips_never_mark_a_row(self):
        rows = marked_rows([("PPL101", "")], ["", "   "])
        assert rows == [["PPL101", "", False]]

    def test_the_history_list_marks_exactly_like_type_ahead(self):
        """There is no list-kind switch left to get wrong.

        The helper sees entries and the field's current values, so a Recent row
        that is also a chip ticks, and clicking it deselects like any other.
        """
        history = marked_rows([("aMe12", "type")], ["aMe12"])
        assert history == [["aMe12", "type", True]]

    def test_caps_at_the_shared_limit(self):
        entries = [(f"n{i}", "type") for i in range(SUGGESTION_LIMIT + 25)]
        rows = marked_rows(entries, [])
        assert len(rows) == SUGGESTION_LIMIT
        assert rows[-1][0] == f"n{SUGGESTION_LIMIT - 1}"


class TestDeselectHelpers:
    """The click-to-deselect side of the contract: a marked row's second click
    has to find the chip it stands for, and only that one."""

    def test_marking_and_deselecting_agree_on_the_match(self):
        chips = ["PPL101", "  aMe12  "]
        for value in ("PPL101", "aMe12"):
            marked = marked_rows([(value, "")], chips)[0][2]
            assert marked is chip_is_marked(chips, value)

    def test_blank_values_never_mark_and_never_remove(self):
        assert chip_is_marked(["PPL101"], "") is False
        assert chip_is_marked([], "PPL101") is False
        assert without_chip(["PPL101"], "") == ["PPL101"]

    def test_removal_drops_one_entry_and_keeps_the_others_spelling(self):
        assert without_chip(["a", "PPL101 ", "b"], "PPL101") == ["a", "b"]
        assert without_chip(["PPL101", "PPL101"], "PPL101") == ["PPL101"]

    def test_a_row_that_lost_its_chip_removes_nothing(self):
        """The model can move under an open list (paste, clear, a second tab);
        an unmatched deselect must not delete an unrelated chip."""
        assert without_chip(["aMe12"], "PPL101") == ["aMe12"]


class TestMarkedRowTitle:
    """Both renderers must offer the same hover text: a marked history row now
    sits next to a prune ``x`` that removes something else entirely."""

    def test_the_query_box_props_the_title_on_a_marked_row(self):
        import inspect

        from ui.components import common

        source = inspect.getsource(common.neuron_list_input)
        assert "MARKED_ROW_TITLE" in source

    def test_the_overlay_carries_the_title_placeholder(self):
        from ui.components import layer_style_editor

        js = layer_style_editor._SUGGESTION_JS
        assert "__MARKED_TITLE_JSON__" in js
        assert "__drocatMarkedTitle" in js

    def test_the_title_survives_substitution_as_one_js_literal(self):
        """The placeholder is replaced by a JS string literal, not bare text.

        ``json.dumps`` owns the escaping (it emits the em dash as ``\\u2014``),
        so what is pinned here is the invariant the renderer depends on: the
        literal the script declares evaluates back to the shared text.
        """
        import json
        import re

        from ui.components import layer_style_editor

        js = layer_style_editor._SUGGESTION_JS.replace(
            "__MARKED_TITLE_JSON__", json.dumps(MARKED_ROW_TITLE))
        declared = re.search(
            r"var __drocatMarkedTitle = (\"[^\"]*\");", js)
        assert declared, js
        assert json.loads(declared.group(1)) == MARKED_ROW_TITLE
        assert "__MARKED_TITLE_JSON__" not in js


class TestClassNamesMatchTheRenderers:
    """The CSS lives in ui/app.py and the overlay renderer is raw JS, so a
    renamed constant would silently drop a style rule rather than fail. The
    query box's own use is already pinned behaviorally in test_ui_e2e."""

    def test_overlay_renderer_emits_the_shared_names(self):
        from ui.components import layer_style_editor

        js = layer_style_editor._SUGGESTION_JS
        # The renderer concatenates class attributes, so match the token
        # rather than a quoted literal.
        for name in (ITEM_CLASS, ADDED_CLASS, CHECK_CLASS, REMOVE_CLASS):
            assert name in js

    def test_app_css_styles_the_shared_names(self):
        from ui.app import DROCAT_CSS

        for name in (ITEM_CLASS, ADDED_CLASS, CHECK_CLASS, REMOVE_CLASS):
            assert f".{name}" in DROCAT_CSS

    def test_the_query_box_tick_is_tinted_like_the_overlay_tick(self):
        """A marked row reads the same on both surfaces.

        The row-level tint cannot reach the icon: ``ui.icon`` carries
        ``.drocat-muted``, which sets its own color. So the box's tick has to
        carry the shared ``drocat-suggest-check`` name AND have its own rule.
        """
        import inspect

        from ui.app import DROCAT_CSS
        from ui.components.common import neuron_list_input

        # The class reaches the row through the constant (an f-string), so the
        # name is what appears in the source -- matching the value instead would
        # pass only if the box hardcoded the string.
        assert "CHECK_CLASS" in inspect.getsource(neuron_list_input)
        assert (f".drocat-suggest-menu .q-item.drocat-suggest-added"
                f" .{CHECK_CLASS}") in DROCAT_CSS


class TestPruneReveal:
    """A history row's prune ``x`` drops the *history entry*, while a marked row's
    tick drops the *chip* -- two removals a few pixels apart. Only one of them is
    on screen until the row is the one being pointed at, and the one that hides
    stays built into the row: adding and removing it would reflow the list under
    the pointer partway through a run of picks."""

    def test_the_prune_hides_until_its_row_is_pointed_at(self):
        import re

        from ui.app import DROCAT_CSS

        assert re.search(
            rf"\.{REMOVE_CLASS} \{{\s*\n\s*opacity: 0;", DROCAT_CSS)
        for reveal in (
            f".{ITEM_CLASS}:hover .{REMOVE_CLASS}",
            f".{ITEM_CLASS}.{ACTIVE_CLASS} .{REMOVE_CLASS}",
            f".drocat-suggest-menu .q-item:hover .{REMOVE_CLASS}",
            f".drocat-suggest-menu .q-item:focus-within .{REMOVE_CLASS}",
        ):
            assert reveal in DROCAT_CSS, reveal

    def test_hiding_is_css_only_on_both_surfaces(self):
        """Neither renderer may decide to skip the control.

        The overlay's is a client-side span, so the pin is that its branch is
        gated on the list being history and not on the row being marked. The
        query box's is a real ``ui.button`` and is pinned behaviorally by
        ``test_neuron_list_input_history_rows_mark_and_deselect``.
        """
        import re

        from ui.components import layer_style_editor

        branch = re.search(
            r"if \(isHistory\) \{[^}]*" + REMOVE_CLASS,
            layer_style_editor._SUGGESTION_JS,
            re.S,
        )
        assert branch, "every history row must build the prune"
        assert "added" not in branch.group(0)


def test_shared_limit_is_the_query_box_default():
    """The overlay caps at SUGGESTION_LIMIT, so the box must default to it."""
    import inspect

    from ui.components.common import neuron_list_input

    defaults = {
        parameter.name: parameter.default
        for parameter in inspect.signature(neuron_list_input).parameters.values()
    }
    assert defaults["suggestion_limit"] == SUGGESTION_LIMIT


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
