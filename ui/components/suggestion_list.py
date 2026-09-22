"""The behavior contract shared by DROCAT's neuron suggestion lists.

Two surfaces render the same list, deliberately differently:
``neuron_list_input`` builds a NiceGUI ``ui.menu`` of ``ui.item`` rows, while the
Skeleton layer editor's per-cell overlay is one client-rendered DIV (creating
NiceGUI items there would re-render the q-table body and remount the focused
q-select, wiping the text being typed). Rendering can differ; what the list
*means* cannot, so both surfaces decide it here: a row that stands for a value
the field already holds keeps its place and reads as marked, and clicking it
takes that value back out.
"""
from typing import Iterable, List, Sequence, Set, Tuple

# Rows offered at once.
SUGGESTION_LIMIT = 50

# Shared by the Python row builders, the overlay's client renderer and the CSS
# in ui/app.py. Renaming one without the others silently loses a style rule.
ITEM_CLASS = "drocat-suggest-item"
ADDED_CLASS = "drocat-suggest-added"
ACTIVE_CLASS = "drocat-suggest-active"
CHECK_CLASS = "drocat-suggest-check"
LABEL_CLASS = "drocat-suggest-label"
HINT_CLASS = "drocat-suggest-hint"

# Hover text for a marked row. The tick alone does not say that the next click
# takes the chip back out, and a history row now has a prune "x" beside it that
# removes something else entirely.
MARKED_ROW_TITLE = "Already added — click again to remove"


def chip_keys(chipped: Iterable[str]) -> Set[str]:
    """The stripped, non-empty spellings a field currently holds.

    Every mark/deselect decision in both surfaces compares through this, so a
    chip stored as ``"PPL101 "`` ticks the ``"PPL101"`` row and clicking that
    row is the same action as clicking the chip's own ``x``.
    """
    return {
        str(value).strip() for value in (chipped or []) if str(value).strip()
    }


def marked_rows(
    entries: Iterable[Tuple[str, str]],
    chipped: Sequence[str],
    *,
    limit: int = SUGGESTION_LIMIT,
) -> List[List]:
    """Normalize ``(value, hint)`` entries into capped ``[value, hint, marked]`` rows.

    A value that is already in the field keeps its row and comes back marked,
    rather than being filtered out: dropping it would move the rows under the
    pointer partway through a run of picks. Marked means tinted and ticked, and
    clicking such a row takes its chip back out -- the list works as a
    multi-select checkbox column, in the type-ahead list and in the
    Recent/Frequent history alike.
    """
    taken = chip_keys(chipped)
    return [
        [str(value), str(hint or ""), str(value).strip() in taken]
        for value, hint in list(entries)[:limit]
    ]


def chip_is_marked(chipped: Sequence[str], value: str) -> bool:
    """Whether a clicked row stands for a chip the field already holds."""
    text = str(value or "").strip()
    return bool(text) and text in chip_keys(chipped)


def without_chip(chipped: Sequence[str], value: str) -> List[str]:
    """The chips with one entry for ``value`` removed, in their own spelling.

    Removes the first whitespace-equal match only, which is what a
    ``add-unique`` chip list can hold anyway. Returns the input unchanged when
    nothing matches, so a row that was marked when it rendered but has since
    lost its chip (a paste, a clear, another tab's edit) cannot delete an
    unrelated entry.
    """
    chips = list(chipped or [])
    text = str(value or "").strip()
    if not text:
        return chips
    for index, chip in enumerate(chips):
        if str(chip).strip() == text:
            return chips[:index] + chips[index + 1:]
    return chips
