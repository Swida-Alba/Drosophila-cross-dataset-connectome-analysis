"""Plan 2026-09-07 Phase G: documentation/skill consistency greps.

Guards the pathfinding docs and skills against stale enumerator names, the
old raw bodyId visualization filenames presented as produced output, and
missing statements about the implemented untyped filter.
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The ui_guides migration (8b96cfe) moved the editable page source to
# docs/ui_guides/src/ and left only the built .html at the old path, so a guard
# that names a prose surface must read the .md; the .html is listed separately
# where the *published* page is what the claim is about. Every list is a module
# constant so test_every_listed_surface_exists can census it — these paths are
# strings, so a migration silently deletes a guard's coverage instead of
# reporting a stale list.
DROP_UNTYPED_FILES = [
    "docs/ui_guides/find_path.html",
    "docs/ui_guides/find_shortest.html",
    "docs/ui_guides/src/find_shortest.md",
    "docs/ui_guides/cross_dataset.html",
    "docs/core-features/FindAllPath_Documentation.md",
    "docs/OUTPUT_FILES.md",
    "docs/technical/PATHFINDING_PIPELINE.md",
    "skills/drocat-usage/tabs/find-path.md",
    "skills/drocat-usage/tabs/find-shortest.md",
    "skills/drocat-usage/tabs/inter-dataset.md",
    "skills/drocat-backend/modules/comparison.md",
]

# The stale name must never be presented as the pipeline's enumerator: any
# occurrence must carry an explicit not-used clarification nearby.
STALE_ENUMERATOR_FILES = [
    "docs/technical/PATHFINDING_PIPELINE.md",
    "docs/technical/STRONGEST_FIRST_PATHFINDING.md",
    "docs/ui_guides/src/find_shortest.md",
    "docs/ui_guides/find_shortest.html",
    "docs/ui_guides/pathfinding_algorithms.html",
    "docs/core-features/PathFinding_Methods.md",
    "skills/drocat-usage/tabs/find-shortest.md",
    "skills/drocat-backend/modules/vispath.md",
    "src/coana.py",
]

ACTUAL_ENUMERATOR_FILES = {
    "src/coana.py": "find_paths_shortest_strongest_first",
    "docs/ui_guides/src/find_shortest.md":
        "find_paths_shortest_strongest_first",
    "skills/drocat-usage/tabs/find-shortest.md":
        "find_paths_shortest_strongest_first",
}

PROVENANCE_FILES = (
    "docs/OUTPUT_FILES.md",
    "docs/technical/PATHFINDING_PIPELINE.md",
    "docs/technical/STRONGEST_FIRST_PATHFINDING.md",
    "skills/drocat-usage/tabs/find-path.md",
)

SHORTEST_NEVER_FLOORED_FILES = (
    "docs/ui_guides/src/find_shortest.md",
    "docs/technical/PATHFINDING_PIPELINE.md",
    "skills/drocat-usage/tabs/find-shortest.md",
)

NOT_USED_MARKERS = ("not used", "not called", "not what the",
                    "does not call", "does not use", "never called",
                    "uncalled", "isn't called", "not the enumerator")


def _read(rel):
    return (PROJECT_ROOT / rel).read_text(encoding="utf-8")


def _listed_surface_paths():
    """Every path literal these guards read, de-duplicated, in list order."""
    seen = []
    for group in (DROP_UNTYPED_FILES, STALE_ENUMERATOR_FILES,
                  list(ACTUAL_ENUMERATOR_FILES), PROVENANCE_FILES,
                  SHORTEST_NEVER_FLOORED_FILES):
        for rel in group:
            if rel not in seen:
                seen.append(rel)
    return seen


def test_every_listed_surface_exists():
    missing = [rel for rel in _listed_surface_paths()
               if not (PROJECT_ROOT / rel).is_file()]
    assert not missing, (
        "stale doc-surface list — a migration moved or deleted these paths, "
        f"so the guards below silently stopped covering them: {missing}")


def test_drop_untyped_documented_across_surfaces():
    for rel in DROP_UNTYPED_FILES:
        text = _read(rel)
        assert "drop_untyped" in text or "Drop Untyped" in text, rel


def test_stale_enumerator_only_with_not_used_note():
    for rel in STALE_ENUMERATOR_FILES:
        text = _read(rel)
        for match in re.finditer(r"find_paths_shortest_backward", text):
            context = text[max(0, match.start() - 400):
                           match.end() + 400]
            # strip markdown emphasis/backticks and collapse whitespace so
            # "is **not**\ncalled" still matches the marker set
            context = re.sub(r"\s+", " ", re.sub(r"[*_`]", "", context))
            context = context.lower()
            assert any(marker in context for marker in NOT_USED_MARKERS), (
                f"{rel}: stale enumerator mention without a not-used note")


def test_actual_enumerator_named():
    for rel, needle in ACTUAL_ENUMERATOR_FILES.items():
        assert needle in _read(rel), rel


def test_no_raw_bodyid_viz_names_as_produced_output():
    """The legacy raw names may only appear in explicitly historical
    (legacy / no longer produced) contexts."""
    candidates = list((PROJECT_ROOT / "docs").rglob("*.md")) + \
        list((PROJECT_ROOT / "docs").rglob("*.html")) + \
        list((PROJECT_ROOT / "skills").rglob("*.md")) + \
        list((PROJECT_ROOT / "ui").rglob("*.py"))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "bodyId_visualization_network" not in text:
            continue
        context = text.lower()
        historical = ("no longer" in context or "legacy" in context
                      or "historical" in context)
        assert historical, (
            f"{path.relative_to(PROJECT_ROOT)}: raw bodyId visualization "
            "name presented without a legacy/no-longer-produced note")


def test_provenance_fields_documented():
    for rel in PROVENANCE_FILES:
        text = _read(rel)
        for key in ("applied_threshold", "applied_threshold_source",
                    "strongest_retained_bottleneck", "paths_complete"):
            assert key in text, f"{rel}: {key}"


def test_shortest_never_floored_statement():
    for rel in SHORTEST_NEVER_FLOORED_FILES:
        text = _read(rel).lower()
        assert ("never" in text and "floor" in text), rel


def test_shortest_tab_default_depth_documented():
    text = _read("skills/drocat-usage/tabs/find-shortest.md")
    assert "max_interlayer=2" not in text
    assert "5" in text
