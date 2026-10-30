"""Apply DROCAT's arrow patches to the vendored Cytoscape bundle.

Two renderer behaviours break translucent arrows:

1. ``drawArrowhead`` clears a full-alpha ``destination-out`` silhouette
   under every filled arrowhead whose alpha is below 1, then paints the
   coloured arrow over it.  The cut-out stops the edge line showing through
   the arrowhead, but it also erases anything already on the canvas, so
   overlapping translucent arrowheads never blend.  The guard

       1===o&&"hollow"!==u||

   cuts out when opacity < 1 OR the arrow is hollow; the patch keeps the
   cut-out only for hollow arrowheads:

       "hollow"===u&&

2. The triangle arrow's line ``gap`` is ``2 * width * arrow-scale``, while
   the drawn triangle is ``arrowShapeWidth * getArrowWidth(width,
   arrow-scale)`` long (about ``0.3 * getArrowWidth``).  The gap is
   therefore several pixels shorter than the head, so the shaft runs into
   the arrowhead; with the cut-out removed that overlap shows as a darker
   stripe inside the translucent triangle.  The patch gives the triangle a
   ``gap`` equal to its actual drawn length, so the shaft ends exactly at
   the head's base:

       gap:function(e){return t.arrowShapeWidth*t.getArrowWidth(...)}

The bundle is a vendor file, so the patches are applied by this script and
verified by ``--check`` in CI (``tests/vispath/test_cytoscape_arrow_blend_patch.py``).
Upgrading Cytoscape means re-vendoring the new bundle and reworking this
script against its new arrowhead guard and arrow geometry.

Usage:
    python scripts/patch_cytoscape_arrow_blend.py          # patch
    python scripts/patch_cytoscape_arrow_blend.py --check  # CI gate
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = PROJECT_ROOT / "src" / "assets" / "cytoscape" / "cytoscape.min.js"

#: Upstream cytoscape 3.28.1 as vendored 2026-10-02 (see src/assets/VERSIONS.md).
PRISTINE_SHA256 = "92d752b48ea949720675865197fd2a0001c95bc5888545e990af60321712d4c6"
ORIGINAL_GUARD = '1===o&&"hollow"!==u||'
PATCHED_GUARD = '"hollow"===u&&'
TRIANGLE_ORIGINAL = 's("triangle",{points:[-.15,-.3,0,0,.15,-.3]})'
TRIANGLE_PATCHED = (
    's("triangle",{points:[-.15,-.3,0,0,.15,-.3],gap:function(e){'
    "return t.arrowShapeWidth*t.getArrowWidth("
    'e.pstyle("width").pfValue,e.pstyle("arrow-scale").value)}})'
)
#: The first vendored patch line; upgraded in place by the geometry patch.
OLD_MARKER = "/* DROCAT patch: arrow fill blends; hollow keeps cut-out " "(Cytoscape 3.28.1) */\n"
MARKER = (
    "/* DROCAT patch (Cytoscape 3.28.1): blended arrow fill; "
    "shaft ends at the arrowhead base; hollow cut-out kept */\n"
)
#: Hash of the bundle after MARKER + both replacements, pinned so --check
#: also catches unrelated edits to the vendor file.
PATCHED_SHA256 = "65cfafc30915fda27501126a3531d4f5667dadd3864261841de9923ce6b5a77f"


def _normalized(text: str) -> str:
    """Hash/write form: LF newlines, so a CRLF checkout still matches."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _replace_once(text: str, original: str, patched: str, label: str) -> str:
    count = text.count(original)
    if count != 1:
        raise SystemExit(
            f"{BUNDLE}: expected exactly one {label}, found {count}; "
            "the bundle is not the pinned cytoscape 3.28.1"
        )
    return text.replace(original, patched)


def patch_text(text: str) -> str:
    """Return the fully patched bundle text (idempotent)."""
    text = _normalized(text)
    if text.startswith(OLD_MARKER):
        text = MARKER + text.removeprefix(OLD_MARKER)
    if text.startswith(MARKER):
        if ORIGINAL_GUARD in text or PATCHED_GUARD not in text:
            raise SystemExit(
                f"{BUNDLE}: marker present but guard does not match the "
                "patched form; re-vendor the bundle and rework the patch"
            )
        if TRIANGLE_ORIGINAL in text:
            return _replace_once(
                text, TRIANGLE_ORIGINAL, TRIANGLE_PATCHED, "triangle arrow definition"
            )
        if TRIANGLE_PATCHED not in text:
            raise SystemExit(
                f"{BUNDLE}: marker present but triangle arrow is neither the "
                "original nor the patched form; rework the patch"
            )
        return text
    text = _replace_once(text, ORIGINAL_GUARD, PATCHED_GUARD, "arrowhead opacity guard")
    text = _replace_once(text, TRIANGLE_ORIGINAL, TRIANGLE_PATCHED, "triangle arrow definition")
    return MARKER + text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the bundle is patched (and unmodified); write nothing",
    )
    args = parser.parse_args(argv)

    if not BUNDLE.exists():
        raise SystemExit(f"missing vendored bundle: {BUNDLE}")
    raw = BUNDLE.read_text(encoding="utf-8")
    text = _normalized(raw)
    before = _sha256(text)

    if args.check:
        try:
            patched = patch_text(text)
        except SystemExit as exc:
            print(exc, file=sys.stderr)
            return 1
        if patched != text:
            print(
                f"{BUNDLE}: patch set incomplete (run the script); " f"current SHA-256 {before}",
                file=sys.stderr,
            )
            return 1
        if before != PATCHED_SHA256:
            print(
                f"{BUNDLE}: patched but SHA-256 {before} != recorded " f"{PATCHED_SHA256}",
                file=sys.stderr,
            )
            return 1
        print(f"OK: patched bundle, SHA-256 {before}")
        return 0

    if not text.startswith(MARKER) and not text.startswith(OLD_MARKER):
        if before != PRISTINE_SHA256:
            raise SystemExit(
                f"{BUNDLE}: SHA-256 {before} is not the pinned pristine "
                f"{PRISTINE_SHA256}; refusing to patch an unknown bundle"
            )

    patched = patch_text(text)
    after = _sha256(patched)
    if after != PATCHED_SHA256:
        raise SystemExit(
            f"patched SHA-256 {after} != recorded {PATCHED_SHA256}; "
            "update the pin only after reviewing the change"
        )
    if patched != text:
        BUNDLE.write_text(patched, encoding="utf-8", newline="\n")
        print(f"patched {BUNDLE}")
    else:
        print(f"already patched {BUNDLE}")
    print(f"patched SHA-256: {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
