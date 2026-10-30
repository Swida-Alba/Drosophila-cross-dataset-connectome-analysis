"""Apply DROCAT's arrowhead alpha-blend patch to the vendored Cytoscape bundle.

Cytoscape 3.28.1 draws every filled arrowhead under an opaque white
``destination-out`` silhouette whenever the arrow alpha is below 1 (i.e.
``line-opacity`` < 1), then paints the coloured arrow over it.  The
cut-out stops the edge line showing through the arrowhead, but it also
erases anything already on the canvas, so overlapping translucent
arrowheads never blend and the arrow cannot composite with its own shaft.
A serializer that emits the cut-out also puts an opaque white backing path
in the SVG.  The guard is the single expression

    1===o&&"hollow"!==u||

which cuts out when opacity < 1 OR the arrow is hollow.  The patch keeps
the cut-out only for hollow arrowheads:

    "hollow"===u&&

so filled arrows composite with what is beneath them, while hollow arrows
keep their cleared interior.

The bundle is a vendor file, so the patch is applied by this script and
verified by ``--check`` in CI (``tests/vispath/test_cytoscape_arrow_blend_patch.py``).
Upgrading Cytoscape means re-vendoring the new bundle and reworking this
script against its new arrowhead guard.

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
MARKER = "/* DROCAT patch: arrow fill blends; hollow keeps cut-out (Cytoscape 3.28.1) */\n"
#: Hash of the bundle after MARKER + guard replacement, pinned so --check
#: also catches unrelated edits to the vendor file.
PATCHED_SHA256 = "13bf8acf27da3d4445fa0fb4fea30144a624858840d18259beed85e1cd40bea5"


def _normalized(text: str) -> str:
    """Hash/write form: LF newlines, so a CRLF checkout still matches."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def patch_text(text: str) -> str:
    """Return the patched bundle text (idempotent)."""
    text = _normalized(text)
    if text.startswith(MARKER):
        if ORIGINAL_GUARD in text or PATCHED_GUARD not in text:
            raise SystemExit(
                f"{BUNDLE}: marker present but guard does not match the "
                "patched form; re-vendor the bundle and rework the patch"
            )
        return text
    count = text.count(ORIGINAL_GUARD)
    if count != 1:
        raise SystemExit(
            f"{BUNDLE}: expected exactly one {ORIGINAL_GUARD!r}, found "
            f"{count}; the bundle is not the pinned cytoscape 3.28.1"
        )
    return MARKER + text.replace(ORIGINAL_GUARD, PATCHED_GUARD)


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
            patch_text(text)
        except SystemExit as exc:
            print(exc, file=sys.stderr)
            return 1
        if not text.startswith(MARKER):
            print(f"{BUNDLE}: NOT patched (pristine {before})", file=sys.stderr)
            return 1
        if before != PATCHED_SHA256:
            print(
                f"{BUNDLE}: patched but SHA-256 {before} != recorded " f"{PATCHED_SHA256}",
                file=sys.stderr,
            )
            return 1
        print(f"OK: patched bundle, SHA-256 {before}")
        return 0

    if not text.startswith(MARKER) and before != PRISTINE_SHA256:
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
