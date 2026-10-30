"""The vendored Cytoscape bundle carries the arrow-blend patch.

Cytoscape 3.28.1 clears a full-alpha ``destination-out`` silhouette under
every filled arrowhead whose opacity is below 1.  That cut-out stops the
edge line showing through a translucent arrow, but it also erases whatever
is already on the canvas, so translucent arrowheads never composite and the
exported arrow cannot blend with its own shaft.  ``drawArrowhead``'s guard

    1===o&&"hollow"!==u||

runs the cut-out when the arrow is translucent OR hollow; the DROCAT patch
keeps it only for hollow arrows:

    "hollow"===u&&

``scripts/patch_cytoscape_arrow_blend.py`` applies the change (idempotent,
SHA-256-guarded); this file is the CI gate that notices an unpatched or
silently re-vendored bundle.

Run:
    python -m pytest tests/vispath/test_cytoscape_arrow_blend_patch.py -q
"""

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = PROJECT_ROOT / "src" / "assets" / "cytoscape" / "cytoscape.min.js"
SCRIPT = PROJECT_ROOT / "scripts" / "patch_cytoscape_arrow_blend.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("patch_cytoscape_arrow_blend", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalized(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_bundle_is_patched_and_pinned():
    patch = _load_script()
    text = _normalized(BUNDLE.read_text(encoding="utf-8"))

    assert text.startswith(patch.MARKER), "bundle has no DROCAT patch marker"
    assert patch.ORIGINAL_GUARD not in text
    assert text.count(patch.PATCHED_GUARD) == 1
    assert _sha(text) == patch.PATCHED_SHA256


def test_patch_script_check_gate_passes():
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True, timeout=120
    )
    assert res.returncode == 0, res.stderr
    assert "OK: patched bundle" in res.stdout


def test_patch_script_reproduces_the_recorded_hashes():
    patch = _load_script()
    patched = _normalized(BUNDLE.read_text(encoding="utf-8"))
    pristine = patched.removeprefix(patch.MARKER).replace(
        patch.PATCHED_GUARD, patch.ORIGINAL_GUARD, 1
    )
    assert (
        _sha(pristine) == patch.PRISTINE_SHA256
    ), "the recorded pristine hash no longer describes the upstream bundle"

    rebuilt = patch.patch_text(pristine)
    assert _sha(rebuilt) == patch.PATCHED_SHA256
    # Idempotent: a second pass must not add a second marker or guard.
    assert patch.patch_text(rebuilt) == rebuilt
