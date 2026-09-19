"""Lint: every text-mode writer in the product must pin ``encoding=``.

Finding F7 of the 2026-09-18 Windows retest: text files were written with the
platform default codec, so on a zh-CN Windows host (GBK / cp936) output was
either mojibake, unreadable by the UTF-8-pinned readers, or a hard
``UnicodeEncodeError`` for payloads carrying glyphs GBK cannot represent
(``→ ✓ ✗`` and every emoji).  Whether a write survives is therefore a
function of the *machine*, not of the code -- which makes it exactly the
kind of bug a static check can retire.

Rule (AST walk over ``src/``, ``scripts/`` and ``ui/``):
  a text-mode write -- ``open()``/``io.open()``/``codecs.open()``/
  ``Path.open(mode=...)`` with ``w``/``a``/``x``/``+`` and no ``b``, or any
  ``Path.write_text()`` -- must pass ``encoding=`` explicitly.

Deliberate non-rules (see ``ALLOWLIST`` and the skip conditions in
``_text_write_calls``): binary writes, ``os.open`` (file-descriptor flags,
not a text mode), and calls whose mode is a runtime value we cannot judge
statically.  Those are skipped rather than allowlisted, so the allowlist
stays a short list of real, documented exceptions.
"""

import ast
import io
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Directories whose product writers must pin an encoding.  ``ui/`` is
#: included on purpose: it is clean today, and pinning it here turns any
#: future regression into a failing test instead of a Windows-only bug.
SCANNED_DIRS = ("src", "scripts", "ui")

#: (relative POSIX path, enclosing function, call kind) -> why it is exempt.
#: Line numbers are deliberately NOT part of the key: they drift on every
#: unrelated edit and a lint that has to be re-anchored is a lint that gets
#: deleted.
ALLOWLIST = {
    ("src/fafb_bundle.py", "__enter__", "open"):
        "POSIX advisory-flock lockfile (.lock sibling).  The handle is only "
        "created and locked -- no text is ever written through it.",
    ("src/morphology.py", "append_vectors", "open"):
        "Same flock lockfile pattern (.parquet.lock); created, locked, "
        "closed.  The text payload next to it (the pending-rows JSON) DOES "
        "pin its encoding and is checked by this lint.",
    ("src/visualize_skeleton.py", "_suppress_output", "open"):
        "Sink for os.devnull used to silence stdout/stderr; nothing is ever "
        "written to it.",
}

_MODE_CHARS = set("rwaxbt+")
_WRITE_CHARS = ("w", "a", "x", "+")
#: Module receivers whose ``.open()`` is the builtin two-argument layout.
_OPEN_MODULES = ("io", "codecs")


def _mode_literal(node):
    """Return the literal open-mode string, or None when undecidable."""
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return None
    mode = node.value
    # Reject anything that is not shaped like an open() mode.  This is what
    # keeps zipfile/gzip/PIL/webbrowser/fitz ``.open(...)`` calls out of the
    # walk: their first argument is a member name or a URL, not a mode.
    if not mode or not _MODE_CHARS.issuperset(mode):
        return None
    return mode


def _call_name(call):
    func = call.func
    if isinstance(func, ast.Name):
        return func.id, None
    if isinstance(func, ast.Attribute):
        owner = func.value
        owner_name = owner.id if isinstance(owner, ast.Name) else None
        return func.attr, owner_name
    return None, None


def _is_two_arg_open_layout(call):
    """True for ``open(...)``/``io.open(...)``/``codecs.open(...)``.

    Those take ``(file, mode, ...)``.  Every other ``.open(...)`` receiver in
    the scanned tree is ``pathlib.Path.open(mode, ...)``, whose mode is the
    *first* positional argument.
    """
    func = call.func
    return (isinstance(func, ast.Name)
            or (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in _OPEN_MODULES))


def _has_encoding(call):
    return any(kw.arg == "encoding" for kw in call.keywords)


def _keyword_mode(call):
    for kw in call.keywords:
        if kw.arg == "mode":
            return _mode_literal(kw.value)
    return None


def _text_write_mode(call, is_bound_method):
    """Mode string for a text write, or None when the call is not one.

    ``is_bound_method`` selects the argument layout: ``pathlib.Path.open()``
    takes the mode as its *first* positional argument, while the builtin
    ``open()`` and ``io.open()``/``codecs.open()`` take it as the second.
    """
    if is_bound_method:
        mode_node = call.args[0] if call.args else None
    else:
        mode_node = call.args[1] if len(call.args) >= 2 else None
    if mode_node is None:
        mode = _keyword_mode(call)
        if mode is None:
            mode = "r"  # both layouts default to a text read
    else:
        mode = _mode_literal(mode_node)
        if mode is None:
            return None  # non-literal / non-mode argument: undecidable
    if "b" in mode:
        return None      # binary: encoding is not applicable
    if not any(c in mode for c in _WRITE_CHARS):
        return None      # read-only
    return mode


def _enclosing_function(tree, lineno):
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= lineno and (
                    best is None or node.lineno > best.lineno):
                best = node
    return best.name if best else "<module>"


def _text_write_calls(source):
    """Yield (lineno, function, kind) for text writes missing ``encoding``."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name, owner = _call_name(node)
        if name is None:
            continue
        if name == "open":
            if owner == "os":
                continue            # os.open(): O_* flags, not a text mode
            mode = _text_write_mode(node, not _is_two_arg_open_layout(node))
            if mode is None:
                continue
            kind = "open"
        elif name == "write_text":
            # Path.write_text() is always a text write.
            kind = "write_text"
        else:
            continue
        if _has_encoding(node):
            continue
        func = _enclosing_function(tree, node.lineno)
        yield node.lineno, func, kind


def _iter_python_files():
    for directory in SCANNED_DIRS:
        base = PROJECT_ROOT / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path.relative_to(PROJECT_ROOT).as_posix(), path


def _scan():
    """Return (violations, allowlisted, files_scanned)."""
    violations = []
    allowlisted = []
    scanned = 0
    for rel, path in _iter_python_files():
        try:
            source = io.open(path, encoding="utf-8").read()
        except (OSError, UnicodeDecodeError) as exc:  # pragma: no cover
            violations.append((rel, 0, "<unreadable>", "scan", str(exc)))
            continue
        scanned += 1
        for lineno, func, kind in _text_write_calls(source):
            entry = (rel, func, kind)
            snippet = source.splitlines()[lineno - 1].strip()
            if entry in ALLOWLIST:
                allowlisted.append((rel, lineno, func, kind))
            else:
                violations.append((rel, lineno, func, kind, snippet))
    return violations, allowlisted, scanned


SCAN = _scan()


def _format(violations):
    lines = [
        "Text-mode writer(s) without an explicit encoding -- on a zh-CN "
        "Windows host these silently use GBK and then crash or emit output "
        "the UTF-8 readers cannot parse (retest F7):",
    ]
    for rel, lineno, func, kind, snippet in violations:
        lines.append(f"  {rel}:{lineno} in {func}() [{kind}]\n"
                     f"      {snippet}")
    lines.append("")
    lines.append("  Fix: pass encoding='utf-8' (UTF-8 is the encoding every "
                 "reader in this repo pins; see ui/output_guide.py).")
    lines.append("  Intentionally exempt: extend ALLOWLIST in this file with "
                 "a documented reason.")
    return "\n".join(lines)


def test_scanned_directories_are_present():
    """Guard against the lint passing because it silently found nothing."""
    assert all((PROJECT_ROOT / d).is_dir() for d in SCANNED_DIRS), (
        f"expected {'/'.join(SCANNED_DIRS)} to exist for the encoding lint")
    assert SCAN[2] >= 100, (
        f"only {SCAN[2]} Python files scanned -- the walk is broken, not "
        "clean")


def test_every_text_writer_pins_encoding():
    violations = SCAN[0]
    assert not violations, _format(violations)


def test_allowlist_entries_are_still_needed():
    """Every exemption must still be suppressing a real hit.

    Without this, an entry would silently rot into dead weight the moment
    the exempted call grew an encoding (or disappeared).
    """
    covered = {(rel, func, kind) for rel, _, func, kind in SCAN[1]}
    unused = set(ALLOWLIST) - covered
    assert not unused, (
        "ALLOWLIST entries no longer match any writer; delete them: "
        + ", ".join(sorted("/".join(e[:2]) + " " + e[2] for e in unused)))


def test_colabeling_report_declares_a_charset():
    """Regression guard for the HTML half of retest F7.

    The co-labeling report is written as UTF-8; with no charset declaration
    a browser sniffing the file on a GBK Windows host still renders it as
    mojibake, so the meta tag is part of the fix.
    """
    source = io.open(PROJECT_ROOT / "src" / "neuronbridge_finder.py",
                     encoding="utf-8").read()
    start = source.index("def _generate_colabeling_report")
    report_body = source[start:source.index("def find_lines_batch", start)]
    assert "<head>" in report_body
    head = report_body[report_body.index("<head>"):report_body.index("</head>")]
    assert re.search(r"<meta\s+charset=[\"']?utf-8", head, re.IGNORECASE), (
        "generated report <head> lost its charset declaration")
    assert "\ufffd" not in source, (
        "a U+FFFD replacement character is mojibake baked into the source "
        "(it was introduced by a default-codec round trip)")


def test_tier_a_glyphs_crash_the_platform_codec(tmp_path):
    """Proves the Tier A fixes were crash fixes, not cosmetics.

    ``summary.txt`` (``run_random_control_test``) and the co-labeling HTML
    report carry these glyphs.  None of them exist in GBK, the default codec
    of a zh-CN Windows host, so the pre-fix writers raised
    ``UnicodeEncodeError`` and produced no output file at all.
    """
    crash_payload = "✓ ✗ ↔ 🔬 📊 📈"
    for glyph in crash_payload.split():
        with pytest.raises(UnicodeEncodeError):
            glyph.encode("gbk")
    hostile = tmp_path / "summary.txt"
    with pytest.raises(UnicodeEncodeError):
        with open(hostile, "w", encoding="gbk") as handle:
            handle.write(crash_payload)
    assert not hostile.read_bytes().endswith(crash_payload.encode("utf-8"))
    # ... and the pinned writer round-trips byte-for-byte.
    with open(hostile, "w", encoding="utf-8") as handle:
        handle.write(crash_payload)
    assert hostile.read_bytes() == crash_payload.encode("utf-8")


def test_tier_b_glyphs_round_trip_corrupt():
    """Why the README/parameters writers mattered even without a crash.

    Box-drawing and math glyphs ARE representable in GBK, so the pre-fix
    write "succeeded" -- it just emitted bytes that are not valid UTF-8,
    which the UTF-8-pinned readers (``ui/output_guide.py`` reads
    ``parameters.txt`` as UTF-8) then reject or garble.
    """
    glyphs = "─ │ └ ├ ± ≤ ≥ × —"
    for glyph in glyphs.split():
        raw = glyph.encode("gbk")            # platform default on zh-CN Windows
        assert glyph.encode("utf-8") != raw or glyph.isascii()
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            decoded = None
        assert decoded != glyph, (
            f"{glyph!r} round-trips identically in both codecs; it is no "
            "longer evidence for the Tier B class -- re-derive the list")
