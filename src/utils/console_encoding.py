"""Console encoding guards for non-UTF-8 terminals (Windows GBK/cp1252).

DROCAT's status output is decorated with Unicode symbols (⚠ ✓ →).  When a
library consumer runs on a legacy Windows code page — most commonly with
stdout redirected to a file or pipe, where Python falls back to the ANSI
code page instead of the console's UTF-8 mode — the first such symbol used
to abort the run with ``UnicodeEncodeError`` (Defect B in the 2026-09-12
Windows test report).  Call :func:`ensure_utf8_stdio` once at module import
to make the standard streams tolerate any character.
"""

import sys


def ensure_utf8_stdio():
    """Reconfigure stdout/stderr to UTF-8 with replacement on encode errors.

    Safe to call multiple times and under any embedding: streams without
    ``reconfigure`` (string buffers, captured streams) and already-UTF-8
    streams are left untouched, and failures are swallowed — this guard
    must never be the reason a run fails.
    """
    for stream in (sys.stdout, sys.stderr):
        _ensure_utf8_stream(stream)


def _ensure_utf8_stream(stream):
    if stream is None:
        return
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
