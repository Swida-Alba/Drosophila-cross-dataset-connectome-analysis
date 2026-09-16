"""Console-encoding guard (Defect B from the 2026-09-12 Windows report).

On legacy Windows code pages (GBK/cp1252) with redirected output, printing
the Unicode status symbols used throughout DROCAT raises
``UnicodeEncodeError``.  ``ensure_utf8_stdio`` reconfigures the standard
streams so the symbols survive (or degrade to replacement characters)
instead of aborting the run.
"""

import io
import sys

from utils.console_encoding import _ensure_utf8_stream, ensure_utf8_stdio


def _make_stream(encoding):
    raw = io.BytesIO()
    return io.TextIOWrapper(raw, encoding=encoding, errors="strict",
                            write_through=True)


def test_non_utf8_stream_is_reconfigured():
    stream = _make_stream("cp936")
    _ensure_utf8_stream(stream)
    assert stream.encoding.lower().replace("-", "") == "utf8"
    # The symbol that crashed the GBK console in the report now encodes.
    stream.write("⚠️ done ✓")
    stream.flush()


def test_utf8_stream_reconfiguration_is_harmless():
    stream = _make_stream("utf-8")
    _ensure_utf8_stream(stream)
    assert stream.encoding.lower().replace("-", "") == "utf8"


def test_unsupported_stream_is_left_alone():
    class Bare:
        pass

    bare = Bare()
    _ensure_utf8_stream(bare)  # must not raise
    assert not hasattr(bare, "reconfigure")


def test_none_stream_is_ignored():
    _ensure_utf8_stream(None)  # must not raise


def test_ensure_utf8_stdio_covers_swapped_streams(monkeypatch):
    out, err = _make_stream("cp1252"), _make_stream("cp1252")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    ensure_utf8_stdio()
    assert out.encoding.lower().replace("-", "") == "utf8"
    assert err.encoding.lower().replace("-", "") == "utf8"


def test_ensure_utf8_stdio_survives_none_streams(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    ensure_utf8_stdio()  # must not raise
