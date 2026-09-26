"""Round-6 Windows feedback fixes that need a live socket/child process:

- F-N1: banc_public_data.http_get resumes a dropped download with a
  ``Range`` continuation instead of restarting from byte 0.
- F-D2: the install verifier decodes probe output as UTF-8 and reports a
  dead reader as a probe failure, never as ``TypeError``.
"""

import http.server
import socketserver
import sys
import threading
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import banc_public_data as bpd  # noqa: E402


BODY = b"x" * 100_000


class _TruncatingHandler(http.server.BaseHTTPRequestHandler):
    requests = []  # reset per test via fixture

    def do_GET(self):  # noqa: N802 — http.server API
        rng = self.headers.get("Range")
        type(self).requests.append(rng)
        if rng is None:
            # serve half the body, then drop the connection mid-stream
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            self.wfile.write(BODY[:50_000])
            self.wfile.flush()
            self.connection.close()
            return
        start = int(rng.split("=")[1].split("-")[0])
        self.send_response(206)
        self.send_header("Content-Range",
                         f"bytes {start}-{len(BODY) - 1}/{len(BODY)}")
        self.send_header("Content-Length", str(len(BODY) - start))
        self.end_headers()
        self.wfile.write(BODY[start:])

    def log_message(self, *args):  # silence the test log
        pass


def test_http_get_resumes_after_truncation():
    _TruncatingHandler.requests = []
    srv = socketserver.TCPServer(("127.0.0.1", 0), _TruncatingHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        data = bpd.http_get(f"http://127.0.0.1:{port}/big",
                            attempts=3, note="test")
    finally:
        srv.shutdown()
        srv.server_close()
    assert data == BODY
    # the retry must CONTINUE from the salvaged partial, not restart:
    # exactly one full request, then one Range continuation from 50_000
    assert _TruncatingHandler.requests == [None, "bytes=50000-"], (
        _TruncatingHandler.requests)


def test_http_get_returns_none_on_404():
    class _NotFound(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(404)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = socketserver.TCPServer(("127.0.0.1", 0), _NotFound)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert bpd.http_get(f"http://127.0.0.1:{port}/gone") is None
    finally:
        srv.shutdown()
        srv.server_close()


def test_verifier_probe_decodes_utf8_and_survives_dead_reader():
    """F-D2: the atomic-write probe's subprocess must decode as UTF-8 and
    the membership test must tolerate a None stdout (round-6: the parent
    decoded cp936, the reader thread died, and 'x' in None raised)."""
    verifier_path = (Path(__file__).resolve().parents[2]
                     / "skills" / "drocat-install" / "scripts"
                     / "verify_install.py")
    source = verifier_path.read_text(encoding="utf-8")
    # every text=True subprocess declares its decoding (line-based: the
    # encoding rides on the same line as text=True)
    text_lines = [l for l in source.splitlines() if 'text=True' in l]
    assert text_lines, "no probe subprocess found"
    for line in text_lines:
        assert "encoding='utf-8'" in line, line
    # the membership test is None-safe
    assert '"atomic-ok" in stdout_text' in source
    assert 'proc.stdout or ""' in source or "stdout_text = proc.stdout or" in source
