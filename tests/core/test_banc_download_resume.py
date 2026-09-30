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


def test_consolidation_temps_use_the_reclaimable_naming():
    """R7-2: the consolidation/profile writers must use the temp-sibling
    scheme (.{name}.{kind}.{pid}.tmp) that remove_stale_temp_files can
    reclaim — the naked 'path + .tmp' naming leaked a permanent orphan on
    Windows and its os.replace raced an existing temp into WinError 183."""
    import re
    import utils.parquet_utils as pu
    import comparison.connectivity_profiler as cp
    import coana
    import fafb_bundle
    import FAFB_file_converter

    src = Path(cp.__file__).read_text(encoding="utf-8")
    # the two consolidation writes name their temps via temp_sibling AND
    # pre-sweep their own kind: the loader's 'profile-cache' sweep cannot
    # see a crashed consolidation temp, so without the matching-kind sweep
    # the heaviest writer leaked a permanent orphan
    assert src.count("'profile-consolidation'") >= 4
    # the swaps overwrite atomically on Windows too (Path.rename raises
    # FileExistsError there) and ride out a briefly-locked destination
    # through replace_with_retry rather than failing immediately
    assert src.count("replace_with_retry(temp_path, main_cache_path)") >= 2
    assert src.count("replace_with_retry(temp_file, batch_file)") >= 2
    assert src.count("replace_with_retry(temp_path, cache_path)") >= 1
    # orphaned per-neuron batch temps are reclaimed before the batch-dir
    # rmdir so they cannot wedge the directory open
    assert src.count(
        "remove_stale_temp_files_in_dir(batch_dir, 'profile-batch')") >= 2
    # the naked '.parquet.tmp' naming survives in exactly ONE place: the
    # reader-side legacy-orphan cleanup in _load_cache_dataframe. A second
    # occurrence would be a writer regressing to the non-reclaimable name.
    assert src.count("with_suffix('.parquet.tmp')") == 1

    fafb_src = Path(FAFB_file_converter.__file__).read_text(encoding="utf-8")
    # the table converter reclaims its temps even on the reuse path
    assert "reclaim_writer_temps(save_path, 'fafb-table')" in fafb_src

    for module in (coana, fafb_bundle, FAFB_file_converter):
        # strip comments: the pattern lives in CODE, and the fix's own
        # comments legitimately mention the naked form
        code_lines = [l for l in Path(module.__file__)
                      .read_text(encoding="utf-8").splitlines()
                      if not l.strip().startswith('#')]
        code = '\n'.join(code_lines)
        for pattern in ("+ '.tmp'", '+ ".tmp"', 'name + ".tmp"'):
            assert pattern not in code, (module.__name__, pattern)

    # the reclaimable naming actually round-trips through the reclaimer
    import os
    import tempfile
    work = Path(tempfile.mkdtemp())
    probe = work / "_r7_probe_target.parquet"
    temp = Path(pu.temp_sibling(str(probe), "probe"))
    try:
        temp.write_bytes(b"x")
        assert pu._temp_pid(temp.name,
                            "._r7_probe_target.parquet.probe.") is not None
    finally:
        temp.unlink(missing_ok=True)
        work.rmdir()


def test_temp_reclamation_round_trips_across_name_formats():
    """R7-2 finish: the reclaimer must match the CURRENT temp names
    (.{name}.{kind}.{pid}.{thread}.tmp) and the pre-thread scheme
    (.{name}.{kind}.{pid}.tmp); a live writer's temp is never touched."""
    import os
    import shutil
    import tempfile
    import utils.parquet_utils as pu

    dead_pid = 999_999_99  # above every platform's pid ceiling
    work = Path(tempfile.mkdtemp())
    try:
        final = work / "connectivity_profiles.parquet"
        final.write_bytes(b"final")
        current = work / (
            f".connectivity_profiles.parquet.profile-consolidation."
            f"{dead_pid}.12345.tmp")
        current.write_bytes(b"orphan")
        pre_thread = work / (
            f".connectivity_profiles.parquet.profile-consolidation."
            f"{dead_pid}.tmp")
        pre_thread.write_bytes(b"orphan")
        live = Path(pu.temp_sibling(str(final), "profile-consolidation"))
        live.write_bytes(b"in-progress")

        pu.remove_stale_temp_files(final, "profile-consolidation")
        assert not current.exists(), "current-scheme orphan not reclaimed"
        assert not pre_thread.exists(), "pre-thread-scheme orphan not reclaimed"
        assert live.exists(), "own in-progress temp was reclaimed"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_batch_dir_temp_reclaim_is_selective():
    """The kind-only directory sweep reclaims ORPHANED batch temps
    (whatever their final name was, dead writer) and keeps live ones and
    the batch finals — the production emptying of the directory happens
    because consolidation unlinks the finals first, then this sweep clears
    the orphans so ``rmdir()`` can succeed."""
    import os
    import shutil
    import tempfile
    import utils.parquet_utils as pu

    dead_pid = 999_999_99
    work = Path(tempfile.mkdtemp())
    batch_dir = work / "profile_batches"
    batch_dir.mkdir()
    try:
        (batch_dir / "123.parquet").write_bytes(b"b")
        (batch_dir / "456.parquet").write_bytes(b"b")
        orphan = batch_dir / f".789.parquet.profile-batch.{dead_pid}.7.tmp"
        orphan.write_bytes(b"orphan")
        live = batch_dir / (
            f".321.parquet.profile-batch.{os.getpid()}.{threading.get_ident()}.tmp")
        live.write_bytes(b"in-progress")

        pu.remove_stale_temp_files_in_dir(batch_dir, "profile-batch")
        assert not orphan.exists()
        assert live.exists()
        assert (batch_dir / "123.parquet").exists()
        assert (batch_dir / "456.parquet").exists()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_reclaim_writer_temps_removes_only_aged_legacy_naked_temps():
    """The legacy naked '{final}.tmp' sibling is reclaimed by AGE only:
    an aged orphan from the pre-R7-2 writers goes, a young one (a live
    unmigrated writer's in-flight temp) stays."""
    import os
    import shutil
    import tempfile
    import time
    import utils.parquet_utils as pu

    work = Path(tempfile.mkdtemp())
    try:
        final = work / "connections.parquet"
        final.write_bytes(b"final")
        young = work / "connections.parquet.tmp"
        young.write_bytes(b"in-flight")
        aged = work / "other.parquet"
        aged_tmp = work / "other.parquet.tmp"
        aged_tmp.write_bytes(b"orphan")
        old = time.time() - (7 * 3600)
        os.utime(aged_tmp, (old, old))

        pu.reclaim_writer_temps(aged, "fafb-table")
        assert not aged_tmp.exists(), "aged legacy orphan not reclaimed"
        pu.reclaim_writer_temps(final, "fafb-table")
        assert young.exists(), "young naked temp (live writer) was reclaimed"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_temp_sibling_separates_concurrent_threads():
    """Two threads sharing one process must not share a temp path."""
    import os
    import tempfile
    import utils.parquet_utils as pu

    work = Path(tempfile.mkdtemp())
    try:
        names = {}

        def worker(key):
            names[key] = pu.temp_sibling(str(work / "t.parquet"), "probe")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert names[0] != names[1]
        assert all(str(os.getpid()) in name for name in names.values())
    finally:
        work.rmdir()
