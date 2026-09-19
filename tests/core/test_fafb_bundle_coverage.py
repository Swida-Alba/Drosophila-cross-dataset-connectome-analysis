"""Coverage tests for src/fafb_bundle.py.

The sibling test_fafb_bundle.py exercises the happy paths, but it drives the
CLI through ``subprocess`` (whose coverage is not captured in-process) and
mostly uses lazy_convert=False readers, leaving the following uncovered:
the in-process ``_cli`` dispatch, header/footer/index corruption errors,
``_read_header`` (never invoked by the reader), the lazy-conversion buffer
flush + auto-compact thresholds, the ZIP-leftover iteration path, the
compact/close/lock exception fallbacks, the no-multiprocessing pack
progress callback, and the verbatim-compaction retry / best-effort logic.

Hermetic: all bundles/zips are tiny synthetic files under pytest tmp_path.
No multiprocessing is used (pack always runs with n_workers=1).
"""

import io
import struct
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import fafb_bundle as fb  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _swc(nid, n=12):
    lines = [f"# SWC {nid}"]
    for i in range(1, n + 1):
        parent = -1 if i == 1 else i - 1
        lines.append(f"{i} 1 {i * 1000.0 + 0.44} {i * 200.0 + 0.5} "
                     f"{i * 50.0} 2.5 {parent}")
    return "\n".join(lines) + "\n"


def make_zip(tmp_path, ids=(101, 102, 103), name="sk_lod1_783_healed.zip"):
    zip_path = tmp_path / name
    with zipfile.ZipFile(zip_path, "w") as z:
        for nid in ids:
            z.writestr(f"{nid}.swc", _swc(nid))
    return zip_path


def _bundle_path(tmp_path):
    return tmp_path / "sk_lod1_783_healed.zst"


class _RaisingClose:
    def close(self):
        raise OSError("boom")


def _run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["fafb_bundle", *argv])
    return fb._cli()


# ---------------------------------------------------------------------------
# module-level guards and parsing
# ---------------------------------------------------------------------------

def test_require_zstd_raises_when_missing(monkeypatch):
    monkeypatch.setattr(fb, "zstd", None)
    with pytest.raises(ImportError):
        fb._compress_frame(b"payload")


def test_parse_swc_rows_skips_short_lines():
    content = b"# comment\n1 1 0 0\n2 1 1.0 2.0 3.0 0.5 -1\n"
    rows = fb._parse_swc_rows(content)
    assert len(rows) == 1
    assert rows[0][0] == 2


# ---------------------------------------------------------------------------
# header / footer / index corruption
# ---------------------------------------------------------------------------

def test_read_header_valid_and_errors(tmp_path):
    zip_path = make_zip(tmp_path)
    bundle_path = _bundle_path(tmp_path)
    fb.pack(zip_path, bundle_path, n_workers=1)
    with open(bundle_path, "rb") as fh:
        header = fb._read_header(fh)
    assert header["node_bytes"] == fb.NODE_BYTES

    bad_magic = tmp_path / "bad_magic.bin"
    bad_magic.write_bytes(b"XXXXXXXX" + b"\x00" * 32)
    with open(bad_magic, "rb") as fh:
        with pytest.raises(fb.BundleCorruptError, match="magic"):
            fb._read_header(fh)

    bad_version = tmp_path / "bad_version.bin"
    with open(bad_version, "wb") as fh:
        fh.write(fb.MAGIC)
        fh.write(struct.pack("<II", 99, fb.HEADER_LEN))
        fh.write(b" " * fb.HEADER_LEN)
    with open(bad_version, "rb") as fh:
        with pytest.raises(fb.BundleCorruptError, match="version"):
            fb._read_header(fh)


def test_read_footer_too_small_and_bad_magic(tmp_path):
    tiny = tmp_path / "tiny.bin"
    tiny.write_bytes(b"\x00" * 10)
    with open(tiny, "rb") as fh:
        with pytest.raises(fb.BundleCorruptError, match="too small"):
            fb._read_footer(fh)

    bad = tmp_path / "badfooter.bin"
    bad.write_bytes(b"\x00" * 200)
    with open(bad, "rb") as fh:
        with pytest.raises(fb.BundleCorruptError, match="footer magic"):
            fb._read_footer(fh)


def test_read_indexes_crc_mismatch(tmp_path):
    zip_path = make_zip(tmp_path)
    bundle_path = _bundle_path(tmp_path)
    fb.pack(zip_path, bundle_path, n_workers=1)
    with open(bundle_path, "rb") as fh:
        footer = fb._read_footer(fh)
        bad_block = dict(footer, crc_block=footer["crc_block"] ^ 0xFFFFFFFF)
        with pytest.raises(fb.BundleCorruptError, match="block index CRC"):
            fb._read_indexes(fh, bad_block)
        bad_neuron = dict(footer, crc_neuron=footer["crc_neuron"] ^ 0xFFFFFFFF)
        with pytest.raises(fb.BundleCorruptError, match="neuron index CRC"):
            fb._read_indexes(fh, bad_neuron)


# ---------------------------------------------------------------------------
# reader lifecycle edge branches
# ---------------------------------------------------------------------------

def test_reload_index_opens_when_handle_missing(tmp_path):
    zip_path = make_zip(tmp_path)
    bundle_path = _bundle_path(tmp_path)
    fb.pack(zip_path, bundle_path, n_workers=1)
    bundle = fb.FAFBSkeletonBundle(bundle_path, lazy_convert=False)
    assert bundle._handle is None
    bundle._reload_index()  # forces _ensure_open
    assert bundle._handle is not None
    bundle.close()


def test_contains_len_and_zip_checks(tmp_path):
    zip_path = make_zip(tmp_path)
    bundle_path = _bundle_path(tmp_path)
    fb.pack(zip_path, bundle_path, n_workers=1)
    bundle = fb.FAFBSkeletonBundle(bundle_path, zip_path=zip_path,
                                   lazy_convert=False)
    try:
        assert bundle.contains(999) is False  # in zip -> True path avoided
        assert bundle.contains(101) is True
        assert len(bundle) == bundle.count()
    finally:
        bundle.close()

    # bundle-only reader: no zip -> contains falls to the False branch
    bundle2 = fb.FAFBSkeletonBundle(bundle_path, lazy_convert=False)
    try:
        assert bundle2._zip() is None
        assert bundle2.contains(101) is True
        assert bundle2.contains(404) is False
        assert list(bundle2.iter_texts())  # zip is None -> early return path
    finally:
        bundle2.close()


def test_iter_texts_yields_zip_leftovers_and_buffers(tmp_path):
    zip_path = make_zip(tmp_path, ids=(101, 102, 103))
    bundle_path = _bundle_path(tmp_path)
    fb.append_entries(zip_path, bundle_path, [101])
    bundle = fb.FAFBSkeletonBundle(bundle_path, zip_path=zip_path,
                                   lazy_convert=True)
    try:
        seen = dict(bundle.iter_texts())
        assert set(seen) == {101, 102, 103}
        assert "2000.4" in seen[102]
    finally:
        bundle.close()


def test_buffer_convert_flushes_when_block_full(tmp_path):
    zip_path = make_zip(tmp_path, ids=(101,))
    bundle_path = _bundle_path(tmp_path)
    # tiny block: 2 nodes max -> a 12-node neuron forces an immediate flush
    bundle = fb.FAFBSkeletonBundle(bundle_path, zip_path=zip_path,
                                   lazy_convert=True, block_bytes=fb.NODE_BYTES * 2)
    try:
        assert bundle.get(101) is not None
        bundle.flush()
        assert bundle.bundle_count() == 1
    finally:
        bundle.close()


def test_buffer_convert_auto_compact_threshold(tmp_path, monkeypatch):
    zip_path = make_zip(tmp_path, ids=(101, 102))
    bundle_path = _bundle_path(tmp_path)
    monkeypatch.setattr(fb, "COMPACT_THRESHOLD_ENTRIES", 1)
    bundle = fb.FAFBSkeletonBundle(bundle_path, zip_path=zip_path,
                                   lazy_convert=True)
    try:
        bundle.get(101)  # buffered then auto-flush+compact threshold path
        bundle.get(102)
        bundle.flush()
    finally:
        bundle.close()


def test_compact_zip_no_zip_and_no_converted(tmp_path):
    bundle_path = _bundle_path(tmp_path)
    bundle = fb.FAFBSkeletonBundle(bundle_path, lazy_convert=False)
    try:
        assert bundle.compact_zip() == 0  # zip_path is None
    finally:
        bundle.close()

    zip_path = make_zip(tmp_path)
    bundle2 = fb.FAFBSkeletonBundle(bundle_path, zip_path=zip_path,
                                    lazy_convert=False)
    try:
        assert bundle2.compact_zip() == 0  # nothing converted yet
    finally:
        bundle2.close()


def test_compact_zip_handles_raising_zip_handle(tmp_path):
    zip_path = make_zip(tmp_path, ids=(101, 102))
    bundle_path = _bundle_path(tmp_path)
    fb.append_entries(zip_path, bundle_path, [101])
    bundle = fb.FAFBSkeletonBundle(bundle_path, zip_path=zip_path,
                                   lazy_convert=False)
    try:
        bundle._zip()  # open the lazy zip handle
        bundle._zip_handle = _RaisingClose()
        removed = bundle.compact_zip()
        assert removed == 1
    finally:
        bundle.close()


def test_close_swallows_exceptions_from_handles(tmp_path):
    bundle_path = _bundle_path(tmp_path)
    make_zip(tmp_path)
    bundle = fb.FAFBSkeletonBundle(bundle_path, lazy_convert=False)
    bundle._ensure_open()

    def _raising_flush():
        raise RuntimeError("flush failed")

    bundle.flush = _raising_flush
    bundle._zip_handle = _RaisingClose()
    bundle._handle = _RaisingClose()
    bundle._mm = _RaisingClose()
    bundle.close()  # must not raise
    assert bundle._handle is None and bundle._mm is None


# ---------------------------------------------------------------------------
# _bundle_file_lock exception fallbacks
# ---------------------------------------------------------------------------

def test_file_lock_enter_falls_back_without_fcntl(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "fcntl", None)
    lock = fb._bundle_file_lock(tmp_path / "bundle.zst")
    with lock:
        assert lock._fd is None


def test_file_lock_exit_swallows_unlock_and_close_errors(
        tmp_path, monkeypatch):
    # flock is POSIX-only: on Windows ``_bundle_file_lock`` deliberately
    # degrades to a no-op, so there is nothing to unlock here.
    fcntl = pytest.importorskip("fcntl")
    lock = fb._bundle_file_lock(tmp_path / "bundle.zst")
    lock.__enter__()
    assert lock._fd is not None
    monkeypatch.setattr(fcntl, "flock",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("x")))
    lock._fd = _RaisingClose()
    assert lock.__exit__(None, None, None) is False


# ---------------------------------------------------------------------------
# _pack_shard branches
# ---------------------------------------------------------------------------

def test_pack_shard_skips_non_swc_empty_and_blocks(tmp_path):
    zip_path = tmp_path / "mixed.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("README.txt", "not a skeleton")      # non .swc -> skip
        z.writestr("201.swc", "# only comment\n")        # no rows -> skip
        z.writestr("202.swc", _swc(202, n=6))
        z.writestr("203.swc", _swc(203, n=6))
    result = fb._pack_shard((str(zip_path), 0, 4, fb.DEFAULT_LEVEL,
                             fb.NODE_BYTES * 4))  # 4-node blocks force flush
    assert result["max_error"] < 0.1
    assert len(result["blocks"]) >= 1


def test_pack_shard_empty_range(tmp_path):
    zip_path = make_zip(tmp_path)
    result = fb._pack_shard((str(zip_path), 0, 0, fb.DEFAULT_LEVEL,
                             fb.BLOCK_BYTES))
    assert result["blocks"] == []


# ---------------------------------------------------------------------------
# pack progress callback (single worker; multiprocessing is not exercised)
# ---------------------------------------------------------------------------

def test_pack_progress_callback_single_worker(tmp_path):
    zip_path = make_zip(tmp_path)
    out = tmp_path / "progress.zst"
    calls = []
    stats = fb.pack(zip_path, out, n_workers=1,
                    progress_callback=lambda done, total: calls.append((done, total)))
    assert stats["neurons"] == 3
    assert calls and calls[-1][0] == stats["blocks"]


# ---------------------------------------------------------------------------
# append_entries branches
# ---------------------------------------------------------------------------

def test_append_entries_skips_empty_rows(tmp_path):
    zip_path = tmp_path / "with_empty.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("301.swc", "# comment only\n")
        z.writestr("302.swc", _swc(302))
    bundle_path = tmp_path / "append_empty.zst"
    appended = fb.append_entries(zip_path, bundle_path, [301, 302, 303])
    assert appended == 1


# ---------------------------------------------------------------------------
# _compact_zip_verbatim branches
# ---------------------------------------------------------------------------

def test_compact_verbatim_missing_zip_and_nothing_to_remove(tmp_path):
    assert fb._compact_zip_verbatim(tmp_path / "absent.zip", {1}) == 0

    zip_path = make_zip(tmp_path, ids=(101,))
    assert fb._compact_zip_verbatim(zip_path, converted=set()) == 0


def test_compact_verbatim_permission_retry_and_raise(tmp_path, monkeypatch):
    zip_path = make_zip(tmp_path, ids=(101, 102))
    monkeypatch.setattr(fb.time, "sleep", lambda *_: None)

    def _always_permission_error(*args, **kwargs):
        raise PermissionError("locked by indexer")

    monkeypatch.setattr(fb.os, "replace", _always_permission_error)
    with pytest.raises(PermissionError):
        fb._compact_zip_verbatim(zip_path, converted={101}, best_effort=False)


def test_compact_verbatim_best_effort_swallows_errors(tmp_path, monkeypatch):
    zip_path = make_zip(tmp_path, ids=(101,))
    monkeypatch.setattr(fb.os, "replace",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("x")))
    assert fb._compact_zip_verbatim(zip_path, converted={101},
                                    best_effort=True) == 0


# ---------------------------------------------------------------------------
# in-process CLI dispatch
# ---------------------------------------------------------------------------

def test_cli_info_and_append_and_compact(tmp_path, monkeypatch, capsys):
    zip_path = make_zip(tmp_path, ids=(101, 102, 103))
    bundle_path = _bundle_path(tmp_path)

    assert _run_cli(monkeypatch, "append", str(zip_path), str(bundle_path),
                    "101", "102") == 0
    assert "appended 2" in capsys.readouterr().out

    assert _run_cli(monkeypatch, "info", str(bundle_path),
                    "--zip", str(zip_path)) == 0
    out = capsys.readouterr().out
    assert '"bundle_neurons": 2' in out and "converted_percent" in out

    assert _run_cli(monkeypatch, "compact", str(zip_path),
                    str(bundle_path)) == 0
    assert "removed 2" in capsys.readouterr().out


def test_cli_pack_and_verify(tmp_path, monkeypatch, capsys):
    zip_path = make_zip(tmp_path)
    out = tmp_path / "cli_pack.zst"
    assert _run_cli(monkeypatch, "pack", str(zip_path), str(out),
                    "--workers", "1") == 0
    assert '"neurons": 3' in capsys.readouterr().out

    assert _run_cli(monkeypatch, "verify", str(out), "--zip", str(zip_path),
                    "--sample", "3") == 0
    assert '"ok": true' in capsys.readouterr().out


def test_cli_pack_delete_source_success(tmp_path, monkeypatch, capsys):
    zip_path = make_zip(tmp_path)
    out = tmp_path / "del_ok.zst"
    monkeypatch.setattr(fb, "verify", lambda *a, **k: {"ok": True})
    assert _run_cli(monkeypatch, "pack", str(zip_path), str(out),
                    "--workers", "1", "--delete-source") == 0
    assert "deleted after successful verify" in capsys.readouterr().out
    assert not zip_path.exists() and out.exists()


def test_cli_pack_delete_source_verify_failure(tmp_path, monkeypatch, capsys):
    zip_path = make_zip(tmp_path)
    out = tmp_path / "del_fail.zst"
    monkeypatch.setattr(fb, "verify", lambda *a, **k: {"ok": False})
    assert _run_cli(monkeypatch, "pack", str(zip_path), str(out),
                    "--workers", "1", "--delete-source") == 1
    assert "verify failed" in capsys.readouterr().out
    assert zip_path.exists()  # kept on failure


def test_cli_verify_failure_returns_nonzero(tmp_path, monkeypatch, capsys):
    zip_path = make_zip(tmp_path)
    bundle_path = _bundle_path(tmp_path)
    fb.pack(zip_path, bundle_path, n_workers=1)
    monkeypatch.setattr(fb, "verify", lambda *a, **k: {"ok": False})
    assert _run_cli(monkeypatch, "verify", str(bundle_path),
                    "--zip", str(zip_path)) == 1


def test_verify_bundle_internal_ids_after_zip_compacted(tmp_path):
    # Once every id has been compacted out of the ZIP, verify degrades those
    # ids to bundle-internal consistency checks (the `not in zip_names` path).
    zip_path = make_zip(tmp_path)
    bundle_path = _bundle_path(tmp_path)
    fb.pack(zip_path, bundle_path, n_workers=1)
    bundle = fb.FAFBSkeletonBundle(bundle_path, zip_path=zip_path,
                                   lazy_convert=False)
    try:
        assert bundle.compact_zip() == 3  # zip now has no .swc entries
    finally:
        bundle.close()
    result = fb.verify(bundle_path, zip_path=zip_path, sample=3)
    assert result["ok"] and result["errors"] == 0
