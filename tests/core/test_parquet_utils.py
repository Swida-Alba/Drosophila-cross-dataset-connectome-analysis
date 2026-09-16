"""Tests for the profile-driven lossless parquet re-encoder."""
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pyarrow as pa
import pyarrow.parquet as pq

import utils.parquet_utils as pu
from utils.parquet_utils import (
    LOSSLESS_MARKER_KEY,
    LOSSLESS_MARKER_VALUE,
    atomic_replace,
    parquet_lossless_marker,
    parquet_readable,
    reencode_parquet_lossless,
    remove_stale_temp_files,
    temp_sibling,
    write_parquet_atomic,
)

BANC_LIKE_PROFILE = {
    "column_encoding": {
        "pre_root_id": "DELTA_BINARY_PACKED",
        "post_root_id": "BYTE_STREAM_SPLIT",
        "syn_count": "BYTE_STREAM_SPLIT",
        "x_pre": "BYTE_STREAM_SPLIT",
        "y_pre": "BYTE_STREAM_SPLIT",
    },
    "use_dictionary": False,
    "compression": "zstd",
    "compression_level": 7,
}


def _write_table(path, frame):
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), path)


def _footer_encodings(path):
    encodings = set()
    metadata = pq.ParquetFile(path).metadata
    for group in range(metadata.num_row_groups):
        for column in range(metadata.num_columns):
            encodings.update(
                str(e) for e in metadata.row_group(group).column(column).encodings)
    return encodings


class TestTempSibling:
    def test_cleanup_matches_production_temp_name(self, tmp_path):
        """The atomic-write temp path and the stale-cleanup glob must
        agree, or interrupted runs would leak temp files forever."""
        target = str(tmp_path / "synapse.parquet")
        temp = temp_sibling(target, "build")
        assert temp.startswith(str(tmp_path) + os.sep)
        assert os.path.basename(temp).startswith(".synapse.parquet.build.")

        # A leftover from another (dead) process uses the same scheme
        # with that process's pid, and must be reclaimed.
        stale = os.path.join(os.path.dirname(temp),
                             ".synapse.parquet.build.999999.tmp")
        Path(stale).write_bytes(b"leftover")
        remove_stale_temp_files(target, "build")
        assert not os.path.exists(stale)

    def test_current_process_temp_is_not_removed(self, tmp_path):
        target = str(tmp_path / "synapse.parquet")
        own = temp_sibling(target, "build")  # carries this process's pid
        Path(own).write_bytes(b"in progress")
        remove_stale_temp_files(target, "build")
        assert os.path.exists(own)

    @pytest.mark.skipif(os.name != "posix", reason="needs POSIX kill(0)")
    def test_live_writer_temp_is_preserved_then_reclaimed(self, tmp_path):
        """A concurrent preparation's in-progress temp must survive cleanup;
        once its process is gone, the leftover is reclaimed."""
        target = str(tmp_path / "synapse.parquet")
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            temp = os.path.join(
                str(tmp_path), f".synapse.parquet.build.{proc.pid}.tmp")
            Path(temp).write_bytes(b"being written")
            remove_stale_temp_files(target, "build")
            assert os.path.exists(temp)  # writer still alive

            proc.terminate()
            proc.wait(timeout=10)
            remove_stale_temp_files(target, "build")
            assert not os.path.exists(temp)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_old_temp_reclaimed_when_liveness_unknown(self, tmp_path, monkeypatch):
        """On platforms without a liveness probe, an aged temp is reclaimed
        but a recent one is left alone."""
        monkeypatch.setattr(pu, "_process_alive", lambda pid: None)
        target = str(tmp_path / "synapse.parquet")
        old = os.path.join(str(tmp_path), ".synapse.parquet.build.999999.tmp")
        recent = os.path.join(str(tmp_path), ".synapse.parquet.build.999998.tmp")
        Path(old).write_bytes(b"old")
        Path(recent).write_bytes(b"recent")
        stale_time = time.time() - pu.DEFAULT_TEMP_MAX_AGE_SECONDS - 60
        os.utime(old, (stale_time, stale_time))

        remove_stale_temp_files(target, "build")
        assert not os.path.exists(old)
        assert os.path.exists(recent)


class TestAtomicWrite:
    def test_write_parquet_atomic_fsyncs_and_replaces(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(os, "fsync", lambda fd: calls.append(fd))
        target = tmp_path / "out.parquet"

        write_parquet_atomic(
            str(target),
            lambda temp: pq.write_table(pa.table({"v": [1, 2, 3]}), temp))

        assert calls  # the temp (and directory) were flushed
        assert pd.read_parquet(target)["v"].tolist() == [1, 2, 3]
        assert not list(tmp_path.glob(".*.tmp"))

    def test_failure_leaves_original_intact_and_no_temp(self, tmp_path):
        target = tmp_path / "out.parquet"
        pq.write_table(pa.table({"v": [1]}), target)
        original = target.read_bytes()

        def boom(temp):
            Path(temp).write_bytes(b"partial")
            raise RuntimeError("write failed")

        with pytest.raises(RuntimeError):
            write_parquet_atomic(str(target), boom)

        assert target.read_bytes() == original
        assert not list(tmp_path.glob(".*.tmp"))

    def test_atomic_replace_is_not_a_plain_move(self, tmp_path):
        """atomic_replace must work across the same directory and leave
        exactly one file."""
        src = tmp_path / ".out.parquet.build.1.tmp"
        dst = tmp_path / "out.parquet"
        pq.write_table(pa.table({"v": [7]}), src)
        atomic_replace(str(src), str(dst))
        assert not src.exists()
        assert pd.read_parquet(dst)["v"].tolist() == [7]

    def test_fsync_failure_is_non_fatal(self, tmp_path, monkeypatch):
        """A failing flush must not abort the write: os.replace is the
        atomicity guarantee (Windows raises [Errno 9] on read-only fds)."""
        def boom(fd):
            raise OSError(9, "Bad file descriptor")

        monkeypatch.setattr(os, "fsync", boom)
        target = tmp_path / "out.parquet"

        write_parquet_atomic(
            str(target),
            lambda temp: pq.write_table(pa.table({"v": [1, 2, 3]}), temp))

        assert pd.read_parquet(target)["v"].tolist() == [1, 2, 3]
        assert not list(tmp_path.glob(".*.tmp"))

    def test_atomic_replace_flushes_through_writable_handle(self, tmp_path, monkeypatch):
        """os.fsync on a read-only descriptor raises [Errno 9] on Windows,
        so the flush handle must be opened writable."""
        import builtins
        modes = []
        real_open = builtins.open

        def tracking_open(file, mode="r", *args, **kwargs):
            modes.append(mode)
            return real_open(file, mode, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", tracking_open)
        src = tmp_path / ".out.parquet.build.1.tmp"
        dst = tmp_path / "out.parquet"
        pq.write_table(pa.table({"v": [7]}), src)

        atomic_replace(str(src), str(dst))

        assert modes and all(set(mode) & {"+", "w", "a"} for mode in modes)
        assert pd.read_parquet(dst)["v"].tolist() == [7]


class TestParquetReadable:
    def test_rejects_truncated_and_empty(self, tmp_path):
        truncated = tmp_path / "trunc.parquet"
        truncated.write_bytes(b"PAR1 not a real footer")
        assert parquet_readable(str(truncated)) is False

        empty = tmp_path / "empty.parquet"
        pq.write_table(pa.table({"v": pa.array([], type=pa.int64())}), empty)
        assert parquet_readable(str(empty)) is False

        good = tmp_path / "good.parquet"
        pq.write_table(pa.table({"v": [1]}), good)
        assert parquet_readable(str(good)) is True

    def test_rejects_footer_present_but_body_lost(self, tmp_path):
        """A partial copy that kept the tail parses its footer but has
        column offsets pointing past EOF; the metadata extent check must
        catch it without reading data pages."""
        src = tmp_path / "src.parquet"
        pq.write_table(
            pa.table({"a": list(range(50000)), "b": [i * 0.5 for i in range(50000)]}),
            src)
        assert parquet_readable(str(src)) is True

        raw = src.read_bytes()
        footer_len = int.from_bytes(raw[-8:-4], "little")
        # PAR1 magic + footer only: the body (most of the file) is gone.
        short = tmp_path / "footer_only.parquet"
        short.write_bytes(b"PAR1" + raw[-(8 + footer_len):])
        assert parquet_readable(str(short)) is False

    def test_equal_length_corruption_is_metadata_invisible(self, tmp_path):
        """Documents the boundary: in-place corruption that keeps the
        length cannot be detected from the footer, so parquet_readable
        still passes it (the page read is what fails)."""
        src = tmp_path / "src.parquet"
        pq.write_table(pa.table({"a": list(range(50000))}), src)
        raw = bytearray(src.read_bytes())
        raw[100:300] = b"\x00" * 200  # corrupt body, same length
        corrupt = tmp_path / "corrupt.parquet"
        corrupt.write_bytes(bytes(raw))
        assert parquet_readable(str(corrupt)) is True
        with pytest.raises(Exception):
            pq.ParquetFile(str(corrupt)).read_row_group(0)


class TestReencodeParquetLossless:
    def test_round_trip_preserves_values_and_dtypes(self, tmp_path):
        frame = pd.DataFrame({
            "pre_root_id": [720575940379280070, 720575940379280112],
            "post_root_id": [720575940381690921, 720575940379287024],
            "syn_count": [1, 13],
            "x_pre": [336848.0, 439920.5],
            "y_pre": [668048.0, 663824.25],
            "z_pre": [212220.0, 254610.75],
        }).astype({"x_pre": "float32", "y_pre": "float32", "z_pre": "float32"})
        path = tmp_path / "t.parquet"
        _write_table(path, frame)

        assert reencode_parquet_lossless(path, BANC_LIKE_PROFILE) is True
        # Decoded values and dtypes are identical to the source.
        pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
        # The requested encodings actually landed in the file.
        encodings = _footer_encodings(path)
        assert "BYTE_STREAM_SPLIT" in encodings
        assert "DELTA_BINARY_PACKED" in encodings

    def test_marker_makes_pass_idempotent(self, tmp_path):
        path = tmp_path / "t.parquet"
        _write_table(path, pd.DataFrame({"v": [1, 2, 3]}))

        assert reencode_parquet_lossless(path, BANC_LIKE_PROFILE) is True
        assert parquet_lossless_marker(path) is True
        schema = pq.read_schema(path)
        assert schema.metadata.get(LOSSLESS_MARKER_KEY) == LOSSLESS_MARKER_VALUE
        assert reencode_parquet_lossless(path, BANC_LIKE_PROFILE) is False

    def test_cast_applied_when_stats_prove_fit(self, tmp_path):
        profile = {"casts": {"coord": "int32"}, "use_dictionary": False,
                   "compression": "zstd", "compression_level": 7}
        path = tmp_path / "t.parquet"
        _write_table(path, pd.DataFrame({"coord": [86080, 912448]}))

        assert reencode_parquet_lossless(path, profile) is True
        out = pd.read_parquet(path)
        assert str(out["coord"].dtype) == "int32"
        assert out["coord"].tolist() == [86080, 912448]

    def test_cast_guard_skips_out_of_range_column(self, tmp_path):
        """A narrowing whose column does not provably fit is skipped, but
        the table is still re-encoded."""
        profile = {"casts": {"coord": "int32"}, "use_dictionary": False,
                   "compression": "zstd", "compression_level": 7}
        path = tmp_path / "t.parquet"
        _write_table(path, pd.DataFrame({"coord": [0, 3_000_000_000]}))

        assert reencode_parquet_lossless(path, profile) is True
        out = pd.read_parquet(path)
        assert str(out["coord"].dtype) == "int64"
        assert out["coord"].tolist() == [0, 3_000_000_000]

    def test_drop_columns_folded_into_same_pass(self, tmp_path):
        profile = {"drop_columns": ("junk",), "use_dictionary": False,
                   "compression": "zstd", "compression_level": 7}
        path = tmp_path / "t.parquet"
        _write_table(path, pd.DataFrame({"keep": [1, 2], "junk": [9, 9]}))

        assert reencode_parquet_lossless(path, profile) is True
        assert list(pd.read_parquet(path).columns) == ["keep"]

    def test_incompatible_encoding_dropped_not_fatal(self, tmp_path):
        """A string-only encoding listed for an integer column must be
        skipped (table still encoded) rather than abort the whole pass."""
        profile = {
            "column_encoding": {"id": "DELTA_LENGTH_BYTE_ARRAY",
                                "v": "BYTE_STREAM_SPLIT"},
            "use_dictionary": False, "compression": "zstd",
            "compression_level": 7,
        }
        path = tmp_path / "t.parquet"
        _write_table(path, pd.DataFrame({"id": [1, 2], "v": [10, 20]}))

        assert reencode_parquet_lossless(path, profile) is True
        out = pd.read_parquet(path)
        assert out["id"].tolist() == [1, 2]
        assert out["v"].tolist() == [10, 20]
        assert parquet_lossless_marker(path) is True

    def test_unreadable_source_left_intact(self, tmp_path):
        """A truncated source fails the pass without being destroyed."""
        path = tmp_path / "t.parquet"
        path.write_bytes(b"PAR1 truncated garbage")

        assert reencode_parquet_lossless(path, BANC_LIKE_PROFILE) is False
        assert path.read_bytes() == b"PAR1 truncated garbage"
