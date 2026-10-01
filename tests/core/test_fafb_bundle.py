"""Tests for the FAFB healed skeleton sources: zip-only serving (the
default application path — no ``.zst`` container is created or appended),
the columnar container tooling (bulk pack, lazy per-skeleton conversion,
ZIP fallback with logical + verbatim-compaction removal), reader
resolution (zip first), and the CLI surface."""

import gzip  # noqa: F401  (parity with sibling cache modules)
import os
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import fafb_bundle as fb  # noqa: E402
import fafb_utils as fau  # noqa: E402


def _swc(nid, n=12, frac=True):
    lines = [f"# SWC {nid}"]
    for i in range(1, n + 1):
        parent = -1 if i == 1 else i - 1
        x = i * 1000.0 + (0.44 if frac else 0.0)
        y = i * 200.0 + 0.5
        z = i * 50.0
        lines.append(f"{i} 1 {x} {y} {z} 2.5 {parent}")
    return "\n".join(lines) + "\n"


def make_zip(tmp_path, ids=(101, 102, 103, 104, 105, 106, 107, 108, 109, 110)):
    zip_path = tmp_path / "sk_lod1_783_healed.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for nid in ids:
            z.writestr(f"{nid}.swc", _swc(nid))
    return zip_path


def _bundle_path(tmp_path):
    return tmp_path / "sk_lod1_783_healed.zst"


class TestBulkPack:
    def test_pack_roundtrip_and_verify(self, tmp_path):
        zip_path = make_zip(tmp_path)
        bundle_path = _bundle_path(tmp_path)
        stats = fb.pack(zip_path, bundle_path, n_workers=2)
        assert stats["neurons"] == 10
        assert stats["max_coord_error_nm"] < 0.1
        assert 0 < stats["blocks"] <= 10

        reader = fb.FAFBSkeletonBundle(bundle_path, zip_path=zip_path,
                                       lazy_convert=False)
        try:
            assert reader.count() == 10
            text = reader.get(105)
            assert text and text.splitlines()[0].startswith("1 1 1000.44")
            assert text and "5000.4" in text.splitlines()[4]
            assert reader.get(999) is None
            assert reader.contains(103)
            assert len(list(reader.iter_texts())) == 10
        finally:
            reader.close()

        result = fb.verify(bundle_path, zip_path=zip_path, sample=10)
        assert result["ok"] and result["errors"] == 0
        assert result["max_node_delta_nm"] < 0.1

    def test_pack_rejects_oversized_roundtrip_error(self, tmp_path, monkeypatch):
        zip_path = make_zip(tmp_path)
        bundle_path = _bundle_path(tmp_path)
        monkeypatch.setattr(fb, "MAX_COORD_ERROR_NM", 1e-9)
        with pytest.raises(ValueError, match="refusing to build"):
            fb.pack(zip_path, bundle_path, n_workers=1)
        assert not bundle_path.exists()

    def test_corrupt_footer_raises(self, tmp_path):
        zip_path = make_zip(tmp_path)
        bundle_path = _bundle_path(tmp_path)
        fb.pack(zip_path, bundle_path, n_workers=1)
        data = bytearray(bundle_path.read_bytes())
        data[-1] ^= 0xFF  # corrupt the footer version byte
        bundle_path.write_bytes(bytes(data))
        reader = fb.FAFBSkeletonBundle(bundle_path, zip_path=zip_path,
                                       lazy_convert=False)
        with pytest.raises(fb.BundleCorruptError):
            reader._ensure_open()


class TestZipOnlyResolution:
    def test_open_bundle_prefers_zip_and_never_creates_zst(self, tmp_path):
        zip_path = make_zip(tmp_path)
        bundle = fb.open_bundle(tmp_path)
        assert bundle is not None
        assert bundle.zip_path == zip_path
        assert bundle.bundle_path is None
        assert bundle.lazy_convert is False
        try:
            assert bundle.get(101) is not None
            assert bundle.count() == 10
            assert bundle.bundle_count() == 0
        finally:
            bundle.close()
        # serving from the zip must not have created a container
        assert not _bundle_path(tmp_path).exists()

    def test_open_bundle_zst_only_opened_readonly(self, tmp_path):
        # no zip anywhere: an existing .zst stays readable
        zip_path = make_zip(tmp_path)
        bundle_path = _bundle_path(tmp_path)
        fb.append_entries(zip_path, bundle_path, [101, 102])
        zip_path.unlink()
        before = bundle_path.stat().st_size
        bundle = fb.open_bundle(tmp_path)
        assert bundle is not None
        assert bundle.bundle_path == bundle_path
        assert bundle.zip_path is None
        try:
            assert bundle.get(101) is not None
            assert bundle.get(103) is None
            assert bundle.count() == 2
        finally:
            bundle.close()
        assert bundle_path.stat().st_size == before

    def test_open_bundle_missing_everything(self, tmp_path):
        assert fb.open_bundle(tmp_path) is None

    def test_zip_only_get_iter_and_flush_noop(self, tmp_path):
        make_zip(tmp_path)
        bundle = fb.open_bundle(tmp_path)
        try:
            texts = dict(bundle.iter_texts())
            assert set(texts) == set(range(101, 111))
            # flush must be a no-op in zip-only mode
            bundle.flush()
            assert not _bundle_path(tmp_path).exists()
        finally:
            bundle.close()

    def test_zip_only_compact_is_noop(self, tmp_path):
        zip_path = make_zip(tmp_path)
        mtime_before = zip_path.stat().st_mtime_ns
        bundle = fb.open_bundle(tmp_path)
        try:
            assert bundle.compact_zip() == 0
        finally:
            bundle.close()
        assert zip_path.stat().st_mtime_ns == mtime_before

    def test_zip_only_never_touches_legacy_zst(self, tmp_path):
        zip_path = make_zip(tmp_path)
        bundle_path = _bundle_path(tmp_path)
        fb.append_entries(zip_path, bundle_path, [101])
        before = bundle_path.stat().st_mtime_ns
        bundle = fb.open_bundle(tmp_path)
        try:
            # the zip serves all ids; the legacy container stays untouched
            assert bundle.get(105) is not None
        finally:
            bundle.close()
        assert bundle_path.stat().st_mtime_ns == before


class TestLazyConversion:
    def test_resolution_zip_first_when_both_exist(self, tmp_path):
        zip_path = make_zip(tmp_path)
        bundle_path = _bundle_path(tmp_path)
        fb.pack(zip_path, bundle_path, n_workers=1)
        # zip-only: the zip is served directly, the packed .zst is ignored
        bundle = fb.open_bundle(tmp_path)
        assert bundle is not None
        assert bundle.zip_path == zip_path
        assert bundle.bundle_path is None
        try:
            assert bundle.get(101) is not None
        finally:
            bundle.close()

    def test_get_converts_lazily_and_dedups(self, tmp_path):
        zip_path = make_zip(tmp_path)
        bundle_path = _bundle_path(tmp_path)
        bundle = fb.FAFBSkeletonBundle(bundle_path, zip_path=zip_path,
                                       lazy_convert=True)
        try:
            text = bundle.get(107)
            assert text is not None and "7000.4" in text
            bundle.get(107)  # second get before flush: no duplicate
            bundle.flush()
            assert bundle._ids() == {107}
            bundle.get(108)
            bundle.get(109)
            bundle.flush()
            assert bundle._ids() == {107, 108, 109}
            # served from the container, not the zip: the float32 columnar
            # round-trip may re-quantize the text (<= 0.1 nm policy), so
            # compare parsed rows with tolerance
            again = bundle.get(107)
            rows_a = fb._parse_swc_rows(text.encode("utf-8"))
            rows_b = fb._parse_swc_rows(again.encode("utf-8"))
            assert len(rows_a) == len(rows_b)
            for ra, rb in zip(rows_a, rows_b):
                assert ra[0] == rb[0] and ra[1] == rb[1] and ra[6] == rb[6]
                for va, vb in zip(ra[2:6], rb[2:6]):
                    assert abs(va - vb) < 0.1
        finally:
            bundle.close()

    def test_compact_removes_entries_verbatim(self, tmp_path):
        zip_path = make_zip(tmp_path)
        bundle_path = _bundle_path(tmp_path)
        bundle = fb.FAFBSkeletonBundle(bundle_path, zip_path=zip_path,
                                       lazy_convert=True)
        try:
            for nid in (107, 108, 109):
                assert bundle.get(nid) is not None
            bundle.flush()
            removed = bundle.compact_zip()
            assert removed == 3
        finally:
            bundle.close()
        with zipfile.ZipFile(zip_path, "r") as z:
            names = z.namelist()
        assert sorted(names) == [
            f"{nid}.swc" for nid in (101, 102, 103, 104, 105, 106, 110)
        ]
        # remaining entries read byte-identically through zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            assert z.read("103.swc") == _swc(103).encode("utf-8")
        reader = fb.FAFBSkeletonBundle(bundle_path, zip_path=zip_path,
                                       lazy_convert=False)
        try:
            assert "3000.4" in reader.get(103)
            assert reader.get(107) is not None  # from the container
            assert reader.count() == 10
        finally:
            reader.close()

    def test_append_entries_bulk(self, tmp_path):
        zip_path = make_zip(tmp_path)
        bundle_path = _bundle_path(tmp_path)
        appended = fb.append_entries(zip_path, bundle_path, [101, 102, 999])
        assert appended == 2
        reader = fb.FAFBSkeletonBundle(bundle_path, zip_path=zip_path,
                                       lazy_convert=False)
        try:
            assert reader._ids() == {101, 102}
            # idempotent: re-appending skips already-converted ids
            assert fb.append_entries(zip_path, bundle_path, [101, 103]) == 1
            reader._reload_index(force=True)
            assert reader._ids() == {101, 102, 103}
        finally:
            reader.close()

    def test_info(self, tmp_path):
        zip_path = make_zip(tmp_path)
        bundle_path = _bundle_path(tmp_path)
        result = fb.info(bundle_path, zip_path=zip_path)
        assert result["zip_entries"] == 10
        fb.append_entries(zip_path, bundle_path, [101])
        result = fb.info(bundle_path, zip_path=zip_path)
        assert result["bundle_neurons"] == 1
        assert result["converted_percent"] == 10.0


class TestFafbUtilsSeam:
    def test_get_fafb_skeleton_bundle_zip_preferred(self, tmp_path):
        zip_path = make_zip(tmp_path)
        fb.append_entries(zip_path, _bundle_path(tmp_path), [101])
        bundle = fau.get_fafb_skeleton_bundle(str(tmp_path))
        assert bundle is not None
        try:
            # zip-only serving: every zip id resolves, no container needed
            assert bundle.get(101) is not None
            assert bundle.count() == 10
            assert bundle.bundle_path is None
        finally:
            bundle.close()

    def test_get_fafb_skeleton_bundle_zip_only(self, tmp_path):
        make_zip(tmp_path)
        assert not _bundle_path(tmp_path).exists()
        bundle = fau.get_fafb_skeleton_bundle(str(tmp_path))
        assert bundle is not None and bundle.zip_path is not None
        try:
            assert bundle.get(105) is not None
        finally:
            bundle.close()
        assert not _bundle_path(tmp_path).exists()

    def test_get_fafb_skeleton_bundle_missing(self, tmp_path):
        assert fau.get_fafb_skeleton_bundle(str(tmp_path)) is None


class TestCli:
    def test_cli_info_verify_append_compact(self, tmp_path):
        zip_path = make_zip(tmp_path)
        bundle_path = _bundle_path(tmp_path)
        fb.append_entries(zip_path, bundle_path, [101, 102])
        env = dict(os.environ)
        run = lambda *args: subprocess.run(
            [sys.executable, "-m", "src.fafb_bundle", *args],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            env=env)
        info = run("info", str(bundle_path), "--zip", str(zip_path))
        assert info.returncode == 0 and "converted_percent" in info.stdout
        append = run("append", str(zip_path), str(bundle_path), "103")
        assert append.returncode == 0 and "appended 1" in append.stdout
        compact = run("compact", str(zip_path), str(bundle_path))
        assert compact.returncode == 0
        verify = run("verify", str(bundle_path), "--zip", str(zip_path),
                     "--sample", "10")
        assert verify.returncode == 0 and '"ok": true' in verify.stdout

    def test_cli_pack_delete_source_gated_on_verify(self, tmp_path):
        zip_path = make_zip(tmp_path)
        out = tmp_path / "full.zst"
        env = dict(os.environ)
        run = subprocess.run(
            [sys.executable, "-m", "src.fafb_bundle", "pack", str(zip_path),
             str(out), "--workers", "2", "--delete-source"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), env=env)
        assert run.returncode == 0 and "deleted" in run.stdout
        assert not zip_path.exists() and out.exists()


class TestCodexOverlay:
    """Overlay bundle: union reads, overlay-wins, rewrite safety."""

    SWC = ("# SWC format file\n"
           "1 1 0 0 0 1 -1\n"
           "2 3 100 0 0 1 1\n")

    def _dataset_dir(self, tmp_path, with_base=True):
        from fafb_bundle import write_overlay_members
        dataset_dir = tmp_path / "flywire_FAFB_v783"
        dataset_dir.mkdir()
        if with_base:
            import zipfile
            with zipfile.ZipFile(dataset_dir / "sk_lod1_783_healed.zip",
                                 "w") as zf:
                zf.writestr("111.swc", self.SWC)
                zf.writestr("222.swc", self.SWC)
        return dataset_dir

    def test_overlay_wins_over_base_and_union_ids(self, tmp_path):
        from fafb_bundle import open_bundle, write_overlay_members
        dataset_dir = self._dataset_dir(tmp_path)
        write_overlay_members(dataset_dir, {"333": self.SWC},
                              origin="fafb_codex_healed")
        bundle = open_bundle(dataset_dir)
        try:
            assert bundle.get(333) is not None      # overlay-only id
            assert bundle.get(111) is not None      # base id
            assert bundle.contains(333) and bundle.contains(111)
            assert bundle.ids() == {111, 222, 333}
        finally:
            bundle.close()

    def test_overlay_replaces_base_member_for_same_id(self, tmp_path):
        from fafb_bundle import open_bundle, write_overlay_members
        dataset_dir = self._dataset_dir(tmp_path)
        replacement = "9 1 0 0 0 1 -1\n"
        write_overlay_members(dataset_dir, {"111": replacement},
                              origin="local_extrusion_fix")
        bundle = open_bundle(dataset_dir)
        try:
            assert bundle.get(111).strip() == replacement.strip()
        finally:
            bundle.close()

    def test_rewrite_preserves_other_members_and_manifest(self, tmp_path):
        from fafb_bundle import (open_bundle, read_overlay_manifest,
                                 write_overlay_members)
        dataset_dir = self._dataset_dir(tmp_path)
        write_overlay_members(dataset_dir, {"111": self.SWC, "333": self.SWC},
                              origin="fafb_codex_healed")
        write_overlay_members(dataset_dir, {"111": "7 1 0 0 0 1 -1\n"},
                              origin="local_extrusion_fix")
        bundle = open_bundle(dataset_dir)
        try:
            assert bundle.get(111).strip() == "7 1 0 0 0 1 -1"
            assert bundle.get(333) is not None  # untouched sibling kept
        finally:
            bundle.close()
        manifest = read_overlay_manifest(dataset_dir)
        assert manifest["111"]["origin"] == "local_extrusion_fix"
        assert manifest["111"]["replaced"] is True
        assert manifest["333"]["replaced"] is False

    def test_overlay_only_without_base_zip(self, tmp_path):
        from fafb_bundle import open_bundle, write_overlay_members
        dataset_dir = self._dataset_dir(tmp_path, with_base=False)
        write_overlay_members(dataset_dir, {"111": self.SWC},
                              origin="fafb_codex_healed")
        bundle = open_bundle(dataset_dir)
        try:
            assert bundle is not None
            assert bundle.get(111) is not None
        finally:
            bundle.close()
