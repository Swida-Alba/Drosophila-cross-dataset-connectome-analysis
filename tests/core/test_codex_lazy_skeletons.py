"""Lazy per-neuron skeleton fetch tests: zip64 parsing, the ETag-pinned
bundle index, per-neuron ranged extraction against a local Range server,
and the provenance/persistence contracts the resolver integration uses."""

import http.server
import io
import json
import socketserver
import struct
import sys
import threading
import zipfile
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import codex_downloader as cd  # noqa: E402

TOKEN = "test-token"
ETAG = '"bundle-etag"'
BODY_IDS = [720575940590515268, 720575940596125868, 720575940623940963]


def _swc_text(seed: int) -> str:
    return (
        "# SWC format file\n"
        "1 1 0.0 0.0 0.0 1.0 -1\n"
        f"2 3 {seed}.0 0.0 0.0 1.0 1\n"
        f"3 3 {seed * 2}.0 5.0 0.0 1.0 2\n"
    )


def _build_bundle() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for index, body_id in enumerate(BODY_IDS):
            zf.writestr(f"{body_id}.swc", _swc_text(index + 1))
    return buffer.getvalue()


BUNDLE = _build_bundle()


class _BundleHandler(http.server.BaseHTTPRequestHandler):
    requests = []
    etag = ETAG

    def do_GET(self):  # noqa: N802 — http.server API
        from urllib.parse import parse_qs, urlparse
        query = parse_qs(urlparse(self.path).query)
        product = (query.get("data_product") or [""])[0]
        token = (query.get("api_token") or [""])[0]
        rng = self.headers.get("Range")
        type(self).requests.append((product, rng))
        if token != TOKEN or product != "skeleton_swc_files":
            self.send_response(401 if token != TOKEN else 404)
            self.end_headers()
            return
        if rng is None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(BUNDLE)))
            self.send_header("ETag", type(self).etag)
            self.end_headers()
            self.wfile.write(BUNDLE)
            return
        if rng == "bytes=0-0":
            self.send_response(206)
            self.send_header("Content-Range", f"bytes 0-0/{len(BUNDLE)}")
            self.send_header("Content-Length", "0")
            self.send_header("ETag", type(self).etag)
            self.end_headers()
            return
        start, end = (int(part) for part in rng.split("=")[1].split("-"))
        end = min(end, len(BUNDLE) - 1)
        self.send_response(206)
        self.send_header("Content-Range",
                         f"bytes {start}-{end}/{len(BUNDLE)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("ETag", type(self).etag)
        self.end_headers()
        self.wfile.write(BUNDLE[start:end + 1])

    def log_message(self, *args):  # silence the test log
        pass


@pytest.fixture()
def bundle_server(monkeypatch, tmp_path):
    _BundleHandler.requests = []
    srv = socketserver.TCPServer(("127.0.0.1", 0), _BundleHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(cd, "CODEX_DOWNLOAD_URL",
                        f"http://127.0.0.1:{port}/api/download_resource")
    project_root = tmp_path / "root"
    project_root.mkdir()
    (project_root / "config_local.json").write_text(
        json.dumps({"tokens": {"flywire_codex": TOKEN}}), encoding="utf-8")
    catalog = dict(cd.PRODUCT_CATALOG)
    catalog["skeleton_swc_files"] = cd.CodexProduct(
        "skeleton_swc_files", "sk_lod1_783_healed.zip", len(BUNDLE),
        cd.LEVEL_SKELETON, "test bundle")
    monkeypatch.setattr(cd, "PRODUCT_CATALOG", catalog)
    monkeypatch.delenv(cd.CODEX_TOKEN_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    yield project_root
    srv.shutdown()
    srv.server_close()


def _index_requests():
    return [r for r in _BundleHandler.requests
            if r[0] == "skeleton_swc_files" and r[1] not in (None, "bytes=0-0")]


def test_bundle_index_fetch_and_cache(bundle_server):
    project_root = bundle_server
    index = cd.ensure_bundle_index(project_root=project_root)
    assert set(index) == {f"{bid}.swc" for bid in BODY_IDS}
    cache_dir = cd.bundle_cache_dir(project_root=project_root)
    assert (cache_dir / "codex_bundle_index.json").exists()
    assert (cache_dir / "codex_bundle_meta.json").exists()
    assert (cache_dir / "codex_bundle_central_directory.bin").exists()
    before = len(_index_requests())
    # second call is served from the ETag-pinned cache
    index2 = cd.ensure_bundle_index(project_root=project_root)
    assert index2 == index
    assert len(_index_requests()) == before


def test_bundle_index_refetches_on_etag_change(bundle_server, monkeypatch):
    project_root = bundle_server
    cd.ensure_bundle_index(project_root=project_root)
    before = len(_index_requests())
    monkeypatch.setattr(_BundleHandler, "etag", '"new-etag"')
    cd.ensure_bundle_index(project_root=project_root)
    assert len(_index_requests()) > before


def test_fetch_skeleton_swcs_byte_equal(bundle_server):
    project_root = bundle_server
    index = cd.ensure_bundle_index(project_root=project_root)
    before = len(_BundleHandler.requests)
    fetched, missing = cd.fetch_skeleton_swcs(
        BODY_IDS, project_root=project_root, index=index)
    assert missing == []
    with zipfile.ZipFile(io.BytesIO(BUNDLE)) as zf:
        reference = {bid: zf.read(f"{bid}.swc").decode()
                     for bid in BODY_IDS}
    assert fetched == reference
    # each neuron costs exactly one ranged GET
    assert len(_BundleHandler.requests) - before == len(BODY_IDS)


def test_fetch_skeleton_swcs_reports_missing(bundle_server):
    project_root = bundle_server
    index = cd.ensure_bundle_index(project_root=project_root)
    _fetched, missing = cd.fetch_skeleton_swcs(
        BODY_IDS + [111], project_root=project_root, index=index)
    assert missing == [111]


def test_fetch_rejects_crc_mismatch(bundle_server, monkeypatch):
    project_root = bundle_server
    index = cd.ensure_bundle_index(project_root=project_root)
    victim = f"{BODY_IDS[0]}.swc"
    poisoned = dict(index)
    entry = dict(poisoned[victim])
    entry["crc"] = (entry["crc"] + 1) & 0xFFFFFFFF
    poisoned[victim] = entry
    monkeypatch.setattr(cd, "ensure_bundle_index", lambda *a, **k: poisoned)
    with pytest.raises(cd.CodexProductUnavailable) as excinfo:
        cd.fetch_skeleton_swcs([BODY_IDS[0]], project_root=project_root)
    assert "mismatch" in str(excinfo.value)


def test_index_fetch_cancelled(bundle_server, monkeypatch):
    project_root = bundle_server
    cancel = threading.Event()
    cancel.set()
    monkeypatch.setattr(cd, "CHUNK_BYTES", 512)
    with pytest.raises(cd.CodexDownloadCancelled):
        cd.ensure_bundle_index(project_root=project_root,
                               cancel_event=cancel)


def test_parse_synthetic_zip64_entry():
    name = b"720575940596125868.swc"
    extra = struct.pack("<HH", 0x0001, 24) + struct.pack(
        "<QQQ", 85_714, 30_000, 13_862_000_000)
    header = (
        b"PK\x01\x02"
        + struct.pack("<HHH", 20, 45, 0)              # made/needed/flags
        + struct.pack("<H", 8)                        # method = deflate
        + struct.pack("<HH", 0, 0)                    # time/date
        + struct.pack("<I", 0)                        # crc32
        + struct.pack("<II", 0xFFFFFFFF, 0xFFFFFFFF)  # comp/uncomp -> zip64
        + struct.pack("<HHH", len(name), len(extra), 0)
        + struct.pack("<HH", 0, 0)                    # disk/internal attrs
        + struct.pack("<I", 0)                        # external attrs
        + struct.pack("<I", 0xFFFFFFFF)               # local offset -> zip64
    )
    entry = header + name + extra
    entries = cd.parse_bundle_central_directory(entry)
    assert entries[name.decode()] == {
        "method": 8, "comp": 30_000, "uncomp": 85_714,
        "crc": 0, "offset": 13_862_000_000}


def test_write_compressed_skeleton_records_source_header(tmp_path):
    import navis
    from morphology import _write_compressed_skeleton
    from skeleton_provenance import (FAFB_CODEX_HEALED, parse_provenance)
    import zstandard as zstd

    neuron = navis.read_swc(io.StringIO(_swc_text(1)))
    path = tmp_path / "720575940596125868.swc.zst"
    _write_compressed_skeleton(path, neuron, simplification=0,
                               source=FAFB_CODEX_HEALED)
    text = zstd.ZstdDecompressor().decompress(
        path.read_bytes(), max_output_size=1 << 20).decode()
    provenance = parse_provenance(text)
    assert provenance.source == FAFB_CODEX_HEALED
    assert provenance.simplification == 0
    assert text.index("# DROCAT source:") < text.index("# DROCAT simpl:")


def test_resolver_chain_wiring():
    """The lazy step sits between the bundle and the extrusion check, joins
    the extrusion pool, caches into the overlay bundle, and a local prune
    replaces the overlay member it produced."""
    source = Path(cd.__file__).with_name("visualize_skeleton.py") \
        .read_text(encoding="utf-8")
    assert "_take_codex_lazy()" in source
    assert source.index("def _take_codex_lazy") > source.index("def _take_bundle")
    assert "{'zip', 'raw_cache', 'codex_lazy'}" in source
    assert "'codex_lazy', 'local_repaired', 'cave'" in source
    assert "write_overlay_members(" in source
    assert "origin=FAFB_CODEX_HEALED" in source
    assert "def _replace_overlay_member" in source
    assert "self._replace_overlay_member(canonical, repaired)" in source
    morphology_source = Path(cd.__file__).with_name("morphology.py") \
        .read_text(encoding="utf-8")
    assert '"source": source or "neuprint.fetch_skeleton"' in morphology_source
    assert "_provenance.make_source_line(source)" in morphology_source
