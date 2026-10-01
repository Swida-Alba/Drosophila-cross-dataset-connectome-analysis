"""Codex downloader tests against a local server that replicates the
measured codex.flywire.ai semantics: Range honoured, If-Range ignored,
401 without a token, 404 JSON for unknown products.

TokenManager also searches the CURRENT WORKING DIRECTORY for config files,
so every test isolates cwd — otherwise the developer's real
config_local.json would leak its flywire_codex token into the chain."""

import http.server
import json
import socketserver
import sys
import threading
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import codex_downloader as cd  # noqa: E402

TOKEN = "test-token"
ETAG = '"test-etag"'
BODIES = {
    "classification": b"c" * 10_000,
    "consolidated_cell_types": b"t" * 2_000,
}


class _CodexHandler(http.server.BaseHTTPRequestHandler):
    requests = []  # (product, range, token)

    def do_GET(self):  # noqa: N802 — http.server API
        from urllib.parse import parse_qs, urlparse
        query = parse_qs(urlparse(self.path).query)
        product = (query.get("data_product") or [""])[0]
        token = (query.get("api_token") or [""])[0]
        rng = self.headers.get("Range")
        type(self).requests.append((product, rng, token))
        if token != TOKEN:
            self._fail(401)
            return
        body = BODIES.get(product)
        if body is None:
            self._fail(404, product)
            return
        if rng is None or rng == "bytes=0-0":
            self.send_response(206 if rng else 200)
            if rng:
                self.send_header("Content-Range", f"bytes 0-0/{len(body)}")
                self.send_header("Content-Length", "0")
            else:
                self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", ETAG)
            self.end_headers()
            if rng is None:
                self.wfile.write(body)
            return
        start, end = (int(part) for part in rng.split("=")[1].split("-"))
        end = min(end, len(body) - 1)
        self.send_response(206)
        self.send_header("Content-Range",
                         f"bytes {start}-{end}/{len(body)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("ETag", ETAG)
        self.end_headers()
        self.wfile.write(body[start:end + 1])

    def _fail(self, code, product=""):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        message = (f"Download resource '{product}' is not available for "
                   f"dataset 'fafb'." if code == 404 else "unauthorized")
        self.wfile.write(json.dumps({"error": message}).encode())

    def log_message(self, *args):  # silence the test log
        pass


@pytest.fixture()
def server(monkeypatch, tmp_path):
    _CodexHandler.requests = []
    srv = socketserver.TCPServer(("127.0.0.1", 0), _CodexHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(cd, "CODEX_DOWNLOAD_URL",
                        f"http://127.0.0.1:{port}/api/download_resource")
    project_root = tmp_path / "root"
    project_root.mkdir()
    (project_root / "config_local.json").write_text(
        json.dumps({"tokens": {"flywire_codex": TOKEN}}), encoding="utf-8")
    catalog = {key: cd.CodexProduct(key, cd.PRODUCT_CATALOG[key].filename,
                                    len(body), cd.LEVEL_NECESSARY, "test")
               for key, body in BODIES.items()}
    monkeypatch.setattr(cd, "PRODUCT_CATALOG", catalog)
    monkeypatch.delenv(cd.CODEX_TOKEN_ENV, raising=False)
    # TokenManager resolves config files from cwd first; keep it clean so
    # the developer's real config_local.json never enters the chain.
    monkeypatch.chdir(tmp_path)
    yield project_root, tmp_path / "downloads"
    srv.shutdown()
    srv.server_close()


def _seed_part(downloads, product, body, fraction, etag):
    downloads.mkdir(parents=True, exist_ok=True)
    dest = downloads / product.filename
    part = downloads / (product.filename + ".part")
    part.write_bytes(body[:int(len(body) * fraction)])
    cd._write_sidecar(cd._sidecar_path(dest), etag, len(body))
    return dest, part


def _data_ranges(product):
    return [r[1] for r in _CodexHandler.requests
            if r[0] == product and r[1] not in (None, "bytes=0-0")]


def test_no_token_raises_before_any_request(monkeypatch, tmp_path):
    monkeypatch.delenv(cd.CODEX_TOKEN_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(cd.CodexTokenRequired) as excinfo:
        cd.download_product("classification", tmp_path / "dl",
                            project_root=tmp_path)
    assert "Account page" in str(excinfo.value)
    assert not (tmp_path / "dl" / "classification.csv.gz").exists()
    assert _CodexHandler.requests == []


def test_token_reads_config_then_env(server, monkeypatch, tmp_path):
    project_root, _ = server
    assert cd.get_codex_token(project_root) == TOKEN
    bare = tmp_path / "bare"
    bare.mkdir()
    monkeypatch.setenv(cd.CODEX_TOKEN_ENV, "env-token")
    assert cd.get_codex_token(bare) == "env-token"


def test_verify_codex_token(server):
    project_root, _ = server
    ok, message = cd.verify_codex_token(project_root)
    assert ok and "accepted" in message.lower()


def test_verify_codex_token_refused(server, tmp_path):
    _, _ = server
    root2 = tmp_path / "root2"
    root2.mkdir()
    (root2 / "config_local.json").write_text(
        json.dumps({"tokens": {"flywire_codex": "wrong"}}), encoding="utf-8")
    ok, message = cd.verify_codex_token(root2)
    assert not ok and "refused" in message.lower()


def test_download_completes_and_finalizes(server):
    project_root, downloads = server
    dest = cd.download_product("classification", downloads,
                               project_root=project_root, chunk_bytes=4096)
    assert dest.read_bytes() == BODIES["classification"]
    assert not (downloads / (dest.name + ".part")).exists()
    assert not cd._sidecar_path(dest).exists()
    probes = [r for r in _CodexHandler.requests if r[1] == "bytes=0-0"]
    assert len(probes) == 1
    assert len(_data_ranges("classification")) == 3


def test_download_resumes_matching_etag(server):
    project_root, downloads = server
    product = cd.PRODUCT_CATALOG["classification"]
    body = BODIES["classification"]
    _seed_part(downloads, product, body, 0.5, ETAG)
    dest = cd.download_product("classification", downloads,
                               project_root=project_root, chunk_bytes=4096)
    assert dest.read_bytes() == body
    ranges = _data_ranges("classification")
    assert ranges, _CodexHandler.requests
    first_start = int(ranges[0].split("=")[1].split("-")[0])
    assert first_start == len(body) // 2, ranges


def test_download_restarts_on_etag_change(server):
    project_root, downloads = server
    product = cd.PRODUCT_CATALOG["classification"]
    body = BODIES["classification"]
    _seed_part(downloads, product, body, 0.5, '"stale-etag"')
    dest = cd.download_product("classification", downloads,
                               project_root=project_root, chunk_bytes=4096)
    assert dest.read_bytes() == body
    ranges = _data_ranges("classification")
    assert int(ranges[0].split("=")[1].split("-")[0]) == 0, ranges


def test_download_cancelled_keeps_part(server):
    project_root, downloads = server
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(cd.CodexDownloadCancelled):
        cd.download_product("classification", downloads,
                            project_root=project_root, cancel_event=cancel)
    product = cd.PRODUCT_CATALOG["classification"]
    part = downloads / (product.filename + ".part")
    assert part.exists()
    assert not (downloads / product.filename).exists()


def test_unknown_product_raises(server, monkeypatch):
    project_root, downloads = server
    catalog = dict(cd.PRODUCT_CATALOG)
    catalog["ghost"] = cd.CodexProduct("ghost", "ghost.csv.gz", 10,
                                       cd.LEVEL_NECESSARY, "test")
    monkeypatch.setattr(cd, "PRODUCT_CATALOG", catalog)
    with pytest.raises(cd.CodexProductUnavailable):
        cd.download_product("ghost", downloads, project_root=project_root)


def test_size_drift_raises(server, monkeypatch):
    project_root, downloads = server
    catalog = dict(cd.PRODUCT_CATALOG)
    key = "classification"
    catalog[key] = cd.CodexProduct(key, catalog[key].filename,
                                   catalog[key].size + 1,
                                   cd.LEVEL_NECESSARY, "test")
    monkeypatch.setattr(cd, "PRODUCT_CATALOG", catalog)
    with pytest.raises(cd.CodexProductUnavailable) as excinfo:
        cd.download_product(key, downloads, project_root=project_root)
    assert "refusing" in str(excinfo.value)


def test_skip_if_exists(server):
    project_root, downloads = server
    product = cd.PRODUCT_CATALOG["classification"]
    downloads.mkdir(parents=True, exist_ok=True)
    dest = downloads / product.filename
    dest.write_bytes(BODIES["classification"])
    result = cd.download_product("classification", downloads,
                                 project_root=project_root)
    assert result == dest
    assert _CodexHandler.requests == []
    status = cd.product_status("classification", downloads)
    assert status["complete"] and status["local"] == product.size


def test_catalog_contract():
    necessary = {p.key for p in cd.level_products(cd.LEVEL_NECESSARY)}
    assert necessary == {
        "classification", "connections_princeton_no_threshold", "names",
        "coordinates", "neurons", "cell_stats", "consolidated_cell_types"}
    assert {p.key for p in cd.level_products(cd.LEVEL_SYNAPSE)} == {
        "synapse_table"}
    assert {p.key for p in cd.level_products(cd.LEVEL_SKELETON)} == {
        "skeleton_swc_files"}
    by_key = {p.key: p for p in cd.PRODUCT_CATALOG.values()}
    assert (by_key["synapse_table"].filename
            == "fafb_v783_princeton_synapse_table.csv.gz")
    assert by_key["skeleton_swc_files"].filename == "sk_lod1_783_healed.zip"
    assert by_key["classification"].filename == "classification.csv.gz"
    assert by_key["connections_princeton_no_threshold"].size == 275_679_780
    assert cd.level_bytes(cd.LEVEL_SYNAPSE) == 2_695_106_039


def test_downloads_dir_for(tmp_path):
    path = cd.downloads_dir_for("flywire_FAFB_v783", project_root=tmp_path)
    assert path == tmp_path / "datasets" / "flywire_FAFB_v783" / "downloads"


def test_synapse_table_ready(tmp_path):
    parquet = cd.synapse_parquet_path("flywire_FAFB_v783", project_root=tmp_path)
    assert parquet == (tmp_path / "datasets" / "flywire_FAFB_v783"
                       / "flywire_FAFB_v783_synapse_table.parquet")
    assert not cd.synapse_table_ready("flywire_FAFB_v783", project_root=tmp_path)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_bytes(b"parquet")
    assert cd.synapse_table_ready("flywire_FAFB_v783", project_root=tmp_path)
