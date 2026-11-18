"""Coverage tests for utils/api_utils.py, neuronbridge_client.py, neuronbridge_cache.py.

These tests exercise the remaining uncovered branches of the API utility,
client and cache modules.  All HTTP is mocked at the session boundary and all
disk I/O uses pytest tmp_path.  No real network access occurs.
"""

import json
import warnings
from io import BytesIO

import pandas as pd
import pytest
import requests
from PIL import Image

from utils import api_utils
from utils.api_utils import (
    APITimeoutError,
    APIRetryExhaustedError,
    APICancelError,
    api_call_with_retry,
    build_cypher_type_condition,
    escape_cypher_string,
    process_batches_with_retry,
)
from neuronbridge_client import APIObject, Client
from neuronbridge_cache import ID_COLUMNS, IMAGE_COLUMNS, NeuronBridgeParquetCache


# ---------------------------------------------------------------------------
# utils/api_utils.py
# ---------------------------------------------------------------------------


class _RetryRecorder:
    def __init__(self):
        self.calls = []
        self.sleeps = []

    def on_retry(self, attempt, exc):
        self.calls.append((attempt, exc))


def test_api_call_with_retry_success(monkeypatch):
    monkeypatch.setattr(api_utils.time, "sleep", lambda *_a, **_k: None)
    assert api_call_with_retry(lambda: 42, timeout=5, max_retries=2) == 42


def test_api_call_with_retry_timeout_retries_then_raises(monkeypatch):
    import time as time_module

    real_sleep = time_module.sleep  # captured before patching below
    sleeps = []
    monkeypatch.setattr(api_utils.time, "sleep", lambda delay: sleeps.append(delay))
    recorder = _RetryRecorder()

    def hang():
        real_sleep(0.6)

    with pytest.raises(APITimeoutError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            api_call_with_retry(
                hang,
                timeout=0.05,
                max_retries=2,
                retry_delay=0.01,
                description="hanging call",
                on_retry=recorder.on_retry,
                verbose=True,
            )

    assert len(recorder.calls) == 1
    assert recorder.calls[0][0] == 1
    assert isinstance(recorder.calls[0][1], APITimeoutError)
    assert sleeps == [0.01]  # exponential backoff delay before the retry


def test_api_call_with_retry_abandoned_call_runs_on_daemon_thread():
    """An abandoned (timed-out) API call must not block interpreter exit.

    ThreadPoolExecutor workers are non-daemon and joined by
    concurrent.futures at shutdown, so one request stuck on a half-open
    socket kept the generated DROCAT script alive after its final
    "[DROCAT] Done." line: the UI waited for the process forever, showing
    no FINISHED line, run guide, or pair report.
    """
    import threading

    before = set(threading.enumerate())
    release = threading.Event()
    with pytest.raises(APITimeoutError):
        api_call_with_retry(
            lambda: release.wait(30),
            timeout=0.05,
            max_retries=1,
            retry_delay=0.0,
            verbose=False,
        )
    abandoned = [
        thread for thread in threading.enumerate()
        if thread not in before and thread.name.startswith("drocat-api-call")
    ]
    assert abandoned, "the timed-out call should still be running"
    assert all(thread.daemon for thread in abandoned)
    release.set()


def test_api_call_with_retry_generic_error_then_success(monkeypatch):
    monkeypatch.setattr(api_utils.time, "sleep", lambda *_a, **_k: None)
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ValueError("transient")
        return "ok"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = api_call_with_retry(
            flaky, timeout=5, max_retries=5, retry_delay=0.001, verbose=True
        )
    assert result == "ok"
    assert attempts["count"] == 3


def test_api_call_with_retry_exhausted(monkeypatch):
    monkeypatch.setattr(api_utils.time, "sleep", lambda *_a, **_k: None)

    def always_fails():
        raise RuntimeError("boom")

    with pytest.raises(APIRetryExhaustedError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            api_call_with_retry(
                always_fails, timeout=5, max_retries=2, retry_delay=0.001, verbose=True
            )


def test_api_call_with_retry_cancel_before_start(monkeypatch):
    """A pre-set cancel_event aborts before the call is even submitted."""
    import threading

    monkeypatch.setattr(api_utils.time, "sleep", lambda *_a, **_k: None)
    cancel = threading.Event()
    cancel.set()
    calls = {"n": 0}

    def never_called():
        calls["n"] += 1
        return 1

    with pytest.raises(APICancelError):
        api_call_with_retry(
            never_called, timeout=5, max_retries=2, cancel_event=cancel
        )
    assert calls["n"] == 0


def test_api_call_with_retry_cancel_aborts_in_flight_wait(monkeypatch):
    """A cancel_event set while the call hangs aborts the wait within ~0.5 s
    instead of waiting out the full timeout (Settings-tab cancel latency)."""
    import threading
    import time as time_module

    monkeypatch.setattr(api_utils.time, "sleep", lambda *_a, **_k: None)
    cancel = threading.Event()
    release = threading.Event()

    def hang():
        release.wait(30)  # would hang until released if waited on
        return 1

    cancel_timer = threading.Timer(0.3, cancel.set)
    cancel_timer.start()
    t0 = time_module.perf_counter()
    with pytest.raises(APICancelError):
        api_call_with_retry(
            hang, timeout=30, max_retries=2, cancel_event=cancel,
            description="hung call",
        )
    elapsed = time_module.perf_counter() - t0
    cancel_timer.cancel()
    release.set()  # let the worker thread finish so the test can exit cleanly
    assert elapsed < 5.0, f"cancel took {elapsed:.1f}s"


def test_api_call_with_retry_cancel_aborts_backoff_sleep(monkeypatch):
    """A cancel_event set during retry backoff aborts the sleep instead of
    sleeping out the exponential delay."""
    import threading
    import time as time_module

    monkeypatch.setattr(api_utils.time, "sleep", lambda *_a, **_k: None)
    cancel = threading.Event()

    def always_fails():
        raise RuntimeError("boom")

    cancel_timer = threading.Timer(0.2, cancel.set)
    cancel_timer.start()
    t0 = time_module.perf_counter()
    with pytest.raises(APICancelError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            api_call_with_retry(
                always_fails, timeout=5, max_retries=5, retry_delay=10,
                cancel_event=cancel, verbose=True,
            )
    elapsed = time_module.perf_counter() - t0
    cancel_timer.cancel()
    # The 10 s retry backoff was interrupted by the cancel.
    assert elapsed < 3.0, f"backoff sleep was not interrupted ({elapsed:.1f}s)"


def test_escape_cypher_string():
    assert escape_cypher_string("KCa'b'-ap1") == "KCa\\'b\\'-ap1"
    assert escape_cypher_string("a\\b") == "a\\\\b"
    assert escape_cypher_string("plain") == "plain"
    # Non-string values are coerced to text
    assert escape_cypher_string(12345) == "12345"


def test_build_cypher_type_condition_all_branches():
    assert build_cypher_type_condition("DNa01") == "n.type = 'DNa01'"
    assert build_cypher_type_condition("KC.*") == "n.type =~ 'KC.*'"
    assert build_cypher_type_condition("KC*") == "n.type =~ 'KC*'"
    assert build_cypher_type_condition("KCa'b") == "n.type = 'KCa\\'b'"
    assert build_cypher_type_condition(12345) == "n.bodyId = 12345"
    assert build_cypher_type_condition([1, 2, 3]) == "n.bodyId IN [1, 2, 3]"
    assert build_cypher_type_condition("T1", alias="m", type_column="cellType") == (
        "m.cellType = 'T1'"
    )
    with pytest.raises(ValueError):
        build_cypher_type_condition(3.14)


def test_process_batches_with_retry_success_and_failures(monkeypatch):
    monkeypatch.setattr(api_utils.time, "sleep", lambda *_a, **_k: None)

    def process(batch):
        if batch == "bad":
            raise RuntimeError("nope")
        if batch == "none":
            return None
        return f"done-{batch}"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = process_batches_with_retry(
            batches=["a", "none", "bad", "b"],
            process_func=process,
            timeout=5,
            max_retries=1,
            description_prefix="Testing",
            show_progress=True,
            verbose=True,
        )

    assert results == ["done-a", "done-b"]  # None results skipped
    texts = [str(w.message) for w in caught]
    assert any("Batch 3 failed permanently" in t for t in texts)
    assert any("1 batches failed" in t for t in texts)


def test_process_batches_with_retry_single_batch_no_progress_bar(monkeypatch):
    monkeypatch.setattr(api_utils.time, "sleep", lambda *_a, **_k: None)
    results = process_batches_with_retry(
        batches=["only"],
        process_func=lambda batch: [batch],
        timeout=5,
        max_retries=1,
        show_progress=True,
        verbose=False,
    )
    assert results == [["only"]]


# ---------------------------------------------------------------------------
# neuronbridge_client.py
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, *, text="", payload=None, content=b"", status=200,
                 json_error=None, status_error=None):
        self.text = text
        self._payload = payload
        self.content = content
        self.raw = BytesIO(content)
        self.status_code = status
        self.closed = False
        self._json_error = json_error
        self._status_error = status_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload

    def raise_for_status(self):
        if self._status_error is not None:
            raise self._status_error
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.urls = []

    def get(self, url, **_kwargs):
        self.urls.append(url)
        return self.responses[url]


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 3), "blue").save(buffer, format="PNG")
    return buffer.getvalue()


def _full_config(version_root):
    return {
        "stores": {
            "brain": {
                "prefixes": {
                    "CDSResults": f"{version_root}/matches/",
                    "PPPMResults": f"{version_root}/ppp/",
                    "CDM": f"{version_root}/images/",
                    "CDMThumbnail": f"{version_root}/thumbs/",
                    "CDMInput": f"{version_root}/input/",
                    "CDMMatch": f"{version_root}/match/",
                    "CDMBest": f"{version_root}/best/",
                    "CDMBestThumbnail": f"{version_root}/bestthumbs/",
                    "AlignedBodySWC": f"{version_root}/swc/",
                    "VisuallyLosslessStack": f"{version_root}/stacks/",
                }
            }
        }
    }


def _make_client(responses, version="v-test"):
    root = "https://test-bucket.s3.us-east-1.amazonaws.com"
    session = FakeSession(responses)
    client = Client(data_bucket="test-bucket", version=version, session=session)
    return client, session, root


def test_client_empty_current_txt_raises():
    root = "https://test-bucket.s3.us-east-1.amazonaws.com"
    session = FakeSession({f"{root}/current.txt": FakeResponse(text="  \n")})
    with pytest.raises(RuntimeError, match="empty version"):
        Client(data_bucket="test-bucket", session=session)


def test_client_http_error_wrapped():
    root = "https://test-bucket.s3.us-east-1.amazonaws.com"
    response = FakeResponse(
        status=500, status_error=requests.exceptions.HTTPError("HTTP 500")
    )
    session = FakeSession({f"{root}/current.txt": response})
    with pytest.raises(RuntimeError, match="Could not retrieve"):
        Client(data_bucket="test-bucket", session=session)
    assert response.closed


def test_client_invalid_json_wrapped():
    responses = {
        "https://test-bucket.s3.us-east-1.amazonaws.com/v-test/config.json":
            FakeResponse(json_error=ValueError("bad json"))
    }
    with pytest.raises(RuntimeError, match="invalid JSON"):
        _make_client(responses)


def test_client_get_text_and_swc_skeleton():
    version_root = "https://test-bucket.s3.us-east-1.amazonaws.com/v-test"
    swc_text = "1 1 0 0 0 1 -1\n"
    responses = {
        f"{version_root}/config.json": FakeResponse(payload=_full_config(version_root)),
        f"{version_root}/swc/body.swc": FakeResponse(text=swc_text),
    }
    client, _session, _root = _make_client(responses)
    match = APIObject(files={"store": "brain", "AlignedBodySWC": "body.swc"})
    assert client.get_swc_skeleton(match) == swc_text


def test_client_files_url_absolute_and_missing_prefix():
    version_root = "https://test-bucket.s3.us-east-1.amazonaws.com/v-test"
    responses = {
        f"{version_root}/config.json": FakeResponse(payload=_full_config(version_root)),
    }
    client, _session, _root = _make_client(responses)

    # Absolute URLs are returned untouched
    files = APIObject(store="brain", CDM="https://other.example.com/img.png")
    assert client._get_files_url(files, "CDM") == "https://other.example.com/img.png"

    # Missing path returns None
    assert client._get_files_url(APIObject(store="brain"), "CDM") is None

    # Unknown store -> no prefix -> RuntimeError
    with pytest.raises(RuntimeError, match="no prefix"):
        client._get_files_url(APIObject(store="unknown-store", CDM="img.png"), "CDM")


def test_client_match_url_missing_file_raises():
    version_root = "https://test-bucket.s3.us-east-1.amazonaws.com/v-test"
    responses = {
        f"{version_root}/config.json": FakeResponse(payload=_full_config(version_root)),
    }
    client, _session, _root = _make_client(responses)
    match = APIObject()  # no files at all
    with pytest.raises(RuntimeError, match="no file with type"):
        client._get_match_url(match, "CDM")
    # Match whose nested image carries the files is also supported
    match2 = APIObject(image={"files": {"store": "brain", "CDM": "x.png"}})
    assert client._get_match_url(match2, "CDM") == f"{version_root}/images/x.png"


def test_client_get_em_image_none_and_ppp_matches_empty():
    version_root = "https://test-bucket.s3.us-east-1.amazonaws.com/v-test"
    responses = {
        f"{version_root}/config.json": FakeResponse(payload=_full_config(version_root)),
        f"{version_root}/metadata/by_body/7.json": FakeResponse(payload={"results": []}),
    }
    client, _session, _root = _make_client(responses)
    assert client.get_em_image(7) is None

    # Image without PPPMResults yields an empty match list without HTTP calls
    image = APIObject(files={"store": "brain", "CDSResults": "x.json"})
    assert client.get_ppp_matches(image) == []


def test_client_image_getters():
    version_root = "https://test-bucket.s3.us-east-1.amazonaws.com/v-test"
    png = _png_bytes()
    urls = {
        "CDM": f"{version_root}/images/full.png",
        "CDMThumbnail": f"{version_root}/thumbs/thumb.png",
        "CDMInput": f"{version_root}/input/input.png",
        "CDMMatch": f"{version_root}/match/match.png",
        "CDMBest": f"{version_root}/best/best.png",
        "CDMBestThumbnail": f"{version_root}/bestthumbs/bestthumb.png",
        "VisuallyLosslessStack": f"{version_root}/stacks/stack.png",
    }
    responses = {
        f"{version_root}/config.json": FakeResponse(payload=_full_config(version_root)),
    }
    for url in urls.values():
        responses[url] = FakeResponse(content=png)
    client, _session, _root = _make_client(responses)

    files = {"store": "brain"}
    files.update({key: path.rsplit("/", 1)[1] for key, path in urls.items()})
    match = APIObject(files=files)

    for getter in (
        lambda: client.get_cds_image(match),
        lambda: client.get_cds_image(match, thumbnail=True),
        lambda: client.get_target_searchable_image(match),
        lambda: client.get_match_searchable_image(match),
        lambda: client.get_ppp_image(match),
        lambda: client.get_ppp_image(match, thumbnail=True),
        lambda: client.get_image_stack(match),
    ):
        image = getter()
        assert image.size == (2, 3)


def test_api_object_mapping_access():
    obj = APIObject(a=1, nested={"b": 2})
    assert obj["a"] == 1
    assert obj["nested"]["b"] == 2
    assert obj.get("missing", "fallback") == "fallback"
    obj["c"] = {"d": 3}
    assert obj.c.d == 3
    assert dict(obj.items())["a"] == 1


# ---------------------------------------------------------------------------
# neuronbridge_cache.py
# ---------------------------------------------------------------------------


def test_cache_corrupted_manifest_recovers(tmp_path):
    cache = NeuronBridgeParquetCache(tmp_path, version="test")
    cache.root.mkdir(parents=True, exist_ok=True)
    cache.manifest_path.write_text("{not-valid-json")

    path = cache.ensure_manifest("v-abc")

    manifest = json.loads(path.read_text())
    assert manifest["format"] == "parquet"
    assert manifest["neuronbridge_version"] == "v-abc"


def test_cache_unreadable_parquet_returns_none(tmp_path):
    cache = NeuronBridgeParquetCache(tmp_path, version="test")
    id_path = cache.id_path("key1")
    id_path.parent.mkdir(parents=True, exist_ok=True)
    id_path.write_bytes(b"not a parquet file")
    assert cache.load_id("key1") is None

    img_path = cache.image_path("img1", "cds")
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(b"also not parquet")
    assert cache.load_image("img1", "cds") is None


def test_cache_normalize_id_edge_cases():
    # None / empty frames yield the canonical empty schema
    empty = NeuronBridgeParquetCache.normalize_id(None)
    assert list(empty.columns) == ID_COLUMNS
    assert empty.empty

    # Missing score column defaults to 0.0; missing string cols become ""
    frame = pd.DataFrame([{"line": "VT000001"}])
    normalized = NeuronBridgeParquetCache.normalize_id(frame)
    assert normalized.iloc[0]["score"] == 0.0
    assert normalized.iloc[0]["library"] == ""


def test_cache_normalize_image_edge_cases(tmp_path):
    empty = NeuronBridgeParquetCache.normalize_image(None)
    assert list(empty.columns) == IMAGE_COLUMNS
    assert empty.empty

    # Missing bodyId and score columns are filled with defaults
    frame = pd.DataFrame(
        [{"image_id": "em-1", "lm_sample": "lm-1", "match_type": "cds",
          "dataset": "hemibrain:v1.2.1"}]
    )
    normalized = NeuronBridgeParquetCache.normalize_image(frame)
    assert normalized.iloc[0]["bodyId"] == ""
    assert normalized.iloc[0]["score"] == 0.0


def test_cache_save_id_empty_and_merge(tmp_path):
    cache = NeuronBridgeParquetCache(tmp_path, version="test")

    # Empty frames are not written
    assert cache.save_id("key1", pd.DataFrame()) is None

    first = pd.DataFrame(
        [{"line": "VT000001", "library": "lib", "score": 0.5,
          "image_id": "i1", "match_type": "cds"}]
    )
    cache.save_id("key1", first)

    # A second save merges with the existing table and deduplicates
    second = pd.DataFrame(
        [
            {"line": "VT000001", "library": "lib", "score": 0.9,
             "image_id": "i1", "match_type": "cds"},
            {"line": "VT000002", "library": "lib", "score": 0.3,
             "image_id": "i2", "match_type": "cds"},
        ]
    )
    cache.save_id("key1", second)
    loaded = cache.load_id("key1")
    assert loaded is not None
    assert len(loaded) == 2
    top = loaded[loaded["line"] == "VT000001"].iloc[0]
    assert top["score"] == 0.9  # best score kept for duplicated edge


def test_cache_save_image_empty_and_merge(tmp_path):
    cache = NeuronBridgeParquetCache(tmp_path, version="test")
    assert cache.save_image("img1", "cds", pd.DataFrame()) is None

    row = {
        "bodyId": 42, "score": 0.5, "image_id": "em-1", "lm_sample": "img1",
        "match_type": "cds", "dataset": "hemibrain:v1.2.1",
    }
    cache.save_image("img1", "cds", pd.DataFrame([row]))
    better = dict(row, score=0.8)
    cache.save_image("img1", "cds", pd.DataFrame([better]))
    loaded = cache.load_image("img1", "cds")
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded.iloc[0]["score"] == 0.8


def test_cache_write_atomic_cleans_tmp_on_failure(tmp_path, monkeypatch):
    cache = NeuronBridgeParquetCache(tmp_path, version="test")
    frame = pd.DataFrame(
        [{"line": "VT000001", "library": "lib", "score": 0.5,
          "image_id": "i1", "match_type": "cds"}]
    )
    path = cache.id_path("broken")

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr("os.replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        cache._write_atomic(frame, path)

    leftovers = [p for p in path.parent.glob(".*.tmp")]
    assert leftovers == []  # temporary file removed in finally block


def test_cache_iter_parquet_files(tmp_path):
    cache = NeuronBridgeParquetCache(tmp_path, version="test")
    assert cache.iter_parquet_files() == []  # root does not exist yet

    frame = pd.DataFrame(
        [{"line": "VT000001", "library": "lib", "score": 0.5,
          "image_id": "i1", "match_type": "cds"}]
    )
    cache.save_id("key1", frame)
    files = cache.iter_parquet_files()
    assert len(files) == 1
    assert files[0].name.startswith("id_to_lines_")
