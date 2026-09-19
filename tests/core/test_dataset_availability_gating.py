"""Dataset-availability gating: pull-memo classification + the core gate.

Two things are pinned here, both motivated by the Windows field-agent run that
turned one missing NeuPrint token into 59 echoed failures:

* ``statvis._ensure_local_dataset_files`` keeps memoizing "we pulled and the
  tables are still not there" (the per-layer loop in
  ``visualize_skeleton.SkeletonVisualizer`` would otherwise re-attempt the same
  doomed pull once per layer), but an authentication/offline/network failure
  must leave the process exactly as if no attempt had been made.
* ``tests/core/conftest.py`` answers the availability question without
  touching the network, and its ``requires_data`` / ``requires_token`` markers
  turn a missing prerequisite into a skip that names the missing piece.

No test in this module performs network I/O: every pull failure is a
monkeypatched ``pull_dataset``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_PROJECT_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import statvis as sv  # noqa: E402


def _load_core_conftest():
    """Import ``tests/core/conftest.py`` as a plain module (it is also a plugin)."""
    try:
        from tests.core import conftest as module
        return module
    except Exception:  # pragma: no cover - fall back to loading by path
        import importlib.util
        path = Path(__file__).with_name("conftest.py")
        spec = importlib.util.spec_from_file_location("drocat_core_conftest", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


core_conftest = _load_core_conftest()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def gate_dir(tmp_path, monkeypatch):
    """Redirect one synthetic dataset at a tmp dir; return its file prefix."""
    root = tmp_path / "datasets" / "gate_ds_v1"
    body = root / "gate_ds_v1_allneurons"

    def fake_path_body(dataset):
        return "gate_ds_v1", str(root), str(body)

    monkeypatch.setattr(sv, "_get_dataset_path_body", fake_path_body)
    monkeypatch.setattr(sv, "clear_neuron_cache", lambda *a, **k: None)
    return body


def _write_tables(save_path):
    """Materialize the files a completed pull is contractually owed."""
    os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
    pd.DataFrame({"bodyId": [1], "type": ["Mi1"]}).to_csv(
        str(save_path) + "_neuron_df.csv", index=False)
    pd.DataFrame({"bodyId": [1], "roi": ["AL"], "pre": [1], "post": [0]}).to_parquet(
        str(save_path) + "_roi_count_df.parquet")


# Failure modes that say something about *this machine's access* to NeuPrint
# and nothing about the local files.  The texts are copied from the real raise
# sites (neuprint's Client, its ``verbose_errors`` wrapper,
# statvis.pull_dataset, requests/urllib3/socket).
_ENVIRONMENT_FAILURES = [
    pytest.param(
        RuntimeError(
            "Dataset 'gate-ds:v1' is not available locally and no NeuPrint "
            "connection exists.\n   (No default Client has been set yet.)"),
        id="no_client"),
    pytest.param(
        RuntimeError(
            "No token provided. Please provide one or set "
            "NEUPRINT_APPLICATION_CREDENTIALS"),
        id="no_token"),
    pytest.param(
        RuntimeError(
            "Did not understand token. Please provide the entire JSON document "
            "or (only) the complete token string"),
        id="malformed_token"),
    pytest.param(
        RuntimeError("Returned Error (401)\n\nInvalid token"),
        id="token_rejected"),
    pytest.param(
        ConnectionError(
            "[WinError 10065] A connection attempt failed because the "
            "connected party did not properly respond"),
        id="offline_socket"),
    pytest.param(
        TimeoutError("Neuron list query timed out after 120.0s (attempt 5/5)"),
        id="timeout"),
    pytest.param(
        RuntimeError(
            "Failed to download neurons for gate-ds:v1: every batch failed "
            "after retries (server unreachable). Check the connection and "
            "re-run."),
        id="server_unreachable"),
    pytest.param(
        sv.DatasetPullCancelled("Dataset pull cancelled."),
        id="user_cancel"),
]


# ---------------------------------------------------------------------------
# 1. environment failures must not poison the rest of the process
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("failure", _ENVIRONMENT_FAILURES)
def test_environment_failure_is_not_memoized(gate_dir, monkeypatch, failure):
    """A later call must attempt the pull again and see the real cause."""
    attempts = []

    def failing_pull(dataset, save_path=None, **kwargs):
        attempts.append(dataset)
        raise failure

    monkeypatch.setattr(sv, "pull_dataset", failing_pull)

    for attempt in range(1, 4):
        with pytest.raises(type(failure)) as excinfo:
            sv._ensure_local_dataset_files("gate-ds:v1", verbose=False)
        assert len(attempts) == attempt, "each call site must try the pull"
        assert "still missing local data files" not in str(excinfo.value)
    assert "gate_ds_v1" not in sv._FAILED_DATASET_DOWNLOADS


def test_environment_failure_recovers_once_access_exists(gate_dir, monkeypatch):
    """Tokenless first, token configured before the retry: it must succeed."""
    state = {"remaining": 1}

    def flaky_pull(dataset, save_path=None, **kwargs):
        if state["remaining"]:
            state["remaining"] -= 1
            raise RuntimeError(
                "Dataset 'gate-ds:v1' is not available locally and no NeuPrint "
                "connection exists.")
        _write_tables(save_path)

    monkeypatch.setattr(sv, "pull_dataset", flaky_pull)
    with pytest.raises(RuntimeError):
        sv._ensure_local_dataset_files("gate-ds:v1", verbose=False)
    normalized, body = sv._ensure_local_dataset_files("gate-ds:v1", verbose=False)
    assert normalized == "gate_ds_v1"
    assert Path(body).name == "gate_ds_v1_allneurons"
    assert "gate_ds_v1" not in sv._FAILED_DATASET_DOWNLOADS


def test_requests_transport_errors_are_classified_as_environment():
    """requests' hierarchy is OSError-based, not local-filesystem-based."""
    requests = pytest.importorskip("requests")
    for exc in (
            requests.exceptions.ConnectionError("Max retries exceeded"),
            requests.exceptions.HTTPError("Returned Error (403)"),
            requests.exceptions.ReadTimeout("Read timed out")):
        assert sv._is_environment_dataset_pull_error(exc), exc


# ---------------------------------------------------------------------------
# 2. the memoization the per-layer visualization loop depends on is intact
# ---------------------------------------------------------------------------

def test_pull_that_wrote_nothing_fast_fails_the_next_time(gate_dir, monkeypatch):
    """"The pull completed but left no tables" stays memoized."""
    attempts = []

    def no_output_pull(dataset, save_path=None, **kwargs):
        attempts.append(dataset)

    monkeypatch.setattr(sv, "pull_dataset", no_output_pull)
    with pytest.raises(FileNotFoundError) as first:
        sv._ensure_local_dataset_files("gate-ds:v1", verbose=False)
    assert "completed without creating the expected files" in str(first.value)
    assert "gate_ds_v1" in sv._FAILED_DATASET_DOWNLOADS

    with pytest.raises(FileNotFoundError) as second:
        sv._ensure_local_dataset_files("gate-ds:v1", verbose=False)
    assert ("still missing local data files after a previous pull attempt"
            in str(second.value))
    assert len(attempts) == 1, "the loop must not re-run a doomed pull"


def test_unclassifiable_pull_failure_keeps_the_conservative_memo(
        gate_dir, monkeypatch):
    """An error with no environment signature is still treated as an attempt.

    ``test_statvis_coverage.TestEnsureLocalDatasetFiles::test_pull_failure_recorded``
    pins that behaviour, so the classifier may only un-memoize what it can
    actually recognize.
    """
    attempts = []

    def failing_pull(dataset, save_path=None, **kwargs):
        attempts.append(dataset)
        raise RuntimeError("offline")

    monkeypatch.setattr(sv, "pull_dataset", failing_pull)
    with pytest.raises(RuntimeError):
        sv._ensure_local_dataset_files("gate-ds:v1", verbose=False)
    assert "gate_ds_v1" in sv._FAILED_DATASET_DOWNLOADS
    with pytest.raises(FileNotFoundError):
        sv._ensure_local_dataset_files("gate-ds:v1", verbose=False)
    assert len(attempts) == 1


def test_local_filesystem_errors_are_not_environment():
    """A path/permission failure is about the files, so it stays memoized."""
    assert not sv._is_environment_dataset_pull_error(
        PermissionError(13, "Permission denied", r"D:\repo\datasets\gate\x.csv"))
    assert not sv._is_environment_dataset_pull_error(
        FileNotFoundError(2, "No such file or directory", r"D:\repo\datasets\gate"))
    # A network failure underneath a local-looking error is still the network.
    wrapped = FileNotFoundError(2, "No such file or directory")
    wrapped.__cause__ = ConnectionResetError("Connection reset by peer")
    assert sv._is_environment_dataset_pull_error(wrapped)
    # Unknown exceptions default to "memoize" (conservative).
    assert not sv._is_environment_dataset_pull_error(RuntimeError("boom"))


def test_clear_failed_dataset_downloads_resets_the_memo():
    sv._FAILED_DATASET_DOWNLOADS.update({"hemibrain_v1_2_1", "other_ds"})
    try:
        sv.clear_failed_dataset_downloads("hemibrain:v1.2.1")
        assert "hemibrain_v1_2_1" not in sv._FAILED_DATASET_DOWNLOADS
        assert "other_ds" in sv._FAILED_DATASET_DOWNLOADS
    finally:
        sv.clear_failed_dataset_downloads()
    assert sv._FAILED_DATASET_DOWNLOADS == set()


def test_autouse_reset_fixture_clears_the_memo_between_tests():
    """Order-independence: this test starts from an empty memo."""
    assert sv._FAILED_DATASET_DOWNLOADS == set()


# ---------------------------------------------------------------------------
# 3. the availability gate itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", [
    "hemibrain:v1.2.1", "male-cns:v0.9", "manc:v1.2.3",
    "nonexistent-ds:v9.9", "banc:v888", "flywire_FAFB_v783",
])
def test_table_path_derivation_matches_statvis(dataset):
    """The gate looks for exactly the files statvis looks for."""
    neuron, roi = core_conftest.dataset_table_paths(dataset)
    _normalized, _dir, body = sv._get_dataset_path_body(dataset)
    assert str(neuron) == body + "_neuron_df.csv"
    assert str(roi) == sv.roi_count_table_path(body)


def test_table_paths_follow_the_folder_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(core_conftest, "DATASETS_DIR", tmp_path)
    # No dataset folder yet: statvis looks for the flat "datasets/<name>_allneurons"
    # spelling, so the gate must not invent a folder that is not there.
    flat_neuron, _flat_roi = core_conftest.dataset_table_paths("hemibrain:v1.2.1")
    assert flat_neuron == tmp_path / "hemibrain_v1_2_1_allneurons_neuron_df.csv"
    assert core_conftest.dataset_files_present("hemibrain:v1.2.1") is False

    folder = tmp_path / "hemibrain_v1_2_1"
    folder.mkdir()
    neuron, roi = core_conftest.dataset_table_paths("hemibrain:v1.2.1")
    assert neuron == folder / "hemibrain_v1_2_1_allneurons_neuron_df.csv"
    assert roi == folder / "hemibrain_v1_2_1_allneurons_roi_count_df.csv"

    neuron.write_text("bodyId\n1\n", encoding="utf-8")
    # both tables are required -- a lone neuron CSV is not "available"
    assert core_conftest.dataset_files_present("hemibrain:v1.2.1") is False
    (folder / "hemibrain_v1_2_1_allneurons_roi_count_df.parquet").write_bytes(b"x")
    assert core_conftest.dataset_files_present("hemibrain:v1.2.1") is True
    assert core_conftest.dataset_table_paths("hemibrain:v1.2.1")[1] == (
        folder / "hemibrain_v1_2_1_allneurons_roi_count_df.parquet")


def test_roi_table_accepts_the_legacy_csv_spelling(tmp_path, monkeypatch):
    monkeypatch.setattr(core_conftest, "DATASETS_DIR", tmp_path)
    (tmp_path / "manc_v1_0").mkdir()
    neuron, roi = core_conftest.dataset_table_paths("manc:v1.0")
    neuron.write_text("bodyId\n1\n", encoding="utf-8")
    assert roi.name == "manc_v1_0_allneurons_roi_count_df.csv"
    roi.write_text("roi\nAL\n", encoding="utf-8")
    assert core_conftest.dataset_files_present("manc:v1.0") is True


def test_token_configured_reads_the_chain_without_probing(monkeypatch):
    import utils.token_manager as tm_module
    for var in core_conftest._TOKEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(tm_module.token_manager, "tokens", {}, raising=False)
    assert core_conftest.token_configured() is False

    # The clean-checkout state: config.json ships an empty/placeholder token.
    monkeypatch.setattr(tm_module.token_manager, "tokens",
                        {"NEUPRINT_TOKEN": "YOUR_NEUPRINT_TOKEN"}, raising=False)
    assert core_conftest.token_configured() is False, "placeholder is not a token"

    monkeypatch.setenv("NEUPRINT_APPLICATION_CREDENTIALS", "")
    assert core_conftest.token_configured() is False, "empty env var is not a token"

    monkeypatch.setattr(tm_module.token_manager, "tokens", {}, raising=False)
    monkeypatch.setenv(
        "NEUPRINT_APPLICATION_CREDENTIALS", "eyJhbGciOiJIUzI1NiJ9." + "x" * 40)
    assert core_conftest.token_configured() is True

    monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS")
    monkeypatch.setattr(tm_module.token_manager, "tokens",
                        {"NEUPRINT_TOKEN": "config-file-token"}, raising=False)
    assert core_conftest.token_configured() is True


def test_availability_verdict_is_token_or_tables(monkeypatch, tmp_path):
    monkeypatch.setattr(core_conftest, "DATASETS_DIR", tmp_path)
    monkeypatch.setattr(core_conftest, "token_configured", lambda: False)
    verdict = core_conftest.DatasetAvailability()
    assert verdict("hemibrain:v1.2.1") is False
    assert verdict.tables_present("hemibrain:v1.2.1") is False
    assert "no NeuPrint token" in verdict.explain("hemibrain:v1.2.1")

    monkeypatch.setattr(core_conftest, "token_configured", lambda: True)
    assert core_conftest.DatasetAvailability()("hemibrain:v1.2.1") is True


def test_require_skips_when_data_is_unreachable(monkeypatch, tmp_path):
    monkeypatch.setattr(core_conftest, "DATASETS_DIR", tmp_path)
    verdict = core_conftest.DatasetAvailability(token_available=False)
    with pytest.raises(pytest.skip.Exception) as excinfo:
        verdict.require("hemibrain:v1.2.1")
    reason = str(excinfo.value)
    assert "no NeuPrint token" in reason
    assert "hemibrain:v1.2.1" in reason


class _FakeItem:
    """Just enough of a pytest Item for the collection hook."""

    def __init__(self, markers):
        self._markers = markers
        self.added = []

    def get_closest_marker(self, name):
        return self._markers.get(name)

    def add_marker(self, marker):
        self.added.append(marker)


def test_collection_gate_skips_only_what_it_must(monkeypatch, tmp_path):
    monkeypatch.setattr(core_conftest, "DATASETS_DIR", tmp_path)
    monkeypatch.setattr(core_conftest, "token_configured", lambda: False)
    missing = _FakeItem({"requires_data": pytest.mark.requires_data(
        "hemibrain:v1.2.1")})
    token_only = _FakeItem({"requires_token": pytest.mark.requires_token()})
    bare = _FakeItem({"requires_data": pytest.mark.requires_data()})
    core_conftest.pytest_collection_modifyitems(
        config=None, items=[missing, token_only, bare])

    assert [m.name for m in missing.added] == ["skip"]
    assert "requires_data" in missing.added[0].kwargs["reason"]
    assert [m.name for m in token_only.added] == ["skip"]
    assert "requires_token" in token_only.added[0].kwargs["reason"]
    assert bare.added, "bare requires_data gates on the default dataset"

    # With a token configured nothing is gated: the pull can fetch the tables.
    monkeypatch.setattr(core_conftest, "token_configured", lambda: True)
    again = _FakeItem({"requires_data": pytest.mark.requires_data("nope:v1")})
    core_conftest.pytest_collection_modifyitems(config=None, items=[again])
    assert again.added == []


def test_collection_gate_passes_when_tables_are_local(monkeypatch, tmp_path):
    monkeypatch.setattr(core_conftest, "DATASETS_DIR", tmp_path)
    monkeypatch.setattr(core_conftest, "token_configured", lambda: False)
    (tmp_path / "male-cns_v1_0").mkdir()
    neuron, roi = core_conftest.dataset_table_paths("male-cns:v1.0")
    neuron.write_text("bodyId\n1\n", encoding="utf-8")
    roi.write_bytes(b"x")
    item = _FakeItem(
        {"requires_data": pytest.mark.requires_data(["male-cns:v1.0"])})
    core_conftest.pytest_collection_modifyitems(config=None, items=[item])
    assert item.added == []


def test_gate_and_markers_agree_with_the_session_fixture(
        local_dataset_available):
    """The session fixture answers the same question the markers gate on."""
    assert isinstance(local_dataset_available, core_conftest.DatasetAvailability)
    assert bool(local_dataset_available()) == bool(
        local_dataset_available.token_available
        or core_conftest.dataset_files_present(core_conftest.DEFAULT_DATASET))


def test_markers_are_declared_in_pyproject():
    """Declared markers keep ``-W error::pytest.PytestUnknownMarkWarning`` usable."""
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10
        try:
            import tomli as tomllib
        except ImportError:
            pytest.skip("no TOML parser available")
    with open(_PROJECT_ROOT / "pyproject.toml", "rb") as stream:
        declared = tomllib.load(stream)["tool"]["pytest"]["ini_options"]["markers"]
    names = {entry.split(":")[0].strip() for entry in declared}
    assert {"e2e", "requires_data", "requires_token"} <= names, names


@pytest.mark.requires_data
def test_requires_data_marker_run_with_data(local_dataset_available):
    """Reached only when the default dataset can be obtained (marker works)."""
    assert local_dataset_available()


@pytest.mark.requires_token
def test_requires_token_marker_run_with_token():
    assert core_conftest.token_configured()
