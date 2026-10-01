"""CodexPuller worker tests with the network and the converter stubbed."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import codex_downloader as cd  # noqa: E402
import FAFB_file_converter  # noqa: E402
from ui.codex_pull import CodexPuller, human_bytes  # noqa: E402

NECESSARY_KEYS = [
    "classification", "connections_princeton_no_threshold", "names",
    "coordinates", "neurons", "cell_stats", "consolidated_cell_types",
]


def _patch_all(monkeypatch, tmp_path, complete_keys=(), fail_on=None,
               cancel_on=None):
    downloads = tmp_path / "downloads"
    monkeypatch.setattr(cd, "downloads_dir_for", lambda *a, **k: downloads)
    done = set(complete_keys)
    monkeypatch.setattr(
        cd, "product_status",
        lambda key, dl: {
            "key": key,
            "filename": cd.PRODUCT_CATALOG[key].filename,
            "expected": cd.PRODUCT_CATALOG[key].size,
            "local": (cd.PRODUCT_CATALOG[key].size if key in done else 0),
            "complete": key in done,
        })
    seen = []

    def fake_download(key, dl, project_root=None, progress_callback=None,
                      cancel_event=None):
        seen.append(key)
        if progress_callback is not None:
            progress_callback(1, 2)
            progress_callback(2, 2)
        if fail_on is not None and key == fail_on:
            raise cd.CodexAuthError("refused")
        if cancel_on is not None and key == cancel_on:
            raise cd.CodexDownloadCancelled(f"{key} cancelled")
        done.add(key)
        return dl / cd.PRODUCT_CATALOG[key].filename

    monkeypatch.setattr(cd, "download_product", fake_download)
    conversions = {"count": 0}

    def fake_ensure(dataset_name, dataset_dir):
        conversions["count"] += 1
        return True

    monkeypatch.setattr(FAFB_file_converter, "ensure_flywire_data",
                        fake_ensure)
    return seen, conversions


def _wait(puller, timeout=10):
    puller._thread.join(timeout=timeout)
    assert not puller._thread.is_alive(), "worker thread did not finish"


def test_downloads_levels_in_order_and_converts(monkeypatch, tmp_path):
    seen, conversions = _patch_all(monkeypatch, tmp_path)
    puller = CodexPuller()
    assert puller.start([cd.LEVEL_NECESSARY, cd.LEVEL_SKELETON],
                        run_converter=True)
    _wait(puller)
    state = puller.state
    assert seen == NECESSARY_KEYS + ["skeleton_swc_files"]
    summary = state["summary"]
    assert len(summary["downloaded"]) == 8 and not summary["skipped"]
    assert summary["converted"] is True and conversions["count"] == 1
    assert state["done"] and not state["running"] and not state["error"]


def test_skips_complete_products(monkeypatch, tmp_path):
    seen, conversions = _patch_all(
        monkeypatch, tmp_path, complete_keys=set(NECESSARY_KEYS))
    puller = CodexPuller()
    assert puller.start([cd.LEVEL_NECESSARY], run_converter=True)
    _wait(puller)
    assert seen == []
    summary = puller.state["summary"]
    assert summary["skipped"] == NECESSARY_KEYS
    assert summary["converted"] is True and conversions["count"] == 1


def test_convert_only_without_levels(monkeypatch, tmp_path):
    seen, conversions = _patch_all(
        monkeypatch, tmp_path, complete_keys=set(NECESSARY_KEYS))
    puller = CodexPuller()
    assert puller.start([], run_converter=True)
    _wait(puller)
    assert seen == []
    assert puller.state["summary"]["converted"] is True
    assert conversions["count"] == 1


def test_converter_skipped_when_necessary_incomplete(monkeypatch, tmp_path):
    seen, conversions = _patch_all(
        monkeypatch, tmp_path, complete_keys={"classification"})
    puller = CodexPuller()
    assert puller.start([], run_converter=True)
    _wait(puller)
    summary = puller.state["summary"]
    assert summary["converted"] is False
    assert "incomplete" in summary["convert_note"]
    assert conversions["count"] == 0


def test_cancel_mid_download(monkeypatch, tmp_path):
    seen, _ = _patch_all(monkeypatch, tmp_path, cancel_on="names")
    puller = CodexPuller()
    assert puller.start([cd.LEVEL_NECESSARY], run_converter=True)
    _wait(puller)
    state = puller.state
    assert state["cancelled"] is True and not state["error"]
    assert "names cancelled" in state["summary"]["cancelled_during"]
    assert state["summary"]["downloaded"] == NECESSARY_KEYS[:2]


def test_error_path(monkeypatch, tmp_path):
    _patch_all(monkeypatch, tmp_path, fail_on="names")
    puller = CodexPuller()
    assert puller.start([cd.LEVEL_NECESSARY], run_converter=True)
    _wait(puller)
    state = puller.state
    assert state["error"] and "CodexAuthError" in state["error"]
    assert state["done"] and not state["running"]


def test_start_rejects_while_running():
    puller = CodexPuller()
    with puller._lock:
        puller._state["running"] = True
    assert puller.start([cd.LEVEL_NECESSARY]) is False
    with puller._lock:
        puller._state["running"] = False


def test_human_bytes():
    assert human_bytes(512) == "512 B"
    assert human_bytes(275_679_780) == "262.9 MB"
    assert human_bytes(13_873_645_070) == "12.9 GB"


def test_fafb_synapse_warning_gates(monkeypatch, tmp_path):
    """The warning fires only for FAFB + non-skip views + a missing local
    synapse table, and names the Settings download path."""
    import codex_downloader as cd
    from ui.tabs.visualization import fafb_synapse_warning

    monkeypatch.setattr(cd, "synapse_table_ready", lambda *a, **k: False)
    assert fafb_synapse_warning("flywire_FAFB_v783", "skip") is None
    assert fafb_synapse_warning("hemibrain:v1.2.1", "synapse") is None
    warning = fafb_synapse_warning("flywire_FAFB_v783", "synapse")
    assert warning and "synapse table" in warning
    assert "FAFB Dataset Downloads" in warning
    monkeypatch.setattr(cd, "synapse_table_ready", lambda *a, **k: True)
    assert fafb_synapse_warning("flywire_FAFB_v783", "synapse") is None


def test_banc_puller_levels_and_cancel(monkeypatch, tmp_path):
    """BancPuller walks the three levels in order; cancel after level 1."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    import banc_public_data as bpd
    import morphology
    from ui.banc_pull import (BancPuller, LEVEL_NECESSARY, LEVEL_ORDER,
                              LEVEL_SKELETONS, LEVEL_SYNAPSE, banc_level_status)

    calls = []

    def fake_prepare(dataset_name, dataset_dir, project_root=None):
        calls.append(("prepare", dataset_name))
        return True

    def fake_synapse(dataset, project_root=None, force=False,
                     progress_callback=None):
        calls.append(("synapse", dataset))
        return Path(tmp_path) / "synapse.parquet"

    def fake_skeletons(dataset, **kwargs):
        calls.append(("skeletons", dataset))
        return {"total_neurons": 1, "cancelled": False}

    monkeypatch.setattr(bpd, "prepare_dataset_tables", fake_prepare)
    monkeypatch.setattr(bpd, "ensure_synapse_table", fake_synapse)
    monkeypatch.setattr(morphology, "download_all_skeletons", fake_skeletons)

    puller = BancPuller()
    assert puller.start("banc_v888", list(LEVEL_ORDER),
                        project_root=str(tmp_path))
    puller._thread.join(timeout=10)
    assert not puller._thread.is_alive()
    state = puller.state
    assert [c[0] for c in calls] == ["prepare", "synapse", "skeletons"]
    assert state["done"] and not state["error"] and not state["cancelled"]
    assert state["summary"]["completed"] == list(LEVEL_ORDER)

    # failure path: preparation returns False
    monkeypatch.setattr(bpd, "prepare_dataset_tables",
                        lambda *a, **k: False)
    puller2 = BancPuller()
    assert puller2.start("banc_v888", [LEVEL_NECESSARY],
                         project_root=str(tmp_path))
    puller2._thread.join(timeout=10)
    state2 = puller2.state
    assert state2["error"] and "did not complete" in state2["error"]


def test_banc_level_status(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from ui.banc_pull import LEVEL_NECESSARY, LEVEL_SYNAPSE, banc_level_status

    dataset_dir = tmp_path / "datasets" / "banc_v888"
    dataset_dir.mkdir(parents=True)
    status = banc_level_status("banc_v888", project_root=str(tmp_path))
    assert status[LEVEL_NECESSARY]["complete"] is False
    assert status[LEVEL_SYNAPSE]["complete"] is False

    (dataset_dir / "banc_v888_merged_connections.parquet").write_bytes(b"x")
    (dataset_dir / "banc_v888_allneurons_neuron_df.parquet").write_bytes(b"x")
    status = banc_level_status("banc_v888", project_root=str(tmp_path))
    assert status[LEVEL_NECESSARY]["complete"] is True
