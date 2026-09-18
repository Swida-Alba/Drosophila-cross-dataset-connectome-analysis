"""Settings → Storage card tests (ui/components/storage_card.py).

Element-level checks following the repo's NiceGUI Client(page) harness:
the card builds inside a fake client, interlock flips work, and selection
sync enables/disables the action buttons. The async scan/delete flows are
covered at the core level (tests/core/test_storage_inventory.py) — here
we keep to what is deterministic without a running server.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from nicegui import Client
from nicegui.page import page

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ui.components.storage_card import (  # noqa: E402
    StorageCard,
    _fmt_mtime,
    _fmt_ts,
    _truncate_lines,
    create_storage_card,
)


def _fake_pullers():
    return (SimpleNamespace(running=False), SimpleNamespace(running=False))


def _build_card():
    client = Client(page("/storage-card-test"))
    with client:
        card = create_storage_card(*_fake_pullers())
    return client, card


def _button_by_label(client, label):
    return next(el for el in client.elements.values()
                if type(el).__name__ == "Button" and el.text == label)


def _table_by_row_key(client, row_key):
    return next(el for el in client.elements.values()
                if type(el).__name__ == "Table" and el.row_key == row_key)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_fmt_ts_formats_and_falls_back():
    assert _fmt_ts("20260913_025744") == "2026-09-13 02:57:44"
    assert _fmt_ts("") == "—"
    assert _fmt_ts("junk") == "—"


def test_fmt_mtime_falls_back():
    assert _fmt_mtime(0.0) == "—"
    assert _fmt_mtime(1.0) != "—"


def test_truncate_lines():
    assert _truncate_lines(["a", "b"]) == "a\nb"
    assert _truncate_lines([str(i) for i in range(15)]) == (
        "\n".join(str(i) for i in range(12)) + "\n… and 3 more")


# ---------------------------------------------------------------------------
# Card construction + interlocks
# ---------------------------------------------------------------------------


def test_card_builds_with_tables_and_disabled_actions():
    client, card = _build_card()
    caches = _table_by_row_key(client, "key")
    runs = _table_by_row_key(client, "name")
    assert caches.selection == "multiple"
    assert runs.selection == "multiple"
    # Actions start disabled (no scan, no selection).
    assert _button_by_label(client, "Clear Selected…").enabled is False
    assert _button_by_label(client, "Prune Source Data…").enabled is False
    assert _button_by_label(client, "Delete Folder…").enabled is False
    assert _button_by_label(client, "Scan Now").enabled is True


def test_pull_interlock_disables_and_restores_scan():
    client, card = _build_card()
    scan = _button_by_label(client, "Scan Now")
    card.set_pull_active(True)
    assert scan.enabled is False
    assert "pull" in card.status_label.text.lower()
    card.set_pull_active(False)
    assert scan.enabled is True
    assert card.status_label.text == ""


def test_selection_sync_enables_actions():
    client, card = _build_card()
    caches = _table_by_row_key(client, "key")
    runs = _table_by_row_key(client, "name")

    caches.selected = [{"key": "connections:male-cns_v1_0"}]
    card._sync_action_buttons()
    assert _button_by_label(client, "Clear Selected…").enabled is True
    caches.selected = []
    card._sync_action_buttons()
    assert _button_by_label(client, "Clear Selected…").enabled is False

    runs.selected = [{
        "name": "NB-find-lines_MCNS_x_20260801_183020",
        "registered": True,
        "source_bytes": 150,
        "locked": False,
    }]
    card._sync_action_buttons()
    assert _button_by_label(client, "Prune Source Data…").enabled is True
    assert _button_by_label(client, "Delete Folder…").enabled is True

    # Locked rows (active runs) never enable the actions.
    runs.selected = [{
        "name": "NB-find-lines_MCNS_live_20260801_183021",
        "registered": True,
        "source_bytes": 150,
        "locked": True,
    }]
    card._sync_action_buttons()
    assert _button_by_label(client, "Prune Source Data…").enabled is False
    assert _button_by_label(client, "Delete Folder…").enabled is False


def test_prune_requires_registered_rows_with_source():
    client, card = _build_card()
    runs = _table_by_row_key(client, "name")
    runs.selected = [{
        "name": "find-paths-complete_MCNS_a_b_20260801_183000",
        "registered": False,
        "source_bytes": -1,
        "locked": False,
    }]
    card._sync_action_buttons()
    assert _button_by_label(client, "Prune Source Data…").enabled is False
    assert _button_by_label(client, "Delete Folder…").enabled is True


# ---------------------------------------------------------------------------
# Recorded output roots (ui/config registry)
# ---------------------------------------------------------------------------


def test_storage_scan_roots_roundtrip(tmp_path, monkeypatch):
    import ui.config as config_module
    cfg_file = tmp_path / "config_local.json"
    monkeypatch.setattr(config_module, "LOCAL_CONFIG_FILE", cfg_file)
    monkeypatch.setattr(config_module, "_LOCAL_CONFIG_CACHE",
                        {"mtime_ns": None, "data": {}})
    assert config_module.get_storage_scan_roots() == []
    assert config_module.add_storage_scan_root(tmp_path / "runs")
    # Deduplicated.
    assert config_module.add_storage_scan_root(tmp_path / "runs") is True
    roots = config_module.get_storage_scan_roots()
    assert roots == [str((tmp_path / "runs").resolve())]
    # Relative paths are resolved against the server CWD and recorded.
    assert config_module.add_storage_scan_root("relative/dir") is True
    assert len(config_module.get_storage_scan_roots()) == 2
    # Removal.
    config_module.remove_storage_scan_root(tmp_path / "runs")
    remaining = config_module.get_storage_scan_roots()
    assert remaining == [str((Path.cwd() / "relative/dir").resolve())]
    config_module.remove_storage_scan_root("relative/dir")
    assert config_module.get_storage_scan_roots() == []


def test_card_rows_carry_tooltip_and_open_paths(tmp_path):
    """scan_caches items expose open_path so the card can fill the
    hover title and the open-dir button."""
    import src.storage_inventory as si
    cache = tmp_path / "cache"
    ds = cache / "male-cns_v1_0"
    (ds / "skeletons").mkdir(parents=True)
    (ds / "skeletons" / "a.swc.zst").write_bytes(b"x" * 5)
    items = {i.key: i for i in si.scan_caches(
        cache_root=cache, index_root=tmp_path / "neuron_indexes")}
    assert items["skeletons:male-cns_v1_0"].open_path == str(ds)


def test_card_builds_with_body_slots_and_roots_row():
    client = Client(page("/storage-card-slots"))
    with client:
        card = create_storage_card(*_fake_pullers())
    tables = [el for el in client.elements.values()
              if type(el).__name__ == "Table"]
    slots = [el.slots.get("body").template for el in tables]
    assert all(s for s in slots), "body slot missing on a table"
    assert all("open_dir" in s for s in slots), "open-dir event missing"
    labels = [el.text for el in client.elements.values()
              if type(el).__name__ == "Label"]
    assert any("Recorded output roots" in (t or "") for t in labels)
