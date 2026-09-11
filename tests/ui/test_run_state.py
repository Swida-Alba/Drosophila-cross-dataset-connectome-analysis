"""Regression tests for durable UI execution state and page teardown."""

import json

import pytest


def test_run_state_store_replays_logs_and_redacts_credentials(tmp_path):
    from ui.run_state import RunStateStore

    store = RunStateStore(tmp_path)
    record = store.create(
        tab_key="cross dataset",
        title="Comparison Output",
        tool_name="inter_dataset",
        constructor_params={"token": "do-not-write"},
        method_params={"threshold": 3},
    )
    assert record["tab_key"] == "cross_dataset"
    assert record["input_summary"]["constructor"]["token"] == "[redacted]"

    store.append_log(record["run_id"], "token=do-not-write", "system")
    store.append_log(record["run_id"], "visible output", "stdout")

    assert store.read_logs(record["run_id"]) == [
        ("system", "token=[redacted]"),
        ("stdout", "visible output"),
    ]
    manifest = json.loads(
        (tmp_path / "runs" / f"{record['run_id']}.json").read_text()
    )
    assert manifest["input_summary"]["constructor"]["token"] == "[redacted]"


def test_run_manager_notifies_subscribers_registered_before_run(tmp_path):
    from ui.run_state import RunManager, RunStateStore

    manager = RunManager(RunStateStore(tmp_path))
    events = []
    manager.subscribe("network", events.append)

    record = manager.begin(
        tab_key="network",
        title="Network Output",
        tool_name="find_network",
    )
    manager.update_progress(record["run_id"], "execute", "Running network analysis")
    manager.append_log(record["run_id"], "network line")
    manager.finish(
        record["run_id"],
        {"returncode": 0, "files": [], "duration": 1.25, "output_folder": None},
    )

    assert [event["kind"] for event in events] == [
        "state",
        "progress",
        "log",
        "state",
    ]
    assert events[-1]["record"]["status"] == "Completed"
    assert manager.store.latest("network")["duration"] == pytest.approx(1.25)


def test_new_manager_reconciles_orphaned_running_record(tmp_path):
    from ui.run_state import RunManager, RunStateStore

    store = RunStateStore(tmp_path)
    first_manager = RunManager(store)
    record = first_manager.begin(
        tab_key="find_path",
        title="Pathfinding Output",
        tool_name="find_path",
    )

    second_manager = RunManager(store)
    interrupted = second_manager.reconcile()

    assert [item["run_id"] for item in interrupted] == [record["run_id"]]
    restored = store.latest("find_path")
    assert restored["status"] == "Interrupted"
    assert "not found" in restored["message"]


def test_manager_shutdown_terminates_active_backend(tmp_path):
    from ui.run_state import RunManager, RunStateStore

    manager = RunManager(RunStateStore(tmp_path), session_id="shutdown-session")
    record = manager.begin(
        tab_key="network",
        title="Network Output",
        tool_name="find_network",
    )

    class FakeRunner:
        is_running = True

        def __init__(self):
            self.cancel_calls = 0

        def cancel(self):
            self.cancel_calls += 1
            self.is_running = False

    runner = FakeRunner()
    manager.register_runner(record["run_id"], runner)
    manager.shutdown()
    manager.shutdown()

    restored = manager.store.read(record["run_id"])
    assert runner.cancel_calls == 1
    assert restored["status"] == "Interrupted"
    assert restored["shutdown_requested"] is True
    assert "terminated the backend" in restored["message"]


def test_startup_cleanup_removes_old_terminal_cache_only(tmp_path):
    from ui.run_state import RunManager, RunStateStore

    store = RunStateStore(tmp_path)
    old_manager = RunManager(store, session_id="old-session")
    old_record = old_manager.begin(
        tab_key="network",
        title="Old Network Output",
        tool_name="find_network",
    )
    old_manager.append_log(old_record["run_id"], "old cached line")
    old_manager.finish(
        old_record["run_id"],
        {"returncode": 0, "files": [], "duration": 0.1},
    )

    new_manager = RunManager(store, session_id="new-session")
    active_record = new_manager.begin(
        tab_key="find_path",
        title="Current Path Output",
        tool_name="find_path",
    )
    removed = new_manager.cleanup_stale_cache()

    assert removed == [old_record["run_id"]]
    assert store.read(old_record["run_id"]) is None
    assert store.read_logs(old_record["run_id"]) == []
    assert store.read(active_record["run_id"])["status"] == "Running"


def test_startup_cleanup_removes_orphaned_active_cache_after_reconcile(tmp_path):
    from ui.run_state import RunManager, RunStateStore

    store = RunStateStore(tmp_path)
    old_manager = RunManager(store, session_id="old-session")
    old_record = old_manager.begin(
        tab_key="network",
        title="Old Network Output",
        tool_name="find_network",
    )

    new_manager = RunManager(store, session_id="new-session")
    removed = new_manager.cleanup_stale_cache()

    assert removed == [old_record["run_id"]]
    assert store.read(old_record["run_id"]) is None


def test_output_panel_is_safe_after_client_delete(tmp_path, monkeypatch):
    from nicegui import Client
    from nicegui.page import page
    import ui.components.output_panel as output_panel_module
    from ui.components.output_panel import OutputPanel
    from ui.run_state import RunManager, RunStateStore

    manager = RunManager(RunStateStore(tmp_path))
    monkeypatch.setattr(output_panel_module, "RUN_MANAGER", manager)

    client = Client(page("/run-state-delete"))
    with client:
        panel = OutputPanel("Test", state_key="teardown")
        panel.create()
        record = manager.begin(
            tab_key="teardown",
            title="Test",
            tool_name="find_network",
        )

    # Client.delete invokes the panel lifecycle hook before removing the
    # elements. Subsequent events should remain a store operation only.
    client.delete()
    manager.append_log(record["run_id"], "late output")
    manager.finish(
        record["run_id"],
        {"returncode": 0, "files": [], "duration": 0},
    )
    assert manager.store.latest("teardown")["status"] == "Completed"


def test_output_panel_run_persists_and_rehydrates_after_refresh(tmp_path, monkeypatch):
    import asyncio

    from nicegui import Client
    from nicegui.page import page
    import ui.components.output_panel as output_panel_module
    from ui.components.output_panel import OutputPanel
    from ui.run_state import RunManager, RunStateStore

    manager = RunManager(RunStateStore(tmp_path))
    monkeypatch.setattr(output_panel_module, "RUN_MANAGER", manager)

    class FakeRunner:
        is_running = False

        async def run(
            self,
            _tool_name,
            _constructor_params,
            _method_name,
            method_params=None,
            log_callback=None,
            progress_callback=None,
            output_dir=None,
        ):
            del method_params, output_dir
            self.is_running = True
            log_callback("restored log line", "stdout")
            progress_callback("execute", "Running fake analysis")
            self.is_running = False
            return {
                "returncode": 0,
                "files": [],
                "duration": 0.1,
                "cancelled": False,
                "output_folder": None,
            }

        def cancel(self):
            self.is_running = False

    first_client = Client(page("/run-state-refresh-first"))
    with first_client:
        first_panel = OutputPanel("Test", state_key="refresh")
        first_panel.create()
        result = asyncio.run(
            first_panel.run(FakeRunner(), "find_network", {}, "find_network")
        )
    assert result["returncode"] == 0
    assert manager.store.latest("refresh")["status"] == "Completed"

    first_client.delete()
    second_client = Client(page("/run-state-refresh-second"))
    with second_client:
        second_panel = OutputPanel("Test", state_key="refresh")
        second_panel.create()

    assert second_panel.status_label.text == "Completed"
    assert any(
        child.text == "restored log line"
        for child in second_panel.log_area.default_slot.children
    )
    second_client.delete()


def test_new_ui_session_does_not_rehydrate_old_terminal_run(tmp_path):
    from ui.run_state import RunManager, RunStateStore

    store = RunStateStore(tmp_path)
    first_manager = RunManager(store, session_id="ui-session-a")
    record = first_manager.begin(
        tab_key="network",
        title="Network Output",
        tool_name="find_network",
    )
    first_manager.append_log(record["run_id"], "old session output")
    first_manager.finish(
        record["run_id"],
        {"returncode": 0, "files": [], "duration": 0.4},
    )

    second_manager = RunManager(store, session_id="ui-session-b")
    events = []
    second_manager.subscribe("network", events.append)

    assert events == []
    assert second_manager.recent_runs() == []
    persisted = store.latest("network")
    assert persisted["status"] == "Completed"
    assert persisted["session_id"] == "ui-session-a"


def test_active_run_updates_the_new_panel_after_refresh(tmp_path, monkeypatch):
    import asyncio

    from nicegui import Client
    from nicegui.page import page
    import ui.components.output_panel as output_panel_module
    from ui.components.output_panel import OutputPanel
    from ui.run_state import RunManager, RunStateStore

    manager = RunManager(RunStateStore(tmp_path))
    monkeypatch.setattr(output_panel_module, "RUN_MANAGER", manager)

    async def scenario():
        completed = asyncio.Event()

        class WaitingRunner:
            is_running = False

            async def run(
                self,
                _tool_name,
                _constructor_params,
                _method_name,
                method_params=None,
                log_callback=None,
                progress_callback=None,
                output_dir=None,
            ):
                del method_params, output_dir
                self.is_running = True
                log_callback("active before refresh", "stdout")
                progress_callback("execute", "Waiting for completion")
                await completed.wait()
                self.is_running = False
                return {
                    "returncode": 0,
                    "files": [],
                    "duration": 0.2,
                    "cancelled": False,
                    "output_folder": None,
                }

            def cancel(self):
                completed.set()

        first_client = Client(page("/run-state-active-first"))
        with first_client:
            first_panel = OutputPanel("Test", state_key="active-refresh")
            first_panel.create()
        run_task = asyncio.create_task(
            first_panel.run(WaitingRunner(), "find_network", {}, "find_network")
        )

        for _ in range(20):
            if manager._active_runners:
                break
            await asyncio.sleep(0)
        first_client.delete()

        second_client = Client(page("/run-state-active-second"))
        with second_client:
            second_panel = OutputPanel("Test", state_key="active-refresh")
            second_panel.create()
        assert second_panel.status_label.text == "Running"
        assert any(
            child.text == "active before refresh"
            for child in second_panel.log_area.default_slot.children
        )

        completed.set()
        await run_task
        assert second_panel.status_label.text == "Completed"
        second_client.delete()

    asyncio.run(scenario())
