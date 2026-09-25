"""Durable execution state for the DROCAT UI.

NiceGUI pages are disposable: a browser refresh creates a new client and
deletes the elements owned by the previous client.  Analysis processes and
their user-visible state therefore cannot be owned exclusively by a page or
an output panel.  This module provides a small application-scoped run store
and notification hub which keeps the execution contract independent from
NiceGUI.

The store is intentionally local and file based.  DROCAT runs locally, so a
JSON manifest plus an append-only JSONL log gives refresh recovery without
introducing a database dependency.  Writes are atomic for manifests and
failures to persist state never interrupt the analysis process.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from .config import PROJECT_ROOT


ACTIVE_STATUSES = {"Queued", "Running"}
TERMINAL_STATUSES = {"Completed", "Failed", "Cancelled", "Interrupted"}

_SAFE_KEY_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_STRUCTURED_PROGRESS_RE = re.compile(
    r"^\[DROCAT\]\[progress\]\s*(?P<step>\d+)\s*/\s*"
    r"(?P<total>\d+)\s*(?P<label>.*)$"
)
_COUNTER_PROGRESS_RE = re.compile(
    r"^\s*(?P<label>.*?):\s*"
    r"(?:(?:\d+(?:\.\d+)?)%\|.*?\|\s*)?"
    r"(?P<current>[\d,]+)\s*/\s*(?P<total>[\d,]+)"
)
_SENSITIVE_KEY_RE = re.compile(
    r"(?:token|password|passwd|secret|credential|api[_-]?key|authorization)",
    re.IGNORECASE,
)
_SENSITIVE_TEXT_RE = re.compile(
    r"(?i)(\b(?:token|password|passwd|secret|credential|api[_-]?key|authorization)"
    r"\s*[:=]\s*)([^,\s;]+)"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_key(value: str, fallback: str = "run") -> str:
    cleaned = _SAFE_KEY_RE.sub("_", str(value or "").strip()).strip("._")
    return (cleaned or fallback)[:120]


def _json_safe(value: Any) -> Any:
    """Convert common UI/backend values into JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_input_summary(value: Any, key: str = "") -> Any:
    """Keep useful run context while excluding credentials from manifests."""
    if _SENSITIVE_KEY_RE.search(str(key)):
        return "[redacted]"
    if isinstance(value, dict):
        return {
            str(item_key): _safe_input_summary(item, str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe_input_summary(item, key) for item in value]
    return _json_safe(value)


def _redact_text(value: Any) -> str:
    """Redact key/value credentials before writing a line to disk."""
    text = str(value or "")
    return _SENSITIVE_TEXT_RE.sub(r"\1[redacted]", text)


class RunStateStore:
    """Persist run manifests and append-only execution logs on disk."""

    def __init__(self, root: Optional[os.PathLike | str] = None) -> None:
        configured = os.environ.get("DROCAT_RUN_STATE_DIR", "").strip()
        self.root = Path(root or configured or (PROJECT_ROOT / "local_data" / ".drocat_runs"))
        self.manifest_dir = self.root / "runs"
        self.index_path = self.root / "index.json"
        self._lock = threading.RLock()

    def _manifest_path(self, run_id: str) -> Path:
        return self.manifest_dir / f"{_safe_key(run_id)}.json"

    def _log_path(self, run_id: str) -> Path:
        return self.manifest_dir / f"{_safe_key(run_id)}.jsonl"

    def _write_json_atomic(self, path: Path, payload: Any) -> bool:
        temporary: Optional[Path] = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(path)
            return True
        except (OSError, TypeError, ValueError):
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            return False

    def _read_json(self, path: Path) -> Optional[dict]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, ValueError, TypeError):
            return None

    def _read_index(self) -> dict:
        return self._read_json(self.index_path) or {}

    def create(
        self,
        *,
        tab_key: str,
        title: str,
        tool_name: str,
        method_name: str = "run",
        constructor_params: Optional[dict] = None,
        method_params: Optional[dict] = None,
        output_dir: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """Create and persist a new run record, returning its snapshot."""
        run_id = uuid.uuid4().hex
        now = _utc_now()
        record = {
            "schema_version": 1,
            "run_id": run_id,
            # A run is restorable in the UI session that created it. The
            # durable manifest remains on disk for diagnostics/history, but a
            # later server process must not silently repopulate its tabs with
            # an old session's terminal log.
            "session_id": _safe_key(session_id, "legacy") if session_id else None,
            "tab_key": _safe_key(tab_key, "tool"),
            "title": str(title or "Output"),
            "tool_name": str(tool_name or ""),
            "method_name": str(method_name or "run"),
            "status": "Running",
            "message": "Starting analysis",
            "progress": {
                "phase": "prepare",
                "label": "Preparing inputs",
                "value": 0.0,
                "active_index": 0,
            },
            "created_at": now,
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "duration": None,
            "return_code": None,
            "cancel_requested": False,
            "output_dir": str(output_dir) if output_dir else None,
            "output_folder": None,
            "files": [],
            "log_file": str(self._log_path(run_id)),
            "input_summary": _safe_input_summary({
                "constructor": constructor_params or {},
                "method": method_params or {},
            }),
        }
        with self._lock:
            self._write_json_atomic(self._manifest_path(run_id), record)
            index = self._read_index()
            latest = index.get("latest", {})
            if not isinstance(latest, dict):
                latest = {}
            latest[record["tab_key"]] = run_id
            index["latest"] = latest
            self._write_json_atomic(self.index_path, index)
        return deepcopy(record)

    def read(self, run_id: str) -> Optional[dict]:
        with self._lock:
            value = self._read_json(self._manifest_path(run_id))
        return deepcopy(value) if value else None

    def latest(self, tab_key: str, session_id: Optional[str] = None) -> Optional[dict]:
        key = _safe_key(tab_key, "tool")
        expected_session = _safe_key(session_id, "session") if session_id else None
        with self._lock:
            index = self._read_index()
            latest = index.get("latest", {})
            run_id = latest.get(key) if isinstance(latest, dict) else None
            record = self._read_json(self._manifest_path(run_id)) if run_id else None
            if expected_session and (
                record is None or record.get("session_id") != expected_session
            ):
                record = None
            if record is None:
                candidates = []
                for path in self._manifests_snapshot():
                    candidate = self._read_json(path)
                    if (
                        candidate
                        and candidate.get("tab_key") == key
                        and (
                            expected_session is None
                            or candidate.get("session_id") == expected_session
                        )
                    ):
                        candidates.append(candidate)
                record = max(
                    candidates,
                    key=lambda item: str(item.get("created_at", "")),
                    default=None,
                )
        return deepcopy(record) if record else None

    def _manifests_snapshot(self) -> list:
        """Directory listing for the scan fallback, memoized per mtime.

        The header activity poll lands here every 2 s per tab when the index
        has no current-session run (worst right after a server restart); the
        glob is the expensive part and the directory rarely changes between
        ticks, so cache the listing keyed on its mtime.
        """
        try:
            mtime = self.manifest_dir.stat().st_mtime_ns
        except OSError:
            return []
        cache = self.__dict__.setdefault("_manifests_snapshot_cache", (None, []))
        if cache[0] != mtime:
            cache = (mtime, sorted(self.manifest_dir.glob("*.json")))
            self._manifests_snapshot_cache = cache
        return cache[1]

    def latest_for_tabs(
        self,
        tab_keys: Iterable[str],
        session_id: Optional[str] = None,
    ) -> Dict[str, dict]:
        return {
            _safe_key(key, "tool"): record
            for key in tab_keys
            if (record := self.latest(key, session_id=session_id)) is not None
        }

    def recent(self, limit: int = 25, session_id: Optional[str] = None) -> list[dict]:
        """Return the most recently updated run records for the activity UI."""
        records = []
        expected_session = _safe_key(session_id, "session") if session_id else None
        with self._lock:
            try:
                paths = list(self.manifest_dir.glob("*.json"))
            except OSError:
                paths = []
            for path in paths:
                record = self._read_json(path)
                if record and (
                    expected_session is None
                    or record.get("session_id") == expected_session
                ):
                    records.append(record)
        records.sort(
            key=lambda item: str(
                item.get("updated_at") or item.get("created_at") or ""
            ),
            reverse=True,
        )
        return deepcopy(records[:max(1, int(limit))])

    def update(self, run_id: str, **changes: Any) -> Optional[dict]:
        with self._lock:
            record = self._read_json(self._manifest_path(run_id))
            if record is None:
                return None
            for key, value in changes.items():
                record[key] = _json_safe(value)
            record["updated_at"] = _utc_now()
            self._write_json_atomic(self._manifest_path(run_id), record)
        return deepcopy(record)

    def append_log(self, run_id: str, message: str, level: str = "stdout") -> None:
        line = str(message or "").rstrip()
        if not line:
            return
        event = {
            "timestamp": _utc_now(),
            "level": str(level or "stdout"),
            "message": _redact_text(line),
        }
        with self._lock:
            try:
                self.manifest_dir.mkdir(parents=True, exist_ok=True)
                with self._log_path(run_id).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            except (OSError, TypeError, ValueError):
                pass

    def read_logs(self, run_id: str, limit: int = 5000) -> list[tuple[str, str]]:
        events = []
        with self._lock:
            try:
                with self._log_path(run_id).open("r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        try:
                            event = json.loads(raw_line)
                        except (ValueError, TypeError):
                            continue
                        if not isinstance(event, dict):
                            continue
                        events.append((
                            str(event.get("level", "stdout")),
                            str(event.get("message", "")),
                        ))
            except OSError:
                return []
        return events[-max(1, int(limit)):]

    def clear(self, run_id: str) -> None:
        """Remove one run's local state; intended for explicit UI cleanup."""
        with self._lock:
            for path in (self._manifest_path(run_id), self._log_path(run_id)):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def cleanup_terminal_runs(self, session_id: str) -> list[str]:
        """Delete terminal run-cache records from older UI sessions.

        Run state is only needed for refresh recovery while its server session
        exists. Terminal records from earlier sessions are therefore safe to
        remove at the next UI startup. Active records are deliberately kept:
        the new process can first determine whether an old backend is still
        alive before reconciling it as interrupted. This method never touches
        output folders or analysis result files.
        """
        expected_session = _safe_key(session_id, "session")
        removed: list[str] = []
        with self._lock:
            try:
                manifests = list(self.manifest_dir.glob("*.json"))
            except OSError:
                manifests = []

            for path in manifests:
                record = self._read_json(path)
                if not record:
                    continue
                if record.get("session_id") == expected_session:
                    continue
                if record.get("status") not in TERMINAL_STATUSES:
                    continue
                run_id = str(record.get("run_id") or path.stem)
                for cache_path in (path, self._log_path(run_id)):
                    try:
                        cache_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                removed.append(run_id)

            # A failed atomic write can leave a JSONL file without its
            # manifest. It is not recoverable and is safe to remove here.
            try:
                for log_path in self.manifest_dir.glob("*.jsonl"):
                    if not log_path.with_suffix(".json").exists():
                        try:
                            log_path.unlink(missing_ok=True)
                        except OSError:
                            pass
            except OSError:
                pass

            # Rebuild the tiny latest-run index after deleting cache entries;
            # stale pointers must not keep old run IDs discoverable.
            try:
                remaining_latest: dict[str, tuple[str, str]] = {}
                for manifest_path in self.manifest_dir.glob("*.json"):
                    record = self._read_json(manifest_path)
                    if not record:
                        continue
                    tab_key = _safe_key(record.get("tab_key"), "tool")
                    run_id = str(record.get("run_id") or manifest_path.stem)
                    updated_at = str(
                        record.get("updated_at") or record.get("created_at") or ""
                    )
                    current = remaining_latest.get(tab_key)
                    if current is None or updated_at >= current[1]:
                        remaining_latest[tab_key] = (run_id, updated_at)
                index = self._read_index()
                index["latest"] = {
                    tab_key: run_id
                    for tab_key, (run_id, _updated_at) in remaining_latest.items()
                }
                self._write_json_atomic(self.index_path, index)
            except OSError:
                pass
        return removed


class RunManager:
    """Coordinate persistent run records and live UI subscribers."""

    def __init__(
        self,
        store: Optional[RunStateStore] = None,
        *,
        session_id: Optional[str] = None,
    ) -> None:
        self.store = store or RunStateStore()
        # A new RunManager represents a new DROCAT server session. Browser
        # refreshes keep this process (and therefore this ID); restarting the
        # terminal/server creates a new ID and starts with an empty restorable
        # view unless a future detached worker explicitly supports reconnect.
        self.session_id = _safe_key(session_id or uuid.uuid4().hex, "session")
        self._active_runners: Dict[str, Any] = {}
        self._records: Dict[str, dict] = {}
        self._listeners: Dict[str, Dict[int, Callable[[dict], None]]] = {}
        self._tab_listeners: Dict[str, Dict[int, Callable[[dict], None]]] = {}
        self._shutdown_started = False
        self._lock = threading.RLock()

    @staticmethod
    def _is_active(record: Optional[dict]) -> bool:
        return bool(record and record.get("status") in ACTIVE_STATUSES)

    def _emit(self, run_id: str, event: dict) -> None:
        with self._lock:
            listeners = list(self._listeners.get(run_id, {}).values())
        for listener in listeners:
            try:
                listener(deepcopy(event))
            except Exception:
                # A NiceGUI client may disappear between taking the snapshot
                # and delivering an event. The store must remain authoritative.
                continue

    def begin(
        self,
        *,
        tab_key: str,
        title: str,
        tool_name: str,
        method_name: str = "run",
        constructor_params: Optional[dict] = None,
        method_params: Optional[dict] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        record = self.store.create(
            tab_key=tab_key,
            title=title,
            tool_name=tool_name,
            method_name=method_name,
            constructor_params=constructor_params,
            method_params=method_params,
            output_dir=output_dir,
            session_id=self.session_id,
        )
        with self._lock:
            self._active_runners[record["run_id"]] = None
            self._records[record["run_id"]] = deepcopy(record)
            tab_listeners = list(
                self._tab_listeners.get(record["tab_key"], {}).items()
            )
            if tab_listeners:
                self._listeners.setdefault(record["run_id"], {}).update(
                    dict(tab_listeners)
                )
        self._emit(record["run_id"], {"kind": "state", "record": record})
        return record

    def register_runner(self, run_id: str, runner: Any) -> None:
        terminate_immediately = False
        with self._lock:
            if run_id in self._active_runners:
                self._active_runners[run_id] = runner
                terminate_immediately = self._shutdown_started
        if terminate_immediately:
            try:
                runner.cancel()
            except Exception:
                pass

    def subscribe(self, tab_key: str, listener: Callable[[dict], None]) -> Callable[[], None]:
        """Subscribe to the latest run for a tab and replay its state/log."""
        self.reconcile()
        record = self.store.latest(tab_key, session_id=self.session_id)
        run_id = record.get("run_id") if record else None
        tab_key = _safe_key(tab_key, "tool")
        listener_id = id(listener)
        with self._lock:
            self._tab_listeners.setdefault(tab_key, {})[listener_id] = listener
            if run_id:
                self._listeners.setdefault(run_id, {})[listener_id] = listener
        if run_id:
            try:
                listener({
                    "kind": "snapshot",
                    "record": deepcopy(record),
                    "logs": self.store.read_logs(run_id),
                })
            except Exception:
                pass

        def unsubscribe() -> None:
            with self._lock:
                tab_listeners = self._tab_listeners.get(tab_key, {})
                tab_listeners.pop(listener_id, None)
                if not tab_listeners:
                    self._tab_listeners.pop(tab_key, None)
                for current_run_id, listeners in list(self._listeners.items()):
                    listeners.pop(listener_id, None)
                    if not listeners:
                        self._listeners.pop(current_run_id, None)

        return unsubscribe

    def update_progress(self, run_id: str, phase: str, label: str = "") -> None:
        with self._lock:
            record = deepcopy(self._records.get(run_id))
        record = record or self.store.read(run_id)
        if record is None:
            return
        progress = dict(record.get("progress") or {})
        phase = str(phase or "").strip().lower()
        progress.update({"phase": phase, "label": str(label or "")})
        if phase in {"prepare", "initialize"}:
            progress["value"] = 0.0
        elif phase == "complete":
            progress["value"] = 1.0
        updated = self.store.update(run_id, progress=progress, message=label or phase)
        if updated:
            with self._lock:
                self._records[run_id] = deepcopy(updated)
            self._emit(run_id, {"kind": "progress", "record": updated})

    def append_log(self, run_id: str, message: str, level: str = "stdout") -> None:
        line = str(message or "").rstrip()
        if not line:
            return
        self.store.append_log(run_id, line, level)
        with self._lock:
            record = deepcopy(self._records.get(run_id))
        record = record or self.store.read(run_id)
        if record is None:
            return
        progress = dict(record.get("progress") or {})
        updated = record
        progress_changed = False
        structured = _STRUCTURED_PROGRESS_RE.match(line)
        if structured:
            step = max(1, int(structured.group("step")))
            total = max(1, int(structured.group("total")))
            progress.update({
                "phase": "execute",
                "step": step,
                "total": total,
                "label": structured.group("label").strip(),
                "value": max(float(progress.get("value", 0.0) or 0.0), (step - 1) / total),
                "active_index": step - 1,
            })
            progress_changed = True
        else:
            counter = _COUNTER_PROGRESS_RE.match(line) if level == "progress" else None
            if counter:
                current = int(counter.group("current").replace(",", ""))
                total = int(counter.group("total").replace(",", ""))
                if total > 0:
                    progress.update({
                        "phase": "execute",
                        "counter_current": current,
                        "counter_total": total,
                        "label": counter.group("label").strip(),
                    })
                    progress_changed = True
        if progress_changed:
            updated = self.store.update(run_id, progress=progress) or record
            with self._lock:
                self._records[run_id] = deepcopy(updated)
        self._emit(run_id, {
            "kind": "log",
            "run_id": run_id,
            "level": str(level or "stdout"),
            "message": line,
            "record": updated,
        })

    def finish(self, run_id: str, result: Optional[dict] = None, error: Optional[str] = None) -> Optional[dict]:
        """Persist a terminal result and release the in-process runner."""
        result = result or {}
        cancelled = bool(result.get("cancelled"))
        return_code = result.get("returncode")
        if error:
            status = "Failed"
            message = error
        elif cancelled:
            status = "Cancelled"
            message = "Execution cancelled by user"
        elif return_code == 0:
            status = "Completed"
            message = "Execution completed successfully"
        else:
            status = "Failed"
            message = f"Execution failed (return code {return_code})"

        with self._lock:
            record = deepcopy(self._records.get(run_id))
        record = record or self.store.read(run_id)
        if record is None:
            with self._lock:
                self._active_runners.pop(run_id, None)
                self._records.pop(run_id, None)
            return None
        # The shutdown hook has already recorded the authoritative terminal
        # state. The runner task may still unwind briefly after SIGTERM; do
        # not let its late result overwrite the shutdown outcome.
        if (
            record.get("status") == "Interrupted"
            and record.get("shutdown_requested")
        ):
            with self._lock:
                self._active_runners.pop(run_id, None)
                self._records.pop(run_id, None)
            return record
        progress = dict(record.get("progress") or {})
        if status == "Completed":
            progress.update({"phase": "complete", "label": "Completed successfully!", "value": 1.0})
        elif status in {"Failed", "Cancelled"}:
            progress["phase"] = "failed" if status == "Failed" else "cancelled"
        duration = result.get("duration")
        if duration is None:
            duration = record.get("duration")
        updated = self.store.update(
            run_id,
            status=status,
            message=message,
            progress=progress,
            finished_at=_utc_now(),
            duration=float(duration) if isinstance(duration, (int, float)) else duration,
            return_code=return_code,
            output_folder=result.get("output_folder"),
            files=result.get("files") or [],
            neuron_match=result.get("neuron_match"),
            error=error,
        )
        with self._lock:
            self._active_runners.pop(run_id, None)
            if updated:
                self._records[run_id] = deepcopy(updated)
        if updated:
            self._emit(run_id, {"kind": "state", "record": updated})
        with self._lock:
            self._records.pop(run_id, None)
        return updated

    def request_cancel(self, run_id: Optional[str]) -> bool:
        """Request cancellation of an active in-process run."""
        if not run_id:
            return False
        with self._lock:
            runner = self._active_runners.get(run_id)
        if runner is None:
            return False
        try:
            runner.cancel()
        except Exception:
            return False
        updated = self.store.update(
            run_id,
            cancel_requested=True,
            message="Cancellation requested",
        )
        if updated:
            with self._lock:
                self._records[run_id] = deepcopy(updated)
            self._emit(run_id, {"kind": "state", "record": updated})
        return True

    def shutdown(self) -> None:
        """Terminate active backends when the DROCAT UI process exits.

        A browser refresh never calls this method: the NiceGUI server remains
        alive, so the runner and its child process continue normally. This is
        reserved for application shutdown and is intentionally idempotent so
        it can also be used as a final cleanup hook by launchers/tests.
        """
        with self._lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True
            active = list(self._active_runners.items())

        for run_id, runner in active:
            if runner is not None:
                try:
                    runner.cancel()
                except Exception:
                    pass
            updated = self.store.update(
                run_id,
                status="Interrupted",
                message="The DROCAT UI process shut down and terminated the backend.",
                finished_at=_utc_now(),
                shutdown_requested=True,
                error="Backend execution stopped because the UI process exited.",
            )
            if updated:
                with self._lock:
                    self._records[run_id] = deepcopy(updated)
                self._emit(run_id, {"kind": "state", "record": updated})

    def recent_runs(self, limit: int = 25) -> list[dict]:
        return self.store.recent(limit, session_id=self.session_id)

    def cleanup_stale_cache(self) -> list[str]:
        """Reconcile and remove stale run state during UI initialization.

        Active records owned by this manager remain active. Active records
        from an older server session have no runner object in this process,
        so reconciliation marks them ``Interrupted`` and the cache cleanup
        removes them in the same startup pass.
        """
        self.reconcile()
        return self.store.cleanup_terminal_runs(self.session_id)

    def reconcile(self) -> list[dict]:
        """Mark active records without an in-process runner as interrupted."""
        interrupted = []
        try:
            manifests = list(self.store.manifest_dir.glob("*.json"))
        except OSError:
            manifests = []
        for path in manifests:
            record = self.store._read_json(path)  # pylint: disable=protected-access
            if not record or not self._is_active(record):
                continue
            run_id = record.get("run_id")
            with self._lock:
                runner = self._active_runners.get(run_id)
                known = run_id in self._active_runners and (
                    runner is None or bool(getattr(runner, "is_running", False))
                )
                if not known:
                    self._active_runners.pop(run_id, None)
            if known:
                continue
            updated = self.store.update(
                run_id,
                status="Interrupted",
                message="The DROCAT process was not found after the UI restarted.",
                finished_at=_utc_now(),
                error="No in-process runner is available to continue this execution.",
            )
            if updated:
                interrupted.append(updated)
                with self._lock:
                    self._records[run_id] = deepcopy(updated)
                self._emit(run_id, {"kind": "state", "record": updated})
        return interrupted


RUN_MANAGER = RunManager()


__all__ = [
    "ACTIVE_STATUSES",
    "TERMINAL_STATUSES",
    "RunManager",
    "RunStateStore",
    "RUN_MANAGER",
]
