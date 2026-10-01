"""Background worker for FlyWire Codex FAFB downloads.

Runs in a worker thread so the UI stays responsive; the Settings tab polls
:attr:`CodexPuller.state` with a ``ui.timer`` and renders progress. One pull
at a time; a pull walks the selected download levels in order (necessary →
synapse → skeleton bundle) and optionally runs the FAFB converter once the
necessary level is complete. Interrupted transfers keep their ``.part``
files on the server-side download contract (resumable ranged GETs), so a
re-run simply continues where the cancel stopped.
"""

import threading
import time
from typing import Dict, List, Optional


class CodexPuller:
    """One-shot background Codex download manager (one pull at a time)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state: Dict = {
            "running": False,
            "levels": [],
            "product": None,
            "phase": "idle",
            "current": 0,
            "total": 0,
            "info": "",
            "done": False,
            "cancelled": False,
            "cancel_requested": False,
            "error": None,
            "summary": None,
            "started_at": None,
            "fetch_started_at": None,
        }

    @property
    def state(self) -> Dict:
        """Snapshot of the current pull state (thread-safe)."""
        with self._lock:
            return dict(self._state)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._state["running"]

    def start(self, levels: List[str], run_converter: bool = False,
              project_root=None,
              dataset: str = "flywire_FAFB_v783") -> bool:
        """Start a background download of *levels* (subset of LEVEL_ORDER).

        An empty *levels* list runs only the converter step. Returns False
        when a pull is already running.
        """
        with self._lock:
            if self._state["running"]:
                return False
            self._state = {
                "running": True,
                "levels": list(levels),
                "product": None,
                "phase": "prepare",
                "current": 0,
                "total": 0,
                "info": "Preparing download...",
                "done": False,
                "cancelled": False,
                "cancel_requested": False,
                "error": None,
                "summary": None,
                "started_at": time.time(),
                "fetch_started_at": None,
            }
            self._cancel_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(list(levels), run_converter, project_root, dataset),
            daemon=True,
            name="codex-pull",
        )
        self._thread.start()
        return True

    def cancel(self) -> None:
        """Request a stop after the current chunk (resume-safe)."""
        self._cancel_event.set()
        with self._lock:
            self._state["cancel_requested"] = True

    def _progress(self, current: int, total: int, info: str) -> None:
        with self._lock:
            self._state["current"] = current
            self._state["total"] = total
            self._state["info"] = info
            if self._state["fetch_started_at"] is None and total > 0:
                self._state["fetch_started_at"] = time.time()

    def _phase(self, name: str, msg: str, product=None) -> None:
        with self._lock:
            self._state["phase"] = name
            self._state["info"] = msg
            self._state["product"] = product
            if name == "download":
                self._state["current"] = 0
                self._state["total"] = 0
                self._state["fetch_started_at"] = None

    def _finish(self, cancelled: bool = False, error: str = None,
                summary: Dict = None) -> None:
        with self._lock:
            if error is not None:
                self._state["error"] = error
                self._state["info"] = "Failed."
            else:
                self._state["cancelled"] = cancelled
                self._state["info"] = "Finished."
            if summary is not None:
                self._state["summary"] = summary
            self._state["done"] = True
            self._state["running"] = False

    def _run(self, levels: List[str], run_converter: bool, project_root,
             dataset: str) -> None:
        import codex_downloader as cd

        downloaded: List[str] = []
        skipped: List[str] = []
        cancelled = False
        try:
            if levels:
                downloads = cd.downloads_dir_for(dataset, project_root)
                products = [p for level in cd.LEVEL_ORDER if level in levels
                            for p in cd.level_products(level)]
                for product in products:
                    if self._cancel_event.is_set():
                        cancelled = True
                        break
                    status = cd.product_status(product.key, downloads)
                    if status["complete"]:
                        skipped.append(product.key)
                        continue
                    self._phase(
                        "download",
                        f"Downloading {product.filename} "
                        f"({human_bytes(product.size)})...",
                        product=product.key)
                    cd.download_product(
                        product.key, downloads, project_root=project_root,
                        progress_callback=lambda pos, total,
                        _name=product.filename: self._progress(
                            pos, total, f"Downloading {_name}..."),
                        cancel_event=self._cancel_event)
                    downloaded.append(product.key)

            converted = False
            convert_note = "not run"
            if run_converter and not cancelled:
                self._phase("convert", "Running the FAFB converter...")
                downloads = cd.downloads_dir_for(dataset, project_root)
                necessary_complete = all(
                    cd.product_status(p.key, downloads)["complete"]
                    for p in cd.level_products(cd.LEVEL_NECESSARY))
                if not necessary_complete:
                    convert_note = (
                        "skipped: the necessary data level is incomplete")
                else:
                    from FAFB_file_converter import ensure_flywire_data
                    converted = bool(ensure_flywire_data(
                        dataset, str(downloads.parent)))
                    convert_note = (
                        "tables ready" if converted
                        else "ran, but files are still missing")
            summary = {
                "downloaded": downloaded,
                "skipped": skipped,
                "converted": converted,
                "convert_note": convert_note,
                "elapsed_time": time.time() - (self._state.get("started_at")
                                               or time.time()),
            }
            self._finish(cancelled=cancelled, summary=summary)
        except cd.CodexDownloadCancelled as exc:
            self._finish(cancelled=True, summary={
                "downloaded": downloaded,
                "skipped": skipped,
                "cancelled_during": str(exc),
            })
        except Exception as exc:
            self._finish(error=f"{type(exc).__name__}: {exc}")


def human_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


# Shared single-flight instance: the Settings download card and the
# visualization tab's synapse offer must not start concurrent pulls.
codex_puller = CodexPuller()
