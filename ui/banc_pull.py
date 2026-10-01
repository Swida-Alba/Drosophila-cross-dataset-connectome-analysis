"""Background worker for BANC dataset downloads (public release bucket).

Mirrors :class:`CodexPuller` with BANC's three levels: necessary data
(``banc_public_data.prepare_dataset_tables`` — meta feather + connections,
~134 MB once), the per-synapse table (``ensure_synapse_table``, ~3.9 GB,
resumable), and a bulk skeleton pull (``morphology.download_all_skeletons``
— per-neuron SWCs from the same public bucket into the raw cache). No
token is involved anywhere. One pull at a time.
"""

import threading
import time
from typing import Dict, List, Optional


class BancPuller:
    """One-shot background BANC download manager (one pull at a time)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state: Dict = {
            "running": False,
            "dataset": None,
            "levels": [],
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
        with self._lock:
            return dict(self._state)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._state["running"]

    def start(self, dataset: str, levels: List[str],
              project_root=None) -> bool:
        """Start a background pull of *levels* for *dataset*.

        Returns False when a pull is already running.
        """
        with self._lock:
            if self._state["running"]:
                return False
            self._state = {
                "running": True,
                "dataset": dataset,
                "levels": list(levels),
                "phase": "prepare",
                "current": 0,
                "total": 0,
                "info": "Preparing BANC download...",
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
            args=(dataset, list(levels), project_root),
            daemon=True,
            name=f"banc-pull-{dataset}",
        )
        self._thread.start()
        return True

    def cancel(self) -> None:
        self._cancel_event.set()
        with self._lock:
            self._state["cancel_requested"] = True

    def _progress(self, current: int, total: int, info: str = "") -> None:
        with self._lock:
            self._state["current"] = current
            self._state["total"] = total
            if info:
                self._state["info"] = info
            if self._state["fetch_started_at"] is None and total:
                self._state["fetch_started_at"] = time.time()

    def _phase(self, name: str, msg: str) -> None:
        with self._lock:
            self._state["phase"] = name
            self._state["info"] = msg
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

    def _run(self, dataset: str, levels: List[str], project_root) -> None:
        from pathlib import Path

        from utils.flywire_readiness import dataset_folder
        try:
            root = Path(project_root) if project_root is not None \
                else Path(__file__).resolve().parents[1]
            dataset_dir = root / "datasets" / dataset_folder(dataset)
            completed: List[str] = []

            if LEVEL_NECESSARY in levels:
                self._phase("download",
                            "Preparing BANC metadata + connections "
                            "(public bucket)...")
                from banc_public_data import prepare_dataset_tables
                if not prepare_dataset_tables(dataset, dataset_dir):
                    raise RuntimeError(
                        "BANC table preparation did not complete "
                        f"for {dataset}")
                completed.append(LEVEL_NECESSARY)

            if LEVEL_SYNAPSE in levels and not self._cancel_event.is_set():
                self._phase("download",
                            "Downloading the BANC per-synapse table "
                            "(~3.9 GB, resumable)...")
                from banc_public_data import ensure_synapse_table
                path = ensure_synapse_table(
                    dataset, project_root=root,
                    progress_callback=lambda pos, total: self._progress(
                        pos, total, "Downloading the per-synapse table..."))
                if path is None:
                    raise RuntimeError(
                        "The BANC release has no synapse table for this "
                        "version")
                completed.append(LEVEL_SYNAPSE)

            if LEVEL_SKELETONS in levels and not self._cancel_event.is_set():
                self._phase("skeletons",
                            "Fetching all BANC skeletons (per-neuron SWCs "
                            "from the public bucket)...")
                from morphology import download_all_skeletons
                download_all_skeletons(
                    dataset, project_root=root, verbose=False,
                    progress_callback=self._progress,
                    cancel_event=self._cancel_event)
                completed.append(LEVEL_SKELETONS)

            cancelled = self._cancel_event.is_set()
            self._finish(cancelled=cancelled, summary={
                "dataset": dataset,
                "completed": completed,
                "elapsed_time": time.time() - (
                    self._state.get("started_at") or time.time()),
            })
        except Exception as exc:
            if self._cancel_event.is_set():
                self._finish(cancelled=True, summary={"dataset": dataset})
            else:
                self._finish(error=f"{type(exc).__name__}: {exc}")


LEVEL_NECESSARY = "necessary"
LEVEL_SYNAPSE = "synapses"
LEVEL_SKELETONS = "skeletons"
LEVEL_ORDER = (LEVEL_NECESSARY, LEVEL_SYNAPSE, LEVEL_SKELETONS)

BANC_SYNAPSE_TABLE_BYTES = 4_200_000_000  # ~3.9 GiB


def banc_level_status(dataset: str, project_root=None) -> Dict[str, dict]:
    """Local-presence status per level for the BANC card."""
    from pathlib import Path

    from banc_public_data import synapse_table_path
    from utils.flywire_readiness import dataset_folder

    root = Path(project_root) if project_root is not None \
        else Path(__file__).resolve().parents[1]
    folder = dataset_folder(dataset)
    dataset_dir = root / "datasets" / folder
    connections = dataset_dir / f"{folder}_merged_connections.parquet"
    neurons = dataset_dir / f"{folder}_allneurons_neuron_df.parquet"
    synapse = synapse_table_path(dataset, project_root=root)
    return {
        LEVEL_NECESSARY: {
            "complete": connections.exists() and neurons.exists(),
            "detail": ("✓ metadata + connections tables local"
                       if connections.exists() and neurons.exists()
                       else "· tables download automatically on first use"),
        },
        LEVEL_SYNAPSE: {
            "complete": synapse.exists(),
            "detail": ("✓ per-synapse table local (~3.9 GB)"
                       if synapse.exists()
                       else "· per-synapse table not downloaded"),
        },
        LEVEL_SKELETONS: {
            "complete": False,
            "detail": ("· skeletons fetch on demand during visualization; "
                       "this level bulk-caches every SWC"),
        },
    }
