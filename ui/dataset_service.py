"""
Dataset Service for DROCAT UI
Fetches available datasets from NeuPrint server dynamically.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .config import PROJECT_ROOT

try:
    # Shared token resolution: the server probe that lets a rejected
    # candidate fall through to the next location in the chain.
    from src.utils.token_manager import token_manager as _shared_token_manager
except ImportError:  # src not on sys.path; fall back to first-found tokens
    _shared_token_manager = None

try:
    from src.utils.naming_utils import canonical_dataset_name
    from src.flywire_ids import (
        dataset_folder,
        is_banc_dataset,
        is_fafb_dataset,
        is_local_connectome_dataset,
    )
except ImportError:  # src not on sys.path; keep the UI importable standalone
    def canonical_dataset_name(value):
        return str(value or "").strip()

    def dataset_folder(value):
        return canonical_dataset_name(value).replace(":", "_").replace(".", "_")

    def is_banc_dataset(dataset):
        return "banc" in str(dataset or "").strip().lower()

    def is_fafb_dataset(dataset):
        # Mirror src.flywire_ids.is_fafb_dataset: the ``fafb`` token or the
        # bare ``flywire``/``fafb`` aliases, never a generic ``flywire_*``
        # prefix and never BANC.
        normalized = canonical_dataset_name(dataset).strip().lower()
        if is_banc_dataset(normalized):
            return False
        return normalized in {"flywire", "fafb"} or "fafb" in normalized

    def is_local_connectome_dataset(dataset):
        return is_fafb_dataset(dataset) or is_banc_dataset(dataset)


# ---------------------------------------------------------------------------
# Status vocabulary
#
# A dataset's readiness is four independent facts, not one bool: server
# reachability, metadata prepared, connectivity prepared, visualization
# source.  Server state is the only dimension that needs a token/network and
# is therefore the only one persisted; the rest are derived from disk.
# ---------------------------------------------------------------------------

SERVER_AVAILABLE = "available"
SERVER_HIDDEN = "hidden"          # listed by NeuPrint but not queryable
SERVER_NO_TOKEN = "no_token"      # no credential configured
SERVER_TIMEOUT = "timeout"
SERVER_UNREACHABLE = "unreachable"
SERVER_UNKNOWN = "unknown"        # never checked

CAP_READY = "ready"               # artifact present locally
CAP_ON_DEMAND = "on_demand"       # fetchable from server/bucket/CAVE
CAP_PARTIAL = "partial"           # some artifacts present
CAP_MISSING = "missing"           # nothing available

ACCESS_STREAMING = "streaming"            # NeuPrint: query on demand
ACCESS_DOWNLOAD_REQUIRED = "download_required"  # BANC (never streamable)


@dataclass
class DatasetInfo:
    """Information about a dataset.

    ``available`` remains the single-bool convenience for callers/badges.
    The dimension fields describe *why* it is or is not ready:

    - ``server_state``: one of the ``SERVER_*`` constants.
    - ``metadata_state`` / ``connectivity_state`` / ``visualization_state``:
      one of the ``CAP_*`` constants.
    - ``access_mode``: ``streaming`` / ``download_required``.
    """
    name: str
    source: str  # 'neuprint', 'flywire' (FAFB), or 'banc' (standalone BANC)
    available: bool = False
    neuron_count: int = 0
    typed_count: int = 0
    local_cache: bool = False
    local_prepared: bool = False  # Has local data files ready to use
    display_name: str = ""  # Human-readable name from Codex/server
    metadata: Dict = field(default_factory=dict)
    error: Optional[str] = None
    # Dimensioned status (derived unless noted).
    family: str = ""                 # 'neuprint' | 'fafb' | 'banc'
    access_mode: str = ""
    server_state: str = SERVER_UNKNOWN  # persisted
    server_checked_at: Optional[str] = None  # persisted
    metadata_state: str = CAP_MISSING
    connectivity_state: str = CAP_MISSING
    visualization_state: str = CAP_MISSING
    visualization_source: Optional[str] = None  # 'local' | 'cave' | 'bucket' | 'server'


# Dataset name normalization: folder name <-> dataset name
# e.g. 'hemibrain_v1_2_1' <-> 'hemibrain:v1.2.1'
def folder_to_dataset(folder_name: str) -> str:
    """Convert a folder name to a dataset identifier."""
    # Local releases use their canonical folder spelling.  This also folds
    # legacy ``flywire_BANC_*`` folders into the standalone BANC namespace.
    if is_local_connectome_dataset(folder_name):
        return dataset_folder(folder_name)
    # NeuPrint: first _ becomes :, remaining _ become .
    # e.g. hemibrain_v1_2_1 -> hemibrain:v1.2.1
    #      male-cns_v0_9 -> male-cns:v0.9
    parts = folder_name.split("_", 1)
    if len(parts) == 2:
        prefix, version = parts
        version = version.replace("_", ".")
        return f"{prefix}:{version}"
    return folder_name


def dataset_to_folder(dataset: str) -> str:
    """Convert a dataset identifier to a folder name."""
    # Canonicalize both FAFB and standalone BANC to the same safe folder
    # spelling used by the backend and cache builders.
    if is_local_connectome_dataset(dataset):
        return dataset_folder(dataset)
    # NeuPrint: : becomes _, . becomes _
    return dataset.replace(":", "_").replace(".", "_")


def is_flywire_dataset(dataset: str) -> bool:
    """Compatibility alias for the FAFB-only FlyWire predicate.

    BANC is a standalone public-release source.  Use
    :func:`is_local_connectome_dataset` when code needs either FAFB or BANC.
    """
    return is_fafb_dataset(dataset)


class DatasetService:
    """Service for fetching and managing dataset availability."""

    AVAILABILITY_CACHE_FORMAT = "drocat_dataset_availability/v2"
    # v2 stores only the server dimension; local readiness is derived from
    # disk on every read (v1 also persisted local rows, which then went
    # stale in the file).
    AVAILABILITY_CACHE_FILENAME = "dataset_availability.json"

    # Known NeuPrint dataset candidates (fallback if /api/dbmeta/datasets fails)
    NEUPRINT_CANDIDATES = [
        "male-cns:v1.0",
        "male-cns:v0.9",
        "hemibrain:v1.2.1",
        "hemibrain:v1.1",
        "optic-lobe:v1.1",
        "optic-lobe:v1.0.1",
        "manc:v1.2.3",
        "manc:v1.2.1",
        "manc:v1.0",
        "fib19:v1.0",
        "mushroombody",
    ]

    # FAFB release datasets (the FlyWire/Codex-backed local source).
    FLYWIRE_DATASETS = [
        "flywire_FAFB_v783",
    ]

    # BANC release datasets. These are fetched from the public BANC bucket;
    # they do not use FlyWire/Codex or a CAVE token.
    BANC_DATASETS = [
        "banc_v888",
        "banc_v626",
    ]

    # Codex display info (fetched from codex.flywire.ai rendered page).
    # BANC counts are the prepared-table row counts from the 2026-09-04
    # public-bucket snapshot: v888 as served (188,508) and v626 after the
    # root_626 dedup (185,165). Only used when no local table can supply a
    # count — kept in sync by tests/ui/test_dataset_service.py.
    CODEX_DATASETS = {
        "flywire_FAFB_v783": {"display": "FAFB v783 (CB)", "desc": "Female Adult Fly Brain", "neurons": 139255},
        "banc_v888": {"display": "BANC v888 (CNS)", "desc": "Brain and Nerve Cord — public BANC release", "neurons": 188508},
        "banc_v626": {"display": "BANC v626 (CNS)", "desc": "Brain and Nerve Cord — public BANC release (older)", "neurons": 185165},
    }

    NEUPRINT_SERVER = "https://neuprint.janelia.org"
    CODEX_URL = "https://codex.flywire.ai/"

    def __init__(self):
        self._token: Optional[str] = None
        self._cave_token: Optional[str] = None
        # Ordered neuprint candidates from the config files:
        # [('config.json', '...'), ('config_local.json', '...')].
        self._token_chain: List[Tuple[str, str]] = []
        self._cache: Dict[str, DatasetInfo] = {}
        self._lock = threading.Lock()
        self._datasets_dir = PROJECT_ROOT / "datasets"
        self._cache_dir = PROJECT_ROOT / "cache"
        self._index_dir = PROJECT_ROOT / "neuron_indexes"
        self._available_neuprint: Optional[List[str]] = None
        self._server_datasets: Dict[str, dict] = {}  # Full server response from /api/dbmeta/datasets
        self._last_fetch_time: float = 0
        self._availability_updated_at: Optional[str] = None
        self._availability_loaded = False
        # Persisted server dimension: canonical name -> {state, checked_at,
        # metadata}.  This is the ONLY thing cached in the file.
        self._server_rows: Dict[str, dict] = {}
        self._server_rows_mtime: Optional[float] = None
        self._banc_bucket_probe_cache: Optional[dict] = None

    @property
    def availability_cache_path(self) -> Path:
        """Persistent file containing the last complete availability refresh."""
        return self._cache_dir / self.AVAILABILITY_CACHE_FILENAME

    @classmethod
    def _read_server_rows(cls, payload: dict) -> Dict[str, dict]:
        """Extract the server dimension from a v2 (or legacy v1) payload.

        v2 stores ``servers`` directly.  A v1 file stored every row under
        ``datasets``; keep only the non-local rows and drop the local-release
        rows, which are re-derived from disk.  This is what retires the
        stale ``flywire_BANC_*`` rows in old files.
        """
        rows: Dict[str, dict] = {}
        servers = payload.get("servers")
        if isinstance(servers, dict):
            for name, raw in servers.items():
                if not isinstance(raw, dict):
                    continue
                key = canonical_dataset_name(str(name))
                rows[key] = {
                    "state": str(raw.get("state") or SERVER_UNKNOWN),
                    "checked_at": raw.get("checked_at"),
                    "metadata": raw.get("metadata")
                    if isinstance(raw.get("metadata"), dict) else {},
                }
            return rows

        datasets = payload.get("datasets")
        if isinstance(datasets, dict):
            for name, raw in datasets.items():
                if not isinstance(raw, dict):
                    continue
                key = canonical_dataset_name(str(name))
                if is_local_connectome_dataset(key):
                    # Local releases are derived from disk; never trust a
                    # frozen local row.
                    continue
                metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) \
                    else {}
                # v1 stored the count as a top-level field; fold it into the
                # metadata the composer reads so a server-only dataset keeps
                # its count until the next refresh.
                if raw.get("neuron_count"):
                    metadata = {**metadata,
                                "neuron_count": raw.get("neuron_count"),
                                "typed_count": raw.get("typed_count")}
                rows[key] = {
                    # v1 did not record *why* a row was unavailable (no token
                    # vs. empty server response vs. network error), so map a
                    # positive to 'available' and leave an unrecorded negative
                    # as 'unknown' rather than asserting a failure mode we
                    # never captured.  The next refresh resolves it.
                    "state": SERVER_AVAILABLE if raw.get("available")
                    else SERVER_UNKNOWN,
                    "checked_at": raw.get("checked_at"),
                    "metadata": metadata,
                }
        return rows

    def _load_persisted_availability(self) -> None:
        """Load the persisted server dimension, if a file exists."""
        path = self.availability_cache_path
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = None
        if self._availability_loaded and mtime == self._server_rows_mtime:
            return

        rows: Dict[str, dict] = {}
        updated_at: Optional[str] = None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                updated_at = str(payload.get("updated_at") or "") or None
                rows = self._read_server_rows(payload)
        except (OSError, ValueError, TypeError):
            # A corrupt or partially-written file must never prevent the
            # Settings page from loading.
            rows = {}
            updated_at = None

        with self._lock:
            self._server_rows = rows
            self._availability_updated_at = updated_at
            self._availability_loaded = True
            self._server_rows_mtime = mtime

    def _compose(self, dataset: str) -> DatasetInfo:
        """Build one dataset's full, disk-derived status row."""
        key = canonical_dataset_name(str(dataset or "").strip())
        server = self._server_rows.get(key) or self._empty_server_status()
        info = DatasetInfo(
            name=key,
            source=self.source_of(key),
            metadata=dict(server.get("metadata") or {}),
            server_state=str(server.get("state") or SERVER_UNKNOWN),
            server_checked_at=server.get("checked_at"),
        )
        return self._derive_local_fields(info)

    def _catalog_names(self) -> List[str]:
        """Every dataset worth reporting, including ones never touched.

        Persisted server rows + the local-release catalogs + the known
        NeuPrint candidates + anything found under ``datasets/``.  The
        NeuPrint candidates matter on a fresh machine / without a token:
        without them the availability card would silently omit every
        NeuPrint dataset instead of listing it as ``server: unknown``.  No
        network.
        """
        names: List[str] = []
        seen = set()

        def _add(name):
            key = canonical_dataset_name(str(name or "").strip())
            if key and key not in seen:
                seen.add(key)
                names.append(key)

        for name in self._server_rows:
            _add(name)
        for name in self.NEUPRINT_CANDIDATES:
            _add(name)
        for name in self.FLYWIRE_DATASETS:
            _add(name)
        for name in self.BANC_DATASETS:
            _add(name)
        for info in self.get_local_datasets():
            _add(info.name)
        return names

    def get_cached_availability(self) -> Tuple[Dict[str, DatasetInfo], Optional[str]]:
        """Return composed availability rows and the server refresh time.

        Server state comes from the persisted file; local readiness is
        derived from disk on every call, so it cannot go stale.
        """
        self._load_persisted_availability()
        results = {name: self.check_dataset_availability(name)
                   for name in self._catalog_names()}
        return results, self._availability_updated_at

    def get_availability_updated_at(self) -> Optional[str]:
        """Return the timestamp of the last successful server refresh."""
        self._load_persisted_availability()
        return self._availability_updated_at

    def _persist_availability(self, server_rows: Dict[str, dict]) -> str:
        """Persist the server dimension (atomically) and update the runtime cache."""
        updated_at = datetime.now().astimezone().isoformat(timespec="seconds")
        payload = {
            "format": self.AVAILABILITY_CACHE_FORMAT,
            "updated_at": updated_at,
            "servers": server_rows,
        }

        path = self.availability_cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.tmp")
        try:
            temp_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            temp_path.replace(path)
        except OSError:
            try:
                temp_path.unlink()
            except OSError:
                pass
            raise

        with self._lock:
            self._server_rows = dict(server_rows)
            self._availability_updated_at = updated_at
            self._availability_loaded = True
            try:
                self._server_rows_mtime = path.stat().st_mtime
            except OSError:
                self._server_rows_mtime = None
            self._cache.clear()
        return updated_at

    def _load_tokens(self):
        """Load tokens from config.json (primary) then config_local.json."""
        if self._token is not None and self._cave_token is not None:
            return

        loaded = {}
        chain = []
        # config.json comes first per key (the file a GitHub-pulled copy
        # edits directly); the gitignored config_local.json follows in the
        # fallback chain.
        for filename in ("config.json", "config_local.json"):
            config_path = PROJECT_ROOT / filename
            if not config_path.exists():
                continue
            try:
                # utf-8-sig tolerates the UTF-8 BOM that Windows editors
                # prepend to saved JSON files.
                with open(config_path, "r", encoding="utf-8-sig") as f:
                    cfg = json.load(f)
                cfg_tokens = cfg.get("tokens") or {}
                if isinstance(cfg_tokens, dict):
                    for key in ("neuprint", "cave"):
                        value = cfg_tokens.get(key)
                        if isinstance(value, str) and value.strip() and not value.startswith("YOUR_"):
                            # First non-empty value wins: config.json is read first.
                            loaded.setdefault(key, value.strip())
                            if key == "neuprint":
                                chain.append((filename, value.strip()))
            except (OSError, ValueError):
                pass

        if self._token is None:
            self._token = loaded.get("neuprint")
        if self._cave_token is None:
            self._cave_token = loaded.get("cave")
        self._token_chain = chain

    def get_token(self) -> Optional[str]:
        """Get NeuPrint token (config.json -> config_local.json -> env).

        Config wins per the standard chain, so a config update overrides a
        shell-exported NEUPRINT_APPLICATION_CREDENTIALS/NEUPRINT_TOKEN.
        Candidates the NeuPrint server refuses (401/403) are skipped and the
        next location is checked instead of failing the request; a probe
        that cannot run (e.g. offline) never disqualifies a candidate. With
        every candidate refused, the first is returned so callers surface
        the rejected-token error.
        """
        self._load_tokens()
        candidates = []
        if self._token:
            candidates.append(("config", self._token))
            # Deeper config candidates (config_local.json when config.json
            # also supplied a value) stay in the chain behind the winner.
            for source, value in self._token_chain:
                if value != self._token:
                    candidates.append((source, value))
        env_token = (os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
                     or os.environ.get("NEUPRINT_TOKEN"))
        if env_token and env_token.strip() \
                and not env_token.strip().startswith("YOUR_"):
            candidates.append(("environment", env_token.strip()))
        if not candidates:
            return None
        if _shared_token_manager is None or len(candidates) == 1:
            return candidates[0][1]
        first = candidates[0][1]
        probed = set()
        for _source, value in candidates:
            if value in probed:
                continue
            probed.add(value)
            if not _shared_token_manager.neuprint_token_rejected(value):
                return value
        return first

    def get_cave_token(self) -> Optional[str]:
        """Get CAVE token (config.json -> config_local.json -> env)."""
        self._load_tokens()
        if self._cave_token:
            return self._cave_token
        return os.environ.get("CAVE_TOKEN")

    def fetch_neuprint_datasets(self) -> List[str]:
        """
        Fetch the FULL list of available NeuPrint datasets from the server API.
        Uses /api/dbmeta/datasets endpoint with Bearer token auth.
        Falls back to probing known candidates if API call fails.
        Returns a list of available dataset names.
        """
        self._load_tokens()

        token = self.get_token()
        if not token:
            return []

        # Try the proper API endpoint first
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-type": "application/json",
            }
            r = requests.get(
                f"{self.NEUPRINT_SERVER}/api/dbmeta/datasets",
                headers=headers,
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict):
                    # Store full server metadata, excluding datasets the
                    # server marks as hidden (e.g. banc:v888): they are
                    # listed but not queryable through the API.
                    self._server_datasets = data
                    available = sorted(
                        name for name, meta in data.items()
                        if not (isinstance(meta, dict)
                                and str(meta.get("hidden", "")).lower() == "true")
                    )
                    self._available_neuprint = available
                    self._last_fetch_time = time.time()
                    return available
        except Exception:
            pass

        # Fallback: probe known candidates individually
        available = []
        for ds in self.NEUPRINT_CANDIDATES:
            info = self._probe_neuprint_dataset(ds)
            if info.available:
                available.append(ds)

        self._available_neuprint = available
        self._last_fetch_time = time.time()
        return available

    def get_all_datasets(self) -> List[str]:
        """
        Get list of all available datasets (NeuPrint + FAFB + BANC).
        If we have fetched from the server, includes ALL server datasets
        (including older versions like hemibrain:v1.1, fib19:v1.0, etc.).
        """
        if self._available_neuprint is not None:
            return self._available_neuprint + self.FLYWIRE_DATASETS + self.BANC_DATASETS
        return self.NEUPRINT_CANDIDATES + self.FLYWIRE_DATASETS + self.BANC_DATASETS

    def get_neuprint_datasets(self) -> List[str]:
        """Get list of available NeuPrint datasets."""
        if self._available_neuprint is not None:
            return self._available_neuprint.copy()
        return self.NEUPRINT_CANDIDATES.copy()

    def get_flywire_datasets(self) -> List[str]:
        """Get the FAFB datasets backed by the FlyWire/Codex release."""
        return self.FLYWIRE_DATASETS.copy()

    def get_banc_datasets(self) -> List[str]:
        """Get the standalone BANC public-release datasets."""
        return self.BANC_DATASETS.copy()

    def check_dataset_availability(self, dataset: str) -> DatasetInfo:
        """Compose one dataset's status: persisted server state + live disk.

        Read-only and network-free — server probing happens in
        :meth:`refresh_availability`.  Local readiness is re-derived on every
        call so it never goes stale; the result is mirrored into ``_cache``
        for selector helpers.  (A build-time memo was tried and reverted:
        callers legitimately mutate files between calls and expect the next
        call to see them. The per-build dedup that matters happens one level
        up, in the selector-label memo, and the Settings poller runs at a 5s
        cadence.)
        """
        self._load_persisted_availability()
        info = self._compose(dataset)
        with self._lock:
            self._cache[info.name] = info
        return info

    def _probe_banc_bucket(self) -> dict:
        """HEAD the public BANC release object. Cached per session."""
        if self._banc_bucket_probe_cache is not None:
            return dict(self._banc_bucket_probe_cache)
        state = SERVER_UNREACHABLE
        try:
            import banc_public_data
            url = f"{banc_public_data.BUCKET_HTTP_BASE}/" \
                  f"{banc_public_data.META_FEATHER_PATH}"
            size = banc_public_data._remote_size(url)
            state = SERVER_AVAILABLE if size else SERVER_UNREACHABLE
        except Exception as exc:
            # Mirror _fetch_neuprint_counts: classify TimeoutError alike
            # regardless of the exception's exact class name.
            name = type(exc).__name__.lower()
            if "timeout" in name or "timeout" in str(exc).lower() \
                    or "timed out" in str(exc).lower():
                state = SERVER_TIMEOUT
            else:
                state = SERVER_UNREACHABLE
        result = {
            "state": state,
            "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "metadata": {},
        }
        self._banc_bucket_probe_cache = dict(result)
        return result

    def _probe_server(self, dataset: str) -> dict:
        """Probe the *server* dimension for one dataset (network).

        Returns ``{state, checked_at, metadata}`` using the ``SERVER_*``
        vocabulary.  Distinguishes timeout / no-token / unreachable / hidden
        instead of collapsing every failure to "unavailable".
        """
        family = self.family_of(dataset)
        if family == "banc":
            return self._probe_banc_bucket()
        if family == "fafb":
            # FAFB has no connectivity server; CAVE is only a visualization
            # fallback, so its "server" state tracks whether a CAVE token
            # exists (never a connection source — too slow for production).
            return {
                "state": SERVER_AVAILABLE if self.get_cave_token()
                else SERVER_NO_TOKEN,
                "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "metadata": {},
            }
        return self._probe_neuprint_server(dataset)

    def _probe_neuprint_server(self, dataset: str) -> dict:
        """Probe a NeuPrint dataset's server availability."""
        checked_at = datetime.now().astimezone().isoformat(timespec="seconds")
        if not self.get_token():
            return {"state": SERVER_NO_TOKEN, "checked_at": checked_at,
                    "metadata": {}}

        # Fast path: dataset list already fetched this session.
        if self._server_datasets and dataset in self._server_datasets:
            server_info = self._server_datasets[dataset] or {}
            if str(server_info.get("hidden", "")).lower() == "true":
                return {"state": SERVER_HIDDEN, "checked_at": checked_at,
                        "metadata": {}}
            metadata = {
                "server": self.NEUPRINT_SERVER,
                "last_mod": server_info.get("last-mod", ""),
                "uuid": server_info.get("uuid", ""),
                "rois": server_info.get("ROIs", []),
            }
            # The server list carries no counts; record them once here (the
            # read path is network-free) when no local table can supply them.
            if not self._neuron_table_present(dataset):
                total, typed = self._fetch_neuprint_counts(dataset)
                if total:
                    metadata["neuron_count"] = total
                    metadata["typed_count"] = typed
            return {"state": SERVER_AVAILABLE, "checked_at": checked_at,
                    "metadata": metadata}

        total, typed, state = self._fetch_neuprint_counts(dataset, with_state=True)
        metadata = {"server": self.NEUPRINT_SERVER, "checked_at": checked_at}
        if total:
            metadata["neuron_count"] = total
            metadata["typed_count"] = typed
        return {"state": state, "checked_at": checked_at, "metadata": metadata}

    def _fetch_neuprint_counts(self, dataset: str, with_state: bool = False):
        """Query the NeuPrint server for a dataset's total/typed count.

        Returns ``(0, 0)`` (or ``(0, 0, state)`` when ``with_state``) when the
        dataset cannot be queried.  Failure modes are classified rather than
        swallowed so the UI can show timeout vs no-token vs unreachable.
        """
        token = self.get_token()
        if not token:
            return (0, 0, SERVER_NO_TOKEN) if with_state else (0, 0)
        try:
            from neuprint import Client

            client = Client(self.NEUPRINT_SERVER, dataset, token)
            result = client.fetch_custom(
                "MATCH (n:Neuron) RETURN count(n) as total, "
                "sum(CASE WHEN n.type IS NOT NULL AND n.type <> '' THEN 1 ELSE 0 END) as typed"
            )
            if not result.empty:
                counts = (int(result["total"].iloc[0]), int(result["typed"].iloc[0]))
                return (*counts, SERVER_AVAILABLE) if with_state else counts
            return (0, 0, SERVER_UNREACHABLE) if with_state else (0, 0)
        except Exception as exc:
            state = SERVER_UNREACHABLE
            name = type(exc).__name__.lower()
            text = str(exc).lower()
            if "timeout" in name or "timeout" in text or "timed out" in text:
                state = SERVER_TIMEOUT
            return (0, 0, state) if with_state else (0, 0)

    def fetch_codex_datasets(self) -> Dict[str, dict]:
        """
        Fetch available FAFB/BANC release names from the Codex catalog
        (codex.flywire.ai).
        Returns dict of {dataset_name: {display, desc, neurons}}.
        Falls back to hardcoded CODEX_DATASETS if fetch fails.
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            r = requests.get(self.CODEX_URL, timeout=10)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                # Look for dataset cards in the rendered HTML
                # The page structure has dataset names like "FAFB v783 (CB)"
                text = soup.get_text()
                # Parse known patterns
                import re
                patterns = [
                    (r'FAFB\s+v(\d+)', 'flywire_FAFB_v{}', 'FAFB v{} (CB)', 'Female Adult Fly Brain'),
                    (r'BANC\s+v(\d+)', 'banc_v{}', 'BANC v{} (CNS)', 'Brain and Nerve Cord'),
                ]
                found = {}
                for pattern, key_fmt, display_fmt, desc in patterns:
                    matches = re.findall(pattern, text)
                    for ver in matches:
                        key = key_fmt.format(ver)
                        display = display_fmt.format(ver)
                        found[key] = {"display": display, "desc": desc, "neurons": 0}
                if found:
                    # Update our CODEX_DATASETS with fetched info
                    self.CODEX_DATASETS.update(found)
                    # Keep the two release catalogs separate. BANC may be
                    # advertised by Codex, but it is not a FlyWire source.
                    for k in found:
                        target = (
                            self.BANC_DATASETS
                            if is_banc_dataset(k)
                            else self.FLYWIRE_DATASETS
                        )
                        if k not in target:
                            target.append(k)
                    return found
        except Exception:
            pass
        return self.CODEX_DATASETS

    def _probe_neuprint_dataset(self, dataset: str) -> DatasetInfo:
        """Compatibility shim: probe one NeuPrint dataset (network).

        Returns a ``DatasetInfo`` whose ``available`` reflects the server
        state and whose ``error`` carries the failure reason.  Used by the
        candidate-probe fallback in :meth:`fetch_neuprint_datasets` and by
        tests; the Settings flow uses :meth:`_probe_neuprint_server`.
        """
        info = DatasetInfo(name=dataset, source="neuprint")
        server = self._probe_neuprint_server(dataset)
        info.server_state = server["state"]
        info.server_checked_at = server["checked_at"]
        info.metadata = dict(server.get("metadata") or {})
        info.available = server["state"] == SERVER_AVAILABLE
        if not info.available:
            info.error = {
                SERVER_NO_TOKEN: "No NeuPrint token configured",
                SERVER_HIDDEN: "Hidden on the NeuPrint server "
                               "(not queryable through the API)",
                SERVER_TIMEOUT: "Timed out contacting the NeuPrint server",
                SERVER_UNREACHABLE: "NeuPrint server unreachable",
            }.get(server["state"], "Unavailable on the NeuPrint server")
        return info

    def _check_local_cache(self, dataset: str) -> bool:
        """Check if dataset has local cache files."""
        dataset_path = self._get_dataset_path(dataset)
        if dataset_path and dataset_path.exists():
            for pattern in ["*_neuron_df.csv", "*_neuron_df.parquet", "*_allneurons*.csv"]:
                if list(dataset_path.glob(pattern)):
                    return True

        cache_path = self._cache_dir / dataset_to_folder(dataset)
        if cache_path.exists():
            if (cache_path / "connections.parquet").exists():
                return True
            # The neuron index is an app-owned "system file" now
            # (neuron_indexes/), not part of the cache.

        return False

    def _check_local_prepared(self, dataset: str) -> bool:
        """Check if dataset has local data files ready for analysis (not just cache)."""
        dataset_path = self._get_dataset_path(dataset)
        if dataset_path and dataset_path.exists():
            if not is_local_connectome_dataset(dataset):
                # NeuPrint can legitimately use a local neuron table while
                # connections remain server-backed; preserve that behavior.
                return any(
                    path
                    for pattern in ("*_neuron_df.csv", "*_neuron_df.parquet")
                    for path in dataset_path.glob(pattern)
                )

            # A FAFB/BANC conversion is usable only when both generated
            # tables exist.  A neuron table by itself is not enough for pathfinding:
            # the converter also writes the merged connection table.
            neuron_ready = any(
                path
                for pattern in (
                    "*_allneurons_neuron_df.parquet",
                    "*_allneurons_neuron_df.csv",
                )
                for path in dataset_path.glob(pattern)
            )
            connections_ready = any(
                path
                for pattern in (
                    "*_merged_connections.parquet",
                    "*_merged_connections.csv",
                )
                for path in dataset_path.glob(pattern)
            )
            if neuron_ready and connections_ready:
                return True
        return False

    def _get_dataset_path(self, dataset: str) -> Optional[Path]:
        """Get the local path for a dataset."""
        safe_name = dataset_to_folder(dataset)
        return self._datasets_dir / safe_name

    def _find_metadata_file(self, dataset: str) -> Optional[Path]:
        """Find metadata file for a dataset."""
        dataset_path = self._get_dataset_path(dataset)
        if not dataset_path or not dataset_path.exists():
            return None

        safe_name = dataset_to_folder(dataset)
        metadata_file = dataset_path / f"{safe_name}_metadata.json"
        if metadata_file.exists():
            return metadata_file

        for f in dataset_path.glob("*_metadata.json"):
            return f

        return None

    # ------------------------------------------------------------------
    # Dimension probes (disk-derived)
    # ------------------------------------------------------------------

    @staticmethod
    def family_of(dataset: str) -> str:
        """Return the dataset family: 'neuprint' | 'fafb' | 'banc'."""
        if is_banc_dataset(dataset):
            return "banc"
        if is_fafb_dataset(dataset):
            return "fafb"
        return "neuprint"

    @staticmethod
    def source_of(dataset: str) -> str:
        """Return the ``DatasetInfo.source`` value for *dataset*.

        Keeps the established vocabulary ('neuprint' | 'flywire' | 'banc')
        for compatibility; ``family`` carries the precise 'fafb' spelling.
        """
        family = DatasetService.family_of(dataset)
        return "flywire" if family == "fafb" else family

    @staticmethod
    def access_mode_of(dataset: str) -> str:
        """NeuPrint streams on demand; standalone BANC is never streamable."""
        return ACCESS_STREAMING if DatasetService.family_of(dataset) == "neuprint" \
            else ACCESS_DOWNLOAD_REQUIRED

    @staticmethod
    def _has_any(dataset_path: Optional[Path], patterns) -> bool:
        if not dataset_path or not dataset_path.exists():
            return False
        return any(matches for pattern in patterns
                   for matches in [list(dataset_path.glob(pattern))])

    def _neuron_table_present(self, dataset: str) -> bool:
        return self._has_any(self._get_dataset_path(dataset), (
            "*_allneurons_neuron_df.parquet", "*_allneurons_neuron_df.csv",
            "*_neuron_df.parquet", "*_neuron_df.csv",
        ))

    def _allneurons_table_present(self, dataset: str) -> bool:
        """The strict converter output used by FAFB/BANC preparation."""
        return self._has_any(self._get_dataset_path(dataset), (
            "*_allneurons_neuron_df.parquet", "*_allneurons_neuron_df.csv",
        ))

    def _roi_table_present(self, dataset: str) -> bool:
        return self._has_any(self._get_dataset_path(dataset), (
            "*_allneurons_roi_count_df.parquet", "*_allneurons_roi_count_df.csv",
            "*_roi_count_df.parquet", "*_roi_count_df.csv",
        ))

    def _neuron_index_present(self, dataset: str) -> bool:
        index = self._index_dir / dataset_to_folder(dataset) / "neuron_index.parquet"
        return index.exists()

    def _merged_connections_present(self, dataset: str) -> bool:
        return self._has_any(self._get_dataset_path(dataset), (
            "*_merged_connections.parquet", "*_merged_connections.csv",
        ))

    def probe_metadata(self, dataset: str) -> str:
        """Metadata readiness — the basic-for-analysis state.

        NeuPrint metadata is the neuron table + ROI table + materialized
        neuron index (what ``Pull Dataset Metadata`` produces and what the
        connectivity pull reads first).  FAFB/BANC metadata is the local
        neuron table from the converter/bucket.
        """
        if self.family_of(dataset) == "neuprint":
            parts = (
                self._neuron_table_present(dataset),
                self._roi_table_present(dataset),
                self._neuron_index_present(dataset),
            )
            if all(parts):
                return CAP_READY
            if any(parts):
                return CAP_PARTIAL
            return CAP_MISSING
        # FAFB/BANC: the converter's ``*_allneurons_neuron_df`` output (the
        # strict spelling ``_check_local_prepared`` requires) is the metadata.
        return CAP_READY if self._allneurons_table_present(dataset) else CAP_MISSING

    def probe_connectivity(self, dataset: str) -> str:
        """Connectivity readiness.  FAFB/BANC are local-only (CAVE is too
        slow for production); NeuPrint streams when no cache exists."""
        family = self.family_of(dataset)
        if family in ("fafb", "banc"):
            return CAP_READY if self._merged_connections_present(dataset) else CAP_MISSING
        cache_conn = self._cache_dir / dataset_to_folder(dataset) / "connections.parquet"
        if cache_conn.exists():
            return CAP_READY
        return CAP_ON_DEMAND  # server-streamed

    def probe_visualization(self, dataset: str) -> Tuple[str, Optional[str]]:
        """Return (state, source) for skeleton/mesh availability.

        BANC fetches on demand from the public bucket; FAFB uses the local
        healed bundle or the slow CAVE fallback; NeuPrint streams.
        """
        family = self.family_of(dataset)
        if family == "banc":
            skel_dir = (self._cache_dir / dataset_to_folder(dataset)
                        / "skeletons" / "raw_skeletons")
            if skel_dir.is_dir() and any(skel_dir.glob("*.swc*")):
                return CAP_READY, "local"
            return CAP_ON_DEMAND, "bucket"
        if family == "fafb":
            dataset_path = self._get_dataset_path(dataset)
            if self._has_any(dataset_path, ("sk_lod1_783_healed.zip", "sk_*.zip")):
                return CAP_READY, "local"
            cave_dir = (self._cache_dir / dataset_to_folder(dataset)
                        / "skeletons" / "cave_skeletons")
            if cave_dir.is_dir() and any(cave_dir.iterdir()):
                return CAP_READY, "local"
            if self.get_cave_token():
                return CAP_ON_DEMAND, "cave"
            return CAP_MISSING, None
        return CAP_ON_DEMAND, "server"

    def _derive_local_fields(self, info: DatasetInfo) -> DatasetInfo:
        """Fill the derived dimensions, counts, and the ``available`` bool."""
        dataset = info.name
        info.family = self.family_of(dataset)
        info.access_mode = self.access_mode_of(dataset)
        info.metadata_state = self.probe_metadata(dataset)
        info.connectivity_state = self.probe_connectivity(dataset)
        info.visualization_state, info.visualization_source = \
            self.probe_visualization(dataset)

        # Reuse the established local-data probes so the card and the dataset
        # selectors (`_dataset_label_parts`) never disagree about "local".
        # The dimension chips above carry the finer-grained truth.
        info.local_prepared = self._check_local_prepared(dataset)
        info.local_cache = self._check_local_cache(dataset)

        if info.neuron_count == 0:
            total, typed = self._load_local_neuron_counts(dataset)
            info.neuron_count = total
            info.typed_count = typed
        if info.neuron_count == 0 and info.metadata:
            # Server-recorded counts (written during refresh for datasets
            # with no local table).
            info.neuron_count = int(info.metadata.get("neuron_count") or 0)
            info.typed_count = int(info.metadata.get("typed_count") or 0)
        if info.neuron_count == 0 and info.family in ("fafb", "banc"):
            codex = self.CODEX_DATASETS.get(dataset) or {}
            if codex.get("neurons"):
                info.neuron_count = int(codex["neurons"])

        if info.family in ("fafb", "banc"):
            # A download-required release is analyzable only once BOTH its
            # neuron table and its connection table exist; server
            # reachability alone is not enough, and neither is one table on
            # its own.  ``local_prepared`` encodes that conjunction.
            info.available = info.local_prepared
            if not info.available:
                source_name = "BANC" if info.family == "banc" else "FAFB"
                info.error = info.error or (
                    f"Local {source_name} neuron and connection tables are not "
                    "both prepared."
                )
        else:
            info.available = info.server_state == SERVER_AVAILABLE

        if not info.display_name:
            info.display_name = (
                (self.CODEX_DATASETS.get(dataset) or {}).get("display")
                or dataset
            )
        return info

    def _empty_server_status(self) -> dict:
        return {"state": SERVER_UNKNOWN, "checked_at": None, "metadata": {}}

    # neuron-count memoization: (dataset, mtime_ns) -> (total, typed).
    # Counting a large neuron CSV on every page load is wasteful; the count is
    # re-read only when the local table changes.
    _neuron_counts_cache: Dict[tuple, tuple] = {}

    def _load_local_neuron_counts(self, dataset: str) -> tuple:
        """
        Total / typed neuron counts for a locally prepared dataset.

        Priority:
          1. ``*_metadata.json`` (``neuron_counts.total`` / ``typed``)
          2. local neuron table (``*_allneurons_neuron_df.parquet`` or
             ``.csv``, or a plain ``*_neuron_df.parquet``/``.csv`` from a
             NeuPrint conversion) counted via a streaming Polars scan - this
             covers datasets that were pulled/downloaded without a metadata
             file.

        The result is memoized per (dataset, table mtime); falls back to
        (0, 0) when neither source is available.
        """
        dataset_path = self._get_dataset_path(dataset)
        if not dataset_path or not dataset_path.exists():
            return 0, 0

        # 1. Metadata file wins when present (it is authoritative and cheap).
        metadata_file = self._find_metadata_file(dataset)
        if metadata_file and metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    meta = json.load(f)
                counts = meta.get("neuron_counts", {}) or {}
                if counts.get("total"):
                    return int(counts["total"]), int(counts.get("typed", 0) or 0)
            except Exception:
                pass

        # 2. Fall back to counting the local neuron table.  Prefer the full
        #    ``*_allneurons_neuron_df.*`` table; the plain ``*_neuron_df.*``
        #    names must match too, or prepared datasets would show no count.
        table = None
        for pattern in (
            "*_allneurons_neuron_df.parquet",
            "*_allneurons_neuron_df.csv",
            "*_neuron_df.parquet",
            "*_neuron_df.csv",
        ):
            matches = sorted(dataset_path.glob(pattern))
            if matches:
                table = matches[0]
                break
        if table is None:
            return 0, 0

        key = (dataset, table.stat().st_mtime_ns)
        if key in self._neuron_counts_cache:
            return self._neuron_counts_cache[key]

        total = typed = 0
        try:
            import polars as pl

            lazy = pl.scan_parquet(table) if table.suffix == ".parquet" else pl.scan_csv(
                table, infer_schema_length=0, ignore_errors=True
            )
            if "type" in lazy.collect_schema().names():
                # NOTE: do not apply a frame-level `.sum()` to `pl.len()` -
                # that corrupts the total (observed as a ~158k-row table
                # summing to billions).
                row = lazy.select(
                    pl.len().alias("total"),
                    (
                        pl.col("type").is_not_null()
                        & (pl.col("type") != "")
                        & (pl.col("type").cast(pl.Utf8) != "nan")
                    ).sum().alias("typed"),
                ).collect().row(0)
                total, typed = int(row[0]), int(row[1])
            else:
                total = int(lazy.select(pl.len()).collect().item())
        except Exception:
            # Last resort: plain pandas row count
            try:
                import pandas as pd

                if table.suffix == ".parquet":
                    df = pd.read_parquet(table)
                else:
                    df = pd.read_csv(table, index_col=0, low_memory=False)
                total = len(df)
                typed = int(df["type"].notna().sum()) if "type" in df.columns else 0
            except Exception:
                return 0, 0

        self._neuron_counts_cache[key] = (total, typed)
        return total, typed

    def _load_cache_neuron_counts(self, dataset: str) -> tuple:
        """
        Total / typed neuron counts from the app-owned neuron index.

        ``neuron_indexes/<dataset>/neuron_index.parquet`` is written by the
        DatasetPuller with one row per cached neuron (or shipped as a bundled
        seed).  The count may be partial (only what has been pulled so far),
        so it is used only as a last resort behind the dataset tables and the
        server query.
        """
        index = self._index_dir / dataset_to_folder(dataset) / "neuron_index.parquet"
        if not index.exists():
            return 0, 0

        key = ("cache", dataset, index.stat().st_mtime_ns)
        if key in self._neuron_counts_cache:
            return self._neuron_counts_cache[key]

        total = typed = 0
        try:
            import polars as pl

            lazy = pl.scan_parquet(index)
            if "type" in lazy.collect_schema().names():
                row = lazy.select(
                    pl.len().alias("total"),
                    (
                        pl.col("type").is_not_null()
                        & (pl.col("type") != "")
                        & (pl.col("type").cast(pl.Utf8) != "nan")
                    ).sum().alias("typed"),
                ).collect().row(0)
                total, typed = int(row[0]), int(row[1])
            else:
                total = int(lazy.select(pl.len()).collect().item())
        except Exception:
            return 0, 0

        self._neuron_counts_cache[key] = (total, typed)
        return total, typed

    def get_local_datasets(self) -> List[DatasetInfo]:
        """Get information about locally present datasets.

        Network-free: each row carries the disk-derived dimensions, so the
        catalog is correct on a fresh machine before any server refresh.
        """
        datasets = []

        if not self._datasets_dir.exists():
            return datasets

        for folder in self._datasets_dir.iterdir():
            if folder.is_dir() and not folder.name.startswith("."):
                name = folder_to_dataset(folder.name)
                info = self._derive_local_fields(
                    DatasetInfo(name=name, source=self.source_of(name)))
                metadata_file = self._find_metadata_file(name)
                if metadata_file and metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            local_meta = json.load(f)
                        # A server row's metadata wins when present; otherwise
                        # surface the local sidecar.
                        info.metadata = info.metadata or local_meta
                    except Exception:
                        pass
                datasets.append(info)

        return datasets

    def refresh_availability(self, datasets: Optional[List[str]] = None) -> Dict[str, DatasetInfo]:
        """Refresh the **server** dimension and return composed rows.

        Fetches release names from Codex, probes each dataset's server state,
        and persists only that server dimension.  Local readiness is always
        derived from disk, so it is never written.  A full refresh (``None``)
        is authoritative and replaces the server rows; a targeted list
        updates only those rows and preserves the rest.
        """
        full_refresh = datasets is None
        with self._lock:
            self._cache.clear()

        if full_refresh:
            # Fetch release names from Codex (FAFB and BANC remain separate).
            self.fetch_codex_datasets()
            # Fetch NeuPrint datasets from server
            neuprint_available = self.fetch_neuprint_datasets()
            datasets = (neuprint_available + self.FLYWIRE_DATASETS
                        + self.BANC_DATASETS)

        self._banc_bucket_probe_cache = None  # re-probe once per refresh
        self._load_persisted_availability()
        server_rows: Dict[str, dict] = (
            {} if full_refresh else dict(self._server_rows))

        for dataset in datasets:
            key = canonical_dataset_name(str(dataset or "").strip())
            if not key:
                continue
            server_rows[key] = self._probe_server(key)

        self._persist_availability(server_rows)

        return {name: self.check_dataset_availability(name)
                for name in self._catalog_names()}

    def is_cache_fresh(self, max_age_seconds: int = 300) -> bool:
        """Check if the cached availability data is still fresh."""
        if self._last_fetch_time == 0:
            return False
        return (time.time() - self._last_fetch_time) < max_age_seconds


# Global instance
_dataset_service: Optional[DatasetService] = None


def get_dataset_service() -> DatasetService:
    """Get the global DatasetService instance."""
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService()
    return _dataset_service
