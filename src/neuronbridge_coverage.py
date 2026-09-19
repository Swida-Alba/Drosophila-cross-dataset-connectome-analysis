"""NeuronBridge dataset-coverage snapshots.

NeuronBridge hosts EM bodies for a subset of DROCAT's datasets, and the
hosted set changes with NeuronBridge data releases (probed live: v3_10_0
hosts hemibrain:v1.2.1, male-cns:v0.9, manc:v1.2.1 and FlyWire FAFB v783;
male-cns:v1.0 bodies live only in v0.9 libraries; BANC/optic-lobe are
absent).  This module keeps an on-disk snapshot of that coverage so the
Find Lines tab can disable datasets NeuronBridge does not serve and route
queries at the releases it does.

Design contract (plan `_plan/plan-nb-find-lines-dataset-aware-queries.md`
§5): the coverage state is ADVISORY.  ``unknown`` (refresh failed, or no
local table to sample from) never disables a dataset, and even
``unavailable`` only warns — a stale snapshot must never lock out a
dataset that actually works.  An unreachable NeuronBridge never invalidates
a persisted snapshot.

There is no name/library index in the NeuronBridge bucket, so coverage is
measured the only reliable way: probe ``metadata/by_body/{id}.json`` for a
small sample of typed bodyIds drawn from each dataset's LOCAL neuron
table, and classify the returned ``publishedName`` identities:

* ``exact``       — some record matches base + version.
* ``aligned``     — no version match, but some record matches the base
                    at another hosted version (male-cns:v1.0 → v0.9).
* ``unavailable`` — the sample answered definitively, nothing matched.
* ``unknown``     — no usable answer (offline, no sample, no table).

A single-body 404 inside a hosted dataset is expected noise, so several
typed bodies are probed and ANY identity match settles the dataset.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote

try:  # ``src/`` on sys.path (runner scripts) vs package imports
    from src.neuron_index_builder import (
        metadata_candidates,
        read_metadata_projection,
    )
    from src.utils.label_utils import is_untyped_type_label
except ImportError:  # pragma: no cover - bare ``src`` layout
    from neuron_index_builder import (
        metadata_candidates,
        read_metadata_projection,
    )
    from utils.label_utils import is_untyped_type_label

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TTL_DAYS = 7.0
SAMPLE_FIRST_ROUND = 2
SAMPLE_SECOND_ROUND = 3
PROBE_WORKERS = 8
SNAPSHOT_FILENAME = "coverage_snapshot.json"

STATUS_EXACT = "exact"
STATUS_ALIGNED = "aligned"
STATUS_UNAVAILABLE = "unavailable"
STATUS_UNKNOWN = "unknown"

_LOCK = threading.Lock()


def default_cache_root() -> Path:
    """The NeuronBridge cache root shared with the Finder."""
    return PROJECT_ROOT / "cache" / "neuronbridge"


def _utc_now() -> datetime:
    """The single clock seam: stamps and TTL ages both derive from it."""
    return datetime.now(timezone.utc)


def _now() -> str:
    return _utc_now().isoformat(timespec="seconds")


@dataclass
class DatasetCoverage:
    """Coverage verdict for one dataset."""

    dataset: str
    status: str = STATUS_UNKNOWN
    hosted_version: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    checked_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "hosted_version": self.hosted_version,
            "evidence": list(self.evidence),
            "checked_at": self.checked_at,
        }

    @classmethod
    def from_dict(cls, dataset: str, payload: Dict[str, Any]) -> "DatasetCoverage":
        payload = payload if isinstance(payload, dict) else {}
        return cls(
            dataset=dataset,
            status=str(payload.get("status") or STATUS_UNKNOWN),
            hosted_version=payload.get("hosted_version"),
            evidence=[
                str(item) for item in (payload.get("evidence") or []) if item
            ],
            checked_at=payload.get("checked_at"),
        )


@dataclass
class CoverageSnapshot:
    """A persisted coverage verdict set for one NeuronBridge version."""

    nb_version: Optional[str]
    created_at: str
    datasets: Dict[str, DatasetCoverage] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nb_version": self.nb_version,
            "created_at": self.created_at,
            "datasets": {
                dataset: coverage.to_dict()
                for dataset, coverage in sorted(self.datasets.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CoverageSnapshot":
        payload = payload if isinstance(payload, dict) else {}
        datasets = {
            str(dataset): DatasetCoverage.from_dict(dataset, entry)
            for dataset, entry in (payload.get("datasets") or {}).items()
        }
        return cls(
            nb_version=payload.get("nb_version"),
            created_at=str(payload.get("created_at") or ""),
            datasets=datasets,
        )

    def coverage_of(self, dataset: str) -> DatasetCoverage:
        return self.datasets.get(
            str(dataset),
            DatasetCoverage(dataset=str(dataset), status=STATUS_UNKNOWN),
        )

    def unavailable_datasets(self, candidates: Sequence[str]) -> List[str]:
        """Datasets the snapshot classifies as not hosted (advisory)."""
        return [
            dataset
            for dataset in candidates
            if self.coverage_of(dataset).status == STATUS_UNAVAILABLE
        ]

    def covered_datasets(self, candidates: Sequence[str]) -> Dict[str, str]:
        """Datasets NeuronBridge can serve, mapped to the release to query.

        ``exact`` datasets map to themselves; ``aligned`` datasets map to
        their hosted release (male-cns:v1.0 → male-cns:v0.9).  Everything
        else — including ``unknown`` — is left out of expansion targets,
        where a mistake would silently route bodies at the wrong library;
        ``unknown`` datasets still go through the ordinary query path.
        """
        covered: Dict[str, str] = {}
        for dataset in candidates:
            coverage = self.coverage_of(dataset)
            if coverage.status == STATUS_EXACT:
                covered[dataset] = dataset
            elif coverage.status == STATUS_ALIGNED and coverage.hosted_version:
                covered[dataset] = canonical_hosted_name(coverage.hosted_version)
        return covered


def snapshot_path(cache_root: Optional[os.PathLike] = None) -> Path:
    root = Path(cache_root) if cache_root else default_cache_root()
    return root / SNAPSHOT_FILENAME


def load_snapshot(
    cache_root: Optional[os.PathLike] = None,
) -> Optional[CoverageSnapshot]:
    """Load the persisted snapshot, or ``None`` when absent/corrupt."""
    path = snapshot_path(cache_root)
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return CoverageSnapshot.from_dict(payload)
    except (OSError, ValueError):
        return None


def _save_snapshot(
    snapshot: CoverageSnapshot, cache_root: Optional[os.PathLike] = None
) -> Path:
    path = snapshot_path(cache_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(snapshot.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _age_days(stamp: Optional[str], now: Optional[datetime] = None) -> float:
    if not stamp:
        return float("inf")
    try:
        created = datetime.fromisoformat(str(stamp))
    except ValueError:
        return float("inf")
    delta = (now if now is not None else _utc_now()) - created
    return max(delta.total_seconds(), 0.0) / 86400.0


def _load_local_projection(dataset: str):
    """Read the dataset's local metadata projection, or ``None``."""
    datasets_dir = PROJECT_ROOT / "datasets"
    try:
        candidates = metadata_candidates(str(dataset), datasets_dir)
    except Exception:
        return None
    for candidate in candidates:
        try:
            return read_metadata_projection(candidate)
        except Exception:
            continue
    return None


def sample_body_ids(
    dataset: str, limit: int = SAMPLE_FIRST_ROUND + SAMPLE_SECOND_ROUND
) -> List[str]:
    """Typed bodyIds to probe, one per distinct type, deterministic.

    The first rows of a dataset table are frequently untyped, and untyped
    neurons carry no library identity worth probing, so sampling walks the
    local table once and picks the first bodyId of ``limit`` distinct real
    types (the shared untyped predicate of the analysis tabs).  Returns
    ``[]`` when the dataset has no local table — the caller then
    classifies the dataset as ``unknown`` instead of ``unavailable``.
    """
    frame = _load_local_projection(dataset)
    if frame is None:
        return []
    columns = set(frame.columns)
    if "bodyId" not in columns:
        return []
    type_column = "type" if "type" in columns else (
        "instance" if "instance" in columns else None
    )
    try:
        import polars as pl

        body_ids = frame["bodyId"].cast(pl.Utf8, strict=False).to_list()
        types = (
            frame[type_column].cast(pl.Utf8, strict=False).to_list()
            if type_column is not None
            else [None] * frame.height
        )
    except Exception:
        return []

    picked: List[str] = []
    seen_types: set = set()
    seen_ids: set = set()
    for body_id, neuron_type in zip(body_ids, types):
        text = str(body_id or "").strip()
        if not text or text in seen_ids:
            continue
        if neuron_type and is_untyped_type_label(neuron_type):
            continue
        if type_column is not None:
            if not neuron_type or neuron_type in seen_types:
                continue
            seen_types.add(neuron_type)
        seen_ids.add(text)
        picked.append(text)
        if len(picked) >= limit:
            break
    return picked


def _identity_of(dataset: str) -> Tuple[str, Optional[str]]:
    """Dataset string → ``(base, version)`` with the Finder's semantics."""
    try:
        from neuronbridge_finder import NeuronBridgeFinder
    except ImportError:  # pragma: no cover - ``src`` layout
        from src.neuronbridge_finder import NeuronBridgeFinder

    return NeuronBridgeFinder._dataset_identity(dataset)


def canonical_hosted_name(hosted: str) -> str:
    """A published identity prefix → DROCAT's dataset display name.

    NeuronBridge publishes ``flywire_fafb:v783`` while the Finder's
    resolvable spelling is ``flywire_FAFB_v783``; identity comparison (not
    string comparison) matches the two.  Unmapped prefixes pass through.
    """
    try:
        from neuronbridge_finder import LIBRARY_TO_DATASET_NAME
    except ImportError:  # pragma: no cover - ``src`` layout
        from src.neuronbridge_finder import LIBRARY_TO_DATASET_NAME

    target = _identity_of(hosted)
    if not target[0]:
        return hosted
    for name in LIBRARY_TO_DATASET_NAME.values():
        if _identity_of(name) == target:
            return name
    return hosted


def _record_field(record: Any, name: str) -> str:
    """Read a field off an APIObject record or its raw-dict form."""
    if isinstance(record, dict):
        return str(record.get(name, "") or "")
    return str(getattr(record, name, "") or "")


def _published_identity(em_record: Any) -> Optional[str]:
    published = _record_field(em_record, "publishedName")
    parts = published.split(":")
    if len(parts) >= 3 and parts[-1].isdigit():
        return ":".join(parts[:-1])
    return published or None


def classify_records(
    dataset: str,
    records: Iterable[Any],
) -> DatasetCoverage:
    """Classify one dataset from the by_body records of its sample.

    Any record matching base + version settles ``exact``; otherwise any
    base-only match yields ``aligned`` at that record's identity.  Records
    with an unrelated base (the optic-lobe/MANC small-id collisions) are
    ignored — they are different neurons that happen to share a number.
    """
    expected_base, expected_version = _identity_of(dataset)
    if not expected_base:
        return DatasetCoverage(
            dataset=dataset, status=STATUS_UNKNOWN, checked_at=_now()
        )
    aligned_version: Optional[str] = None
    evidence: List[str] = []
    for record in records:
        actual = _published_identity(record)
        if not actual:
            continue
        actual_base, actual_version = _identity_of(actual)
        library = _record_field(record, "libraryName") or actual
        if actual_base != expected_base:
            continue
        if expected_version is None or actual_version == expected_version:
            if library not in evidence:
                evidence.append(library)
            return DatasetCoverage(
                dataset=dataset,
                status=STATUS_EXACT,
                hosted_version=actual,
                evidence=evidence,
                checked_at=_now(),
            )
        if aligned_version is None:
            aligned_version = actual
        if library not in evidence:
            evidence.append(library)
    if aligned_version is not None:
        return DatasetCoverage(
            dataset=dataset,
            status=STATUS_ALIGNED,
            hosted_version=aligned_version,
            evidence=evidence,
            checked_at=_now(),
        )
    return DatasetCoverage(
        dataset=dataset,
        status=STATUS_UNAVAILABLE,
        evidence=evidence,
        checked_at=_now(),
    )


def _fetch_body_records(client: Any, body_id: str):
    """Fetch by_body records; ``(id, None, exc)`` means a failed request."""
    encoded = quote(str(body_id), safe="")
    url = f"{client.data_url}/metadata/by_body/{encoded}.json"
    try:
        payload = client._get_json(url)
    except Exception as exc:  # noqa: BLE001 - classified by the caller
        return str(body_id), None, exc
    results = getattr(payload, "results", None)
    if results is None and isinstance(payload, dict):
        results = payload.get("results")
    return str(body_id), list(results or []), None


def _is_miss(exc: Optional[Exception]) -> bool:
    """True for a definitive 404-style miss (body simply not hosted)."""
    if exc is None:
        return False
    response = getattr(getattr(exc, "__cause__", None), "response", None)
    status = getattr(response, "status_code", None)
    if status is not None:
        return int(status) == 404
    return "404" in str(exc)


def probe_coverage(
    dataset: str,
    body_ids: Sequence[str],
    client: Any,
    max_workers: int = PROBE_WORKERS,
) -> DatasetCoverage:
    """Probe one dataset's sample and return its coverage verdict.

    Two rounds of probes; the accumulated record set is re-classified
    after each round and ANY exact match settles the dataset early.  A
    round in which every request failed for network reasons (no definitive
    misses, no records) short-circuits to ``unknown``.
    """
    if not body_ids:
        # No local table to sample: never claim "unavailable" from silence.
        return DatasetCoverage(
            dataset=dataset, status=STATUS_UNKNOWN, checked_at=_now()
        )

    rounds = (
        list(body_ids[:SAMPLE_FIRST_ROUND]),
        list(body_ids[SAMPLE_FIRST_ROUND:SAMPLE_FIRST_ROUND + SAMPLE_SECOND_ROUND]),
    )
    all_records: List[Any] = []
    verdict: Optional[DatasetCoverage] = None

    for round_ids in rounds:
        if not round_ids:
            break
        round_records: List[Any] = []
        round_misses = 0
        round_errors = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_fetch_body_records, client, body_id)
                for body_id in round_ids
            ]
            for future in as_completed(futures):
                _body_id, body_records, exc = future.result()
                if exc is not None:
                    if _is_miss(exc):
                        round_misses += 1
                    else:
                        round_errors += 1
                elif body_records:
                    round_records.extend(body_records)
                else:
                    round_misses += 1
        if round_errors and not round_misses and not round_records:
            # Network failure, not a hosting verdict.
            return DatasetCoverage(
                dataset=dataset, status=STATUS_UNKNOWN, checked_at=_now()
            )
        if round_records:
            all_records.extend(round_records)
            verdict = classify_records(dataset, all_records)
            if verdict.status == STATUS_EXACT:
                return verdict

    if verdict is not None:
        return verdict
    return DatasetCoverage(
        dataset=dataset, status=STATUS_UNAVAILABLE, checked_at=_now()
    )


def _make_client(client: Any):
    if client is not None:
        return client
    try:
        from neuronbridge_client import Client as NBClient
    except ImportError:  # pragma: no cover - ``src`` layout
        from src.neuronbridge_client import Client as NBClient
    return NBClient(timeout=15)


def refresh(
    datasets: Sequence[str],
    client: Any = None,
    force: bool = False,
    ttl_days: float = DEFAULT_TTL_DAYS,
    cache_root: Optional[os.PathLike] = None,
    nb_version: Optional[str] = None,
) -> CoverageSnapshot:
    """Refresh and persist the coverage snapshot for ``datasets``.

    Per-dataset TTL: verdicts younger than ``ttl_days`` survive without
    re-probing, and a NeuronBridge version change invalidates everything.
    An unreachable NeuronBridge returns the persisted snapshot untouched —
    never a blanket ``unknown`` rewrite.
    """
    requested = [str(dataset) for dataset in datasets or [] if str(dataset)]
    with _LOCK:
        try:
            active_client = _make_client(client)
        except Exception:
            active_client = None

        live_version = nb_version
        if active_client is not None and live_version is None:
            try:
                live_version = str(active_client.version)
            except Exception:
                live_version = None

        existing = load_snapshot(cache_root)
        if live_version is None:
            # Offline: never discard prior knowledge.
            if existing is not None:
                return existing
            return CoverageSnapshot(
                nb_version=None,
                created_at=_now(),
                datasets={
                    dataset: DatasetCoverage(
                        dataset=dataset, status=STATUS_UNKNOWN
                    )
                    for dataset in requested
                },
            )

        merged: Dict[str, DatasetCoverage] = {}
        same_version = existing is not None and existing.nb_version == live_version
        if same_version:
            for dataset, coverage in existing.datasets.items():
                if _age_days(coverage.checked_at) <= ttl_days:
                    merged[dataset] = coverage

        to_probe = [
            dataset
            for dataset in requested
            if force or dataset not in merged
        ]
        if to_probe and active_client is not None:
            samples: Dict[str, List[str]] = {}
            for dataset in to_probe:
                try:
                    samples[dataset] = sample_body_ids(dataset)
                except Exception:
                    samples[dataset] = []
            with ThreadPoolExecutor(max_workers=PROBE_WORKERS) as executor:
                futures = {
                    executor.submit(
                        probe_coverage, dataset, samples.get(dataset, []),
                        active_client,
                    ): dataset
                    for dataset in to_probe
                }
                for future in as_completed(futures):
                    dataset = futures[future]
                    try:
                        merged[dataset] = future.result()
                    except Exception:
                        merged[dataset] = DatasetCoverage(
                            dataset=dataset,
                            status=STATUS_UNKNOWN,
                            checked_at=_now(),
                        )

        snapshot = CoverageSnapshot(
            nb_version=live_version,
            created_at=_now(),
            datasets=merged,
        )
        try:
            _save_snapshot(snapshot, cache_root)
        except OSError:
            pass
        return snapshot


def warnings_for(
    datasets: Sequence[str],
    snapshot: Optional[CoverageSnapshot],
) -> List[str]:
    """User-facing warnings for selected datasets the snapshot cannot serve.

    Advisory only: callers notify and proceed.  ``unknown`` datasets (and
    a missing snapshot, e.g. after a failed refresh) never warn — the
    dataset may well work.
    """
    if snapshot is None:
        return []
    warnings: List[str] = []
    for dataset in datasets or []:
        coverage = snapshot.coverage_of(dataset)
        if coverage.status == STATUS_UNAVAILABLE:
            warnings.append(
                f"{dataset} appears not hosted by NeuronBridge "
                f"(coverage checked {coverage.checked_at or 'at an unknown time'}); "
                "searching it anyway — results may be empty."
            )
    return warnings
