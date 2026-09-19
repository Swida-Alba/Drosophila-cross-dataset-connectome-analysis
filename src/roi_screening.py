"""
ROI-distribution candidate screening for morphological similarity.

Candidate discovery for ``MorphologyComparer`` that ranks every neuron in a
dataset by the similarity of its innervation profile: per-neuron input
(``post``) and output (``pre``) synapse counts over the dataset's PRIMARY
ROIs, compared with cosine similarity and bilateral mirroring (the better of
the same-orientation and L/R-swapped orientations, as in NBLAST). This
complements the connection-cache screen (shared partners), which cannot see
neurons without shared partners at all.

Data sources:

- counts: the long-form ``datasets/{folder}/{folder}_allneurons_roi_count_df.csv``
  ONLY (``bodyId, roi, pre, post``). The per-neuron ``roiInfo`` /
  ``inputRois`` / ``outputRois`` string columns of the neuron table are never
  parsed.
- ROI level: the primary ROI list of the locally prepared
  ``{folder}_metadata.json`` sidecar (written by ``statvis.pull_dataset``).
  Primary ROIs are a disjoint partition of each neuron's synapses, unlike the
  hierarchical labels (``CentralBrain``, ``ME_R_layer_*``, ``LOP_L_col_*``)
  also present in the ROI table, which would double-count. The list is
  validated with a partition sum-check before use; a missing sidecar is
  backfilled once through the same ``statvis._build_dataset_metadata``
  preparation code that a dataset pull uses.

The per-dataset matrices are cached as
``cache/{folder}/morphology/roi_profiles.npz`` (fingerprinted on the source
CSVs and sidecar) so repeated queries cost two matrix-vector products.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROI_PROFILE_CACHE_VERSION = 1


class RoiScreeningUnavailable(ValueError):
    """ROI screening prerequisites are missing (no ROI table / metadata).

    Raised instead of a plain ValueError so ``MorphologyComparer`` can tell
    preparation problems apart from programming errors and fall back to the
    connection-cache screen for auto-resolved runs.
    """


# Partition validation window: the median per-neuron ratio of (synapses
# counted inside the ROI list) to (the neuron's own total) must fall in this
# range for the list to be a disjoint primary partition. Hierarchical lists
# (every ROI label in the table) double-count and land well above it.
PARTITION_RATIO_MIN = 0.95
PARTITION_RATIO_MAX = 1.05

LogFn = Optional[Callable[[str], None]]


def _dataset_folder(dataset: str) -> str:
    """Map a dataset name to its folder (hemibrain:v1.2.1 -> hemibrain_v1_2_1)."""
    return dataset.replace(":", "_").replace(".", "_")


def _project_root(project_root: Optional[str] = None) -> Path:
    return Path(project_root) if project_root else Path(__file__).parent.parent


def dataset_paths(dataset: str,
                  project_root: Optional[str] = None) -> Tuple[Path, Path]:
    """(dataset folder, allneurons path prefix) for a dataset name."""
    root = _project_root(project_root)
    folder = _dataset_folder(dataset)
    dataset_dir = root / "datasets" / folder
    return dataset_dir, dataset_dir / f"{folder}_allneurons"


def roi_count_table_path(dataset: str, project_root: Optional[str] = None) -> Path:
    """Path of the dataset's ROI-count table: parquet when present, else CSV.

    ``pull_dataset`` writes ``_roi_count_df.parquet`` (zstd); CSVs from older
    pulls remain a valid source, so callers resolve whichever exists.
    """
    _, prefix = dataset_paths(dataset, project_root)
    parquet = Path(str(prefix) + "_roi_count_df.parquet")
    if parquet.exists():
        return parquet
    return Path(str(prefix) + "_roi_count_df.csv")


# Backwards-compatible alias for the CSV-era name.
roi_count_csv_path = roi_count_table_path


def read_roi_count_columns(path: Path, columns: Sequence[str]):
    """Read selected columns of an ROI-count table (parquet or CSV)."""
    import polars as pl

    if path.suffix == ".parquet":
        return pl.read_parquet(path, columns=list(columns))
    return pl.read_csv(path, columns=list(columns), infer_schema_length=10000)


def metadata_json_path(dataset: str, project_root: Optional[str] = None) -> Path:
    # Same convention as pull_dataset (statvis): the sidecar drops the
    # "_allneurons" infix -> datasets/{folder}/{folder}_metadata.json.
    dataset_dir, _ = dataset_paths(dataset, project_root)
    return dataset_dir / f"{_dataset_folder(dataset)}_metadata.json"


def load_primary_rois(dataset: str, project_root: Optional[str] = None,
                      log: LogFn = None) -> Optional[List[str]]:
    """Primary ROI list from the locally prepared metadata sidecar, if any."""
    path = metadata_json_path(dataset, project_root)
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text())
    except Exception as exc:
        if log:
            log(f"ROI screening: metadata sidecar unreadable ({exc}).")
        return None
    rois = ((meta.get("roi_coverage") or {}).get("roi_list")) or []
    rois = [str(r) for r in rois if str(r) and str(r) != "NotPrimary"]
    return rois or None


def validate_primary_rois(rois: Sequence[str], dataset: str,
                          project_root: Optional[str] = None,
                          log: LogFn = None) -> bool:
    """Whether ``rois`` partitions each neuron's synapses (no hierarchy).

    For every neuron, the ``pre`` counts summed over the ROI list should
    match the neuron's own ``pre`` total (median ratio ~1.0). A hierarchical
    list counts synapses in parent and child ROIs alike and fails the check.
    """
    import polars as pl

    root = _project_root(project_root)
    rc_path = roi_count_table_path(dataset, str(root))
    nd_path = root / "datasets" / _dataset_folder(dataset) / \
        f"{_dataset_folder(dataset)}_allneurons_neuron_df.csv"
    if not rc_path.exists() or not nd_path.exists():
        return False
    try:
        rc = read_roi_count_columns(rc_path, ("bodyId", "roi", "pre"))
        rc = rc.filter(pl.col("roi").is_in(list(rois)))
        sums = rc.group_by("bodyId").agg(pl.col("pre").sum().alias("sum_pre"))
        nd = pl.read_csv(nd_path, columns=["bodyId", "pre"])
        joined = nd.join(sums, on="bodyId", how="left").fill_null(0)
        if joined.height == 0:
            return False
        ratio = joined["sum_pre"] / joined["pre"].clip(1)
        median = float(ratio.median())
    except Exception as exc:
        if log:
            log(f"ROI screening: partition validation failed ({exc}).")
        return False
    ok = PARTITION_RATIO_MIN <= median <= PARTITION_RATIO_MAX
    if log and not ok:
        log(f"ROI screening: ROI list is not a disjoint partition "
            f"(median coverage {median:.2f}); treating it as unusable.")
    return ok


def backfill_dataset_metadata(dataset: str,
                              project_root: Optional[str] = None,
                              log: LogFn = None) -> Optional[dict]:
    """Fetch and save the metadata sidecar for a locally prepared dataset.

    Datasets prepared before ``pull_dataset`` started writing the sidecar
    have neuron/ROI tables but no ``*_metadata.json``. Instead of re-pulling
    the whole dataset, load the local tables and rerun the same preparation
    code the pull uses (``statvis._build_dataset_metadata``), which takes
    the primary ROI list from the NeuPrint server. Returns the saved
    metadata dict, or None when the server is unreachable.
    """
    root = _project_root(project_root)
    _, prefix = dataset_paths(dataset, str(root))
    neuron_csv = Path(str(prefix) + "_neuron_df.csv")
    roi_table = roi_count_table_path(dataset, str(root))
    if not neuron_csv.exists() or not roi_table.exists():
        return None
    try:
        from statvis import _build_dataset_metadata

        neuron_df = pd.read_csv(
            neuron_csv, usecols=lambda c: c in ("bodyId", "type", "pre", "post")
        )
        if roi_table.suffix == ".parquet":
            roi_count_df = pd.read_parquet(
                roi_table, columns=["bodyId", "roi", "pre", "post"]
            )
        else:
            roi_count_df = pd.read_csv(
                roi_table, usecols=lambda c: c in ("bodyId", "roi", "pre", "post")
            )
        from neuprint import Client
        try:
            from utils.token_manager import token_manager
            token = token_manager.get_token("NEUPRINT_TOKEN")
        except Exception:
            token = ""
        client = Client("neuprint.janelia.org", dataset=dataset, token=token)
        meta = _build_dataset_metadata(dataset, neuron_df, roi_count_df, client)
        meta_path = metadata_json_path(dataset, str(root))
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        if log:
            log(f"ROI screening: metadata sidecar saved to {meta_path}.")
        return meta
    except Exception as exc:
        if log:
            log(f"ROI screening: metadata backfill failed ({exc}).")
        return None


def _flip_roi(name: str) -> str:
    """Mirror an ROI name across the midline (L <-> R; midline ROIs fixed)."""
    if name.endswith("(L)"):
        return name[:-3] + "(R)"
    if name.endswith("(R)"):
        return name[:-3] + "(L)"
    return name


def mirror_permutation(rois: Sequence[str]) -> np.ndarray:
    """Column permutation mirroring the ROI space; -1 where no mirror exists."""
    index = {r: i for i, r in enumerate(rois)}
    return np.array([index.get(_flip_roi(r), -1) for r in rois], dtype=np.int64)


class RoiProfileStore:
    """Per-neuron input/output synapse distributions over the primary ROIs.

    ``ensure()`` loads the fingerprinted npz cache or rebuilds it from the
    ROI-count CSV (one polars pass, a few seconds on male-cns). ``screen()``
    ranks every neuron against one query by mirrored cosine similarity of
    the raw-count distributions.
    """

    def __init__(self, dataset: str, project_root: Optional[str] = None,
                 verbose: bool = False, log: LogFn = None):
        self.dataset = dataset
        self.root = _project_root(project_root)
        folder = _dataset_folder(dataset)
        self.cache_file = self.root / "cache" / folder / "morphology" / \
            "roi_profiles.npz"
        self.verbose = verbose
        self._ext_log = log
        self.bodyIds: Optional[np.ndarray] = None   # int64, sorted
        self.rois: List[str] = []
        self.pre: Optional[np.ndarray] = None       # raw counts, float32
        self.post: Optional[np.ndarray] = None
        self._post_n = self._pre_n = None           # row-normalized
        # Derived index built once per load/build: bodyId -> row position,
        # the midline-mirror column permutation, and the non-zero-candidate
        # mask (zero-vector neurons are excluded by ``screen``).
        self._pos_of: Dict[int, int] = {}
        self._perm: np.ndarray = np.array([], dtype=np.int64)
        self._valid: np.ndarray = np.array([], dtype=bool)

    def _log(self, msg: str):
        if self._ext_log is not None:
            self._ext_log(msg)
        elif self.verbose:
            print(msg)

    # ------------------------------------------------------------ build/load
    def _fingerprint(self, rois: Sequence[str]) -> str:
        def _stat(path: Path):
            try:
                st = path.stat()
                return [st.st_size, int(st.st_mtime)]
            except OSError:
                return None

        payload = {
            "version": ROI_PROFILE_CACHE_VERSION,
            "roi_csv": _stat(roi_count_table_path(self.dataset, str(self.root))),
            "metadata": _stat(metadata_json_path(self.dataset, str(self.root))),
            "rois": hashlib.sha256(
                "\n".join(sorted(rois)).encode("utf-8")
            ).hexdigest(),
        }
        return json.dumps(payload, sort_keys=True)

    def load(self) -> bool:
        """Populate from the npz cache when its fingerprint still matches."""
        if not self.cache_file.exists():
            return False
        try:
            with np.load(self.cache_file, allow_pickle=False) as data:
                fingerprint = str(data["fingerprint"])
                rois = [str(r) for r in data["rois"].tolist()]
                if fingerprint != self._fingerprint(rois):
                    self._log("ROI profiles cache outdated; rebuilding.")
                    return False
                self.bodyIds = data["bodyIds"].astype(np.int64)
                self.pre = data["pre"].astype(np.float32)
                self.post = data["post"].astype(np.float32)
            self.rois = rois
            self._rebuild_index()
            return True
        except Exception as exc:
            self._log(f"ROI profiles cache unreadable ({exc}); rebuilding.")
            return False

    def _resolve_primary_rois(self) -> List[str]:
        rois = load_primary_rois(self.dataset, str(self.root), log=self._log)
        if rois is None:
            self._log("ROI screening: no local metadata sidecar; fetching it "
                      "once (dataset preparation backfill).")
            meta = backfill_dataset_metadata(self.dataset, str(self.root),
                                             log=self._log)
            if meta is not None:
                rois = load_primary_rois(self.dataset, str(self.root))
        if not rois:
            raise RoiScreeningUnavailable(
                f"No primary ROI list for {self.dataset}. Prepare the dataset "
                "(Settings: pull/prepare dataset) so "
                f"{metadata_json_path(self.dataset, str(self.root)).name} "
                "exists, or use Candidate Source 'profile'/'cache'."
            )
        if not validate_primary_rois(rois, self.dataset, str(self.root),
                                     log=self._log):
            # A sidecar written without a server client falls back to every
            # ROI label (hierarchical, double-counting). Refetching the real
            # primary list fixes it; give up cleanly when offline.
            self._log("ROI screening: local ROI list failed the partition "
                      "check; refetching metadata once.")
            meta = backfill_dataset_metadata(self.dataset, str(self.root),
                                             log=self._log)
            if meta is not None:
                rois = load_primary_rois(self.dataset, str(self.root)) or rois
            if not validate_primary_rois(rois, self.dataset, str(self.root)):
                raise RoiScreeningUnavailable(
                    f"The primary ROI list of {self.dataset} is not a disjoint "
                    "partition (hierarchical labels double-count synapses). "
                    "Re-prepare the dataset metadata or use Candidate Source "
                    "'profile'/'cache'."
                )
        return rois

    def build(self) -> "RoiProfileStore":
        """Build the matrices from the ROI-count table and cache them."""
        import polars as pl

        table = roi_count_table_path(self.dataset, str(self.root))
        if not table.exists():
            try:
                from .flywire_ids import is_local_connectome_dataset
            except ImportError:  # pragma: no cover - direct module import
                from flywire_ids import is_local_connectome_dataset
            if is_local_connectome_dataset(self.dataset):
                # FAFB/BANC local releases have no per-ROI synapse count table at
                # all (the ROI screen is NeuPrint-only); "pull/prepare" can
                # never produce one, so the guidance must say so.
                raise RoiScreeningUnavailable(
                    f"ROI screening is not available for {self.dataset}: "
                    "FAFB/BANC local releases have no per-ROI synapse count "
                    "table. "
                    "Use Candidate Source 'profile' (shared connectivity "
                    "partners) or 'cache' (full vector-cache search) instead."
                )
            raise RoiScreeningUnavailable(
                f"No ROI-count table for {self.dataset} (expected {table}). "
                "Pull/prepare the dataset first, or use Candidate Source "
                "'profile'/'cache'."
            )
        rois = self._resolve_primary_rois()

        rc = read_roi_count_columns(table, ("bodyId", "roi", "pre", "post"))
        present = set(rc["roi"].unique().to_list())
        kept = [r for r in rois if r in present]
        dropped = len(rois) - len(kept)
        if dropped:
            self._log(f"ROI screening: {dropped} primary ROIs absent from the "
                      "ROI table (kept as zero columns).")
        if not kept:
            raise RoiScreeningUnavailable(
                f"None of the primary ROIs of {self.dataset} appear in {table}."
            )

        body_ids = np.sort(rc["bodyId"].unique().to_numpy()).astype(np.int64)
        row_of = pl.DataFrame({"bodyId": body_ids}).with_row_index("row")
        rc = (rc.join(row_of, on="bodyId")
                .filter(pl.col("roi").is_in(kept))
                .with_columns(pl.col("bodyId").cast(pl.Int64)))
        col_of = {r: k for k, r in enumerate(kept)}
        cols = rc["roi"].replace_strict(col_of).to_numpy().astype(np.int64)
        rows = rc["row"].to_numpy().astype(np.int64)

        n, r = len(body_ids), len(kept)
        pre = np.zeros((n, r), dtype=np.float32)
        post = np.zeros((n, r), dtype=np.float32)
        pre[rows, cols] = rc["pre"].to_numpy().astype(np.float32)
        post[rows, cols] = rc["post"].to_numpy().astype(np.float32)

        self.bodyIds, self.rois, self.pre, self.post = body_ids, kept, pre, post
        self._post_n = self._pre_n = None
        self._rebuild_index()

        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_file.with_suffix(".npz.tmp")
        with open(tmp, "wb") as handle:
            # File object: np.savez_compressed would otherwise append ".npz"
            # to the temp name and break the atomic replace below.
            np.savez_compressed(
                handle, bodyIds=body_ids, rois=np.array(kept),
                pre=pre, post=post,
                fingerprint=np.array(self._fingerprint(kept)),
            )
        tmp.replace(self.cache_file)
        self._log(f"ROI screening: built {n} neurons x {r} primary ROIs "
                  f"(cache: {self.cache_file}).")
        return self

    def ensure(self) -> "RoiProfileStore":
        return self if self.load() else self.build()

    # -------------------------------------------------------------- screening
    def _rebuild_index(self):
        """Derive the row index, mirror permutation and valid-candidate mask.

        Runs once per load/build. The valid mask replaces the per-call
        full-matrix norm reductions the previous ``screen`` implementation
        recomputed for every query.
        """
        self._pos_of = {int(b): i for i, b in enumerate(self.bodyIds.tolist())}
        self._perm = mirror_permutation(self.rois)
        self._valid = (
            (self.pre * self.pre).sum(axis=1)
            + (self.post * self.post).sum(axis=1)
        ) > 0

    def _ensure_normalized(self):
        if self._post_n is not None:
            return

        def _norm(M):
            return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True),
                                  1e-9).astype(np.float32)

        self._post_n = _norm(self.post)
        self._pre_n = _norm(self.pre)
        # Raw count matrices are only needed for normalization and the valid
        # mask (both derived above); drop them so male-cns-scale stores do
        # not hold two unused n x r float32 copies.
        self.pre = None
        self.post = None

    def _mirrored_query(self, q: np.ndarray) -> np.ndarray:
        """The query vector reindexed so ``M @ q_m`` scores the mirrored
        orientation of the candidate distributions.

        Mirroring the query instead of materializing mirrored copies of the
        full candidate matrices halves the store's memory footprint: for
        ``M @ q_m``, ``q_m[k]`` carries the query mass of every ROI whose
        mirror column is ``k`` (midline ROIs map to themselves; ROIs without
        a mirror contribute nothing).
        """
        q_m = np.zeros_like(q)
        valid = self._perm >= 0
        q_m[self._perm[valid]] = q[valid]
        return q_m

    def screen(self, query_body_ids: Sequence[int],
               top_k: Optional[int] = None) -> pd.DataFrame:
        """Rank all neurons against the query by mirrored ROI cosine.

        The query distribution is the renormalized mean of its members'
        row-normalized input/output vectors (only blocks the query actually
        has). Each candidate scores the mean block cosine, taking the better
        of the same-orientation and mirrored orientations; zero-vector
        candidates and query members are excluded. Returns a DataFrame
        [bodyId, roi_similarity] sorted by similarity descending (ties by
        ascending bodyId). ``top_k`` bounds the result to the best K rows
        (partial-selection fast path for pool truncation).
        """
        if self.bodyIds is None:
            raise RuntimeError("RoiProfileStore not loaded/built.")
        self._ensure_normalized()

        query_rows = []
        for bid in query_body_ids:
            row = self._pos_of.get(int(bid))
            if row is None:
                self._log(f"ROI screening: query bodyId {bid} not in the ROI "
                          "table; skipped.")
            else:
                query_rows.append(row)
        if not query_rows:
            return pd.DataFrame({"bodyId": [], "roi_similarity": []})

        # One pass over both synapse blocks; the mirrored orientation is the
        # same matrix-vector product with the permuted query (no mirrored
        # candidate matrices are ever materialized).
        same = np.zeros(len(self.bodyIds), dtype=np.float32)
        flipped = np.zeros(len(self.bodyIds), dtype=np.float32)
        n_blocks = 0
        for cand in (self._post_n, self._pre_n):
            q = cand[query_rows].mean(axis=0)
            norm = float(np.linalg.norm(q))
            if norm <= 1e-9:
                continue   # query has no arbor in this block
            q = (q / norm).astype(np.float32)
            same += cand @ q
            flipped += cand @ self._mirrored_query(q)
            n_blocks += 1
        if not n_blocks:
            return pd.DataFrame({"bodyId": [], "roi_similarity": []})
        scores = np.maximum(same, flipped) / n_blocks

        scores[query_rows] = -np.inf   # query members are never candidates
        scores[~self._valid] = np.nan  # zero-synapse neurons are unrankable

        keep = np.isfinite(scores)
        idx = np.flatnonzero(keep)
        if not len(idx):
            return pd.DataFrame({"bodyId": [], "roi_similarity": []})
        if top_k is not None and 0 < top_k < len(idx):
            # Partial selection: only the best K rows are ordered (float
            # ties at the boundary are a measure-zero event; the full path
            # below keeps the deterministic bodyId-ascending tie-break).
            part = np.argpartition(-scores[keep], top_k - 1)[:top_k]
            order = idx[part[np.argsort(-scores[keep][part], kind="stable")]]
        else:
            order = idx[np.argsort(-scores[keep], kind="stable")]
        kept_ids = self.bodyIds[order]
        return pd.DataFrame({
            "bodyId": kept_ids.astype(np.int64),
            "roi_similarity": scores[order].astype(float),
        })

    def expansion_rows(self, body_ids: Sequence[int],
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Hellinger pre/post expansion rows for specific bodyIds.

        Returns ``(found_ids, block)`` where ``block[i]`` is
        ``sqrt([pre_fractions, post_fractions])`` (2R dims) of
        ``found_ids[i]``. Composed from the cached npz on demand — the raw
        count matrices are dropped from memory after normalization — so the
        screen's own state is never disturbed. Used by the V2 morphology
        comparison as its optional ROI-expansion block; datasets without a
        store (FAFB/FlyWire) simply omit the block.
        """
        if self.bodyIds is None or self.cache_file is None \
                or not Path(self.cache_file).exists():
            return np.array([], dtype=np.int64), np.zeros((0, 0))
        wanted = [int(b) for b in body_ids]
        pos = {int(b): i for i, b in enumerate(self.bodyIds.tolist())}
        rows = [pos[b] for b in wanted if b in pos]
        found = [b for b in wanted if b in pos]
        if not rows:
            return np.array([], dtype=np.int64), np.zeros((0, 0))
        try:
            with np.load(self.cache_file, allow_pickle=False) as data:
                pre = data["pre"][rows].astype(np.float64)
                post = data["post"][rows].astype(np.float64)
        except Exception:
            return np.array([], dtype=np.int64), np.zeros((0, 0))
        for mat in (pre, post):
            totals = mat.sum(axis=1)
            mat /= np.maximum(totals, 1.0)[:, None]
            mat[totals <= 0, :] = 0.0
        block = np.sqrt(np.hstack([pre, post]))
        return np.asarray(found, dtype=np.int64), block
