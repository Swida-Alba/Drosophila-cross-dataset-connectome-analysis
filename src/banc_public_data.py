"""Public BANC release data access (GCS bucket, no authentication).

BANC is analyzed from its own public release products instead of through
FlyWire/Codex or the token-gated CAVE ``brain_and_nerve_cord`` datastack:

- **Skeletons** are SWC files in nanometres (BANC space), one file per
  neuron under ``compiled_data/banc_888/banc_banc_space_swc/``:
  ``{id}_skeleton.swc`` (full resolution, proofread neurons) or
  ``{id}_l2.swc`` (coarse L2 approximation).  Every file stem is an
  888-namespace id; the resolution lives only in the suffix, so a v626
  dataset id resolves through the meta-feather crosswalk
  (``root_626`` -> ``banc_888_id``).
- **Metadata / connections** are release products that replace the manual
  Codex downloads (see :func:`prepare_dataset`).

All fetched skeletons are cached in the shared raw skeleton store
(``cache/{dataset}/skeletons/raw_skeletons/{bodyId}.swc.zst``) at raw level
(simplification 0), identical to the FAFB raw-cache representation.
"""


from __future__ import annotations

# Round-7 report: this module prints non-ASCII status glyphs on paths that
# never import coana (its own __main__, converter CLIs), so install the
# UTF-8 stdio guard at the source — idempotent, never raises.
try:
    from utils.console_encoding import ensure_utf8_stdio
except ImportError:
    try:
        from .utils.console_encoding import ensure_utf8_stdio
    except ImportError:
        ensure_utf8_stdio = None
if ensure_utf8_stdio is not None:
    ensure_utf8_stdio()

import io
import os
import time
import http.client
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

try:
    from .flywire_ids import is_banc_dataset
except ImportError:  # pragma: no cover - src laid bare on sys.path
    from flywire_ids import is_banc_dataset

try:
    from .utils.naming_utils import canonical_dataset_name, dataset_version
except ImportError:  # pragma: no cover - src laid bare on sys.path
    from utils.naming_utils import canonical_dataset_name, dataset_version

try:
    from .utils.parquet_utils import (
        parquet_is_reusable, parquet_readable, reencode_parquet_lossless,
        write_parquet_atomic)
except ImportError:  # pragma: no cover - src laid bare on sys.path
    from utils.parquet_utils import (
        parquet_is_reusable, parquet_readable, reencode_parquet_lossless,
        write_parquet_atomic)

# Shared compressed-SWC provenance contract (headers, parser, writer).
try:
    from . import skeleton_provenance as _provenance
except ImportError:  # pragma: no cover - src laid bare on sys.path
    import skeleton_provenance as _provenance


# ---------------------------------------------------------------------------
# Public bucket locations (verified live; plain HTTPS, no token anywhere)
# ---------------------------------------------------------------------------

BUCKET_HTTP_BASE = (
    "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome"
)

# One SWC per neuron; nm coordinates; 888-namespace stems.
SWC_DIR = "compiled_data/banc_888/banc_banc_space_swc"

# All-neuron L2 set in MICROMETRES with v626-namespace stems.  Last-resort
# fallback only: values are scaled x1000 to nm and the cache entry records
# the coarser provenance.
PCG_SKEL_DIR = "neuron_skeletons/swcs-from-pcg-skel"

# Release metadata (188,508 rows): ids across versions, curated types,
# proofread flags, neurotransmitters, soma positions.
META_FEATHER_PATH = "compiled_data/banc_888/banc_888_meta.feather"

# Per-release connection counts (synapse size >= 3, count >= 3 thresholds are
# baked into the product; the Codex download matches this closely).
CONNECTION_PRODUCTS = {
    "v626": ("neuron_connectivity/v626/"
             "synapses_v1_human_readable_sizethresh3_"
             "connectioncountsperneuropil_countthresh3.parquet"),
    "v888": ("neuron_connectivity/v888/"
             "synapses_v1_human_readable_sizethresh3_"
             "connectioncountsperneuropil_countthresh3.parquet"),
}
DEFAULT_CONNECTION_VERSION = "v888"

# Per-synapse tables (nanometre pre-synaptic site coordinates; the release
# publishes only pre-site positions — no post-site columns).  One-time
# download (≈3.9 GB) per release, then locally filtered per query.
SYNAPSE_TABLES = {
    "v626": ("neuron_connectivity/v626/synapses_v1_human_readable_"
             "id_size_prerootid_postrootid_prex_prey_prez_neuropil.parquet"),
    "v888": ("neuron_connectivity/v888/synapses_v1_human_readable_"
             "id_size_prerootid_postrootid_prex_prey_prez_neuropil.parquet"),
}

# CloudVolume precomputed layers (public; vertices already in nanometres).
NEURON_MESHES_URL = (
    "precomputed://" + BUCKET_HTTP_BASE + "/neuron_meshes"
)
REGION_OUTLINES_URL = (
    "precomputed://" + BUCKET_HTTP_BASE + "/region_outlines"
)

# Brain/VNC coordinate segmentation of the whole-CNS outline.  Measured
# from the outline's cross-section profile along the AP axis: brain lobes
# span y ≈ 60–240k, the thin cervical connective (neck) y ≈ 360–500k
# (deepest waist 470k), VNC somas start ≈ 538k.  The cut goes at the
# brain/neck boundary so the neck stays with the VNC:
BANC_BRAIN_VNC_Y_CUTOFF = 350_000

# Aggregated outline segments in the region_outlines layer (nm coordinates).
# (The rendered tube radius target lives in the renderer/UI: 240 nm default
# via banc_radius_target_nm.)
REGION_AGGREGATES = {
    1: "BANC_outline",
    2: "BANC_neuropil",
    3: "BANC_brain_neuropil",
    4: "BANC_vnc_neuropil",
}

# Skeleton resolution preference.  One unified chain for every caller:
# 888 L2 first (the cache-level product), then the 888 full-resolution
# skeleton, then the v626-era pcg-skel set.  ``DEFAULT_RESOLUTION`` and
# ``_VALID_RESOLUTIONS`` only survive for the deprecated ``resolution``
# argument of :func:`fetch_banc_swc` (accepted, validated, ignored).
DEFAULT_RESOLUTION = "l2"
_VALID_RESOLUTIONS = ("l2", "full", "full_auto")
_SKELETON_SUFFIXES = ("_l2.swc", "_skeleton.swc")

# Provenance line written into cached SWC headers (next to the level line).
_SOURCE_HEADER = _provenance.SOURCE_HEADER

# Dataset folders whose crosswalk download failed this process (negative
# cache: do not re-attempt the 58 MB download per neuron).
_CROSSWALK_FAILURES: set = set()

# Process-lifetime crosswalk memo ({(dataset folder, cache path) -> dict}):
# a successful mapping is read from disk once per process, not once per
# neuron.
_CROSSWALK_CACHE: dict = {}

# Memoized 888 -> v626 inverse (same key shape).  The pcg-skeleton
# fallback resolves through it per neuron; re-deriving the O(185k) dict
# on every call is pure churn (issue-report BANC-08).
_CROSSWALK_REVERSE_CACHE: dict = {}


def _project_root(project_root) -> Path:
    if project_root is not None:
        return Path(project_root)
    return Path(__file__).resolve().parent.parent


def _canonical_banc_name(dataset) -> str:
    """Return the canonical spelling used by every BANC release operation."""
    return canonical_dataset_name(str(dataset or "").strip())


def _dataset_folder(dataset) -> str:
    return _canonical_banc_name(dataset).replace(":", "_").replace(".", "_")


def _connection_version(dataset) -> str:
    # ``banc`` and ``flywire_BANC`` are pinned by the shared canonicalizer to
    # the v626 release.  Extract the version AFTER that normalization or the
    # aliases silently fall through to the v888 default.
    version = (dataset_version(_canonical_banc_name(dataset)) or "").lower()
    if version in CONNECTION_PRODUCTS:
        return version
    return DEFAULT_CONNECTION_VERSION


def reject_unknown_banc_release(dataset) -> None:
    """Refuse an explicit-but-unsupported BANC release (round-7 R7-1a).

    ``banc`` / ``flywire_BANC`` legitimately resolve to a pinned default
    release, and ``banc_v626`` / ``banc_v888`` are served directly — but an
    identifier carrying an explicit version the bucket does not publish
    (e.g. ``banc_v999``) used to fall through to the default release and
    download it under the requested name: v888 data labelled v999. Raise
    instead; the caller's refusal path takes over.
    """
    version = (dataset_version(str(dataset or "").strip()) or "").lower()
    if version and version not in CONNECTION_PRODUCTS:
        raise ValueError(
            f"BANC release {version!r} (from {dataset!r}) is not published "
            f"by the public bucket — supported releases: "
            f"{sorted(CONNECTION_PRODUCTS)}. Refusing to download a "
            "different release and label it with this dataset name.")


def _is_v626_release(dataset) -> bool:
    """Return whether *dataset* addresses the v626 BANC id namespace."""
    return _connection_version(dataset) == "v626"


# ---------------------------------------------------------------------------
# HTTP helper (the public JSON/API endpoints drop connections occasionally;
# every fetch retries before giving up)
# ---------------------------------------------------------------------------

def http_get(url: str, timeout: float = 120, attempts: int = 3,
             note: str = "") -> Optional[bytes]:
    """GET *url* with retries and byte-range RESUME.  Returns None on HTTP
    404, bytes on success.

    Round-6 finding F-N1: every retry used to restart from byte 0, so one
    lossy link exhausted the attempts mid-file and aborted BANC
    preparation — resume only worked BETWEEN files.  The partial body of a
    dropped read is now kept and the next attempt continues it with a
    ``Range: bytes=<n>-`` request; a server that answers 200 instead of
    206 (range ignored) restarts cleanly from zero.
    """
    last_error = None
    received = bytearray()
    for attempt in range(attempts):
        try:
            headers = {"User-Agent": "DROCAT/banc-public-data"}
            if received:
                headers["Range"] = f"bytes={len(received)}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, 'status', 200)
                if received and status == 206:
                    received.extend(response.read())  # continuation
                else:
                    received = bytearray(response.read())  # full restart
                return bytes(received)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last_error = exc
        except (urllib.error.URLError, http.client.HTTPException,
                OSError, ConnectionError) as exc:
            # http.client.HTTPException covers IncompleteRead: a connection
            # dropped mid-body used to escape the retry loop entirely and
            # abort BANC preparation after tens of MB (re-test finding F3,
            # 2026-09-16).  The partial bytes read so far survive in
            # ``received`` and the next attempt resumes from them.
            # http.client.IncompleteRead.partial is the BYTES read so far
            # (not a file object) — salvage them so the retry resumes.
            partial = getattr(exc, 'partial', None)
            if isinstance(partial, (bytes, bytearray)):
                received.extend(partial)
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(
        f"Failed to download {note or url} after {attempts} attempts: "
        f"{last_error}")


# ---------------------------------------------------------------------------
# SWC skeleton fetch
# ---------------------------------------------------------------------------

def _skeleton_cache_path(dataset, body_id, project_root) -> Path:
    return _provenance.raw_skeleton_cache_path(
        _project_root(project_root), _dataset_folder(dataset), body_id)


def _resolution_from_source_header(content: str) -> str:
    """Bucket resolution recorded in a cached SWC's provenance header.

    ``# DROCAT source: banc_gcs_full`` -> 'full'; every other provenance
    (banc_gcs_l2, banc_gcs_pcg_um_x1000, headerless legacy entries) reads
    as the coarse 'l2'.  navis drops comment lines, so the resolution must
    be rebuilt from the raw text on every cache load (R2).
    """
    return _provenance.resolution_from_source(
        _provenance.read_stored_source(content))


def _load_cached_swc(cache_path: Path):
    """Load a cached TreeNeuron (units nm, id set) or None."""
    import navis

    if not cache_path.exists():
        return None
    try:
        import zstandard as zstd

        with open(cache_path, "rb") as handle:
            with zstd.ZstdDecompressor().stream_reader(handle) as reader:
                content = reader.read().decode("utf-8", "replace")
        neuron = navis.read_swc(io.StringIO(content))
        if neuron is None:
            return None
        neuron.id = int(cache_path.name.replace(".swc.zst", "")
                        .replace(".swc.gz", ""))
        neuron.units = "nm"
        # R2: restore the resolution provenance so cache hits honour the
        # requested resolution instead of defaulting to 'l2'.
        neuron._drocat_banc_resolution = _resolution_from_source_header(
            content)
        return neuron
    except Exception as exc:  # a corrupt entry behaves like a cache miss
        print(f"  [banc] Ignoring unreadable cache entry "
              f"{cache_path.name}: {exc}")
        return None


def _write_cached_swc(cache_path: Path, swc_text: bytes, source: str) -> None:
    """Persist raw SWC bytes (level 0) with a provenance header line."""
    payload = _provenance.make_source_line(source) + swc_text
    _provenance.write_compressed_swc_zst(str(cache_path), payload,
                                         simplification=0)


def _crosswalk_cache_path(dataset, project_root) -> Path:
    return (_project_root(project_root) / "cache"
            / _dataset_folder(dataset) / "banc_id_crosswalk.parquet")


def _crosswalk_memo_key(dataset, project_root) -> tuple:
    """Memo key binding the dataset folder to its concrete cache path."""
    return (_dataset_folder(dataset),
            os.path.abspath(str(_crosswalk_cache_path(dataset, project_root))))


def get_id_crosswalk(dataset, project_root=None, force_refresh: bool = False):
    """Return the v626 -> 888 id map (dict), downloading/caching it once.

    The crosswalk is derived from the release meta feather
    (``root_626`` -> ``banc_888_id``) and cached as a small two-column
    parquet under ``cache/{dataset}/``.  A successful mapping is memoized
    for the process lifetime, so per-neuron callers do not re-read the
    parquet.  Returns an empty dict on failure so callers fall back to
    direct ids.
    """
    import pandas as pd

    key = _crosswalk_memo_key(dataset, project_root)
    if force_refresh:
        _CROSSWALK_CACHE.pop(key, None)
        _CROSSWALK_REVERSE_CACHE.pop(key, None)
    elif key in _CROSSWALK_CACHE:
        # Copy: the memo is process-wide, so handing out the live dict
        # would let any future caller mutate the shared mapping
        # (issue-report BANC-07).
        return dict(_CROSSWALK_CACHE[key])

    cache_path = _crosswalk_cache_path(dataset, project_root)
    if cache_path.exists() and not force_refresh:
        try:
            frame = pd.read_parquet(cache_path)
            # A readable cache supersedes any earlier download failure.
            _CROSSWALK_FAILURES.discard(key[0])
            _CROSSWALK_CACHE[key] = dict(
                zip(frame["root_626"], frame["banc_888_id"]))
            return dict(_CROSSWALK_CACHE[key])
        except Exception as exc:
            print(f"  [banc] Ignoring unreadable crosswalk cache: {exc}")

    if key[0] in _CROSSWALK_FAILURES and not force_refresh:
        return {}

    url = f"{BUCKET_HTTP_BASE}/{META_FEATHER_PATH}"
    print(f"  [banc] Downloading release metadata for the id crosswalk "
          f"(~58 MB, one-time)...")
    try:
        blob = http_get(url, timeout=300, attempts=3,
                        note="banc_888_meta.feather")
        if blob is None:
            raise RuntimeError("meta feather not found on the bucket")
        frame = pd.read_feather(io.BytesIO(blob))
    except Exception as exc:
        # Negative-cache the failure so per-neuron fetches do not retry the
        # 58 MB download over and over; the raw id fallback applies.
        _CROSSWALK_FAILURES.add(key[0])
        print(f"  [banc] Crosswalk unavailable ({exc}); using direct ids.")
        return {}
    crosswalk = frame[["root_626", "banc_888_id"]].dropna().drop_duplicates(
        subset=["root_626"])
    crosswalk.columns = ["root_626", "banc_888_id"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_parquet(cache_path, index=False)
    mapping = dict(zip(crosswalk["root_626"], crosswalk["banc_888_id"]))
    # A successful download clears any earlier negative cache (a forced
    # refresh after a transient failure must stick).
    _CROSSWALK_FAILURES.discard(key[0])
    _CROSSWALK_CACHE[key] = mapping
    return mapping


def get_id_crosswalk_reverse(dataset, project_root=None):
    """Return the 888 -> v626 id map (dict), from the same cached crosswalk.

    Needed by the pcg-skel fallback: those micrometre skeletons are named
    with v626-era ids, so a v888 dataset body id must be translated before
    probing that set.  Memoized like the forward map — the fallback calls
    this per neuron, and re-deriving the O(185k) inverse every time was
    pure churn (issue-report BANC-08).
    """
    key = _crosswalk_memo_key(dataset, project_root)
    if key in _CROSSWALK_REVERSE_CACHE:
        return dict(_CROSSWALK_REVERSE_CACHE[key])
    forward = get_id_crosswalk(dataset, project_root=project_root)
    if not forward:
        return {}
    reverse = {v888: v626 for v626, v888 in forward.items()}
    _CROSSWALK_REVERSE_CACHE[key] = reverse
    return reverse


def resolve_banc_stem(dataset, body_id, project_root=None) -> str:
    """Map a dataset body id to its 888-namespace SWC file stem.

    v888-dataset ids already are 888-namespace ids; v626-dataset ids resolve
    through the meta crosswalk (falling back to the raw id when the
    crosswalk is unavailable).
    """
    body_id = str(body_id).strip()
    if not _is_v626_release(dataset):
        return body_id
    crosswalk = get_id_crosswalk(dataset, project_root=project_root)
    return crosswalk.get(body_id, body_id)


def fetch_banc_swc(dataset, body_id, resolution: str = DEFAULT_RESOLUTION,
                   project_root=None, use_cache: bool = True):
    """Fetch one BANC neuron as a navis TreeNeuron (nanometres).

    One unified chain for every caller: 888-namespace L2 first (the
    cache-level product), then the 888 full-resolution skeleton, then the
    v626-era micrometre pcg-skel set (scaled to nm via the reverse
    crosswalk).  ``resolution`` is accepted for backward compatibility,
    validated, and IGNORED — there is no per-call source selection anymore.
    Returns None when no skeleton exists for the neuron (true gaps are
    ~1.7% of the release, tiny fragments).
    """
    if not is_banc_dataset(dataset):
        raise ValueError(f"fetch_banc_swc requires a BANC dataset, "
                         f"got {dataset!r}")
    if resolution not in _VALID_RESOLUTIONS:
        raise ValueError(f"Invalid BANC skeleton resolution {resolution!r}; "
                         f"expected one of {_VALID_RESOLUTIONS}")

    import navis

    body_id = str(body_id).strip()
    cache_path = _skeleton_cache_path(dataset, body_id, project_root)
    if use_cache:
        cached = _load_cached_swc(cache_path)
        if cached is not None:
            # Any cached entry satisfies the unified chain: whatever it
            # holds is what the bucket served for this neuron before.
            return cached

    stem = resolve_banc_stem(dataset, body_id, project_root=project_root)

    for suffix in _SKELETON_SUFFIXES:
        url = f"{BUCKET_HTTP_BASE}/{SWC_DIR}/{stem}{suffix}"
        blob = http_get(url, note=f"{stem}{suffix}")
        if blob is None:
            continue
        resolution_name = 'l2' if suffix == '_l2.swc' else 'full'
        try:
            # Parse before caching: a malformed download must never enter
            # the cache (R5).
            neuron = navis.read_swc(io.StringIO(blob.decode("utf-8", "replace")))
        except Exception as exc:
            print(f"  [banc] Unreadable SWC for {stem}{suffix}: {exc}")
            continue
        neuron.id = int(body_id)
        neuron.units = "nm"
        neuron._drocat_banc_resolution = resolution_name
        if use_cache:
            _write_cached_swc(cache_path, blob,
                              f"banc_gcs_{resolution_name}")
        return neuron

    # Last resort: the micrometre pcg-skel set (v626-named ids).  Covers
    # neurons whose compiled_data export is missing entirely — including
    # v888 dataset ids via the reverse crosswalk.
    pcg_candidates = []
    if _is_v626_release(dataset):
        pcg_candidates.append(body_id)
    else:
        reverse = get_id_crosswalk_reverse(dataset, project_root)
        twin = reverse.get(body_id)
        if twin:
            pcg_candidates.append(twin)
    for pcg_id in pcg_candidates:
        url = f"{BUCKET_HTTP_BASE}/{PCG_SKEL_DIR}/{pcg_id}.swc"
        blob = http_get(url, note=f"pcg-skel {pcg_id}.swc")
        if blob is None:
            continue
        try:
            neuron = navis.read_swc(
                io.StringIO(blob.decode("ascii", "replace")))
        except Exception as exc:
            print(f"  [banc] Unreadable pcg-skel SWC for {pcg_id}: {exc}")
            continue
        # The pcg-skel product is in micrometres: scale to the nm
        # convention every other DROCAT skeleton uses.
        for axis in ("x", "y", "z"):
            neuron.nodes[axis] = neuron.nodes[axis] * 1000.0
        neuron.id = int(body_id)
        neuron.units = "nm"
        neuron._drocat_banc_resolution = "l2"
        if use_cache:
            # R1: persist the SCALED coordinates.  Writing the raw
            # micrometre download would poison the nm cache on reload.
            _write_cached_swc(
                cache_path, _swc_text_of(neuron), "banc_gcs_pcg_um_x1000")
        return neuron

    return None


def _swc_text_of(neuron) -> bytes:
    """Serialize a TreeNeuron to SWC bytes (navis dialect)."""
    import tempfile

    import navis

    with tempfile.NamedTemporaryFile(suffix=".swc", delete=False) as handle:
        temp_path = handle.name
    try:
        navis.write_swc(neuron, temp_path, write_meta=False)
        with open(temp_path, "rb") as handle:
            return handle.read()
    finally:
        os.unlink(temp_path)


# ---------------------------------------------------------------------------
# Release metadata / connections (replaces the manual Codex downloads)
# ---------------------------------------------------------------------------

_ALT_TYPE_COLUMNS = (
    "cell_type",
    "fafb_cell_type",
    "fafb_alignment_cell_type",
    "manc_cell_type",
    "malecns_cell_type",
    "hemibrain_cell_type",
    "fanc_cell_type",
)


def _join_alternative_types(row) -> str:
    """Comma-joined deduped alternative types for one meta-feather row.

    ``cell_type`` is the neuron's current curated name (97.6% identical to
    the Codex ``Alternative Cell Type(s)`` passthrough on v888); the
    per-dataset cross-fly columns round out the searchable aliases.  The
    mapper's ``Alternative Cell Type(s)`` column must keep its exact header,
    and keeping ``cell_type`` as the first token mirrors the Codex data.
    """
    tokens = []
    for column in _ALT_TYPE_COLUMNS:
        value = row.get(column)
        if value is None or (isinstance(value, float) and value != value):
            continue
        text = str(value).strip()
        if not text or text.lower().startswith("auto:") or text.lower() == "nan":
            continue
        if text not in tokens:
            tokens.append(text)
    return ",".join(tokens)


def _remote_size(url: str) -> Optional[int]:
    """Content-Length of *url* via HEAD, or None."""
    import urllib.error

    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return int(response.headers["Content-Length"])
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    # URLError / OSError propagate: callers retry with backoff.


def _fetch_range(url: str, start: int, end: int) -> bytes:
    """Bytes of *url* in [start, end] inclusive (HTTP Range)."""
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{end}",
                      "User-Agent": "DROCAT/banc-public-data"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()


def fetch_range_to_file(url: str, dest: Path,
                        progress_callback=None,
                        chunk_bytes: int = 8 * 1024 * 1024,
                        attempts: int = 12,
                        force: bool = False) -> Optional[Path]:
    """Download *url* into *dest*, resuming from any partial local file.

    Uses HTTP Range requests (the public bucket supports them), so an
    interrupted multi-gigabyte download continues where it stopped.  The
    file is considered complete when its size matches the remote
    Content-Length; the result is verified before returning.
    """
    last_error = None
    for attempt in range(attempts):
        try:
            total = _remote_size(url)
            if total is None:
                return None
            have = 0 if force else (
                dest.stat().st_size if dest.exists() else 0)
            if have > total:
                dest.unlink()
                have = 0
            if have == total:
                return dest
            dest.parent.mkdir(parents=True, exist_ok=True)
            mode = "ab" if have else "wb"
            with open(dest, mode) as out:
                pos = have
                while pos < total:
                    end = min(pos + chunk_bytes, total) - 1
                    chunk = _fetch_range(url, pos, end)
                    out.write(chunk)
                    pos += len(chunk)
                    if progress_callback is not None:
                        progress_callback(pos, total)
            if pos == total:
                return dest
            last_error = RuntimeError(
                f"incomplete download: {have}/{total} bytes")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last_error = exc
        except (urllib.error.URLError, http.client.HTTPException,
                OSError, ConnectionError) as exc:
            # http.client.HTTPException covers IncompleteRead: the bytes
            # already written stay on disk and the next attempt resumes.
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(min(30.0, 3.0 * (attempt + 1)))
    raise RuntimeError(
        f"Failed to download {dest.name} after {attempts} attempts: "
        f"{last_error}")


def synapse_table_path(dataset, project_root=None) -> Path:
    """Local path of the downloaded per-synapse parquet for *dataset*."""
    version = _connection_version(dataset)
    product = SYNAPSE_TABLES[version].rsplit("/", 1)[-1]
    return (_project_root(project_root) / "datasets"
            / _dataset_folder(dataset) / "downloads" / product)


def ensure_synapse_table(dataset, project_root=None, force: bool = False,
                         progress_callback=None) -> Optional[Path]:
    """Ensure the per-synapse parquet is fully downloaded locally.

    Resumable: a partial file continues from its current length.  Returns
    the local path on success, None when the bucket has no table for the
    release.
    """
    local = synapse_table_path(dataset, project_root)
    version = _connection_version(dataset)
    url = f"{BUCKET_HTTP_BASE}/{SYNAPSE_TABLES[version]}"
    # Never trust a bare existence check: fetch_range_to_file no-ops when
    # the local file matches the remote size, resumes when partial, and
    # restarts when force is set — a truncated leftover is completed, not
    # handed to the reader.
    print(f"  [banc] Ensuring the per-synapse table is local "
          f"(~3.9 GB; downloaded once, resumed when partial)...")
    return fetch_range_to_file(
        url, local, force=force, progress_callback=progress_callback)


def build_synapse_table(per_synapse_parquet: Path, derived_path: Path,
                        progress_callback=None) -> bool:
    """Aggregate the per-synapse parquet into a per-pair synapse table.

    The remote table stores one row per detected synapse (pre-site
    coordinates only, nanometres, root-id 0 placeholders included).  The
    derived table carries one row per (pre, post) pair: ``syn_count``
    (synapses per pair) and the mean pre-site position.  The release has
    no post-site coordinates, so no post columns are written — the reader
    mirrors the pre-site position onto ``x/y/z_post`` after loading,
    which keeps BANC synapse markers on the pre-synaptic site without
    persisting three duplicate columns (39.5% of the old file).

    Streams row-group by row-group so peak memory stays bounded; partial
    aggregates are re-folded periodically.  The output is written to a
    temporary sibling and atomically renamed, so an interrupted run never
    leaves a truncated table at ``derived_path``.
    """
    try:
        import polars as pl

        running = None
        columns = ["pre_root_id", "post_root_id", "pre_x", "pre_y", "pre_z"]
        import pyarrow.parquet as pq

        reader = pq.ParquetFile(per_synapse_parquet)
        n_groups = reader.metadata.num_row_groups

        def fold(frame):
            nonlocal running
            part = frame.group_by(["pre_root_id", "post_root_id"]).agg(
                pl.len().alias("syn_count"),
                pl.col("pre_x").sum().alias("sx"),
                pl.col("pre_y").sum().alias("sy"),
                pl.col("pre_z").sum().alias("sz"),
            )
            running = (part if running is None
                       else pl.concat([running, part], rechunk=True))
            if running.height > 5_000_000:
                running = running.group_by(
                    ["pre_root_id", "post_root_id"]).agg(
                    pl.col("syn_count").sum(),
                    pl.col("sx").sum(), pl.col("sy").sum(), pl.col("sz").sum())

        reader = pq.ParquetFile(per_synapse_parquet)
        for group in range(n_groups):
            frame = pl.from_arrow(
                reader.read_row_group(group, columns=columns))
            frame = frame.filter(
                (pl.col("pre_root_id") > 0)
                & (pl.col("post_root_id") > 0))
            if frame.height:
                fold(frame.with_columns(
                    pl.col("pre_root_id").cast(pl.Int64),
                    pl.col("post_root_id").cast(pl.Int64),
                    pl.col("pre_x").cast(pl.Float64),
                    pl.col("pre_y").cast(pl.Float64),
                    pl.col("pre_z").cast(pl.Float64),
                ))
            if progress_callback is not None:
                progress_callback(group + 1, n_groups)

        if running is None or running.height == 0:
            return False
        final = running.group_by(["pre_root_id", "post_root_id"]).agg(
            pl.col("syn_count").sum().alias("syn_count"),
            (pl.col("sx").sum() / pl.col("syn_count").sum())
                .cast(pl.Float32).alias("x_pre"),
            (pl.col("sy").sum() / pl.col("syn_count").sum())
                .cast(pl.Float32).alias("y_pre"),
            (pl.col("sz").sum() / pl.col("syn_count").sum())
                .cast(pl.Float32).alias("z_pre"),
        )
        # The release publishes only pre-site coordinates.  The generic
        # reader mirrors them onto the post columns in memory, so the
        # duplicates are not persisted.
        final = final.sort(["pre_root_id", "post_root_id"])
        Path(derived_path).parent.mkdir(parents=True, exist_ok=True)
        # Atomic output: an interrupted write leaves a temp sibling (never
        # a truncated table), which the next run cleans up and rebuilds.
        write_parquet_atomic(
            str(derived_path),
            lambda temp: final.write_parquet(temp, compression="zstd"))
        print(f"  ✓ Synapse table written: {derived_path} "
              f"({final.height:,} connections)")
        # Born encoded: apply the lossless layout right away so no later
        # upgrade pass is ever needed for fresh derivations.
        reencode_parquet_lossless(derived_path, BANC_SYNAPSE_PROFILE,
                                  progress_callback=progress_callback)
        return True
    except Exception as exc:
        print(f"  ⚠️ Failed to build the BANC synapse table: {exc}")
        return False


# Lossless storage profile for the derived per-pair synapse table.
# Encodings decoded transparently on read; measured on the v888 release
# (1.84 GB -> 1.38 GB, -25%, values bit-identical).  post_root_id is
# scattered within the pre-sorted table so it takes BYTE_STREAM_SPLIT
# instead of dictionary; z_pre stays PLAIN (split measured worse there).
BANC_SYNAPSE_PROFILE = {
    "drop_columns": ("x_post", "y_post", "z_post"),
    "column_encoding": {
        "pre_root_id": "DELTA_BINARY_PACKED",
        "post_root_id": "BYTE_STREAM_SPLIT",
        "syn_count": "BYTE_STREAM_SPLIT",
        "x_pre": "BYTE_STREAM_SPLIT",
        "y_pre": "BYTE_STREAM_SPLIT",
    },
    "use_dictionary": False,
    "compression": "zstd",
    "compression_level": 7,
}


def compact_synapse_table(derived_path, progress_callback=None) -> bool:
    """Bring a derived synapse table to the current lossless layout.

    One streaming pass drops the legacy mirrored post-site columns
    (pre-v4.5 tables), applies the measured encodings and stamps the
    footer marker; tables already carrying the marker are left
    untouched.  Values are preserved exactly.
    """
    return reencode_parquet_lossless(
        derived_path, BANC_SYNAPSE_PROFILE,
        progress_callback=progress_callback)


def ensure_synapse_derived_table(dataset, dataset_dir,
                                 project_root=None,
                                 progress_callback=None) -> bool:
    """Ensure the derived per-pair synapse table exists for *dataset*.

    Downloads the remote per-synapse parquet once (resumable), then
    aggregates it into ``datasets/<dataset>/<dataset>_synapse_table.parquet``
    — the exact fallback name the generic FlyWire synapse reader probes.
    A table left over from an older DROCAT generation is compacted in
    place (mirrored post-site columns dropped) before reuse.  A table
    truncated by an interrupted run is detected via its footer, discarded,
    and rebuilt — re-fetching the raw download if that was already
    reclaimed.
    """
    dataset_dir = str(dataset_dir)
    derived = os.path.join(
        dataset_dir, f"{_dataset_folder(dataset)}_synapse_table.parquet")
    if os.path.exists(derived):
        if parquet_readable(derived):
            # One-time reclaim for tables from before the reader-side
            # mirroring; a no-op for the current 6-column schema.
            compact_synapse_table(derived, progress_callback)
            return True
        print("  ⚠️ Existing synapse table is incomplete (the previous "
              "run was interrupted); discarding and rebuilding it...")
        try:
            os.remove(derived)
        except OSError:
            pass
    local = ensure_synapse_table(dataset, project_root)
    if local is None:
        return False
    version = _connection_version(dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    ok = build_synapse_table(
        local, derived,
        progress_callback=progress_callback)
    if ok:
        # The per-synapse download is re-fetchable; reclaim its space once
        # the derived per-pair table exists.
        try:
            reclaimed_gb = local.stat().st_size / 1e9
            local.unlink()
            print(f"  ✓ Removed the raw per-synapse download "
                  f"({reclaimed_gb:.1f} GB reclaimed).")
        except OSError:
            pass
    return ok


def meta_feather_path(dataset, project_root=None) -> Path:
    return (_project_root(project_root) / "datasets"
            / _dataset_folder(dataset) / "downloads"
            / "banc_888_meta.feather")


def connections_product_path(dataset, project_root=None) -> Path:
    version = _connection_version(dataset)
    return (_project_root(project_root) / "datasets"
            / _dataset_folder(dataset) / "downloads"
            / f"connections_{version}.parquet")


def download_meta_feather(dataset, project_root=None,
                          force: bool = False) -> Optional[Path]:
    """Download the release meta feather into the dataset downloads dir."""
    local = meta_feather_path(dataset, project_root)
    if local.exists() and not force:
        return local
    url = f"{BUCKET_HTTP_BASE}/{META_FEATHER_PATH}"
    blob = http_get(url, timeout=300, attempts=3, note="banc_888_meta.feather")
    if blob is None:
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    temp = local.with_name(f".{local.name}.{os.getpid()}.tmp")
    temp.write_bytes(blob)
    os.replace(temp, local)
    return local


def download_connections_product(dataset, project_root=None,
                                 force: bool = False) -> Optional[Path]:
    """Download the release connection-count product (thresholds baked in)."""
    local = connections_product_path(dataset, project_root)
    if local.exists() and not force:
        return local
    product = CONNECTION_PRODUCTS[_connection_version(dataset)]
    url = f"{BUCKET_HTTP_BASE}/{product}"
    blob = http_get(url, timeout=600, attempts=3, note=product)
    if blob is None:
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    temp = local.with_name(f".{local.name}.{os.getpid()}.tmp")
    temp.write_bytes(blob)
    os.replace(temp, local)
    return local


def build_neurons_dataframe(meta_feather: Path, version: str):
    """Convert the meta feather into the converter's neuron-table schema.

    ``type`` keeps the Codex convention of ``'Unknown'`` for unannotated
    neurons; ``Alternative Cell Type(s)`` preserves the exact header the
    cross-dataset type mapper requires (it reads the CSV variant of this
    table and hard-fails the whole mapper on header drift).
    """
    import pandas as pd

    frame = pd.read_feather(meta_feather)

    id_column = "root_626" if version == "v626" else "root_888"
    if id_column not in frame.columns:
        id_column = "root_888"
    # Merge every retained column positionally FIRST, dedupe on bodyId
    # LAST: v626 ids carry duplicates (coarser materialization) and a
    # pre-dedupe would desynchronize the column merge.
    neurons = pd.DataFrame({"bodyId": frame[id_column].astype(str).str.strip()})
    for column in frame.columns:
        if column in ("root_626", "root_888", "root_850", "root_890",
                      "root_id", "banc_888_id", "supervoxel_id", "nucleus_id"):
            continue
        neurons[column] = frame[column].values
    neurons = neurons.drop_duplicates(subset=["bodyId"], keep="first")

    neurons["type"] = neurons["cell_type"].astype("string").str.strip()
    neurons["type"] = neurons["type"].fillna("Unknown").replace("", "Unknown")

    neurons["Alternative Cell Type(s)"] = neurons.apply(
        _join_alternative_types, axis=1)

    rename_map = {
        "cell_class": "Class",
        "cell_sub_class": "Sub Class",
        "side": "Soma side",
        "neurotransmitter_predicted": "nt_type",
        "neurotransmitter_score": "Predicted NT confidence",
        "neurotransmitter_verified": "Verified NT type",
        "neuropeptide_verified": "Verified Neuropeptide",
        "body_part_sensory": "Body Part",
        "cell_function": "Function",
        # The release column is explicitly micrometres — label it honestly
        # (the Codex CSV mislabeled the same values as "(nm)").
        "l2_cable_length_um": "Cable length (µm)",
        "volume_nm3": "Volume (nm^3)",
        "super_class": "super_class",
        "flow": "flow",
        "nerve": "nerve",
        "hemilineage": "hemilineage",
    }
    neurons = neurons.rename(columns=rename_map)

    neurons["instance"] = neurons["type"]
    neurons["post"] = 0

    standard_cols = ["bodyId", "type", "instance", "post", "super_class",
                     "Class", "Sub Class", "Soma side", "hemilineage",
                     "nerve", "flow", "nt_type"]
    all_cols = list(neurons.columns)
    neurons = neurons[[c for c in standard_cols if c in all_cols]
                      + [c for c in all_cols if c not in standard_cols]]
    return neurons.sort_values("bodyId").reset_index(drop=True)


def prepare_dataset_tables(dataset_name, dataset_dir,
                           project_root=None) -> bool:
    """Prepare neuron + connection tables from the public bucket products.

    The meta feather stays in ``<dataset_dir>/downloads/`` (it is read at
    runtime by the neuron index and the type mapper) so re-runs skip the
    network entirely.  The connections product is only fetched when the
    merged table is missing — it has no consumer after the derivation and
    is safe to delete once ``{dataset}_merged_connections.parquet``
    exists.  Returns True when both parquet tables exist afterwards.
    """
    from BANC_file_converter import (
        process_connections_dataframe,
        process_neurons_dataframe,
    )

    # Keep the dataset directory, but make every filename and release lookup
    # use the same safe canonical namespace.  This preserves legacy callers
    # that pass ``flywire_BANC_*`` or ``banc:v888`` while avoiding split table
    # names inside the canonical dataset directory.
    reject_unknown_banc_release(dataset_name)
    dataset_name = _dataset_folder(dataset_name)
    version = _connection_version(dataset_name)
    dataset_dir = str(dataset_dir)
    # The caller's dataset_dir wins: derive the downloads root from it so
    # temporary/test dataset directories stay fully isolated.
    project_root = str(Path(dataset_dir).parent.parent)

    meta_local = download_meta_feather(dataset_name, project_root)
    if meta_local is None:
        return False

    neuron_pq = os.path.join(dataset_dir,
                             f"{dataset_name}_allneurons_neuron_df.parquet")
    neuron_csv = os.path.join(dataset_dir,
                              f"{dataset_name}_allneurons_neuron_df.csv")
    conn_pq = os.path.join(dataset_dir,
                           f"{dataset_name}_merged_connections.parquet")

    ok = True
    if not parquet_is_reusable(neuron_pq):
        neurons = build_neurons_dataframe(meta_local, version)
        ok = process_neurons_dataframe(neurons, neuron_pq,
                                       save_csv_path=neuron_csv) and ok
    if not parquet_is_reusable(conn_pq):
        # The connections product has no consumer beyond this derivation;
        # skip the 76 MB download entirely while the merged table exists.
        conn_local = download_connections_product(dataset_name, project_root)
        if conn_local is None:
            return False
        ok = process_connections_dataframe(conn_local, conn_pq) and ok
    return ok
