import gc
import os
import re
import sys
import json
import threading
from collections import OrderedDict
from copy import copy
from types import SimpleNamespace
import warnings

# Make the project root importable regardless of how this module was loaded
# (scripts put only src/ on sys.path; the `from src...` fallbacks below
# silently failed from a non-repo cwd - e.g. the NeuPrint token stayed empty).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Console encoding guard (Defect B): progress/status output prints ⚠/✓,
# which aborts with UnicodeEncodeError on legacy Windows code pages when
# stdout is redirected.
try:
    from .utils.console_encoding import ensure_utf8_stdio
except ImportError:
    try:
        from utils.console_encoding import ensure_utf8_stdio
    except ImportError:
        ensure_utf8_stdio = None
if ensure_utf8_stdio is not None:
    ensure_utf8_stdio()

# Suppress FutureWarning from neuprint about Series.__getitem__
warnings.filterwarnings("ignore", category=FutureWarning, module="neuprint")

import bokeh.palettes
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import navis
import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
from neuprint import *
# Explicit imports for Pylance static analysis (already imported via *)
from neuprint import Client, fetch_neurons
from tqdm import tqdm

try:
    from .utils.naming_utils import canonical_dataset_name
except ImportError:  # pragma: no cover - src laid bare on sys.path
    from utils.naming_utils import canonical_dataset_name

try:
    from .flywire_ids import (
        is_banc_dataset,
        is_fafb_dataset,
        is_flywire_dataset,
        is_local_connectome_dataset,
        normalize_flywire_body_id,
        normalize_flywire_id_columns,
        resolve_flywire_dataset_dir,
    )
except ImportError:
    from flywire_ids import (
        is_banc_dataset,
        is_fafb_dataset,
        is_flywire_dataset,
        is_local_connectome_dataset,
        normalize_flywire_body_id,
        normalize_flywire_id_columns,
        resolve_flywire_dataset_dir,
    )

# FlyWire client support removed


def _tqdm_print(*args, **kwargs):
    """Write a message without tearing an active progress bar.

    The UI runner captures stdout and stderr independently. ``tqdm`` bars
    normally use stderr, while ordinary ``print`` calls use stdout; a message
    written between a bar's clear and redraw can therefore arrive out of
    order in the live log. ``tqdm.write`` performs the clear/write/refresh
    sequence under tqdm's lock, so keep dataset-pull messages on the same
    stdout stream as the NeuronBridge bars.

    The fallback keeps the helper compatible with the lightweight tqdm fakes
    used by tests and with environments where tqdm is replaced by a callable.
    """
    writer = getattr(tqdm, "write", None)
    if writer is None:
        print(*args, **kwargs)
        return

    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    file = kwargs.pop("file", sys.stdout)
    flush = kwargs.pop("flush", False)
    message = sep.join(str(arg) for arg in args)

    if end == "\n":
        # tqdm.write clears active bars and refreshes them after the message.
        writer(message, file=file, end=end)
        if flush:
            file.flush()
        return

    # Preserve print(..., end=...) for the few progress-adjacent status
    # messages that intentionally stay on one line.
    external_write_mode = getattr(tqdm, "external_write_mode", None)
    if external_write_mode is None:
        print(message, end=end, file=file, flush=flush)
        return
    with external_write_mode(file=file):
        print(message, end=end, file=file, flush=flush)

# ============================================================================
# Local HTML/JS escaping helpers for the deprecated local heatmap template.
# The canonical implementation lives in vispath_pkg.shared_controls; these
# copies exist because the local fallback path must stay self-contained.
# ============================================================================

def _statvis_html_escape(value):
    """Escape a string for safe embedding as HTML text content."""
    if value is None:
        return ""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _statvis_js_escape(value):
    """Escape a string for safe embedding inside a JS single-quoted literal."""
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\\'")
    text = text.replace('"', '\\"')
    text = text.replace("\n", "\\n").replace("\r", "\\r")
    text = text.replace("<", "\\u003c").replace(">", "\\u003e")
    text = text.replace("&", "\\u0026")
    return text

# ============================================================================
# In-Memory Cache for Neuron DataFrames
# ============================================================================
# Avoids repeated CSV reads when getNeurons() is called multiple times
# Structure: {dataset: {'neuron_df': DataFrame, 'roi_df': DataFrame}}
_NEURON_DF_CACHE = {}
_DATASET_DOWNLOAD_LOCKS = {}
_FAILED_DATASET_DOWNLOADS = set()


# Signature of a pull that failed because of the *environment* rather than
# because the data itself could not be written: no NeuPrint client/token
# configured, a rejected token, or a transport failure.  neuprint raises plain
# RuntimeErrors with these words ("No token provided. Please provide one or set
# NEUPRINT_APPLICATION_CREDENTIALS", "No default Client has been set ..."), and
# this module wraps the no-client case in "no NeuPrint connection exists";
# requests/urllib3/socket failures are OSError descendants but their server
# text carries the status code.  Word boundaries keep a ``401`` inside a bodyId,
# port or coordinate from reading as a rejected token (same guard as
# ``visualize_skeleton._token_rejected``).
_DATASET_PULL_ENV_ERROR_RE = re.compile(
    r'no neuprint connection'
    r'|not available locally and no'
    r'|default client'
    r'|\btoken\b'
    r'|\bcredentials?\b'
    r'|\b40[13]\b'
    r'|unauthorized|forbidden'
    r'|\bunreachable\b'
    r'|connection \w*(?:refus|reset|abort|fail|clos|error)'
    r'|\btimed? ?out\b|\btimeout\b'
    r'|name resolution|getaddrinfo'
    r'|\bssl\b|certificate'
    r'|\bproxy\b',
    re.IGNORECASE,
)

# OSError subclasses that describe the local filesystem, not the network: a
# request that died because a path or permission was wrong is a genuine
# "we could not write the files" outcome and stays memoized.
_LOCAL_FILESYSTEM_ERRORS = (
    FileNotFoundError, FileExistsError, PermissionError,
    NotADirectoryError, IsADirectoryError,
)


def _is_environment_dataset_pull_error(exc: BaseException) -> bool:
    """Whether a ``pull_dataset`` failure is an environment/access problem.

    Environment failures (no token, rejected token, offline, refused or timed
    out connection, user cancel) tell us nothing about whether a later attempt
    would succeed — success depends on state outside this process.  They must
    therefore propagate as if no attempt had been made; see
    :func:`_ensure_local_dataset_files`.

    Anything else — including an unclassifiable exception — returns False, so
    the conservative "we tried and the files are still not there" memoization
    is preserved.
    """
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, DatasetPullCancelled):
            return True
        if isinstance(current, OSError) and not isinstance(
                current, _LOCAL_FILESYSTEM_ERRORS):
            # requests.RequestException/ConnectionError/Timeout, urllib3 and
            # socket errors are all OSError descendants.
            return True
        if _DATASET_PULL_ENV_ERROR_RE.search(str(current)):
            return True
        current = current.__cause__ or current.__context__
    return False


def clear_failed_dataset_downloads(dataset: str = None) -> None:
    """Drop the "do not pull this dataset again" memo (one dataset, or all).

    Test hook: a fixture calls this between tests so a failed pull in one test
    cannot change the outcome of the next one.  Production code does not need
    it — a successful pull clears its own entry and an environment failure
    never records one.
    """
    if dataset is None:
        _FAILED_DATASET_DOWNLOADS.clear()
        return
    _FAILED_DATASET_DOWNLOADS.discard(
        canonical_dataset_name(dataset).replace(':', '_').replace('.', '_'))


def _flywire_neuron_table_path(dataset: str, project_root: str):
    """Resolve an exact FAFB/BANC local-release table without fallback."""

    data_dir = resolve_flywire_dataset_dir(project_root, dataset)
    if data_dir is None:
        return None
    dataset_name = data_dir.name
    candidates = [
        data_dir / f"{dataset_name}_allneurons_neuron_df.parquet",
        data_dir / f"{dataset_name}_allneurons_neuron_df.csv",
    ]
    return next((str(path) for path in candidates if path.exists()), None)

# Polars is a hard dependency (pinned in requirements.txt and imported at
# module scope above); HAS_POLARS is kept for the legacy fast-loader preference.
HAS_POLARS = True


# Column names produced when a CSV was written with a serialized pandas
# index: an empty header, ``index``, pandas' ``Unnamed: 0`` placeholders or
# polars' ``column_1`` auto-names.
_INDEX_LIKE_COLUMN_RE = re.compile(
    r'^(?:|index|level_\d+|Unnamed: \d+(?: level_\d+)?|column_\d+)$'
)


def drop_leading_index_columns(df):
    """Drop leading serialized-index columns from a DataFrame.

    Neuron-metadata CSVs written with ``to_csv(index=True)`` carry a leading
    positional index column; depending on the reader it arrives unnamed, as
    ``Unnamed: 0`` (pandas) or ``column_1`` (polars).  These columns are
    noise for downstream exports (source_neurons.csv, neuron_info.csv, ...),
    so drop the leading run of index-like columns.

    Guards keep real data safe: at least one non-index column must remain and
    the first dropped column must hold integer values, which a positional
    index always does but a genuine data column practically never does.
    Accepts pandas and polars frames.
    """
    columns = list(df.columns)
    drop_names = []
    for name in columns:
        if not _INDEX_LIKE_COLUMN_RE.match(str(name)):
            break
        if len(columns) - len(drop_names) <= 1:
            break
        drop_names.append(name)
    if not drop_names:
        return df

    try:
        import polars as pl
        is_polars = isinstance(df, pl.DataFrame)
    except ImportError:
        is_polars = False

    if is_polars:
        first_values = df.get_column(drop_names[0]).to_numpy()
    else:
        first_values = df[drop_names[0]].to_numpy()
    numeric = pd.to_numeric(pd.Series(first_values), errors='coerce')
    if numeric.isna().any() or not (numeric % 1 == 0).all():
        return df
    if is_polars:
        return df.drop(*drop_names)
    return df.drop(columns=drop_names)


def _load_dataframe_fast(file_path: str, dtype_overrides: dict = None) -> pd.DataFrame:
    """
    Load a DataFrame using the fastest available method.
    
    Priority: parquet > polars CSV > pandas CSV
    
    Args:
        file_path: Path to CSV or parquet file
        dtype_overrides: Dict of column dtypes (e.g., {'bodyId': str})
    
    Returns:
        pandas DataFrame
    """
    # A parquet sibling is a speed optimization, not a second source of
    # truth.  Pulls write the ``*_neuron_df.csv`` metadata first; if that CSV
    # is newer, reading the older parquet would let the dataframe resolver
    # disagree with the freshly built neuron index.  Prefer parquet only when
    # it is at least as new as the requested source file (or when the source
    # path itself is absent).
    parquet_path = file_path.rsplit('.', 1)[0] + '.parquet'
    parquet_is_current = False
    if os.path.exists(parquet_path):
        if not os.path.exists(file_path):
            parquet_is_current = True
        else:
            try:
                parquet_is_current = (
                    os.stat(parquet_path).st_mtime_ns
                    >= os.stat(file_path).st_mtime_ns
                )
            except OSError:
                parquet_is_current = False
    if parquet_is_current:
        df = pd.read_parquet(parquet_path)
        if dtype_overrides:
            for col, dtype in dtype_overrides.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
        return drop_leading_index_columns(df)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Use polars for CSV if available (3-10x faster)
    if HAS_POLARS:
        try:
            # Polars reads CSV very fast
            schema_overrides = None
            if dtype_overrides:
                string_columns = {
                    col: pl.Utf8
                    for col, dtype in dtype_overrides.items()
                    if dtype is str or str(dtype).lower() in {'string', 'object'}
                }
                schema_overrides = string_columns or None
            pl_df = pl.read_csv(
                file_path,
                infer_schema_length=10000,
                schema_overrides=schema_overrides,
            )
            df = pl_df.to_pandas()
            # Apply dtype overrides after conversion
            if dtype_overrides:
                for col, dtype in dtype_overrides.items():
                    if col in df.columns:
                        df[col] = df[col].astype(dtype)
            return drop_leading_index_columns(df)
        except Exception:
            pass  # Fall back to pandas
    
    # Fallback to pandas
    df = pd.read_csv(file_path, low_memory=False, dtype=dtype_overrides)
    return drop_leading_index_columns(df)


def _get_cached_neuron_df(dataset: str, dataset_path_body: str):
    """
    Load neuron DataFrame from cache or disk.
    
    First call loads from disk and caches in memory.
    Subsequent calls return the cached DataFrame instantly.
    
    Uses polars for fast CSV reading if available, falls back to pandas.
    Prefers parquet format if available (fastest).
    
    Args:
        dataset: Dataset identifier (normalized, e.g., 'hemibrain_v1_2_1')
        dataset_path_body: Path prefix for data files (without suffix)
    
    Returns:
        Tuple of (neuron_df, roi_count_df)
    """
    global _NEURON_DF_CACHE
    
    if dataset in _NEURON_DF_CACHE:
        # Return cached DataFrames
        cached = _NEURON_DF_CACHE[dataset]
        return cached['neuron_df'].copy(), cached['roi_df'].copy()
    
    # File paths
    neuron_csv = dataset_path_body + '_neuron_df.csv'
    roi_table = roi_count_table_path(dataset_path_body)

    # Load using fast loader (handles parquet/polars/pandas priority)
    ndf = _load_dataframe_fast(neuron_csv)
    rdf = _load_dataframe_fast(roi_table)
    
    if is_local_connectome_dataset(dataset):
        normalize_flywire_id_columns(ndf, ['bodyId'])
        normalize_flywire_id_columns(rdf, ['bodyId'])
    else:
        # NeuPrint's legacy client expects integer body IDs. This conversion
        # is intentionally restricted to non-local-release datasets.
        if 'bodyId' in ndf.columns:
            try:
                ndf['bodyId'] = ndf['bodyId'].astype('int64')
            except (ValueError, TypeError):
                pass
        if 'bodyId' in rdf.columns:
            try:
                rdf['bodyId'] = rdf['bodyId'].astype('int64')
            except (ValueError, TypeError):
                pass
    
    # Cache in memory
    _NEURON_DF_CACHE[dataset] = {
        'neuron_df': ndf,
        'roi_df': rdf
    }
    
    return ndf.copy(), rdf.copy()


def clear_neuron_cache(dataset: str = None):
    """
    Clear the neuron DataFrame cache.
    
    Args:
        dataset: Specific dataset to clear. If None, clears all.
    """
    global _NEURON_DF_CACHE
    
    if dataset is None:
        _NEURON_DF_CACHE.clear()
    elif dataset in _NEURON_DF_CACHE:
        del _NEURON_DF_CACHE[dataset]


def _get_cached_neuron_search(dataset: str):
    """Return the shared parquet search reader when a local cache exists."""
    try:
        from src.neuron_search import get_cached_neuron_search
    except ImportError:
        try:
            from neuron_search import get_cached_neuron_search
        except ImportError:
            return None
    return get_cached_neuron_search(dataset)


_TOKEN_RESOLUTION_CACHE_LIMIT = 20000
_TOKEN_RESOLUTION_CACHE: OrderedDict = OrderedDict()


def _token_cache_key(required_neuron, frame, dataset, search_columns):
    return (str(dataset), str(required_neuron), str(search_columns or 'auto'))


def _lookup_token_resolution(required_neuron, frame, dataset, search_columns):
    """Return a prior resolution of the same token against the same frame.

    A run resolves the same chip list twice (source and target passes) and
    often repeats tokens across steps; each resolution otherwise re-scans the
    whole metadata frame.  Entries hold a weak reference to the frame so a
    reloaded table can never inherit stale results.
    """
    import weakref
    key = _token_cache_key(required_neuron, frame, dataset, search_columns)
    entry = _TOKEN_RESOLUTION_CACHE.get(key)
    if entry is None:
        return None
    frame_ref, frame_len, body_ids, info = entry
    if frame_ref() is not frame or frame_len != len(frame):
        _TOKEN_RESOLUTION_CACHE.pop(key, None)
        return None
    _TOKEN_RESOLUTION_CACHE.move_to_end(key)
    return list(body_ids), dict(info)


def _store_token_resolution(required_neuron, frame, dataset, search_columns, body_ids, search_info):
    import weakref
    try:
        frame_ref = weakref.ref(frame)
    except TypeError:
        return
    key = _token_cache_key(required_neuron, frame, dataset, search_columns)
    _TOKEN_RESOLUTION_CACHE[key] = (
        frame_ref, len(frame), list(body_ids), dict(search_info or {})
    )
    while len(_TOKEN_RESOLUTION_CACHE) > _TOKEN_RESOLUTION_CACHE_LIMIT:
        _TOKEN_RESOLUTION_CACHE.popitem(last=False)


_QUERY_PRINT_CAP = 10


class _ResolutionSummary:
    """Aggregate per-chip resolution outcomes for one getNeurons query.

    Beyond a handful of chips one print per chip is noise (a 219-chip run
    printed 219 near-identical lines); once the query exceeds
    ``_QUERY_PRINT_CAP`` entries the loop stays silent and this summary
    carries the totals instead: chips in, neurons out, and any chip that
    matched nothing.
    """

    def __init__(self, total_chips: int):
        self.total_chips = total_chips
        self.cached_hits = 0
        self.dataframe_hits = 0
        self.missed_chips = []
        self.body_id_keys = set()

    def add(self, chip, body_ids, search_info=None):
        if body_ids:
            if (search_info or {}).get('cache'):
                self.cached_hits += 1
            else:
                self.dataframe_hits += 1
            self.body_id_keys.update(str(body_id) for body_id in body_ids)
        else:
            self.missed_chips.append(str(chip))

    def report(self):
        print(
            f'✓ Resolved {self.total_chips} queries → '
            f'{len(self.body_id_keys)} neurons '
            f'(cached search: {self.cached_hits}; '
            f'dataframe search: {self.dataframe_hits})'
        )
        if self.missed_chips:
            shown = ', '.join(self.missed_chips[:5])
            extra = (
                f' +{len(self.missed_chips) - 5} more'
                if len(self.missed_chips) > 5 else ''
            )
            print(
                f'\033[33m⚠️  {len(self.missed_chips)} of {self.total_chips} '
                f'queries matched no neurons: {shown}{extra}\033[0m'
            )


def _resolve_single_neuron(
    required_neuron,
    ndf_alltypes,
    body_id_alltypes,
    *,
    dataset,
    cached_search=None,
    verbose=True,
    search_columns='auto',
    search_info_sink=None,
):
    """Resolve through the shared cache, validating it against the frame."""
    try:
        from src.neuron_search import resolve_cached_or_dataframe_query
    except ImportError:  # pragma: no cover - ``src/`` on sys.path imports
        from neuron_search import resolve_cached_or_dataframe_query
    cached = _lookup_token_resolution(
        required_neuron, ndf_alltypes, dataset, search_columns
    )
    if cached is not None:
        body_ids, search_info = cached
    else:
        body_ids, search_info = resolve_cached_or_dataframe_query(
            cached_search,
            ndf_alltypes,
            required_neuron,
            search_columns=search_columns,
        )
        _store_token_resolution(
            required_neuron, ndf_alltypes, dataset, search_columns,
            body_ids, search_info,
        )
    if search_info_sink is not None:
        search_info_sink.append(dict(search_info or {}))
    if verbose:
        if body_ids:
            column = search_info.get('matched_column') or 'metadata'
            source = 'cached search' if search_info.get('cache') else 'dataframe search'
            print(
                f'Found {len(body_ids)} neurons for "{required_neuron}" '
                f'({source}; first: {column})'
            )
        else:
            print(
                f'No neurons found for "{required_neuron}" in the shared neuron index'
            )
    return body_ids, search_info

def _get_dataset_path_body(dataset: str) -> tuple[str, str, str]:
    """Return normalized dataset name, dataset directory, and file prefix."""
    dataset_normalized = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
    project_root = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(project_root, "datasets", dataset_normalized)

    if os.path.exists(dataset_dir):
        dataset_path_body = os.path.join(dataset_dir, f"{dataset_normalized}_allneurons")
    else:
        dataset_path_body = os.path.join(project_root, "datasets", f"{dataset_normalized}_allneurons")

    return dataset_normalized, dataset_dir, dataset_path_body


def roi_count_table_path(dataset_path_body: str) -> str:
    """Path of a dataset's ROI-count table: parquet when present, else CSV.

    ``pull_dataset`` writes ``_roi_count_df.parquet`` (zstd); CSVs from older
    pulls are still a valid source, so readers resolve whichever exists.
    """
    parquet = dataset_path_body + '_roi_count_df.parquet'
    if os.path.exists(parquet):
        return parquet
    return dataset_path_body + '_roi_count_df.csv'


def _ensure_local_dataset_files(dataset: str, client=None, verbose: bool = True) -> tuple[str, str]:
    """
    Ensure local dataset CSVs exist exactly once per dataset per process.

    This prevents repeated pull attempts when visualization requests the same
    dataset layer-by-layer and a previous pull failed or did not materialize
    the expected CSV files.

    Only the outcomes that are genuinely stable inside a process are memoized:

    * ``pull_dataset`` returned but the expected tables are still absent, or
    * ``pull_dataset`` raised something that is *not* an environment failure
      (see :func:`_is_environment_dataset_pull_error`).

    An authentication/offline/network failure is *not* recorded: it says
    nothing about the local files, and memoizing it turned one missing NeuPrint
    token into every later call in the process failing with a misleading
    "still missing local data files after a previous pull attempt".  Such a
    failure propagates unchanged so each call site sees the real cause.
    """
    global _DATASET_DOWNLOAD_LOCKS, _FAILED_DATASET_DOWNLOADS

    dataset_normalized, dataset_dir, dataset_path_body = _get_dataset_path_body(dataset)
    neuron_csv = dataset_path_body + '_neuron_df.csv'
    roi_table = roi_count_table_path(dataset_path_body)

    if os.path.exists(neuron_csv) and os.path.exists(roi_table):
        return dataset_normalized, dataset_path_body

    if dataset_normalized in _FAILED_DATASET_DOWNLOADS:
        raise FileNotFoundError(
            f"Dataset '{dataset}' is still missing local data files after a previous pull attempt. "
            f"Expected: {neuron_csv} and {roi_table}"
        )

    lock = _DATASET_DOWNLOAD_LOCKS.setdefault(dataset_normalized, threading.Lock())
    with lock:
        if os.path.exists(neuron_csv) and os.path.exists(roi_count_table_path(dataset_path_body)):
            return dataset_normalized, dataset_path_body

        if verbose:
            print(f'\033[33mcsv files of dataset "{dataset}" not found, downloading...\033[0m')

        dataset_path_body = os.path.join(dataset_dir, f"{dataset_normalized}_allneurons")
        neuron_csv = dataset_path_body + '_neuron_df.csv'

        try:
            pull_dataset(dataset, save_path=dataset_path_body, omitNoneType=False, client=client)
        except Exception as exc:
            if not _is_environment_dataset_pull_error(exc):
                _FAILED_DATASET_DOWNLOADS.add(dataset_normalized)
            raise

        if not os.path.exists(neuron_csv) or not os.path.exists(roi_count_table_path(dataset_path_body)):
            _FAILED_DATASET_DOWNLOADS.add(dataset_normalized)
            raise FileNotFoundError(
                f"pull_dataset('{dataset}') completed without creating the expected files: "
                f"{neuron_csv} and {dataset_path_body + '_roi_count_df.parquet'}"
            )

        _FAILED_DATASET_DOWNLOADS.discard(dataset_normalized)
        clear_neuron_cache(dataset_normalized)
        return dataset_normalized, dataset_path_body


# ============================================================================
# Neuron Query Helper Functions
# ============================================================================
# These functions provide convenient access to neuron info by various identifiers
# Similar query syntax as FindNeuronConnection: bodyIds, types, instances, regex

def _get_neuron_df(dataset: str = 'male-cns:v0.9', verbose: bool = False) -> pd.DataFrame:
    """
    Load the neuron DataFrame for a dataset from local cache/files.
    
    Args:
        dataset: Dataset identifier (e.g., 'male-cns:v0.9', 'hemibrain:v1.2.1')
        verbose: Print loading messages
    
    Returns:
        DataFrame with neuron information
    
    Raises:
        FileNotFoundError: If dataset files are not found locally
    """
    # Normalize dataset name
    dataset_normalized = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
    
    # FAFB and BANC both use exact local-release tables, but they remain
    # separate sources and therefore use separate cache namespaces.
    if is_local_connectome_dataset(dataset):
        cache_prefix = "banc" if is_banc_dataset(dataset) else "flywire"
        cache_key = f"{cache_prefix}_{dataset}"
        if cache_key in _NEURON_DF_CACHE:
            return _NEURON_DF_CACHE[cache_key]['neuron_df'].copy()
        
        # Try to load from local files
        try:
            import fafb_utils
            project_root = os.path.dirname(os.path.dirname(__file__))
            data_dir = resolve_flywire_dataset_dir(project_root, dataset)

            if data_dir is not None:
                neuron_file, _ = fafb_utils.prepare_flywire_data(data_dir)
                full_neuron_df = _load_dataframe_fast(neuron_file, dtype_overrides={'bodyId': str})
                if 'bodyId' in full_neuron_df.columns:
                    normalize_flywire_id_columns(full_neuron_df, ['bodyId'])
                _NEURON_DF_CACHE[cache_key] = {'neuron_df': full_neuron_df}
                return full_neuron_df.copy()
        except Exception as e:
            source = "BANC" if is_banc_dataset(dataset) else "FAFB"
            raise FileNotFoundError(f"Could not load {source} data: {e}")
        
        raise FileNotFoundError(f"Dataset '{dataset}' not found locally")
    
    # Standard neuprint datasets
    _, _, dataset_path_body = _get_dataset_path_body(dataset)

    neuron_csv = dataset_path_body + '_neuron_df.csv'
    if not os.path.exists(neuron_csv):
        raise FileNotFoundError(
            f"Dataset '{dataset}' not found locally at {neuron_csv}. "
            f"Run pull_dataset('{dataset}') to download it first."
        )
    
    ndf, _ = _get_cached_neuron_df(dataset_normalized, dataset_path_body)
    return ndf


def get_types(
    query,
    dataset: str = 'male-cns:v0.9',
    verbose: bool = True,
    return_simple: bool = False
):
    """
    Get neuron types for a query (bodyIds/types/instances/regex patterns/dict filters).
    
    Similar to FindNeuronConnection query syntax, this function searches local
    data files and returns type information for matching neurons.
    
    Args:
        query: Neuron query - can be:
            - Single identifier: 'aMe12', 720575940610453042
            - List of identifiers: ['Mi1', 'Tm3', 720575940610453042]
            - Regex pattern: 'aMe.*', 'Mi[1-9]'
            - Mixed: ['aMe.*', 'Mi1', 720575940610453042]
            - Dict filter: {'contains': 'DN'}  # Auto-searches type, instance, etc.
            - Dict filter: {'startswith': ['aMe', 'Mi']}  # OR for list values
            - Dict filter: {'contains': 'DN', 'endswith': '_R'}  # AND across operators
        dataset: Dataset identifier (default: 'male-cns:v0.9')
        verbose: Print search progress messages
        return_simple: If True, return just the type_list instead of tuple
    
    Returns:
        If return_simple=True:
            list: List of unique types found
        Otherwise:
            tuple: (type_list, map_dict, dataset)
                - type_list: List of unique types found
                - map_dict: Dict mapping {type: [input_items_that_matched]}
                - dataset: The dataset used (normalized)
    
    Examples:
        >>> # Simple return (recommended for most use cases)
        >>> type_list = get_types('aMe.*', return_simple=True)
        >>> print(type_list)  # ['aMe12', 'aMe17a', 'aMe17b', ...]
        
        >>> # Dict filter with simple return
        >>> type_list = get_types({'contains': 'DN'}, return_simple=True)
        
        >>> # Full return (legacy)
        >>> type_list, map_dict, ds = get_types('aMe.*')
        >>> print(type_list)  # ['aMe12', 'aMe17a', 'aMe17b', ...]
        
        >>> type_list, map_dict, ds = get_types(['Mi1', 'Tm3'])
        >>> print(map_dict)  # {'Mi1': ['Mi1'], 'Tm3': ['Tm3']}
        
        >>> # Dict filter examples
        >>> types = get_types({'startswith': ['DN', 'AN']}, return_simple=True)
        >>> types = get_types({'contains': 'DN', 'endswith': '_R'}, return_simple=True)
    """
    # Load neuron DataFrame
    ndf = _get_neuron_df(dataset, verbose=verbose)
    dataset_normalized = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
    
    # Check if query is dict-based filter
    if isinstance(query, dict):
        # Use NeuronFilter for dict-based queries
        try:
            from utils.neuron_filter import NeuronFilter
        except ImportError:
            from src.utils.neuron_filter import NeuronFilter
        
        nf = NeuronFilter(query)
        matched_df = nf.apply(ndf)
        type_list = sorted(matched_df['type'].dropna().unique().tolist()) if 'type' in matched_df.columns else []
        
        if verbose:
            print(f"Found {len(type_list)} unique types matching filter")
        
        if return_simple:
            return type_list
        
        # For dict queries, map_dict shows filter -> types
        map_dict = {t: [str(query)] for t in type_list}
        return type_list, map_dict, dataset_normalized
    
    # Legacy query handling
    # Normalize query to list
    if not isinstance(query, list):
        query = [query]
    
    bodyId_alltypes = ndf['bodyId'].tolist()
    cached_search = _get_cached_neuron_search(dataset)
    
    # Process each query item
    type_set = set()
    map_dict = {}  # type -> [input_items]
    
    for item in query:
        # Get matching bodyIds
        bodyId_list, search_info = _resolve_single_neuron(
            item,
            ndf,
            bodyId_alltypes,
            dataset=dataset,
            cached_search=cached_search,
            verbose=verbose,
        )
        
        if bodyId_list:
            # Get types for these bodyIds
            matched_df = ndf[ndf['bodyId'].isin(bodyId_list)]
            types_found = matched_df['type'].dropna().unique().tolist()
            
            for t in types_found:
                type_set.add(t)
                if t not in map_dict:
                    map_dict[t] = []
                map_dict[t].append(item)
    
    type_list = sorted(list(type_set))
    
    if verbose:
        print(f"Found {len(type_list)} unique types from {len(query)} query items")
    
    if return_simple:
        return type_list
    
    return type_list, map_dict, dataset_normalized


def get_bodyIds(
    query,
    dataset: str = 'male-cns:v0.9',
    verbose: bool = True,
    return_simple: bool = False
):
    """
    Get bodyIds for a query (bodyIds/types/instances/regex patterns/dict filters).
    
    Similar to FindNeuronConnection query syntax, this function searches local
    data files and returns bodyId information for matching neurons.
    
    Args:
        query: Neuron query - can be:
            - Single identifier: 'aMe12', 720575940610453042
            - List of identifiers: ['Mi1', 'Tm3', 720575940610453042]
            - Regex pattern: 'aMe.*', 'Mi[1-9]'
            - Mixed: ['aMe.*', 'Mi1', 720575940610453042]
            - Dict filter: {'contains': 'DN'}  # Auto-searches type, instance, etc.
            - Dict filter: {'startswith': ['aMe', 'Mi']}  # OR for list values
        dataset: Dataset identifier (default: 'male-cns:v0.9')
        verbose: Print search progress messages
        return_simple: If True, return just the bodyId_list instead of tuple
    
    Returns:
        If return_simple=True:
            list: List of all bodyIds found
        Otherwise:
            tuple: (bodyId_list, map_dict, dataset)
                - bodyId_list: List of all bodyIds found
                - map_dict: Dict mapping {bodyId: input_item_that_matched}
                - dataset: The dataset used (normalized)
    
    Examples:
        >>> # Simple return (recommended for most use cases)
        >>> bodyIds = get_bodyIds('aMe12', return_simple=True)
        >>> print(len(bodyIds))  # Number of aMe12 neurons
        
        >>> # Dict filter with simple return
        >>> bodyIds = get_bodyIds({'contains': 'DN'}, return_simple=True)
        
        >>> # Full return (legacy)
        >>> bodyIds, map_dict, ds = get_bodyIds('aMe12')
        >>> print(len(bodyIds))  # Number of aMe12 neurons
        
        >>> bodyIds, map_dict, ds = get_bodyIds(['Mi1', 'Tm3'])
        >>> print(len(bodyIds))  # Total Mi1 + Tm3 neurons
    """
    # Load neuron DataFrame
    ndf = _get_neuron_df(dataset, verbose=verbose)
    dataset_normalized = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
    
    # Check if query is dict-based filter
    if isinstance(query, dict):
        # Use NeuronFilter for dict-based queries
        try:
            from utils.neuron_filter import NeuronFilter
        except ImportError:
            from src.utils.neuron_filter import NeuronFilter
        
        nf = NeuronFilter(query)
        matched_df = nf.apply(ndf)
        bodyId_list = matched_df['bodyId'].tolist() if 'bodyId' in matched_df.columns else []
        
        if verbose:
            print(f"Found {len(bodyId_list)} bodyIds matching filter")
        
        if return_simple:
            return bodyId_list
        
        # For dict queries, map all bodyIds to the filter string
        map_dict = {bid: str(query) for bid in bodyId_list}
        return bodyId_list, map_dict, dataset_normalized
    
    # Legacy query handling
    # Normalize query to list
    if not isinstance(query, list):
        query = [query]
    
    bodyId_alltypes = ndf['bodyId'].tolist()
    cached_search = _get_cached_neuron_search(dataset)
    
    # Process each query item
    all_bodyIds = []
    map_dict = {}  # bodyId -> input_item
    
    for item in query:
        # Get matching bodyIds
        bodyId_list, search_info = _resolve_single_neuron(
            item,
            ndf,
            bodyId_alltypes,
            dataset=dataset,
            cached_search=cached_search,
            verbose=verbose,
        )
        
        for bid in bodyId_list:
            all_bodyIds.append(bid)
            map_dict[bid] = item
    
    # Remove duplicates while preserving order
    seen = set()
    bodyId_list = []
    for bid in all_bodyIds:
        if bid not in seen:
            seen.add(bid)
            bodyId_list.append(bid)
    
    if verbose:
        print(f"Found {len(bodyId_list)} unique bodyIds from {len(query)} query items")
    
    if return_simple:
        return bodyId_list
    
    return bodyId_list, map_dict, dataset_normalized


def get_instances(
    query,
    dataset: str = 'male-cns:v0.9',
    verbose: bool = True,
    return_simple: bool = False
):
    """
    Get neuron instances for a query (bodyIds/types/instances/regex patterns/dict filters).
    
    Similar to FindNeuronConnection query syntax, this function searches local
    data files and returns instance information for matching neurons.
    
    Args:
        query: Neuron query - can be:
            - Single identifier: 'aMe12', 720575940610453042
            - List of identifiers: ['Mi1', 'Tm3', 720575940610453042]
            - Regex pattern: 'aMe.*', 'Mi[1-9]'
            - Mixed: ['aMe.*', 'Mi1', 720575940610453042]
            - Dict filter: {'contains': 'DN'}  # Auto-searches type, instance, etc.
            - Dict filter: {'endswith': '_R'}  # Suffix match
        dataset: Dataset identifier (default: 'male-cns:v0.9')
        verbose: Print search progress messages
        return_simple: If True, return just the instance_list instead of tuple
    
    Returns:
        If return_simple=True:
            list: List of unique instances found
        Otherwise:
            tuple: (instance_list, map_dict, dataset)
                - instance_list: List of unique instances found
                - map_dict: Dict mapping {instance: [input_items_that_matched]}
                - dataset: The dataset used (normalized)
    
    Examples:
        >>> # Simple return (recommended for most use cases)
        >>> instances = get_instances('aMe12', return_simple=True)
        >>> print(instances)  # ['aMe12_L', 'aMe12_R', ...]
        
        >>> # Dict filter with simple return
        >>> instances = get_instances({'endswith': '_R'}, return_simple=True)
        
        >>> # Full return (legacy)
        >>> instances, map_dict, ds = get_instances('aMe12')
        >>> print(instances)  # ['aMe12_L', 'aMe12_R', ...]
        
        >>> instances, map_dict, ds = get_instances(['Mi1', 'Tm3'])
    """
    # Load neuron DataFrame
    ndf = _get_neuron_df(dataset, verbose=verbose)
    dataset_normalized = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
    
    # Check if instance column exists
    if 'instance' not in ndf.columns:
        if verbose:
            print(f"Warning: 'instance' column not found in dataset '{dataset}'")
        if return_simple:
            return []
        return [], {}, dataset_normalized
    
    # Check if query is dict-based filter
    if isinstance(query, dict):
        # Use NeuronFilter for dict-based queries
        try:
            from utils.neuron_filter import NeuronFilter
        except ImportError:
            from src.utils.neuron_filter import NeuronFilter
        
        nf = NeuronFilter(query)
        matched_df = nf.apply(ndf)
        instance_list = sorted(matched_df['instance'].dropna().unique().tolist())
        
        if verbose:
            print(f"Found {len(instance_list)} unique instances matching filter")
        
        if return_simple:
            return instance_list
        
        # For dict queries, map_dict shows filter -> instances
        map_dict = {inst: [str(query)] for inst in instance_list}
        return instance_list, map_dict, dataset_normalized
    
    # Legacy query handling
    # Normalize query to list
    if not isinstance(query, list):
        query = [query]
    
    bodyId_alltypes = ndf['bodyId'].tolist()
    cached_search = _get_cached_neuron_search(dataset)
    
    # Process each query item
    instance_set = set()
    map_dict = {}  # instance -> [input_items]
    
    for item in query:
        # Get matching bodyIds
        bodyId_list, search_info = _resolve_single_neuron(
            item,
            ndf,
            bodyId_alltypes,
            dataset=dataset,
            cached_search=cached_search,
            verbose=verbose,
        )
        
        if bodyId_list:
            # Get instances for these bodyIds
            matched_df = ndf[ndf['bodyId'].isin(bodyId_list)]
            instances_found = matched_df['instance'].dropna().unique().tolist()
            
            for inst in instances_found:
                instance_set.add(inst)
                if inst not in map_dict:
                    map_dict[inst] = []
                map_dict[inst].append(item)
    
    instance_list = sorted(list(instance_set))
    
    if verbose:
        print(f"Found {len(instance_list)} unique instances from {len(query)} query items")
    
    if return_simple:
        return instance_list
    
    return instance_list, map_dict, dataset_normalized


def get_info(
    query,
    dataset: str = 'male-cns:v0.9',
    columns: list = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Get full neuron information for a query (similar to getNeurons but simpler).
    
    This is a lightweight version of getNeurons() that returns just the 
    neuron DataFrame without ROI counts or criteria objects.
    
    Args:
        query: Neuron query - can be:
            - Single identifier: 'aMe12', 720575940610453042
            - List of identifiers: ['Mi1', 'Tm3', 720575940610453042]
            - Regex pattern: 'aMe.*', 'Mi[1-9]'
            - Mixed: ['aMe.*', 'Mi1', 720575940610453042]
            - None: Return all neurons in dataset
            - Dict filter: {'contains': 'DN'}  # Auto-searches type, instance, etc.
            - Dict filter: {'startswith': ['aMe', 'Mi'], 'endswith': '_R'}  # AND logic
        dataset: Dataset identifier (default: 'male-cns:v0.9')
        columns: List of columns to return. If None, returns all columns.
        verbose: Print search progress messages
    
    Returns:
        pd.DataFrame: Neuron information for matching neurons
    
    Examples:
        >>> df = get_info('aMe.*')
        >>> print(df[['bodyId', 'type', 'instance']].head())
        
        >>> df = get_info(['Mi1', 'Tm3'], columns=['bodyId', 'type', 'instance', 'soma'])
        
        >>> df = get_info(None)  # Get all neurons in dataset
        
        >>> # Dict filter examples
        >>> df = get_info({'contains': 'DN'})
        >>> df = get_info({'startswith': 'aMe', 'endswith': '_R'})
    """
    # Load neuron DataFrame
    ndf = _get_neuron_df(dataset, verbose=verbose)
    
    # Return all neurons if query is None
    if query is None:
        result = ndf.copy()
        if columns:
            available_cols = [c for c in columns if c in result.columns]
            result = result[available_cols]
        if verbose:
            print(f"Returning all {len(result)} neurons from dataset '{dataset}'")
        return result
    
    # Check if query is dict-based filter
    if isinstance(query, dict):
        # Use NeuronFilter for dict-based queries
        try:
            from utils.neuron_filter import NeuronFilter
        except ImportError:
            from src.utils.neuron_filter import NeuronFilter
        
        nf = NeuronFilter(query)
        result = nf.apply(ndf)
        
        # Select columns if specified
        if columns:
            available_cols = [c for c in columns if c in result.columns]
            missing_cols = [c for c in columns if c not in result.columns]
            if missing_cols and verbose:
                print(f"Warning: Columns not found: {missing_cols}")
            result = result[available_cols]
        
        if verbose:
            print(f"Found {len(result)} neurons matching filter")
        
        return result
    
    # Legacy query handling
    # Normalize query to list
    if not isinstance(query, list):
        query = [query]
    
    bodyId_alltypes = ndf['bodyId'].tolist()
    cached_search = _get_cached_neuron_search(dataset)
    
    # Process each query item and collect bodyIds
    all_bodyIds = []
    for item in query:
        bodyId_list, _ = _resolve_single_neuron(
            item,
            ndf,
            bodyId_alltypes,
            dataset=dataset,
            cached_search=cached_search,
            verbose=verbose,
        )
        all_bodyIds.extend(bodyId_list)
    
    # Remove duplicates
    unique_bodyIds = list(dict.fromkeys(all_bodyIds))
    
    # Filter DataFrame
    result = ndf[ndf['bodyId'].isin(unique_bodyIds)].copy()
    
    # Select columns if specified
    if columns:
        available_cols = [c for c in columns if c in result.columns]
        missing_cols = [c for c in columns if c not in result.columns]
        if missing_cols and verbose:
            print(f"Warning: Columns not found: {missing_cols}")
        result = result[available_cols]
    
    if verbose:
        print(f"Returning info for {len(result)} neurons")
    
    return result


class CreateHeatmap:
    """
    A class for creating and managing heatmap visualizations of connection matrices.
    
    This class provides a clean interface for generating multiple heatmaps with
    consistent styling and automatic color scale selection based on matrix type.
    
    Attributes
    ----------
    output_folder : str
        Directory where heatmap HTML files will be saved
    showfig : bool
        Whether to automatically open heatmaps in browser (default: False)
    default_fontsize : int
        Default font size for heatmap labels (default: 12)
    
    Examples
    --------
    >>> # Create heatmap generator
    >>> hm = CreateHeatmap(output_folder='./my_heatmaps', showfig=False)
    >>> 
    >>> # Add single heatmap
    >>> hm.add_heatmap(
    ...     matrix=conn_matrix_type,
    ...     name='connection_matrix_type',
    ...     title='Connection Matrix by Type',
    ...     color_scale='green'  # or custom [[0, 'white'], [1, 'green']]
    ... )
    >>> 
    >>> # Add multiple heatmaps at once
    >>> hm.add_heatmaps({
    ...     'conn_matrix': conn_matrix,
    ...     'ratio_matrix': ratio_matrix
    ... }, base_title='My Analysis')
    >>> 
    >>> # Generate all heatmaps
    >>> created_files = hm.create_all()
    """
    
    # Predefined color scales for different matrix types
    COLOR_SCALES = {
        'green': [[0, 'rgb(255,255,255)'], [1, 'rgb(14,83,13)']],
        'purple': [[0, 'rgb(255,255,255)'], [1, 'rgb(104,55,164)']],
        'orange': [[0, 'rgb(255,255,255)'], [1, 'rgb(204,102,0)']],
        'blue': [[0, 'rgb(255,255,255)'], [1, 'rgb(31,119,180)']],
        'red': [[0, 'rgb(255,255,255)'], [1, 'rgb(214,39,40)']],
    }
    
    def __init__(self, output_folder, showfig=False, default_fontsize=12):
        """
        Initialize CreateHeatmap instance.
        
        Parameters
        ----------
        output_folder : str
            Directory to save heatmap HTML files
        showfig : bool, optional
            Whether to auto-open heatmaps in browser (default: False)
        default_fontsize : int, optional
            Default font size for labels (default: 12)
        """
        self.output_folder = output_folder
        self.showfig = showfig
        self.default_fontsize = default_fontsize
        self.heatmaps = []
        
        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def add_heatmap(self, matrix, name, title=None, color_scale='purple', fontsize=None, scale='linear', interactive=False, conn_df=None):
        """
        Add a single heatmap to the generation queue.
        
        Parameters
        ----------
        matrix : pd.DataFrame
            Connection matrix to visualize
        name : str
            Base filename (without .html extension)
        title : str, optional
            Title to display on heatmap. If None, uses name.
        color_scale : str or list, optional
            Either a preset name ('green', 'purple', 'orange', 'blue', 'red')
            or a custom Plotly color scale [[0, 'color1'], [1, 'color2']]
        fontsize : int, optional
            Font size for this heatmap. If None, uses default_fontsize.
        scale : str, optional
            Scale for color mapping: 'linear', 'log2', or 'log10' (default: 'linear')
            Only used if interactive=False
        interactive : bool, optional
            If True, creates interactive heatmap with scale controls (default: False)
        conn_df : pd.DataFrame, optional
            Connection dataframe with type information for enhanced hover labels
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        if title is None:
            title = name.replace('_', ' ').title()
        
        if fontsize is None:
            fontsize = self.default_fontsize
        
        # Resolve color scale
        if isinstance(color_scale, str):
            if color_scale in self.COLOR_SCALES:
                color_scale = self.COLOR_SCALES[color_scale]
            else:
                # Default to purple if unknown preset
                color_scale = self.COLOR_SCALES['purple']
        
        self.heatmaps.append({
            'matrix': matrix,
            'name': name,
            'title': title,
            'color_scale': color_scale,
            'fontsize': fontsize,
            'scale': scale,
            'interactive': interactive,
            'conn_df': conn_df
        })
        
        return self  # Allow method chaining
    
    def add_heatmaps(self, matrices_dict, titles_dict=None, color_scales_dict=None, fontsize=None, scale='linear', interactive=False):
        """
        Add multiple heatmaps at once from dictionaries.
        
        Parameters
        ----------
        matrices_dict : dict
            Dictionary of {name: matrix_dataframe}
        titles_dict : dict, optional
            Dictionary of {name: title_string}. If None, auto-generates titles.
        color_scales_dict : dict, optional
            Dictionary of {name: color_scale}. If None, auto-detects based on name.
        fontsize : int, optional
            Font size for all heatmaps. If None, uses default_fontsize.
        scale : str, optional
            Scale for color mapping: 'linear', 'log2', or 'log10' (default: 'linear')
        interactive : bool, optional
            If True, creates interactive heatmaps with scale controls (default: False)
        
        Returns
        -------
        self
            Returns self for method chaining
        
        Examples
        --------
        >>> matrices = {
        ...     'conn_matrix_type': conn_mat,
        ...     'ratio_matrix_type': ratio_mat
        ... }
        >>> hm.add_heatmaps(matrices)
        """
        if titles_dict is None:
            titles_dict = {}
        if color_scales_dict is None:
            color_scales_dict = {}
        
        for name, matrix in matrices_dict.items():
            # Get or generate title
            title = titles_dict.get(name, name.replace('_', ' ').title())
            
            # Get or auto-detect color scale
            if name in color_scales_dict:
                color_scale = color_scales_dict[name]
            else:
                # Auto-detect based on name
                if 'ratio' in name.lower():
                    color_scale = 'orange'
                elif 'transmission' in name.lower() or 'prob' in name.lower():
                    color_scale = 'purple'
                elif 'bodyid' in name.lower():
                    color_scale = 'green'
                else:
                    color_scale = 'purple'
            
            self.add_heatmap(matrix, name, title, color_scale, fontsize, scale, interactive)
        
        return self
    
    def create_all(self):
        """
        Generate all queued heatmaps.
        
        Returns
        -------
        list
            List of created file paths
        """
        if not self.heatmaps:
            print('No heatmaps to create.')
            return []
        
        print(f'Creating {len(self.heatmaps)} heatmap(s)...')
        created_files = []
        
        for hm in self.heatmaps:
            filename = os.path.join(self.output_folder, f"{hm['name']}.html")
            
            # Use interactive version if requested
            if hm.get('interactive', False):
                VisConnMatInteractive(
                    hm['matrix'],
                    filename=filename,
                    title=hm['title'],
                    color_scale=hm['color_scale'],
                    showfig=self.showfig,
                    fontsize=hm['fontsize'],
                    conn_df=hm.get('conn_df')
                )
            else:
                VisConnMat(
                    hm['matrix'],
                    filename=filename,
                    title=hm['title'],
                    color_scale=hm['color_scale'],
                    showfig=self.showfig,
                    fontsize=hm['fontsize'],
                    scale=hm['scale']
            )
            
            created_files.append(filename)
            print(f"  Created: {hm['name']}.html")
        
        print('Done\n')
        self.heatmaps = []  # Clear queue after creation
        return created_files
    
    def clear(self):
        """Clear the heatmap queue without creating them."""
        self.heatmaps = []
        return self


def LogInHemibrain(token,dataset='hemibrain:v1.2.1'): # log in to hemibrain dataset
    '''
    Log in to hemibrain dataset;
    Please provide your own token, which can be obtained from https://neuprint.janelia.org/account
    '''
    # Use TokenManager
    try:
        from .utils.token_manager import token_manager
        token = token_manager.get_token('NEUPRINT_TOKEN', token)
    except ImportError:
        try:
            from src.utils.token_manager import token_manager
            token = token_manager.get_token('NEUPRINT_TOKEN', token)
        except ImportError:
            pass

    client = Client(
        'neuprint.janelia.org',
        dataset = dataset,
        token = token,
    )
    print("Logged in \ndataset: " + dataset)
    return client, dataset

def getCriteriaAndName(requiredNeurons):
    from neuprint import NeuronCriteria as NC
    if requiredNeurons == None:
        criteria = None
        fname = 'ALL'
    elif type(requiredNeurons) != list:
        raise ValueError('requiredNeurons must be a list or None')
    elif type(requiredNeurons[0]) == int: # bodyId
        criteria = NC(bodyId=requiredNeurons)
        fname = str(requiredNeurons[0])
    elif requiredNeurons[0].find('.*') != -1: # instance
        criteria = NC(instance=requiredNeurons)
        fname = requiredNeurons[0].replace('.*','')
    else: # type
        criteria = NC(type=requiredNeurons)
        fname = requiredNeurons[0]
    if requiredNeurons != None and len(requiredNeurons) > 1:
        fname += '_etc'
    return criteria, fname

def _get_coverage_notes(dataset_name):
    '''Known dataset coverage notes (mirrors ComparisonAnalyzer._get_coverage_notes).'''
    notes = {
        'hemibrain': "Central brain only. Missing: optic lobe, ventral nerve cord, subesophageal zone.",
        'male-cns': "Full male CNS including central brain, optic lobes, VNC. Mostly bilateral symmetric.",
        'manc': "Male adult nerve cord (VNC) connectome.",
        'flywire': "Full adult female brain (FAFB). Complete brain coverage with optic lobes.",
        'fafb': "Full adult female brain. Complete brain coverage with optic lobes.",
        'optic-lobe': "Optic lobe only. Missing: central brain, VNC.",
        'banc': "Full brain and VNC connectome.",
    }
    dataset_lower = dataset_name.lower()
    for key, note in notes.items():
        if key in dataset_lower:
            return note
    return "Coverage information not available."

def _build_dataset_metadata(dataset, neuron_df, roi_count_df, client=None):
    '''Build the ``<dataset>_metadata.json`` sidecar from freshly pulled frames.

    Computes the statistics directly from the in-memory tables instead of
    re-querying the server, so every pulled dataset carries identical
    metadata regardless of whether a cross-dataset comparison has ever run
    (previously only ComparisonAnalyzer wrote this file, and only for the
    datasets it compared).
    '''
    import datetime

    total = len(neuron_df)
    if 'type' in neuron_df.columns:
        # 'Unknown' is the untyped placeholder (NeuPrint/BANC convention):
        # counting it as typed reported a misleading 100% coverage.
        type_vals = neuron_df['type'].astype('string').str.strip().fillna('')
        typed = int(((type_vals != '') & (type_vals != 'Unknown')).sum())
    else:
        typed = 0

    def _sum(col):
        return int(neuron_df[col].fillna(0).sum()) if col in neuron_df.columns else 0

    total_pre = _sum('pre')
    total_post = _sum('post')

    # Prefer the server's primary ROI list when the client exposes it; fall
    # back to every ROI present in the long-form roi-count table.
    primary_rois = getattr(client, 'primary_rois', None)
    has_roi_col = 'roi' in roi_count_df.columns
    if primary_rois:
        roi_list = list(primary_rois)
    elif has_roi_col and not roi_count_df.empty:
        roi_list = sorted(set(roi_count_df['roi'].dropna()) - {'NotPrimary'})
    else:
        roi_list = []

    neuron_counts_per_roi = {}
    if roi_list and has_roi_col and not roi_count_df.empty \
            and 'bodyId' in roi_count_df.columns:
        rc = roi_count_df[roi_count_df['roi'].isin(roi_list)]
        if 'pre' in rc.columns and 'post' in rc.columns:
            # A neuron counts toward an ROI when it has any synapse there.
            rc = rc[(rc['pre'].fillna(0) + rc['post'].fillna(0)) > 0]
        counts = rc.groupby('roi')['bodyId'].nunique()
        neuron_counts_per_roi = {roi: int(counts.get(roi, 0)) for roi in roi_list}

    return {
        'dataset': dataset,
        'source': 'neuprint',
        'fetched_at': datetime.datetime.now().isoformat(),
        'neuron_counts': {
            'total': total,
            'typed': typed,
            'untyped': total - typed,
            'type_coverage': typed / total if total else 0,
        },
        'synapse_counts': {
            'total_presynaptic': total_pre,
            'total_postsynaptic': total_post,
            'total': total_pre + total_post,
        },
        'roi_coverage': {
            'roi_list': roi_list,
            'roi_count': len(roi_list),
            'neuron_counts_per_roi': neuron_counts_per_roi,
        },
        'coverage_notes': _get_coverage_notes(dataset),
    }


class DatasetPullCancelled(Exception):
    """Raised when ``pull_dataset`` detects a cancel request mid-download.

    The download stops without writing partial tables, so a dataset's
    ``_neuron_df`` / ``_roi_count_df`` files are never left incomplete; the
    caller (Settings metadata / connections / skeleton pull) treats it as a
    user cancel instead of a failure.
    """


def pull_dataset(dataset, save_path=None, omitNoneType=False, client=None, batch_size=2000, fetch_fn=None, drop_roi_cols=True, progress_callback=None, cancel_event=None, max_workers=1):
    '''
    Download the complete neuron table of a NeuPrint dataset (including
    neurons with type=None) and save it as CSV (neurons) and zstd parquet
    (ROI counts).

    The download is CHUNKED, time-bounded and retried: fetching the whole
    dataset in one ``fetch_neurons(None)`` call has no timeout and no
    progress feedback, so a large dataset stalls for minutes and looks like
    a hang. Instead the bodyIds are listed with one light query, then the
    neuron info is fetched in chunks of ``batch_size`` under
    ``api_call_with_retry`` (timeout + 5 reconnect attempts per chunk) with
    a live progress bar.

    Parameters
    ----------
    dataset : str
        NeuPrint dataset identifier, e.g. 'male-cns:v1.0'.
    save_path : str, optional
        Output path prefix; ``_neuron_df.csv`` and ``_roi_count_df.parquet``
        are written next to it.  By default the per-neuron ROI columns
        ``roiInfo``, ``inputRois`` and ``outputRois`` are dropped from
        ``_neuron_df.csv`` before saving (see ``drop_roi_cols``); the same
        data is always kept long-form in ``_roi_count_df.parquet``.
    omitNoneType : bool
        Drop rows without a type before saving (default False = keep).
    client : object, optional
        NeuPrint client; uses the default client when None.
    batch_size : int
        Neurons fetched per API call (default 2000).
    fetch_fn : callable, optional
        Fetch function with the neuprint signature ``(criteria, client)``
        returning ``(neuron_df, roi_count_df)``. Defaults to the module-level
        ``fetch_neurons``; injectable for tests/adapters.
    drop_roi_cols : bool
        Drop the per-neuron ROI columns ``roiInfo``, ``inputRois`` and
        ``outputRois`` from ``_neuron_df.csv`` before saving (default True).
        These columns are fully derivable from the long-form
        ``_roi_count_df.parquet`` and dominate the neuron CSV size
        (male-cns); set False only when the raw per-neuron columns must be
        stored locally.
    progress_callback : callable, optional
        Callback ``(current, total)`` invoked after each chunk is fetched,
        giving the cumulative number of neurons downloaded out of the total.
        Used by the Settings-tab dataset-metadata pull to drive the
        determinate progress bar; the terminal tqdm bar is always shown.
    cancel_event : threading.Event, optional
        When set, the download stops between chunks and raises
        ``DatasetPullCancelled`` without writing partial tables, so the
        Settings-tab pulls can honour the Cancel button mid-download.
    max_workers : int
        Batches fetched concurrently (1 = sequential).  A long dataset pull
        is a chain of server requests that NeuPrint rate-limits, so a bound
        of a small number of workers (e.g. 4) overlaps the fetches to reduce
        wall time; only raise it when the server tolerates concurrent
        requests.

    Memory: the per-neuron ROI detail columns (``roiInfo``/``inputRois``/
    ``outputRois``) are dropped per chunk and the ROI rows are streamed to a
    temp dir then consolidated, so a large dataset does not accumulate every
    frame in RAM during a long download.

    A ``<dataset>_metadata.json`` sidecar (neuron/synapse counts, ROI
    coverage) is always written next to the tables, computed from the pulled
    frames.
    '''
    if cancel_event is not None and cancel_event.is_set():
        raise DatasetPullCancelled("Dataset pull cancelled.")

    # requires login to hemibrain dataset
    if save_path is None:
        # Go up from src/ to project root, then into datasets/
        dataset_normalized = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
        project_root = os.path.dirname(os.path.dirname(__file__))
        dataset_dir = os.path.join(project_root, "datasets", dataset_normalized)
        save_path = os.path.join(dataset_dir, f"{dataset_normalized}_allneurons")
        # The directory is created only right before the tables are written
        # (below), so a failed or cancelled download leaves no empty
        # datasets/<dataset>/ folder behind.

    from neuprint import NeuronCriteria as NC
    from neuprint import default_client

    try:
        from src.utils.api_utils import api_call_with_retry, APITimeoutError, APIRetryExhaustedError
    except ImportError:
        # Inline fallback when src is not on sys.path (scripts launched
        # without the src prefix).
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

        class APITimeoutError(Exception):
            pass

        class APIRetryExhaustedError(Exception):
            pass

        def api_call_with_retry(func, timeout=60, max_retries=5, retry_delay=2.0,
                                description="API call", on_retry=None, verbose=True):
            import time
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    # shutdown(wait=False): a hung API call must not block the
                    # retry loop (with-block would wait forever).
                    executor = ThreadPoolExecutor(max_workers=1)
                    try:
                        future = executor.submit(func)
                        return future.result(timeout=timeout)
                    finally:
                        executor.shutdown(wait=False)
                except FuturesTimeoutError:
                    last_exc = APITimeoutError(f"{description} timed out after {timeout}s (attempt {attempt}/{max_retries})")
                    if on_retry is not None:
                        on_retry(attempt, last_exc)
                    if attempt < max_retries:
                        time.sleep(retry_delay * (2 ** (attempt - 1)))
                except Exception as e:
                    last_exc = e
                    if on_retry is not None:
                        on_retry(attempt, e)
                    if attempt < max_retries:
                        time.sleep(retry_delay * (2 ** (attempt - 1)))
            raise last_exc or Exception("Unknown error")

    if client is None:
        try:
            client = default_client()
        except RuntimeError as e:
            # No client configured (cache-only context / no token): the
            # neuprint RuntimeError is opaque to users — say what is missing
            # and how to fix it.
            raise RuntimeError(
                f"Dataset '{dataset}' is not available locally and no NeuPrint "
                f"connection exists.\n"
                f"   ({e})\n"
                f"   Set a NEUPRINT token (UI Settings tab or config.json) so the "
                f"dataset tables can be downloaded, or place the prepared tables "
                f"in the datasets/ folder."
            ) from e
    else:
        # NeuronCriteria construction itself requires the neuprint DEFAULT
        # client (the criteria factory is client-bound in this neuprint
        # version), so make the passed client the default for this pull.
        try:
            from neuprint import set_default_client
            set_default_client(client)
        except Exception:
            pass

    # The neuprint fetch function to use (module-level by default). Tests
    # inject a fake here directly — avoiding module-global monkeypatching
    # that races with api_call_with_retry's worker threads.
    fetch_fn = fetch_fn or fetch_neurons

    # 1) Light bodyId-only query (bounded + retried): chunking needs the id
    #    list, and the ids-only response is far smaller than the full table.
    ids_df = api_call_with_retry(
        lambda: client.fetch_custom("MATCH (n:Neuron) RETURN n.bodyId AS bodyId"),
        timeout=120.0,
        max_retries=5,
        retry_delay=5.0,
        description='Neuron list query',
        on_retry=lambda attempt, exc: _tqdm_print(
            f'⚠️ Server not responding (neuron list) — reconnecting, attempt {attempt}/5...'),
        verbose=True,
    )
    all_ids = [int(x) for x in ids_df['bodyId'].tolist()]
    total = len(all_ids)

    # 2) Chunked neuron download with timeout/retry + live progress bar,
    #    optional bounded parallelism, and memory-efficient streaming: the
    #    per-neuron ROI detail columns are dropped per chunk and the ROI rows
    #    are written to a temp dir, so a huge dataset (male-cns ROI table is
    #    millions of rows) does not accumulate every frame in RAM during a
    #    long, slow download.
    import tempfile
    import shutil
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

    neuron_frames = []
    n_batches = (total + batch_size - 1) // batch_size
    # Keep this bar on stdout, alongside NeuronBridgeFinder._vprint().  The
    # UI runner can then preserve the clear/write/refresh sequence as one
    # ordered stream instead of racing stdout against stderr.
    progress = tqdm(
        total=total,
        desc='Downloading neuron list',
        unit='neuron',
        leave=False,
        file=sys.stdout,
    )
    stream_dir = tempfile.mkdtemp(prefix='drocat_pull_')
    roi_batch_paths = []

    def _record_chunk(i, ndf, rdf):
        """Consume one fetched chunk (drop ROI detail cols, stream ROI rows)."""
        if ndf is not None and not ndf.empty:
            if drop_roi_cols:
                ndf = ndf.drop(
                    columns=[c for c in ('roiInfo', 'inputRois', 'outputRois')
                             if c in ndf.columns]
                )
            neuron_frames.append(ndf)
        if rdf is not None and not rdf.empty:
            path = os.path.join(stream_dir, f'roi_{i:06d}.parquet')
            rdf.to_parquet(path, index=False)
            roi_batch_paths.append(path)

    def _fetch_chunk(i):
        """Fetch one chunk; never raises (returns a failure flag)."""
        chunk = all_ids[i:i + batch_size]
        batch_num = i // batch_size + 1
        try:
            ndf, rdf = api_call_with_retry(
                lambda c=chunk: fetch_fn(NC(bodyId=c), client=client),
                timeout=180.0,
                max_retries=5,
                retry_delay=5.0,
                description=f'Neuron batch {batch_num}/{n_batches}',
                on_retry=lambda attempt, exc, b=batch_num: _tqdm_print(
                    f'⚠️ Server not responding (neuron batch {b}/{n_batches}) '
                    f'— reconnecting, attempt {attempt}/5...'),
                verbose=True,
            )
            return i, ndf, rdf, False
        except (APITimeoutError, APIRetryExhaustedError) as e:
            _tqdm_print(
                f'⚠️ Neuron batch {batch_num}/{n_batches} failed after retries: {e}'
            )
            return i, None, None, True

    try:
        current = 0
        if max_workers and max_workers > 1:
            # Bounded in-flight parallel fetch (mirrors build_connection_cache):
            # appends/streaming stay in this single thread; only the network
            # fetches overlap.  Cancellation stops new submissions; in-flight
            # fetches finish in the background and their results are discarded.
            batch_indices = iter(range(0, total, batch_size))
            executor = ThreadPoolExecutor(max_workers=max_workers)
            try:
                pending = {}
                for _ in range(max_workers):
                    i = next(batch_indices, None)
                    if i is None:
                        break
                    pending[executor.submit(_fetch_chunk, i)] = i
                while pending:
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                    if not done:
                        continue
                    for fut in done:
                        i = pending.pop(fut)
                        _, ndf, rdf, failed = fut.result()
                        if not failed:
                            _record_chunk(i, ndf, rdf)
                            n = len(all_ids[i:i + batch_size])
                            current += n
                            progress.update(n)
                            if progress_callback is not None:
                                progress_callback(min(current, total), total)
                        next_i = next(batch_indices, None)
                        if (
                            next_i is not None
                            and (cancel_event is None or not cancel_event.is_set())
                        ):
                            pending[executor.submit(_fetch_chunk, next_i)] = next_i
                    if cancel_event is not None and cancel_event.is_set():
                        break
            finally:
                executor.shutdown(wait=False)
        else:
            for i in range(0, total, batch_size):
                if cancel_event is not None and cancel_event.is_set():
                    raise DatasetPullCancelled("Dataset pull cancelled.")
                _, ndf, rdf, failed = _fetch_chunk(i)
                if not failed:
                    _record_chunk(i, ndf, rdf)
                    n = len(all_ids[i:i + batch_size])
                    current += n
                    progress.update(n)
                    if progress_callback is not None:
                        progress_callback(min(current, total), total)
        if cancel_event is not None and cancel_event.is_set():
            raise DatasetPullCancelled("Dataset pull cancelled.")
        # Consolidate the streamed ROI rows HERE (while the temp files still
        # exist); the finally-block removes the temp dir right after.
        if roi_batch_paths:
            import polars as pl
            roi_count_df = pl.concat(
                [pl.scan_parquet(p) for p in roi_batch_paths],
                how='vertical_relaxed',
            ).collect().to_pandas()
        else:
            roi_count_df = pd.DataFrame()
    finally:
        progress.close()
        shutil.rmtree(stream_dir, ignore_errors=True)

    if not neuron_frames:
        raise RuntimeError(
            f'Failed to download neurons for {dataset}: every batch failed after retries '
            f'(server unreachable). Check the connection and re-run.'
        )
    neuron_df = pd.concat(neuron_frames, ignore_index=True)

    if omitNoneType:
        # delete rows with type is empty
        neuron_df = neuron_df[neuron_df['type'].notna()]
    _tqdm_print(f'Pulled {len(neuron_df)} neurons from {dataset}')
    _tqdm_print('Writing to', save_path, end='...')
    # First disk side effect of the pull: create the destination folder only
    # once the data is in hand (see the save_path resolution above).
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    # write neuron table as csv; the ROI-count table is numeric long-form
    # (bodyId/roi/count columns), so a zstd parquet is ~5x smaller than the
    # equivalent CSV and loads without schema inference
    neuron_df.to_csv(save_path + '_neuron_df.csv', index=True)
    roi_count_df.to_parquet(save_path + '_roi_count_df.parquet', index=False, compression='zstd')
    stale_roi_csv = save_path + '_roi_count_df.csv'
    if os.path.exists(stale_roi_csv):
        # A previous pull wrote CSV; drop it so the parquet is the only copy
        os.remove(stale_roi_csv)

    # Metadata sidecar: computed from the frames just saved, so every pulled
    # dataset gets its statistics without waiting for a cross-dataset
    # comparison run to touch it.
    metadata = _build_dataset_metadata(dataset, neuron_df, roi_count_df, client)
    meta_path = save_path
    if meta_path.endswith('_allneurons'):
        meta_path = meta_path[:-len('_allneurons')]
    meta_file = meta_path + '_metadata.json'
    with open(meta_file, 'w', encoding='utf-8') as mf:
        json.dump(metadata, mf, indent=2, default=str)
    _tqdm_print('Done! (metadata saved to', meta_file + ')')

def getNeurons(requiredNeurons, dataset='hemibrain:v1.2.1', custom_group_names=None, client=None, verbose=True, search_columns='auto', search_info_sink=None):
    '''get neurons locally from a given dataset
    
    Parameters
    ----------
    requiredNeurons : list, dict, or None
        Neuron query. Supports multiple formats:
        
        Legacy formats:
        - None: Return all neurons
        - list of types/instances: ['aMe12', 'Mi1']
        - list of regex patterns: ['aMe.*', 'Mi.*']
        - list of bodyIds: [12345, 67890]
        - nested lists for grouping: ['A', 'B', ['C', 'D']]
        
        Dict filter format:
        - {'type': {'contains': 'DN'}}  # Types containing 'DN'
        - {'type': {'startswith': ['aMe', 'Mi']}}  # Types starting with 'aMe' or 'Mi'
        - {'type': {'endswith': '_R'}}  # Types ending with '_R'
        - {'type': {'regex': 'DN[a-z]\\d+'}}  # Types matching regex
        - {'bodyId': [12345, 67890]}  # Specific bodyIds
        - {'type': {'contains': 'DN'}, 'instance': {'endswith': '_R'}}  # Combined filters (AND)
        
    dataset : str
        Dataset name
    custom_group_names : list, optional
        Custom names for groups when using nested lists
    client : object, optional
        Client object (NeuPrint or FlyWire) for direct fetching if local dataset missing
    verbose : bool, optional
        Whether to print progress messages (default: True)
    search_columns : str, optional
        Which columns to search when resolving neuron names:
        'auto' (default) searches the prioritized identity/type/taxonomy
        columns: bodyId -> type -> instance -> flywireType -> hemibrainType
        -> mancType -> other *Type -> class/subclass/superclass; 'type', 'instance'
        and 'bodyId' restrict the search to that single column.
    search_info_sink : list, optional
        Internal collector for the shared resolver's match metadata. When
        supplied, one dictionary is appended for each name resolution; this
        lets the analysis run record when a query was resolved after the
        identity columns without changing the public return tuple.
        
    Returns
    -------
    neuron_df : pd.DataFrame
        DataFrame of neurons with 'custom_group' column for nested list groups
    roi_count_df : pd.DataFrame
        ROI count DataFrame
    auto_name : str
        Auto-generated name for the neuron set
    criteria : NeuronCriteria
        Neuprint criteria object
        
    Examples
    --------
    >>> # Legacy format
    >>> neuron_df, roi_df, name, criteria = getNeurons(['aMe.*'], dataset='hemibrain:v1.2.1')
    
    >>> # Dict filter format (recommended for complex queries)
    >>> neuron_df, _, _, _ = getNeurons({'type': {'contains': 'DN'}}, dataset='hemibrain:v1.2.1')
    >>> neuron_df, _, _, _ = getNeurons({'type': {'startswith': ['aMe', 'Mi']}}, dataset='hemibrain:v1.2.1')
    '''
    from neuprint import NeuronCriteria as NC

    # Check if requiredNeurons is a dict-based filter
    if isinstance(requiredNeurons, dict):
        # Use NeuronFilter for dict-based queries
        try:
            from utils.neuron_filter import NeuronFilter
        except ImportError:
            from src.utils.neuron_filter import NeuronFilter
        
        nf = NeuronFilter(requiredNeurons)
        
        # Load the dataset first
        if is_local_connectome_dataset(dataset):
            ndf = _get_neuron_df(dataset, verbose=verbose)
        else:
            # Standard neuprint datasets
            dataset_normalized = dataset.replace(':','_').replace('.','_')
            project_root = os.path.dirname(os.path.dirname(__file__))
            dataset_dir = os.path.join(project_root, "datasets", dataset_normalized)
            if os.path.exists(dataset_dir):
                dataset_path_body = os.path.join(dataset_dir, f"{dataset_normalized}_allneurons")
            else:
                dataset_path_body = os.path.join(project_root, "datasets", f"{dataset_normalized}_allneurons")
            ndf, roi_df = _get_cached_neuron_df(dataset_normalized, dataset_path_body)
        
        # Apply filter
        matched_df = nf.apply(ndf)
        
        # Generate auto_name from filter description
        auto_name = nf.describe().replace(' ', '_')[:30]  # Truncate for file names
        if len(auto_name) > 20:
            auto_name = 'filter_result'
        
        if verbose:
            print(f"Found {len(matched_df)} neurons matching filter: {nf.describe()}")
        
        return matched_df, pd.DataFrame(), auto_name, None
    
    # Original getNeurons logic for legacy formats
    from neuprint import NeuronCriteria as NC
    
    # FAFB and standalone BANC use their exact local-release tables.
    if is_local_connectome_dataset(dataset):
        # Try to use local data first
        try:
            import fafb_utils
            # Go up from src/ to project root, then into datasets/
            project_root = os.path.dirname(os.path.dirname(__file__))
            data_dir = resolve_flywire_dataset_dir(project_root, dataset)

            if data_dir is not None:
                # Use dataset name in message instead of hardcoded "FAFB"
                dataset_short = dataset.split('_')[1] if '_' in dataset else dataset
                
                # Check if already cached
                cache_prefix = "banc" if is_banc_dataset(dataset) else "flywire"
                cache_key = f"{cache_prefix}_{dataset}"
                if cache_key in _NEURON_DF_CACHE:
                    if verbose:
                        print(f"Using cached {dataset_short} data...")
                    full_neuron_df = _NEURON_DF_CACHE[cache_key]['neuron_df'].copy()
                else:
                    if verbose:
                        print(f"Loading {dataset_short} data from {data_dir}...")
                    neuron_file, _ = fafb_utils.prepare_flywire_data(data_dir)
                    
                    # Load using fast loader (polars if available)
                    full_neuron_df = _load_dataframe_fast(neuron_file, dtype_overrides={'bodyId': str})
                    
                    # Ensure bodyId is string
                    normalize_flywire_id_columns(full_neuron_df, ['bodyId'])
                    
                    # Cache for future calls
                    _NEURON_DF_CACHE[cache_key] = {'neuron_df': full_neuron_df}
                    if verbose:
                        print(f"  ✓ Loaded {len(full_neuron_df):,} neurons (cached for reuse)")
                
                # Handle special cases: None = all neurons, [] = all typed neurons
                if requiredNeurons is None:
                    return full_neuron_df, pd.DataFrame(), 'ALL_FAFB', None
                
                if len(requiredNeurons) == 0:
                    # Empty list = all typed neurons (neurons with non-null type)
                    if 'type' in full_neuron_df.columns:
                        typed_df = full_neuron_df[full_neuron_df['type'].notna() & (full_neuron_df['type'] != '')].copy()
                        if verbose:
                            print(f"Returning all typed neurons: {len(typed_df):,} neurons")
                        return typed_df, pd.DataFrame(), 'allneurons', None
                    else:
                        # No type column, return all
                        return full_neuron_df, pd.DataFrame(), 'allneurons', None
                
                # Filter based on requiredNeurons
                selected_dfs = []
                
                # Handle nested lists (custom groups)
                # Structure: [item1, item2, [group_item1, group_item2]]
                
                # Flatten for simple filtering first
                flat_list = []
                custom_groups = {} # Map bodyId/type -> group_name
                
                group_idx = 0
                for i, item in enumerate(requiredNeurons):
                    if isinstance(item, list):
                        # It's a custom group
                        group_name = custom_group_names[group_idx] if custom_group_names and group_idx < len(custom_group_names) else f"Group_{group_idx+1}"
                        group_idx += 1
                        for subitem in item:
                            flat_list.append(subitem)
                            custom_groups[str(subitem)] = group_name
                    else:
                        flat_list.append(item)
                
                # Use the shared priority search for each item:
                # bodyId -> type -> instance -> useful type/taxonomy fields.
                bodyId_alltypes = full_neuron_df['bodyId'].astype(str).tolist()
                cached_search = _get_cached_neuron_search(dataset)

                fafb_summary = (
                    _ResolutionSummary(len(flat_list))
                    if verbose and len(flat_list) > _QUERY_PRINT_CAP
                    else None
                )
                for item in flat_list:
                    item_bodyIds, search_info = _resolve_single_neuron(
                        item,
                        full_neuron_df,
                        bodyId_alltypes,
                        dataset=dataset,
                        cached_search=cached_search,
                        verbose=verbose and fafb_summary is None,
                        search_columns=search_columns,
                        search_info_sink=search_info_sink,
                    )
                    if fafb_summary is not None:
                        fafb_summary.add(item, item_bodyIds, search_info)
                    if item_bodyIds:
                        # Get matching rows
                        item_df = full_neuron_df[full_neuron_df['bodyId'].astype(str).isin([str(b) for b in item_bodyIds])].copy()
                        selected_dfs.append(item_df)
                if fafb_summary is not None:
                    fafb_summary.report()
                
                if selected_dfs:
                    filtered_df = pd.concat(selected_dfs).drop_duplicates(subset=['bodyId'])
                    
                    # Apply custom groups
                    if custom_groups:
                        filtered_df['custom_group'] = filtered_df.apply(
                            lambda row: custom_groups.get(str(row['bodyId'])) or custom_groups.get(str(row['type'])), axis=1
                        )
                    
                    # Generate auto_name based on requiredNeurons
                    if len(requiredNeurons) == 1:
                        if isinstance(requiredNeurons[0], list):
                             # Single group
                             items_str = [str(x).replace('.*', '') for x in requiredNeurons[0]]
                             auto_name = items_str[0] + '_etc' if len(items_str) > 1 else items_str[0]
                        else:
                             auto_name = str(requiredNeurons[0]).replace('.*', '')
                    elif len(requiredNeurons) > 1:
                        first_item = requiredNeurons[0]
                        if isinstance(first_item, list):
                             items_str = [str(x).replace('.*', '') for x in first_item]
                             first_name = items_str[0] + '_etc' if len(items_str) > 1 else items_str[0]
                        else:
                             first_name = str(first_item).replace('.*', '')
                        auto_name = first_name + '_etc'
                    else:
                        auto_name = "fafb_selection"

                    return filtered_df, pd.DataFrame(), auto_name, None
                else:
                    print("Warning: No neurons found matching criteria in local FAFB data.")
                    return pd.DataFrame(), pd.DataFrame(), "empty", None
                    
        except ImportError:
            print("Warning: fafb_utils not found.")
        except Exception as e:
            print(f"Warning: Error loading local FAFB data: {e}.")

        print("Warning: FlyWire API fetching has been removed. Please ensure local data is available.")
        return pd.DataFrame(), pd.DataFrame(), "error", None

    if requiredNeurons == None:
        criteria = None
        auto_name = 'ALL'
        neuron_df, roi_count_df = fetch_neurons(criteria, client=client)
        return neuron_df, roi_count_df, auto_name, criteria
    if type(requiredNeurons) != list:
        requiredNeurons = [requiredNeurons]
    
    dataset_normalized, dataset_path_body = _ensure_local_dataset_files(
        dataset,
        client=client,
        verbose=verbose,
    )

    # Use in-memory cache for neuron DataFrames (avoids repeated CSV reads)
    ndf_alltypes, rdf_alltypes = _get_cached_neuron_df(dataset_normalized, dataset_path_body)
    bodyId_alltypes = ndf_alltypes['bodyId'].tolist()
    cached_search = _get_cached_neuron_search(dataset)
    
    if len(requiredNeurons) == 0:
        # Empty list = all typed neurons (neurons with non-null type)
        if 'type' in ndf_alltypes.columns:
            neuron_df = ndf_alltypes[ndf_alltypes['type'].notna() & (ndf_alltypes['type'] != '')].copy()
            # Also filter roi_count_df to match
            roi_count_df = rdf_alltypes[rdf_alltypes.index.isin(neuron_df['bodyId'])].copy() if not rdf_alltypes.empty else rdf_alltypes
            if verbose:
                print(f"Returning all typed neurons: {len(neuron_df):,} neurons")
        else:
            neuron_df = ndf_alltypes
            roi_count_df = rdf_alltypes
        auto_name = 'allneurons'
        bodyId_list = neuron_df['bodyId'].tolist()
    else:
        # Check if we have nested lists for custom grouping
        has_nested = any(isinstance(item, list) for item in requiredNeurons)
        
        if has_nested:
            # Process with custom grouping
            bodyId_list = []
            group_names = []
            group_custom_idx = 0

            # Same print cap as the flat list: count every item across all
            # groups and switch to the totals summary past the threshold.
            total_nested_items = sum(
                len(item) if isinstance(item, list) else 1
                for item in requiredNeurons
            )
            summary = (
                _ResolutionSummary(total_nested_items)
                if verbose and total_nested_items > _QUERY_PRINT_CAP
                else None
            )

            for i, requiredNeuron in enumerate(requiredNeurons):
                if isinstance(requiredNeuron, list):
                    # Nested list - create custom group
                    group_bodyIds = []
                    group_items = []

                    for item in requiredNeuron:
                        group_items.append(str(item).replace('.*', ''))
                        item_bodyIds, item_info = _resolve_single_neuron(
                            item,
                            ndf_alltypes,
                            bodyId_alltypes,
                            dataset=dataset,
                            cached_search=cached_search,
                            verbose=verbose and summary is None,
                            search_columns=search_columns,
                            search_info_sink=search_info_sink,
                        )
                        if summary is not None:
                            summary.add(item, item_bodyIds, item_info)
                        group_bodyIds.extend(item_bodyIds)
                    
                    # Generate group name
                    if custom_group_names and group_custom_idx < len(custom_group_names):
                        group_name = custom_group_names[group_custom_idx]
                    else:
                        # Auto-generate name from first item
                        if len(group_items) == 1:
                            group_name = group_items[0]
                        else:
                            group_name = group_items[0] + '_etc'
                    
                    group_names.append(group_name)
                    bodyId_list.extend(group_bodyIds)
                    group_custom_idx += 1
                    
                    print(f'Custom group "{group_name}": {len(group_bodyIds)} neurons from {len(requiredNeuron)} items')
                else:
                    # Regular item
                    item_bodyIds, item_info = _resolve_single_neuron(
                        requiredNeuron,
                        ndf_alltypes,
                        bodyId_alltypes,
                        dataset=dataset,
                        cached_search=cached_search,
                        verbose=verbose and summary is None,
                        search_columns=search_columns,
                        search_info_sink=search_info_sink,
                    )
                    if summary is not None:
                        summary.add(requiredNeuron, item_bodyIds, item_info)
                    bodyId_list.extend(item_bodyIds)
                    group_names.append(str(requiredNeuron).replace('.*', ''))

            if summary is not None:
                summary.report()
            
            # Create auto_name from group names
            if len(group_names) == 1:
                auto_name = group_names[0]
            elif len(group_names) == 2:
                auto_name = '_'.join(group_names)
            else:
                auto_name = group_names[0] + '_etc'
            
            # Ensure bodyId_list type matches DataFrame's bodyId column type for .isin() to work
            if bodyId_list and len(ndf_alltypes) > 0:
                sample_df_bid = ndf_alltypes['bodyId'].iloc[0]
                if isinstance(sample_df_bid, (int, np.integer)):
                    bodyId_list = [int(b) if isinstance(b, str) and str(b).isdigit() else b for b in bodyId_list]
                else:
                    bodyId_list = [str(b) for b in bodyId_list]
            
            # Build neuron_df with custom_group column
            neuron_df = ndf_alltypes[ndf_alltypes['bodyId'].isin(bodyId_list)].copy()
            roi_count_df = rdf_alltypes[rdf_alltypes['bodyId'].isin(bodyId_list)]
            
            # Add custom_group column by matching original type or creating merged type
            neuron_df['custom_group'] = neuron_df['type']  # Default to original type
            
            # Reassign custom groups for nested list items. This pass only
            # re-resolves the same items to map group names onto bodyIds, so
            # it never prints — the resolution outcome was already reported
            # (per chip under the cap, or by the summary above).
            group_custom_idx = 0
            for i, requiredNeuron in enumerate(requiredNeurons):
                if isinstance(requiredNeuron, list):
                    # Get bodyIds for this custom group
                    group_bodyIds = []
                    for item in requiredNeuron:
                        item_bodyIds, _ = _resolve_single_neuron(
                            item,
                            ndf_alltypes,
                            bodyId_alltypes,
                            dataset=dataset,
                            cached_search=cached_search,
                            verbose=False,
                            search_columns=search_columns,
                            search_info_sink=search_info_sink,
                        )
                        group_bodyIds.extend(item_bodyIds)
                    
                    # Ensure type consistency for .isin() matching
                    if group_bodyIds and len(ndf_alltypes) > 0:
                        sample_df_bid = ndf_alltypes['bodyId'].iloc[0]
                        if isinstance(sample_df_bid, (int, np.integer)):
                            group_bodyIds = [int(b) if isinstance(b, str) and str(b).isdigit() else b for b in group_bodyIds]
                        else:
                            group_bodyIds = [str(b) for b in group_bodyIds]
                    
                    # Assign custom group name
                    if custom_group_names and group_custom_idx < len(custom_group_names):
                        group_name = custom_group_names[group_custom_idx]
                    else:
                        items_str = [str(item).replace('.*', '') for item in requiredNeuron]
                        group_name = items_str[0] + '_etc' if len(items_str) > 1 else items_str[0]
                    
                    neuron_df.loc[neuron_df['bodyId'].isin(group_bodyIds), 'custom_group'] = group_name
                    group_custom_idx += 1
            
        else:
            # Original logic for flat list
            bodyId_list = []
            # Beyond the print cap one summary line replaces the per-chip
            # prints (219 chips used to print 219 "Found 1 neurons" lines).
            summary = (
                _ResolutionSummary(len(requiredNeurons))
                if verbose and len(requiredNeurons) > _QUERY_PRINT_CAP
                else None
            )
            for i, requiredNeuron in enumerate(requiredNeurons):
                if i == 0:
                    auto_name = str(requiredNeuron).replace('.*','')
                elif i == 1:
                    auto_name += '_etc'

                item_bodyIds, item_info = _resolve_single_neuron(
                    requiredNeuron,
                    ndf_alltypes,
                    bodyId_alltypes,
                    dataset=dataset,
                    cached_search=cached_search,
                    verbose=verbose and summary is None,
                    search_columns=search_columns,
                    search_info_sink=search_info_sink,
                )
                if summary is not None:
                    summary.add(requiredNeuron, item_bodyIds, item_info)
                bodyId_list.extend(item_bodyIds)
            if summary is not None:
                summary.report()

            # Ensure bodyId_list type matches DataFrame's bodyId column type for .isin() to work
            if bodyId_list and len(ndf_alltypes) > 0:
                sample_df_bid = ndf_alltypes['bodyId'].iloc[0]
                if isinstance(sample_df_bid, (int, np.integer)):
                    bodyId_list = [int(b) if isinstance(b, str) and str(b).isdigit() else b for b in bodyId_list]
                else:
                    bodyId_list = [str(b) for b in bodyId_list]

            neuron_df = ndf_alltypes[ndf_alltypes['bodyId'].isin(bodyId_list)]
            roi_count_df = rdf_alltypes[rdf_alltypes['bodyId'].isin(bodyId_list)]
    
    # Create NeuronCriteria only if client is available (skip in cache-only mode)
    criteria = None
    try:
        criteria = NC(bodyId=bodyId_list)
    except RuntimeError:
        # No default client set (cache-only mode) - criteria not needed for local data
        pass
    return neuron_df, roi_count_df, auto_name, criteria

def _process_single_neuron(
    requiredNeuron,
    ndf_alltypes,
    bodyId_alltypes,
    verbose=True,
    search_columns='auto',
):
    """Resolve one neuron query using the shared dataframe contract.

    The public helper is retained for notebook/backward compatibility.  The
    implementation is intentionally thin: cache-backed callers use
    ``resolve_neuron_query`` directly, while dataframe callers use the same
    ``resolve_dataframe_query`` fallback.
    """
    try:
        from src.neuron_search import resolve_dataframe_query
    except ImportError:  # pragma: no cover - ``src/`` on sys.path imports
        from neuron_search import resolve_dataframe_query
    resolved_ids, resolved_info = resolve_dataframe_query(
        ndf_alltypes,
        requiredNeuron,
        search_columns=search_columns,
        verbose=verbose,
    )
    if verbose:
        if resolved_ids:
            print(
                f'Found {len(resolved_ids)} neurons for "{requiredNeuron}" '
                f'(dataframe search; first: '
                f'{resolved_info.get("matched_column") or "metadata"})'
            )
        else:
            print(
                f'\033[33m"{requiredNeuron}" not found in any column, '
                'please check your input (skipped)\033[0m'
            )
    return resolved_ids, resolved_info

def removeSearchedNeurons(conn_df,searchedNeurons,exempt_neurons=None):
    '''remove neurons on searched layers, except those in exempt_neurons'''
    neurons_post = conn_df['bodyId_post'].unique()
    
    # Identify neurons to remove: those in searchedNeurons
    to_remove = np.intersect1d(neurons_post, searchedNeurons, assume_unique=True)
    
    # If exempt_neurons provided, keep them even if they are in searchedNeurons
    if exempt_neurons is not None and len(exempt_neurons) > 0:
        # Remove exempt neurons from the to_remove list
        to_remove = np.setdiff1d(to_remove, exempt_neurons, assume_unique=True)
        
    df = conn_df[~conn_df['bodyId_post'].isin(to_remove)]
    return df

def Conn2FullMat(source_df,target_df,conn_df,conn_type,weight_col='weight'): 
    '''convert connection table (conn_df) to a full connection matrix (keep zero connections)'''
    # Append type to bodyId for row/column names if available
    if 'type' in source_df.columns:
        sbodyId = [f"{row.bodyId}_{row.type}" for _, row in source_df.iterrows()]
    else:
        sbodyId = source_df.bodyId.tolist()
        
    if 'type' in target_df.columns:
        tbodyId = [f"{row.bodyId}_{row.type}" for _, row in target_df.iterrows()]
    else:
        tbodyId = target_df.bodyId.tolist()
    if 'type' in source_df.columns:
        stype = source_df.type.unique().tolist()
    else:
        # No type annotation: fall back to bodyId labels so the type-level
        # matrix mirrors the bodyId-level one.
        stype = [str(x) for x in source_df.bodyId.unique().tolist()]
    if 'type' in target_df.columns:
        ttype = target_df.type.unique().tolist()
    else:
        ttype = [str(x) for x in target_df.bodyId.unique().tolist()]
    sbodyId.sort()
    tbodyId.sort()
    stype.sort()
    ttype.sort()
    # Convert bodyId to strings to ensure consistent treatment as labels, not numbers
    sbodyId = [str(x) for x in sbodyId]
    tbodyId = [str(x) for x in tbodyId]
    cmat_bodyId = pd.DataFrame(data=np.zeros([len(sbodyId),len(tbodyId)],dtype=int),index=sbodyId,columns=tbodyId)
    cmat_type = pd.DataFrame(data=np.zeros([len(stype),len(ttype)],dtype=int),index=stype,columns=ttype)
    # Create mappings for bodyId lookup
    source_map = {}
    if 'type' in source_df.columns:
        for _, row in source_df.iterrows():
            source_map[str(row.bodyId)] = f"{row.bodyId}_{row.type}"
    else:
        for _, row in source_df.iterrows():
            source_map[str(row.bodyId)] = str(row.bodyId)
            
    target_map = {}
    if 'type' in target_df.columns:
        for _, row in target_df.iterrows():
            target_map[str(row.bodyId)] = f"{row.bodyId}_{row.type}"
    else:
        for _, row in target_df.iterrows():
            target_map[str(row.bodyId)] = str(row.bodyId)

    for i in conn_df.index:
        raw_pre = str(conn_df.at[i,'bodyId_pre'])
        raw_post = str(conn_df.at[i,'bodyId_post'])
        
        bpre = source_map.get(raw_pre, raw_pre)
        bpost = target_map.get(raw_post, raw_post)
        
        if bpre in cmat_bodyId.index and bpost in cmat_bodyId.columns:
            bweight = conn_df.at[i,weight_col]
            cmat_bodyId.at[bpre,bpost] = bweight
    for i in conn_type.index:
        tpre  = conn_type.at[i,'type_pre']
        tpost = conn_type.at[i,'type_post']
        tweight = conn_type.at[i,weight_col]
        cmat_type.at[tpre,tpost] = tweight
    return cmat_bodyId,cmat_type

def calRC(cmat,threshold=0):
    '''calculate row and column sums of a connection matrix'''
    n_row,n_col = cmat.shape
    sourceN = [0]*n_col 
    targetN = [0]*n_row
    sum_col = [0]*n_col
    sum_row = [0]*n_row
    for i in range(n_row):
        for j in range(n_col):
            val = cmat.iat[i,j]
            sum_row[i] += val
            sum_col[j] += val
            if val > threshold:
                targetN[i] += 1
                sourceN[j] += 1
    cmat_new = pd.DataFrame(np.insert(cmat.values, len(cmat.index), values=sourceN, axis=0))
    cmat_new = pd.DataFrame(np.insert(cmat_new.values, len(cmat_new.index), values=sum_col, axis=0))
    cmat_new.columns = cmat.columns
    cmat_new.index = list(cmat.index) + ['sourceN','sum_col']
    cmat_new.insert(loc=len(cmat.columns),column='targetN',value=targetN+[0,0])
    cmat_new.insert(loc=len(cmat.columns)+1,column='sum_row',value=sum_row+[0,sum(sum_row)])
    return cmat_new

def filtMat(cmat,axis=0,filt_range=None,by='MR'): 
    '''identify columns whose maximums are in the range'''
    if filt_range is None:
        filt_range = [0, 1]
    if by == 'MR': # maximum ratio
        nval = cmat.shape # nval = (n_row, n_col)
        criterion = [1]*nval[1-axis]
        maxVal = cmat.max(axis=axis)
        if filt_range[0] != filt_range[1]:
            for j in range(nval[1-axis]):
                if maxVal[j] <= filt_range[0] or maxVal[j] > filt_range[1]: # left open, right closed interval
                    criterion[j] = 0
        else:
            for j in range(nval[1-axis]):
                if maxVal[j] != filt_range[0]: # left open, right closed interval
                    criterion[j] = 0
        if axis == 0:
            cmat_new = pd.DataFrame(np.insert(cmat.values, len(cmat.index), values=criterion, axis=0))
            cmat_new.index = list(cmat.index) + ['sourceCriterion']
            cmat_new.columns = cmat.columns
            cmat_new = cmat_new.loc[:,cmat_new.loc['sourceCriterion'] == 1]
            cmat_new = cmat_new.iloc[:-1,:]
        elif axis == 1:
            cmat_new = cmat.copy()
            cmat_new.insert(loc=len(cmat.columns), column='targetCriterion', value=criterion)
            cmat_new = cmat_new.loc[cmat_new['targetCriterion'] == 1,:]
            cmat_new = cmat_new.iloc[:,:-1]
    elif by == 'N': # synapse number
        cmat_t = calRC(cmat) # new connection matrix
        if axis == 0:
            if filt_range[0] != None and filt_range[1] != None:
                cmat_t = cmat_t.loc[:,cmat_t.loc['sourceN'] >= filt_range[0]]
                cmat_t = cmat_t.loc[:,cmat_t.loc['sourceN'] <= filt_range[1]]
            elif filt_range[0] == None:
                cmat_t = cmat_t.loc[:,cmat_t.loc['sourceN'] <= filt_range[1]]
            elif filt_range[1] == None:
                cmat_t = cmat_t.loc[:,cmat_t.loc['sourceN'] >= filt_range[0]]
            cmat_t = cmat_t.iloc[:-2,:]
        elif axis == 1:
            if filt_range[0] != None and filt_range[1] != None:
                cmat_t = cmat_t.loc[cmat_t['targetN'] >= filt_range[0],:]
                cmat_t = cmat_t.loc[cmat_t['targetN'] <= filt_range[1],:]
            elif filt_range[0] == None:
                cmat_t = cmat_t.loc[cmat_t['targetN'] <= filt_range[1],:]
            elif filt_range[1] == None:
                cmat_t = cmat_t.loc[cmat_t['targetN'] >= filt_range[0],:]
            cmat_t = cmat_t.iloc[:,:-2]
        cmat_new = cmat_t
    return cmat_new

def stMat(mat,axis=0):
    '''standardize matrix by row or column'''
    matt = calRC(mat)
    rowN,colN = matt.shape
    if axis == 0: # standardize by column
        for i in range(rowN-2):
            for j in range(colN-2):
                matt.iat[i,j] /= matt.iat[-1,j]
    elif axis == 1: # standardize by row
        for i in range(rowN-2):
            for j in range(colN-2):
                matt.iat[i,j] /= matt.iat[i,-1]
    return matt.iloc[:-2,:-2]

def VisConnMat(cmat,filename,title='',color_scale=None,showfig=True,fontsize=12,scale='linear'): 
    if color_scale is None:
        color_scale = [[0, 'rgb(255,255,255)'], [1, 'rgb(104,55,164)']]
    '''visualize connection matrix with enhanced labels and hover information
    
    Parameters
    ----------
    cmat : pd.DataFrame
        Connection matrix to visualize
    filename : str
        Output HTML filename
    title : str, optional
        Title for the heatmap
    color_scale : list, optional
        Plotly color scale
    showfig : bool, optional
        Whether to open in browser
    fontsize : int, optional
        Font size for labels
    scale : str, optional
        Scale for color mapping: 'linear', 'log2', or 'log10' (default: 'linear')
    '''
    
    # Determine the metric type from the title or filename
    metric_type = 'synapses'  # default
    if 'ratio' in title.lower() or 'ratio' in filename.lower():
        metric_type = 'ratio'
    elif 'transmission' in title.lower() or 'probability' in title.lower():
        metric_type = 'probability'
    
    # Check if this is a large matrix (optimization for performance)
    is_large = cmat.shape[0] > 100 or cmat.shape[1] > 100
    is_very_large = cmat.shape[0] > 500 or cmat.shape[1] > 500
    
    # Calculate sparsity for additional optimization decisions
    sparsity = (cmat.values == 0).sum() / cmat.size
    is_sparse = sparsity > 0.7  # More than 70% zeros
    
    # Deep optimization: For extremely large matrices, consider showing only non-zero entries
    # This creates a scatter plot instead of full heatmap for massive size reduction
    # DISABLED FOR NOW - needs more work to properly handle all cases
    use_scatter_mode = False  # is_very_large and is_sparse and cmat.size > 250000
    
    # if use_scatter_mode:
    #     print(f"  ⚡ Ultra-optimization: Using scatter mode for {cmat.shape[0]}×{cmat.shape[1]} sparse matrix")
    
    # Apply scale transformation to data
    z_data = cmat.values.copy()
    scale_label = ''
    
    if scale == 'log2':
        # Apply log2 transformation (add 1 to avoid log(0))
        z_data = np.log2(z_data + 1)
        scale_label = ' (log2)'
    elif scale == 'log10':
        # Apply log10 transformation (add 1 to avoid log(0))
        z_data = np.log10(z_data + 1)
        scale_label = ' (log10)'
    # else: scale == 'linear', use original values
    
    # Create custom hover text (only for smaller matrices)
    if not is_large:
        hover_text = []
        for i, row_label in enumerate(cmat.index):
            hover_row = []
            for j, col_label in enumerate(cmat.columns):
                value = cmat.iloc[i, j]
                # Format value based on metric type (show original value)
                if metric_type == 'ratio' or metric_type == 'probability':
                    value_str = f'{value:.4f}'
                else:
                    value_str = f'{int(value):,}' if value == int(value) else f'{value:,.2f}'
                
                hover_row.append(
                    f'<b>Source:</b> {row_label}<br>'
                    f'<b>Target:</b> {col_label}<br>'
                    f'<b>{metric_type.capitalize()}:</b> {value_str}'
                )
            hover_text.append(hover_row)
    else:
        # For large matrices, use simplified hover with automatic formatting
        hover_text = None

    # Determine color bar range
    zmin = z_data.min()
    zmax = z_data.max()
    if metric_type == 'synapses' and scale == 'linear':
        # For synapse counts, outliers can skew the color bar. Use 99th percentile for zmax.
        if zmax > 0:
            zmax = np.percentile(z_data, 99)
    elif metric_type in ['ratio', 'probability'] and scale == 'linear':
        zmin = 0.0
        zmax = min(zmax, 1.0) # Cap at 1.0

    # Create visualization with appropriate mode
    if use_scatter_mode:
        # Ultra-optimization: Use scatter plot showing only non-zero values
        # This reduces file size by 90%+ for sparse matrices
        non_zero_mask = z_data != 0
        rows, cols = np.where(non_zero_mask)
        values = z_data[non_zero_mask]
        
        # Create scatter plot (much more efficient for sparse data)
        fig = go.Figure(data=go.Scatter(
            x=cols,
            y=rows,
            mode='markers',
            marker=dict(
                size=8,
                color=values,
                colorscale=color_scale,
                cmin=zmin,
                cmax=zmax,
                colorbar=dict(
                    title=dict(
                        text=metric_type.capitalize() + scale_label,
                        side='right'
                    )
                ),
                line=dict(width=0.5, color='rgba(0,0,0,0.2)')
            ),
            hovertemplate=(
                'Row: %{y}<br>'
                'Col: %{x}<br>'
                'Value: %{marker.color:.2f}<br>'
                '<extra></extra>'
            )
        ))
        
        # Add note about visualization mode
        title = f"{title}<br><sub style='color:#666;'>Sparse matrix visualization (showing {len(values):,} non-zero connections)</sub>"
        
    else:
        # Standard heatmap mode
        heatmap_config = {
            'z': z_data,
            'colorscale': color_scale,
            'zmin': zmin,
            'zmax': zmax,
            'colorbar': dict(
                title=dict(
                    text=metric_type.capitalize() + scale_label,
                    side='right'
                )
            )
        }
        
        # Deep optimization: For large matrices, use indices instead of full labels
        if is_large:
            # Use numeric indices to drastically reduce JSON size
            # Labels stored separately and not embedded in every data point
            heatmap_config['x'] = list(range(len(cmat.columns)))
            heatmap_config['y'] = list(range(len(cmat.index)))
            
            # Simplified hover using indices (Plotly will auto-format)
            heatmap_config['hovertemplate'] = (
                'Row: %{y}<br>'
                'Col: %{x}<br>'
                'Value: %{z:.2f}<br>'
                '<extra></extra>'
            )
        else:
            # For small matrices, keep full labels
            heatmap_config['x'] = cmat.columns.astype(str)
            heatmap_config['y'] = cmat.index.astype(str)
            
            if hover_text is not None:
                heatmap_config['text'] = hover_text
                heatmap_config['hoverinfo'] = 'text'
            else:
                heatmap_config['hovertemplate'] = (
                    '<b>Source:</b> %{y}<br>'
                    '<b>Target:</b> %{x}<br>'
                    '<b>Value:</b> %{z:.2f}<br>'
                    '<extra></extra>'
                )
        
        fig = go.Figure(data=go.Heatmap(**heatmap_config))
    
    # Update layout with axis labels
    layout_config = {
        'title_text': title,
        'font_size': fontsize,
        'xaxis': dict(
            title=dict(text='<b>Target</b>', font=dict(size=fontsize+2, color='#333333')),
            side='bottom',
            tickangle=-45 if len(cmat.columns) > 1 else 0,  # Always rotate when multiple labels
            range=[-0.5, len(cmat.columns) - 0.5] if use_scatter_mode else None
        ),
        'yaxis': dict(
            title=dict(text='<b>Source</b>', font=dict(size=fontsize+2, color='#333333')),
            side='left',
            autorange='reversed',  # Keep the original order (top to bottom)
            range=[len(cmat.index) - 0.5, -0.5] if use_scatter_mode else None
        ),
        'hoverlabel': dict(
            bgcolor='white',
            font_size=12,
            font_family='Arial'
        ),
        'autosize': True,
        'margin': dict(l=120, r=40, b=120, t=140 if use_scatter_mode else 100, pad=4)
    }
    
    # For scatter mode, ensure proper aspect ratio
    if use_scatter_mode:
        layout_config['xaxis']['constrain'] = 'domain'
        layout_config['yaxis']['scaleanchor'] = 'x'
        layout_config['plot_bgcolor'] = 'white'
        layout_config['xaxis']['showgrid'] = True
        layout_config['yaxis']['showgrid'] = True
        layout_config['xaxis']['gridcolor'] = 'rgba(0,0,0,0.1)'
        layout_config['yaxis']['gridcolor'] = 'rgba(0,0,0,0.1)'
    
    # For large matrices, hide tick labels to reduce file size
    if is_large or use_scatter_mode:
        layout_config['xaxis']['showticklabels'] = False
        layout_config['yaxis']['showticklabels'] = False
        layout_config['xaxis']['title'] = f'<b>Target</b> ({len(cmat.columns)} neurons)'
        layout_config['yaxis']['title'] = f'<b>Source</b> ({len(cmat.index)} neurons)'
    
    fig.update_layout(**layout_config)
    
    # Write HTML with deep backend optimizations
    write_config = {
        'auto_open': showfig,
        'include_plotlyjs': 'cdn',  # Use CDN instead of embedding 3MB library
        'config': {
            'displayModeBar': True, 
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'] if is_large else []
        }
    }
    
    # Deep optimization for large matrices (but not scatter mode)
    if is_large and not use_scatter_mode:
        # Use div-only output for embedding (smaller than full HTML)
        write_config['include_mathjax'] = False  # Disable MathJax (not needed)
        write_config['div_id'] = 'heatmap'  # Enable div mode for potential embedding
        
        # Reduce decimal precision in JSON to save space
        # Plotly uses default precision which can be excessive for visualization
        # This is handled by converting figure to dict and rounding
        fig_dict = fig.to_dict()
        
        # Round z values to reduce JSON size
        if 'data' in fig_dict and len(fig_dict['data']) > 0:
            z_values = fig_dict['data'][0].get('z', [])
            if isinstance(z_values, (list, np.ndarray)):
                # For sparse matrices, more aggressive rounding
                decimals = 1 if is_sparse else 2
                
                # Convert to numpy array if needed
                z_array = np.array(z_values)
                
                # Round values
                z_rounded = np.round(z_array, decimals)
                
                # For very sparse matrices, explicitly convert zeros to reduce storage
                if is_sparse:
                    # Set very small values to exactly 0 (reduces file size via compression)
                    z_rounded[np.abs(z_rounded) < 0.01] = 0
                
                fig_dict['data'][0]['z'] = z_rounded.tolist()
        
        # Recreate figure from optimized dict
        fig = go.Figure(fig_dict)
        
        # Add compression hint in title for user awareness
        if is_sparse and not use_scatter_mode:
            sparsity_pct = int(sparsity * 100)
            fig.update_layout(
                title_text=f"{title}<br><sub style='color:#666;'>Matrix {sparsity_pct}% sparse - optimized for file size</sub>"
            )
    
    fig.write_html(filename, **write_config)


# ============================================================================
# DEPRECATED: VisConnMatInteractive in statvis.py
# ============================================================================
# This function is now maintained in vispath_pkg/vispath.py
# Use: from vispath_pkg import VisConnMatInteractive
# The statvis.py version is kept for backwards compatibility but will redirect
# to vispath_pkg if available.
# ============================================================================

def VisConnMatInteractive(cmat, filename, title='', color_scale=None, showfig=True, fontsize=12, conn_df=None, matrices_dict=None, verbose=True, zmin=None, zmax=None, init_width=None, init_height=None, init_clustered=True, metric_name=None):
    if color_scale is None:
        color_scale = [[0, 'rgb(255,255,255)'], [1, 'rgb(104,55,164)']]
    '''Create interactive heatmap with comprehensive controls.
    
    DEPRECATED: This function is now maintained in vispath_pkg.
    Please use: from vispath_pkg import VisConnMatInteractive
    
    This wrapper will redirect to vispath_pkg if available.
    '''
    # Try to use vispath_pkg version first (more up-to-date)
    try:
        from vispath_pkg import VisConnMatInteractive as VisPkgVisConnMatInteractive
        return VisPkgVisConnMatInteractive(
            cmat=cmat, 
            filename=filename, 
            title=title, 
            color_scale=color_scale, 
            showfig=showfig, 
            fontsize=fontsize, 
            conn_df=conn_df, 
            matrices_dict=matrices_dict, 
            verbose=verbose,
            zmin=zmin,
            zmax=zmax,
            init_width=init_width,
            init_height=init_height,
            init_clustered=init_clustered,
            metric_name=metric_name
        )
    except ImportError:
        pass  # Fall through to local implementation
    
    # Local implementation (kept for backwards compatibility)
    _VisConnMatInteractive_local(cmat, filename, title, color_scale, showfig, fontsize, conn_df, matrices_dict, verbose, zmin, zmax)


def _VisConnMatInteractive_local(cmat, filename, title='', color_scale=[[0, 'rgb(255,255,255)'], [1, 'rgb(104,55,164)']], showfig=True, fontsize=12, conn_df=None, matrices_dict=None, verbose=True, zmin=None, zmax=None):
    '''[DEPRECATED LOCAL VERSION] Create interactive heatmap with comprehensive controls similar to network visualization
    
    Features:
    - Metric toggle: Switch between weight/ratio/probability (if provided)
    - Clustering toggle: Toggle between original and clustered ordering (hierarchical clustering)
    - Scale switcher: Linear / Log2 / Log10 / Sqrt
    - Colorscale selector with presets (Greens, Purples, Oranges, Blues, Reds, Viridis, etc.)
    - Font size slider
    - Export to SVG with adjustable resolution
    - Zoom/pan controls
    - Save/load layout state
    
    Parameters
    ----------
    cmat : pd.DataFrame
        Connection matrix to visualize (weight matrix if matrices_dict not provided)
    filename : str
        Output HTML filename
    title : str, optional
        Title for the heatmap
    color_scale : list, optional
        Plotly color scale (default starting point)
    showfig : bool, optional
        Whether to open in browser
    fontsize : int, optional
        Default font size for labels
    conn_df : pd.DataFrame, optional
        Connection dataframe with type information for enhanced hover labels (bodyId heatmaps only)
    matrices_dict : dict, optional
        Dictionary with keys 'weight', 'ratio', 'probability' containing different metric matrices
        If provided, enables metric toggle. Otherwise uses cmat as weight matrix only.
    verbose : bool, optional
        Whether to print progress messages (default: True)
    zmin : float, optional
        Minimum value for color scale. If None, auto-computed from data.
    zmax : float, optional
        Maximum value for color scale. If None, auto-computed from data.
    '''
    
    # Handle multiple matrices for metric toggle
    has_multiple_metrics = matrices_dict is not None and isinstance(matrices_dict, dict)
    
    if has_multiple_metrics:
        # Use provided matrices dictionary
        available_metrics = []
        matrices_data = {}
        
        if 'weight' in matrices_dict and matrices_dict['weight'] is not None:
            available_metrics.append('weight')
            matrices_data['weight'] = matrices_dict['weight'].values.copy()
        
        if 'ratio' in matrices_dict and matrices_dict['ratio'] is not None:
            available_metrics.append('ratio')
            matrices_data['ratio'] = matrices_dict['ratio'].values.copy()
        
        if 'probability' in matrices_dict and matrices_dict['probability'] is not None:
            available_metrics.append('probability')
            matrices_data['probability'] = matrices_dict['probability'].values.copy()
        
        # Use first available metric as default
        default_metric = available_metrics[0] if available_metrics else 'weight'
        data_linear = matrices_data.get(default_metric, cmat.values.copy())
        metric_type = default_metric
    else:
        # Single matrix mode - determine metric type from title/filename
        available_metrics = ['weight']  # Only one metric available
        matrices_data = {}
        
        metric_type = 'weight'
        if 'ratio' in title.lower() or 'ratio' in filename.lower():
            metric_type = 'ratio'
            available_metrics = ['ratio']
        elif 'transmission' in title.lower() or 'probability' in title.lower():
            metric_type = 'probability'
            available_metrics = ['probability']
        
        data_linear = cmat.values.copy()
        matrices_data[metric_type] = data_linear
    
    is_large = cmat.shape[0] > 100 or cmat.shape[1] > 100
    
    # Check sparsity for potential optimization
    zero_count = np.count_nonzero(data_linear == 0)
    sparsity_ratio = zero_count / data_linear.size
    is_sparse = sparsity_ratio > 0.5  # More than 50% zeros
    
    # Compute hierarchical clustering with multiple methods for row/column ordering
    if verbose:
        print("  Computing hierarchical clustering...")
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    
    # Store clustering results for all methods
    clustering_methods = ['ward', 'average', 'complete', 'single']
    clustering_results = {}
    
    try:
        for method in clustering_methods:
            method_results = {}
            
            # Cluster rows (source neurons)
            if data_linear.shape[0] > 1:
                # Use euclidean distance (required for ward, good for others)
                row_distances = pdist(data_linear, metric='euclidean')
                # Check for non-finite values
                if not np.all(np.isfinite(row_distances)):
                    raise ValueError("Non-finite distances in row clustering")
                row_linkage = linkage(row_distances, method=method)
                method_results['row_order'] = leaves_list(row_linkage).tolist()
            else:
                method_results['row_order'] = [0]
            
            # Cluster columns (target neurons)
            if data_linear.shape[1] > 1:
                col_distances = pdist(data_linear.T, metric='euclidean')
                # Check for non-finite values
                if not np.all(np.isfinite(col_distances)):
                    raise ValueError("Non-finite distances in column clustering")
                col_linkage = linkage(col_distances, method=method)
                method_results['col_order'] = leaves_list(col_linkage).tolist()
            else:
                method_results['col_order'] = [0]
            
            clustering_results[method] = method_results
        
        # Use Ward as default (best for most connectome data)
        row_order_clustered = np.array(clustering_results['ward']['row_order'])
        col_order_clustered = np.array(clustering_results['ward']['col_order'])
        
        clustering_successful = True
        print(f"  ✓ Clustering complete: {len(row_order_clustered)} rows, {len(col_order_clustered)} cols")
        print(f"  Available methods: Ward (default), Average, Complete, Single")
    except Exception as e:
        print(f"  ⚠ Clustering failed: {e}")
        print(f"  Using original order")
        row_order_clustered = np.array(range(data_linear.shape[0]))
        col_order_clustered = np.array(range(data_linear.shape[1]))
        clustering_successful = False
        clustering_results = {}

    
    # Store both original and clustered orders
    row_order_original = list(range(data_linear.shape[0]))
    col_order_original = list(range(data_linear.shape[1]))
    
    # For large matrices, reduce precision to save HTML size
    # Keep more precision for ratio/probability metrics
    if is_large:
        if metric_type in ['ratio', 'probability']:
            # Keep 4 decimal places for ratios/probabilities
            data_linear = np.round(data_linear, 4)
        else:
            # For synapse counts, round to integers (no precision loss)
            data_linear = np.round(data_linear, 0)
    
    # Deep optimization: For very large matrices, compute transforms in JavaScript
    # This saves ~75% of HTML file size by not embedding pre-computed transforms
    use_lazy_transforms = is_large and data_linear.size > 50000
    
    # Sparse matrix optimization: For matrices with >70% zeros, use COO format
    use_sparse_format = is_large and sparsity_ratio > 0.7 and data_linear.size > 50000
    sparse_data = None
    
    if use_sparse_format:
        # Convert to COO (Coordinate) format: store only non-zero values
        rows, cols = np.nonzero(data_linear)
        values = data_linear[rows, cols]
        sparse_data = {
            'rows': rows.tolist(),
            'cols': cols.tolist(),
            'values': values.tolist(),
            'shape': list(data_linear.shape)
        }
        print(f"  Using sparse format: {sparsity_ratio*100:.1f}% zeros, storing {len(values)} values instead of {data_linear.size}")
    
    if use_lazy_transforms:
        # Store only linear data; transforms computed client-side
        data_log2 = None
        data_log10 = None
        data_sqrt = None
    else:
        # Pre-compute for small matrices (faster initial display)
        # Handle negative values: sign(v) * transform(|v|)
        data_log2 = np.where(data_linear >= 0, 
                             np.log2(data_linear + 1), 
                             -np.log2(-data_linear + 1))
        data_log10 = np.where(data_linear >= 0, 
                              np.log10(data_linear + 1), 
                              -np.log10(-data_linear + 1))
        data_sqrt = np.where(data_linear >= 0, 
                            np.sqrt(data_linear), 
                            -np.sqrt(-data_linear))
        
        if is_large:
            if metric_type in ['ratio', 'probability']:
                data_log2 = np.round(data_log2, 4)
                data_log10 = np.round(data_log10, 4)
                data_sqrt = np.round(data_sqrt, 4)
            else:
                data_log2 = np.round(data_log2, 2)
                data_log10 = np.round(data_log10, 2)
                data_sqrt = np.round(data_sqrt, 2)
    
    # Create hover text with original values
    # If conn_df is provided, create type lookup for bodyId heatmaps
    type_lookup = None
    
    if conn_df is not None and 'bodyId_pre' in conn_df.columns and 'type_pre' in conn_df.columns:
        # Create lookup dictionaries for bodyId -> type
        # Convert bodyId keys to strings to match matrix index/columns
        type_lookup = {
            'pre': {str(k): v for k, v in conn_df.set_index('bodyId_pre')['type_pre'].to_dict().items()},
            'post': {str(k): v for k, v in conn_df.set_index('bodyId_post')['type_post'].to_dict().items()}
        }
    # Generate hover text with actual labels for all matrix sizes
    # No longer use compact mode - always show full information with proper labels
    hover_text = []
    for i, row_label in enumerate(cmat.index):
        hover_row = []
        for j, col_label in enumerate(cmat.columns):
            value = cmat.iloc[i, j]
            if metric_type == 'ratio' or metric_type == 'probability':
                value_str = f'{value:.4f}'
            else:
                value_str = f'{int(value):,}' if value == int(value) else f'{value:,.2f}'
            
            # Always use actual labels with type info if available
            if type_lookup:
                try:
                    # Labels are already strings, use them directly for type lookup
                    row_id = str(row_label);
                    col_id = str(col_label);
                    row_type = type_lookup['pre'].get(row_id, 'Unknown')
                    col_type = type_lookup['post'].get(col_id, 'Unknown')
                    hover_row.append(f'<b>Source:</b> {row_label} ({row_type})<br><b>Target:</b> {col_label} ({col_type})<br><b>{metric_type.capitalize()}:</b> {value_str}')
                except:
                    # Fall back to label-only display if type lookup fails
                    hover_row.append(f'<b>Source:</b> {row_label}<br><b>Target:</b> {col_label}<br><b>{metric_type.capitalize()}:</b> {value_str}')
            else:
                # No type info available - just show labels
                hover_row.append(f'<b>Source:</b> {row_label}<br><b>Target:</b> {col_label}<br><b>{metric_type.capitalize()}:</b> {value_str}')
        hover_text.append(hover_row)
    
    # Determine axis labels - ALWAYS use actual names, not numeric indices
    # Even for large matrices, show proper labels (optimization only affects hover text)
    x_labels = cmat.columns.astype(str).tolist()
    y_labels = cmat.index.astype(str).tolist();
    
    # Generate unique storage key for this heatmap
    from datetime import datetime
    output_name = os.path.splitext(os.path.basename(filename))[0]
    timestamp_hash = datetime.now().strftime('%Y%m%d%H%M%S');
    storage_key = f"heatmap_settings_{output_name}#{timestamp_hash}"
    
    # Determine default colorscale name
    default_colorscale = 'Greens'
    if 'ratio' in filename.lower():
        default_colorscale = 'Oranges'
    elif 'transmission' in filename.lower() or 'probability' in filename.lower():
        default_colorscale = 'Purples'
    
    # Create HTML with comprehensive interactive controls
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{_statvis_html_escape(title)}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            user-select: text;
        }}
        
        .main-container {{
            max-width: 1800px;
            margin: 0 auto;
        }}
        
        .controls {{
            background: white;
            padding: 12px;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }}
        
        .controls-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 8px;
            margin-bottom: 10px;
        }}
        
        .control-section {{
            background: #f8f9fa;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #e9ecef;
        }}
        
        .control-section h3 {{
            margin: 0 0 8px 0;
            font-size: 12px;
            font-weight: 600;
            color: #495057;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}
        
        .button-group {{
            display: flex;
            gap: 4px;
            flex-wrap: wrap;
        }}
        
        button {{
            padding: 6px 10px;
            border: 1px solid #dee2e6;
            background: white;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
            font-weight: 500;
            transition: all 0.2s;
            color: #495057;
        }}
        
        button:hover {{
            background: #f8f9fa;
            border-color: #adb5bd;
        }}
        
        button.active {{
            background: #4CAF50;
            color: white;
            border-color: #4CAF50;
        }}
        
        button.export-btn {{
            background: #2196F3;
            color: white;
            border-color: #2196F3;
        }}
        
        button.export-btn:hover {{
            background: #1976D2;
            border-color: #1976D2;
        }}
        
        button.save-btn {{
            background: #FF9800;
            color: white;
            border-color: #FF9800;
        }}
        
        button.save-btn:hover {{
            background: #F57C00;
            border-color: #F57C00;
        }}
        
        select {{
            width: 100%;
            padding: 4px 6px;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            font-size: 11px;
            background: white;
            cursor: pointer;
            color: #495057;
        }}
        
        select:focus {{
            outline: none;
            border-color: #4CAF50;
            box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.1);
        }}
        
        .slider-control {{
            margin-bottom: 6px;
        }}
        
        .slider-control label {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 3px;
            font-size: 10px;
            color: #495057;
            font-weight: 500;
        }}
        
        .slider-value {{
            color: #4CAF50;
            font-weight: 600;
        }}
        
        input[type="range"] {{
            width: 100%;
            height: 4px;
            border-radius: 2px;
            background: #dee2e6;
            outline: none;
            -webkit-appearance: none;
        }}
        
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4CAF50;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        input[type="range"]::-webkit-slider-thumb:hover {{
            background: #45a049;
            transform: scale(1.15);
        }}
        
        input[type="range"]::-moz-range-thumb {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4CAF50;
            cursor: pointer;
            border: none;
        }}
        
        #heatmap-container {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        #heatmap {{
            width: 100%;
            height: 1200px;
        }}
        
        .status-message {{
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            text-align: center;
            margin-top: 8px;
        }}
        
        .status-success {{
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        
        .status-info {{
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }}
        
        .status-error {{
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        
        .info-box {{
            background: #e7f3ff;
            border-left: 3px solid #2196F3;
            padding: 8px;
            border-radius: 3px;
            font-size: 10px;
            color: #1976D2;
            margin-top: 8px;
            line-height: 1.4;
        }}
        
        .info-box strong {{
            display: block;
            margin-bottom: 3px;
            font-size: 11px;
        }}
        
        .drag-item {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 3px;
            padding: 6px 8px;
            margin-bottom: 4px;
            cursor: move;
            user-select: none;
            display: flex;
            align-items: center;
            transition: all 0.2s;
        }}
        
        .drag-item:hover {{
            background: #f0f0f0;
            border-color: #4CAF50;
        }}
        
        .drag-item.dragging {{
            opacity: 0.5;
            background: #e3f2fd;
        }}
        
        .drag-item.drag-over {{
            border-top: 3px solid #4CAF50;
        }}
        
        .drag-handle {{
            margin-right: 6px;
            color: #999;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="main-container">
        <div class="controls">
            <div class="controls-grid">
                <!-- Metric, Ordering & Scale Combined Section -->
                {'<div class="control-section" id="metricOrderingSection">' if has_multiple_metrics else '<div class="control-section">'}
                    {'<h3>📊 Metric, Ordering & Scale</h3>' if has_multiple_metrics else '<h3>🔀 Ordering & Scale</h3>'}
                    
                    <!-- Metric Selection (if multiple metrics available) -->
                    {'<div style="margin-bottom: 8px;"><label style="font-size: 10px; display: block; margin-bottom: 2px;">Metric:</label>' if has_multiple_metrics else '<!-- Single metric mode -->'}
                    {'<select id="metricSelect" onchange="updateMetric()">' if has_multiple_metrics else ''}
                        {'<option value="weight">Synapse Count</option>' if has_multiple_metrics and 'weight' in available_metrics else ''}
                        {'<option value="ratio"' + (' selected' if metric_type == 'ratio' else '') + '>Connection Ratio</option>' if has_multiple_metrics and 'ratio' in available_metrics else ''}
                        {'<option value="probability"' + (' selected' if metric_type == 'probability' else '') + '>Traversal Probability</option>' if has_multiple_metrics and 'probability' in available_metrics else ''}
                    {'</select></div>' if has_multiple_metrics else ''}
                    
                    <!-- Clustering Toggle -->
                    <div style="margin-bottom: 8px;">
                        <label style="font-size: 10px; display: block; margin-bottom: 2px;">Ordering:</label>
                        <div class="button-group">
                            <button id="btn-original" class="active" onclick="toggleClustering('original')">Original</button>
                            <button id="btn-clustered" onclick="toggleClustering('clustered')">Clustered</button>
                        </div>
                    </div>
                    
                    <!-- Clustering Method Selection -->
                    <div id="clusteringMethodSection" style="margin-bottom: 8px; display: none;">
                        <label style="font-size: 10px; display: block; margin-bottom: 2px;">Clustering Method:</label>
                        <select id="clusteringMethodSelect" onchange="updateClusteringMethod()" style="width: 100%; font-size: 10px; padding: 4px;">
                            <option value="ward">Ward (Compact Clusters)</option>
                            <option value="average">Average (Balanced)</option>
                            <option value="complete">Complete (Tight Clusters)</option>
                            <option value="single">Single (Loose Clusters)</option>
                        </select>
                    </div>
                    
                    <!-- Scale Selection -->
                    <div>
                        <label style="font-size: 10px; display: block; margin-bottom: 2px;">Scale:</label>
                        <div class="button-group">
                            <button id="btn-linear" class="active" onclick="setScale('linear')">Linear</button>
                            <button id="btn-log2" onclick="setScale('log2')">Log₂</button>
                            <button id="btn-log10" onclick="setScale('log10')">Log₁₀</button>
                            <button id="btn-sqrt" onclick="setScale('sqrt')">√</button>
                        </div>
                    </div>
                </div>
                
                <!-- Color -->
                <div class="control-section" id="colorscaleSection">
                    <h3>🎨 Color</h3>
                    <select id="colorscaleSelect" onchange="updateColorscale()" style="margin-bottom: 8px;">
                        <option value="Greens" {'selected' if default_colorscale == 'Greens' else ''}>Greens</option>
                        <option value="Purples" {'selected' if default_colorscale == 'Purples' else ''}>Purples</option>
                        <option value="Oranges" {'selected' if default_colorscale == 'Oranges' else ''}>Oranges</option>
                        <option value="Blues" {'selected' if default_colorscale == 'Blues' else ''}>Blues</option>
                        <option value="Reds">Reds</option>
                        <option value="Viridis">Viridis</option>
                        <option value="Plasma">Plasma</option>
                        <option value="Inferno">Inferno</option>
                        <option value="Magma">Magma</option>
                        <option value="Cividis">Cividis</option>
                        <option value="Hot">Hot</option>
                        <option value="Jet">Jet</option>
                        <option value="RdBu">Red-Blue (Diverging)</option>
                        <option value="RdYlGn">Red-Yellow-Green</option>
                        <option value="Custom">Custom</option>
                    </select>
                    
                    <div id="customColorSection">
                        <div style="margin-bottom: 6px;">
                            <label style="display: block; margin-bottom: 3px; font-size: 10px;">
                                <input type="checkbox" id="use3PointScale" onchange="toggle3PointScale()"> 
                                3-Point Scale (diverging)
                            </label>
                        </div>
                        <div id="twoPointColors">
                            <div style="margin-bottom: 4px;">
                                <label style="font-size: 10px; display: block; margin-bottom: 2px;">Min (value):</label>
                                <div style="display: flex; gap: 4px;">
                                    <input type="color" id="colorMin" value="#ffffff" style="width: 40px; height: 26px; cursor: pointer;">
                                    <input type="number" id="valueMin2" placeholder="Auto" step="any" style="flex: 1; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;">
                                </div>
                            </div>
                            <div style="margin-bottom: 4px;">
                                <label style="font-size: 10px; display: block; margin-bottom: 2px;">Max (value):</label>
                                <div style="display: flex; gap: 4px;">
                                    <input type="color" id="colorMax" value="#68379c" style="width: 40px; height: 26px; cursor: pointer;">
                                    <input type="number" id="valueMax2" placeholder="Auto" step="any" style="flex: 1; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;">
                                </div>
                            </div>
                        </div>
                        <div id="threePointColors" style="display: none;">
                            <div style="margin-bottom: 4px;">
                                <label style="font-size: 10px; display: block; margin-bottom: 2px;">Min (value):</label>
                                <div style="display: flex; gap: 4px;">
                                    <input type="color" id="colorMin3" value="#0000ff" style="width: 40px; height: 26px; cursor: pointer;">
                                    <input type="number" id="valueMin3" value="0" step="any" style="flex: 1; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;">
                                </div>
                            </div>
                            <div style="margin-bottom: 4px;">
                                <label style="font-size: 10px; display: block; margin-bottom: 2px;">Mid (value):</label>
                                <div style="display: flex; gap: 4px;">
                                    <input type="color" id="colorMid3" value="#ffffff" style="width: 40px; height: 26px; cursor: pointer;">
                                    <input type="number" id="valueMid3" value="0.5" step="any" style="flex: 1; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;">
                                </div>
                            </div>
                            <div style="margin-bottom: 4px;">
                                <label style="font-size: 10px; display: block; margin-bottom: 2px;">Max (value):</label>
                                <div style="display: flex; gap: 4px;">
                                    <input type="color" id="colorMax3" value="#ff0000" style="width: 40px; height: 26px; cursor: pointer;">
                                    <input type="number" id="valueMax3" value="1" step="any" style="flex: 1; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;">
                                </div>
                            </div>
                        </div>
                        <div style="display: flex; gap: 4px; margin-top: 4px;">
                            <button onclick="applyCustomColors()" style="flex: 1; font-size: 10px;">Apply</button>
                            <button onclick="resetToAutoColors()" style="flex: 1; font-size: 10px;">Auto</button>
                        </div>
                    </div>
                </div>
                
                <!-- Font Size & Colorbar Settings -->
                <div class="control-section">
                    <h3>🎚️ Display</h3>
                    <div class="slider-control">
                        <label>
                            <span>Font Size:</span>
                            <span class="slider-value" id="fontSizeValue">{fontsize}px</span>
                        </label>
                        <input type="range" id="fontSizeSlider" min="8" max="48" value="{fontsize}" step="1" oninput="updateFontSize(this.value)">
                    </div>
                    <div style="margin-top: 8px; display: flex; gap: 4px;">
                        <button id="toggleLabelsBtn" onclick="toggleLabels()" style="flex: 1;">
                            {'🏷️ Hide Text' if not is_large else '🏷️ Show Text'}
                        </button>
                        <button id="toggleCellValuesBtn" onclick="toggleCellValues()" style="flex: 1;">
                            🔢 Hide Values
                        </button>
                    </div>
                    <div class="slider-control" style="margin-top: 8px;">
                        <label>
                            <span>Cell Value Size:</span>
                            <span class="slider-value" id="cellValueSizeValue">10px</span>
                        </label>
                        <input type="range" id="cellValueSizeSlider" min="6" max="48" value="10" step="1" oninput="updateCellValueSize(this.value)">
                    </div>
                    <div style="margin-top: 8px;">
                        <label style="font-size: 11px; display: block; margin-bottom: 4px;">Ignore Values (comma-separated):</label>
                        <input type="text" id="ignoreValuesInput" placeholder="e.g., 0, >20, <=5" style="width: 100%; padding: 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px; box-sizing: border-box;" oninput="updateIgnoredValues()">
                    </div>
                    <div style="margin-top: 8px;">
                        <label style="font-size: 11px; display: block; margin-bottom: 4px;">
                            🔍 Data Filter (hide rows/cols):
                            <button onclick="resetDataFilter()" style="padding: 2px 6px; font-size: 9px; background: #6c757d; color: white; border: none; border-radius: 3px; cursor: pointer; margin-left: 4px;" title="Reset filter">🔄 Reset</button>
                        </label>
                        <input type="text" id="dataFilterInput" placeholder="e.g., <5, <=10, >100" style="width: 100%; padding: 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px; box-sizing: border-box;" oninput="applyDataFilter()">
                        <div id="filterStatus" style="font-size: 9px; color: #666; margin-top: 2px; min-height: 14px;"></div>
                    </div>
                    <div class="slider-control" style="margin-top: 8px;">
                        <label>
                            <span>Contrast Threshold:</span>
                            <span class="slider-value" id="contrastThresholdValue">0.5000</span>
                            <button onclick="reverseContrastColors()" style="padding: 2px 6px; font-size: 10px; background: #6c757d; color: white; border: none; border-radius: 3px; cursor: pointer; margin-left: 4px;" title="Reverse black/white colors">🔄</button>
                        </label>
                        <input type="range" id="contrastThresholdSlider" min="0" max="1" value="0.5" step="0.0001" oninput="updateContrastThreshold(this.value)">
                    </div>
                </div>
                
                <!-- Plot Dimensions -->
                <div class="control-section">
                    <h3>📐 Plot Size</h3>
                    <div class="slider-control">
                        <label>
                            <span>Width:</span>
                            <span class="slider-value" id="widthValue">800px</span>
                        </label>
                        <div style="display: flex; gap: 4px; align-items: center;">
                            <input type="range" id="widthSlider" min="400" max="2400" value="800" step="20" oninput="updatePlotSize()" style="flex: 1;">
                            <input type="number" id="widthInput" value="800" min="100" step="20" style="width: 70px; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;" oninput="updatePlotSizeFromInput()">
                        </div>
                   
                                      
                    <div class="slider-control">
                        <label>
                            <span>Height:</span>
                            <span class="slider-value" id="heightValue">800px</span>
                        </label>
                        <div style="display: flex; gap: 4px; align-items: center;">
                            <input type="range" id="heightSlider" min="400" max="2400" value="800" step="20" oninput="updatePlotSize()" style="flex: 1;">
                            <input type="number" id="heightInput" value="800" min="100" step="20" style="width: 70px; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;" oninput="updatePlotSizeFromInput()">
                        </div>
                    </div>
                    <div style="display: flex; gap: 4px;">
                        <button id="squareCellsBtn" onclick="makeSquareCells()" style="flex: 1;">⬜ Square Cells</button>
                        <button onclick="resetPlotSize()" style="flex: 1;">🔄 Reset</button>
                    </div>
                    <div style="margin-top: 8px;">
                        <button id="transposeBtn" onclick="transposeMatrix()" style="width: 100%;">🔄 Swap Rows ↔ Columns</button>
                    </div>
                </div>
                
                <!-- Row/Column Ordering -->
                <div class="control-section">
                    <h3>📋 Row/Column Order</h3>
                    <button onclick="toggleOrderPanel('rows')" style="width: 100%; font-size: 10px; margin-bottom: 4px;">📑 Reorder Rows</button>
                    <button onclick="toggleOrderPanel('cols')" style="width: 100%; font-size: 10px; margin-bottom: 4px;">📑 Reorder Columns</button>
                    <button onclick="resetOrder()" style="width: 100%; font-size: 10px;">🔄 Reset to Original</button>
                    
                </div>
                
                <!-- Export & Saving -->
                <div class="control-section">
                    <h3>💾 Export & Saving</h3>
                    <div class="slider-control" style="margin-bottom: 8px;">
                        <label>
                            <span>Export Scale:</span>
                            <span class="slider-value" id="exportScaleValue">2x</span>
                        </label>
                        <input type="range" id="exportScaleSlider" min="1" max="5" value="2" step="0.5" oninput="updateExportScale(this.value)">
                    </div>
                    <div class="button-group" style="flex-direction: column; margin-bottom: 8px;">
                        <button class="export-btn" onclick="exportSVG()" style="width: 100%;">📥 Export SVG</button>
                    </div>
                    <div class="button-group">
                        <button class="save-btn" onclick="saveSettings()">💾 Save</button>
                        <button class="save-btn" onclick="loadSettings()">📂 Load</button>
                        <button onclick="resetSettings()">🔄 Reset</button>
                    </div>
                    <div id="settingsStatus"></div>
                </div>
            </div>
            
            <div class="info-box">
                <strong>💡 Tips:</strong>
                Use Log₂ or Log₁₀ scales for large dynamic ranges • 
                Adjust plot size with width/height sliders for better visualization • 
                Use export scale (1x-5x) to control SVG resolution • 
                3-point custom colors ideal for diverging data (negative → zero → positive) • 
                Hover over cells for details • 
                Zoom and pan with mouse • 
                Settings persist across sessions
            </div>
        </div>
        
        <div id="heatmap-container">
            <div id="heatmap"></div>
        </div>
    </div>
    
    <!-- Floating Reorder Panel -->
    <div id="orderPanel" style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                                 background: white; border: 2px solid #333; border-radius: 8px; padding: 16px; 
                                 box-shadow: 0 4px 20px rgba(0,0,0,0.3); z-index: 10000; min-width: 300px; max-width: 400px; 
                                 max-height: 70vh; flex-direction: column; display: none;">
        <div style="margin-bottom: 12px; border-bottom: 2px solid #ddd; padding-bottom: 8px;">
            <label id="orderPanelLabel" style="font-size: 14px; font-weight: bold; color: #333;"></label>
        </div>
        <div id="orderList" style="font-size: 12px; overflow-y: auto; flex: 1; margin-bottom: 12px;"></div>
        <button onclick="closeOrderPanel()" style="width: 100%; font-size: 12px; padding: 8px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">✓ Close</button>
    </div>
    
    <!-- Overlay backdrop for floating panel -->
    <div id="orderPanelBackdrop" style="display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; 
                                        background: rgba(0,0,0,0.5); z-index: 9999;" onclick="closeOrderPanel()"></div>
    
    <script>
        // Metric toggle support
        const availableMetrics = {json.dumps(available_metrics)};
        const hasMultipleMetrics = availableMetrics.length > 1;
        let currentMetric = '{metric_type}';
        
        // Store all metric matrices
        const metricsData = {{}};
'''
    
    # Add metric data assignments
    for metric in available_metrics:
        html_content += f"        metricsData['{metric}'] = {json.dumps(matrices_data[metric].tolist())};\n"
    
    html_content += f'''
        
        // Data for different scales
        const sparseData = {json.dumps(sparse_data) if sparse_data is not None else 'null'};
        const useSparseFormat = sparseData !== null;
        
        // Get current metric data
        let dataLinear = metricsData[currentMetric];
        
        const dataLog2 = {'null' if data_log2 is None else json.dumps(data_log2.tolist())};
        const dataLog10 = {'null' if data_log10 is None else json.dumps(data_log10.tolist())};
        const dataSqrt = {'null' if data_sqrt is None else json.dumps(data_sqrt.tolist())};
        const xLabels = {json.dumps(x_labels)};
        const yLabels = {json.dumps(y_labels)};
        const storageKey = '{_statvis_js_escape(storage_key)}';
        const useLazyTransforms = {json.dumps(use_lazy_transforms)};
        
        // Track current row/column order (for interactive reordering)
        let currentXLabels = xLabels.slice();
        let currentYLabels = yLabels.slice();
        
        // Hover text - always use full array with proper labels (no compact mode)
        const hoverText = {json.dumps(hover_text)};
        
        // Cache for lazy-computed transforms
        let cachedDataLog2 = null;
        let cachedDataLog10 = null;
        let cachedDataSqrt = null;
        
        // Clustering data - row and column orders for all methods
        const rowOrderOriginal = {json.dumps(row_order_original)};
        const colOrderOriginal = {json.dumps(col_order_original)};
        const clusteringAvailable = {json.dumps(clustering_successful)};
        
        // All clustering method results
        const clusteringResults = {json.dumps(clustering_results)};
        
        // Default to Ward method
        const rowOrderClustered = {json.dumps(row_order_clustered.tolist())};
        const colOrderClustered = {json.dumps(col_order_clustered.tolist())};
        
        // Current settings
        let currentScale = 'linear';
        let currentColorscale = '{default_colorscale}';
        let currentFontSize = {fontsize};
        let useAutoRange = {json.dumps(zmin is None and zmax is None)};
        let customZmin = {json.dumps(zmin)};
        let customZmax = {json.dumps(zmax)};
        let customColorScale = null;  // Store custom color scale
        let use3PointScale = false;
        let currentWidth = 800;
        let currentHeight = 800;
        let exportScale = 2;
        let squareCellsLocked = false;  // Track if square cells are locked
        let showLabels = !{json.dumps(is_large)};  // Show labels for small matrices, hide for large
        let showCellValues = true;  // Track if cell values should be displayed in cells (default: true)
        let cellValueFontSize = 10;  // Font size for cell value annotations
        let ignoredValues = new Set();  // Set of values to ignore when displaying cell values
        let contrastThreshold = 0.5;  // Luminance threshold for contrast color (0-1, default: 0.5)
        let reverseContrast = false;  // Whether to reverse black/white contrast colors
        let useClusteredOrder = false;  // Track current ordering mode
        let currentClusteringMethod = 'ward';  // Current clustering method (ward, average, complete, single)
        let isTransposed = false;  // Track if matrix is transposed
        const metricType = '{metric_type}';
        const isLarge = {json.dumps(is_large)};
        const originalTitle = '{_statvis_js_escape(title)}';
        
        // Data filter state
        let dataFilterActive = false;
        let dataFilterExpressions = [];
        let filteredRowIndices = [];  // Indices of rows to show after filtering
        let filteredColIndices = [];  // Indices of columns to show after filtering
        
        // Function to generate hover text dynamically when needed
        // Hover text is pre-generated in Python with proper labels
        // This function regenerates it when switching metrics (multi-metric mode)
        function generateHoverText() {{
            if (!hasMultipleMetrics) {{
                return hoverText;  // Use pre-generated hover text for single-metric mode
            }}
            
            // Generate hover text on-the-fly for multi-metric mode
            const rows = dataLinear.length;
            const cols = dataLinear[0].length;
            const result = new Array(rows);
            
            // Get metric display name
            const metricNames = {{
                'weight': 'Synapses',
                'ratio': 'Ratio',
                'probability': 'Probability'
            }};
            const currentMetricName = metricNames[currentMetric] || currentMetric;
            
            for (let i = 0; i < rows; i++) {{
                result[i] = new Array(cols);
                for (let j = 0; j < cols; j++) {{
                    const value = dataLinear[i][j];
                    let valueStr;
                    if (currentMetric === 'ratio' || currentMetric === 'probability') {{
                        valueStr = value.toFixed(4);
                    }} else {{
                        valueStr = Math.floor(value) === value ? 
                            value.toLocaleString() : 
                            value.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
                    }}

                    // Always use actual labels from yLabels and xLabels
                    const srcLabel = yLabels[i];
                    const tgtLabel = xLabels[j];
                    result[i][j] = '<b>Source:</b> ' + srcLabel + '<br><b>Target:</b> ' + tgtLabel + '<br><b>' + currentMetricName + ':</b> ' + valueStr;
                }}
            }}
            return result;
        }}
        
        function getDataForScale(scale) {{
            if (!useLazyTransforms) {{
                // Use pre-computed data for small matrices
                switch(scale) {{
                    case 'log2': return dataLog2;
                    case 'log10': return dataLog10;
                    case 'sqrt': return dataSqrt;
                    default: return dataLinear;
                }}
            }}
            
            // Lazy computation for large matrices
            switch(scale) {{
                case 'log2':
                    if (cachedDataLog2 === null) {{
                        console.log('Computing log₂ transform...');
                        cachedDataLog2 = dataLinear.map(row => row.map(v => {{
                            // Handle negative values: sign(v) * log2(|v| + 1)
                            if (v < 0) return -Math.log2(-v + 1);
                            return Math.log2(v + 1);
                        }}));
                    }}
                    return cachedDataLog2;
                case 'log10':
                    if (cachedDataLog10 === null) {{
                        console.log('Computing log₁₀ transform...');
                        cachedDataLog10 = dataLinear.map(row => row.map(v => {{
                            // Handle negative values: sign(v) * log10(|v| + 1)
                            if (v < 0) return -Math.log10(-v + 1);
                            return Math.log10(v + 1);
                        }}));
                    }}
                    return cachedDataLog10;
                case 'sqrt':
                    if (cachedDataSqrt === null) {{
                        console.log('Computing √ transform...');
                        cachedDataSqrt = dataLinear.map(row => row.map(v => {{
                            // Handle negative values: sign(v) * sqrt(|v|)
                            if (v < 0) return -Math.sqrt(-v);
                            return Math.sqrt(v);
                        }}));
                    }}
                    return cachedDataSqrt;
                default:
                    return dataLinear;
            }}
        }}
        
        function getScaleLabel(scale) {{
            switch(scale) {{
                case 'log2': return ' (log₂)';
                case 'log10': return ' (log₁₀)';
                case 'sqrt': return ' (√)';
                default: return '';
            }}
        }}
        
        function getDataRange(data) {{
            let min = Infinity;
            let max = -Infinity;
            for (let row of data) {{
                for (let val of row) {{
                    if (val < min) min = val;
                    if (val > max) max = val;
                }}
            }}
            return {{min, max}};
        }}
        
        function reorderData(data, rowOrder, colOrder) {{
            // Reorder rows and columns of the data matrix according to given orders
            const reordered = new Array(rowOrder.length);
            for (let i = 0; i < rowOrder.length; i++) {{
                reordered[i] = new Array(colOrder.length);
                for (let j = 0; j < colOrder.length; j++) {{
                    reordered[i][j] = data[rowOrder[i]][colOrder[j]];
                }}
            }}
            return reordered;
        }}
        
        function reorderLabels(labels, order) {{
            // Reorder labels according to given order
            const reordered = new Array(order.length);
            for (let i = 0; i < order.length; i++) {{
                reordered[i] = labels[order[i]];
            }}
            return reordered;
        }}
        
        function reorderHoverText(hoverText, rowOrder, colOrder) {{
            // Reorder hover text according to given orders
            if (hoverText === null) return null;
            const reordered = new Array(rowOrder.length);
            for (let i = 0; i < rowOrder.length; i++) {{
                reordered[i] = new Array(colOrder.length);
                for (let j = 0; j < colOrder.length; j++) {{
                    reordered[i][j] = hoverText[rowOrder[i]][colOrder[j]];
                }}
            }}
            return reordered;
        }}
        
        function createHeatmap() {{
            // Safety check: ensure data is available
            if (!dataLinear || dataLinear.length === 0) {{
                console.error('Cannot create heatmap: data not available');
                return;
            }}
            
            let data = getDataForScale(currentScale);
            let dataOriginal = dataLinear.map(row => row.slice()); // Keep original for cell values
            const scaleLabel = getScaleLabel(currentScale);
            
            // Determine which labels to use based on transpose state
            let displayXLabels, displayYLabels;
            let currentHoverText = generateHoverText();
            
            if (isTransposed) {{
                // When transposed: rows become columns, columns become rows
                // So we use the swapped tracking variables
                displayXLabels = currentYLabels.slice();
                displayYLabels = currentXLabels.slice();
                
                // Transpose the data matrix
                data = data[0].map((_, colIndex) => data.map(row => row[colIndex]));
                dataOriginal = dataOriginal[0].map((_, colIndex) => dataOriginal.map(row => row[colIndex]));
                
                // Transpose hover text if available
                if (currentHoverText !== null) {{
                    currentHoverText = currentHoverText[0].map((_, colIndex) => 
                        currentHoverText.map(row => row[colIndex])
                    );
                }}
                
                // Now apply reordering based on current tracked order (already transposed)
                const baseXLabels = yLabels;
                const baseYLabels = xLabels;
                
                const rowMapping = displayYLabels.map(label => baseYLabels.indexOf(label));
                const colMapping = displayXLabels.map(label => baseXLabels.indexOf(label));
                
                // Reorder transposed data
                data = rowMapping.map(rowIdx => 
                    colMapping.map(colIdx => data[rowIdx][colIdx])
                );
                dataOriginal = rowMapping.map(rowIdx => 
                    colMapping.map(colIdx => dataOriginal[rowIdx][colIdx])
                );
                
                // Reorder hover text if available
                if (currentHoverText !== null) {{
                    currentHoverText = rowMapping.map(rowIdx => 
                        colMapping.map(colIdx => currentHoverText[rowIdx][colIdx])
                    );
                }}
            }} else {{
                // Normal (non-transposed) mode
                displayXLabels = currentXLabels.slice();
                displayYLabels = currentYLabels.slice();
                
                // Apply reordering if different from base labels
                const baseXLabels = xLabels;
                const baseYLabels = yLabels;
                
                const needsRowReorder = !arraysEqual(displayYLabels, baseYLabels);
                const needsColReorder = !arraysEqual(displayXLabels, baseXLabels);
                
                if (needsRowReorder || needsColReorder) {{
                    const rowMapping = displayYLabels.map(label => baseYLabels.indexOf(label));
                    const colMapping = displayXLabels.map(label => baseXLabels.indexOf(label));
                    
                    // Reorder data matrix
                    data = rowMapping.map(rowIdx => 
                        colMapping.map(colIdx => data[rowIdx][colIdx])
                    );
                    dataOriginal = rowMapping.map(rowIdx => 
                        colMapping.map(colIdx => dataOriginal[rowIdx][colIdx])
                    );
                    
                    // Reorder hover text if available
                    if (currentHoverText !== null) {{
                        currentHoverText = rowMapping.map(rowIdx => 
                            colMapping.map(colIdx => currentHoverText[rowIdx][colIdx])
                        );
                    }}
                }}
            }}
            
            // Apply clustering reordering if enabled (after transpose and custom reordering)
            if (useClusteredOrder && clusteringAvailable) {{
                // Get clustering results for the selected method
                const selectedMethod = clusteringResults[currentClusteringMethod];
                let methodRowOrder = rowOrderClustered;
                let methodColOrder = colOrderClustered;
                
                if (selectedMethod) {{
                    methodRowOrder = selectedMethod.row_order;
                    methodColOrder = selectedMethod.col_order;
                }} else {{
                    console.warn('Clustering method not found:', currentClusteringMethod, '- using default');
                }}
                
                // When transposed, swap the cluster orders to match the transposed dimensions
                const effectiveRowOrder = isTransposed ? methodColOrder : methodRowOrder;
                const effectiveColOrder = isTransposed ? methodRowOrder : methodColOrder;
                
                data = reorderData(data, effectiveRowOrder, effectiveColOrder);
                dataOriginal = reorderData(dataOriginal, effectiveRowOrder, effectiveColOrder);
                displayXLabels = reorderLabels(displayXLabels, effectiveColOrder);
                displayYLabels = reorderLabels(displayYLabels, effectiveRowOrder);
                // Reorder hover text if available
                if (currentHoverText !== null) {{
                    currentHoverText = reorderHoverText(currentHoverText, effectiveRowOrder, effectiveColOrder);
                }}
            }}
            
            // Apply data filter if active (hide rows/columns based on their max values)
            if (dataFilterActive && filteredRowIndices.length > 0 && filteredColIndices.length > 0) {{
                // Filter data matrix
                data = filteredRowIndices.map(rowIdx => 
                    filteredColIndices.map(colIdx => data[rowIdx][colIdx])
                );
                dataOriginal = filteredRowIndices.map(rowIdx => 
                    filteredColIndices.map(colIdx => dataOriginal[rowIdx][colIdx])
                );
                
                // Filter labels
                displayXLabels = filteredColIndices.map(idx => displayXLabels[idx]);
                displayYLabels = filteredRowIndices.map(idx => displayYLabels[idx]);
                
                // Filter hover text if available
                if (currentHoverText !== null) {{
                    currentHoverText = filteredRowIndices.map(rowIdx => 
                        filteredColIndices.map(colIdx => currentHoverText[rowIdx][colIdx])
                    );
                }}
                
                console.log(`Data filter: showing ${{filteredRowIndices.length}} rows × ${{filteredColIndices.length}} cols`);
            }}
            
            const range = getDataRange(data);
            
            // Determine which colorscale to use
            let colorscaleToUse;
            
            // Check if we should use custom colorscale
            if (currentColorscale === 'Custom' && customColorScale && Array.isArray(customColorScale) && customColorScale.length > 0) {{
                // Use the custom colorscale array directly
                colorscaleToUse = customColorScale;
                console.log('✓ createHeatmap: Using CUSTOM colorscale:', {{
                    scale: customColorScale,
                    length: customColorScale.length,
                    positions: customColorScale.map(c => c[0])
                }});
            }} else {{
                // For preset colorscales, convert to array format for Plotly compatibility
                // Plotly v1.58.5 doesn't recognize all colorscale names, so we define them explicitly
                colorscaleToUse = getPlotlyColorscaleArray(currentColorscale);
                console.log('createHeatmap: Using preset colorscale:', {{
                    name: currentColorscale,
                    isArray: Array.isArray(colorscaleToUse),
                    length: Array.isArray(colorscaleToUse) ? colorscaleToUse.length : 'N/A'
                }});
            }}
            
            // Get metric display name for colorbar
            const metricDisplayNames = {{
                'weight': 'Synapses',
                'ratio': 'Ratio',
                'probability': 'Probability'
            }};
            const metricDisplayName = metricDisplayNames[currentMetric] || currentMetric;
            
            const trace = {{
                z: data,
                x: displayXLabels.map((_, i) => i),  // Use indices for positioning
                y: displayYLabels.map((_, i) => i),  // Use indices for positioning
                type: 'heatmap',
                colorscale: colorscaleToUse,
                colorbar: {{
                    title: metricDisplayName + scaleLabel,
                    titleside: 'right'
                }}
            }};
            
            // Configure text display for cell values
            console.log('createHeatmap: showCellValues =', showCellValues);
            if (showCellValues) {{
                // Show cell values: use texttemplate to display z values
                console.log('Setting texttemplate to show cell values');
                
                // Create a text array from the data for display
                const textArray = data.map(row => row.map(val => val.toString()));
                
                trace.text = textArray;  // Text array for display
                trace.texttemplate = '%{{text}}';  // Use the text array
                trace.textfont = {{
                    size: Math.max(8, Math.min(16, currentFontSize * 0.8))
                }};
                // For hover, use the detailed hover text
                trace.hovertext = currentHoverText;
                trace.hoverinfo = 'text';
                trace.hovertemplate = '%{{hovertext}}<extra></extra>';  // <extra></extra> hides "trace 0"
            }} else {{
                // Hide cell values: no texttemplate, only hover text
                console.log('NOT setting texttemplate - hiding cell values');
                trace.text = currentHoverText;  // Text for hover only
                trace.hoverinfo = 'text';  // Show hover text on hover
                trace.hovertemplate = '%{{text}}<extra></extra>';  // <extra></extra> hides "trace 0"
            }}
            console.log('trace texttemplate:', trace.texttemplate);
            console.log('trace text sample:', trace.text ? trace.text[0] : 'none');
            
            // Apply custom colorbar range
            // Priority: 1) Custom color range (for cross-heatmap comparison)
            //           2) Manual slider range (if not auto)
            //           3) Auto range (default)
            if (window.customColorRange) {{
                trace.zmin = window.customColorRange.min;
                trace.zmax = window.customColorRange.max;
                console.log('Using custom color range:', window.customColorRange);
            }} else if (!useAutoRange && customZmin !== null && customZmax !== null) {{
                trace.zmin = customZmin;
                trace.zmax = customZmax;
            }}
            
            // Store current range for slider scaling
            window.currentDataRange = range;
            
            // Update 2-point color value inputs to show current data range in auto mode
            if (!window.customColorRange) {{
                document.getElementById('valueMin2').value = formatValueDisplay(range.min);
                document.getElementById('valueMax2').value = formatValueDisplay(range.max);
            }}
            
            // Determine axis titles based on transpose state
            const xAxisLabel = isTransposed ? 'Source' : 'Target';
            const yAxisLabel = isTransposed ? 'Target' : 'Source';
            const xAxisCount = displayXLabels.length;
            const yAxisCount = displayYLabels.length;
            
            const layout = {{
                title: originalTitle,
                font: {{size: currentFontSize}},
                autosize: false,
                xaxis: {{
                    title: isLarge ? '<b>' + xAxisLabel + '</b> (' + xAxisCount + ' neurons)' : '<b>' + xAxisLabel + '</b>',
                    side: 'bottom',
                    titlefont: {{size: currentFontSize + 2, color: '#333333'}},
                    tickangle: displayXLabels.length > 1 ? -45 : 0,  // Always rotate when multiple labels
                    showticklabels: showLabels,
                    tickmode: 'array',  // Use explicit tick values
                    tickvals: displayXLabels.map((_, i) => i),  // Use indices as tick positions
                    ticktext: displayXLabels  // Use labels as tick text
                }},
                yaxis: {{
                    title: isLarge ? '<b>' + yAxisLabel + '</b> (' + yAxisCount + ' neurons)' : '<b>' + yAxisLabel + '</b>',
                    side: 'left',
                    titlefont: {{size: currentFontSize + 2, color: '#333333'}},
                    autorange: 'reversed',
                    showticklabels: showLabels,
                    tickmode: 'array',  // Use explicit tick values
                    tickvals: displayYLabels.map((_, i) => i),  // Use indices as tick positions
                    ticktext: displayYLabels  // Use labels as tick text
                }},
                hoverlabel: {{
                    bgcolor: 'white',
                    font_size: 12,
                    font_family: 'Arial'
                }},
                width: currentWidth,
                height: currentHeight,
                margin: {{l: 120, r: 40, b: 120, t: 100, pad: 4}}
            }};
            
            // Add compression hint in title for user awareness
            if (is_sparse && !use_scatter_mode) {{
                const sparsity_pct = Math.round(sparsity * 100);
                layout.title.text += `<br><sub style='color:#666;'>Matrix ${{
                    sparsity_pct
                }}% sparse - optimized for file size</sub>`;
            }}
            
            // For scatter mode, lock aspect ratio and adjust margins
            if (use_scatter_mode) {{
                layout.xaxis.constrain = 'domain';
                layout.yaxis.scaleanchor = 'x';
                layout.plot_bgcolor = 'white';
                layout.xaxis.showgrid = true;
                layout.yaxis.showgrid = true;
                layout.xaxis.gridcolor = 'rgba(0,0,0,0.1)';
                layout.yaxis.gridcolor = 'rgba(0,0,0,0.1)';
                layout.margin.t = 100;
                layout.margin.b = 120;
            }}
            
            // Create or update the heatmap using Plotly.react (handles both creation and updates)
            Plotly.react('heatmap', [trace], layout);
        }}
        
        function toggleClustering(mode) {{
            // Toggle between original and clustered ordering
            useClusteredOrder = (mode === 'clustered');
            
            // Update button states
            document.getElementById('btn-original').classList.toggle('active', mode === 'original');
            document.getElementById('btn-clustered').classList.toggle('active', mode === 'clustered');
            
            // Show/hide clustering method selector
            const methodSection = document.getElementById('clusteringMethodSection');
            if (methodSection) {{
                methodSection.style.display = (mode === 'clustered' && clusteringAvailable) ? 'block' : 'none';
            }}
            
            // If clustering is not available, show message and revert
            if (mode === 'clustered' && !clusteringAvailable) {{
                alert('Clustering is not available for this matrix. Using original order.');
                useClusteredOrder = false;
                document.getElementById('btn-original').classList.add('active');
                document.getElementById('btn-clustered').classList.remove('active');
                return;
            }}
            
            // Update data filter state (disables when clustering is active)
            applyDataFilter();
            
            // Recreate heatmap with new ordering
            createHeatmap();
        }}
        
        function updateClusteringMethod() {{
            // Get selected clustering method
            const methodSelect = document.getElementById('clusteringMethodSelect');
            currentClusteringMethod = methodSelect.value;
            
            console.log('Switching to clustering method:', currentClusteringMethod);
            
            // Update the heatmap with new clustering method
            if (useClusteredOrder) {{
                createHeatmap();
            }}
        }}
        
        function setScale(scale) {{
            currentScale = scale;
            
            // Update button states
            document.querySelectorAll('[id^="btn-"]').forEach(btn => {{
                btn.classList.remove('active');
            }});
            document.getElementById('btn-' + scale).classList.add('active');
            
            createHeatmap();
        }}
        
        function updateMetric() {{
            // Switch to the selected metric
            currentMetric = document.getElementById('metricSelect').value;
            console.log('Switching to metric:', currentMetric);
            
            // Update dataLinear with the new metric's data (always use original ordering)
            dataLinear = metricsData[currentMetric];
            
            // Clear cached transforms so they're recomputed for new metric
            cachedDataLog2 = null;
            cachedDataLog10 = null;
            cachedDataSqrt = null;
            
            // Recreate the heatmap with new metric data (clustering will be applied in createHeatmap)
            createHeatmap();
        }}
        
        function updateColorscale() {{
            currentColorscale = document.getElementById('colorscaleSelect').value;
            
            // If switching to Custom and no custom scale exists, create default
            if (currentColorscale === 'Custom' && !customColorScale) {{
                applyCustomColors();
            }}
            
            createHeatmap();
        }}
        
        function toggleCustomColorPanel() {{
            const panel = document.getElementById('customColorPanel');
            if (panel.style.display === 'none') {{
                panel.style.display = 'block';
            }} else {{
                panel.style.display = 'none';
            }}
        }}
        
        function toggle3PointScale() {{
            use3PointScale = document.getElementById('use3PointScale').checked;
            const twoPoint = document.getElementById('twoPointColors');
            const threePoint = document.getElementById('threePointColors');
            
            if (use3PointScale) {{
                twoPoint.style.display = 'none';
                threePoint.style.display = 'block';
                
                // Set default values based on current data range
                if (window.currentDataRange) {{
                    const range = window.currentDataRange;
                    const mid = (range.min + range.max) / 2;
                    document.getElementById('valueMin3').value = formatValueDisplay(range.min);
                    document.getElementById('valueMid3').value = formatValueDisplay(mid);
                    document.getElementById('valueMax3').value = formatValueDisplay(range.max);
                }}
            }} else {{
                twoPoint.style.display = 'block';
                threePoint.style.display = 'none';
            }}
        }}
        
        function rgbToPlotlyFormat(hex) {{
            // Convert hex color to RGB format for Plotly
            const r = parseInt(hex.slice(1, 3), 16);
            const g = parseInt(hex.slice(3, 5), 16);
            const b = parseInt(hex.slice(5, 7), 16);
            return `rgb(${{r}},${{g}},${{b}})`;
        }}
        
        // Helper function to compare two arrays for equality
        function arraysEqual(arr1, arr2) {{
            if (arr1.length !== arr2.length) return false;
            for (let i = 0; i < arr1.length; i++) {{
                if (arr1[i] !== arr2[i]) return false;
            }}
            return true;
        }}
        
        function formatValueDisplay(value) {{
            // Format number to remove trailing zeros and unnecessary decimal point
            // Examples: 0.000000 -> "0", 250.123456 -> "250.123456", 1.500000 -> "1.5"
            if (value === 0) return "0";
            const str = value.toFixed(6);
            // Remove trailing zeros and decimal point if not needed
            return str.replace(/\.?0+$/, '');
        }}
        
        function applyCustomColors() {{
            if (use3PointScale) {{
                // 3-point scale with custom value mapping
                const colorMin = document.getElementById('colorMin3').value;
                const colorMid = document.getElementById('colorMid3').value;
                const colorMax = document.getElementById('colorMax3').value;
                
                const valueMin = parseFloat(document.getElementById('valueMin3').value);
                const valueMid = parseFloat(document.getElementById('valueMid3').value);
                const valueMax = parseFloat(document.getElementById('valueMax3').value);
                
                // Get current data range
                const range = window.currentDataRange;
                if (!range) {{
                    alert('Please wait for data to load before applying custom colors.');
                    return;
                }}
                
                // Map custom values to [0, 1] range - allows values beyond actual data range
                const normalizeValue = (val, rangeMin, rangeMax) => {{
                    if (rangeMax === rangeMin) return 0.5;
                    return (val - rangeMin) / (rangeMax - rangeMin);
                }};
                
                // Use custom value range for normalization (allows cross-heatmap comparison)
                const customRangeMin = valueMin;
                const customRangeMax = valueMax;
                
                if (customRangeMax === customRangeMin) {{
                    alert('Custom min and max values cannot be the same.');
                    return;
                }}
                
                // Map custom value points to [0, 1] colorscale positions
                // This defines where each color appears on the scale
                const posMid = normalizeValue(valueMid, customRangeMin, customRangeMax);
                
                // Clamp mid position to [0, 1]
                const clampedPosMid = Math.max(0, Math.min(1, posMid));
                
                // Create color scale array spanning 0 to 1
                // Plotly will map data values to this scale based on customColorRange
                customColorScale = [
                    [0, rgbToPlotlyFormat(colorMin)],
                    [clampedPosMid, rgbToPlotlyFormat(colorMid)],
                    [1, rgbToPlotlyFormat(colorMax)]
                ];
                
                // Set custom range for Plotly to use
                window.customColorRange = {{min: valueMin, max: valueMax}};
                
                // Sort by position (required by Plotly)
                customColorScale.sort((a, b) => a[0] - b[0]);
                
                // Ensure positions are distinct (avoid duplicates)
                const epsilon = 0.001;
                for (let i = 1; i < customColorScale.length; i++) {{
                    if (Math.abs(customColorScale[i][0] - customColorScale[i-1][0]) < epsilon) {{
                        customColorScale[i][0] = customColorScale[i-1][0] + epsilon;
                    }}
                }}
                
                console.log('Applied 3-point scale:', {{
                    inputs: {{
                        min: {{value: valueMin, color: colorMin}},
                        mid: {{value: valueMid, color: colorMid}},
                        max: {{value: valueMax, color: colorMax}}
                    }},
                    customRange: {{min: valueMin, max: valueMax}},
                    midPosition: clampedPosMid,
                    colorScale: customColorScale
                }});
            }} else {{
                // 2-point scale with optional custom value mapping
                const colorMin = document.getElementById('colorMin').value;
                const colorMax = document.getElementById('colorMax').value;
                
                const valueMin2Input = document.getElementById('valueMin2').value;
                const valueMax2Input = document.getElementById('valueMax2').value;
                
                // Check if custom values are specified
                if (valueMin2Input !== '' && valueMax2Input !== '') {{
                    // Use custom values for cross-heatmap comparison
                    const valueMin = parseFloat(valueMin2Input);
                    const valueMax = parseFloat(valueMax2Input);
                    
                    if (valueMax === valueMin) {{
                        alert('Custom min and max values cannot be the same.');
                        return;
                    }}
                    
                    // Colorscale spans from 0 to 1 (representing valueMin to valueMax)
                    // Plotly will map data values to this scale automatically
                    customColorScale = [
                        [0, rgbToPlotlyFormat(colorMin)],
                        [1, rgbToPlotlyFormat(colorMax)]
                    ];
                    
                    // Override the data normalization by setting colorscale range
                    window.customColorRange = {{min: valueMin, max: valueMax}};
                    
                    console.log('Applied 2-point scale with custom values:', {{
                        customRange: {{min: valueMin, max: valueMax}},
                        colorScale: customColorScale
                    }});
                }} else {{
                    // Auto mode: use full data range
                    customColorScale = [
                        [0, rgbToPlotlyFormat(colorMin)],
                        [1, rgbToPlotlyFormat(colorMax)]
                    ];
                    
                    // Clear custom range
                    window.customColorRange = null;
                    
                    console.log('Applied 2-point scale (auto):', customColorScale);
                }}
            }}
            
            // Switch to Custom colorscale and update
            currentColorscale = 'Custom';
            
            // Update dropdown without triggering the onchange handler
            const selectElement = document.getElementById('colorscaleSelect');
            const oldOnchange = selectElement.onchange;
            selectElement.onchange = null;
            selectElement.value = 'Custom';
            selectElement.onchange = oldOnchange;
            
            console.log('About to create heatmap with custom scale:', {{
                currentColorscale: currentColorscale,
                customColorScale: customColorScale,
                dropdownValue: selectElement.value
            }});
            
            createHeatmap();
        }}
        
        function resetToAutoColors() {{
            // Clear custom color range
            window.customColorRange = null;
            
            // Update value input boxes to show current data range
            const range = window.currentDataRange;
            if (range) {{
                document.getElementById('valueMin2').value = formatValueDisplay(range.min);
                document.getElementById('valueMax2').value = formatValueDisplay(range.max);
            }}
            
            // Recreate heatmap with auto colors
            createHeatmap();
            
            console.log('Reset to auto color mode');
        }}
        
        function updateFontSize(size) {{
            currentFontSize = parseInt(size);
            document.getElementById('fontSizeValue').textContent = size + 'px';
            createHeatmap();
        }}
        
        function toggleLabels() {{
            showLabels = !showLabels;
            const btn = document.getElementById('toggleLabelsBtn');
            btn.textContent = showLabels ? '🏷️ Hide Text' : '🏷️ Show Text';
            
            // Update the layout to hide/show ALL text elements including colorbar
            const gd = document.getElementById('heatmap');
            
            // Update colorbar text (trace-level property)
            const traceUpdate = {{
                'colorbar.title.text': showLabels ? (metricType.charAt(0).toUpperCase() + metricType.slice(1)) : '',
                'colorbar.showticklabels': showLabels
            }};
            
            // Update layout elements
            const layoutUpdate = {{
                'title.text': showLabels ? originalTitle : '',
                'xaxis.showticklabels': showLabels,
                'yaxis.showticklabels': showLabels,
                'xaxis.title.text': showLabels ? (isLarge ? '<b>Target</b> (' + gd.data[0].x.length + ' neurons)' : '<b>Target</b>') : '',
                'yaxis.title.text': showLabels ? (isLarge ? '<b>Source</b> (' + gd.data[0].y.length + ' neurons)' : '<b>Source</b>') : '',
                'xaxis.ticks': showLabels ? 'outside' : '',
                'yaxis.ticks': showLabels ? 'outside' : '',
                // Prevent autosize from expanding the plot
                'autosize': false,
                // Keep margins fixed to prevent rescaling
                'margin.l': 120,
                'margin.r': 40,
                'margin.t': 100,
                'margin.b': 120,
                // Preserve dimensions explicitly
                'width': currentWidth,
                'height': currentHeight
            }};
            
            // Add compression hint in title for user awareness
            if (is_sparse && !use_scatter_mode) {{
                const sparsity_pct = Math.round(sparsity * 100);
                layout.title.text += `<br><sub style='color:#666;'>Matrix ${{
                    sparsity_pct
                }}% sparse - optimized for file size</sub>`;
            }}
            
            // For scatter mode, lock aspect ratio and adjust margins
            if (use_scatter_mode) {{
                layout.xaxis.constrain = 'domain';
                layout.yaxis.scaleanchor = 'x';
                layout.plot_bgcolor = 'white';
                layout.xaxis.showgrid = true;
                layout.yaxis.showgrid = true;
                layout.xaxis.gridcolor = 'rgba(0,0,0,0.1)';
                layout.yaxis.gridcolor = 'rgba(0,0,0,0.1)';
                layout.margin.t = 100;
                layout.margin.b = 120;
            }}
            
            // Create or update the heatmap using Plotly.react (handles both creation and updates)
            Plotly.react('heatmap', [trace], layout);
        }}
        
        function toggleCellValues() {{
            showCellValues = !showCellValues;
            const btn = document.getElementById('toggleCellValuesBtn');
            btn.textContent = showCellValues ? '🔢 Hide Values' : '🔢 Show Values';
            
            console.log('toggleCellValues called, showCellValues is now:', showCellValues);
            
            // Recreate heatmap to add/remove cell value annotations
            createHeatmap();
        }}
        
        function updateCellValueSize(size) {{
            cellValueFontSize = parseInt(size);
            document.getElementById('cellValueSizeValue').textContent = cellValueFontSize + 'px';
            
            // Only recreate if cell values are currently shown
            if (showCellValues) {{
                createHeatmap();
            }}
        }}
        
        function updateContrastThreshold(value) {{
            contrastThreshold = parseFloat(value);
            document.getElementById('contrastThresholdValue').textContent = contrastThreshold.toFixed(4);
            console.log('Contrast threshold updated to:', contrastThreshold);
            
            // Recreate heatmap if cell values are currently shown
            if (showCellValues) {{
                createHeatmap();
            }}
        }}
        
        function reverseContrastColors() {{
            reverseContrast = !reverseContrast;
            console.log('Contrast colors reversed:', reverseContrast);
            
            // Recreate heatmap if cell values are currently shown
            if (showCellValues) {{
                createHeatmap();
            }}
        }}
        
        function updateIgnoredValues() {{
            const input = document.getElementById('ignoreValuesInput');
            const expressions = input.value.split(',').map(v => v.trim()).filter(v => v !== '');
            
            // Clear and repopulate the ignored values array
            // Store both exact values and comparison expressions
            ignoredValues.clear();
            ignoredValues.expressions = [];  // Array to store comparison expressions
            
            expressions.forEach(expr => {{
                // Check if it's a comparison expression (>, <, >=, <=)
                const compMatch = expr.match(/^([><]=?|==|!=)\\s*(-?\\d+\\.?\\d*)$/);
                if (compMatch) {{
                    // It's a comparison expression
                    const operator = compMatch[1];
                    const threshold = parseFloat(compMatch[2]);
                    ignoredValues.expressions.push({{ operator, threshold }});
                }} else {{
                    // Try to parse as exact number
                    const num = parseFloat(expr);
                    if (!isNaN(num)) {{
                        ignoredValues.add(num);
                    }}
                }}
            }});
            
            console.log('Ignored exact values:', Array.from(ignoredValues));
            console.log('Ignored expressions:', ignoredValues.expressions);
            
            // Recreate heatmap if cell values are shown
            if (showCellValues) {{
                createHeatmap();
            }}
        }}
        
        function shouldIgnoreValue(value) {{
            // Check if value matches any exact value
            if (ignoredValues.has(value)) {{
                return true;
            }}
            
            // Check if value matches any comparison expression
            if (ignoredValues.expressions && ignoredValues.expressions.length > 0) {{
                for (const expr of ignoredValues.expressions) {{
                    let matches = false;
                    switch (expr.operator) {{
                        case '>':
                            matches = value > expr.threshold;
                            break;
                        case '<':
                            matches = value < expr.threshold;
                            break;
                        case '>=':
                            matches = value >= expr.threshold;
                            break;
                        case '<=':
                            matches = value <= expr.threshold;
                            break;
                        case '==':
                            matches = value === expr.threshold;
                            break;
                        case '!=':
                            matches = value !== expr.threshold;
                            break;
                    }}
                    if (matches) {{
                        return true;
                    }}
                }}
            }}
            
            return false;
        }}
        
        // ===== DATA FILTER FUNCTIONS =====
        // Filter entire rows/columns based on their maximum values
        
        function parseFilterExpressions(inputString) {{
            const expressions = inputString.split(',').map(v => v.trim()).filter(v => v !== '');
            const parsedExpressions = [];
            
            expressions.forEach(expr => {{
                // Check if it's a comparison expression (>, <, >=, <=, ==, !=)
                const compMatch = expr.match(/^([><]=?|==|!=)\\s*(-?\\d+\\.?\\d*)$/);
                if (compMatch) {{
                    const operator = compMatch[1];
                    const threshold = parseFloat(compMatch[2]);
                    parsedExpressions.push({{ operator, threshold }});
                }} else {{
                    // Try to parse as exact number (will hide if max == this value)
                    const num = parseFloat(expr);
                    if (!isNaN(num)) {{
                        parsedExpressions.push({{ operator: '==', threshold: num }});
                    }}
                }}
            }});
            
            return parsedExpressions;
        }}
        
        function shouldHideRowOrColumn(maxValue, expressions) {{
            if (expressions.length === 0) return false;
            
            for (const expr of expressions) {{
                let matches = false;
                switch (expr.operator) {{
                    case '>':
                        matches = maxValue > expr.threshold;
                        break;
                    case '<':
                        matches = maxValue < expr.threshold;
                        break;
                    case '>=':
                        matches = maxValue >= expr.threshold;
                        break;
                    case '<=':
                        matches = maxValue <= expr.threshold;
                        break;
                    case '==':
                        matches = maxValue === expr.threshold;
                        break;
                    case '!=':
                        matches = maxValue !== expr.threshold;
                        break;
                }}
                if (matches) {{
                    return true;  // Hide if any expression matches
                }}
            }}
            
            return false;
        }}
        
        function applyDataFilter() {{
            const input = document.getElementById('dataFilterInput');
            const statusDiv = document.getElementById('filterStatus');
            const filterValue = input.value.trim();
            
            // Disable data filtering when clustering is active
            if (useClusteredOrder && clusteringAvailable) {{
                statusDiv.textContent = '⚠️ Data filter disabled during clustering';
                statusDiv.style.color = '#ff9800';
                input.disabled = true;
                dataFilterActive = false;
                dataFilterExpressions = [];
                filteredRowIndices = [];
                filteredColIndices = [];
                return;
            }} else {{
                input.disabled = false;
            }}
            
            if (!filterValue) {{
                // No filter - show all rows/columns
                dataFilterActive = false;
                dataFilterExpressions = [];
                filteredRowIndices = [];
                filteredColIndices = [];
                statusDiv.textContent = '';
                createHeatmap();
                return;
            }}
            
            // Parse filter expressions
            dataFilterExpressions = parseFilterExpressions(filterValue);
            
            if (dataFilterExpressions.length === 0) {{
                statusDiv.textContent = '⚠️ Invalid filter format';
                statusDiv.style.color = '#d32f2f';
                return;
            }}
            
            // Get current data based on scale
            let currentData = dataLinear;
            if (currentScale === 'log2' && (useLazyTransforms ? cachedDataLog2 : dataLog2)) {{
                currentData = useLazyTransforms ? cachedDataLog2 : dataLog2;
            }} else if (currentScale === 'log10' && (useLazyTransforms ? cachedDataLog10 : dataLog10)) {{
                currentData = useLazyTransforms ? cachedDataLog10 : dataLog10;
            }} else if (currentScale === 'sqrt' && (useLazyTransforms ? cachedDataSqrt : dataSqrt)) {{
                currentData = useLazyTransforms ? cachedDataSqrt : dataSqrt;
            }}
            
            // Use original unscaled data for filtering
            const filterData = metricsData[currentMetric];
            
            const nRows = filterData.length;
            const nCols = filterData[0].length;
            
            // Calculate max value for each row and column
            const rowMaxValues = new Array(nRows).fill(-Infinity);
            const colMaxValues = new Array(nCols).fill(-Infinity);
            
            for (let i = 0; i < nRows; i++) {{
                for (let j = 0; j < nCols; j++) {{
                    const value = filterData[i][j];
                    if (value > rowMaxValues[i]) rowMaxValues[i] = value;
                    if (value > colMaxValues[j]) colMaxValues[j] = value;
                }}
            }}
            
            // Determine which rows and columns to keep
            filteredRowIndices = [];
            filteredColIndices = [];
            
            for (let i = 0; i < nRows; i++) {{
                if (!shouldHideRowOrColumn(rowMaxValues[i], dataFilterExpressions)) {{
                    filteredRowIndices.push(i);
                }}
            }}
            
            for (let j = 0; j < nCols; j++) {{
                if (!shouldHideRowOrColumn(colMaxValues[j], dataFilterExpressions)) {{
                    filteredColIndices.push(j);
                }}
            }}
            
            dataFilterActive = true;
            
            const hiddenRows = nRows - filteredRowIndices.length;
            const hiddenCols = nCols - filteredColIndices.length;
            
            if (filteredRowIndices.length === 0 || filteredColIndices.length === 0) {{
                statusDiv.textContent = '⚠️ Filter hides all data!';
                statusDiv.style.color = '#d32f2f';
                dataFilterActive = false;
                return;
            }}
            
            statusDiv.textContent = `✓ Showing ${{filteredRowIndices.length}}/${{nRows}} rows, ${{filteredColIndices.length}}/${{nCols}} cols`;
            statusDiv.style.color = '#2e7d32';
            
            console.log(`Data filter applied: hiding ${{hiddenRows}} rows and ${{hiddenCols}} cols`);
            
            createHeatmap();
        }}
        
        function resetDataFilter() {{
            document.getElementById('dataFilterInput').value = '';
            document.getElementById('filterStatus').textContent = '';
            dataFilterActive = false;
            dataFilterExpressions = [];
            filteredRowIndices = [];
            filteredColIndices = [];
            createHeatmap();
        }}
        
        // ===== END DATA FILTER FUNCTIONS =====
        
        function getPlotlyColorscaleArray(scaleName) {{
            // Return colorscale array for Plotly heatmap
            // Plotly v1.58.5 doesn't recognize all colorscale names, so we define them as arrays
            const colorscales = {{
                'Greens': [
                    [0.0, 'rgb(247,252,245)'],
                    [0.125, 'rgb(229,245,224)'],
                    [0.25, 'rgb(199,233,192)'],
                    [0.375, 'rgb(161,217,155)'],
                    [0.5, 'rgb(116,196,118)'],
                    [0.625, 'rgb(65,171,93)'],
                    [0.75, 'rgb(35,139,69)'],
                    [0.875, 'rgb(0,109,44)'],
                    [1.0, 'rgb(0,68,27)']
                ],
                'Blues': [
                    [0.0, 'rgb(247,251,255)'],
                    [0.125, 'rgb(222,235,247)'],
                    [0.25, 'rgb(198,219,239)'],
                    [0.375, 'rgb(158,202,225)'],
                    [0.5, 'rgb(107,174,214)'],
                    [0.625, 'rgb(66,146,198)'],
                    [0.75, 'rgb(33,113,181)'],
                    [0.875, 'rgb(8,81,156)'],
                    [1.0, 'rgb(8,48,107)']
                ],
                'Reds': [
                    [0.0, 'rgb(255,245,240)'],
                    [0.125, 'rgb(254,224,210)'],
                    [0.25, 'rgb(252,187,161)'],
                    [0.375, 'rgb(252,146,114)'],
                    [0.5, 'rgb(251,106,74)'],
                    [0.625, 'rgb(239,59,44)'],
                    [0.75, 'rgb(203,24,29)'],
                    [0.875, 'rgb(165,15,21)'],
                    [1.0, 'rgb(103,0,13)']
                ],
                'Purples': [
                    [0.0, 'rgb(252,251,253)'],
                    [0.125, 'rgb(239,237,245)'],
                    [0.25, 'rgb(218,218,235)'],
                    [0.375, 'rgb(188,189,220)'],
                    [0.5, 'rgb(158,154,200)'],
                    [0.625, 'rgb(128,125,186)'],
                    [0.75, 'rgb(106,81,163)'],
                    [0.875, 'rgb(84,39,143)'],
                    [1.0, 'rgb(63,0,125)']
                ],
                'Oranges': [
                    [0.0, 'rgb(255,245,235)'],
                    [0.125, 'rgb(254,230,206)'],
                    [0.25, 'rgb(253,208,162)'],
                    [0.375, 'rgb(253,174,107)'],
                    [0.5, 'rgb(253,141,60)'],
                    [0.625, 'rgb(241,105,19)'],
                    [0.75, 'rgb(217,72,1)'],
                    [0.875, 'rgb(166,54,3)'],
                    [1.0, 'rgb(127,39,4)']
                ],
                'Viridis': [
                    [0, 'rgb(68,1,84)'],
                    [0.25, 'rgb(59,82,139)'],
                    [0.5, 'rgb(33,145,140)'],
                    [0.75, 'rgb(94,201,98)'],
                    [1, 'rgb(253,231,37)']
                ],
                'Plasma': [
                    [0, 'rgb(13,8,135)'],
                    [0.25, 'rgb(126,3,168)'],
                    [0.5, 'rgb(204,71,120)'],
                    [0.75, 'rgb(248,149,64)'],
                    [1, 'rgb(240,249,33)']
                ],
                'Inferno': [
                    [0, 'rgb(0,0,4)'],
                    [0.25, 'rgb(87,16,110)'],
                    [0.5, 'rgb(188,55,84)'],
                    [0.75, 'rgb(249,142,9)'],
                    [1, 'rgb(252,255,164)']
                ],
                'Magma': [
                    [0, 'rgb(0,0,4)'],
                    [0.25, 'rgb(81,18,124)'],
                    [0.5, 'rgb(182,54,121)'],
                    [0.75, 'rgb(251,136,97)'],
                    [1, 'rgb(252,253,191)']
                ],
                'Cividis': [
                    [0, 'rgb(0,32,76)'],
                    [0.25, 'rgb(0,79,110)'],
                    [0.5, 'rgb(53,133,136)'],
                    [0.75, 'rgb(149,189,161)'],
                    [1, 'rgb(253,231,37)']
                ],
                'Hot': [
                    [0, 'rgb(0,0,0)'],
                    [0.33, 'rgb(255,0,0)'],
                    [0.66, 'rgb(255,255,0)'],
                    [1, 'rgb(255,255,255)']
                ],
                'Jet': [
                    [0, 'rgb(0,0,143)'],
                    [0.25, 'rgb(0,159,255)'],
                    [0.5, 'rgb(0,255,0)'],
                    [0.75, 'rgb(255,159,0)'],
                    [1, 'rgb(143,0,0)']
                ],
                'RdBu': [
                    [0, 'rgb(5,10,172)'],
                    [0.35, 'rgb(106,137,247)'],
                    [0.5, 'rgb(190,190,190)'],
                    [0.65, 'rgb(220,170,132)'],
                    [1, 'rgb(178,10,28)']
                ],
                'RdYlGn': [
                    [0, 'rgb(165,0,38)'],
                    [0.25, 'rgb(253,174,97)'],
                    [0.5, 'rgb(255,255,191)'],
                    [0.75, 'rgb(166,217,106)'],
                    [1, 'rgb(0,104,55)']
                ]
            }};
            
            // Return the colorscale array, or fallback to the name string
            return colorscales[scaleName] || scaleName;
        }}
        
        function getColorFromPlotlyScale(scaleName, normalized) {{
            // Map of Plotly colorscales to their RGB interpolations
            // These are approximations of Plotly's built-in scales
            const colorscales = {{
                'Greens': [
                    [0.0, 'rgb(247,252,245)'],
                    [0.125, 'rgb(229,245,224)'],
                    [0.25, 'rgb(199,233,192)'],
                    [0.375, 'rgb(161,217,155)'],
                    [0.5, 'rgb(116,196,118)'],
                    [0.625, 'rgb(65,171,93)'],
                    [0.75, 'rgb(35,139,69)'],
                    [0.875, 'rgb(0,109,44)'],
                    [1.0, 'rgb(0,68,27)']
                ],
                'Blues': [
                    [0.0, 'rgb(247,251,255)'],
                    [0.125, 'rgb(222,235,247)'],
                    [0.25, 'rgb(198,219,239)'],
                    [0.375, 'rgb(158,202,225)'],
                    [0.5, 'rgb(107,174,214)'],
                    [0.625, 'rgb(66,146,198)'],
                    [0.75, 'rgb(33,113,181)'],
                    [0.875, 'rgb(8,81,156)'],
                    [1.0, 'rgb(8,48,107)']
                ],
                'Reds': [
                    [0.0, 'rgb(255,245,240)'],
                    [0.125, 'rgb(254,224,210)'],
                    [0.25, 'rgb(252,187,161)'],
                    [0.375, 'rgb(252,146,114)'],
                    [0.5, 'rgb(251,106,74)'],
                    [0.625, 'rgb(239,59,44)'],
                    [0.75, 'rgb(203,24,29)'],
                    [0.875, 'rgb(165,15,21)'],
                    [1.0, 'rgb(103,0,13)']
                ],
                'Purples': [
                    [0.0, 'rgb(252,251,253)'],
                    [0.125, 'rgb(239,237,245)'],
                    [0.25, 'rgb(218,218,235)'],
                    [0.375, 'rgb(188,189,220)'],
                    [0.5, 'rgb(158,154,200)'],
                    [0.625, 'rgb(128,125,186)'],
                    [0.75, 'rgb(106,81,163)'],
                    [0.875, 'rgb(84,39,143)'],
                    [1.0, 'rgb(63,0,125)']
                ],
                'Oranges': [
                    [0.0, 'rgb(255,245,235)'],
                    [0.125, 'rgb(254,230,206)'],
                    [0.25, 'rgb(253,208,162)'],
                    [0.375, 'rgb(253,174,107)'],
                    [0.5, 'rgb(253,141,60)'],
                    [0.625, 'rgb(241,105,19)'],
                    [0.75, 'rgb(217,72,1)'],
                    [0.875, 'rgb(166,54,3)'],
                    [1.0, 'rgb(127,39,4)']
                ],
                'Viridis': [
                    [0, 'rgb(68,1,84)'],
                    [0.25, 'rgb(59,82,139)'],
                    [0.5, 'rgb(33,145,140)'],
                    [0.75, 'rgb(94,201,98)'],
                    [1, 'rgb(253,231,37)']
                ],
                'Plasma': [
                    [0, 'rgb(13,8,135)'],
                    [0.25, 'rgb(126,3,168)'],
                    [0.5, 'rgb(204,71,120)'],
                    [0.75, 'rgb(248,149,64)'],
                    [1, 'rgb(240,249,33)']
                ],
                'Inferno': [
                    [0, 'rgb(0,0,4)'],
                    [0.25, 'rgb(87,16,110)'],
                    [0.5, 'rgb(188,55,84)'],
                    [0.75, 'rgb(249,142,9)'],
                    [1, 'rgb(252,255,164)']
                ],
                'Magma': [
                    [0, 'rgb(0,0,4)'],
                    [0.25, 'rgb(81,18,124)'],
                    [0.5, 'rgb(182,54,121)'],
                    [0.75, 'rgb(251,136,97)'],
                    [1, 'rgb(252,253,191)']
                ],
                'Cividis': [
                    [0, 'rgb(0,32,76)'],
                    [0.25, 'rgb(0,79,110)'],
                    [0.5, 'rgb(53,133,136)'],
                    [0.75, 'rgb(149,189,161)'],
                    [1, 'rgb(253,231,37)']
                ],
                'Hot': [
                    [0, 'rgb(0,0,0)'],
                    [0.33, 'rgb(255,0,0)'],
                    [0.66, 'rgb(255,255,0)'],
                    [1, 'rgb(255,255,255)']
                ],
                'Jet': [
                    [0, 'rgb(0,0,143)'],
                    [0.25, 'rgb(0,159,255)'],
                    [0.5, 'rgb(0,255,0)'],
                    [0.75, 'rgb(255,159,0)'],
                    [1, 'rgb(143,0,0)']
                ],
                'RdBu': [
                    [0, 'rgb(5,10,172)'],
                    [0.35, 'rgb(106,137,247)'],
                    [0.5, 'rgb(190,190,190)'],
                    [0.65, 'rgb(220,170,132)'],
                    [1, 'rgb(178,10,28)']
                ],
                'RdYlGn': [
                    [0, 'rgb(165,0,38)'],
                    [0.25, 'rgb(253,174,97)'],
                    [0.5, 'rgb(255,255,191)'],
                    [0.75, 'rgb(166,217,106)'],
                    [1, 'rgb(0,104,55)']
                ]
            }};
            
            // Interpolate color from a custom colorscale array
            // colorscale format: [[0, 'color1'], [0.5, 'color2'], [1, 'color3'], ...]
            
            if (!Array.isArray(colorscale) || colorscale.length === 0) {{
                return 'rgb(128, 128, 128)';  // fallback gray
            }}
            
            // Handle edge cases
            if (normalized <= 0 || normalized <= colorscale[0][0]) {{
                return Array.isArray(colorscale[0]) && colorscale[0].length > 1 ? colorscale[0][1] : 'rgb(128, 128, 128)';
            }} else if (normalized >= 1 || normalized >= colorscale[colorscale.length - 1][0]) {{
                const last = colorscale[colorscale.length - 1];
                return Array.isArray(last) && last.length > 1 ? last[1] : 'rgb(128, 128, 128)';
            }}
            
            // Find the two color stops to interpolate between
            let lower = colorscale[0];
            let upper = colorscale[colorscale.length - 1];
            
            for (let i = 0; i < colorscale.length - 1; i++) {{
                if (normalized >= colorscale[i][0] && normalized <= colorscale[i + 1][0]) {{
                    lower = colorscale[i];
                    upper = colorscale[i + 1];
                    break;
                }}
            }}
            
            // Interpolate between the two colors
            const t = (normalized - lower[0]) / (upper[0] - lower[0]);
            const lowerRgb = hexToRgb(lower[1]);
            const upperRgb = hexToRgb(upper[1]);
            
            const r = Math.round(lowerRgb[0] + t * (upperRgb[0] - lowerRgb[0]));
            const g = Math.round(lowerRgb[1] + t * (upperRgb[1] - lowerRgb[1]));
            const b = Math.round(lowerRgb[2] + t * (upperRgb[2] - lowerRgb[2]));
            
            return `rgb(${{r}},${{g}},${{b}})`;
        }}
        
        function getContrastColor(rgb) {{
            // Calculate luminance from RGB color
            // If luminance is high (light background), use dark text; otherwise use light text
            const r = rgb[0];
            const g = rgb[1];
            const b = rgb[2];
            
            // Calculate relative luminance using the formula for sRGB
            const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            
            // Convert normalized threshold (0-1) to 0-255 range for comparison
            const threshold = contrastThreshold * 255;
            
            // Compare against the adjustable threshold
            // Normal: high luminance (light bg) → black text, low luminance (dark bg) → white text
            // Reverse: swap the logic
            if (reverseContrast) {{
                return luminance > threshold ? 'white' : 'black';
            }} else {{
                return luminance > threshold ? 'black' : 'white';
            }}
        }}
        
        function getColorForValue(value, zmin, zmax, colorscale) {{
            // Normalize value to 0-1 range
            const normalized = (value - zmin) / (zmax - zmin);
            
            // Get RGB color from the colorscale at the normalized position
            // This is a simplified version - Plotly has complex colorscale interpolation
            // For now, we'll sample the colorscale array
            if (Array.isArray(colorscale) && colorscale.length > 0) {{
                const idx = Math.floor(normalized * (colorscale.length - 1));
                const colorStop = colorscale[Math.max(0, Math.min(idx, colorscale.length - 1))];
                if (Array.isArray(colorStop) && colorStop.length > 1) {{
                    return colorStop[1];
                }}
            }}
            
            // Fallback: return a color based on normalized value
            if (normalized < 0.5) {{
                return `rgb(${{Math.round(normalized * 510)}}, ${{Math.round(normalized * 510)}}, 255)`;
            }} else {{
                return `rgb(255, ${{Math.round((1 - normalized) * 510)}}, ${{Math.round((1 - normalized) * 510)}})`;
            }}
        }}
        
        function hexToRgb(hex) {{
            // Convert hex color to RGB array
            if (hex.startsWith('#')) {{
                const result = /^#?([a-f\\d]{{2}})([a-f\\d]{{2}})([a-f\\d]{{2}})$/i.exec(hex);
                return result ? [
                    parseInt(result[1], 16),
                    parseInt(result[2], 16),
                    parseInt(result[3], 16)
                ] : [128, 128, 128];
            }} else if (hex.startsWith('rgb')) {{
                const match = hex.match(/\\d+/g);
                return match ? match.slice(0, 3).map(Number) : [128, 128, 128];
            }}
            return [128, 128, 128];
        }}
        
        function updatePlotSize() {{
            const gd = document.getElementById('heatmap');
            currentWidth = parseInt(document.getElementById('widthSlider').value);
            
            // If square cells are locked, auto-adjust height
            if (squareCellsLocked && gd.data && gd.data[0]) {{
                const numRows = gd.data[0].y.length;
                const numCols = gd.data[0].x.length;
                const margins = gd.layout.margin || {{l: 120, r: 40, b: 120, t: 100}};
                const marginHorizontal = margins.l + margins.r;
                const marginVertical = margins.t + margins.b;
                const plotAreaWidth = currentWidth - marginHorizontal;
                const plotAreaHeight = plotAreaWidth * numRows / numCols;
                currentHeight = Math.round(plotAreaHeight + marginVertical);
            }} else {{
                currentHeight = parseInt(document.getElementById('heightSlider').value);
            }}
            
            // Sync input boxes with sliders
            document.getElementById('widthInput').value = currentWidth;
            document.getElementById('heightInput').value = currentHeight;
            document.getElementById('widthValue').textContent = currentWidth + 'px';
            document.getElementById('heightValue').textContent = currentHeight + 'px';
            document.getElementById('heightSlider').value = Math.min(2400, Math.max(400, currentHeight));
            
            // Update the layout without recreating the entire plot
            Plotly.relayout(gd, {{
                width: currentWidth,
                height: currentHeight
            }});
        }}
        
        function updatePlotSizeFromInput() {{
            const gd = document.getElementById('heatmap');
            const widthInput = parseInt(document.getElementById('widthInput').value);
            
            // Update width
            currentWidth = widthInput;
            
            // If square cells are locked, auto-adjust height
            if (squareCellsLocked && gd.data && gd.data[0]) {{
                const numRows = gd.data[0].y.length;
                const numCols = gd.data[0].x.length;
                const margins = gd.layout.margin || {{l: 120, r: 40, b: 120, t: 100}};
                const marginHorizontal = margins.l + margins.r;
                const marginVertical = margins.t + margins.b;
                const plotAreaWidth = currentWidth - marginHorizontal;
                const plotAreaHeight = plotAreaWidth * numRows / numCols;
                currentHeight = Math.round(plotAreaHeight + marginVertical);
            }} else {{
                currentHeight = parseInt(document.getElementById('heightInput').value);
            }}
            
            // Update sliders (clamped to their range) and displays
            document.getElementById('widthSlider').value = Math.min(2400, Math.max(400, currentWidth));
            document.getElementById('heightSlider').value = Math.min(2400, Math.max(400, currentHeight));
            document.getElementById('widthInput').value = currentWidth;
            document.getElementById('heightInput').value = currentHeight;
            document.getElementById('widthValue').textContent = currentWidth + 'px';
            document.getElementById('heightValue').textContent = currentHeight + 'px';
            
            // Update the layout
            Plotly.relayout(gd, {{
                width: currentWidth,
                height: currentHeight
            }});
        }}
        
        function makeSquareCells() {{
            const gd = document.getElementById('heatmap');
            if (!gd.data || !gd.data[0]) return;
            
            const btn = document.getElementById('squareCellsBtn');
            squareCellsLocked = !squareCellsLocked;
            
            if (squareCellsLocked) {{
                // Lock to square cells
                const numRows = gd.data[0].y.length;
                const numCols = gd.data[0].x.length;
                
                // Get margins (l=120, r=40, b=120, t=100)
                const margins = gd.layout.margin || {{l: 120, r: 40, b: 120, t: 100}};
                const marginHorizontal = margins.l + margins.r;  // 160px
                const marginVertical = margins.t + margins.b;    // 220px
                
                // Calculate height for square cells based on current width
                const plotAreaWidth = currentWidth - marginHorizontal;
                const plotAreaHeight = plotAreaWidth * numRows / numCols;
                const targetHeight = Math.round(plotAreaHeight + marginVertical);
                
                // Update height
                currentHeight = targetHeight;
                document.getElementById('heightSlider').value = Math.min(2400, Math.max(400, targetHeight));
                document.getElementById('heightInput').value = targetHeight;
                document.getElementById('heightValue').textContent = targetHeight + 'px';
                
                // Lock aspect ratio
                Plotly.relayout(gd, {{
                    width: currentWidth,
                    height: targetHeight,
                    'xaxis.scaleanchor': 'y',
                    'xaxis.scaleratio': 1,
                    'yaxis.constrain': 'domain'
                }});
                
                btn.textContent = '🔓 Unlock Cells';
                btn.style.backgroundColor = '#28a745';
                
                console.log('Square cells LOCKED:', {{
                    numCols: numCols,
                    numRows: numRows,
                    width: currentWidth,
                    height: targetHeight,
                    cellAspectRatio: 1.0
                }});
            }} else {{
                // Unlock - remove aspect ratio constraint
                Plotly.relayout(gd, {{
                    'xaxis.scaleanchor': null,
                    'xaxis.scaleratio': null,
                    'yaxis.constrain': null
                }});
                
                btn.textContent = '⬜ Square Cells';
                btn.style.backgroundColor = '';
                
                console.log('Square cells UNLOCKED - free adjustment enabled');
            }}
        }}
        
        function transposeMatrix() {{
            isTransposed = !isTransposed;
            
            // Update button text
            const btn = document.getElementById('transposeBtn');
            btn.textContent = isTransposed ? '🔄 Restore Original' : '🔄 Swap Rows ↔ Columns';
            btn.style.backgroundColor = isTransposed ? '#17a2b8' : '';
            
            console.log('Matrix transposed:', isTransposed);
            
            // Recreate heatmap with transposed data
            createHeatmap();
        }}
        
        // Row/Column reordering functions
        function resetOrder() {{
            // Reset to original order (before any reordering operations)
            currentXLabels = xLabels.slice();
            currentYLabels = yLabels.slice();
            console.log('Reset to original order');
            closeOrderPanel();  // Close panel if open
            createHeatmap();
        }}
        
        // Drag and drop ordering
        let currentOrderType = null;  // 'rows' or 'cols'
        let draggedItem = null;
        let tempOrder = [];
        
        function toggleOrderPanel(type) {{
            currentOrderType = type;
            const panel = document.getElementById('orderPanel');
            const backdrop = document.getElementById('orderPanelBackdrop');
            const label = document.getElementById('orderPanelLabel');
            const listContainer = document.getElementById('orderList');
            
            // Get current labels based on type and transpose state
            // We need to show the ACTUAL order displayed on heatmap, including clustering
            let labels;
            if (type === 'rows') {{
                // Visual rows = Y-axis
                labels = isTransposed ? currentXLabels.slice() : currentYLabels.slice();
                
                // Apply clustering if enabled
                if (useClusteredOrder && clusteringAvailable) {{
                    const effectiveRowOrder = isTransposed ? colOrderClustered : rowOrderClustered;
                    labels = reorderLabels(labels, effectiveRowOrder);
                }}
                label.textContent = 'Reorder Rows (Y-axis)';
            }} else {{
                // Visual columns = X-axis
                labels = isTransposed ? currentYLabels.slice() : currentXLabels.slice();
                
                // Apply clustering if enabled
                if (useClusteredOrder && clusteringAvailable) {{
                    const effectiveColOrder = isTransposed ? rowOrderClustered : colOrderClustered;
                    labels = reorderLabels(labels, effectiveColOrder);
                }}
                label.textContent = 'Reorder Columns (X-axis)';
            }}
            
            tempOrder = labels.slice();
            console.log('toggleOrderPanel:', type, 'isTransposed:', isTransposed, 'clustered:', useClusteredOrder, 'labels:', labels);
            
            // Create draggable list
            listContainer.innerHTML = '';
            labels.forEach((item, index) => {{
                const div = document.createElement('div');
                div.className = 'drag-item';
                div.draggable = true;
                div.dataset.label = item;
                div.innerHTML = '<span class="drag-handle">☰</span>' + item;
                
                div.addEventListener('dragstart', handleDragStart);
                div.addEventListener('dragover', handleDragOver);
                div.addEventListener('drop', handleDrop);
                div.addEventListener('dragend', handleDragEnd);
                div.addEventListener('dragenter', handleDragEnter);
                div.addEventListener('dragleave', handleDragLeave);
                
                listContainer.appendChild(div);
            }});
            
            // Show panel and backdrop
            panel.style.display = 'flex';
            backdrop.style.display = 'block';
        }}
        
        function closeOrderPanel() {{
            document.getElementById('orderPanel').style.display = 'none';
            document.getElementById('orderPanelBackdrop').style.display = 'none';
            currentOrderType = null;
            draggedItem = null;
            tempOrder = [];
        }}
        
        function handleDragStart(e) {{
            draggedItem = this;
            this.classList.add('dragging');
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/html', this.innerHTML);
        }}
        
        function handleDragOver(e) {{
            if (e.preventDefault) {{
                e.preventDefault();
            }}
            e.dataTransfer.dropEffect = 'move';
            return false;
        }}
        
        function handleDragEnter(e) {{
            if (this !== draggedItem) {{
                this.classList.add('drag-over');
            }}
        }}
        
        function handleDragLeave(e) {{
            this.classList.remove('drag-over');
        }}
        
        function handleDrop(e) {{
            if (e.stopPropagation) {{
                e.stopPropagation();
            }}
            
            if (draggedItem !== this) {{
                // Reorder in DOM - insert before the target
                const draggedLabel = draggedItem.dataset.label;
                const targetLabel = this.dataset.label;
                
                const listContainer = document.getElementById('orderList');
                
                // Always insert before the target element
                // This gives consistent behavior: dropping on X puts item before X
                this.parentNode.insertBefore(draggedItem, this);
                
                // Read the new order from DOM to ensure perfect sync
                const itemsAfter = Array.from(listContainer.children);
                tempOrder = itemsAfter.map(item => item.dataset.label);
                
                console.log('Dragged', draggedLabel, 'before', targetLabel, '| New order:', tempOrder);
                
                // Apply immediately to heatmap
                applyReorderImmediate();
            }}
            
            this.classList.remove('drag-over');
            return false;
        }}
        
        function handleDragEnd(e) {{
            this.classList.remove('dragging');
            
            // Remove drag-over class from all items
            const items = document.querySelectorAll('.drag-item');
            items.forEach(item => item.classList.remove('drag-over'));
        }}
        
        function applyReorderImmediate() {{
            if (!currentOrderType || tempOrder.length === 0) return;
            
            // When user manually reorders, disable clustering to respect their choice
            if (useClusteredOrder) {{
                useClusteredOrder = false;
                const orderBtn = document.getElementById('orderBtn');
                if (orderBtn) {{
                    orderBtn.textContent = '🔀 Clustered Order';
                }}
                console.log('Disabled clustering due to manual reordering');
            }}
            
            if (currentOrderType === 'rows') {{
                if (isTransposed) {{
                    currentXLabels = tempOrder.slice();
                }} else {{
                    currentYLabels = tempOrder.slice();
                }}
                console.log('Applied immediate reorder to rows:', tempOrder);
            }} else {{
                if (isTransposed) {{
                    currentYLabels = tempOrder.slice();
                }} else {{
                    currentXLabels = tempOrder.slice();
                }}
                console.log('Applied immediate reorder to columns:', tempOrder);
            }}
            
            createHeatmap();
        }}
        
        function applyDragOrder() {{
            // Just close the panel - reordering already applied immediately
            closeOrderPanel();
        }}
        
        function resetPlotSize() {{
            currentWidth = 800;
            currentHeight = 800;
            document.getElementById('widthSlider').value = 800;
            document.getElementById('heightSlider').value = 800;
            document.getElementById('widthValue').textContent = '800px';
            document.getElementById('heightValue').textContent = '800px';
            document.getElementById('exportScaleSlider').value = 2;
            document.getElementById('exportScaleValue').textContent = '2x';
            
            // Reset custom color inputs
            document.getElementById('colorMin').value = '#ffffff';
            document.getElementById('colorMax').value = '#68379c';
            document.getElementById('colorMin3').value = '#0000ff';
            document.getElementById('colorMid3').value = '#ffffff';
            document.getElementById('colorMax3').value = '#ff0000';
            document.getElementById('use3PointScale').checked = false;
            toggle3PointScale();
            
            createHeatmap();
            showStatus('✅ Reset to defaults', 'success');
        }}
        
        function showStatus(message, type) {{
            console.log('showStatus called:', message, type);
            const statusDiv = document.getElementById('settingsStatus');
            console.log('statusDiv found:', statusDiv);
            if (!statusDiv) {{
                console.error('settingsStatus div not found!');
                return;
            }}
            statusDiv.innerHTML = '<div class="status-message status-' + type + '">' + message + '</div>';
            console.log('Status message displayed:', statusDiv.innerHTML);
            setTimeout(() => {{
                statusDiv.innerHTML = '';
            }}, 3000);
        }}
        
        function updateExportScale(value) {{
            exportScale = parseFloat(value);
            document.getElementById('exportScaleValue').textContent = value + 'x';
        }}
        
        function exportSVG() {{
            const filename = 'heatmap_' + currentScale + '_' + new Date().getTime() + '.svg';
            const gd = document.getElementById('heatmap');
            // SVG at native size. Heatmaps with up to 100 cells export as
            // vector <rect> cells (editable shapes); larger heatmaps keep
            // Plotly's embedded pixel image so the colors stay crisp after
            // PowerPoint's Convert-to-Shape.
            Plotly.toImage(gd, {{
                format: 'svg',
                width: currentWidth,
                height: currentHeight
            }}).then(function(dataUrl) {{
                const svgString = decodeURIComponent(dataUrl.split(',')[1]);
                const imgMatch = svgString.match(/<image[^>]*xlink:href="data:image\/png;base64,[^"]+"[^>]*>/);
                if (!imgMatch) {{
                    downloadSvgString(svgString, filename);
                    showStatus('✅ SVG exported: ' + currentWidth + 'x' + currentHeight + 'px', 'success');
                    return;
                }}
                const imgTag = imgMatch[0];
                const pngUrl = imgTag.match(/xlink:href="([^"]+)"/)[1];
                vectorizeHeatmapCells(svgString, imgTag, pngUrl).then(function(finalSvg) {{
                    downloadSvgString(finalSvg, filename);
                    showStatus('✅ SVG exported: ' + currentWidth + 'x' + currentHeight + 'px', 'success');
                }});
            }}).catch(function(error) {{
                console.error('SVG export failed:', error);
                showStatus('⚠️ SVG export failed. See console.', 'error');
            }});
        }}
        
        function downloadSvgString(svgString, filename) {{
            const blob = new Blob([svgString], {{ type: 'image/svg+xml' }});
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.download = filename;
            link.href = url;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            setTimeout(function() {{ URL.revokeObjectURL(url); }}, 1000);
        }}
        
        // Replace the rasterized heatmap-cell <image> with one <rect> per cell
        // (colors taken from the rendered pixels, so the visual result is
        // identical) - but only for heatmaps with at most 100 cells. Larger
        // heatmaps keep the embedded pixel image to avoid color diffusion.
        function vectorizeHeatmapCells(svgString, imgTag, pngUrl) {{
            return new Promise(function(resolve) {{
                const img = new Image();
                img.onload = function() {{
                    const cols = img.naturalWidth;
                    const rows = img.naturalHeight;
                    if (!rows || !cols || rows * cols > 100) {{
                        resolve(svgString);  // >100 cells: keep the embedded pixel image
                        return;
                    }}
                    const x = parseFloat((imgTag.match(/ x="([^"]+)"/) || [])[1] || '0');
                    const y = parseFloat((imgTag.match(/ y="([^"]+)"/) || [])[1] || '0');
                    const w = parseFloat((imgTag.match(/ width="([^"]+)"/) || [])[1] || '0');
                    const h = parseFloat((imgTag.match(/ height="([^"]+)"/) || [])[1] || '0');
                    const canvas = document.createElement('canvas');
                    canvas.width = cols;
                    canvas.height = rows;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    const data = ctx.getImageData(0, 0, cols, rows).data;
                    const cellW = w / cols;
                    const cellH = h / rows;
                    let rects = '<g class="heatmap-cells-vector" shape-rendering="crispEdges">';
                    for (let r = 0; r < rows; r++) {{
                        for (let c = 0; c < cols; c++) {{
                            const i = (r * cols + c) * 4;
                            const a = data[i + 3];
                            if (a === 0) continue;  // fully transparent (masked) cell
                            const fill = a < 255
                                ? 'rgba(' + data[i] + ',' + data[i + 1] + ',' + data[i + 2] + ',' + (a / 255) + ')'
                                : 'rgb(' + data[i] + ',' + data[i + 1] + ',' + data[i + 2] + ')';
                            rects += '<rect x="' + (x + c * cellW).toFixed(2) + '" y="' + (y + r * cellH).toFixed(2) +
                                '" width="' + cellW.toFixed(2) + '" height="' + cellH.toFixed(2) + '" fill="' + fill + '"/>';
                        }}
                    }}
                    rects += '</g>';
                    resolve(svgString.replace(imgTag, rects));
                }};
                img.onerror = function() {{ resolve(svgString); }};
                img.src = pngUrl;
            }});
        }}
        
        function resetSettings() {{
            resetPlotSize();
        }}
        
        function saveSettings() {{
            try {{
                const settings = {{
                    scale: currentScale,
                    colorscale: currentColorscale,
                    fontSize: currentFontSize,
                    width: currentWidth,
                    height: currentHeight,
                    exportScale: exportScale,
                    showLabels: showLabels,
                    showCellValues: showCellValues,
                    useClusteredOrder: useClusteredOrder,
                    clusteringMethod: currentClusteringMethod,
                    isTransposed: isTransposed
                }};
                localStorage.setItem(storageKey, JSON.stringify(settings));
                showStatus('✅ Settings saved', 'success');
            }} catch (err) {{
                console.error('Save failed:', err);
                showStatus('⚠️ Save failed. See console.', 'error');
            }}
        }}
        
        function loadSettings(silent) {{
            let saved = null;
            try {{
                const raw = localStorage.getItem(storageKey);
                saved = raw ? JSON.parse(raw) : null;
            }} catch (err) {{
                console.error('Load failed:', err);
            }}
            if (!saved) {{
                if (!silent) showStatus('⚠️ No saved settings found', 'warning');
                createHeatmap();
                return;
            }}
            
            // Restore state and sync controls
            if (saved.scale !== undefined) {{
                currentScale = saved.scale;
                document.querySelectorAll('[id^="btn-"]').forEach(btn => btn.classList.remove('active'));
                const scaleBtn = document.getElementById('btn-' + currentScale);
                if (scaleBtn) scaleBtn.classList.add('active');
            }}
            if (saved.fontSize !== undefined) {{
                currentFontSize = saved.fontSize;
                document.getElementById('fontSizeSlider').value = currentFontSize;
                document.getElementById('fontSizeValue').textContent = currentFontSize + 'px';
            }}
            if (saved.width !== undefined) {{
                currentWidth = saved.width;
                document.getElementById('widthSlider').value = Math.min(2400, Math.max(400, currentWidth));
                document.getElementById('widthValue').textContent = currentWidth + 'px';
            }}
            if (saved.height !== undefined) {{
                currentHeight = saved.height;
                document.getElementById('heightSlider').value = Math.min(2400, Math.max(400, currentHeight));
                document.getElementById('heightValue').textContent = currentHeight + 'px';
            }}
            if (saved.exportScale !== undefined) {{
                exportScale = saved.exportScale;
                document.getElementById('exportScaleSlider').value = exportScale;
                document.getElementById('exportScaleValue').textContent = exportScale + 'x';
            }}
            if (saved.showLabels !== undefined) {{
                showLabels = saved.showLabels;
                const btn = document.getElementById('toggleLabelsBtn');
                if (btn) btn.textContent = showLabels ? '🏷️ Hide Labels' : '🏷️ Show Labels';
            }}
            if (saved.showCellValues !== undefined) {{
                showCellValues = saved.showCellValues;
                const btn = document.getElementById('toggleCellValuesBtn');
                if (btn) btn.textContent = showCellValues ? '🔢 Hide Values' : '🔢 Show Values';
            }}
            if (saved.useClusteredOrder !== undefined) {{
                useClusteredOrder = saved.useClusteredOrder;
                document.getElementById('btn-original').classList.toggle('active', !useClusteredOrder);
                document.getElementById('btn-clustered').classList.toggle('active', useClusteredOrder);
            }}
            if (saved.clusteringMethod !== undefined) {{
                currentClusteringMethod = saved.clusteringMethod;
                const sel = document.getElementById('clusteringMethodSelect');
                if (sel) sel.value = currentClusteringMethod;
            }}
            if (saved.isTransposed !== undefined) {{
                isTransposed = saved.isTransposed;
            }}
            
            createHeatmap();
            if (!silent) showStatus('✅ Settings loaded', 'success');
        }}
        
        // Try to load saved settings on page load
        window.addEventListener('load', () => {{
            // Initialize custom color range if zmin/zmax were provided
            if (customZmin !== null && customZmax !== null) {{
                window.customColorRange = {{min: customZmin, max: customZmax}};
                console.log('Initialized custom color range from parameters:', window.customColorRange);
            }}
            
            const saved = localStorage.getItem(storageKey);
            if (saved) {{
                loadSettings(false);  // Silent load on initialization
            }} else {{
                createHeatmap();
            }}
        }});
    </script>
</body>
</html>
'''
    
    # Write HTML file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    if showfig:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(filename))


def split_path(path_df):
    """
    Convert path list to string representation.
    """
    if path_df.empty:
        return path_df
    
    if 'path' in path_df.columns:
        # If path_str already exists, preserve it (it might contain the original list)
        if 'path_str' in path_df.columns:
            # Ensure path is string if it's still a list
            if not path_df.empty and isinstance(path_df['path'].iloc[0], list):
                path_df['path'] = path_df['path'].apply(lambda x: '->'.join(map(str, x)))
            return path_df

        # Generate string representation
        path_strings = path_df['path'].apply(lambda x: '->'.join(map(str, x)) if isinstance(x, list) else str(x))
        
        # Save original list to path_str
        path_df['path_str'] = path_df['path']
        # Overwrite path with string
        path_df['path'] = path_strings
        
    return path_df

def path_filter(path_df, keyword_in_path_to_remove=None):
    """
    Filter paths containing specific keywords.
    """
    if path_df.empty or not keyword_in_path_to_remove:
        return path_df, pd.DataFrame()
        
    if isinstance(keyword_in_path_to_remove, str):
        keywords = [keyword_in_path_to_remove]
    else:
        keywords = keyword_in_path_to_remove
        
    mask = pd.Series(False, index=path_df.index)
    # After split_path(), 'path_str' holds the ORIGINAL list column and the
    # string form lives in 'path'; pick whichever column holds strings.
    match_col = None
    for col in ('path_str', 'path'):
        if col in path_df.columns:
            sample = path_df[col].dropna()
            if sample.empty or isinstance(sample.iloc[0], str):
                match_col = col
                break
    if match_col is not None:
        for kw in keywords:
            mask |= path_df[match_col].str.contains(kw, na=False)
    
    excluded = path_df[mask].copy()
    kept = path_df[~mask].copy()
    
    return kept, excluded

def build_path_dataframe_from_paths(paths, conn_data, targets, real_layer_map=None, level='type', type_lookup=None, edge_lookup=None, engine='auto'):
    """
    Build a DataFrame from a list of paths, calculating weights and probabilities.
    Optimized with dictionary lookup for O(1) edge access.
    
    UNIFIED ENTRY POINT: dispatches to the pandas implementation (below) or the
    Polars implementation (``build_path_dataframe_from_paths_polars``)
    based on ``engine`` / the input type. Both engines return the same schema
    for the same input (the polars engine stringifies list-valued columns for
    CSV compatibility and omits the pandas-only 'path_str' column).
    """
    if engine == 'polars' or (engine == 'auto' and not isinstance(conn_data, pd.DataFrame)):
        if not isinstance(conn_data, pl.DataFrame):
            conn_data = pl.from_pandas(conn_data)
        return build_path_dataframe_from_paths_polars(paths, conn_data, targets,
                                                      real_layer_map=real_layer_map,
                                                      level=level, type_lookup=type_lookup)
    if engine == 'pandas' and not isinstance(conn_data, pd.DataFrame):
        conn_data = conn_data.to_pandas()
    if not paths:
        return pd.DataFrame()
        
    # Determine source and target columns based on level
    src_col = f'{level}_pre' if f'{level}_pre' in conn_data.columns else 'bodyId_pre'
    tgt_col = f'{level}_post' if f'{level}_post' in conn_data.columns else 'bodyId_post'
    
    if edge_lookup is None:
        # Pre-process connection data into a lookup dictionary
        # This avoids O(N) filtering inside the loop
        print(f"Optimizing path building: Pre-indexing {len(conn_data)} connections...")
        
        # Ensure we work with strings for consistent lookup
        # (The original code handled mixed types by trying both, so we normalize to string)
        conn_data_str = conn_data.copy()
        conn_data_str['src_str'] = conn_data_str[src_col].astype(str)
        conn_data_str['tgt_str'] = conn_data_str[tgt_col].astype(str)
        
        # Define aggregation columns and aggregate with Polars (vectorized;
        # a pandas groupby with the per-group nt lambda is slow on large
        # bodyId-level tables). Semantics are identical to the old pandas agg:
        # weight = sum, probabilities/ratios = mean, nt_type = '|'-joined
        # sorted unique non-empty values ('Unknown' when none) - see
        # _unique_nt_types.
        agg_cols = ['src_str', 'tgt_str', 'weight']
        for col in ('traversal_probability', 'connection_ratio', 'nt_type'):
            if col in conn_data.columns:
                agg_cols.append(col)
        pl_conn = pl.from_pandas(conn_data_str[agg_cols])
        agg_exprs = [pl.col('weight').sum().alias('weight')]
        if 'traversal_probability' in conn_data.columns:
            agg_exprs.append(pl.col('traversal_probability').mean().alias('traversal_probability'))
        if 'connection_ratio' in conn_data.columns:
            agg_exprs.append(pl.col('connection_ratio').mean().alias('connection_ratio'))
        if 'nt_type' in conn_data.columns:
            agg_exprs.append(
                pl.col('nt_type').unique().implode().list.sort().list.drop_nulls().list.eval(
                    pl.element().cast(pl.Utf8).filter(~pl.element().is_in(['None', 'nan']))
                ).list.join('|').alias('nt_type')
            )
        grouped = pl_conn.group_by(['src_str', 'tgt_str']).agg(agg_exprs).to_pandas()
        if 'nt_type' in grouped.columns:
            # Empty valid set -> 'Unknown' (matches _unique_nt_types)
            grouped['nt_type'] = grouped['nt_type'].replace('', 'Unknown')
        
        # Convert to dictionary for fast lookup
        # Key: (src_str, tgt_str), Value: dict of metrics
        edge_lookup = grouped.set_index(['src_str', 'tgt_str']).to_dict('index')
    
    rows = []
    
    # Use tqdm for progress bar
    path_iterator = tqdm(paths, desc=f"Enriching {level}-level paths", unit="path", 
                         disable=len(paths) < 100)
    
    for path in path_iterator:
        # Calculate path metrics
        weights = []
        probs = []
        ratios = []
        nt_types = []
        
        valid_path = True
        path_str_nodes = [str(n) for n in path]
        
        for i in range(len(path) - 1):
            u_str, v_str = path_str_nodes[i], path_str_nodes[i+1]
            
            # Fast lookup
            metrics = edge_lookup.get((u_str, v_str))
            
            if not metrics:
                # Debug print for first failure
                if len(rows) == 0 and i == 0:
                    print(f"Debug: Failed to find connection {path[i]} -> {path[i+1]} in conn_data")
                    print(f"  src_col: {src_col}, tgt_col: {tgt_col}")
                valid_path = False
                break
                
            # Get metrics
            w = metrics.get('weight', 0)
            p = metrics.get('traversal_probability', 0)
            r = metrics.get('connection_ratio', 0)
            
            weights.append(_integer_synapse_count(w))
            probs.append(p)
            ratios.append(r)
            
            if 'nt_type' in metrics:
                nt_types.append(metrics['nt_type'])
            
        if valid_path:
            # Format path string with types
            if type_lookup:
                path_formatted_parts = []
                for node in path:
                    node_type = 'Unknown'
                    # Handle potential string/int mismatch in lookup keys
                    t = type_lookup.get(node)
                    if t is None:
                        t = type_lookup.get(str(node))
                    if t is None and isinstance(node, str) and node.isdigit():
                         t = type_lookup.get(int(node))
                    if t is not None:
                        node_type = str(t)
                    path_formatted_parts.append(f"{node}_{node_type}")
                path_formatted = "->".join(path_formatted_parts)
            else:
                # No type lookup provided (e.g. level='type'), just join nodes
                path_formatted = "->".join(map(str, path))

            row = {
                'path_str': path,
                'path': path_formatted,
                'weights': weights,
                'probabilities': probs,
                'ratios': ratios,
                'min_weight': min(weights) if weights else 0,
                'path_prob': np.prod(probs) if probs else 0,
                'min_ratio': min(ratios) if ratios else 0,
                'length': len(path) - 1
            }
            
            if nt_types:
                row['nt_types'] = nt_types
            
            # Add path types if lookup is provided
            if type_lookup:
                path_types = []
                for node in path:
                    # Handle potential string/int mismatch in lookup keys
                    node_type = type_lookup.get(node)
                    if node_type is None:
                        node_type = type_lookup.get(str(node))
                    if node_type is None and isinstance(node, str) and node.isdigit():
                         node_type = type_lookup.get(int(node))
                    
                    path_types.append(str(node_type) if node_type is not None else 'Unknown')
                row['path_types'] = path_types
                
            rows.append(row)
            
    return pd.DataFrame(rows)


def _integer_synapse_count(value):
    """Return integral synapse counts as Python ``int`` values.

    Connection tables can arrive through pandas as ``5.0`` even though the
    underlying synapse count is integral.  Keep genuinely non-integral values
    unchanged so formatting does not hide malformed input data.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if np.isfinite(numeric) and numeric.is_integer():
        return int(numeric)
    return value

# =============================================================================
# Unified streaming path writer (Polars implementation, moved from
# statvis_polars.py).
# =============================================================================

def process_paths_streaming(path_gen, conn_data, targets, output_path, 
                          excluded_path=None, real_layer_map=None, 
                          level='type', type_lookup=None, 
                          keyword_in_path_to_remove=None,
                          batch_size=100000,
                          verbose=True,
                          type_to_label_map=None):
    """
    Stream paths from generator, process in batches using Polars, and write to CSV.
    Returns total count of saved paths.
    
    OPTIMIZED: Uses buffered batch collection to reduce file I/O overhead.
    Collects 20 batches (~2M paths) before writing to minimize disk I/O.
    
    Args:
        type_to_label_map: Optional dict mapping original type names to standardized labels.
                           Types are fetched using original names but output uses mapped labels
                           for cross-dataset comparison.
    """
    if verbose:
        print(f"Optimizing path building: Pre-indexing {len(conn_data)} connections (Polars)...")
    
    # Prepare connection data once
    df_conn = prepare_connection_data(conn_data, level)
    
    batch = []
    total_saved = 0
    total_excluded = 0
    
    # Collect batches in memory before writing (reduces I/O overhead)
    write_buffer = []
    excl_buffer = []
    write_every_n_batches = 20  # Write every 20 batches (~2M paths) to balance memory vs I/O
    batch_count = 0
    
    # Track if we've written to files yet
    first_write = True
    first_excl_write = True
    
    # Single-line progress display (\r-based, so it refreshes in place even
    # when the output is piped/captured). When the generator carries its own
    # pathfinding bars (the L{...} stages), they take over the line while
    # they run; this bar then writes its final summary once they are done.
    if verbose:
        try:
            from vispath_pkg.fast_graph_core import LineProgress
        except ImportError:
            LineProgress = None
        if LineProgress is not None:
            iterator = LineProgress(path_gen, desc=f"Streaming {level}-level paths",
                                    unit="path", leave=True)
        else:
            iterator = path_gen
    else:
        iterator = path_gen
        
    for path in iterator:
        batch.append(path)
        
        if len(batch) >= batch_size:
            df_batch, df_excl = process_batch_polars(batch, df_conn, level, keyword_in_path_to_remove,
                                                      type_to_label_map=type_to_label_map)
            
            if not df_batch.is_empty():
                write_buffer.append(df_batch)
                total_saved += len(df_batch)
                
            if excluded_path and not df_excl.is_empty():
                excl_buffer.append(df_excl)
                total_excluded += len(df_excl)
            
            batch_count += 1
            batch = []
            
            # Write buffered batches periodically
            if batch_count >= write_every_n_batches:
                if write_buffer:
                    _write_buffer_to_csv(write_buffer, output_path, append=not first_write)
                    first_write = False
                    write_buffer = []
                if excl_buffer and excluded_path:
                    _write_buffer_to_csv(excl_buffer, excluded_path, append=not first_excl_write)
                    first_excl_write = False
                    excl_buffer = []
                batch_count = 0
                gc.collect()
            
    # Process remaining paths
    if batch:
        df_batch, df_excl = process_batch_polars(batch, df_conn, level, keyword_in_path_to_remove,
                                                  type_to_label_map=type_to_label_map)
        
        if not df_batch.is_empty():
            write_buffer.append(df_batch)
            total_saved += len(df_batch)
            
        if excluded_path and not df_excl.is_empty():
            excl_buffer.append(df_excl)
            total_excluded += len(df_excl)
    
    # Write any remaining buffered data
    if write_buffer:
        _write_buffer_to_csv(write_buffer, output_path, append=not first_write)
    if excl_buffer and excluded_path:
        _write_buffer_to_csv(excl_buffer, excluded_path, append=not first_excl_write)
            
    return total_saved

def _type_probability_series(pairs_df, group_pre: str, group_post: str, aggregate_method: str):
    """Type-level traversal_probability from the deduplicated bodyId pairs.

    Returns a Series indexed by (group_pre, group_post), or None when
    *aggregate_method* is 'ratio' (the caller derives it from the type-level
    connection_ratio instead). The pairs frame must carry 'weight',
    'block_probability' and 'traversal_probability' columns.

    - 'product' (default): ``1 - prod(1 - p_pair)`` - the type edge is a
      bundle of parallel channels, so it transmits if ANY pair transmits
      (reliability/OR model; recommended for path analysis).
    - 'average': weight-weighted mean of the pair probabilities.
    - 'ratio': input-share model (``min(ratio / 0.3, 1)``).
    """
    if aggregate_method == 'product':
        blocks = pairs_df.groupby([group_pre, group_post])['block_probability'].prod()
        return (1.0 - blocks).rename('traversal_probability')
    if aggregate_method == 'average':
        tmp = pairs_df.assign(_wt=pairs_df['weight'] * pairs_df['traversal_probability'])
        grouped = tmp.groupby([group_pre, group_post])[['_wt', 'weight']].sum()
        return (grouped['_wt'] / grouped['weight'].replace(0, np.nan)).fillna(0.0).rename('traversal_probability')
    return None


def EnrichConnectionTable(conn_table, traversal_probability_threshold=0, dataset=None, script_path=None, target_neurons_df=None, aggregate_method='product', label_mapper=None, global_incoming_weights=None, separate_hemispheres=False, engine='auto', global_incoming_body_weights=None):
    '''Add traversal probability, connection ratio, and layer information to the connection table
    
    UNIFIED ENTRY POINT: dispatches to the pandas implementation (below) or the
    Polars implementation (``statvis_polars.EnrichConnectionTablePolars``)
    based on ``engine`` / the input type. Both engines produce the same
    type-level and group-level output schema (type_pre/type_post or
    custom_group_pre/custom_group_post, weight, connection_ratio,
    traversal_probability, block_probability, nt_type) and the same values for
    the same input.
    
    NOTE: When separate_hemispheres=True, the caller is expected to have already applied
    hemisphere suffixes (_L/_R/_U) to type_pre/type_post columns. This function will
    aggregate by those already-suffixed types. The parameter is accepted for API
    compatibility but does not change the aggregation behavior.
    
    Parameters
    ----------
    conn_table : DataFrame
        Connection table to enrich (pandas DataFrame, or Polars DataFrame with
        ``engine='polars'`` / ``engine='auto'``)
    traversal_probability_threshold : float, optional
        Minimum traversal probability threshold (default: 0)
    dataset : str, optional
        Dataset name (e.g., 'optic-lobe:v1.1') for local dataset lookup
    script_path : str, optional
        Path to script directory containing 'datasets' folder
    target_neurons_df : DataFrame, optional
        Full dataframe of target neurons (with bodyId, type, post columns).
        Used to get correct type-level denominators. If not provided, only
        neurons appearing in connections will be used (less accurate).
    aggregate_method : str, optional
        How the type-level traversal_probability is derived from the
        bodyId-level pairs (default 'product'):
        - 'product': ``1 - prod(1 - p_pair)`` over the deduplicated pairs -
          the type edge is a bundle of parallel channels, so it transmits if
          ANY pair transmits (reliability/OR model; recommended for path
          analysis).
        - 'average': weight-weighted mean of the pair probabilities.
        - 'ratio': ``min(connection_ratio / 0.3, 1)`` (input-share model).
        Same semantics as the Polars engine and coana._apply_type_level_filters.
    label_mapper : LabelMapper, optional
        LabelMapper object to standardize types in the local dataset for accurate ratio calculation.
    global_incoming_weights : DataFrame, optional
        Pre-computed total incoming weights for each post-synaptic type.
        Should have columns [type_post, total_incoming_weight].
        If provided, used for calculating GLOBAL type-level ratios
        (denominator = ALL incoming connections in the dataset, per
        ScoreCalculation_Guide). If None, local ratios (from the provided
        connections only) are calculated - these are inflated when the table
        only covers a subset of the dataset.
    global_incoming_body_weights : DataFrame, optional
        Pre-computed total incoming weights for each post-synaptic bodyId.
        Should have columns [bodyId_post, total_incoming_weight].
        If provided, used for calculating GLOBAL bodyId-level ratios.
        Post neurons missing from this table (and all untyped neurons, which
        are grouped by bodyId at type level) fall back to LOCAL totals so
        ratios/probabilities never collapse to 0.
    separate_hemispheres : bool, optional
        Whether hemisphere separation is enabled. The actual suffix application
        should be done by the caller before passing the connection table.
    engine : str, optional
        'auto' (default): Polars engine when conn_table is a Polars DataFrame,
        pandas engine otherwise. 'pandas' / 'polars': force the engine
        (the input frame is converted when needed).
    
    Returns
    -------
    conn_df : DataFrame
        Enriched connection table with bodyId-level metrics
    conn_type : DataFrame
        Type-level aggregation (always based on original type column)
    conn_group : DataFrame or None
        Custom group-level aggregation (only if custom_group columns exist)
    '''
    if engine == 'polars' or (engine == 'auto' and not isinstance(conn_table, pd.DataFrame)):
        if not isinstance(conn_table, pl.DataFrame):
            conn_table = pl.from_pandas(conn_table)
        # Empty input: return engine-typed empty outputs (see the pandas guard below)
        if conn_table.is_empty() or conn_table.width == 0:
            return pl.DataFrame(), pl.DataFrame(), None
        return EnrichConnectionTablePolars(
            conn_table,
            traversal_probability_threshold=traversal_probability_threshold,
            dataset=dataset,
            script_path=script_path,
            target_neurons_df=target_neurons_df,
            aggregate_method=aggregate_method,
            label_mapper=label_mapper,
            global_incoming_weights=global_incoming_weights,
            separate_hemispheres=separate_hemispheres,
            global_incoming_body_weights=global_incoming_body_weights,
        )
    if engine == 'pandas' and not isinstance(conn_table, pd.DataFrame):
        conn_table = conn_table.to_pandas()
    # Empty input: return engine-typed empty outputs instead of crashing on
    # missing columns / dtype-driven operations (both engines used to raise
    # KeyError/ColumnNotFound/InvalidOperation for empty or column-less tables).
    if conn_table.shape[0] == 0 or conn_table.shape[1] == 0:
        return pd.DataFrame(), pd.DataFrame(), None
    conn_df = conn_table.copy()
    
    # Determine grouping columns (use custom_group if available, otherwise type)
    group_pre = 'custom_group_pre' if 'custom_group_pre' in conn_df.columns else 'type_pre'
    group_post = 'custom_group_post' if 'custom_group_post' in conn_df.columns else 'type_post'
    
    # Try to use local dataset first
    use_local = False
    ndf_complete = None
    if dataset and script_path:
        dataset_clean = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
        dataset_path = (
            _flywire_neuron_table_path(dataset, script_path)
            if is_local_connectome_dataset(dataset) else
            os.path.join(
                script_path,
                'datasets',
                dataset_clean,
                f"{dataset_clean}_allneurons_neuron_df.csv"
            )
        )
        
        # Enhanced dataset discovery logic
        if dataset_path is not None and not os.path.exists(dataset_path):
            # Fallback: Try root datasets folder (legacy)
            legacy_path = os.path.join(
                script_path,
                'datasets',
                f"{dataset_clean}_allneurons_neuron_df.csv"
            )
            if os.path.exists(legacy_path):
                dataset_path = legacy_path
            else:
                # Try globbing for any *_allneurons_neuron_df.csv in subdir
                subdir_path = os.path.join(script_path, 'datasets', dataset_clean)
                if os.path.exists(subdir_path):
                    import glob
                    candidates = glob.glob(os.path.join(subdir_path, "*_allneurons_neuron_df.csv"))
                    if candidates:
                        dataset_path = candidates[0]

        if dataset_path is not None and os.path.exists(dataset_path):
            use_local = True
            # FAFB/BANC local-release tables use exact string bodyIds.
            if is_local_connectome_dataset(dataset):
                if str(dataset_path).lower().endswith('.parquet'):
                    ndf_complete = pd.read_parquet(dataset_path)
                else:
                    ndf_complete = pd.read_csv(
                        dataset_path, header=0, index_col=None,
                        dtype={'bodyId': 'string'}, low_memory=False
                    )
                normalize_flywire_id_columns(ndf_complete, ['bodyId'])
            else:
                # Read robustly instead of hard-coding index_col=0: the
                # coana-saved CSVs carry a leading unnamed index column, but
                # standard CSVs (bodyId as the first data column) must work
                # too - and so must legacy files written with bodyId as the
                # index (the Polars engine reads all of these fine).
                ndf_complete = pd.read_csv(dataset_path, header=0, index_col=None, low_memory=False)
                if len(ndf_complete.columns) and ndf_complete.columns[0].startswith('Unnamed:'):
                    ndf_complete = ndf_complete.drop(columns=[ndf_complete.columns[0]])
                if 'bodyId' not in ndf_complete.columns and ndf_complete.index.name == 'bodyId':
                    ndf_complete = ndf_complete.reset_index()
            
            # Ensure bodyId column is string for comparison
            if 'bodyId' in ndf_complete.columns:
                ndf_complete['bodyId'] = ndf_complete['bodyId'].astype(str)
            
            mapped_dict = {}
            # Apply label mapping to local dataset if provided
            if label_mapper and 'type' in ndf_complete.columns:
                # Create a copy to avoid modifying the original cache
                ndf_complete = ndf_complete.copy()

                # Vectorized label mapping (same semantics as the old
                # row-wise apply, but one get_label call per UNIQUE bodyId/
                # type instead of two calls per neuron row).
                body_ids = ndf_complete['bodyId'].astype(str)
                bid_map = {
                    b: label_mapper.get_label(dataset, b)
                    for b in body_ids.unique()
                }
                mapped_body = body_ids.map(bid_map)
                # BodyId mapping wins when it differs from the raw ID.
                # (None results are treated as unmapped, matching the old
                # fillna fallback behaviour downstream.)
                mask_body = mapped_body.notna() & (mapped_body != body_ids)

                type_vals = ndf_complete['type'].fillna('').astype(str)
                type_map = {
                    t: label_mapper.get_label(dataset, t)
                    for t in type_vals[~mask_body].unique()
                    if t != ''
                }
                mapped_type = type_vals.map(type_map)
                mask_type = (
                    (~mask_body)
                    & (type_vals != '')
                    & (type_vals != body_ids)
                    & mapped_type.notna()
                    & (mapped_type != type_vals)
                )

                std_label = pd.Series('', index=ndf_complete.index, dtype=object)
                std_label.loc[mask_body] = mapped_body[mask_body]
                std_label.loc[mask_type] = mapped_type[mask_type]
                ndf_complete['std_label'] = std_label
                
                # Overwrite type with std_label where available
                mask = ndf_complete['std_label'] != ''
                ndf_complete.loc[mask, 'type'] = ndf_complete.loc[mask, 'std_label']
                
                # Create a dictionary of ONLY the mapped labels
                mapped_dict = ndf_complete.loc[mask].set_index('bodyId')['std_label'].to_dict()
                
                # Also overwrite custom_group if present, to ensure grouping uses the mapped label
                if 'custom_group' in ndf_complete.columns:
                    ndf_complete.loc[mask, 'custom_group'] = ndf_complete.loc[mask, 'std_label']
                
                # print(f"DEBUG: Unique types after mapping: {ndf_complete['type'].unique()}")
                # if 'custom_group' in ndf_complete.columns:
                #      print(f"DEBUG: custom_group sample: {ndf_complete['custom_group'].unique()[:5]}")
                
            bodyIds_needed = conn_df.bodyId_post.astype(str).unique().tolist()
            if 'post' in ndf_complete.columns:
                df_post = ndf_complete[ndf_complete['bodyId'].isin(bodyIds_needed)][['bodyId', 'post']].copy()
            else:
                # No post counts in the local table: leave post missing so the
                # merge below produces NaN -> filled with 0, matching the
                # Polars engine's graceful handling.
                df_post = pd.DataFrame(columns=['bodyId', 'post'])
    
    if not use_local:
        # Fallback to API call
        # Note: fetch_neurons is not available in this context, so we rely on target_neurons_df or existing data
        if target_neurons_df is not None:
            # Ensure consistent types for filtering (convert to string to handle int64 vs str mismatch)
            target_bodyIds = target_neurons_df['bodyId'].astype(str);
            conn_bodyIds = conn_df.bodyId_post.astype(str).unique();
            
            # Apply label mapping to target_neurons_df if provided
            if label_mapper and 'type' in target_neurons_df.columns:
                target_neurons_df = target_neurons_df.copy()
                
                # Define mapping function (same as above)
                def get_mapped_label_target(row):
                    body_id = str(row['bodyId'])
                    # Use get_label to check all roles
                    mapped_body = label_mapper.get_label(dataset, body_id)
                    if mapped_body != body_id:
                         return mapped_body
                    
                    if pd.notna(row['type']):
                        type_val = str(row['type'])
                        if type_val != body_id:
                            mapped_type = label_mapper.get_label(dataset, type_val)
                            if mapped_type != type_val:
                                return mapped_type
                    return ''

                target_neurons_df['std_label'] = target_neurons_df.apply(get_mapped_label_target, axis=1)
                mask = target_neurons_df['std_label'] != ''
                target_neurons_df.loc[mask, 'type'] = target_neurons_df.loc[mask, 'std_label']
            
            df_post = target_neurons_df[target_bodyIds.isin(conn_bodyIds)][['bodyId', 'post']].copy();
        else:
            # Last resort: assume post info might be in conn_df or we can't get it
            # If we can't get it, we can't calculate connection_ratio accurately
            print("Warning: Could not fetch neuron info for connection ratio calculation.");
            df_post = pd.DataFrame(columns=['bodyId', 'post']);
    
    post_info = df_post.copy();
    post_info.columns = ['bodyId_post','post'];
    
    # Handle case where type_pre/type_post columns already exist (from cache enrichment)
    # null/empty types are temporarily marked 'Unknown' for the type_map
    # lookup below; the original null-ness is remembered so that untyped
    # neurons can be grouped per-bodyId at the end (Polars std_label
    # semantics) instead of lumping them into one mixed 'Unknown' group.
    was_null_pre = conn_df['type_pre'].isnull() | (conn_df['type_pre'] == '') if 'type_pre' in conn_df.columns else pd.Series(True, index=conn_df.index)
    was_null_post = conn_df['type_post'].isnull() | (conn_df['type_post'] == '') if 'type_post' in conn_df.columns else pd.Series(True, index=conn_df.index)
    if 'type_pre' in conn_df.columns:
        conn_df.loc[conn_df.type_pre.isnull(),'type_pre'] = 'Unknown';
    else:
        conn_df['type_pre'] = 'Unknown';
        
    if 'type_post' in conn_df.columns:
        conn_df.loc[conn_df.type_post.isnull(),'type_post'] = 'Unknown';
    else:
        conn_df['type_post'] = 'Unknown';

    # Update types from local dataset if available
    if use_local and ndf_complete is not None and 'type' in ndf_complete.columns:
        # Ensure bodyIds are strings
        conn_df['bodyId_pre'] = conn_df['bodyId_pre'].astype(str);
        conn_df['bodyId_post'] = conn_df['bodyId_post'].astype(str);
        
        # Map types if missing or empty
        type_map = ndf_complete.set_index('bodyId')['type'].to_dict();
        
        if label_mapper:
            # 1. Apply explicit mappings (overwrite types for mapped neurons)
            # IMPORTANT: When separate_hemispheres=True, type_pre/type_post may already have
            # hemisphere suffixes (e.g., "PPL101_L"). mapped_dict contains base types (e.g., "PPL101").
            # We need to preserve the hemisphere suffix when applying the mapping.
            if mapped_dict:
                if separate_hemispheres:
                    # Helper to extract hemisphere suffix from type name
                    def extract_hemi_suffix(type_name):
                        if pd.notna(type_name) and isinstance(type_name, str):
                            if type_name.endswith('_L'):
                                return '_L'
                            elif type_name.endswith('_R'):
                                return '_R'
                            elif type_name.endswith('_U'):
                                return '_U'
                        return ''
                    
                    # Apply mapping to pre neurons, preserving hemisphere suffix
                    mapped_pre = conn_df['bodyId_pre'].map(mapped_dict)
                    hemi_suffix_pre = conn_df['type_pre'].map(extract_hemi_suffix)
                    mapped_pre_with_suffix = mapped_pre.astype(str) + hemi_suffix_pre
                    # Only apply where mapping exists (mapped_pre is not NaN)
                    conn_df['type_pre'] = mapped_pre_with_suffix.where(mapped_pre.notna(), conn_df['type_pre'])
                    
                    # Apply mapping to post neurons, preserving hemisphere suffix
                    mapped_post = conn_df['bodyId_post'].map(mapped_dict)
                    hemi_suffix_post = conn_df['type_post'].map(extract_hemi_suffix)
                    mapped_post_with_suffix = mapped_post.astype(str) + hemi_suffix_post
                    conn_df['type_post'] = mapped_post_with_suffix.where(mapped_post.notna(), conn_df['type_post'])
                else:
                    # No hemisphere separation: apply mapping directly
                    conn_df['type_pre'] = conn_df['bodyId_pre'].map(mapped_dict).fillna(conn_df['type_pre'])
                    conn_df['type_post'] = conn_df['bodyId_post'].map(mapped_dict).fillna(conn_df['type_post'])
            
            # 2. Fill unknowns with type_map (original types + mapped types)
            # This preserves existing valid types in conn_df that were NOT mapped, preventing fallback to bodyIds
            mask_pre = (conn_df['type_pre'].isnull()) | (conn_df['type_pre'] == '') | (conn_df['type_pre'] == 'Unknown')
            if mask_pre.any():
                conn_df.loc[mask_pre, 'type_pre'] = conn_df.loc[mask_pre, 'bodyId_pre'].map(type_map).fillna('Unknown')
                
            mask_post = (conn_df['type_post'].isnull()) | (conn_df['type_post'] == '') | (conn_df['type_post'] == 'Unknown')
            if mask_post.any():
                conn_df.loc[mask_post, 'type_post'] = conn_df.loc[mask_post, 'bodyId_post'].map(type_map).fillna('Unknown')
        else:
            # Update type_pre
            mask_pre = (conn_df['type_pre'].isnull()) | (conn_df['type_pre'] == '') | (conn_df['type_pre'] == 'Unknown')
            if mask_pre.any():
                conn_df.loc[mask_pre, 'type_pre'] = conn_df.loc[mask_pre, 'bodyId_pre'].map(type_map).fillna('Unknown')
                
            # Update type_post
            mask_post = (conn_df['type_post'].isnull()) | (conn_df['type_post'] == '') | (conn_df['type_post'] == 'Unknown')
            if mask_post.any():
                conn_df.loc[mask_post, 'type_post'] = conn_df.loc[mask_post, 'bodyId_post'].map(type_map).fillna('Unknown')

    # Untyped neurons group by their bodyId (Polars std_label semantics): a
    # null/empty type that stayed 'Unknown' after the type_map lookup is
    # replaced by the bodyId, so untyped neurons form per-bodyId groups
    # instead of one mixed 'Unknown' group. Pre-existing 'Unknown' values in
    # the input are left alone (the Polars engine keeps them as a type).
    if was_null_pre.any():
        mask_pre = was_null_pre & conn_df['type_pre'].isin(['Unknown', ''])
        conn_df.loc[mask_pre, 'type_pre'] = conn_df.loc[mask_pre, 'bodyId_pre'].astype(str)
    if was_null_post.any():
        mask_post = was_null_post & conn_df['type_post'].isin(['Unknown', ''])
        conn_df.loc[mask_post, 'type_post'] = conn_df.loc[mask_post, 'bodyId_post'].astype(str)

    # Custom-group labels follow the same exclusivity chain as the Polars
    # engine: custom_group -> type -> bodyId (first non-empty wins; runs
    # after the untyped conversion above so untyped neurons resolve their
    # group to the bodyId). A neuron never appears under both its custom
    # group and its raw type, and null group keys never reach the groupby.
    if 'custom_group_pre' in conn_df.columns:
        fill_pre = conn_df['custom_group_pre'].isnull() | (conn_df['custom_group_pre'] == '')
        if fill_pre.any():
            conn_df.loc[fill_pre, 'custom_group_pre'] = conn_df.loc[fill_pre, 'type_pre']
            still_missing = conn_df['custom_group_pre'].isnull() | (conn_df['custom_group_pre'] == '')
            conn_df.loc[still_missing, 'custom_group_pre'] = conn_df.loc[still_missing, 'bodyId_pre'].astype(str)
    if 'custom_group_post' in conn_df.columns:
        fill_post = conn_df['custom_group_post'].isnull() | (conn_df['custom_group_post'] == '')
        if fill_post.any():
            conn_df.loc[fill_post, 'custom_group_post'] = conn_df.loc[fill_post, 'type_post']
            still_missing = conn_df['custom_group_post'].isnull() | (conn_df['custom_group_post'] == '')
            conn_df.loc[still_missing, 'custom_group_post'] = conn_df.loc[still_missing, 'bodyId_post'].astype(str)
    
    # Ensure bodyId columns are strings for merging to avoid warnings
    conn_df['bodyId_post'] = conn_df['bodyId_post'].astype(str)
    post_info['bodyId_post'] = post_info['bodyId_post'].astype(str)

    conn_df = conn_df.merge(post_info,how='left',on='bodyId_post')
    
    # Handle potential column collision if 'post' already existed (creates post_x, post_y)
    if 'post_x' in conn_df.columns and 'post_y' in conn_df.columns:
        # Prefer the new info (post_y) if available, otherwise keep old (post_x)
        conn_df['post'] = conn_df['post_y'].fillna(conn_df['post_x'])
        conn_df = conn_df.drop(columns=['post_x', 'post_y'])
    
    # Unify with the Polars engine: the no-local fallback leaves 'post'
    # missing (NaN); fill with 0 so both engines emit the same bodyId-level
    # schema (real runs carry real post counts and are unaffected).
    if 'post' in conn_df.columns:
        conn_df['post'] = conn_df['post'].fillna(0)
    
    # Check if connection_ratio already exists and has valid values (from coana.py global calculation)
    # If so, preserve it to maintain the correct global ratio calculation
    has_valid_ratio = ('connection_ratio' in conn_df.columns and 
                       conn_df['connection_ratio'].notna().any() and 
                       (conn_df['connection_ratio'] > 0).any())
    
    if not has_valid_ratio:
        # Only recalculate if ratio doesn't exist or has no valid values.
        # GLOBAL bodyId denominators (total incoming to each post neuron from
        # ALL sources) are used when supplied; post neurons missing from the
        # global table fall back to the LOCAL total over this table so ratios
        # never collapse to 0/undefined.
        if global_incoming_body_weights is not None and 'bodyId_post' in global_incoming_body_weights.columns:
            total_incoming = global_incoming_body_weights[['bodyId_post', 'total_incoming_weight']].copy()
            total_incoming['bodyId_post'] = total_incoming['bodyId_post'].astype(str)
            local_total = conn_df.groupby('bodyId_post')['weight'].sum().reset_index(name='local_total_incoming')
            total_incoming = total_incoming.merge(local_total, on='bodyId_post', how='left')
            total_incoming['total_incoming_weight'] = total_incoming['total_incoming_weight'].fillna(
                total_incoming['local_total_incoming']
            )
            total_incoming = total_incoming.drop(columns=['local_total_incoming'])
            conn_df = conn_df.merge(total_incoming, how='left', on='bodyId_post')
        else:
            total_incoming = conn_df.groupby('bodyId_post')['weight'].sum().reset_index(name='total_incoming_weight')
            conn_df = conn_df.merge(total_incoming, how='left', on='bodyId_post')
        
        # Calculate connection_ratio using the (global if available, else local) denominator
        weight_arr = conn_df['weight'].to_numpy(dtype=float)
        total_arr = conn_df['total_incoming_weight'].to_numpy(dtype=float)
        valid_mask = ~np.isnan(total_arr) & (total_arr > 0)
        conn_df['connection_ratio'] = np.divide(
            weight_arr, total_arr,
            out=np.full(len(conn_df), np.nan, dtype=float),
            where=valid_mask,
        )
        
        # Drop temporary column
        conn_df = conn_df.drop(columns=['total_incoming_weight'], errors='ignore')
        
    # Calculate traversal_probability from connection_ratio
    if 'traversal_probability' in conn_df.columns:
        conn_df['traversal_probability'] = conn_df.connection_ratio / 0.3
    else:
        conn_df.insert(loc=3,column='traversal_probability',value=conn_df.connection_ratio/0.3)
    
    conn_df.loc[conn_df.traversal_probability > 1,'traversal_probability'] = 1
    
    # Calculate block_probability (handle if already exists)
    if 'block_probability' in conn_df.columns:
        conn_df['block_probability'] = 1 - conn_df.traversal_probability
    else:
        conn_df.insert(loc=len(conn_df.columns),column='block_probability',value= 1 - conn_df.traversal_probability)
    
    # Drop temporary column
    conn_df = conn_df.drop(columns=['total_incoming_weight'], errors='ignore')
    
    conn_df = conn_df.loc[conn_df.traversal_probability >= traversal_probability_threshold]
    
    # Aggregate connection data by neuron type/group
    # Use custom_group if available, otherwise fall back to type
    # Calculate from bodyId level to ensure accuracy (neurons in connections, not types in connections)
    # First deduplicate by bodyId pairs to avoid counting same connection multiple times
    cols_to_keep = ['bodyId_pre', 'bodyId_post', group_pre, group_post, 'weight']
    # Per-pair probability columns feed the type-level aggregate method
    # ('product' compound / 'average' weighted mean).
    for col in ('block_probability', 'traversal_probability'):
        if col in conn_df.columns and col not in cols_to_keep:
            cols_to_keep.append(col)
    
    # Check for NT type column - prefer nt_type_pre (presynaptic NT), fallback to nt_type
    nt_col = None
    if 'nt_type_pre' in conn_df.columns:
        nt_col = 'nt_type_pre'
    elif 'nt_type' in conn_df.columns:
        nt_col = 'nt_type'
    
    has_nt = nt_col is not None
    if has_nt:
        cols_to_keep.append(nt_col)
        
    bodyid_pairs = conn_df[cols_to_keep].drop_duplicates(subset=['bodyId_pre', 'bodyId_post'])
    
    # Add progress indicator for large aggregations
    pbar = None
    # if len(bodyid_pairs) > 50000:
    #     pbar = tqdm(total=5, desc=f"  Enriching {len(bodyid_pairs):,} connections", unit="step")
    
    # Weight aggregation via Polars (vectorized; the pandas groupby with a
    # per-group mode() lambda is the dominant cost on million-row tables).
    # Semantics: weight = sum over deduplicated bodyId pairs; nt_type = mode
    # (most frequent, first on ties - identical to the pandas mode().iloc[0]).
    pl_pairs = pl.from_pandas(bodyid_pairs)
    if has_nt:
        weight_sum = (
            pl_pairs.group_by([group_pre, group_post])
            .agg([pl.col('weight').sum().alias('weight'),
                  pl.col(nt_col).mode().sort().first().alias('nt_type')])
            .to_pandas()
        )
    else:
        weight_sum = (
            pl_pairs.group_by([group_pre, group_post])
            .agg(pl.col('weight').sum().alias('weight'))
            .to_pandas()
        )
    
    if pbar: pbar.update(1) # Step 1: Weight aggregation complete
    
    # Calculate total incoming weights per group_post (sum across all group_pre sources)
    total_incoming_per_type = weight_sum.groupby(group_post)['weight'].sum().reset_index(name='total_incoming_weight')

    if pbar: pbar.update(1) # Step 2: Incoming weights calculated

    # (A legacy 'total_post per group' computation lived here and fed the old
    # static-ratio formula; the dynamic ratio below uses total_incoming_per_type
    # instead, so that block was dead code and was removed.)

    # Calculate group-to-group connection_ratio
    # Ratio = weight(A→B) / total_incoming_weight(→B)
    # With global_incoming_weights, the denominator covers ALL incoming
    # connections to B in the dataset (true fraction of B's total input from A,
    # per ScoreCalculation_Guide). Without them, the denominator is the sum
    # over the threshold-filtered connections in this table only, which
    # inflates ratios when the table covers a subset of the dataset.
    if global_incoming_weights is not None and group_post in global_incoming_weights.columns:
        conn_type = weight_sum.merge(
            global_incoming_weights[[group_post, 'total_incoming_weight']],
            on=group_post, how='left',
        )
    else:
        conn_type = weight_sum.merge(total_incoming_per_type, on=group_post, how='left')
    
    # Fall back to LOCAL totals for groups missing from the global table
    # (untyped neurons grouped by bodyId, std_labels absent from the global
    # type table, ...). Without this the ratio would be NaN and the
    # traversal_probability would be filled with 0, zeroing every path
    # through such an edge.
    conn_type['total_incoming_weight'] = conn_type['total_incoming_weight'].fillna(
        conn_type[group_post].map(
            total_incoming_per_type.set_index(group_post)['total_incoming_weight']
        )
    )
    
    # Calculate ratio using the chosen denominator
    conn_type['connection_ratio'] = conn_type.apply(
        lambda row: row['weight'] / row['total_incoming_weight'] 
        if pd.notnull(row['total_incoming_weight']) and row['total_incoming_weight'] > 0 
        else float('nan'),
        axis=1
    )

    if pbar: pbar.update(1) # Step 4: Ratios calculated

    # Type-level traversal_probability from the aggregate method. 'product'
    # (default) compounds the per-pair block probabilities over the deduped
    # pairs (reliability/OR model); 'average' takes the weight-weighted mean;
    # 'ratio' uses min(connection_ratio / 0.3, 1) (input-share model).
    prob_series = _type_probability_series(
        bodyid_pairs, group_pre, group_post, aggregate_method
    )
    if prob_series is None:
        conn_type['traversal_probability'] = conn_type['connection_ratio'] / 0.3
        conn_type.loc[conn_type['traversal_probability'] > 1, 'traversal_probability'] = 1
    else:
        conn_type = conn_type.merge(
            prob_series.reset_index(), how='left', on=[group_pre, group_post]
        )
    conn_type['block_probability'] = 1 - conn_type['traversal_probability']
    
    if pbar: 
        pbar.update(1) # Step 5: Traversal probabilities calculated
        pbar.close()
    
    # Fix FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated
    # Explicitly convert to numeric before filling NaNs to avoid object-dtype downcasting issues
    # Note: connection_ratio NaN means no incoming connections (undefined), keep it as NaN
    conn_aggregated = conn_type.copy()
    conn_aggregated['connection_ratio'] = pd.to_numeric(conn_aggregated['connection_ratio'], errors='coerce')
    conn_aggregated['traversal_probability'] = pd.to_numeric(conn_aggregated['traversal_probability'], errors='coerce').fillna(0.0)
    conn_aggregated['block_probability'] = pd.to_numeric(conn_aggregated['block_probability'], errors='coerce').fillna(1.0)
    # conn_aggregated = conn_aggregated.infer_objects() # No longer needed as we explicitly converted
    
    # Keep nt_type in the type-level output when the input carries it (the
    # path builders use it to annotate edges; the Polars engine emits it too).
    type_keep_cols = [group_pre, group_post, 'weight', 'connection_ratio',
                      'traversal_probability', 'block_probability']
    if 'nt_type' in conn_aggregated.columns:
        type_keep_cols.append('nt_type')
    conn_aggregated = conn_aggregated[type_keep_cols]
    
    # Check if we're using custom groups
    has_custom_groups = (group_pre == 'custom_group_pre' and group_post == 'custom_group_post')
    
    if has_custom_groups:
        # Compute BOTH type-based and custom group-based aggregations
        # 1. Custom group aggregation (already computed). Columns stay
        #    custom_group_pre/custom_group_post (same schema as the Polars
        #    engine; legacy 'group_pre'/'group_post' naming was removed).
        conn_group = conn_aggregated.copy()
        
        # 2. Original type-based aggregation
        # Calculate from bodyId level for accuracy
        bodyid_pairs_type_cols = ['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post', 'weight']
        for col in ('block_probability', 'traversal_probability'):
            if col in conn_df.columns and col not in bodyid_pairs_type_cols:
                bodyid_pairs_type_cols.append(col)
        if has_nt:
            bodyid_pairs_type_cols.append(nt_col)
        bodyid_pairs_type = conn_df[bodyid_pairs_type_cols].drop_duplicates(subset=['bodyId_pre', 'bodyId_post'])
        
        pbar_type = None
        # if len(bodyid_pairs_type) > 50000:
        #     pbar_type = tqdm(total=4, desc=f"  Enriching {len(bodyid_pairs_type):,} type-level connections", unit="step")

        # Polars aggregation (same semantics as the main branch above)
        pl_pairs_type = pl.from_pandas(bodyid_pairs_type)
        if has_nt:
            weight_sum_type = (
                pl_pairs_type.group_by(['type_pre', 'type_post'])
                .agg([pl.col('weight').sum().alias('weight'),
                      pl.col(nt_col).mode().sort().first().alias('nt_type')])
                .to_pandas()
            )
        else:
            weight_sum_type = (
                pl_pairs_type.group_by(['type_pre', 'type_post'])
                .agg(pl.col('weight').sum().alias('weight'))
                .to_pandas()
            )
        total_incoming_per_type_orig = weight_sum_type.groupby('type_post')['weight'].sum().reset_index(name='total_incoming_weight')
        
        if pbar_type: pbar_type.update(1) # Step 1: Weight aggregation

        if pbar_type: pbar_type.update(1) # Step 2: (skipped - not using static post counts)

        # Calculate type-level ratio using GLOBAL denominator when available
        if global_incoming_weights is not None and 'type_post' in global_incoming_weights.columns:
            conn_type = weight_sum_type.merge(
                global_incoming_weights[['type_post', 'total_incoming_weight']],
                on='type_post', how='left',
            )
        else:
            conn_type = weight_sum_type.merge(total_incoming_per_type_orig, on='type_post', how='left')
        # Local-total fallback for types missing from the global table (same
        # rationale as the main aggregation above - no 0/NaN ratios).
        conn_type['total_incoming_weight'] = conn_type['total_incoming_weight'].fillna(
            conn_type['type_post'].map(
                total_incoming_per_type_orig.set_index('type_post')['total_incoming_weight']
            )
        )
        conn_type['connection_ratio'] = conn_type.apply(
            lambda row: row['weight'] / row['total_incoming_weight'] 
            if pd.notnull(row['total_incoming_weight']) and row['total_incoming_weight'] > 0 
            else float('nan'),
            axis=1
        )
        
        if pbar_type: pbar_type.update(1) # Step 3: Ratios calculated

        # Type-level traversal probability from the aggregate method (same
        # semantics as the main aggregation above).
        prob_series = _type_probability_series(
            bodyid_pairs_type, 'type_pre', 'type_post', aggregate_method
        )
        if prob_series is None:
            conn_type['traversal_probability'] = conn_type['connection_ratio'] / 0.3
            conn_type.loc[conn_type['traversal_probability'] > 1, 'traversal_probability'] = 1
        else:
            conn_type = conn_type.merge(
                prob_series.reset_index(), how='left', on=['type_pre', 'type_post']
            )
        conn_type['block_probability'] = 1 - conn_type['traversal_probability']
        
        if pbar_type: 
            pbar_type.update(1) # Step 4: Traversal probabilities
            pbar_type.close()
        
        conn_type = conn_type.fillna({'traversal_probability': 0.0, 'block_probability': 1.0}).infer_objects()
        type_keep_cols = ['type_pre', 'type_post', 'weight', 'connection_ratio',
                          'traversal_probability', 'block_probability']
        if 'nt_type' in conn_type.columns:
            type_keep_cols.append('nt_type')
        conn_type = conn_type[type_keep_cols]
        
        # if len(bodyid_pairs) > 50000:
        #     print(f"  Enrichment complete. Result shape: {conn_type.shape}. Returning results...", flush=True)
        return conn_df, conn_type, conn_group
    else:
        # No custom groups - return original type aggregation only
        conn_type = conn_aggregated.rename(columns={group_pre: 'type_pre', group_post: 'type_post'})
        # if len(bodyid_pairs) > 50000:
        #     print(f"  Enrichment complete. Result shape: {conn_type.shape}. Returning results...", flush=True)
        return conn_df, conn_type, None


def SankeyDirect(cmat, file_path, showfig=True, node_color='rgba(31, 119, 180, 0.8)', link_color='rgba(0, 0, 0, 0.2)'):
    """
    Create a Sankey diagram from a connection matrix.
    
    Parameters
    ----------
    cmat : pd.DataFrame
        Connection matrix (rows=source, cols=target)
    file_path : str
        Path to save the HTML file
    showfig : bool
        Whether to show the figure
    node_color : str
        Color for nodes
    link_color : str
        Color for links
    """
    import plotly.graph_objects as go
    
    # Get sources and targets
    sources = cmat.index.tolist()
    targets = cmat.columns.tolist()
    
    # Create node list (sources + targets)
    # Treat them as bipartite for "Direct" connection
    # Source nodes indices: 0 to len(sources)-1
    # Target nodes indices: len(sources) to len(sources)+len(targets)-1
    
    labels = sources + targets
    
    # Create links
    source_indices = []
    target_indices = []
    values = []
    
    for i, src in enumerate(sources):
        for j, tgt in enumerate(targets):
            val = cmat.at[src, tgt]
            if val > 0:
                source_indices.append(i)
                target_indices.append(len(sources) + j)
                values.append(val)

    if not values:
        print("No connections to plot in Sankey diagram.")
        return

    # Create figure
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = labels,
          color = node_color
        ),
        link = dict(
          source = source_indices,
          target = target_indices,
          value = values,
          color = link_color
      ))])

    fig.update_layout(title_text="Direct Connections Sankey Diagram", font_size=10)
    
    # Save
    fig.write_html(file_path)
    print(f"Sankey diagram saved to {file_path}")
    
    if showfig:
        fig.show()

def Vis3S(data_df,**kwargs): 
    """ Visualize Soma, Skeletons, Synapses or synapse distributions
    Args:
        data_df (pandas.DataFrame): dataframe contains centroid, classification, axis lengths (Ellipse) or radius (Circle).
    """
    
    options = {
        "save_path" : '_3S',
        "title"     : 'MyTitle',
        "classby"   : 'type',
        "plane"     : 'xz',
        "alpha"     : .3,
        "dpi"       : 300,
        "toPlot"    : 'soma', # "soma" or "synapse_distribution" or "synapse" or "skeleton"
        "xlim"      : (0,50000),
        "ylim"      : (50000,0), # reversed
        "showfig"   : False, # faster than True
        "facecolor" : bokeh.palettes.Set1[9],
        "site"      : None, # None, 'pre' or 'post'
        "snp_rois"   : None,
        "show_mesh"  : True,
        "mesh_roi"   : None,
        "roi_range"  : 'primary_rois', # {"primary_rois", "all_rois"}, see more details in neuprint.
        "mesh_color"    : [0.1,0.1,0.1],
        "mesh_alpha"    : 0.1,
        "confidence"    : 0,
        "synapseRadius" : 100,
        "synpase_file_path" : None,
        "save_format": '.png',
        "dataset": 'hemibrain',
        "data_folder": None
    }
    options.update(kwargs)
    if options['snp_rois'] != None and options['mesh_roi'] == None: 
        options['mesh_roi'] = options['snp_rois']
    elif options['snp_rois'] == None and options['mesh_roi'] == None:
        options['mesh_roi'] = ['LH(R)', 'AL(R)', 'EB']
    op = SimpleNamespace(**options)
    
    # Mesh loading
    roimesh = None
    if op.show_mesh:
        roiunits = []
        # Only load meshes for hemibrain for now, or if we have FAFB meshes
        # FAFB meshes are not standard in this repo yet
        if 'hemibrain' in str(op.dataset).lower():
            for roi in op.mesh_roi:
                mesh_file = os.path.join('navis_roi_meshes_json',op.roi_range,roi+'.json')
                if os.path.exists(mesh_file):
                    mesh = navis.Volume.from_json(mesh_file)
                    roiunits.append(mesh)
                else:
                    print('mesh file %s.json not found!'%(roi))
            if roiunits:
                roimesh = navis.Volume.combine(roiunits)
    
    if op.toPlot == 'synapse':
        if op.synapse_file_path:
            snp_file = pd.ExcelFile(op.synapse_file_path)
        else:
            print("Warning: synapse_file_path not provided for synapse plot")
            return

    summary_df = data_df.copy()
    if op.toPlot == 'soma':
        if 'somaLocation' in summary_df.columns:
            print('not found soma of %d neurons'%(summary_df['somaLocation'].isnull().sum()))
            summary_df = summary_df[summary_df['somaLocation'].notnull()]
        else:
            print("Warning: somaLocation column missing")
    elif op.toPlot == 'synapse_distribution':
        if 'snpN_roi' in summary_df.columns:
            print('drop %d neurons having no more than 1 synapses in the ROI'%((summary_df['snpN_roi']<=1).sum()))
            summary_df = summary_df[summary_df['snpN_roi'] > 1]
            
    if op.classby in summary_df.columns:
        print('drop %d unclassified neurons'%(summary_df[op.classby].isnull().sum()))
        summary_df = summary_df[summary_df[op.classby].notnull()]
    summary_df = summary_df.reset_index(drop=True)
    
    classes = sorted(summary_df[op.classby].unique().tolist()) if op.classby in summary_df.columns else ['All']
    classN = len(classes)
    print('categorized by %s:'%(op.classby), classes)
    
    # Color handling
    colors = op.facecolor
    if isinstance(colors, list):
        multi_factor = int(np.ceil(classN / len(colors)))
        if multi_factor > 1: 
            print('Repeated colors were used in plot.')
            colors *= multi_factor
        colors = colors[:classN]
    
    legend_handles = [mp.Patch(color=colors[i],label=classes[i]) for i in range(len(classes))]
    
    # Subplot layout
    lower = int(np.sqrt(classN))
    upper = int(np.ceil(np.sqrt(classN)))
    if lower**2 <= classN <= lower*upper:
        rowN = lower
        colN = upper
    elif lower*upper < classN < upper**2:
        rowN = upper
        colN = upper
    else:
        rowN = upper
        colN = upper
    rowN = max(rowN,2)
    colN = max(colN,2)
    print("subplot size: rowN = %d,colN = %d"%(rowN,colN))
    
    fig, ax = plt.subplots(tight_layout=True,dpi=op.dpi,subplot_kw={'aspect': 'equal'})
    fig_sup, axes = plt.subplots(nrows=rowN,ncols=colN,sharex=True,sharey=True,dpi=op.dpi,subplot_kw={'aspect': 'equal'})
    np.vectorize(lambda axes:axes.axis('off'))(axes)
    fig_sup.suptitle(op.title+'_subplots')
    
    ellipses = []
    skeletons = []
    
    # Load skeletons if needed
    if op.toPlot == 'skeleton':
        if is_fafb_dataset(str(op.dataset)):
            import fafb_utils
            import zipfile
            import io
            
            if op.data_folder is None:
                project_root = os.path.dirname(os.path.dirname(__file__))
                resolved_dir = resolve_flywire_dataset_dir(project_root, op.dataset)
                op.data_folder = str(resolved_dir) if resolved_dir is not None else None

            # The healed zip is served directly (read-only; a legacy .zst
            # is opened read-only only when no zip exists).
            bundle = (
                fafb_utils.get_fafb_skeleton_bundle(op.data_folder)
                if op.data_folder is not None else None
            )
            if bundle is not None:
                print("Loading skeletons from the healed zip...")
                try:
                    for ind in summary_df.index:
                        bodyid = str(summary_df.at[ind, 'bodyId'])
                        content = bundle.get(int(bodyid))
                        if content is None:
                            continue
                        try:
                            n = navis.read_swc(io.StringIO(content))
                            n.name = bodyid
                            cls = summary_df.at[ind, op.classby]
                            cls_idx = classes.index(cls)
                            n.color = colors[cls_idx]
                            skeletons.append(n)
                        except Exception as e:
                            print(f"Error reading SWC for {bodyid}: {e}")
                except Exception as e:
                    print(f"Error opening healed zip: {e}")
                finally:
                    try:
                        bundle.close()
                    except Exception:
                        pass
            else:
                zip_path = (
                    fafb_utils.get_fafb_skeleton_zip(op.data_folder)
                    if op.data_folder is not None else None
                )
                if zip_path:
                    print(f"Loading skeletons from {zip_path}...")
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as z:
                            all_files = set(z.namelist())
                            for ind in summary_df.index:
                                bodyid = str(summary_df.at[ind,'bodyId'])
                                filename = f"{bodyid}.swc"
                                if filename in all_files:
                                    with z.open(filename) as f:
                                        content = f.read()
                                        try:
                                            n = navis.read_swc(io.BytesIO(content))
                                            n.name = bodyid
                                            # Assign color based on class
                                            cls = summary_df.at[ind, op.classby]
                                            cls_idx = classes.index(cls)
                                            n.color = colors[cls_idx]
                                            skeletons.append(n)
                                        except Exception as e:
                                            print(f"Error reading SWC for {bodyid}: {e}")
                                else:
                                    # print(f"Skeleton for {bodyid} not found in zip.")
                                    pass
                    except Exception as e:
                        print(f"Error opening zip file: {e}")
        else:
            # NeuPrint fetch: use the shared cache-aware batched path so this
            # legacy 2D view follows the same online-fetch policy as 3D
            # visualization and Find Similar.
            try:
                from morphology import fetch_skeletons_on_demand_batch
                bodyids = [int(summary_df.at[ind, 'bodyId'])
                           for ind in summary_df.index]
                by_id = fetch_skeletons_on_demand_batch(
                    str(op.dataset), bodyids, project_root=_REPO_ROOT,
                    persist=False,
                )
                for ind in summary_df.index:
                    bodyid = int(summary_df.at[ind, 'bodyId'])
                    n = by_id.get(bodyid)
                    if n is None:
                        continue
                    n.name = str(summary_df.at[ind, 'bodyId'])
                    cls = summary_df.at[ind, op.classby]
                    cls_idx = classes.index(cls)
                    n.color = colors[cls_idx]
                    skeletons.append(n)
            except Exception as e:
                print(f"Warning: could not fetch hemibrain skeletons: {e}")

    for i,cla in enumerate(classes):
        df = summary_df[summary_df[op.classby] == cla]
        ax_x = i % rowN
        ax_y = int(i / rowN)
        
        # Plot mesh
        if roimesh:
            navis.plot2d(roimesh,method='2d',ax=axes[ax_x,ax_y],view=(op.plane[0],op.plane[1]),color=op.mesh_color,alpha=op.mesh_alpha)
        
        if op.toPlot == 'skeleton':
            # Filter skeletons for this class
            class_skels = [s for s in skeletons if getattr(s, 'name', '') in df['bodyId'].astype(str).values]
            if class_skels:
                navis.plot2d(class_skels, method='2d', ax=axes[ax_x,ax_y], view=(op.plane[0],op.plane[1]), color=colors[i], alpha=op.alpha)
                
        elif op.toPlot == 'soma':
            for ind in df.index:
                if 'somaLocation' in df.columns and isinstance(df.at[ind,'somaLocation'], str):
                    somaLoc_str = df.at[ind,'somaLocation'][1:-1].split(', ')
                    name_str = 'xyz'
                    somaLoc = {name_str[k]: int(somaLoc_str[k]) for k in range(3)}
                    e = mp.Circle(xy = (somaLoc[op.plane[0]], somaLoc[op.plane[1]]),
                            radius = df.at[ind,'somaRadius'] if 'somaRadius' in df.columns else 100,
                            alpha = op.alpha,
                            facecolor = colors[i],
                    )
                    ellipses.append(copy(e))
                    axes[ax_x,ax_y].add_patch(copy(e))
                    
    # Save
    if op.save_path:
        plt.savefig(op.save_path + op.save_format)
        print(f"Saved figure to {op.save_path + op.save_format}")
        
    if op.showfig:
        plt.show()

def build_synapse_mesh(pre_coords, post_coords, mode='sphere', size=100, color='red', opacity=1.0, name='synapses'):
    """
    Build a single mesh containing all synapses with specified geometry.
    
    Parameters
    ----------
    pre_coords : pd.DataFrame or np.ndarray
        (N, 3) array of pre-synaptic coordinates (x, y, z)
    post_coords : pd.DataFrame or np.ndarray
        (N, 3) array of post-synaptic coordinates (x, y, z)
    mode : str
        'sphere', 'cone', or 'tetrahedron'
    size : float or np.ndarray
        Size of the glyphs (radius or scale). Can be a single float or an array of shape (N,).
    color : str or list
        Color of the mesh
    opacity : float
        Opacity of the mesh
    name : str
        Name for the trace
        
    Returns
    -------
    go.Mesh3d
        Plotly Mesh3d trace
    """
    import numpy as np
    import plotly.graph_objects as go
    
    # Convert to numpy arrays
    if hasattr(pre_coords, 'values'):
        pre_coords = pre_coords.values
    if hasattr(post_coords, 'values'):
        post_coords = post_coords.values
        
    n_synapses = len(pre_coords)
    if n_synapses == 0:
        return go.Mesh3d()
        
    # Calculate midpoints and direction vectors
    midpoints = (pre_coords + post_coords) / 2
    vectors = post_coords - pre_coords
    
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1)
    # Handle zero length vectors (shouldn't happen but good to be safe)
    norms[norms == 0] = 1
    directions = vectors / norms[:, np.newaxis]
    
    # Define template geometry (centered at origin, aligned with Z axis)
    # Use unit size for template, scale later
    template_size = 1.0
    
    if mode == 'sphere':
        # UV Sphere with 48 faces (8 segments, 4 rings)
        # This provides a smoother approximation than the icosahedron (20 faces)
        N = 8 # segments
        M = 4 # rings (stacks)
        verts = []
        
        # North Pole
        verts.append([0, 0, template_size])
        
        # Rings
        for i in range(1, M):
            phi = np.pi * i / M
            z = template_size * np.cos(phi)
            r_ring = template_size * np.sin(phi)
            for j in range(N):
                theta = 2 * np.pi * j / N
                x = r_ring * np.cos(theta)
                y = r_ring * np.sin(theta)
                verts.append([x, y, z])
        
        # South Pole
        verts.append([0, 0, -template_size])
        
        verts_template = np.array(verts)
        
        faces = []
        # Top cap: North Pole (0) to Ring 1 (1..N)
        for j in range(N):
            idx1 = 1 + j
            idx2 = 1 + (j + 1) % N
            faces.append([0, idx1, idx2])
            
        # Middle rings
        for i in range(M - 2):
            start_curr = 1 + i * N
            start_next = 1 + (i + 1) * N
            for j in range(N):
                curr1 = start_curr + j
                curr2 = start_curr + (j + 1) % N
                next1 = start_next + j
                next2 = start_next + (j + 1) % N
                
                faces.append([curr1, next1, curr2])
                faces.append([next1, next2, curr2])
                
        # Bottom cap: Ring M-1 to South Pole (last index)
        last_idx = len(verts) - 1
        start_last_ring = 1 + (M - 2) * N
        for j in range(N):
            idx1 = start_last_ring + j
            idx2 = start_last_ring + (j + 1) % N
            faces.append([last_idx, idx2, idx1])
            
        faces_template = np.array(faces)
        
    elif mode == 'cone':
        # Cone with 48 faces (24 segments)
        # Pointing along Z.
        s = template_size
        N = 24 # segments
        
        verts = []
        # Tip
        verts.append([0, 0, s]) # Index 0
        # Base Center
        verts.append([0, 0, -s]) # Index 1
        
        # Base Ring
        for i in range(N):
            theta = 2 * np.pi * i / N
            x = s * np.cos(theta)
            y = s * np.sin(theta)
            verts.append([x, y, -s])
            
        verts_template = np.array(verts)
        
        faces = []
        # Sides: Tip (0) to Ring (2..N+1)
        for i in range(N):
            idx1 = 2 + i
            idx2 = 2 + (i + 1) % N
            faces.append([0, idx1, idx2])
            
        # Base: Base Center (1) to Ring (2..N+1)
        # Note: Order reversed for correct normal
        for i in range(N):
            idx1 = 2 + i
            idx2 = 2 + (i + 1) % N
            faces.append([1, idx2, idx1])
            
        faces_template = np.array(faces)
        
    elif mode == 'tetrahedron':
        # Tetrahedron (4 vertices, 4 faces)
        # "pre-post direction length ... 50% longer"
        # Standard tetrahedron inscribed in cube
        s = template_size
        z_scale = 1.5
        verts_template = np.array([
            [s, s, s*z_scale],
            [s, -s, -s*z_scale],
            [-s, s, -s*z_scale],
            [-s, -s, s*z_scale]
        ])
        faces_template = np.array([
            [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]
        ])
    else:
        raise ValueError(f"Unknown mode: {mode}")
        
    n_verts_template = len(verts_template)
    n_faces_template = len(faces_template)
    
    # Construct rotation matrices to align Z axis (0,0,1) to 'directions'
    # R = [X_new, Y_new, Z_new]
    # Z_new = direction
    # X_new = cross(up, Z_new). If Z_new ~ up, use another up.
    
    z_axis = directions
    
    # Arbitrary up vector (0, 1, 0)
    up = np.array([0, 1, 0])
    
    # Check for parallel vectors
    cross_prod = np.cross(up, z_axis)
    cross_norms = np.linalg.norm(cross_prod, axis=1)
    
    # If parallel to Y, use X as up
    mask_parallel = cross_norms < 1e-6
    if np.any(mask_parallel):
        # Create a copy of up vectors
        up_vectors = np.tile(up, (n_synapses, 1))
        up_vectors[mask_parallel] = np.array([1, 0, 0])
        x_axis = np.cross(up_vectors, z_axis)
    else:
        x_axis = np.cross(up, z_axis)
        
    # Normalize X
    x_norms = np.linalg.norm(x_axis, axis=1)
    x_axis = x_axis / x_norms[:, np.newaxis]
    
    # Y = Z x X
    y_axis = np.cross(z_axis, x_axis)
    
    # Construct Rotation Matrices (N, 3, 3)
    # R[i] = [x_axis[i], y_axis[i], z_axis[i]] (columns)
    # So R[i] @ v_template should align v_template to world
    # v_world = R @ v_local
    
    # Stack axes to form R: shape (N, 3, 3)
    # Transpose to get columns: stack along last axis
    R = np.stack([x_axis, y_axis, z_axis], axis=2)
    
    # Apply rotation
    rotated_verts = np.einsum('nij,vj->nvi', R, verts_template)
    
    # Apply scaling
    if np.isscalar(size):
        rotated_verts *= size
    else:
        size_arr = np.array(size)
        if size_arr.ndim == 1 and len(size_arr) == n_synapses:
             rotated_verts *= size_arr[:, np.newaxis, np.newaxis]
        else:
             # Fallback if size is not compatible
             rotated_verts *= size
    
    # Add midpoints
    # (N, V, 3) + (N, 1, 3)
    final_verts = rotated_verts + midpoints[:, np.newaxis, :]
    
    # Flatten vertices
    all_verts = final_verts.reshape(-1, 3)
    
    # Construct faces
    # Faces indices need to be offset
    # (N, F, 3)
    offsets = np.arange(n_synapses) * n_verts_template
    all_faces = faces_template[np.newaxis, :, :] + offsets[:, np.newaxis, np.newaxis]
    all_faces = all_faces.reshape(-1, 3)
    
    # Create Mesh3d
    return go.Mesh3d(
        x=all_verts[:, 0],
        y=all_verts[:, 1],
        z=all_verts[:, 2],
        i=all_faces[:, 0],
        j=all_faces[:, 1],
        k=all_faces[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.1, specular=0.1),
        lightposition=dict(x=1000, y=1000, z=2000)
    )


def build_site_mesh(coords, mode='sphere', size=20.0, color='red', opacity=1.0, name='sites'):
    """Build a single Mesh3d containing site glyphs (spheres/cones) at *coords*.

    Unlike :func:`build_synapse_mesh`, which draws a glyph from a pre- to a
    post-synaptic site along a direction vector, this helper places a glyph
    centered on each coordinate. Cones point along +z (a fixed orientation),
    so it is used for the pre/post-site markers of ``pre_post`` synapse mode.

    Parameters
    ----------
    coords : pd.DataFrame or np.ndarray
        (N, 3) array of site coordinates (x, y, z).
    mode : str
        'sphere' or 'cone'.
    size : float
        Glyph size (radius/scale).
    color : str
        Color of the mesh.
    opacity : float
        Opacity of the mesh.
    name : str
        Name for the trace.

    Returns
    -------
    go.Mesh3d
        Plotly Mesh3d trace (an empty Mesh3d when there are no points).
    """
    import numpy as np
    import plotly.graph_objects as go

    if hasattr(coords, 'values'):
        coords = coords.values
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(1, 3)
    n_sites = coords.shape[0]
    if n_sites == 0:
        return go.Mesh3d()

    template_size = 1.0
    if mode == 'sphere':
        # UV sphere (8 segments, 4 rings), same topology as build_synapse_mesh.
        N = 8
        M = 4
        verts = [[0, 0, template_size]]
        for i in range(1, M):
            phi = np.pi * i / M
            z = template_size * np.cos(phi)
            r_ring = template_size * np.sin(phi)
            for j in range(N):
                theta = 2 * np.pi * j / N
                verts.append([r_ring * np.cos(theta), r_ring * np.sin(theta), z])
        verts.append([0, 0, -template_size])
        verts_template = np.array(verts)
        faces = []
        for j in range(N):
            faces.append([0, 1 + j, 1 + (j + 1) % N])
        for i in range(M - 2):
            start_curr = 1 + i * N
            start_next = 1 + (i + 1) * N
            for j in range(N):
                curr1 = start_curr + j
                curr2 = start_curr + (j + 1) % N
                next1 = start_next + j
                next2 = start_next + (j + 1) % N
                faces.append([curr1, next1, curr2])
                faces.append([next1, next2, curr2])
        last_idx = len(verts) - 1
        start_last = 1 + (M - 2) * N
        for j in range(N):
            faces.append([last_idx, start_last + (j + 1) % N, start_last + j])
        faces_template = np.array(faces)
    elif mode == 'cone':
        # Cone (24 segments) pointing along +z.
        s = template_size
        N = 24
        verts = [[0, 0, s], [0, 0, -s]]
        for i in range(N):
            theta = 2 * np.pi * i / N
            verts.append([s * np.cos(theta), s * np.sin(theta), -s])
        verts_template = np.array(verts)
        faces = []
        for i in range(N):
            faces.append([0, 2 + i, 2 + (i + 1) % N])
        for i in range(N):
            faces.append([1, 2 + (i + 1) % N, 2 + i])
        faces_template = np.array(faces)
    else:
        raise ValueError(f"Unknown site mode: {mode}")

    n_v = len(verts_template)
    n_f = len(faces_template)
    # Scale each template instance, then translate to its site coordinate.
    scaled = verts_template * size
    all_verts = scaled[np.newaxis, :, :] + coords[:, np.newaxis, :]
    all_verts = all_verts.reshape(-1, 3)
    offsets = np.arange(n_sites) * n_v
    all_faces = faces_template[np.newaxis, :, :] + offsets[:, np.newaxis, np.newaxis]
    all_faces = all_faces.reshape(-1, 3)

    return go.Mesh3d(
        x=all_verts[:, 0],
        y=all_verts[:, 1],
        z=all_verts[:, 2],
        i=all_faces[:, 0],
        j=all_faces[:, 1],
        k=all_faces[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.1, specular=0.1),
        lightposition=dict(x=1000, y=1000, z=2000),
    )



# =============================================================================
# Module-level cache for the Polars engine neuron table (moved from
# statvis_polars.py; renamed from _NEURON_DF_CACHE to avoid clashing with
# the pandas engine cache of the same name above).
_PL_NEURON_DF_CACHE = {}  # (dataset_path, mtime_ns) -> pl.DataFrame
_PL_NEURON_DF_CACHE_MAX = 4


# Polars engine (moved verbatim from statvis_polars.py; the unified public
# entry points above dispatch here by input type / engine=).
# =============================================================================


def _load_local_neuron_df_cached(dataset_path: str, is_fafb: bool) -> pl.DataFrame:
    """Load a FAFB/BANC local table once per (path, mtime), cached.

    ``is_fafb`` is retained as the private function's historical parameter
    name; callers now pass ``True`` for either local release so exact body-ID
    normalization is applied to BANC as well.
    """
    try:
        # mtime_ns: getmtime() has only second resolution on some filesystems,
        # so a file regenerated within the same second would hit a stale entry.
        mtime = os.stat(dataset_path).st_mtime_ns
    except OSError:
        mtime = None
    cache_key = (dataset_path, mtime)
    cached = _PL_NEURON_DF_CACHE.get(cache_key)
    if cached is not None:
        return cached

    is_local_release = bool(is_fafb)
    # Handle FAFB/BANC local-release tables which might use exact string IDs.
    # NOTE: polars >= 1.0 removed the `dtypes=` kwarg (use schema_overrides).
    if str(dataset_path).lower().endswith('.parquet'):
        ndf = pl.read_parquet(dataset_path)
    elif is_local_release:
        ndf = pl.read_csv(
            dataset_path,
            infer_schema_length=10000,
            schema_overrides={'bodyId': pl.Utf8},
        )
    else:
        ndf = pl.read_csv(dataset_path, infer_schema_length=10000)
        if 'bodyId' in ndf.columns:
            ndf = ndf.with_columns(pl.col('bodyId').cast(pl.Utf8))

    if is_local_release and 'bodyId' in ndf.columns:
        # A numeric parquet column may already have lost precision before it
        # reached this reader, but any exact integer/string representation is
        # still validated here.  Keep the Polars enrichment path consistent
        # with the pandas reader instead of merely changing the display dtype.
        ndf = ndf.with_columns(
            pl.col('bodyId').map_elements(
                lambda value: normalize_flywire_body_id(value, field='bodyId'),
                return_dtype=pl.Utf8,
            ).alias('bodyId')
        )

    if len(_PL_NEURON_DF_CACHE) >= _PL_NEURON_DF_CACHE_MAX:
        _PL_NEURON_DF_CACHE.pop(next(iter(_PL_NEURON_DF_CACHE)))
    _PL_NEURON_DF_CACHE[cache_key] = ndf
    return ndf


def build_bodyid_label_map(label_mapper, dataset: str, neuron_df: pl.DataFrame) -> dict:
    """
    Build a comprehensive bodyId → std_label map from label_mapper.
    
    This implements the user's 6-step label mapping approach:
    Step 3: Convert label_mapper's type/bodyId/instance → std_label map 
            to a complete bodyId → std_label map using the neuron index file.
    
    The label_mapper may contain mappings by:
    - bodyId: Direct bodyId → std_label
    - type: type_name → std_label (need to expand to all bodyIds of that type)
    - instance: instance_name → std_label (need to expand to all bodyIds with that instance)
    
    Args:
        label_mapper: LabelMapper object with source/target/intermediate mappings
        dataset: Dataset name (e.g., 'hemibrain:v1.2.1')
        neuron_df: Polars DataFrame with neuron index (must have 'bodyId', 'type', optionally 'instance')
        
    Returns:
        Dict[str, str]: Mapping of bodyId → std_label
    """
    if label_mapper is None or neuron_df is None or neuron_df.is_empty():
        return {}
    
    bodyid_label_map = {}
    
    # Ensure bodyId is string
    if 'bodyId' in neuron_df.columns:
        neuron_df = neuron_df.with_columns(pl.col('bodyId').cast(pl.Utf8))
    else:
        return {}
    
    # Build lookup dictionaries for efficient mapping
    # type → [bodyIds]
    type_to_bodyids = {}
    if 'type' in neuron_df.columns:
        type_groups = neuron_df.group_by('type').agg(pl.col('bodyId').alias('bodyIds'))
        for row in type_groups.iter_rows(named=True):
            if row['type'] is not None:
                type_to_bodyids[str(row['type'])] = row['bodyIds']
    
    # instance → [bodyIds]
    instance_to_bodyids = {}
    if 'instance' in neuron_df.columns:
        instance_groups = neuron_df.group_by('instance').agg(pl.col('bodyId').alias('bodyIds'))
        for row in instance_groups.iter_rows(named=True):
            if row['instance'] is not None:
                instance_to_bodyids[str(row['instance'])] = row['bodyIds']
    
    # Helper to normalize dataset name for lookups
    def sanitize(name: str) -> str:
        return canonical_dataset_name(name).replace(':', '_').replace('.', '_').replace('-', '_')
    
    dataset_sanitized = sanitize(dataset)
    
    # Process all mappings (source, target, intermediate)
    all_mappings = []
    for mapping_dict in [label_mapper._source_mapping, label_mapper._target_mapping, label_mapper._intermediate_mapping]:
        for std_label, ds_dict in mapping_dict.items():
            # Try both original and sanitized dataset names
            neuron_ids = []
            if dataset in ds_dict:
                neuron_ids = ds_dict[dataset]
            elif dataset_sanitized in ds_dict:
                neuron_ids = ds_dict[dataset_sanitized]
            
            for neuron_id in neuron_ids:
                all_mappings.append((str(neuron_id), std_label))
    
    # Process each mapping and expand to bodyIds
    # Precompute the set of existing bodyIds once (O(1) lookups) instead of
    # filtering the whole neuron_df for every mapping (O(M*N) per call).
    bodyid_set = set(neuron_df['bodyId'].to_list())

    for neuron_id, std_label in all_mappings:
        # First, check if neuron_id is a direct bodyId
        # If it matches a bodyId in the neuron_df, map it directly
        if neuron_id in bodyid_set:
            bodyid_label_map[neuron_id] = std_label
            continue
        
        # Check if neuron_id is a type name
        if neuron_id in type_to_bodyids:
            for bid in type_to_bodyids[neuron_id]:
                # Don't overwrite existing mappings (first mapping wins)
                if bid not in bodyid_label_map:
                    bodyid_label_map[bid] = std_label
            continue
        
        # Check if neuron_id is an instance name
        if neuron_id in instance_to_bodyids:
            for bid in instance_to_bodyids[neuron_id]:
                if bid not in bodyid_label_map:
                    bodyid_label_map[bid] = std_label
            continue
        
        # If none of the above, just store the mapping in case it's used directly
        # This handles cases where the ID might be used elsewhere
        bodyid_label_map[neuron_id] = std_label
    
    return bodyid_label_map


def prepare_connection_data(conn_data, level='type'):
    """
    Pre-process connection data into a Polars DataFrame optimized for joining.
    Aggregates multiple edges between same nodes.
    """
    # Determine source and target columns
    src_col = f'{level}_pre' if f'{level}_pre' in conn_data.columns else 'bodyId_pre'
    tgt_col = f'{level}_post' if f'{level}_post' in conn_data.columns else 'bodyId_post'
    
    # Convert to Polars if needed
    if isinstance(conn_data, pd.DataFrame):
        df = pl.from_pandas(conn_data)
    else:
        df = conn_data
        
    # Cast columns to string for consistent joining
    df = df.with_columns([
        pl.col(src_col).cast(pl.Utf8).alias('src'),
        pl.col(tgt_col).cast(pl.Utf8).alias('tgt')
    ])
    
    # Define aggregations. Synapse counts are integral even when an upstream
    # pandas/Polars conversion represents them as 5.0, so normalize the
    # semantic count column before summing and writing path metrics.
    aggs = [
        pl.col('weight').cast(pl.Int64, strict=False).sum().alias('weight')
    ]
    
    if 'traversal_probability' in df.columns:
        aggs.append(pl.col('traversal_probability').mean().alias('traversal_probability'))
    
    if 'connection_ratio' in df.columns:
        aggs.append(pl.col('connection_ratio').mean().alias('connection_ratio'))
        
    if 'nt_type' in df.columns:
        # For nt_type, take the first value (most edges between same nodes have same NT)
        aggs.append(pl.col('nt_type').first().alias('nt_type'))

    # Group and aggregate
    df_agg = df.group_by(['src', 'tgt']).agg(aggs)
    
    return df_agg


def process_batch_polars(paths_batch, df_conn, level='type', keyword_in_path_to_remove=None,
                         type_to_label_map=None):
    """
    Process a batch of paths using Polars.
    
    Args:
        type_to_label_map: Optional dict mapping original type names to standardized labels
    """
    if not paths_batch:
        return pl.DataFrame(), pl.DataFrame()
        
    # 1. Create DataFrame from paths
    # paths_batch is list of lists
    # We want: path_id, node_idx, node
    
    # Create a DataFrame with a single column 'path' containing lists
    # Note: Polars creation from list of lists might infer types. Ensure strings.
    # It's safer to convert all nodes to strings first in Python if they are mixed
    paths_str = [[str(n) for n in p] for p in paths_batch]
    
    df_paths = pl.DataFrame({'path_nodes': paths_str})
    df_paths = df_paths.with_row_index('path_id')
    
    # 2. Explode to get edges
    # We need to create edges (u, v) for each path
    # Strategy: Explode nodes, then shift to get next node
    
    df_exploded = df_paths.explode('path_nodes')
    
    # We need to group by path_id to perform shift operation safely
    # But explode keeps order.
    
    df_edges = df_exploded.with_columns([
        pl.col('path_nodes').alias('src'),
        pl.col('path_nodes').shift(-1).over('path_id').alias('tgt')
    ])
    
    # Filter out the last node which has no target (tgt is null)
    df_edges = df_edges.filter(pl.col('tgt').is_not_null())
    
    # 3. Join with connection data
    # df_conn has ['src', 'tgt', 'weight', 'traversal_probability', ...]
    
    df_joined = df_edges.join(df_conn, on=['src', 'tgt'], how='left')
    
    # Fill missing values (if any edge not found)
    # Check if nt_type exists
    has_nt = 'nt_type' in df_joined.columns
    
    df_joined = df_joined.with_columns([
        pl.col('weight').fill_null(0).cast(pl.Int64, strict=False),
        pl.col('traversal_probability').fill_null(0),
        pl.col('connection_ratio').fill_null(0)
    ])
    
    # 4. Aggregate back to path level
    # We want lists of weights, probs, etc. and summary stats
    
    aggs = [
        pl.col('src').alias('path_nodes_flat'), # We'll reconstruct path string later
        pl.col('weight').alias('weights'),
        pl.col('traversal_probability').alias('probabilities'),
        pl.col('connection_ratio').alias('ratios'),
        
        pl.col('weight').min().alias('min_weight'),
        pl.col('traversal_probability').product().alias('path_prob'),
        pl.col('connection_ratio').min().alias('min_ratio'),
        pl.count('src').alias('length')
    ]
    
    # Add nt_type aggregation if available
    if has_nt:
        aggs.append(pl.col('nt_type').alias('nt_types'))
    
    df_results = df_joined.group_by('path_id', maintain_order=True).agg(aggs)
    
    # 5. Filter zero-weight paths (any edge has weight 0)
    # In Polars, we can check if min_weight > 0
    df_results = df_results.filter(pl.col('min_weight') > 0)
    
    if df_results.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    
    # 6. Reconstruct path string and add original path list
    # We need to join back with df_paths to get the original full path list (including last node)
    # because df_edges lost the last node in 'src' column
    
    df_final = df_results.join(df_paths, on='path_id', how='left')
    
    # Create formatted path string "A->B->C"
    # Apply type_to_label_map if provided to rename types in output
    if type_to_label_map:
        # Map each node in the path list using type_to_label_map dict
        def map_node(node):
            return type_to_label_map.get(str(node), str(node))
        
        # Apply mapping to path_nodes list
        df_final = df_final.with_columns(
            pl.col('path_nodes').list.eval(
                pl.element().map_elements(map_node, return_dtype=pl.Utf8)
            ).alias('path_nodes_mapped')
        )
        df_final = df_final.with_columns(
            pl.col('path_nodes_mapped').list.join('->').alias('path')
        ).drop('path_nodes_mapped')
    else:
        # Polars list join without mapping
        df_final = df_final.with_columns(
            pl.col('path_nodes').list.join('->').alias('path')
        )
    
    # Convert list columns to string for CSV compatibility
    # Format as "[w1, w2, w3]" to match original statvis output
    list_format_cols = [
        (pl.lit("[") + pl.col('weights').list.eval(pl.element().cast(pl.Utf8)).list.join(', ') + pl.lit("]")).alias('weights'),
        (pl.lit("[") + pl.col('probabilities').list.eval(pl.element().cast(pl.Utf8)).list.join(', ') + pl.lit("]")).alias('probabilities'),
        (pl.lit("[") + pl.col('ratios').list.eval(pl.element().cast(pl.Utf8)).list.join(', ') + pl.lit("]")).alias('ratios')
    ]
    
    # Add nt_types formatting if available - use quoted strings for proper parsing
    if 'nt_types' in df_final.columns:
        # Format as ["ACH", "GABA"] so ast.literal_eval can parse it
        list_format_cols.append(
            (pl.lit('["') + pl.col('nt_types').list.eval(pl.element().cast(pl.Utf8)).list.join('", "') + pl.lit('"]')).alias('nt_types')
        )
    
    df_final = df_final.with_columns(list_format_cols)
    
    # Rename path_nodes to path_str (to match statvis output convention)
    # But statvis uses 'path_str' for the list object in pandas.
    # Here we can keep 'path_nodes' as the list column.
    
    # 7. Filter keywords
    excluded = pl.DataFrame()
    if keyword_in_path_to_remove:
        if isinstance(keyword_in_path_to_remove, str):
            keywords = [keyword_in_path_to_remove]
        else:
            keywords = keyword_in_path_to_remove
            
        # Build filter expression
        filter_expr = pl.lit(False)
        for kw in keywords:
            filter_expr = filter_expr | pl.col('path').str.contains(kw, literal=True)
            
        excluded = df_final.filter(filter_expr)
        df_final = df_final.filter(~filter_expr)
        
    # Select and rename columns to match statvis output
    # statvis output: path_str (list), path (str), weights, probabilities, ratios, min_weight, path_prob, min_ratio, length
    
    cols_to_keep = [
        'path', 'weights', 'probabilities', 'ratios', 
        'min_weight', 'path_prob', 'min_ratio', 'length'
    ]
    
    # Add nt_types if available
    if 'nt_types' in df_final.columns:
        cols_to_keep.append('nt_types')
    
    # Note: 'path_nodes' is the list. We can keep it if needed, but CSV writing might stringify it.
    # statvis writes 'path' as string "A->B->C".
    
    df_final_selected = df_final.select(cols_to_keep)
    
    # Handle excluded DataFrame - ensure it has the same schema even if empty
    if excluded.is_empty():
        excluded_selected = df_final_selected.clear()  # Empty DataFrame with same schema
    else:
        excluded_selected = excluded.select(cols_to_keep)
    
    return df_final_selected, excluded_selected


def _write_buffer_to_csv(buffer_list, output_path, append=False):
    """
    Helper function to write buffered DataFrames to CSV efficiently.
    Concatenates all DataFrames first, then writes once.
    
    Performance: ~10-20x faster than writing each batch individually.
    """
    if not buffer_list:
        return
        
    # Concatenate all DataFrames in buffer (Polars concat is very fast)
    if len(buffer_list) == 1:
        df_combined = buffer_list[0]
    else:
        df_combined = pl.concat(buffer_list, rechunk=False)  # rechunk=False is faster
    
    # Write to CSV using Polars native I/O (faster than Python file handles)
    if append:
        # For append mode, use file handle with explicit UTF-8 encoding
        # This is critical on Windows where default encoding is often cp1252
        with open(output_path, 'a', encoding='utf-8', buffering=1024*1024) as f:  # 1MB buffer
            df_combined.write_csv(f, include_header=False)
    else:
        # For initial write, use Polars native (faster)
        df_combined.write_csv(output_path)


def EnrichConnectionTablePolars(conn_table, traversal_probability_threshold=0, dataset=None, script_path=None, target_neurons_df=None, aggregate_method='product', label_mapper=None, global_incoming_weights=None, separate_hemispheres=False, global_incoming_body_weights=None):
    '''Add traversal probability, connection ratio, and layer information to the connection table using Polars
    
    NOTE: When separate_hemispheres=True, the caller is expected to have already applied
    hemisphere suffixes (_L/_R/_U) to type_pre/type_post columns. This function will
    aggregate by those already-suffixed types. The parameter is accepted for API
    compatibility but does not change the aggregation behavior.
    
    IMPLEMENTS USER's 6-STEP LABEL MAPPING APPROACH:
    Step 1: Fetch neurons using original type/bodyId/instance (done by caller)
    Step 2: Aggregate source/target/intermediate maps from label_mapper
    Step 3: Convert label_mapper's type/bodyId/instance → std_label to complete bodyId → std_label
    Step 4: Aggregate bodyId-level graph using std_label from label_mapper  
    Step 5: Aggregate remaining (unmapped) bodyIds by type
    Step 6: Mark source and target by the classification map
    
    Parameters
    ----------
    conn_table : DataFrame
        Connection table to enrich (bodyId-level)
    traversal_probability_threshold : float, optional
        Minimum traversal probability threshold (default: 0)
    dataset : str, optional
        Dataset name (e.g., 'optic-lobe:v1.1') for local dataset lookup
    script_path : str, optional
        Path to script directory containing 'datasets' folder
    target_neurons_df : DataFrame, optional
        Full dataframe of target neurons (with bodyId, type, post columns).
        Used to get correct type-level denominators. If not provided, only
        neurons appearing in connections will be used (less accurate).
    aggregate_method : str, optional
        How the type-level traversal_probability is derived from the
        bodyId-level pairs (default 'product'):
        - 'product': ``1 - prod(1 - p_pair)`` over the deduplicated pairs -
          the type edge is a bundle of parallel channels, so it transmits if
          ANY pair transmits (reliability/OR model; recommended for path
          analysis).
        - 'average': weight-weighted mean of the pair probabilities.
        - 'ratio': ``min(connection_ratio / 0.3, 1)`` (input-share model).
        Same semantics as the pandas engine and coana._apply_type_level_filters.
    label_mapper : LabelMapper, optional
        LabelMapper object for cross-dataset comparison.
        When provided, aggregation uses std_label for mapped neurons and type for unmapped.
    global_incoming_weights : DataFrame, optional
        Pre-computed total incoming weights for each post-synaptic type.
        Should have columns [type_post, total_incoming_weight].
        If provided, used for calculating GLOBAL type-level ratios.
        If None, local ratios (from provided connections only) are calculated.
    global_incoming_body_weights : DataFrame, optional
        Pre-computed total incoming weights for each post-synaptic bodyId.
        Should have columns [bodyId_post, total_incoming_weight].
        If provided, used for calculating GLOBAL bodyId-level ratios.
        Post neurons missing from this table (and all untyped neurons, which
        are grouped by bodyId at type level) fall back to LOCAL totals so
        ratios/probabilities never collapse to 0.
    
    Returns
    -------
    conn_df : DataFrame
        Enriched connection table with bodyId-level metrics
    conn_type : DataFrame
        Type-level aggregation (or std_label-level if label_mapper provided)
    conn_group : DataFrame or None
        Custom group-level aggregation (only if custom_group columns exist)
    '''
    # Convert to Polars
    if isinstance(conn_table, pd.DataFrame):
        conn_df = pl.from_pandas(conn_table)
    else:
        conn_df = conn_table
        
    # Ensure string types for IDs
    conn_df = conn_df.with_columns([
        pl.col('bodyId_pre').cast(pl.Utf8),
        pl.col('bodyId_post').cast(pl.Utf8)
    ])
    
    # Determine grouping columns (use custom_group if available, otherwise type)
    group_pre = 'custom_group_pre' if 'custom_group_pre' in conn_df.columns else 'type_pre'
    group_post = 'custom_group_post' if 'custom_group_post' in conn_df.columns else 'type_post'
    
    # Try to use local dataset first
    use_local = False
    ndf_complete = None
    if dataset and script_path:
        dataset_clean = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
        dataset_path = (
            _flywire_neuron_table_path(dataset, script_path)
            if is_local_connectome_dataset(dataset) else
            os.path.join(
                script_path,
                'datasets',
                dataset_clean,
                f"{dataset_clean}_allneurons_neuron_df.csv"
            )
        )
        
        # Enhanced dataset discovery logic
        if dataset_path is not None and not os.path.exists(dataset_path):
            # Fallback: Try root datasets folder (legacy)
            legacy_path = os.path.join(
                script_path,
                'datasets',
                f"{dataset_clean}_allneurons_neuron_df.csv"
            )
            if os.path.exists(legacy_path):
                dataset_path = legacy_path
            else:
                # Try globbing for any *_allneurons_neuron_df.csv in subdir
                subdir_path = os.path.join(script_path, 'datasets', dataset_clean)
                if os.path.exists(subdir_path):
                    import glob
                    candidates = glob.glob(os.path.join(subdir_path, "*_allneurons_neuron_df.csv"))
                    if candidates:
                        dataset_path = candidates[0]

        if dataset_path is not None and os.path.exists(dataset_path):
            use_local = True
            is_local_release = is_local_connectome_dataset(dataset)
            ndf_complete = _load_local_neuron_df_cached(dataset_path,
                                                        is_local_release)
    
    # Step 3: Build complete bodyId → std_label map from label_mapper
    bodyid_label_map = {}
    if label_mapper and ndf_complete is not None:
        bodyid_label_map = build_bodyid_label_map(label_mapper, dataset, ndf_complete)
    
    # Apply bodyId → std_label mapping to connection table
    # For mapped bodyIds: use std_label
    # For unmapped bodyIds: use original type (Step 5)
    # For untyped neurons (empty/null type): use bodyId as fallback
    #
    # IMPORTANT: When separate_hemispheres=True, the type_pre/type_post columns already 
    # have hemisphere suffixes (e.g., "PPL101_L"). The std_label from label_mapper is the
    # base type (e.g., "PPL101"). We need to preserve the hemisphere suffix by extracting
    # it from the type column and appending it to the std_label.
    if bodyid_label_map:
        # Create a Polars-friendly mapping for vectorized lookup
        map_df = pl.DataFrame({
            'bodyId': list(bodyid_label_map.keys()),
            'std_label': list(bodyid_label_map.values())
        })
        
        # Map pre neurons
        conn_df = conn_df.join(
            map_df.rename({'bodyId': 'bodyId_pre', 'std_label': '_mapped_std_label_pre'}),
            on='bodyId_pre',
            how='left'
        )
        
        if separate_hemispheres:
            # When hemisphere separation is enabled, append the hemisphere suffix from type_pre to std_label
            # Extract suffix: if type_pre ends with _L, _R, or _U, extract that suffix
            conn_df = conn_df.with_columns(
                pl.when(pl.col('type_pre').str.ends_with('_L'))
                    .then(pl.lit('_L'))
                    .when(pl.col('type_pre').str.ends_with('_R'))
                    .then(pl.lit('_R'))
                    .when(pl.col('type_pre').str.ends_with('_U'))
                    .then(pl.lit('_U'))
                    .otherwise(pl.lit(''))
                    .alias('_hemi_suffix_pre')
            )
            # Build std_label_pre: mapped_std_label + hemisphere_suffix, else fall back to type_pre, else bodyId
            conn_df = conn_df.with_columns(
                pl.coalesce([
                    pl.when(pl.col('_mapped_std_label_pre').is_not_null())
                        .then(pl.col('_mapped_std_label_pre') + pl.col('_hemi_suffix_pre'))
                        .otherwise(None),
                    pl.when(pl.col('type_pre').is_not_null() & (pl.col('type_pre') != '')).then(pl.col('type_pre')).otherwise(None),
                    pl.col('bodyId_pre')
                ]).alias('std_label_pre')
            )
            conn_df = conn_df.drop('_mapped_std_label_pre', '_hemi_suffix_pre')
        else:
            # No hemisphere separation: use std_label directly
            conn_df = conn_df.with_columns(
                pl.coalesce([
                    pl.col('_mapped_std_label_pre'),
                    pl.when(pl.col('type_pre').is_not_null() & (pl.col('type_pre') != '')).then(pl.col('type_pre')).otherwise(None),
                    pl.col('bodyId_pre')
                ]).alias('std_label_pre')
            )
            conn_df = conn_df.drop('_mapped_std_label_pre')
        
        # Map post neurons
        conn_df = conn_df.join(
            map_df.rename({'bodyId': 'bodyId_post', 'std_label': '_mapped_std_label_post'}),
            on='bodyId_post',
            how='left'
        )
        
        if separate_hemispheres:
            # Extract hemisphere suffix from type_post
            conn_df = conn_df.with_columns(
                pl.when(pl.col('type_post').str.ends_with('_L'))
                    .then(pl.lit('_L'))
                    .when(pl.col('type_post').str.ends_with('_R'))
                    .then(pl.lit('_R'))
                    .when(pl.col('type_post').str.ends_with('_U'))
                    .then(pl.lit('_U'))
                    .otherwise(pl.lit(''))
                    .alias('_hemi_suffix_post')
            )
            conn_df = conn_df.with_columns(
                pl.coalesce([
                    pl.when(pl.col('_mapped_std_label_post').is_not_null())
                        .then(pl.col('_mapped_std_label_post') + pl.col('_hemi_suffix_post'))
                        .otherwise(None),
                    pl.when(pl.col('type_post').is_not_null() & (pl.col('type_post') != '')).then(pl.col('type_post')).otherwise(None),
                    pl.col('bodyId_post')
                ]).alias('std_label_post')
            )
            conn_df = conn_df.drop('_mapped_std_label_post', '_hemi_suffix_post')
        else:
            conn_df = conn_df.with_columns(
                pl.coalesce([
                    pl.col('_mapped_std_label_post'),
                    pl.when(pl.col('type_post').is_not_null() & (pl.col('type_post') != '')).then(pl.col('type_post')).otherwise(None),
                    pl.col('bodyId_post')
                ]).alias('std_label_post')
            )
            conn_df = conn_df.drop('_mapped_std_label_post')
    else:
        # No label_mapper: std_label = type, or bodyId if type is empty/null
        conn_df = conn_df.with_columns([
            pl.coalesce([
                pl.when(pl.col('type_pre').is_not_null() & (pl.col('type_pre') != '')).then(pl.col('type_pre')).otherwise(None),
                pl.col('bodyId_pre')
            ]).alias('std_label_pre'),
            pl.coalesce([
                pl.when(pl.col('type_post').is_not_null() & (pl.col('type_post') != '')).then(pl.col('type_post')).otherwise(None),
                pl.col('bodyId_post')
            ]).alias('std_label_post')
        ])

    # Custom-group labels follow the same exclusivity chain as std_labels:
    # custom_group -> type -> bodyId (first non-empty wins). A neuron
    # without a custom group must not be dropped from the group aggregation,
    # and it never appears under both its custom group and its raw type.
    if 'custom_group_pre' in conn_df.columns:
        conn_df = conn_df.with_columns([
            pl.coalesce([
                pl.when(pl.col('custom_group_pre').is_not_null()
                        & (pl.col('custom_group_pre') != ''))
                .then(pl.col('custom_group_pre')).otherwise(None),
                pl.when(pl.col('type_pre').is_not_null()
                        & (pl.col('type_pre') != ''))
                .then(pl.col('type_pre')).otherwise(None),
                pl.col('bodyId_pre'),
            ]).alias('custom_group_pre'),
            pl.coalesce([
                pl.when(pl.col('custom_group_post').is_not_null()
                        & (pl.col('custom_group_post') != ''))
                .then(pl.col('custom_group_post')).otherwise(None),
                pl.when(pl.col('type_post').is_not_null()
                        & (pl.col('type_post') != ''))
                .then(pl.col('type_post')).otherwise(None),
                pl.col('bodyId_post'),
            ]).alias('custom_group_post'),
        ])

    # 1. Enrich BodyId Level
    # Need to join 'post' count to conn_df
    
    # Prepare reference dataframe for joining
    ref_df = None
    if ndf_complete is not None:
        ref_df = ndf_complete
    elif target_neurons_df is not None:
        if isinstance(target_neurons_df, pd.DataFrame):
            ref_df = pl.from_pandas(target_neurons_df)
        else:
            ref_df = target_neurons_df
            
    if ref_df is not None:
        # Ensure types
        if 'bodyId' in ref_df.columns:
            ref_df = ref_df.with_columns(pl.col('bodyId').cast(pl.Utf8))
        
        # Join to get 'post'
        # We only need bodyId and post for this step
        if 'post' in ref_df.columns:
            post_lookup = ref_df.select(['bodyId', 'post']).rename({'bodyId': 'bodyId_post'})
            
            # Drop 'post' from conn_df if it exists to avoid collision
            if 'post' in conn_df.columns:
                conn_df = conn_df.drop('post')
                
            conn_df = conn_df.join(post_lookup, on='bodyId_post', how='left')
            conn_df = conn_df.with_columns(pl.col('post').fill_null(0))
        else:
             if 'post' not in conn_df.columns:
                 conn_df = conn_df.with_columns(pl.lit(0).alias('post'))
    else:
        if 'post' not in conn_df.columns:
            conn_df = conn_df.with_columns(pl.lit(0).alias('post'))
    
    # Normalize numeric columns at the engine boundary. Empty pandas frames
    # (for example a no-cache FAFB fallback) can otherwise arrive as Polars
    # String columns and make the ratio division fail even when the local
    # connection weights are numeric.
    if 'weight' in conn_df.columns:
        conn_df = conn_df.with_columns(
            # Synapse counts are integer-valued. Keep that semantic dtype so
            # edge CSVs, path metrics, and weight matrices do not emit 5.0.
            pl.col('weight').cast(pl.Int64, strict=False).alias('weight')
        )

    # Check if connection_ratio already exists and has valid values (from coana.py global calculation)
    # If so, preserve it to maintain the correct global ratio calculation
    has_valid_ratio = False
    if 'connection_ratio' in conn_df.columns:
        ratio_stats = conn_df.select([
            pl.col('connection_ratio').is_not_null().any().alias('has_any'),
            (pl.col('connection_ratio') > 0).any().alias('has_positive')
        ]).to_dicts()[0]
        has_valid_ratio = ratio_stats['has_any'] and ratio_stats['has_positive']

    # Convert global_incoming_body_weights to Polars if provided
    # (bodyId keys are normalized to strings to match conn_df's bodyId_post)
    global_incoming_body_pl = None
    if global_incoming_body_weights is not None:
        if isinstance(global_incoming_body_weights, pd.DataFrame):
            global_incoming_body_pl = pl.from_pandas(global_incoming_body_weights)
        else:
            global_incoming_body_pl = global_incoming_body_weights
        if 'bodyId_post' in global_incoming_body_pl.columns:
            global_incoming_body_pl = global_incoming_body_pl.with_columns(
                pl.col('bodyId_post').cast(pl.Utf8)
            )
        if 'total_incoming_weight' in global_incoming_body_pl.columns:
            global_incoming_body_pl = global_incoming_body_pl.with_columns(
                pl.col('total_incoming_weight')
                .cast(pl.Float64, strict=False)
                .alias('total_incoming_weight')
            )
    
    if not has_valid_ratio:
        # Only recalculate if ratio doesn't exist or has no valid values.
        # GLOBAL bodyId denominators (total incoming to each post neuron from
        # ALL sources) are used when supplied; post neurons missing from the
        # global table fall back to the LOCAL total over this table so ratios
        # never collapse to 0/undefined.
        if global_incoming_body_pl is not None:
            conn_df = conn_df.join(global_incoming_body_pl, on='bodyId_post', how='left')
        else:
            conn_df = conn_df.with_columns(
                pl.lit(None, dtype=pl.Float64).alias('total_incoming_weight')
            )
        local_totals = conn_df.group_by('bodyId_post').agg(
            pl.col('weight').sum().alias('_local_total')
        )
        conn_df = conn_df.join(local_totals, on='bodyId_post', how='left')
        conn_df = conn_df.with_columns(
            pl.coalesce([pl.col('total_incoming_weight'), pl.col('_local_total')])
            .cast(pl.Float64, strict=False)
            .alias('total_incoming_weight')
        ).drop('_local_total')
        
        # Calculate connection_ratio using the (global if available, else local) denominator
        conn_df = conn_df.with_columns(
            pl.when(pl.col('total_incoming_weight') > 0)
            .then(pl.col('weight') / pl.col('total_incoming_weight'))
            .otherwise(None)
            .alias('connection_ratio')
        )
        
        # Drop temporary column
        conn_df = conn_df.drop('total_incoming_weight')
    
    # traversal_probability = connection_ratio / 0.3 (capped at 1.0)
    conn_df = conn_df.with_columns(
        (pl.col('connection_ratio') / 0.3).clip(0.0, 1.0).alias('traversal_probability')
    )
    
    # block_probability = 1 - traversal_probability
    conn_df = conn_df.with_columns(
        (1.0 - pl.col('traversal_probability')).alias('block_probability')
    )
    
    # Filter by threshold
    if traversal_probability_threshold > 0:
        conn_df = conn_df.filter(pl.col('traversal_probability') >= traversal_probability_threshold)
        
    # 2. Aggregation (Step 4 & 5: Aggregate by std_label for mapped, type for unmapped)
    # First deduplicate by bodyId pairs to avoid counting same connection multiple times
    cols_to_keep = ['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post', 
                    'std_label_pre', 'std_label_post', 'weight', 'block_probability', 'traversal_probability']
    if 'custom_group_pre' in conn_df.columns:
        cols_to_keep.extend(['custom_group_pre', 'custom_group_post'])
    
    # Check for NT type column - prefer nt_type_pre (presynaptic NT), fallback to nt_type
    nt_col = None
    if 'nt_type_pre' in conn_df.columns:
        nt_col = 'nt_type_pre'
        cols_to_keep.append('nt_type_pre')
    elif 'nt_type' in conn_df.columns:
        nt_col = 'nt_type'
        cols_to_keep.append('nt_type')
    
    # Keep only existing columns
    cols_to_keep = [c for c in cols_to_keep if c in conn_df.columns]
    bodyid_pairs = conn_df.select(cols_to_keep).unique(subset=['bodyId_pre', 'bodyId_post'])

    # Polars preserves a Null dtype when a whole column contains only nulls.
    # That is valid for the bodyId-level table, but it cannot be joined to the
    # String-typed global type-total table below (this is common for untyped
    # post-synaptic neurons).  Normalize all optional grouping keys before any
    # aggregation/join so an all-null column remains nullable String rather
    # than becoming an unjoinable Null column.
    string_columns = [
        'type_pre', 'type_post', 'std_label_pre', 'std_label_post',
        'custom_group_pre', 'custom_group_post',
    ]
    bodyid_pairs = bodyid_pairs.with_columns([
        pl.col(column).cast(pl.Utf8, strict=False).alias(column)
        for column in string_columns
        if column in bodyid_pairs.columns
    ])
    
    # Rename nt_type_pre to nt_type for consistency in downstream processing
    if nt_col == 'nt_type_pre' and 'nt_type_pre' in bodyid_pairs.columns:
        bodyid_pairs = bodyid_pairs.rename({'nt_type_pre': 'nt_type'})
    
    # Also add std_label to ref_df for total_post calculation
    ref_df_with_labels = None
    if ref_df is not None and bodyid_label_map:
        # Add std_label to ref_df
        map_df = pl.DataFrame({
            'bodyId': list(bodyid_label_map.keys()),
            'std_label': list(bodyid_label_map.values())
        })
        ref_df_with_labels = ref_df.join(map_df, on='bodyId', how='left')
        ref_df_with_labels = ref_df_with_labels.with_columns(
            pl.coalesce([pl.col('std_label'), pl.col('type')]).alias('std_label')
        )
    elif ref_df is not None:
        ref_df_with_labels = ref_df.with_columns(pl.col('type').alias('std_label'))
    
    # Convert global_incoming_weights to Polars if provided
    global_incoming_pl = None
    if global_incoming_weights is not None:
        if isinstance(global_incoming_weights, pd.DataFrame):
            global_incoming_pl = pl.from_pandas(global_incoming_weights)
        else:
            global_incoming_pl = global_incoming_weights

        # Match the normalized nullable-string key above.  Without this cast,
        # an all-null `type_post` on the left is Polars' Null dtype and joining
        # it to a normal String key raises SchemaError.
        if 'type_post' in global_incoming_pl.columns:
            global_incoming_pl = global_incoming_pl.with_columns(
                pl.col('type_post').cast(pl.Utf8, strict=False).alias('type_post')
            )
        if 'total_incoming_weight' in global_incoming_pl.columns:
            global_incoming_pl = global_incoming_pl.with_columns(
                pl.col('total_incoming_weight')
                .cast(pl.Float64, strict=False)
                .alias('total_incoming_weight')
            )
    
    # Check if nt_type exists
    has_nt_type = 'nt_type' in bodyid_pairs.columns
    
    # Function to aggregate
    def aggregate_connections(group_pre_col, group_post_col, ref_group_col=None):
        # Sum weights from deduplicated bodyId pairs
        agg_list = [pl.col('weight').sum()]
        
        # Add nt_type aggregation if available (mode; ties break to the
        # lexicographically first value so both engines - and any row order -
        # produce identical results)
        if has_nt_type:
            agg_list.append(pl.col('nt_type').mode().sort().first().alias('nt_type'))
        
        agg_df = bodyid_pairs.group_by([group_pre_col, group_post_col]).agg(agg_list)
        
        # Type-level traversal_probability follows *aggregate_method* (same
        # semantics as the pandas engine, see the parameter docstring):
        # 'product' (default) compounds the per-pair block probabilities over
        # the deduplicated bodyId pairs, 'average' takes the weight-weighted
        # mean, 'ratio' uses min(connection_ratio / 0.3, 1).

        # Calculate Connection Ratio (Type Level)
        # Use GLOBAL incoming weights if provided, otherwise fall back to LOCAL calculation.
        # LOCAL totals are also used as a fallback for groups MISSING from the
        # global table - without it every group whose post type is absent (e.g.
        # untyped neurons grouped by bodyId, or std_labels that map to no known
        # raw type) would get a null ratio and a traversal_probability of 0.
        local_totals = agg_df.group_by(group_post_col).agg(
            pl.col('weight').sum().alias('_local_total')
        )
        
        if global_incoming_pl is not None and group_post_col in ['type_post', 'std_label_post']:
            # Use global incoming weights from the full dataset
            # global_incoming_weights has 'type_post' and 'total_incoming_weight' columns
            
            if group_post_col == 'std_label_post':
                # For std_label aggregation, we need to join bodyid_pairs with global weights
                # first, then aggregate. This handles the case where std_label differs from type.
                
                # Get unique std_label_post -> type_post mappings from bodyid_pairs
                # Since bodyid_pairs still has both type_post and std_label_post
                if 'type_post' in bodyid_pairs.columns:
                    # Build std_label -> sum of global incoming weights
                    # Each std_label_post may map to multiple type_post, so we sum their incoming weights
                    std_label_type_map = bodyid_pairs.select(['std_label_post', 'type_post']).unique()
                    
                    # Add global incoming weight for each type_post (vectorized
                    # join instead of row-wise map_elements with a Python dict)
                    # NOTE: type_post stays NULL for untyped post neurons (their
                    # std_label is the bodyId), so those rows never match the
                    # global type table and are handled by the local fallback.
                    std_label_type_map = std_label_type_map.join(
                        global_incoming_pl.select(['type_post', 'total_incoming_weight']),
                        on='type_post',
                        how='left',
                    ).with_columns(
                        pl.col('total_incoming_weight')
                        .alias('type_incoming')
                    )
                    
                    # Sum by std_label_post (in case one std_label maps to multiple types).
                    # If ANY of a group's types is missing from the global table the
                    # sum would undercount, so the whole group falls back to local.
                    global_incoming_by_std_label = std_label_type_map.group_by('std_label_post').agg(
                        pl.col('type_incoming').sum().alias('_global_sum'),
                        pl.col('type_incoming').is_null().any().alias('_any_missing'),
                    )
                    global_incoming_by_std_label = global_incoming_by_std_label.with_columns(
                        pl.when(~pl.col('_any_missing'))
                        .then(pl.col('_global_sum'))
                        .otherwise(None)
                        .alias('total_incoming_weight')
                    ).drop(['_global_sum', '_any_missing'])
                    
                    # Join with agg_df
                    agg_df = agg_df.join(global_incoming_by_std_label, on='std_label_post', how='left')
                else:
                    # Fallback: rename type_post to std_label_post (they should be the same)
                    global_incoming_renamed = global_incoming_pl.rename({'type_post': 'std_label_post'})
                    agg_df = agg_df.join(global_incoming_renamed, on='std_label_post', how='left')
            else:
                # Direct join for type_post grouping
                agg_df = agg_df.join(global_incoming_pl, on='type_post', how='left')
        else:
            agg_df = agg_df.with_columns(
                pl.lit(None, dtype=pl.Float64).alias('total_incoming_weight')
            )
        
        # LOCAL-total fallback for groups missing from the global table (see
        # the comment above the global branch).
        agg_df = agg_df.join(local_totals, on=group_post_col, how='left')
        agg_df = agg_df.with_columns(
            pl.coalesce([pl.col('total_incoming_weight'), pl.col('_local_total')])
            .cast(pl.Float64, strict=False)
            .alias('total_incoming_weight')
        ).drop('_local_total')
        
        # Calculate ratio using the (global if available, else local) denominator
        agg_df = agg_df.with_columns(
            pl.when(pl.col('total_incoming_weight') > 0)
            .then(pl.col('weight') / pl.col('total_incoming_weight'))
            .otherwise(None)
            .alias('connection_ratio')
        )
        agg_df = agg_df.drop('total_incoming_weight')
        
        # Type-level traversal_probability from the aggregate method (see the
        # parameter docstring). 'product' (default) compounds the per-pair
        # block probabilities over the deduplicated bodyId pairs
        # (reliability/OR model); 'average' takes the weight-weighted mean;
        # 'ratio' uses min(connection_ratio / 0.3, 1) (input-share model).
        # fill_null(0.0) matches the pandas fillna(0.0) semantics for groups
        # with no computable probability; nulls_equal=True keeps parity with
        # pandas merge for null group labels (e.g. unassigned custom groups).
        if aggregate_method == 'product':
            pair_probs = (
                bodyid_pairs.group_by([group_pre_col, group_post_col])
                .agg(
                    (1.0 - pl.col('traversal_probability'))
                    .product()
                    .alias('_block_prod')
                )
                .with_columns(
                    (1.0 - pl.col('_block_prod'))
                    .fill_null(0.0)
                    .alias('traversal_probability')
                )
                .drop('_block_prod')
            )
            agg_df = agg_df.join(
                pair_probs, on=[group_pre_col, group_post_col], how='left', nulls_equal=True
            )
        elif aggregate_method == 'average':
            pair_probs = (
                bodyid_pairs.group_by([group_pre_col, group_post_col])
                .agg(
                    (pl.col('weight') * pl.col('traversal_probability'))
                    .sum()
                    .alias('_wt'),
                    pl.col('weight').sum().alias('_wsum'),
                )
                .with_columns(
                    (pl.col('_wt') / pl.col('_wsum'))
                    .fill_null(0.0)
                    .alias('traversal_probability')
                )
                .drop(['_wt', '_wsum'])
            )
            agg_df = agg_df.join(
                pair_probs, on=[group_pre_col, group_post_col], how='left', nulls_equal=True
            )
        else:
            # 'ratio' model (input-share). fill_null(0.0) matches the pandas
            # fillna(0.0) semantics for types with no known incoming
            # connections.
            agg_df = agg_df.with_columns(
                (pl.col('connection_ratio') / 0.3).clip(0.0, 1.0).fill_null(0.0).alias('traversal_probability')
            )
        
        # block_probability = 1 - traversal_probability (null -> 1.0, matching
        # the pandas fillna(1.0) semantics so both engines emit the same schema)
        agg_df = agg_df.with_columns(
            (1.0 - pl.col('traversal_probability')).fill_null(1.0).alias('block_probability')
        )
        
        return agg_df

    # Aggregate by std_label (Step 4 & 5)
    # This uses std_label for mapped neurons and type for unmapped (since std_label = type for unmapped)
    conn_type = aggregate_connections('std_label_pre', 'std_label_post', ref_group_col='std_label')
    
    # Rename std_label columns to type columns for backward compatibility
    conn_type = conn_type.rename({
        'std_label_pre': 'type_pre',
        'std_label_post': 'type_post'
    })
    
    # Aggregate Group
    conn_group = None
    if 'custom_group_pre' in conn_df.columns:
        conn_group = aggregate_connections('custom_group_pre', 'custom_group_post', ref_group_col='custom_group')
        
    # Return Polars DataFrames directly
    return conn_df, conn_type, conn_group


def build_path_dataframe_from_paths_polars(paths, conn_data, targets, real_layer_map=None, level='type', type_lookup=None):
    """
    Build a Polars DataFrame from a list of paths.
    """
    # Prepare connection data
    df_conn = prepare_connection_data(conn_data, level)
    
    # Process all paths
    # We reuse process_batch_polars logic
    # Note: type_lookup is currently ignored in Polars implementation for path string formatting
    # to maintain high performance. Path strings will contain IDs only.
    
    df_final, _ = process_batch_polars(paths, df_conn, level)
    
    return df_final
