"""
NeuronBridge Finder Module

This module provides a wrapper around the NeuronBridge Python API for finding
correspondences between EM reconstructed neurons (bodyIds) and light microscopy 
(LM) GAL4 driver lines.

Key features:
- Convert neuron bodyIds to matching driver lines (id_to_lines)
- Convert neuron queries (bodyId/instance/type) to lines (neuron_to_lines)
- Reverse search: find neurons matching a driver line (line_to_neuron)
- Support for both CDS and PPPM matching algorithms
- Multi-dataset support (hemibrain, male-cns, FAFB, standalone BANC, etc.)
- Automatic dataset detection from NeuronBridge image metadata
- Caching of API results to reduce redundant calls
- Versioned, compressed Parquet caching

Dependencies:
- requests (via the bundled lightweight NeuronBridge client)
- pandas

Example usage:
    from neuronbridge_finder import NeuronBridgeFinder
    
    nbf = NeuronBridgeFinder()
    
    # Find lines matching a bodyId
    lines = nbf.id_to_lines(636798093, top_n=10)
    
    # Find lines matching a neuron type
    results = nbf.neuron_to_lines('MBON01', top_n=5)
    
    # Find neurons matching a driver line (returns neurons from all datasets)
    neurons = nbf.line_to_neuron('LH173', top_n=10)

Author: Generated for DROCAT project
"""

import json
import os
import re
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

# Make the project root importable regardless of how this module was loaded
# (scripts put only src/ on sys.path; the `from src...` fallbacks below
# silently failed from a non-repo cwd).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import bokeh.palettes
import numpy as np
import pandas as pd

try:
    from .utils.naming_utils import canonical_dataset_name
except ImportError:  # pragma: no cover - src laid bare on sys.path
    from utils.naming_utils import canonical_dataset_name
try:
    from .flywire_ids import (
        is_local_connectome_dataset,
        normalize_flywire_body_id,
        normalize_flywire_id_columns,
        resolve_flywire_dataset_dir,
    )
except ImportError:
    from flywire_ids import (
        is_local_connectome_dataset,
        normalize_flywire_body_id,
        normalize_flywire_id_columns,
        resolve_flywire_dataset_dir,
    )
try:
    from .utils.naming_utils import dataset_abbrev
except ImportError:
    try:
        from utils.naming_utils import dataset_abbrev
    except ImportError:
        # Last-resort fallback with the same mapping as utils.naming_utils,
        # so run-folder names stay meaningful even if that module is missing.
        _DATASET_ABBREVIATIONS_FALLBACK = {
            "male-cns": "MCNS", "male_cns": "MCNS", "hemibrain": "HEMI",
            "optic-lobe": "OL", "optic_lobe": "OL", "manc": "MANC",
            "banc": "BANC", "fib19": "FIB", "mushroombody": "MB",
            "flywire_fafb": "FAFB", "fafb": "FAFB", "flywire_banc": "BANC",
        }

        def dataset_abbrev(dataset):
            if not dataset:
                return "UNKN"
            ds = str(dataset).lower()
            for key, abbrev in _DATASET_ABBREVIATIONS_FALLBACK.items():
                if key in ds:
                    return abbrev
            letters = "".join(c for c in ds.split(":")[0] if c.isalpha())
            return (letters[:4] or "DS").upper()

try:
    from .visualization_options import default_analysis_skeleton_mesh_simplification
except ImportError:
    from visualization_options import default_analysis_skeleton_mesh_simplification

# Try to import polars for faster DataFrame operations on large datasets
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None

if TYPE_CHECKING:
    from comparison.label_mapper import LabelMapper

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

# Try to import LabelMapper for cross-dataset queries
try:
    from comparison.label_mapper import LabelMapper
    HAS_LABELMAPPER = True
except ImportError:
    try:
        from src.comparison.label_mapper import LabelMapper
        HAS_LABELMAPPER = True
    except ImportError:
        HAS_LABELMAPPER = False
        LabelMapper = None  # type: ignore

# Try to import CrossDatasetTypeMapper for type name merging
try:
    from comparison.cross_dataset_type_mapper import CrossDatasetTypeMapper, get_type_mapper
    HAS_TYPE_MAPPER = True
except ImportError:
    try:
        from src.comparison.cross_dataset_type_mapper import CrossDatasetTypeMapper, get_type_mapper
        HAS_TYPE_MAPPER = True
    except ImportError:
        HAS_TYPE_MAPPER = False
        CrossDatasetTypeMapper = None  # type: ignore
        get_type_mapper = None  # type: ignore

# Output-detail policy (Compact prune passes for the three batch methods)
try:
    from neuronbridge_output_policy import (
        prune_colabel_run,
        prune_find_lines_run,
        prune_find_neurons_run,
    )
except ImportError:  # pragma: no cover - ``src/`` on sys.path imports
    try:
        from src.neuronbridge_output_policy import (
            prune_colabel_run,
            prune_find_lines_run,
            prune_find_neurons_run,
        )
    except ImportError:
        prune_find_lines_run = None  # type: ignore
        prune_find_neurons_run = None  # type: ignore
        prune_colabel_run = None  # type: ignore

# Try to import img2pptx for PPTX generation
try:
    from utils.report_utils import img2pptx
    HAS_IMG2PPTX = True
except ImportError:
    try:
        from src.utils.report_utils import img2pptx
        HAS_IMG2PPTX = True
    except ImportError:
        HAS_IMG2PPTX = False
        img2pptx = None  # type: ignore

# DROCAT only needs the public, read-only metadata subset of NeuronBridge.
# The bundled client avoids the upstream package's unrelated Ray/Memray and
# Pydantic 2.9 constraints, which cannot coexist with the NiceGUI runtime.
try:
    from .neuronbridge_client import Client as NBClient
except ImportError:
    from neuronbridge_client import Client as NBClient

try:
    from .neuronbridge_cache import NeuronBridgeParquetCache
except ImportError:
    from neuronbridge_cache import NeuronBridgeParquetCache

NEURONBRIDGE_AVAILABLE = True

# Mapping from NeuronBridge library names to our local dataset folder names
LIBRARY_TO_DATASET = {
    'FlyEM_Hemibrain_v1.2.1': 'hemibrain_v1_2_1',
    'FlyEM_Hemibrain_v1.2': 'hemibrain_v1_2_1',
    'FlyEM_Hemibrain': 'hemibrain_v1_2_1',
    'FlyEM_MANC_v1.2.1': 'manc_v1_2_1',
    'FlyEM_MANC_v1.0': 'manc_v1_0',
    'FlyEM_MANC': 'manc_v1_0',
    'FlyEM_Male_CNS_Brain_v0.9': 'male-cns_v0_9',
    'FlyEM_Male_CNS_VNC_v0.9': 'male-cns_v0_9',
    'FlyEM_Male_CNS_v0.9': 'male-cns_v0_9',
    'FlyEM_Male_CNS': 'male-cns_v0_9',
    'FlyEM_Male_CNS_Brain_v1.0': 'male-cns_v1_0',
    'FlyEM_Male_CNS_VNC_v1.0': 'male-cns_v1_0',
    'FlyEM_Male_CNS_v1.0': 'male-cns_v1_0',
    'FlyWire_FAFB': 'flywire_FAFB_v783',
    'FlyWire_FAFB_v783': 'flywire_FAFB_v783',
    'FlyWire_FAFB_v783_realign': 'flywire_FAFB_v783',
    'FlyWire_BANC': 'banc_v626',
    'FlyWire_BANC_v626': 'banc_v626',
    'FlyWire_BANC_v888': 'banc_v888',
    'FlyEM_Optic_Lobe': 'optic-lobe_v1_1',
}

# Mapping to human-readable dataset names (for display/neuprint format)
LIBRARY_TO_DATASET_NAME = {
    'FlyEM_Hemibrain_v1.2.1': 'hemibrain:v1.2.1',
    'FlyEM_Hemibrain_v1.2': 'hemibrain:v1.2.1',
    'FlyEM_Hemibrain': 'hemibrain:v1.2.1',
    'FlyEM_MANC_v1.2.1': 'manc:v1.2.1',
    'FlyEM_MANC_v1.0': 'manc:v1.0',
    'FlyEM_MANC': 'manc:v1.0',
    'FlyEM_Male_CNS_Brain_v0.9': 'male-cns:v0.9',
    'FlyEM_Male_CNS_VNC_v0.9': 'male-cns:v0.9',
    'FlyEM_Male_CNS_v0.9': 'male-cns:v0.9',
    'FlyEM_Male_CNS': 'male-cns:v0.9',
    'FlyEM_Male_CNS_Brain_v1.0': 'male-cns:v1.0',
    'FlyEM_Male_CNS_VNC_v1.0': 'male-cns:v1.0',
    'FlyEM_Male_CNS_v1.0': 'male-cns:v1.0',
    'FlyWire_FAFB': 'flywire_FAFB_v783',
    'FlyWire_FAFB_v783': 'flywire_FAFB_v783',
    'FlyWire_FAFB_v783_realign': 'flywire_FAFB_v783',
    'FlyWire_BANC': 'banc_v626',
    'FlyWire_BANC_v626': 'banc_v626',
    'FlyWire_BANC_v888': 'banc_v888',
    'FlyEM_Optic_Lobe': 'optic-lobe:v1.1',
}

# Dataset abbreviations for expression matrix type naming
# Format: {dataset_folder_name: abbreviation}
DATASET_ABBREVIATIONS = {
    'hemibrain_v1_2_1': 'HEMI',
    'hemibrain:v1.2.1': 'HEMI',
    'male-cns_v0_9': 'MCNS',
    'male-cns:v0.9': 'MCNS',
    'male-cns_v1_0': 'MCNS',
    'male-cns:v1.0': 'MCNS',
    'manc_v1_0': 'MANC',
    'manc:v1.0': 'MANC',
    'manc_v1_2_1': 'MANC',
    'manc:v1.2.1': 'MANC',
    'flywire_FAFB_v783': 'FAFB',
    'banc_v626': 'BANC',
    'banc_v888': 'BANC',
    'optic-lobe_v1_1': 'OLOB',
    'optic-lobe:v1.1': 'OLOB',
}


def _to_int_bodyid(val):
    """
    Convert a bodyId value to integer, handling various input formats.
    
    Handles:
    - Integer values (passthrough)
    - String values like '5813128953' 
    - Float values like 5813128953.0
    - String floats like '5813128953.0'
    
    Returns the original value if conversion fails.
    """
    try:
        # The normalizer rejects unsafe FlyWire floats instead of silently
        # rounding them through ``int(float(value))``.
        return int(normalize_flywire_body_id(val, field='bodyId'))
    except (TypeError, ValueError):
        return val


# Line name prefixes for classification
# GAL4/LexA lines: typically VT (Vienna Tile), R (Rubin), GMR, etc.
GAL4_LEXA_PREFIXES = ('VT', 'R', 'GMR')
# Split-GAL4 lines: SS (Split Screen), LH (Lateral Horn), MB (Mushroom Body), etc.
SPLIT_GAL4_PREFIXES = ('SS', 'LH', 'MB', 'IS', 'OL', 'LC', 'LLPC', 'LPC', 'JRC_SS', 'BJD_SS')

# Valid input values for validation (case-insensitive)
VALID_MATCH_TYPES = {'cds', 'pppm', 'both'}
VALID_REGIONS = {'brain', 'vnc', 'all'}
VALID_SIMILARITY_METHODS = {'jaccard', 'weighted_jaccard', 'rank_correlation'}
_EM_IMAGE_NOT_PROVIDED = object()
VALID_SORT_BY = {'completeness', 'max'}


def _extract_base_type(prefixed_type: str, merge_mapping=None) -> str:
    """Extract the base type name from a dataset-prefixed type.

    Shared by the expression-matrix visualizations: if a merge mapping is
    available (auto type mapping), a prefixed type that maps to a merged
    type is resolved through it first; otherwise everything after the first
    underscore is returned ('MCNS_aMe12' -> 'aMe12').
    """
    if merge_mapping and prefixed_type in merge_mapping:
        return merge_mapping[prefixed_type]
    # KNOWN LIMITATION (mapper-absent fallback): the naive first-underscore
    # split mangles type names that CONTAIN underscores (e.g.
    # '5thsLNv_LNd6' -> 'LNd6').  Only reached when no auto-mapper merge
    # mapping is available; the mapper-driven path above is authoritative.
    parts = prefixed_type.split('_', 1)
    if len(parts) > 1:
        return parts[1]  # Return everything after first underscore
    return prefixed_type  # Return as-is if no underscore


def _create_mountain_order(scores, labels):
    """Rearrange so the highest score is in the middle, descending to both sides.

    Shared by the labeling/colabeling distribution visualizations.
    """
    sorted_pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    n = len(sorted_pairs)
    if n == 0:
        return [], []

    result_scores = [0] * n
    result_labels = [''] * n

    # Write pointers start NEXT TO the center slot; starting them at the
    # center itself made the first alternating write overwrite the peak.
    left = n // 2 - 1
    right = n // 2 + 1

    for i, (score, label) in enumerate(sorted_pairs):
        if i == 0:
            center = n // 2
            result_scores[center] = score
            result_labels[center] = label
        elif i % 2 == 1:
            result_scores[left] = score
            result_labels[left] = label
            left -= 1
        else:
            result_scores[right] = score
            result_labels[right] = label
            right += 1

    return result_scores, result_labels


def image_summary_skip_note(summary_format, download_images) -> Optional[str]:
    """Explain why a requested PDF/PPTX summary will not be generated.

    The image summary contact sheets are built FROM downloaded line images;
    when image download is disabled a requested summary format is a silent
    no-op. Returns a one-line user-facing note in that case, else ``None``.
    """
    formats = summary_format if isinstance(summary_format, list) else [summary_format]
    formats = [str(f).lower() for f in formats if f]
    if formats and not download_images:
        return (
            f"⚠️  Note: summary format {formats} requested but image download "
            "is disabled. The PDF/PPTX summary is built from downloaded images "
            "and will be skipped — enable image download to generate it."
        )
    return None


# Result layer renders at most this many neurons per type (mirrors
# morphology.TYPE_RENDER_MEMBER_CAP); large types are capped with a warning.
TYPE_RENDER_MEMBER_CAP = 20


@dataclass
class NeuronBridgeFinder:
    """
    A class for finding correspondences between EM neurons and LM driver lines.
    
    This class wraps the NeuronBridge API to provide convenient methods for
    mapping between EM body IDs and GAL4 driver lines across multiple datasets.
    
    Parameters
    ----------
    datasets_path : str, optional
        Path to the datasets folder containing neuron_df CSV files.
        Default: auto-detect from module location.
    use_cache : bool
        Whether to cache API results locally. Default: False — NeuronBridge
        queries are large and rarely reused (unlike the connection-data
        caches), so the match cache would grow without ever paying off.
        The Find Lines / Find EM Neurons / Co-Labeling tabs expose this
        default as the Settings → NeuronBridge Match Cache option.
    cache_folder : str, optional
        Folder for cached results. Default: auto-detect.
    verbose : bool
        Print progress messages. Default: True
    separate_splitgal4 : bool
        If True, separate results into GAL4/LexA and Split-GAL4 categories.
        When enabled, download_img_for_top_n_lines applies separately to each category.
        Default: False
    neuprint_token : str, optional
        NeuPrint API token for pulling missing datasets. If not provided, will check
        NEUPRINT_TOKEN or NEUPRINT_APPLICATION_CREDENTIALS environment variables.
        Without a token, local dataset features (type lookups, specificity) will be skipped.
        Get your token at: https://neuprint.janelia.org/account
    neuprint_server : str
        NeuPrint server URL. Default: 'https://neuprint.janelia.org'
    match_type : str
        Default match algorithm: 'cds', 'pppm', or 'both' (case-insensitive). Default: 'cds'
    region : str
        Filter images by anatomical region: 'Brain', 'VNC', or 'All' (case-insensitive). Default: 'All'
    max_api_images_per_line : int
        Maximum LM images to process per driver line for API calls. Use -1 for unlimited. Default: -1
        Images are pre-filtered by match_type availability before limiting.
    max_workers : int
        Maximum number of parallel threads for processing images. Set to 1 for sequential
        processing. Higher values speed up API calls but may hit rate limits. Default: 8
    
    Attributes
    ----------
    _client : NBClient
        The NeuronBridge API client.
    _neuron_dfs : dict
        Dictionary mapping dataset names to loaded neuron DataFrames.
    
    Example
    -------
    >>> nbf = NeuronBridgeFinder()
    >>> lines = nbf.id_to_lines(636798093, top_n=5)
    >>> print(lines)
    
    # Separate GAL4/LexA from Split-GAL4 lines
    >>> nbf = NeuronBridgeFinder(separate_splitgal4=True)
    >>> results = nbf.find_lines_batch('MBON01', download_img_for_top_n_lines=5)  # 5 GAL4 + 5 Split-GAL4
    
    # With NeuPrint token for pulling missing datasets
    >>> nbf = NeuronBridgeFinder(neuprint_token='your_token_here')
    
    # With parallel processing for faster image searches
    >>> nbf = NeuronBridgeFinder(max_workers=12)  # Use 12 parallel threads
    """
    
    datasets_path: Optional[str] = None
    use_cache: bool = False
    cache_folder: Optional[str] = None
    verbose: bool = True
    separate_splitgal4: bool = False
    neuprint_token: Optional[str] = None
    neuprint_server: str = 'https://neuprint.janelia.org'
    match_type: str = 'cds'
    region: str = 'All'
    max_api_images_per_line: int = -1
    max_workers: int = 8
    
    # Private fields
    _client: Any = field(init=False, repr=False, default=None)
    _neuron_dfs: Dict[str, pd.DataFrame] = field(init=False, repr=False, default_factory=dict)
    _suppress_loading_msgs: bool = field(init=False, repr=False, default=False)
    _batch_mode: bool = field(init=False, repr=False, default=False)
    _warning_collector: List[str] = field(init=False, repr=False, default_factory=list)
    _cache_lock: threading.Lock = field(init=False, repr=False, default_factory=threading.Lock)
    _cache_index: Dict[str, set] = field(init=False, repr=False, default_factory=dict)  # {cache_type: set of cache_keys}
    _parquet_cache: Any = field(init=False, repr=False, default=None)
    
    def __post_init__(self):
        """Initialize the finder after dataclass initialization."""
        # Validate and normalize match_type (case-insensitive)
        if isinstance(self.match_type, str):
            normalized_match_type = self.match_type.lower().strip()
            if normalized_match_type not in VALID_MATCH_TYPES:
                raise ValueError(
                    f"Invalid match_type: '{self.match_type}'. "
                    f"Must be one of: {', '.join(sorted(VALID_MATCH_TYPES))}"
                )
            # Use object.__setattr__ for frozen dataclass compatibility
            object.__setattr__(self, 'match_type', normalized_match_type)
        
        # Validate and normalize region (case-insensitive)
        if isinstance(self.region, str):
            normalized_region = self.region.lower().strip()
            if normalized_region not in VALID_REGIONS:
                raise ValueError(
                    f"Invalid region: '{self.region}'. "
                    f"Must be one of: {', '.join(sorted(VALID_REGIONS))}"
                )
            # Store in title case for display consistency (Brain, VNC, All)
            region_display = {'brain': 'Brain', 'vnc': 'VNC', 'all': 'All'}
            object.__setattr__(self, 'region', region_display[normalized_region])
        
        # Set default datasets path
        if self.datasets_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(module_dir)
            self.datasets_path = os.path.join(project_root, 'datasets')
        
        # Set default cache folder
        if self.cache_folder is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(module_dir)
            self.cache_folder = os.path.join(project_root, 'cache', 'neuronbridge')
        
        # Create cache folder if needed
        if self.use_cache:
            os.makedirs(self.cache_folder, exist_ok=True)
        
        # Initialize client
        self._init_client()

        if self.use_cache:
            self._get_parquet_cache().ensure_manifest(
                getattr(self._client, 'version', None)
            )
            # Build cache index for fast lookups in the current Parquet store.
            self._build_cache_index()
    
    def _vprint(self, msg: str, end: str = '\n', force: bool = False):
        """
        Print message if verbose mode is enabled.
        
        Uses tqdm.write() if tqdm is available and progress bars might be active,
        otherwise falls back to regular print().
        
        Parameters
        ----------
        msg : str
            Message to print
        end : str
            String appended after the message (default: newline)
        force : bool
            If True, print even in batch mode (default: False)
        """
        if self.verbose and (not self._batch_mode or force):
            # Use tqdm.write if available to avoid conflicts with progress bars
            if HAS_TQDM:
                try:
                    from tqdm import tqdm
                    # Progress bars in this module use stdout so the UI
                    # runner receives messages and bar redraws in order.
                    tqdm.write(msg, end=end, file=sys.stdout)
                except:
                    print(msg, end=end, flush=True)
            else:
                print(msg, end=end, flush=True)

    def _progress(self, step: int, total: int, label: str = ""):
        """Emit a structured step-progress event consumed by the web UI.

        The line ``[DROCAT][progress] <step>/<total> <label>`` drives the
        determinate progress bar + step label in the results panel; it is a
        control event, never shown in the execution log.  Uses ``tqdm.write``
        like :meth:`_vprint` so it never interleaves with an active bar.
        """
        if self.verbose:
            msg = f"[DROCAT][progress] {int(step)}/{int(total)} {label}".rstrip()
            if HAS_TQDM:
                try:
                    from tqdm import tqdm
                    tqdm.write(msg, file=sys.stdout)
                    return
                except Exception:
                    pass
            print(msg, flush=True)

    def _get_parquet_cache(self) -> NeuronBridgeParquetCache:
        """Return the lazy Parquet cache store for this Finder instance."""
        cache = getattr(self, '_parquet_cache', None)
        if cache is None:
            cache = NeuronBridgeParquetCache(
                self.cache_folder,
                version=getattr(getattr(self, '_client', None), 'version', None),
            )
            self._parquet_cache = cache
        return cache
    
    def _build_cache_index(self):
        """
        Build an in-memory index of all cached files for fast O(1) lookups.
        
        This scans the current NeuronBridge-versioned Parquet folder once at
        initialization and builds a set of id-to-lines cache keys.
        
        The index maps cache_type -> set of full cache keys (filename without extension).
        """
        self._cache_index = {}
        
        if not self.use_cache or not self.cache_folder:
            return
        
        if not os.path.exists(self.cache_folder):
            return
        
        try:
            cache_files = []
            current_parquet_root = self._get_parquet_cache().root
            if current_parquet_root.exists():
                cache_files = [
                    path for path in current_parquet_root.rglob('*.parquet')
                    if path.is_file() and path.parent.name == 'id_to_lines'
                ]
            
            # Parse each filename and build index
            # Format: id_to_lines_{identifier}.parquet
            for cache_path in cache_files:
                filename = cache_path.name
                cache_key = cache_path.stem
                
                if filename.startswith('id_to_lines_'):
                    self._cache_index.setdefault('id_to_lines', set()).add(cache_key)
            
            # Report stats
            total_cached = sum(len(keys) for keys in self._cache_index.values())
            if total_cached > 0 and self.verbose:
                self._vprint(f"  📁 Cache index: {total_cached} cached files "
                           f"({len(self._cache_index.get('id_to_lines', set()))} id tables)")
        except Exception as e:
            # If indexing fails, continue without index (will fallback to file checks)
            self._cache_index = {}
            if self.verbose:
                self._vprint(f"  ⚠️ Could not build cache index: {e}")
    
    def _is_cached(self, cache_type: str, identifier: str) -> bool:
        """
        Check if a cache entry exists using the in-memory index.
        
        This is O(1) instead of checking file existence on disk.
        
        Parameters
        ----------
        cache_type : str
        Type of cache (currently only 'id_to_lines')
        identifier : str
            The identifier used to build the cache key
            
        Returns
        -------
        bool
            True if the entry is in the cache index
        """
        # Cache-off mode (the default since the output-modes change): the
        # partition must treat everything as uncached, otherwise bodies are
        # routed to the bulk cache loader and silently vanish.
        if not self.use_cache:
            return False
        # Direct checks cover newly-created entries even if another worker
        # saved them after the startup index was built.
        if os.path.exists(self._get_cache_path(cache_type, identifier)):
            return True
        if not self._cache_index:
            return False
        
        safe_id = str(identifier).replace('/', '_').replace('\\', '_').replace(':', '_')
        cache_key = f"{cache_type}_{safe_id}"
        
        return cache_key in self._cache_index.get(cache_type, set())
    
    def _add_to_cache_index(self, cache_type: str, identifier: str):
        """Add a new entry to the cache index after saving to disk."""
        safe_id = str(identifier).replace('/', '_').replace('\\', '_').replace(':', '_')
        cache_key = f"{cache_type}_{safe_id}"
        
        if cache_type not in self._cache_index:
            self._cache_index[cache_type] = set()
        self._cache_index[cache_type].add(cache_key)
    
    def _validate_match_type(self, match_type: str) -> str:
        """
        Validate and normalize match_type parameter.
        
        Parameters
        ----------
        match_type : str
            Match type to validate ('cds', 'pppm', or 'both')
            
        Returns
        -------
        str
            Normalized (lowercase) match_type
            
        Raises
        ------
        ValueError
            If match_type is not valid
        """
        normalized = match_type.lower().strip()
        if normalized not in VALID_MATCH_TYPES:
            raise ValueError(
                f"Invalid match_type: '{match_type}'. "
                f"Must be one of: {', '.join(sorted(VALID_MATCH_TYPES))}"
            )
        return normalized
    
    def _validate_similarity_method(self, method: str) -> str:
        """
        Validate and normalize similarity_method parameter.
        
        Parameters
        ----------
        method : str
            Similarity method to validate
            
        Returns
        -------
        str
            Normalized (lowercase) similarity method
            
        Raises
        ------
        ValueError
            If method is not valid
        """
        normalized = method.lower().strip()
        if normalized not in VALID_SIMILARITY_METHODS:
            raise ValueError(
                f"Invalid similarity_method: '{method}'. "
                f"Must be one of: {', '.join(sorted(VALID_SIMILARITY_METHODS))}"
            )
        return normalized
    
    def _validate_sort_by(self, sort_by: str) -> str:
        """
        Validate and normalize sort_by parameter.
        
        Parameters
        ----------
        sort_by : str
            Sort method: 'completeness' or 'max'
            
        Returns
        -------
        str
            Normalized (lowercase) sort_by value
            
        Raises
        ------
        ValueError
            If sort_by is not valid
        """
        normalized = sort_by.lower().strip()
        if normalized not in VALID_SORT_BY:
            raise ValueError(
                f"Invalid sort_by: '{sort_by}'. "
                f"Must be one of: {', '.join(sorted(VALID_SORT_BY))}"
            )
        return normalized
    
    def _print_warning_summary(self):
        """Print collected warnings as a summary."""
        if not self._warning_collector:
            return
        
        
        # Group warnings by type
        server_unreachable = []
        no_gal4_files = []
        no_files_at_all = []
        
        for warning in self._warning_collector:
            if "server not accessible" in warning:
                server_unreachable.append(warning)
            elif "No files from" in warning:
                no_gal4_files.append(warning)
            elif "No images found" in warning:
                no_files_at_all.append(warning)
        
        # Print summary
        if server_unreachable:
            # Extract unique servers and line counts
            server_issues = {}
            for w in server_unreachable:
                if "flimg.janelia.org" in w:
                    line = w.split("for ")[-1].split(":")[0].strip()
                    server_issues.setdefault("flimg.janelia.org (VT GAL4)", []).append(line)
            
            if server_issues:
                self._vprint("\n⚠️  Server Access Issues:", force=True)
                for server, lines_affected in server_issues.items():
                    self._vprint(f"   • {server}: {len(lines_affected)} line(s) unreachable", force=True)
                    self._vprint(f"     (Attempted MCFO fallback from S3)", force=True)
        
        if no_gal4_files:
            lines = [w.split("for ")[-1].split(",")[0].strip() for w in no_gal4_files]
            self._vprint(f"\n📋 {len(lines)} line(s) used MCFO fallback (no GAL4/SplitGAL4 images)", force=True)
        
        if no_files_at_all:
            lines = [w.split("for ")[-1].split(" in")[0].strip() for w in no_files_at_all]
            if lines:
                self._vprint(f"\n❌ {len(lines)} line(s) had no FlyLight images after all collection fallbacks:", force=True)
                self._vprint(f"   {', '.join(lines[:10])}{'...' if len(lines) > 10 else ''}", force=True)
        
        # Clear warnings after printing
        self._warning_collector.clear()
    
    def _retry_with_backoff(
        self,
        func,
        *args,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        **kwargs
    ):
        """
        Retry a function with exponential backoff for transient network errors.
        
        Parameters
        ----------
        func : callable
            Function to retry
        max_retries : int
            Maximum number of retry attempts (default: 3)
        initial_delay : float
            Initial delay in seconds (default: 1.0)
        *args, **kwargs
            Arguments to pass to func
            
        Returns
        -------
        Any
            Result from func, or None if all retries failed
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if it's a retryable network error
                is_retryable = any([
                    'incompleteread' in error_str,
                    'ssl' in error_str and 'eof' in error_str,
                    'connection' in error_str and 'broken' in error_str,
                    'max retries exceeded' in error_str,
                    'connection reset' in error_str,
                    'timeout' in error_str
                ])
                
                if not is_retryable or attempt >= max_retries - 1:
                    # Not retryable or last attempt - give up
                    raise
                
                # Wait with exponential backoff
                delay = initial_delay * (2 ** attempt)
                time.sleep(delay)
        
        # Should never reach here, but just in case
        if last_error:
            raise last_error
    
    def _normalize_dataset_name(self, dataset: str) -> str:
        """
        Normalize dataset name for comparison.
        
        Handles different naming conventions:
        - 'flywire_FAFB_v783' -> 'flywire-fafb'
        - 'flywire_fafb:v783' -> 'flywire-fafb'
        - 'male-cns:v0.9' -> 'male-cns'
        - 'hemibrain:v1.2.1' -> 'hemibrain'
        - 'hemibrain_v1_2_1' -> 'hemibrain'
        
        Parameters
        ----------
        dataset : str
            Dataset name in any format
            
        Returns
        -------
        str
            Normalized dataset name (lowercase, no version, hyphens for separators)
        """
        import re
        
        # Convert to lowercase
        ds = dataset.lower()
        
        # Remove version info in various formats:
        # - ':v1.2.1' or ':v783' (colon format)
        # - '_v1_2_1' or '_v783' (underscore format)
        # First handle colon-separated version
        ds = ds.split(':')[0]
        
        # Remove underscore-separated version suffix (e.g., '_v783', '_v1_2_1')
        # Match _v followed by digits and optional underscored sub-versions
        ds = re.sub(r'_v\d+(_\d+)*$', '', ds)
        
        # Normalize separators: convert underscores to hyphens
        ds = ds.replace('_', '-')
        
        return ds
    
    def _save_parameters(
        self,
        output_path: str,
        function_name: str,
        function_params: Dict[str, Any],
        filename: str = 'parameters.json'
    ) -> str:
        """
        Save all module-level and function-level parameters to a JSON file.
        
        This ensures reproducibility by recording all configuration used
        to generate the analysis results.
        
        Parameters
        ----------
        output_path : str
            Directory to save the parameters file.
        function_name : str
            Name of the function being called (e.g., 'analyze_colabeling').
        function_params : dict
            Dictionary of function-level parameters.
        filename : str
            Output filename. Default: 'parameters.json'
            
        Returns
        -------
        str
            Path to the saved parameters file.
        """
        import json
        from datetime import datetime
        
        # Collect module-level (instance) parameters
        module_params = {
            'datasets_path': self.datasets_path,
            'use_cache': self.use_cache,
            'cache_folder': self.cache_folder,
            'verbose': self.verbose if isinstance(self.verbose, bool) else str(self.verbose),
            'separate_splitgal4': self.separate_splitgal4,
            'neuprint_server': self.neuprint_server,
            'match_type': self.match_type,
            'region': self.region,
            'max_api_images_per_line': self.max_api_images_per_line,
            # Note: neuprint_token is sensitive, store presence only
            'has_neuprint_token': self.neuprint_token is not None
        }
        
        # Process function params to make them JSON serializable
        serializable_params = {}
        for key, value in function_params.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                serializable_params[key] = value
            elif isinstance(value, (list, tuple)):
                # Handle lists/tuples - convert to list
                serializable_params[key] = list(value)
            elif isinstance(value, dict):
                serializable_params[key] = value
            elif hasattr(value, 'tolist'):
                # Handle numpy arrays
                serializable_params[key] = value.tolist()
            elif hasattr(value, '__dict__'):
                # Handle objects with __dict__
                serializable_params[key] = str(value)
            else:
                serializable_params[key] = str(value)
        
        # Build complete parameters dict
        params = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'function': function_name,
                'neuronbridge_finder_version': '3.1'
            },
            'module_params': module_params,
            'function_params': serializable_params
        }
        
        # Save to JSON file
        params_path = os.path.join(output_path, filename)
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        
        return params_path

    def _save_user_warning_notes(
        self,
        output_path: str,
        notes: List[str],
        filename: str = 'user_warning_notes.txt',
    ) -> Optional[str]:
        """Save output-affecting caveats for a completed NeuronBridge run.

        The notes are intentionally kept separate from ``parameters.json`` so
        they are visible alongside the generated result files and can explain
        when a user-facing cutoff changes an aggregate view without changing
        the retained top-N match records.
        """
        if not notes:
            return None

        notes_path = os.path.join(output_path, filename)
        with open(notes_path, 'w', encoding='utf-8') as f:
            f.write('User warning notes\n')
            f.write('==================\n\n')
            for note in notes:
                f.write(f'- {note}\n')
        return notes_path
    
    def _filter_images_by_region(self, images: List[Any]) -> List[Any]:
        """
        Filter images based on the anatomical region setting.
        
        Filters LM or EM images based on their anatomicalArea attribute.
        Only processes filtering when self.region is 'Brain' or 'VNC'.
        
        Parameters
        ----------
        images : list
            List of LM or EM image objects from NeuronBridge API.
            
        Returns
        -------
        list
            Filtered list of images matching the region criteria.
        """
        if self.region == 'All':
            return images
        
        filtered = []
        for img in images:
            # Get anatomicalArea attribute if it exists
            area = getattr(img, 'anatomicalArea', None)
            if area is None:
                # If no area specified, include by default
                filtered.append(img)
                continue
            
            # Normalize area name (case-insensitive)
            area_lower = area.lower()
            
            if self.region == 'Brain':
                # Include if area contains 'brain' but not 'vnc'
                if 'brain' in area_lower and 'vnc' not in area_lower:
                    filtered.append(img)
            elif self.region == 'VNC':
                # Include if area contains 'vnc'
                if 'vnc' in area_lower:
                    filtered.append(img)
        
        return filtered
    
    def _filter_images_by_match_availability(
        self, 
        images: List[Any], 
        match_type: str
    ) -> List[Any]:
        """
        Filter images to only include those with available match results for the specified type.
        
        Pre-scans image metadata to check for CDSResults/PPPMResults URLs,
        skipping images that don't have the required match type data.
        
        Parameters
        ----------
        images : list
            List of LM image objects from NeuronBridge API.
        match_type : str
            Match algorithm: 'cds', 'pppm', or 'both'.
            
        Returns
        -------
        list
            Filtered list of images that have the required match results available.
        """
        if match_type == 'both':
            # For 'both', include images that have either CDS or PPPM
            filtered = []
            for img in images:
                files = getattr(img, 'files', None)
                if files is None:
                    continue
                has_cds = getattr(files, 'CDSResults', None) is not None
                has_pppm = getattr(files, 'PPPMResults', None) is not None
                if has_cds or has_pppm:
                    filtered.append(img)
            return filtered
        
        elif match_type == 'cds':
            # Only include images with CDS results
            filtered = []
            for img in images:
                files = getattr(img, 'files', None)
                if files is None:
                    continue
                if getattr(files, 'CDSResults', None) is not None:
                    filtered.append(img)
            return filtered
        
        elif match_type == 'pppm':
            # Only include images with PPPM results
            filtered = []
            for img in images:
                files = getattr(img, 'files', None)
                if files is None:
                    continue
                if getattr(files, 'PPPMResults', None) is not None:
                    filtered.append(img)
            return filtered
        
        # Unknown match_type, return all
        return images
    
    def _classify_line_type(self, line_name: str) -> str:
        """
        Classify a driver line as 'gal4_lexa' or 'split_gal4'.
        
        Classification rules:
        - GAL4/LexA: Lines starting with VT, R, GMR (e.g., VT037867, R10A06)
        - Split-GAL4: Lines starting with SS, LH, MB, IS, OL, LC, etc.
        
        Parameters
        ----------
        line_name : str
            The driver line name to classify.
            
        Returns
        -------
        str
            'gal4_lexa' or 'split_gal4'
        """
        if not line_name:
            return 'gal4_lexa'  # default
        
        line_upper = line_name.upper()
        
        # Check Split-GAL4 prefixes first (more specific)
        for prefix in SPLIT_GAL4_PREFIXES:
            if line_upper.startswith(prefix.upper()):
                return 'split_gal4'
        
        # Check GAL4/LexA prefixes
        for prefix in GAL4_LEXA_PREFIXES:
            if line_upper.startswith(prefix.upper()):
                return 'gal4_lexa'
        
        # Default to gal4_lexa for unknown prefixes
        return 'gal4_lexa'
    
    def _aggregate_results_polars(
        self,
        combined_df: pd.DataFrame,
        match_type: str,
        is_multi_dataset: bool,
        sort_by: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregate results using Polars for fast processing of large datasets.
        
        This optimized version uses Polars DataFrames for aggregation operations,
        which is significantly faster than pandas for datasets with >100k rows.
        
        Parameters
        ----------
        combined_df : pd.DataFrame
            Combined results DataFrame.
        match_type : str
            Match type: 'cds', 'pppm', or 'both'.
        is_multi_dataset : bool
            Whether multi-dataset mode is enabled.
        sort_by : str
            Sorting method: 'completeness' or 'max'.
            
        Returns
        -------
        tuple
            (combined_df, line_stats) - processed DataFrames.
        """
        import polars as pl
        
        # Normalize every object column before ``pl.from_pandas``:
        # columns mixing python ints with numeric strings (large bodyId /
        # rootId values arrive as strings from the API, ints from local
        # matches) make pyarrow fail with "Could not convert ... tried to
        # convert to int64". All-numeric columns become numeric; anything
        # with non-numeric content is forced to strings so inference never
        # fails. Integer-valued columns beyond int64 stay strings so the
        # huge IDs never lose precision in float64.
        for col in combined_df.columns:
            ser = combined_df[col]
            if ser.dtype != object:
                continue
            numeric = pd.to_numeric(ser, errors='coerce')
            if numeric.notna().all():
                if numeric.dtype.kind == 'i':
                    combined_df[col] = numeric
                elif (numeric % 1 == 0).all():
                    combined_df[col] = ser.astype(str)
                else:
                    combined_df[col] = numeric
            else:
                combined_df[col] = ser.astype(str)
        
        # Ensure consistent column types before Polars conversion:
        # bodyId columns can have mixed int/str types (especially FlyWire IDs)
        for col in ['source_bodyId', 'bodyId']:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].astype(str)
        
        # Convert to polars for fast aggregation
        pl_df = pl.from_pandas(combined_df)
        
        # Calculate total unique query neurons
        total_query_neurons = 0
        if 'source_bodyId' in pl_df.columns:
            total_query_neurons = pl_df.select(
                pl.col('source_bodyId').cast(pl.Utf8).drop_nulls().n_unique()
            ).item()
        
        self._vprint(f"   📊 Total unique query neurons: {total_query_neurons}")
        
        # Add line_type classification if needed
        if self.separate_splitgal4 and 'line' in pl_df.columns:
            # Vectorized line type classification
            split_prefixes = [p.upper() for p in SPLIT_GAL4_PREFIXES]
            
            # Create classification expression
            line_type_expr = pl.when(
                pl.col('line').str.to_uppercase().str.starts_with(pl.lit(split_prefixes[0]))
            ).then(pl.lit('split_gal4'))
            
            for prefix in split_prefixes[1:]:
                line_type_expr = line_type_expr.when(
                    pl.col('line').str.to_uppercase().str.starts_with(pl.lit(prefix))
                ).then(pl.lit('split_gal4'))
            
            line_type_expr = line_type_expr.otherwise(pl.lit('gal4_lexa'))
            
            pl_df = pl_df.with_columns(line_type_expr.alias('line_type'))
        
        # Build type lookup table for enrichment (vectorized)
        if 'source_bodyId' in pl_df.columns and 'source_dataset' in pl_df.columns:
            # Get unique (bodyId, dataset) pairs
            unique_pairs = pl_df.select(['source_bodyId', 'source_dataset']).unique()
            
            # Build type mapping from neuron_dfs
            type_mappings = []
            for ds in unique_pairs.select('source_dataset').unique().to_series().to_list():
                if ds is None:
                    continue
                ds_folder = self._dataset_name_to_folder(ds)
                neuron_df = self._load_neuron_df_for_dataset(ds_folder)
                if neuron_df is not None and 'bodyId' in neuron_df.columns and 'type' in neuron_df.columns:
                    # Convert to polars and create mapping
                    type_df = pl.from_pandas(neuron_df[['bodyId', 'type']].copy())
                    type_df = type_df.with_columns([
                        pl.col('bodyId').cast(pl.Utf8).alias('source_bodyId'),
                        pl.lit(ds).alias('source_dataset')
                    ]).select(['source_bodyId', 'source_dataset', 'type'])
                    type_mappings.append(type_df)
            
            if type_mappings:
                type_lookup = pl.concat(type_mappings)
                pl_df = pl_df.with_columns(pl.col('source_bodyId').cast(pl.Utf8))
                pl_df = pl_df.join(
                    type_lookup.rename({'type': 'source_type'}),
                    on=['source_bodyId', 'source_dataset'],
                    how='left'
                )
            elif 'source_type' not in pl_df.columns:
                pl_df = pl_df.with_columns(pl.lit('').alias('source_type'))
        
        # Aggregate line-level results
        if 'line' in pl_df.columns:
            self._vprint(f"   🔄 Aggregating {pl_df.shape[0]:,} rows by line...")
            
            # Build aggregation expressions
            agg_exprs = [
                pl.col('score').mean().alias('agg_mean_score'),
                pl.col('score').max().alias('agg_max_score'),
                pl.col('source_bodyId').cast(pl.Utf8).drop_nulls().n_unique().alias('match_count'),
                pl.col('source_bodyId').cast(pl.Utf8).drop_nulls().unique().sort().str.join(',').alias('matched_bodyIds'),
            ]
            
            if 'source_type' in pl_df.columns:
                agg_exprs.append(
                    pl.col('source_type').drop_nulls().filter(pl.col('source_type') != '').unique().sort().str.join(',').alias('matched_types')
                )
            
            if self.separate_splitgal4 and 'line_type' in pl_df.columns:
                agg_exprs.append(pl.col('line_type').first().alias('line_type'))
            
            if is_multi_dataset and 'source_dataset' in pl_df.columns:
                agg_exprs.extend([
                    pl.col('source_dataset').drop_nulls().n_unique().alias('datasets_labeled'),
                    pl.col('source_dataset').drop_nulls().unique().sort().str.join(',').alias('matched_datasets'),
                ])
            
            # Perform aggregation
            line_stats_pl = pl_df.group_by('line').agg(agg_exprs)
            
            # Calculate cross-dataset scores if multi-dataset
            if is_multi_dataset and 'source_dataset' in pl_df.columns:
                self._vprint(f"   🔄 Calculating cross-dataset scores...")
                
                # Get max score per (line, dataset) pair
                line_ds_scores = pl_df.group_by(['line', 'source_dataset']).agg(
                    pl.col('score').max().alias('max_score_per_ds')
                )
                
                # Calculate min and mean of max scores per line
                cross_ds_stats = line_ds_scores.group_by('line').agg([
                    pl.col('max_score_per_ds').min().alias('min_score_per_dataset'),
                    pl.col('max_score_per_ds').mean().alias('cross_dataset_score'),
                ])
                
                # Join back to line_stats
                line_stats_pl = line_stats_pl.join(cross_ds_stats, on='line', how='left')
            
            # Calculate weighted_score
            if total_query_neurons > 0:
                line_stats_pl = line_stats_pl.with_columns([
                    (pl.col('match_count') / total_query_neurons).alias('coverage_ratio'),
                    (pl.col('agg_mean_score') * pl.col('match_count') / total_query_neurons).alias('weighted_score'),
                ])
                
                # Sort based on sort_by parameter
                if sort_by == 'completeness':
                    # Best coverage of all queries: highest fraction of
                    # query neurons matched, regardless of score.
                    line_stats_pl = line_stats_pl.sort('coverage_ratio', descending=True)
                    self._vprint(f"   📊 Sorting by coverage_ratio (fraction of query neurons matched)")
                else:
                    # Default 'max': score-weighted coverage (agg_mean_score
                    # × coverage_ratio) — rewards both strong matches and
                    # broad coverage.
                    line_stats_pl = line_stats_pl.sort('weighted_score', descending=True)
                    self._vprint(f"   📊 Sorting by weighted_score (agg_mean_score × coverage_ratio)")
            elif is_multi_dataset and 'min_score_per_dataset' in line_stats_pl.columns:
                line_stats_pl = line_stats_pl.sort('min_score_per_dataset', descending=True)
                self._vprint(f"   📊 Multi-dataset sorting: by min_score_per_dataset")
            else:
                line_stats_pl = line_stats_pl.sort('agg_mean_score', descending=True)
            
            # Convert back to pandas
            line_stats = line_stats_pl.to_pandas()
            
            # Reorder columns
            priority_cols = ['line', 'line_type', 'match_count', 'weighted_score', 'coverage_ratio', 
                             'agg_mean_score', 'agg_max_score', 'min_score_per_dataset', 
                             'cross_dataset_score', 'datasets_labeled', 'matched_bodyIds', 
                             'matched_types', 'matched_datasets']
            cols = list(line_stats.columns)
            ordered_cols = [col for col in priority_cols if col in cols]
            remaining_cols = [col for col in cols if col not in ordered_cols]
            line_stats = line_stats[ordered_cols + remaining_cols]
            
            # Add matched info back to combined_df
            merge_cols = ['line', 'matched_bodyIds']
            if 'matched_types' in line_stats.columns:
                merge_cols.append('matched_types')
            
            # Convert pl_df back to pandas for merge
            combined_df = pl_df.to_pandas()
            combined_df = combined_df.merge(
                line_stats[merge_cols],
                on='line',
                how='left',
                suffixes=('', '_agg')
            )
            
            # Clean up duplicate columns
            for col in list(combined_df.columns):
                if col.endswith('_agg'):
                    base_col = col[:-4]
                    if base_col in combined_df.columns:
                        combined_df[base_col] = combined_df[col]
                    combined_df = combined_df.drop(columns=[col])
            
            self._vprint(f"   ✅ Aggregation complete: {len(line_stats):,} unique lines")
            
            return combined_df, line_stats
        
        return combined_df, pd.DataFrame()
    
    def _aggregate_results_pandas(
        self,
        combined_df: pd.DataFrame,
        match_type: str,
        is_multi_dataset: bool,
        sort_by: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregate results using pandas (original implementation).
        
        Used for smaller datasets where Polars overhead isn't worth it.
        
        Parameters
        ----------
        combined_df : pd.DataFrame
            Combined results DataFrame.
        match_type : str
            Match type: 'cds', 'pppm', or 'both'.
        is_multi_dataset : bool
            Whether multi-dataset mode is enabled.
        sort_by : str
            Sorting method: 'completeness' or 'max'.
            
        Returns
        -------
        tuple
            (combined_df, line_stats) - processed DataFrames.
        """
        # Calculate total unique query neurons
        total_query_neurons = 0
        if 'source_bodyId' in combined_df.columns:
            total_query_neurons = len(set(str(v) for v in combined_df['source_bodyId'].dropna().unique()))
        
        # Add line_type classification if separate_splitgal4 is enabled
        if self.separate_splitgal4 and 'line' in combined_df.columns:
            combined_df['line_type'] = combined_df['line'].apply(self._classify_line_type)
        
        # Enrich with type information from neuron_df if available
        if 'source_bodyId' in combined_df.columns and 'source_dataset' in combined_df.columns:
            # Get unique (bodyId, dataset) pairs first to minimize lookups
            unique_pairs = combined_df[['source_bodyId', 'source_dataset']].drop_duplicates()
            
            type_cache = {}
            for _, row in unique_pairs.iterrows():
                body_id = row['source_bodyId']
                ds = row['source_dataset']
                if pd.isna(body_id) or pd.isna(ds):
                    continue
                key = (str(body_id), ds)
                if key not in type_cache:
                    ds_folder = self._dataset_name_to_folder(ds)
                    # Unrecognized dataset values (e.g. 'unknown' placeholders
                    # for skipped bodyIds) have no local neuron table; skip
                    # enrichment for them instead of crashing on os.path.join.
                    if ds_folder is None:
                        type_cache[key] = None
                        continue
                    neuron_df = self._load_neuron_df_for_dataset(ds_folder)
                    if neuron_df is not None and 'bodyId' in neuron_df.columns and 'type' in neuron_df.columns:
                        match = neuron_df[neuron_df['bodyId'].astype(str) == str(body_id)]
                        if not match.empty:
                            type_cache[key] = match['type'].iloc[0]
                        else:
                            type_cache[key] = None
                    else:
                        type_cache[key] = None
            
            # Vectorized lookup
            combined_df['source_type'] = combined_df.apply(
                lambda row: type_cache.get((str(row.get('source_bodyId', '')), row.get('source_dataset', '')), ''),
                axis=1
            )
        
        line_stats = pd.DataFrame()
        
        # Aggregate line-level results
        if 'line' in combined_df.columns:
            # Build aggregation dict
            agg_dict = {
                'score': ['mean', 'max'],
                'source_bodyId': [
                    lambda x: len(set(str(v) for v in x.dropna().unique())),
                    lambda x: ','.join(sorted(set(str(v) for v in x.dropna().unique())))
                ]
            }
            
            if 'source_type' in combined_df.columns:
                agg_dict['source_type'] = lambda x: ','.join(sorted(set(str(v) for v in x.dropna().unique() if v)))
            
            if is_multi_dataset and 'source_dataset' in combined_df.columns:
                agg_dict['source_dataset'] = [
                    lambda x: len(set(str(v) for v in x.dropna().unique())),
                    lambda x: ','.join(sorted(set(str(v) for v in x.dropna().unique())))
                ]
            
            line_stats = combined_df.groupby('line').agg(agg_dict).reset_index()
            
            # Flatten multi-level columns
            new_cols = ['line']
            for col in line_stats.columns[1:]:
                if isinstance(col, tuple):
                    new_cols.append(f"{col[0]}_{col[1] if isinstance(col[1], str) else 'agg'}")
                else:
                    new_cols.append(col)
            line_stats.columns = new_cols
            
            # Rename columns
            rename_map = {'score_mean': 'agg_mean_score', 'score_max': 'agg_max_score'}
            for col in line_stats.columns:
                if 'source_bodyId' in col and 'lambda' in col and '0' in col:
                    rename_map[col] = 'match_count'
                elif 'source_bodyId' in col and 'lambda' in col:
                    rename_map[col] = 'matched_bodyIds'
                elif 'source_type' in col and 'lambda' in col:
                    rename_map[col] = 'matched_types'
                elif 'source_dataset' in col and 'lambda' in col and '0' in col:
                    rename_map[col] = 'datasets_labeled'
                elif 'source_dataset' in col and 'lambda' in col:
                    rename_map[col] = 'matched_datasets'
            
            line_stats = line_stats.rename(columns=rename_map)
            
            # Add line_type if available
            if self.separate_splitgal4 and 'line_type' in combined_df.columns:
                line_type_map = combined_df.groupby('line')['line_type'].first()
                line_stats['line_type'] = line_stats['line'].map(line_type_map)
            
            # Calculate cross-dataset scores
            if is_multi_dataset and 'source_dataset' in combined_df.columns:
                # Vectorized approach: group by (line, source_dataset), get max score
                line_ds_scores = combined_df.groupby(['line', 'source_dataset'])['score'].max().reset_index()
                
                # Group by line to get min and mean of max scores
                cross_ds_stats = line_ds_scores.groupby('line').agg(
                    min_score_per_dataset=('score', 'min'),
                    cross_dataset_score=('score', 'mean')
                ).reset_index()
                
                line_stats = line_stats.merge(cross_ds_stats, on='line', how='left')
            
            # Calculate weighted_score
            if 'match_count' in line_stats.columns and total_query_neurons > 0:
                line_stats['coverage_ratio'] = line_stats['match_count'] / total_query_neurons
                line_stats['weighted_score'] = line_stats['agg_mean_score'] * line_stats['coverage_ratio']
                
                if sort_by == 'completeness':
                    # Best coverage of all queries: highest fraction of
                    # query neurons matched, regardless of score.
                    line_stats = line_stats.sort_values('coverage_ratio', ascending=False)
                    self._vprint(f"   📊 Sorting by coverage_ratio (fraction of query neurons matched)")
                else:
                    # Default 'max': score-weighted coverage (agg_mean_score
                    # × coverage_ratio) — rewards both strong matches and
                    # broad coverage.
                    line_stats = line_stats.sort_values('weighted_score', ascending=False)
                    self._vprint(f"   📊 Sorting by weighted_score (agg_mean_score × coverage_ratio)")
                self._vprint(f"      Total query neurons: {total_query_neurons}")
            elif is_multi_dataset:
                line_stats = line_stats.sort_values('min_score_per_dataset', ascending=False)
                self._vprint(f"   📊 Multi-dataset sorting: by min_score_per_dataset")
            else:
                line_stats = line_stats.sort_values('agg_mean_score', ascending=False)
            
            # Reorder columns
            priority_cols = ['line', 'line_type', 'match_count', 'weighted_score', 'coverage_ratio', 
                             'agg_mean_score', 'agg_max_score', 'min_score_per_dataset', 
                             'cross_dataset_score', 'datasets_labeled', 'matched_bodyIds', 
                             'matched_types', 'matched_datasets']
            cols = list(line_stats.columns)
            ordered_cols = [col for col in priority_cols if col in cols]
            remaining_cols = [col for col in cols if col not in ordered_cols]
            line_stats = line_stats[ordered_cols + remaining_cols]
            
            # Add matched info back to combined_df
            merge_cols = ['line', 'matched_bodyIds']
            if 'matched_types' in line_stats.columns:
                merge_cols.append('matched_types')
            combined_df = combined_df.merge(line_stats[merge_cols], on='line', how='left')
        
        return combined_df, line_stats
    
    def _calculate_expression_entropy(self, type_counts: Dict[str, int]) -> float:
        """
        Calculate Shannon entropy of expression pattern.
        
        H = -Σ p_i × log2(p_i)
        
        Lower entropy = more specific (labels fewer types more strongly)
        Higher entropy = more promiscuous (labels many types equally)
        
        Parameters
        ----------
        type_counts : dict
            Dictionary mapping neuron types to their occurrence counts.
            
        Returns
        -------
        float
            Shannon entropy in bits. Range: 0 (perfectly specific) to log2(N) (uniform).
        """
        if not type_counts:
            return 0.0
        
        total = sum(type_counts.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in type_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _calculate_weighted_specificity(
        self, 
        neurons_df: pd.DataFrame,
        queried_types_lower: set,
        score_column: str = 'score'
    ) -> Dict[str, float]:
        """
        Calculate expression-strength-weighted specificity metrics.
        
        Weights each neuron's contribution by its NeuronBridge match score,
        so high-confidence matches count more than low-confidence ones.
        
        Parameters
        ----------
        neurons_df : pd.DataFrame
            DataFrame from line_to_neuron with 'type' and score columns.
        queried_types_lower : set
            Set of queried neuron types (lowercase).
        score_column : str
            Name of score column to use for weighting.
            
        Returns
        -------
        dict
            Dictionary with:
            - 'weighted_type_proportion': Score-weighted proportion of queried types
            - 'weighted_queried_score': Total score for queried types
            - 'weighted_total_score': Total score for all types
            - 'mean_queried_score': Mean score for queried types
        """
        if neurons_df.empty or score_column not in neurons_df.columns:
            return {
                'weighted_type_proportion': 0.0,
                'weighted_queried_score': 0.0,
                'weighted_total_score': 0.0,
                'mean_queried_score': 0.0
            }
        
        # Calculate total weighted scores
        total_score = neurons_df[score_column].sum()
        
        # Get scores for queried types
        queried_mask = neurons_df['type'].fillna('').str.lower().isin(queried_types_lower)
        queried_scores = neurons_df.loc[queried_mask, score_column]
        
        queried_total = queried_scores.sum()
        queried_mean = queried_scores.mean() if len(queried_scores) > 0 else 0.0
        
        # Weighted proportion
        weighted_proportion = queried_total / total_score if total_score > 0 else 0.0
        
        return {
            'weighted_type_proportion': weighted_proportion,
            'weighted_queried_score': queried_total,
            'weighted_total_score': total_score,
            'mean_queried_score': queried_mean
        }
    
    def _build_colabeling_matrix(
        self,
        lines: List[str],
        match_type: str = 'cds',
        top_n: int = 100,
        similarity_method: str = 'weighted_jaccard',
        min_score: float = 0.0,
        min_type_avg_score: float = 0.0
    ) -> Tuple[pd.DataFrame, Dict[str, set]]:
        """
        Build a co-labeling matrix showing how often pairs of lines label the same cell types.
        
        Supports multiple similarity measures:
        - 'jaccard': Binary Jaccard similarity (presence/absence of types)
        - 'weighted_jaccard': Jaccard weighted by match scores
        - 'rank_correlation': Spearman correlation of type rankings based on scores
        
        Parameters
        ----------
        lines : list of str
            List of driver line names.
        match_type : str
            Match algorithm for line_to_neuron.
        top_n : int
            Number of top matches to consider per line.
        similarity_method : str
            Similarity method: 'jaccard', 'weighted_jaccard', or 'rank_correlation'.
            Default: 'weighted_jaccard'
        min_score : float
            Minimum score threshold for individual neurons. Default: 0.0 (no filter).
        min_type_avg_score : float
            Minimum average score threshold for types (across all lines). Default: 0.0 (no filter).
            
        Returns
        -------
        tuple
            (co_labeling_matrix, line_type_sets)
            - co_labeling_matrix: DataFrame with similarities based on cell types
            - line_type_sets: Dict mapping line names to sets of cell type names
        """
        # Validate inputs using helper methods
        match_type = self._validate_match_type(match_type)
        similarity_method = self._validate_similarity_method(similarity_method)
        
        self._vprint(f"\n🔗 Building co-labeling matrix for {len(lines)} lines...")
        self._vprint(f"   📊 Similarity method: {similarity_method}")
        if min_score > 0:
            self._vprint(f"   📊 Min neuron score: {min_score:,.0f}")
        if min_type_avg_score > 0:
            self._vprint(f"   📊 Min type avg score: {min_type_avg_score:,.0f}")
        self._vprint(f"   ⏱️  Note: Fetching neurons for each line to build type-based similarity matrix")
        
        # Collect type sets AND scores for each line
        line_neuron_sets = {}
        line_type_scores = {}
        
        # Enable batch mode to suppress individual cache messages
        self._batch_mode = True
        
        if HAS_TQDM and self.verbose:
            from tqdm import tqdm as tqdm_progress
            iterator = tqdm_progress(
                lines, 
                desc="   🔍 Collecting neurons per line", 
                unit="line",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                ncols=100,
                file=sys.stdout,
            )
        else:
            iterator = lines
        
        for idx, line_name in enumerate(iterator):
            if HAS_TQDM and self.verbose:
                iterator.set_description(f"   🔍 [{idx+1}/{len(lines)}] Fetching neurons for {line_name}")
            try:
                neurons_df = self.line_to_neuron(
                    line_name,
                    match_type=match_type,
                    top_n=top_n,
                )
                if not neurons_df.empty:
                    # Always select the ranked top-N records first.  A
                    # positive score cutoff is a downstream filter and must
                    # never reduce the number of retained top-N records.
                    neurons_top = neurons_df.head(top_n) if top_n > 0 else neurons_df
                    # Apply minimum score filter if specified
                    if min_score > 0:
                        neurons_top = neurons_top[neurons_top['score'] >= min_score]
                    # Use cell types for similarity (not bodyIds) - this gives meaningful co-labeling
                    # Two lines labeling the same cell types are considered similar
                    type_set = set()
                    type_scores = {}  # type -> score mapping
                    for _, row in neurons_top.iterrows():
                        n_type = row.get('type', '')
                        score = row.get('score', 0.0)
                        if n_type and str(n_type).lower() not in ['', 'nan', 'none', 'unknown']:
                            type_key = str(n_type).lower()
                            type_set.add(type_key)
                            # Keep the maximum score if type appears multiple times
                            type_scores[type_key] = max(type_scores.get(type_key, 0.0), float(score))
                    line_neuron_sets[line_name] = type_set
                    line_type_scores[line_name] = type_scores
                else:
                    line_neuron_sets[line_name] = set()
                    line_type_scores[line_name] = {}
            except Exception as e:
                line_neuron_sets[line_name] = set()
                line_type_scores[line_name] = {}
                if self.verbose:
                    self._vprint(f"   ⚠️ Error getting neurons for {line_name}: {e}")
        
        # Disable batch mode after fetching
        self._batch_mode = False
        
        # Apply min_type_avg_score filter: remove types with low average scores
        if min_type_avg_score > 0:
            # Collect all types and their average scores across lines
            all_type_scores = {}  # type -> list of scores
            for line_scores in line_type_scores.values():
                for type_key, score in line_scores.items():
                    if type_key not in all_type_scores:
                        all_type_scores[type_key] = []
                    all_type_scores[type_key].append(score)
            
            # Filter types by average score
            types_to_keep = set()
            for type_key, scores in all_type_scores.items():
                avg_score = sum(scores) / len(scores)
                if avg_score >= min_type_avg_score:
                    types_to_keep.add(type_key)
            
            # Remove filtered types from line sets and scores
            removed_types = set(all_type_scores.keys()) - types_to_keep
            if removed_types:
                self._vprint(f"   📊 Filtered {len(removed_types)} types with avg score < {min_type_avg_score:,.0f}")
                self._vprint(f"   📊 Keeping {len(types_to_keep)} high-confidence types")
                for line_name in line_neuron_sets:
                    line_neuron_sets[line_name] = line_neuron_sets[line_name] & types_to_keep
                    line_type_scores[line_name] = {
                        k: v for k, v in line_type_scores[line_name].items() 
                        if k in types_to_keep
                    }
        
        # Build similarity matrix based on selected method
        self._vprint(f"   🔢 Computing {similarity_method} similarities between {len(lines)} lines...")
        n = len(lines)
        matrix = np.zeros((n, n))
        
        if similarity_method == 'jaccard':
            # Binary Jaccard (original implementation)
            for i, line_i in enumerate(lines):
                set_i = line_neuron_sets.get(line_i, set())
                for j, line_j in enumerate(lines):
                    if i == j:
                        matrix[i, j] = 1.0
                    elif j > i:
                        set_j = line_neuron_sets.get(line_j, set())
                        if set_i and set_j:
                            intersection = len(set_i & set_j)
                            union = len(set_i | set_j)
                            jaccard = intersection / union if union > 0 else 0.0
                        else:
                            jaccard = 0.0
                        matrix[i, j] = jaccard
                        matrix[j, i] = jaccard
                        
        elif similarity_method == 'weighted_jaccard':
            # Weighted Jaccard using match scores
            for i, line_i in enumerate(lines):
                scores_i = line_type_scores.get(line_i, {})
                for j, line_j in enumerate(lines):
                    if i == j:
                        matrix[i, j] = 1.0
                    elif j > i:
                        scores_j = line_type_scores.get(line_j, {})
                        if scores_i and scores_j:
                            all_types = set(scores_i.keys()) | set(scores_j.keys())
                            intersection_sum = sum(
                                min(scores_i.get(t, 0.0), scores_j.get(t, 0.0)) 
                                for t in all_types
                            )
                            union_sum = sum(
                                max(scores_i.get(t, 0.0), scores_j.get(t, 0.0)) 
                                for t in all_types
                            )
                            w_jaccard = intersection_sum / union_sum if union_sum > 0 else 0.0
                        else:
                            w_jaccard = 0.0
                        matrix[i, j] = w_jaccard
                        matrix[j, i] = w_jaccard
                        
        elif similarity_method == 'rank_correlation':
            # Spearman correlation on overlapping types
            from scipy.stats import spearmanr
            for i, line_i in enumerate(lines):
                scores_i = line_type_scores.get(line_i, {})
                for j, line_j in enumerate(lines):
                    if i == j:
                        matrix[i, j] = 1.0
                    elif j > i:
                        scores_j = line_type_scores.get(line_j, {})
                        if scores_i and scores_j:
                            # Find types in common
                            common_types = set(scores_i.keys()) & set(scores_j.keys())
                            if len(common_types) > 1:
                                # Get scores for common types
                                values_i = [scores_i[t] for t in common_types]
                                values_j = [scores_j[t] for t in common_types]
                                # Compute Spearman correlation
                                corr, _ = spearmanr(values_i, values_j)
                                # Handle NaN (can occur with constant values)
                                if np.isnan(corr):
                                    corr = 0.0
                            else:
                                corr = 0.0
                        else:
                            corr = 0.0
                        matrix[i, j] = corr
                        matrix[j, i] = corr
        else:
            raise ValueError(f"Unknown similarity method: {similarity_method}")
        
        co_labeling_df = pd.DataFrame(matrix, index=lines, columns=lines)
        
        return co_labeling_df, line_neuron_sets
    
    def _calculate_colabeling_sparsity(self, co_labeling_matrix: pd.DataFrame, threshold: float = 0.1) -> Dict[str, float]:
        """
        Calculate sparsity metrics from co-labeling matrix.
        
        Sparsity measures how unique each line's labeling pattern is.
        High sparsity = line labels neurons that few other lines label.
        
        Parameters
        ----------
        co_labeling_matrix : pd.DataFrame
            Jaccard similarity matrix from _build_colabeling_matrix.
        threshold : float
            Similarity threshold to consider lines as "co-labeling".
            
        Returns
        -------
        dict
            Dictionary mapping line names to sparsity scores (0-1, higher = more unique).
        """
        sparsity_scores = {}
        
        for line in co_labeling_matrix.index:
            # Get similarities with other lines (exclude self)
            similarities = co_labeling_matrix.loc[line].drop(line)
            
            # Count how many lines have significant overlap
            n_colabeling = (similarities > threshold).sum()
            total_other_lines = len(similarities)
            
            # Sparsity = 1 - (proportion of co-labeling lines)
            sparsity = 1.0 - (n_colabeling / total_other_lines) if total_other_lines > 0 else 1.0
            
            # Also compute mean non-self similarity (lower = more unique)
            mean_similarity = similarities.mean() if len(similarities) > 0 else 0.0
            
            sparsity_scores[line] = {
                'colabel_sparsity': sparsity,
                'n_colabeling_lines': int(n_colabeling),
                'mean_colabel_similarity': mean_similarity
            }
        
        return sparsity_scores
    
    def _sort_expression_matrix(
        self,
        expression_df: pd.DataFrame,
        as_types_rows: bool = True,
        input_format: str = 'auto'
    ) -> pd.DataFrame:
        """
        Sort expression matrix by co-labeling quality.
        
        Sorting criteria (when as_types_rows=True, types as rows):
        1. Types labeled in ALL lines (no zeros) come first
        2. Within complete types: sorted by min_score (higher = more consistent)
        3. Types with partial labeling: sorted by num_nonzero (desc), total_score (desc)
        
        Parameters
        ----------
        expression_df : pd.DataFrame
            Expression matrix. Orientation per ``input_format``.
        as_types_rows : bool
            If True (default), output has types as rows and lines as columns.
            If False, output has lines as rows and types as columns.
        input_format : str
            'lines_x_types' or 'types_x_lines' states the input orientation
            explicitly.  'auto' (legacy) guesses from the shape, which is
            unreliable for small matrices (fewer types than lines).
            
        Returns
        -------
        pd.DataFrame
            Sorted expression matrix with types as rows (if as_types_rows=True).
        """
        # Determine if input needs transposing to get Types × Lines format
        if input_format == 'lines_x_types':
            needs_transpose = True
        elif input_format == 'types_x_lines':
            needs_transpose = False
        else:
            # Legacy heuristic: 'line'-named index, or clearly more columns
            # than rows (Lines × Types with many types).
            needs_transpose = (
                expression_df.index.name == 'line' or 
                (as_types_rows and len(expression_df.columns) > len(expression_df))
            )
        
        if needs_transpose:
            expression_transposed = expression_df.T
        else:
            expression_transposed = expression_df.copy()
        
        if len(expression_transposed) == 0 or len(expression_transposed.columns) == 0:
            return expression_transposed if as_types_rows else expression_transposed.T
        
        n_lines = len(expression_transposed.columns)
        
        # Calculate metrics for each type (row)
        nonzero_count = (expression_transposed > 0).sum(axis=1)
        is_complete = (nonzero_count == n_lines).astype(int)
        
        # Create sorting key:
        # - Primary: whether labeled in all lines (1 if all, 0 if not) - descending
        # - Secondary: min_score except 0s iteratedly (tuple of sorted non-zero scores) - descending
        #   This sorts by the smallest non-zero score, then the next smallest, etc.
        
        # Helper to get sorted non-zeros tuple
        def get_nonzero_tuple(row):
            return tuple(sorted(row[row > 0]))
            
        nonzero_tuples = expression_transposed.apply(get_nonzero_tuple, axis=1)
        
        # Create temp dataframe for sorting
        sort_df = pd.DataFrame({
            'is_complete': is_complete,
            'nonzero_count': nonzero_count,
            'nonzero_tuples': nonzero_tuples
        }, index=expression_transposed.index)
        
        # Sort by the composite key
        sorted_index = sort_df.sort_values(
            by=['is_complete', 'nonzero_count', 'nonzero_tuples'], 
            ascending=[False, False, False]
        ).index
        expression_transposed = expression_transposed.loc[sorted_index]
        
        if as_types_rows:
            return expression_transposed
        else:
            return expression_transposed.T
    
    def visualize_colabeling_matrix(
        self,
        co_labeling_matrix: pd.DataFrame,
        output_path: str,
        title: str = "Co-Labeling Matrix",
        color_scale: str = 'purple',
        filename: str = 'colabeling_matrix.html',
        zmin: Optional[float] = 0.0,
        zmax: Optional[float] = 1.0
    ) -> str:
        """
        Visualize co-labeling matrix as a heatmap using VisConnMatInteractive.
        
        Parameters
        ----------
        co_labeling_matrix : pd.DataFrame
            Similarity matrix from _build_colabeling_matrix.
        output_path : str
            Directory to save the heatmap HTML file.
        title : str
            Title for the heatmap.
        color_scale : str
            Color scale preset: 'purple', 'green', 'blue', 'orange', 'red'.
        filename : str
            Filename for the HTML file. Default: 'colabeling_matrix.html'
        zmin : float, optional
            Minimum value for color scale. Default: 0.0
        zmax : float, optional
            Maximum value for color scale. Default: 1.0 (for similarity matrices)
            
        Returns
        -------
        str
            Path to the created heatmap file.
        """
        try:
            from vispath_pkg import VisConnMatInteractive
        except ImportError:
            # Try to add vispath-subproject/src to path
            import sys
            from pathlib import Path
            
            # Find repo root (assuming we are in src/)
            current_file = Path(__file__).resolve()
            repo_root = current_file.parent.parent
            vispath_src = repo_root / 'vispath-subproject' / 'src'
            
            if vispath_src.exists() and str(vispath_src) not in sys.path:
                sys.path.append(str(vispath_src))
            
            try:
                from vispath_pkg import VisConnMatInteractive
            except ImportError:
                self._vprint("   ⚠️ Could not import VisConnMatInteractive from vispath_pkg")
                return ""
        
        # Create output directory if needed
        os.makedirs(output_path, exist_ok=True)
        
        # Resolve color scale preset to Plotly format
        color_scales = {
            'green': [[0, 'rgb(255,255,255)'], [1, 'rgb(14,83,13)']],
            'purple': [[0, 'rgb(255,255,255)'], [1, 'rgb(104,55,164)']],
            'orange': [[0, 'rgb(255,255,255)'], [1, 'rgb(204,102,0)']],
            'blue': [[0, 'rgb(255,255,255)'], [1, 'rgb(31,119,180)']],
            'red': [[0, 'rgb(255,255,255)'], [1, 'rgb(214,39,40)']],
        }
        resolved_color_scale = color_scales.get(color_scale, color_scales['purple'])
        
        # Create heatmap file path
        full_path = os.path.join(output_path, filename)
        
        # Create heatmap using VisConnMatInteractive with fixed color scale range
        VisConnMatInteractive(
            co_labeling_matrix,
            filename=full_path,
            title=title,
            color_scale=resolved_color_scale,
            zmin=zmin,
            zmax=zmax,
            showfig=False,
            verbose=self.verbose,
            init_clustered=False
        )
        
        self._vprint(f"   📊 Created heatmap: {full_path}")
        return full_path

    def _calculate_mutual_information(
        self,
        lines: List[str],
        queried_types: List[str],
        match_type: str = 'cds',
        top_n: int = 100,
        output_path: Optional[str] = None,
        min_score: float = 0.0,
        min_type_avg_score: float = 0.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Calculate mutual information between driver lines and neuron types.
        
        MI(L; T) = Σ p(l,t) × log2(p(l,t) / (p(l) × p(t)))
        
        This measures how much knowing a line's expression tells you about neuron type.
        High MI = line expression is highly informative about neuron type.
        
        Parameters
        ----------
        lines : list of str
            List of driver line names to analyze.
        queried_types : list of str
            List of queried neuron types (for highlighting in visualization).
        match_type : str
            Match algorithm for line_to_neuron.
        top_n : int
            Number of top matches to consider per line.
        output_path : str, optional
            If provided, save per-line neuron details CSVs to this directory.
        min_score : float
            Minimum score used to filter the expression matrix and labeling
            information after the ranked top-N records have been retrieved.
            The per-line neuron tables retain all top-N records and mark rows
            below this cutoff. Default: 0.0.
        min_type_avg_score : float
            NOT USED in this function. Kept for API compatibility.
            Type filtering is done in _build_colabeling_matrix for similarity calculations.
            Expression matrix filtering is controlled independently by min_score.
            
        Returns
        -------
        tuple
            (mi_df, expression_matrix, line_neurons_dict, labeling_info)
            - mi_df: DataFrame with MI values per line
            - expression_matrix: Score-based expression matrix (lines × types)
              Types are prefixed with dataset abbreviation: {ABBREV}_{type}
            - line_neurons_dict: Dict mapping line name to neurons DataFrame
            - labeling_info: DataFrame with type, dataset, and per-line score columns
        """
        self._vprint(f"\n📊 Calculating mutual information for {len(lines)} lines...")
        if min_score > 0:
            self._vprint(f"   📊 Min neuron score (expression threshold): {min_score:,.0f}")
        # Note: min_type_avg_score is NOT used here - filtering happens in _build_colabeling_matrix
        self._vprint(f"   ⏱️  Note: Fetching neuron types for each line (may take time)")
        
        # Normalize queried types for comparison (but keep original case)
        queried_types_lower = set(t.lower() for t in queried_types if t)
        
        # Enable batch mode to suppress cache messages
        self._batch_mode = True
        
        # Build expression matrix: rows = lines, cols = neuron types with dataset prefix
        # Value = max score if line labels that type, 0 otherwise
        # Format: {dataset_abbrev}_{type} e.g., HEMI_DM4, MCNS_dm4
        all_prefixed_types = set()  # Case-sensitive types with dataset prefix
        line_type_sets = {}  # Lowercase types per line (for MI calculation)
        line_prefixed_type_scores = {}  # Prefixed type -> score per line
        line_neurons_dict = {}  # Store neurons DataFrame for each line
        
        # For labeling_info: store (type, dataset) -> {line: score}
        labeling_info_data = {}  # (type, dataset) -> {line: score}
        
        if HAS_TQDM and self.verbose:
            from tqdm import tqdm as tqdm_progress
            iterator = tqdm_progress(
                lines, 
                desc="   🧬 Building expression matrix", 
                unit="line",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                ncols=100,
                file=sys.stdout,
            )
        else:
            iterator = lines
        
        for idx, line_name in enumerate(iterator):
            if HAS_TQDM and self.verbose:
                iterator.set_description(f"   🧬 [{idx+1}/{len(lines)}] Processing {line_name}")
            try:
                # ``line_to_neuron`` applies top-N to the ranked records before
                # this method applies the score cutoff.  This preserves the
                # requested top-N result list, including rows below the cutoff.
                neurons_df = self.line_to_neuron(line_name, match_type=match_type, top_n=top_n)
                if not neurons_df.empty:
                    neurons_to_store = neurons_df.copy()
                    neurons_to_store['_passes_min_score'] = neurons_to_store['score'] >= (
                        min_score if min_score > 0 else 0
                    )
                    line_neurons_dict[line_name] = neurons_to_store

                    # The cutoff affects aggregate expression outputs, but not
                    # the raw top-N per-line tables retained above.
                    expression_neurons = neurons_df
                    if min_score > 0:
                        expression_neurons = neurons_df[neurons_df['score'] >= min_score]

                    # Get lowercase types for MI calculation from the same
                    # score-filtered data used by the expression matrix.
                    types = expression_neurons['type'].fillna('Unknown').unique()
                    types_lower = set(t.lower() for t in types)
                    line_type_sets[line_name] = types_lower

                    if expression_neurons.empty:
                        line_prefixed_type_scores[line_name] = {}
                        continue
                    
                    # Build prefixed type scores: {ABBREV}_{type} with original case
                    line_prefixed_scores = {}
                    for dataset in expression_neurons['dataset'].dropna().unique():
                        ds_df = expression_neurons[expression_neurons['dataset'] == dataset]
                        # Get dataset abbreviation
                        ds_abbrev = DATASET_ABBREVIATIONS.get(dataset, dataset[:4].upper())
                        
                        # Group by original case type and get max score
                        for type_name, type_df in ds_df.groupby(ds_df['type'].fillna('Unknown')):
                            max_score = type_df['score'].max()
                            prefixed_type = f"{ds_abbrev}_{type_name}"
                            
                            # Store for expression matrix
                            all_prefixed_types.add(prefixed_type)
                            line_prefixed_scores[prefixed_type] = max(
                                line_prefixed_scores.get(prefixed_type, 0.0), 
                                max_score
                            )
                    
                    # Build labeling_info from the same filtered data.
                    for dataset in expression_neurons['dataset'].dropna().unique():
                        ds_df = expression_neurons[expression_neurons['dataset'] == dataset]
                        ds_abbrev = DATASET_ABBREVIATIONS.get(dataset, dataset[:4].upper())
                        
                        for type_name, type_df in ds_df.groupby(ds_df['type'].fillna('Unknown')):
                            max_score = type_df['score'].max()
                            
                            # Store for labeling_info
                            key = (type_name, dataset)
                            if key not in labeling_info_data:
                                labeling_info_data[key] = {}
                            labeling_info_data[key][line_name] = max(
                                labeling_info_data[key].get(line_name, 0.0),
                                max_score
                            )
                    
                    line_prefixed_type_scores[line_name] = line_prefixed_scores
                    
                else:
                    line_type_sets[line_name] = set()
                    line_prefixed_type_scores[line_name] = {}
                    line_neurons_dict[line_name] = pd.DataFrame()
            except Exception as e:
                line_type_sets[line_name] = set()
                line_prefixed_type_scores[line_name] = {}
                line_neurons_dict[line_name] = pd.DataFrame()
        
        # Disable batch mode after fetching
        self._batch_mode = False
        
        if not all_prefixed_types:
            self._vprint("   ⚠️ No types found for any line")
            return pd.DataFrame(), pd.DataFrame(), line_neurons_dict, pd.DataFrame()
        
        # NOTE: min_type_avg_score filtering is NOT applied here. The
        # expression matrix includes all types that remain after min_score;
        # type filtering for similarity/clustering is done in
        # _build_colabeling_matrix.
        
        # Create score-based expression matrix with prefixed types (case-sensitive)
        # Value = max match score if line labels that type, 0 otherwise
        all_prefixed_types_list = sorted(all_prefixed_types)
        expression_matrix = np.zeros((len(lines), len(all_prefixed_types_list)))
        
        for i, line in enumerate(lines):
            for j, prefixed_type in enumerate(all_prefixed_types_list):
                score = line_prefixed_type_scores.get(line, {}).get(prefixed_type, 0.0)
                expression_matrix[i, j] = score
        
        expression_df = pd.DataFrame(
            expression_matrix,
            index=lines,
            columns=all_prefixed_types_list
        )
        
        # Create labeling_info DataFrame
        # Columns: type, dataset, {line1_score}, {line2_score}, ...
        labeling_rows = []
        for (type_name, dataset), line_scores in labeling_info_data.items():
            row = {'type': type_name, 'dataset': dataset}
            for line in lines:
                row[line] = line_scores.get(line, 0.0)
            labeling_rows.append(row)
        
        labeling_info = pd.DataFrame(labeling_rows)
        if not labeling_info.empty:
            # Sort by max score across lines (descending)
            score_cols = [col for col in labeling_info.columns if col not in ['type', 'dataset']]
            labeling_info['_max_score'] = labeling_info[score_cols].max(axis=1)
            labeling_info['_min_score'] = labeling_info[score_cols].min(axis=1)
            labeling_info['_nonzero'] = (labeling_info[score_cols] > 0).sum(axis=1)
            # Sort: all-lines-labeled first (by min_score), then partial (by nonzero, max_score)
            labeling_info['_is_complete'] = labeling_info['_nonzero'] == len(score_cols)
            labeling_info = labeling_info.sort_values(
                ['_is_complete', '_min_score', '_nonzero', '_max_score'],
                ascending=[False, False, False, False]
            )
            labeling_info = labeling_info.drop(columns=['_max_score', '_min_score', '_nonzero', '_is_complete'])
        
        # Calculate MI for each line using lowercase types (original logic)
        # MI(L; T) = H(T) - H(T|L)
        all_types_lower = set()
        for types_set in line_type_sets.values():
            all_types_lower.update(types_set)
        all_types_lower_list = sorted(all_types_lower)
        
        n_lines = len(lines)
        n_types = len(all_types_lower_list)
        
        # Build binary matrix for MI calculation (based on lowercase types)
        mi_binary_matrix = np.zeros((len(lines), n_types))
        for i, line in enumerate(lines):
            types_in_line = line_type_sets.get(line, set())
            for j, ntype in enumerate(all_types_lower_list):
                if ntype in types_in_line:
                    mi_binary_matrix[i, j] = 1.0
        
        # Marginal probability of each type (across all lines)
        type_counts = mi_binary_matrix.sum(axis=0)  # How many lines label each type
        p_type = type_counts / n_lines  # P(type)
        p_type = np.clip(p_type, 1e-10, 1)  # Avoid log(0)
        
        # Entropy of type distribution H(T)
        H_T = -np.sum(p_type * np.log2(p_type))
        
        mi_results = []
        
        for i, line in enumerate(lines):
            types_labeled = line_type_sets.get(line, set())
            n_labeled = len(types_labeled)
            
            if n_labeled == 0:
                mi_results.append({
                    'line': line,
                    'mutual_information': 0.0,
                    'normalized_mi': 0.0,
                    'n_types_labeled': 0,
                    'queried_type_coverage': 0.0
                })
                continue
            
            # Conditional entropy H(T|L=l)
            # P(t|L=l) = 1/n_labeled if line labels type t, 0 otherwise
            p_t_given_l = np.zeros(n_types)
            for j, ntype in enumerate(all_types_lower_list):
                if ntype in types_labeled:
                    p_t_given_l[j] = 1.0 / n_labeled
            
            # H(T|L=l) = -Σ p(t|l) log2(p(t|l))
            p_t_given_l_nonzero = p_t_given_l[p_t_given_l > 0]
            H_T_given_L = -np.sum(p_t_given_l_nonzero * np.log2(p_t_given_l_nonzero))
            
            # Point-wise MI for this line
            # This is simplified: we compute reduction in entropy
            mi_line = H_T - H_T_given_L
            
            # Normalized MI (0-1 scale)
            max_mi = H_T  # Maximum possible MI is H(T)
            normalized_mi = mi_line / max_mi if max_mi > 0 else 0.0
            
            # Queried type coverage
            covered = len(types_labeled & queried_types_lower)
            coverage = covered / len(queried_types_lower) if queried_types_lower else 0.0
            
            mi_results.append({
                'line': line,
                'mutual_information': mi_line,
                'normalized_mi': normalized_mi,
                'n_types_labeled': n_labeled,
                'queried_type_coverage': coverage
            })
        
        mi_df = pd.DataFrame(mi_results)
        
        self._vprint(f"   ✓ MI calculated for {len(mi_df)} lines")
        self._vprint(f"   Total prefixed types: {len(all_prefixed_types_list)}")
        self._vprint(f"   Total unique types (lowercase): {n_types}")
        self._vprint(f"   Type entropy H(T): {H_T:.3f} bits")
        
        return mi_df, expression_df, line_neurons_dict, labeling_info
    
    def visualize_expression_matrix(
        self,
        expression_df: pd.DataFrame,
        output_path: str,
        queried_types: Optional[List[str]] = None,
        title: str = "Line × Type Expression Matrix",
        color_scale: str = 'green',
        top_n_types: int = 100,
        type_filter: Optional[Dict[str, Union[str, List[str]]]] = None
    ) -> str:
        """
        Visualize the expression matrix (lines × types) as a heatmap.
        
        Parameters
        ----------
        expression_df : pd.DataFrame
            Binary expression matrix from _calculate_mutual_information.
        output_path : str
            Directory to save the heatmap HTML file.
        queried_types : list of str, optional
            Queried types to highlight in the matrix.
        title : str
            Title for the heatmap.
        color_scale : str
            Color scale preset.
        top_n_types : int
            Maximum number of types to display (default: 100).
            Types are sorted by total expression score across all lines.
        type_filter : dict, optional
            Filter neuron types by name pattern for visualization.
            Uses a dict with filter type as key and pattern(s) as value.
            
            Filter types:
            - 'contains': Match types containing the pattern
            - 'startswith': Match types starting with the pattern
            - 'endswith': Match types ending with the pattern
            - 'regex': Match types matching the regex pattern
            
            Values: str or list of str patterns
            Multiple patterns within same key use OR logic.
            Multiple keys use AND logic.
            The rank r{N} will be preserved from the original ranking before filtering.
            
            Examples:
            - type_filter = {'contains': 'DN'}  # Types containing 'DN'
            - type_filter = {'startswith': ['DN', 'IN']}  # Types starting with 'DN' or 'IN'
            - type_filter = {'endswith': '_R'}  # Types ending with '_R'
            - type_filter = {'regex': r'DN[a-z]\\d+'}  # Types matching regex pattern
            - type_filter = {'contains': 'DN', 'endswith': '_R'}  # Both conditions (AND)
            
            Set to None or {} to disable filtering (visualize all top types).
            Does not affect the full data CSV - only the viz CSV and heatmap.
            
        Returns
        -------
        str
            Path to the created heatmap file.
        """
        try:
            from vispath_pkg import VisConnMatInteractive
        except ImportError:
            # Try to add vispath-subproject/src to path
            import sys
            from pathlib import Path
            
            # Find repo root (assuming we are in src/)
            current_file = Path(__file__).resolve()
            repo_root = current_file.parent.parent
            vispath_src = repo_root / 'vispath-subproject' / 'src'
            
            if vispath_src.exists() and str(vispath_src) not in sys.path:
                sys.path.append(str(vispath_src))
            
            try:
                from vispath_pkg import VisConnMatInteractive
            except ImportError:
                self._vprint("   ⚠️ Could not import VisConnMatInteractive from vispath_pkg")
                return ""
        
        os.makedirs(output_path, exist_ok=True)
        
        # Transpose first to get types as rows
        # Preserve input order (already sorted by co-labeling quality)
        expression_transposed = expression_df.T.copy()
        
        # Save full data CSV first (before any filtering/truncation)
        full_csv_path = os.path.join(output_path, 'expression_matrix.csv')
        expression_transposed.to_csv(full_csv_path)
        self._vprint(f"   💾 Saved full expression matrix: {full_csv_path}")
        
        # Apply type_filter if specified
        total_types_before_filter = len(expression_transposed)
        if type_filter:
            # For prefixed types (e.g., 'HEMI_aMe12'), extract base type name for filtering
            # Create ranked type list with both full name and base name for filtering
            
            # Create ranked type list using BASE type names for filtering
            ranked_types = [(i, _extract_base_type(name)) for i, name in enumerate(expression_transposed.index, start=1)]
            filtered_base_types = self._apply_type_filter(ranked_types, type_filter)
            filtered_base_names = set(name for _, name in filtered_base_types)
            
            # Keep full prefixed names whose base type matches
            expression_transposed = expression_transposed.loc[
                [idx for idx in expression_transposed.index if _extract_base_type(idx) in filtered_base_names]
            ]
            self._vprint(f"   🔍 Type filter applied: {len(expression_transposed)}/{total_types_before_filter} types match")
        
        # Limit to top N types by total expression score
        total_types = len(expression_transposed)
        if total_types > top_n_types:
            # Filter to top N by total score, but preserve quality-based sorting
            type_totals = expression_transposed.sum(axis=1)
            top_types = set(type_totals.nlargest(top_n_types).index)
            # Keep only top types but maintain original order (sorted by quality)
            expression_transposed = expression_transposed.loc[
                [idx for idx in expression_transposed.index if idx in top_types]
            ]
            # Re-sort by quality (min_score descending within complete/incomplete groups)
            expression_transposed = self._sort_expression_matrix(
                expression_transposed.T, as_types_rows=True,
                input_format='lines_x_types'
            )
            self._vprint(f"   📊 Showing top {top_n_types} types (of {total_types} after filter)")
            # Add filter info to title
            title = f"{title} [Top {top_n_types} types]"
        else:
            self._vprint(f"   📊 Showing all {total_types} types")
        
        # Save viz data CSV (filtered/truncated version)
        viz_csv_path = os.path.join(output_path, 'expression_matrix_viz.csv')
        expression_transposed.to_csv(viz_csv_path)
        self._vprint(f"   💾 Saved viz expression matrix: {viz_csv_path}")
        
        # Handle empty dataframe gracefully
        if expression_transposed.empty:
            self._vprint(f"   ⚠️ No types to visualize after filtering - skipping heatmap creation")
            return ""
        
        # Resolve color scale preset to Plotly format
        color_scales = {
            'green': [[0, 'rgb(255,255,255)'], [1, 'rgb(14,83,13)']],
            'purple': [[0, 'rgb(255,255,255)'], [1, 'rgb(104,55,164)']],
            'orange': [[0, 'rgb(255,255,255)'], [1, 'rgb(204,102,0)']],
            'blue': [[0, 'rgb(255,255,255)'], [1, 'rgb(31,119,180)']],
            'red': [[0, 'rgb(255,255,255)'], [1, 'rgb(214,39,40)']],
        }
        resolved_color_scale = color_scales.get(color_scale, color_scales['green'])
        
        # Create heatmap file path
        filename = os.path.join(output_path, 'expression_matrix.html')
        
        # Calculate initial dimensions based on matrix size
        # For expression matrices: width ~8px per column, height ~20px per row
        # Minimum: 800x800, Maximum: 2400x4000
        n_rows = len(expression_transposed)
        n_cols = len(expression_transposed.columns)
        init_width = max(800, min(2400, n_cols * 80 + 200))  # 80px per column + margin
        init_height = max(800, min(4000, n_rows * 20 + 200))  # 20px per row + margin
        
        # Create heatmap using VisConnMatInteractive (transposed: types × lines)
        VisConnMatInteractive(
            expression_transposed,
            filename=filename,
            title=title.replace("Lines × Types", "Types × Lines"),
            color_scale=resolved_color_scale,
            showfig=False,
            verbose=self.verbose,
            init_width=init_width,
            init_height=init_height,
            init_clustered=False
        )
        
        self._vprint(f"   📊 Created expression matrix heatmap: {filename}")
        return filename

    def visualize_expression_matrix_merged(
        self,
        expression_df: pd.DataFrame,
        output_path: str,
        queried_types: Optional[List[str]] = None,
        title: str = "Line × Type Expression Matrix (Merged)",
        color_scale: str = 'green',
        top_n_types: int = 100,
        aggregation: str = 'max',
        type_filter: Optional[Dict[str, Union[str, List[str]]]] = None,
        use_auto_type_mapping: bool = True,
    ) -> str:
        """
        Visualize the expression matrix with same neuron types merged across datasets.
        
        Types from different datasets (e.g., 'MCNS_aMe12', 'FAFB_aMe12', 'HEMI_aMe12')
        are merged into a single row (e.g., 'aMe12') using the specified aggregation.
        
        When use_auto_type_mapping=True (default), types with different names across
        datasets are also merged (e.g., 'MCNS_MeVPLo2' and 'FAFB_MTe07' → 'MeVPLo2').
        
        Parameters
        ----------
        expression_df : pd.DataFrame
            Expression matrix from _calculate_mutual_information (Lines × Types format).
            Column names should be prefixed with dataset abbreviation (e.g., 'MCNS_aMe12').
        output_path : str
            Directory to save the heatmap HTML file.
        queried_types : list of str, optional
            Queried types to highlight in the matrix.
        title : str
            Title for the heatmap.
        color_scale : str
            Color scale preset.
        top_n_types : int
            Maximum number of types to display (default: 100).
        aggregation : str
            How to aggregate scores across datasets: 'max' (default), 'mean', 'sum'.
        type_filter : dict, optional
            Filter neuron types by name pattern for visualization.
            Uses a dict with filter type as key and pattern(s) as value.
            
            Filter types:
            - 'contains': Match types containing the pattern
            - 'startswith': Match types starting with the pattern
            - 'endswith': Match types ending with the pattern
            - 'regex': Match types matching the regex pattern
            
            Values: str or list of str patterns
            Multiple patterns within same key use OR logic.
            Multiple keys use AND logic.
            The rank r{N} will be preserved from the original ranking before filtering.
            
            Examples:
            - type_filter = {'contains': 'DN'}  # Types containing 'DN'
            - type_filter = {'startswith': ['DN', 'IN']}  # Types starting with 'DN' or 'IN'
            - type_filter = {'endswith': '_R'}  # Types ending with '_R'
            - type_filter = {'regex': r'DN[a-z]\\d+'}  # Types matching regex pattern
            - type_filter = {'contains': 'DN', 'endswith': '_R'}  # Both conditions (AND)
            
            Set to None or {} to disable filtering (visualize all top types).
            Does not affect the full data CSV - only the viz CSV and heatmap.
        use_auto_type_mapping : bool, default True
            Use CrossDatasetTypeMapper to merge types that have different names
            across datasets (e.g., 'MeVPLo2' in male-cns = 'MTe07' in FAFB).
            
        Returns
        -------
        str
            Path to the created heatmap file.
        """
        try:
            from vispath_pkg import VisConnMatInteractive
        except ImportError:
            # Try to add vispath-subproject/src to path
            import sys
            from pathlib import Path
            
            # Find repo root (assuming we are in src/)
            current_file = Path(__file__).resolve()
            repo_root = current_file.parent.parent
            vispath_src = repo_root / 'vispath-subproject' / 'src'
            
            if vispath_src.exists() and str(vispath_src) not in sys.path:
                sys.path.append(str(vispath_src))
            
            try:
                from vispath_pkg import VisConnMatInteractive
            except ImportError:
                self._vprint("   ⚠️ Could not import VisConnMatInteractive from vispath_pkg")
                return ""
        
        os.makedirs(output_path, exist_ok=True)
        
        # Step 1: Build merge mapping
        # Try to use auto type mapping for smarter merging
        merge_mapping = {}
        type_mapper = None
        aggregated_types = []  # Track types that are being aggregated due to N-to-1/1-to-N
        
        if use_auto_type_mapping and HAS_TYPE_MAPPER and get_type_mapper is not None:
            try:
                type_mapper = get_type_mapper()
                if type_mapper and type_mapper._loaded:
                    merge_mapping = type_mapper.get_merge_mapping_for_types(
                        list(expression_df.columns),
                        verbose=self.verbose,
                    )
            except Exception as e:
                self._vprint(f"   ⚠️ Auto type mapping not available: {e}")
        
        # Fallback: simple prefix stripping if no mapper available
        
        # Build mapping: base_type -> list of prefixed columns
        base_type_to_columns = {}
        for col in expression_df.columns:
            base_type = _extract_base_type(col, merge_mapping)
            if base_type not in base_type_to_columns:
                base_type_to_columns[base_type] = []
            base_type_to_columns[base_type].append(col)
        
        # Track types that got merged from different original names
        for base_type, columns in base_type_to_columns.items():
            # Check if columns have different original base names (after prefix)
            original_names = set()
            for col in columns:
                parts = col.split('_', 1)
                if len(parts) > 1:
                    original_names.add(parts[1])
            if len(original_names) > 1:
                aggregated_types.append((base_type, original_names, columns))
        
        if aggregated_types and self.verbose:
            self._vprint(f"   🔗 Auto type mapping merged {len(aggregated_types)} type groups:")
            for base_type, orig_names, cols in aggregated_types[:5]:
                self._vprint(f"      • {sorted(orig_names)} → '{base_type}'")
            if len(aggregated_types) > 5:
                self._vprint(f"      ... and {len(aggregated_types) - 5} more")
            self._vprint(f"   💡 To disable merging, set use_auto_type_mapping=False")
        
        # Step 2: Merge columns by base type using specified aggregation
        merged_data = {}
        for base_type, columns in base_type_to_columns.items():
            subset = expression_df[columns]
            if aggregation == 'max':
                merged_data[base_type] = subset.max(axis=1)
            elif aggregation == 'mean':
                # Mean of non-zero values only
                merged_data[base_type] = subset.apply(
                    lambda row: row[row > 0].mean() if (row > 0).any() else 0.0, axis=1
                )
            elif aggregation == 'sum':
                merged_data[base_type] = subset.sum(axis=1)
            else:
                merged_data[base_type] = subset.max(axis=1)  # Default to max
        
        merged_df = pd.DataFrame(merged_data)
        
        # Log merge statistics
        original_types = len(expression_df.columns)
        merged_types = len(merged_df.columns)
        self._vprint(f"   📊 Merged {original_types} prefixed types → {merged_types} base types (aggregation: {aggregation})")
        
        # Step 3: Transpose to get types as rows
        expression_transposed = merged_df.T.copy()
        
        # Step 4: Sort by co-labeling quality (same logic as original)
        n_lines = len(expression_transposed.columns)
        nonzero_count = (expression_transposed > 0).sum(axis=1)
        total_score = expression_transposed.sum(axis=1)
        min_score_col = expression_transposed.min(axis=1)
        
        is_complete = (nonzero_count == n_lines).astype(int)
        sort_df = pd.DataFrame({
            'is_complete': -is_complete,
            'min_score': -min_score_col,
            'nonzero_count': -nonzero_count,
            'total_score': -total_score
        }, index=expression_transposed.index)
        sorted_index = sort_df.sort_values(['is_complete', 'min_score', 'nonzero_count', 'total_score']).index
        expression_transposed = expression_transposed.loc[sorted_index]
        
        # Save FULL merged data CSV (before any filtering/truncation)
        full_csv_path = os.path.join(output_path, 'expression_matrix_merged.csv')
        expression_transposed.to_csv(full_csv_path)
        self._vprint(f"   💾 Saved full merged expression matrix: {full_csv_path}")
        
        # Step 5: Apply type_filter if specified
        total_types_before_filter = len(expression_transposed)
        if type_filter:
            # Create ranked type list for filtering (using merged type names)
            ranked_types = [(i, name) for i, name in enumerate(expression_transposed.index, start=1)]
            filtered_types = self._apply_type_filter(ranked_types, type_filter)
            filtered_type_names = [name for _, name in filtered_types]
            expression_transposed = expression_transposed.loc[
                [idx for idx in expression_transposed.index if idx in filtered_type_names]
            ]
            self._vprint(f"   🔍 Type filter applied: {len(expression_transposed)}/{total_types_before_filter} merged types match")
        
        # Step 6: Limit to top N types
        total_types = len(expression_transposed)
        if total_types > top_n_types:
            type_totals = expression_transposed.sum(axis=1)
            top_types = set(type_totals.nlargest(top_n_types).index)
            expression_transposed = expression_transposed.loc[
                [idx for idx in expression_transposed.index if idx in top_types]
            ]
            # Re-sort after filtering
            n_lines = len(expression_transposed.columns)
            nonzero_count = (expression_transposed > 0).sum(axis=1)
            total_score = expression_transposed.sum(axis=1)
            min_score_col = expression_transposed.min(axis=1)
            is_complete = (nonzero_count == n_lines).astype(int)
            sort_df = pd.DataFrame({
                'is_complete': -is_complete,
                'min_score': -min_score_col,
                'nonzero_count': -nonzero_count,
                'total_score': -total_score
            }, index=expression_transposed.index)
            sorted_index = sort_df.sort_values(['is_complete', 'min_score', 'nonzero_count', 'total_score']).index
            expression_transposed = expression_transposed.loc[sorted_index]
            
            self._vprint(f"   📊 Showing top {top_n_types} merged types (of {total_types} after filter)")
            title = f"{title} [Top {top_n_types} types]"
        else:
            self._vprint(f"   📊 Showing all {total_types} merged types")
        
        # Save viz data CSV (filtered/truncated version)
        viz_csv_path = os.path.join(output_path, 'expression_matrix_merged_viz.csv')
        expression_transposed.to_csv(viz_csv_path)
        self._vprint(f"   💾 Saved viz merged expression matrix: {viz_csv_path}")
        
        # Handle empty dataframe gracefully
        if expression_transposed.empty:
            self._vprint(f"   ⚠️ No types to visualize after filtering - skipping merged heatmap creation")
            return ""
        
        # Step 7: Resolve color scale
        color_scales = {
            'green': [[0, 'rgb(255,255,255)'], [1, 'rgb(14,83,13)']],
            'purple': [[0, 'rgb(255,255,255)'], [1, 'rgb(104,55,164)']],
            'orange': [[0, 'rgb(255,255,255)'], [1, 'rgb(204,102,0)']],
            'blue': [[0, 'rgb(255,255,255)'], [1, 'rgb(31,119,180)']],
            'red': [[0, 'rgb(255,255,255)'], [1, 'rgb(214,39,40)']],
        }
        resolved_color_scale = color_scales.get(color_scale, color_scales['green'])
        
        # Step 7: Create heatmap
        filename = os.path.join(output_path, 'expression_matrix_merged.html')
        
        n_rows = len(expression_transposed)
        n_cols = len(expression_transposed.columns)
        init_width = max(800, min(2400, n_cols * 80 + 200))
        init_height = max(800, min(4000, n_rows * 20 + 200))
        
        VisConnMatInteractive(
            expression_transposed,
            filename=filename,
            title=title.replace("Lines × Types", "Types × Lines"),
            color_scale=resolved_color_scale,
            showfig=False,
            verbose=self.verbose,
            init_width=init_width,
            init_height=init_height,
            init_clustered=False
        )
        
        self._vprint(f"   📊 Created merged expression matrix heatmap: {filename}")
        return filename

    def visualize_labeling_distribution(
        self,
        data: pd.DataFrame,
        output_path: str,
        score_column: str = 'score',
        label_column: str = 'type',
        title: str = "Labeling Distribution",
        color: str = '#1f77b4',
        filename: str = 'labeling_distribution.html',
        show_threshold: Optional[float] = None,
        group_by: Optional[str] = None
    ) -> str:
        """
        Visualize labeling distribution as a mountain-shaped histogram.
        
        The plot shows scores with highest values in the center, descending
        symmetrically to both sides (like a mountain/pyramid shape).
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with score and label columns.
        output_path : str
            Directory to save the HTML file.
        score_column : str
            Column name for scores. Default: 'score'
        label_column : str
            Column name for labels (types/neurons). Default: 'type'
        title : str
            Plot title. Default: "Labeling Distribution"
        color : str
            Bar color. Default: '#1f77b4' (Category10 blue)
        filename : str
            Output filename. Default: 'labeling_distribution.html'
        show_threshold : float, optional
            If provided, draw a horizontal threshold line.
        group_by : str, optional
            If provided, group data by this column and create subplots.
            
        Returns
        -------
        str
            Path to the created HTML file.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        os.makedirs(output_path, exist_ok=True)
        
        if data.empty or score_column not in data.columns:
            self._vprint(f"   ⚠️ No data for labeling distribution")
            return ""
        
        # Category10 palette (same as bokeh.palettes.Category10)
        CATEGORY10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        full_path = os.path.join(output_path, filename)
        
        if group_by and group_by in data.columns:
            groups = data[group_by].dropna().unique()
            n_groups = len(groups)
            
            if n_groups == 0:
                return ""
            
            # Use 2 columns layout
            n_cols = 2
            n_rows = (n_groups + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=[f"{g}" for g in groups],
                vertical_spacing=0.1,
                horizontal_spacing=0.08
            )
            
            for idx, group in enumerate(groups):
                row = idx // n_cols + 1
                col = idx % n_cols + 1
                group_data = data[data[group_by] == group].copy()
                
                if label_column in group_data.columns:
                    agg_data = group_data.groupby(label_column)[score_column].max().reset_index()
                else:
                    agg_data = group_data[[score_column]].copy()
                    agg_data[label_column] = range(len(agg_data))
                
                scores = agg_data[score_column].tolist()
                labels = agg_data[label_column].tolist()
                
                if len(scores) > 0:
                    mountain_scores, mountain_labels = _create_mountain_order(scores, labels)
                    
                    fig.add_trace(
                        go.Bar(
                            x=list(range(len(mountain_scores))),
                            y=mountain_scores,
                            marker_color=CATEGORY10[idx % len(CATEGORY10)],
                            name=str(group),
                            hovertemplate=f'<b>%{{customdata}}</b><br>Score: %{{y:,.0f}}<extra>{group}</extra>',
                            customdata=mountain_labels,
                            width=1.0  # No gap between bars
                        ),
                        row=row, col=col
                    )
                    
                    if show_threshold:
                        fig.add_hline(
                            y=show_threshold, 
                            line_dash="dash", 
                            line_color="red",
                            row=row, col=col
                        )
                    
                    # Clean axis style
                    fig.update_xaxes(
                        showticklabels=False, 
                        showgrid=False, 
                        zeroline=False,
                        showline=False,
                        row=row, col=col
                    )
                    fig.update_yaxes(
                        showgrid=False, 
                        zeroline=False,
                        showline=False,
                        row=row, col=col
                    )
            
            fig.update_layout(
                title=dict(text=title, x=0.5),
                height=250 * n_rows,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                bargap=0
            )
        else:
            if label_column in data.columns:
                agg_data = data.groupby(label_column)[score_column].max().reset_index()
            else:
                agg_data = data[[score_column]].copy()
                agg_data[label_column] = range(len(agg_data))
            
            scores = agg_data[score_column].tolist()
            labels = agg_data[label_column].tolist()
            
            if len(scores) == 0:
                return ""
            
            mountain_scores, mountain_labels = _create_mountain_order(scores, labels)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=list(range(len(mountain_scores))),
                y=mountain_scores,
                marker_color=color,
                hovertemplate='<b>%{customdata}</b><br>Score: %{y:,.0f}<extra></extra>',
                customdata=mountain_labels,
                width=1.0
            ))
            
            if show_threshold:
                fig.add_hline(
                    y=show_threshold, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Threshold: {show_threshold:,.0f}"
                )
            
            fig.update_layout(
                title=dict(text=title, x=0.5),
                xaxis_title="Neurons/Types (sorted by score, highest in center)",
                yaxis_title="Score",
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False),
                yaxis=dict(showgrid=False, zeroline=False, showline=False),
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                bargap=0
            )
        
        fig.write_html(full_path)
        self._vprint(f"   📊 Created labeling distribution: {full_path}")
        return full_path

    def visualize_colabeling_distribution(
        self,
        line_neurons_dict: Dict[str, pd.DataFrame],
        output_path: str,
        min_score: float = 0.0,
        title: str = "Co-Labeling Score Distribution"
    ) -> Tuple[str, str]:
        """
        Visualize labeling distribution for multiple lines (co-labeling analysis).
        
        Creates two multi-panel plots:
        1. By neuron: individual neuron scores for each line
        2. By type: aggregated type scores for each line
        
        Parameters
        ----------
        line_neurons_dict : dict
            Dictionary mapping line names to neurons DataFrames.
        output_path : str
            Directory to save the HTML files.
        min_score : float
            Minimum score threshold to highlight. Default: 0.0
        title : str
            Plot title.
            
        Returns
        -------
        tuple of str
            Paths to the created HTML files (by_neuron, by_type).
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        os.makedirs(output_path, exist_ok=True)
        
        if not line_neurons_dict:
            return "", ""
        
        # Filter out empty DataFrames - keep all data for saving
        valid_lines_all = {k: v for k, v in line_neurons_dict.items() if not v.empty}
        if not valid_lines_all:
            return "", ""
        
        # Create filtered version for visualization (apply min_score filter if score column exists)
        valid_lines = {}
        for k, df in valid_lines_all.items():
            if min_score > 0 and 'score' in df.columns:
                # Use _passes_min_score column if available (from _calculate_mutual_information)
                if '_passes_min_score' in df.columns:
                    filtered_df = df[df['_passes_min_score']].copy()
                else:
                    filtered_df = df[df['score'] >= min_score].copy()
                if not filtered_df.empty:
                    valid_lines[k] = filtered_df
            else:
                valid_lines[k] = df
        
        # If all data was filtered out, use all data for visualization
        if not valid_lines:
            valid_lines = valid_lines_all
        
        lines = list(valid_lines.keys())
        n_lines = len(lines)
        
        # Category10 palette (same as bokeh.palettes.Category10)
        CATEGORY10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        def create_distribution_plot(by_type: bool, filename: str) -> str:
            """Create distribution plot (by neuron or by type).
            
            Each line gets its own line-specific mountain shape distribution.
            
            Key features:
            - c_line: count of items above min_score threshold for each line
            - If c_line < 100, expand to top-100 items (get t_line = score at expansion boundary)
            - X-axis limited by max(c_line) across all INDIVIDUAL lines (not combined)
            - Y-axis: 0.9 * t_line to max_score for each subplot
            - Peak is ALWAYS centered by padding x-axis
            """
            # Use 2 columns layout, with combined as last panel
            n_panels = n_lines + 1  # lines + combined
            n_cols = 2
            n_rows = (n_panels + n_cols - 1) // n_cols
            
            subplot_titles = [f"{line}" for line in lines]
            subplot_titles.append("Combined (avg)" if by_type else "All Neurons")
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=subplot_titles,
                vertical_spacing=0.08,
                horizontal_spacing=0.06
            )
            
            # Expansion target when items above threshold < 100
            EXPANSION_TARGET = 100
            
            # Pre-calculate all data and per-line statistics
            line_all_data = {}  # Store all data per line (sorted by score desc)
            line_max_scores = {}  # Store max score per line
            line_c_values = {}  # c_line: count of items above threshold
            line_t_values = {}  # t_line: threshold for y-axis (score at expansion boundary)
            line_display_counts = {}  # Number of items to display per line
            
            for line, df in valid_lines.items():
                # Get corresponding unfiltered data from valid_lines_all
                df_all = valid_lines_all.get(line, df)
                
                if by_type and 'type' in df_all.columns:
                    type_scores = df_all.groupby('type')['score'].max()
                    sorted_scores = type_scores.sort_values(ascending=False)
                    line_all_data[line] = sorted_scores
                    line_max_scores[line] = sorted_scores.max() if len(sorted_scores) > 0 else 50000
                    
                    # Calculate c_line (count above threshold)
                    c_line = (sorted_scores >= min_score).sum() if min_score > 0 else len(sorted_scores)
                    line_c_values[line] = c_line
                    
                    # Determine display count and t_line
                    if c_line < EXPANSION_TARGET:
                        # Expand to top-100 (or all if < 100 items available)
                        display_count = min(EXPANSION_TARGET, len(sorted_scores))
                        # t_line is the score at the expansion boundary
                        if display_count > 0:
                            t_line = sorted_scores.iloc[display_count - 1] if display_count <= len(sorted_scores) else sorted_scores.iloc[-1]
                        else:
                            t_line = min_score if min_score > 0 else 0
                    else:
                        # Use only items above threshold
                        display_count = c_line
                        t_line = min_score if min_score > 0 else 0
                    
                    line_display_counts[line] = display_count
                    line_t_values[line] = t_line
                else:
                    sorted_df = df_all.sort_values('score', ascending=False)
                    line_all_data[line] = sorted_df
                    line_max_scores[line] = sorted_df['score'].max() if len(sorted_df) > 0 else 50000
                    
                    # Calculate c_line (count above threshold)
                    c_line = (sorted_df['score'] >= min_score).sum() if min_score > 0 else len(sorted_df)
                    line_c_values[line] = c_line
                    
                    # Determine display count and t_line
                    if c_line < EXPANSION_TARGET:
                        display_count = min(EXPANSION_TARGET, len(sorted_df))
                        if display_count > 0:
                            t_line = sorted_df['score'].iloc[display_count - 1] if display_count <= len(sorted_df) else sorted_df['score'].iloc[-1]
                        else:
                            t_line = min_score if min_score > 0 else 0
                    else:
                        display_count = c_line
                        t_line = min_score if min_score > 0 else 0
                    
                    line_display_counts[line] = display_count
                    line_t_values[line] = t_line
            
            # X-axis range is limited by max(c_line) across individual lines
            # This ensures the combined plot doesn't inflate the x-axis
            max_c_line = max(line_c_values.values()) if line_c_values else EXPANSION_TARGET
            # But we need at least the display count to show expanded items
            max_display_count = max(line_display_counts.values()) if line_display_counts else EXPANSION_TARGET
            global_x_range = max(max_c_line, max_display_count)
            
            # Plot each line with its OWN mountain shape
            for idx, line in enumerate(lines):
                row = idx // n_cols + 1
                col = idx % n_cols + 1
                
                display_count = line_display_counts.get(line, EXPANSION_TARGET)
                t_line = line_t_values.get(line, min_score)
                
                if by_type:
                    # Get type scores for this line (sorted by score desc)
                    type_scores = line_all_data.get(line, pd.Series(dtype=float))
                    
                    # Take display_count items
                    top_types = type_scores.head(display_count)
                    
                    scores = top_types.values.tolist()
                    type_names = top_types.index.tolist()
                    
                    hover_texts = []
                    for type_name, score in zip(type_names, scores):
                        if score >= min_score:
                            hover_texts.append(f"{type_name}<br>Score: {score:,.0f}")
                        else:
                            hover_texts.append(f"{type_name}<br>Score: {score:,.0f}<br>(below threshold)")
                    
                    mountain_scores, _ = _create_mountain_order(scores, type_names)
                    _, mountain_hovers = _create_mountain_order(scores, hover_texts)
                else:
                    # Get all neurons for this line (sorted by score desc)
                    df_all = line_all_data.get(line, pd.DataFrame())
                    
                    # Take display_count items
                    top_neurons = df_all.head(display_count)
                    
                    scores = top_neurons['score'].tolist()
                    if 'bodyId' in top_neurons.columns:
                        labels = top_neurons['bodyId'].astype(str).tolist()
                    elif 'type' in top_neurons.columns:
                        labels = top_neurons['type'].tolist()
                    else:
                        labels = [f"neuron_{i}" for i in range(len(scores))]
                    
                    hover_texts = []
                    for _, row_data in top_neurons.iterrows():
                        body_id = row_data.get('bodyId', 'Unknown')
                        neuron_type = row_data.get('type', 'Unknown')
                        dataset = row_data.get('dataset', 'Unknown')
                        score = row_data.get('score', 0)
                        suffix = "<br>(below threshold)" if score < min_score else ""
                        hover_texts.append(f"bodyId: {body_id}<br>Type: {neuron_type}<br>Dataset: {dataset}{suffix}")
                    
                    mountain_scores, _ = _create_mountain_order(scores, labels)
                    _, mountain_hovers = _create_mountain_order(scores, hover_texts)
                
                if len(mountain_scores) > 0:
                    bar_color = CATEGORY10[idx % len(CATEGORY10)]
                    
                    # Center the peak by calculating offset
                    actual_n_items = len(mountain_scores)
                    offset = (global_x_range - actual_n_items) // 2
                    x_positions = [i + offset for i in range(actual_n_items)]
                    
                    fig.add_trace(
                        go.Bar(
                            x=x_positions,
                            y=mountain_scores,
                            marker=dict(
                                color=bar_color,
                                line=dict(width=0)
                            ),
                            name=line,
                            hovertemplate=f'<b>%{{customdata}}</b><extra>{line}</extra>',
                            customdata=mountain_hovers,
                            width=1.0
                        ),
                        row=row, col=col
                    )
                    
                    if min_score > 0:
                        fig.add_hline(
                            y=min_score, 
                            line_dash="dash", 
                            line_color="red",
                            row=row, col=col
                        )
                    
                    # Y-axis: 0.9 * t_line to max_score * 1.1
                    y_min_subplot = t_line * 0.9 if t_line > 0 else 0
                    y_max_subplot = line_max_scores.get(line, 50000) * 1.1
                    
                    fig.update_xaxes(
                        showticklabels=False, 
                        showgrid=False, 
                        zeroline=False,
                        showline=False,
                        range=[-0.5, global_x_range - 0.5],
                        row=row, col=col
                    )
                    fig.update_yaxes(
                        showgrid=False, 
                        zeroline=False,
                        showline=False,
                        range=[y_min_subplot, y_max_subplot],
                        row=row, col=col
                    )
            
            # Combined plot
            combined_idx = n_lines
            combined_row = combined_idx // n_cols + 1
            combined_col = combined_idx % n_cols + 1
            
            if by_type:
                # Collect all types across all lines with their average scores
                all_types_scores = {}  # type -> list of scores
                for line, type_scores in line_all_data.items():
                    if isinstance(type_scores, pd.Series):
                        for type_name, score in type_scores.items():
                            if type_name not in all_types_scores:
                                all_types_scores[type_name] = []
                            all_types_scores[type_name].append(score)
                
                # Calculate average score per type
                type_avg_scores = {t: sum(s) / len(s) for t, s in all_types_scores.items()}
                sorted_types = sorted(type_avg_scores.items(), key=lambda x: x[1], reverse=True)
                
                # Combined plot uses global_x_range (based on max(c_line))
                combined_labels = [t[0] for t in sorted_types[:global_x_range]]
                combined_scores = [t[1] for t in sorted_types[:global_x_range]]
                combined_hovers = [f"{t}<br>Avg: {s:,.0f}" for t, s in sorted_types[:global_x_range]]
                
                mountain_scores, _ = _create_mountain_order(combined_scores, combined_labels)
                _, mountain_hovers = _create_mountain_order(combined_scores, combined_hovers)
            else:
                # All neurons from all lines
                combined_scores = []
                combined_labels = []
                combined_hovers = []
                for line, df_all in line_all_data.items():
                    if isinstance(df_all, pd.DataFrame):
                        # Only take items up to display_count for this line
                        display_count = line_display_counts.get(line, EXPANSION_TARGET)
                        df_subset = df_all.head(display_count)
                        combined_scores.extend(df_subset['score'].tolist())
                        if 'bodyId' in df_subset.columns:
                            combined_labels.extend(df_subset['bodyId'].astype(str).tolist())
                        else:
                            combined_labels.extend([f"{line}_{i}" for i in range(len(df_subset))])
                        
                        for _, row_data in df_subset.iterrows():
                            body_id = row_data.get('bodyId', 'Unknown')
                            neuron_type = row_data.get('type', 'Unknown')
                            dataset = row_data.get('dataset', 'Unknown')
                            combined_hovers.append(f"bodyId: {body_id}<br>Type: {neuron_type}<br>Dataset: {dataset}<br>Line: {line}")
                
                # Sort and take top items limited by global_x_range
                if combined_scores:
                    sorted_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)
                    combined_target = min(global_x_range * n_lines, len(combined_scores))
                    combined_scores = [combined_scores[i] for i in sorted_indices[:combined_target]]
                    combined_labels = [combined_labels[i] for i in sorted_indices[:combined_target]]
                    combined_hovers = [combined_hovers[i] for i in sorted_indices[:combined_target]]
                
                mountain_scores, _ = _create_mountain_order(combined_scores, combined_labels)
                _, mountain_hovers = _create_mountain_order(combined_scores, combined_hovers)
            
            if mountain_scores:
                # Combined plot y-axis: use min_score * 0.9 as base
                combined_max = max(mountain_scores) if mountain_scores else 50000
                y_max_combined = combined_max * 1.1
                y_min_combined = min_score * 0.9 if min_score > 0 else 0
                
                # Center the combined plot
                actual_combined_items = len(mountain_scores)
                # Combined plot x-range should match individual plots (global_x_range)
                offset_combined = (global_x_range - actual_combined_items) // 2 if actual_combined_items < global_x_range else 0
                x_positions_combined = [i + offset_combined for i in range(actual_combined_items)]
                
                fig.add_trace(
                    go.Bar(
                        x=x_positions_combined,
                        y=mountain_scores,
                        marker=dict(
                            color='#17becf',
                            line=dict(width=0)
                        ),
                        name='Combined',
                        hovertemplate='<b>%{customdata}</b><br>Score: %{y:,.0f}<extra>Combined</extra>',
                        customdata=mountain_hovers,
                        width=1.0
                    ),
                    row=combined_row, col=combined_col
                )
                
                if min_score > 0:
                    fig.add_hline(
                        y=min_score, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text=f"Threshold: {min_score:,.0f}",
                        row=combined_row, col=combined_col
                    )
                
                # Combined plot x-range matches individual line plots
                combined_x_max = max(global_x_range, actual_combined_items)
                
                fig.update_xaxes(
                    showticklabels=False, 
                    showgrid=False, 
                    zeroline=False,
                    showline=False,
                    range=[-0.5, combined_x_max - 0.5],
                    row=combined_row, col=combined_col
                )
                fig.update_yaxes(
                    showgrid=False, 
                    zeroline=False,
                    showline=False,
                    range=[y_min_combined, y_max_combined],
                    row=combined_row, col=combined_col
                )
            
            suffix = "by Type" if by_type else "by Neuron"
            fig.update_layout(
                title=dict(text=f"{title} ({suffix})", x=0.5),
                height=250 * n_rows,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                bargap=0
            )
            
            full_path = os.path.join(output_path, filename)
            fig.write_html(full_path)
            return full_path
        
        # Create both visualizations
        path_by_type = create_distribution_plot(by_type=True, filename='labeling_distribution_by_type.html')
        path_by_neuron = create_distribution_plot(by_type=False, filename='labeling_distribution_by_neuron.html')
        
        # Create stacked mountain-shaped histogram
        path_stacked = self._create_stacked_mountain_plot(
            line_neurons_dict=valid_lines,
            output_path=output_path,
            min_score=min_score,
            title=title,
            colors=CATEGORY10
        )
        
        # Save ALL visualization data as CSV (unfiltered, for reproducibility)
        # Use valid_lines_all which contains all data regardless of min_score filter
        self._save_distribution_data(
            line_neurons_dict=valid_lines_all,
            output_path=output_path
        )
        
        self._vprint(f"   📊 Created labeling distribution (by type): {path_by_type}")
        self._vprint(f"   📊 Created labeling distribution (by neuron): {path_by_neuron}")
        self._vprint(f"   📊 Created labeling distribution (stacked): {path_stacked}")
        
        return path_by_type, path_by_neuron
    
    def _save_distribution_data(
        self,
        line_neurons_dict: Dict[str, pd.DataFrame],
        output_path: str
    ) -> None:
        """
        Save the data used for distribution visualizations as CSV files.
        
        Saves two CSV files:
        1. distribution_data_by_neuron.csv - All neurons with scores per line
        2. distribution_data_by_type.csv - Aggregated type scores per line
        
        Parameters
        ----------
        line_neurons_dict : dict
            Dictionary mapping line names to neurons DataFrames.
        output_path : str
            Directory to save the CSV files.
        """
        # Save by neuron (all data)
        all_neurons = []
        for line, df in line_neurons_dict.items():
            if not df.empty:
                df_copy = df.copy()
                df_copy['source_line'] = line
                all_neurons.append(df_copy)
        
        if all_neurons:
            combined_neurons = pd.concat(all_neurons, ignore_index=True)
            neurons_path = os.path.join(output_path, 'distribution_data_by_neuron.csv')
            combined_neurons.to_csv(neurons_path, index=False)
        
        # Save by type (aggregated max score per type per line)
        type_data = []
        for line, df in line_neurons_dict.items():
            if 'type' in df.columns and not df.empty:
                type_scores = df.groupby('type')['score'].max().reset_index()
                type_scores['source_line'] = line
                # Also get dataset info if available
                if 'dataset' in df.columns:
                    type_datasets = df.groupby('type')['dataset'].first().reset_index()
                    type_scores = type_scores.merge(type_datasets, on='type', how='left')
                type_data.append(type_scores)
        
        if type_data:
            combined_types = pd.concat(type_data, ignore_index=True)
            types_path = os.path.join(output_path, 'distribution_data_by_type.csv')
            combined_types.to_csv(types_path, index=False)
    
    def _create_stacked_mountain_plot(
        self,
        line_neurons_dict: Dict[str, pd.DataFrame],
        output_path: str,
        min_score: float = 0.0,
        title: str = "Co-Labeling Score Distribution",
        colors: List[str] = None
    ) -> str:
        """
        Create a stacked mountain-shaped histogram showing all lines overlaid.
        
        Each line's scores are shown as bars stacked on top of each other,
        with transparency to see overlapping regions.
        
        Parameters
        ----------
        line_neurons_dict : dict
            Dictionary mapping line names to neurons DataFrames.
        output_path : str
            Directory to save the HTML file.
        min_score : float
            Minimum score threshold to highlight.
        title : str
            Plot title.
        colors : list
            List of colors to use for each line.
            
        Returns
        -------
        str
            Path to the created HTML file.
        """
        import plotly.graph_objects as go
        
        if colors is None:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        lines = list(line_neurons_dict.keys())
        n_lines = len(lines)
        
        # Collect all types across all lines with their scores per line
        all_types_data = {}  # type_name -> {line_name: max_score}
        
        for line, df in line_neurons_dict.items():
            if 'type' in df.columns:
                type_scores = df.groupby('type')['score'].max()
                for type_name, score in type_scores.items():
                    if type_name not in all_types_data:
                        all_types_data[type_name] = {}
                    all_types_data[type_name][line] = score
        
        if not all_types_data:
            return ""
        
        # Calculate total score for each type (sum across all lines)
        type_totals = {t: sum(scores.values()) for t, scores in all_types_data.items()}
        
        # Sort types by total score (descending) for mountain shape
        sorted_types = sorted(type_totals.keys(), key=lambda t: type_totals[t], reverse=True)
        
        # Create mountain shape (highest in middle)
        n_types = len(sorted_types)
        mountain_types = [''] * n_types
        left = n_types // 2 - 1 if n_types % 2 == 0 else n_types // 2
        right = n_types // 2 if n_types % 2 == 0 else n_types // 2
        
        for i, type_name in enumerate(sorted_types):
            if i == 0:
                mountain_types[n_types // 2] = type_name
            elif i % 2 == 1:
                mountain_types[left] = type_name
                left -= 1
            else:
                mountain_types[right] = type_name
                right += 1
        
        # Create figure with stacked bars
        fig = go.Figure()
        
        # Add traces for each line (in reverse order so first line is on top visually)
        for idx, line in enumerate(reversed(lines)):
            line_idx = len(lines) - 1 - idx
            scores = []
            hover_texts = []
            
            for type_name in mountain_types:
                score = all_types_data.get(type_name, {}).get(line, 0)
                scores.append(score)
                hover_texts.append(f"{type_name}<br>{line}: {score:,.0f}")
            
            # Convert hex color to rgba with alpha
            hex_color = colors[line_idx % len(colors)]
            # Parse hex to RGB
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            rgba_color = f'rgba({r}, {g}, {b}, 0.7)'
            
            fig.add_trace(
                go.Bar(
                    x=list(range(len(mountain_types))),
                    y=scores,
                    name=line,
                    marker=dict(
                        color=rgba_color,
                        line=dict(width=0)  # No edge
                    ),
                    hovertemplate='<b>%{customdata}</b><extra></extra>',
                    customdata=hover_texts,
                    width=1.0
                )
            )
        
        # Add threshold line if specified
        if min_score > 0:
            fig.add_hline(
                y=min_score,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {min_score:,.0f}",
                annotation_position="top right"
            )
        
        fig.update_layout(
            title=dict(text=f"{title} (Stacked by Type)", x=0.5),
            barmode='stack',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.3)',
                borderwidth=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            bargap=0,
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showline=False,
                title="Types (sorted by total score, mountain-shaped)"
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                title="Score"
            )
        )
        
        full_path = os.path.join(output_path, 'labeling_distribution_stacked.html')
        fig.write_html(full_path)
        return full_path

    def _init_client(self):
        """Initialize the NeuronBridge client with retry logic."""
        import time
        
        max_retries = 5
        # One wait per retry: max_retries - 1 entries, otherwise the fourth
        # failure raises IndexError instead of sleeping for the final retry.
        retry_delays = [2, 5, 10, 15]  # Seconds to wait before each retry
        
        self._vprint("🔌 Initializing NeuronBridge client...")
        
        for attempt in range(max_retries):
            try:
                # Clear any previous client state before retry
                # Use getattr to safely check if _client exists and is not None
                if getattr(self, '_client', None) is not None:
                    try:
                        del self._client
                    except AttributeError:
                        pass
                    import gc
                    gc.collect()
                
                self._client = NBClient()
                # Fix S3 URL version mismatch if needed
                self._fix_store_prefixes()
                self._vprint("  ✓ Client initialized successfully")
                return  # Success
                
            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    wait_time = retry_delays[attempt]
                    self._vprint(f"  ⚠️  Initialization failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
                    self._vprint(f"  ⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"Failed to initialize NeuronBridge client after {max_retries} attempts: {error_msg}\n"
                        f"Please check your network connection and try again later.\n"
                        f"See TROUBLESHOOTING.md for common solutions."
                    )
    
    def _fix_store_prefixes(self):
        """
        Fix S3 URL version mismatch in the NeuronBridge client.
        
        The config may reference v3_8_0 but data might exist at v3_8_1.
        Try production bucket first, as it's more reliable.
        """
        if hasattr(self._client, 'config') and hasattr(self._client.config, 'stores'):
            stores = self._client.config.stores
            # Stores/prefixes support mapping access in both the bundled
            # client and older upstream-client cache objects.
            if hasattr(stores, 'items'):
                for store_key, store in stores.items():
                    if hasattr(store, 'prefixes') and store.prefixes:
                        # prefixes is a dict like {'CDSResults': 'url', ...}
                        if hasattr(store.prefixes, 'items'):
                            for key, url in store.prefixes.items():
                                if isinstance(url, str):
                                    new_url = url
                                    # Keep production bucket but fix version
                                    if 'v3_8_0' in new_url:
                                        new_url = new_url.replace('v3_8_0', 'v3_8_1')
                                    # Switch dev to prod (prod is more reliable)
                                    if 'data-dev' in new_url:
                                        new_url = new_url.replace('data-dev', 'data-prod')
                                    store.prefixes[key] = new_url
    
    def _parse_library_name(self, library_name: str) -> Tuple[str, str]:
        """
        Parse a NeuronBridge library name into base name and version.
        
        Parameters
        ----------
        library_name : str
            Library name like 'FlyEM_Hemibrain_v1.2.1' or 'FlyEM_MANC_v1.2.1'
            
        Returns
        -------
        Tuple[str, str]
            (base_name, version) e.g. ('FlyEM_Hemibrain', '1.2.1') or ('FlyEM_MANC', '1.2.1')
        """
        if not library_name:
            return ('', '')
        
        # Split on '_v' to get base and version
        if '_v' in library_name:
            parts = library_name.rsplit('_v', 1)
            return (parts[0], parts[1] if len(parts) > 1 else '')
        return (library_name, '')
    
    def _get_dataset_from_library(self, library_name: str) -> Optional[str]:
        """
        Get local dataset folder name from NeuronBridge library name.
        
        Parameters
        ----------
        library_name : str
            NeuronBridge library name (e.g., 'FlyEM_Hemibrain_v1.2.1')
            
        Returns
        -------
        str or None
            Local dataset folder name, or None if not found.
        """
        if not library_name:
            return None
        
        # Direct match
        if library_name in LIBRARY_TO_DATASET:
            return LIBRARY_TO_DATASET[library_name]
        
        # Try to parse and match by base name with different versions
        base_name, version = self._parse_library_name(library_name)
        if base_name and version:
            # Look for any existing mapping with the same base name
            for lib_pattern, dataset in LIBRARY_TO_DATASET.items():
                pattern_base, _ = self._parse_library_name(lib_pattern)
                if pattern_base == base_name:
                    # Found a matching base - try to construct the dataset folder
                    # Convert version to folder format (1.2.1 -> v1_2_1)
                    version_folder = 'v' + version.replace('.', '_')
                    # Extract dataset base name from the mapped folder
                    dataset_base = dataset.rsplit('_v', 1)[0] if '_v' in dataset else dataset.replace('_', '-')
                    new_dataset_folder = f"{dataset_base}_{version_folder}"
                    
                    # Print warning about unmapped library
                    self._vprint(f"  ⚠️  UNMAPPED LIBRARY: '{library_name}'")
                    self._vprint(f"      Attempting auto-mapping to: {new_dataset_folder}")
                    self._vprint(f"      (Based on similar library: {lib_pattern} -> {dataset})")
                    self._vprint(f"      Consider adding this to LIBRARY_TO_DATASET in neuronbridge_finder.py")
                    
                    return new_dataset_folder
        
        # Partial match (try prefixes) - legacy fallback
        for lib_pattern, dataset in LIBRARY_TO_DATASET.items():
            if library_name.startswith(lib_pattern.split('_v')[0]):
                self._vprint(f"  ⚠️  PARTIAL MATCH: '{library_name}' -> {dataset}")
                self._vprint(f"      Consider adding explicit mapping to LIBRARY_TO_DATASET")
                return dataset
        
        # No match found - print explicit warning
        self._vprint(f"  ⚠️  UNKNOWN LIBRARY: '{library_name}'")
        self._vprint(f"      This library is not mapped to any local dataset.")
        self._vprint(f"      To add support, update LIBRARY_TO_DATASET and LIBRARY_TO_DATASET_NAME")
        self._vprint(f"      in src/neuronbridge_finder.py with the appropriate mappings.")
        
        return None
    
    def _get_dataset_name_from_library(self, library_name: str) -> Optional[str]:
        """
        Get human-readable dataset name from NeuronBridge library name.
        
        Parameters
        ----------
        library_name : str
            NeuronBridge library name (e.g., 'FlyEM_Male_CNS_Brain_v0.9')
            
        Returns
        -------
        str or None
            Human-readable dataset name (e.g., 'male-cns:v0.9'), or None if not found.
        """
        if not library_name:
            return None
        
        # Direct match
        if library_name in LIBRARY_TO_DATASET_NAME:
            return LIBRARY_TO_DATASET_NAME[library_name]
        
        # Try to parse and match by base name with different versions
        base_name, version = self._parse_library_name(library_name)
        if base_name and version:
            # Look for any existing mapping with the same base name
            for lib_pattern, dataset_name in LIBRARY_TO_DATASET_NAME.items():
                pattern_base, _ = self._parse_library_name(lib_pattern)
                if pattern_base == base_name:
                    # Found a matching base - construct the dataset name
                    # Extract dataset base from the mapped name (e.g., 'hemibrain' from 'hemibrain:v1.2.1')
                    if ':v' in dataset_name:
                        dataset_base = dataset_name.split(':v')[0]
                        new_dataset_name = f"{dataset_base}:v{version}"
                        return new_dataset_name
                    elif '_v' in dataset_name:
                        # Handle folder-style names
                        dataset_base = dataset_name.rsplit('_v', 1)[0]
                        return f"{dataset_base}:v{version}"
        
        # Partial match (try prefixes) - legacy fallback
        for lib_pattern, dataset_name in LIBRARY_TO_DATASET_NAME.items():
            if library_name.startswith(lib_pattern.split('_v')[0]):
                return dataset_name
        
        return None
    
    def _ensure_datasets_loaded(self, datasets: List[str]) -> List[str]:
        """
        Ensure all required datasets are loaded, pulling from NeuPrint if needed.
        
        This method identifies unique datasets from NeuronBridge results and
        ensures each one is available locally (loading from cache or pulling
        from NeuPrint).
        
        Parameters
        ----------
        datasets : list of str
            List of dataset names (e.g., ['hemibrain:v1.2.1', 'male-cns:v0.9', 'manc:v1.2.1'])
            
        Returns
        -------
        list of str
            List of successfully loaded dataset folder names.
        """
        loaded = []
        
        for dataset in datasets:
            # Convert dataset name to folder format
            dataset_folder = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
            
            # Check if already loaded in cache
            if dataset_folder in self._neuron_dfs:
                loaded.append(dataset_folder)
                continue

            # FAFB/BANC local-release tables are prepared artifacts, not
            # NeuPrint datasets. Prefer the exact Parquet/CSV resolver and
            # never attempt a NeuPrint/FAFB fallback for a missing BANC table.
            if is_local_connectome_dataset(dataset_folder):
                local_df = self._load_neuron_df_for_dataset(dataset_folder)
                if local_df is not None and not local_df.empty:
                    loaded.append(dataset_folder)
                else:
                    self._vprint(
                        f"   ⚠️ Could not load local release dataset: {dataset_folder}"
                    )
                continue
            
            # Try to load from local file
            if self.datasets_path:
                neuron_df_path = os.path.join(
                    self.datasets_path, dataset_folder, f"{dataset_folder}_allneurons_neuron_df.csv"
                )
                
                if os.path.exists(neuron_df_path):
                    try:
                        df = pd.read_csv(neuron_df_path, dtype={'bodyId': str}, low_memory=False)
                        self._neuron_dfs[dataset_folder] = df
                        self._vprint(f"   ✓ Loaded {len(df):,} neurons from {dataset_folder}")
                        loaded.append(dataset_folder)
                        continue
                    except Exception as e:
                        self._vprint(f"   ⚠️ Could not load {dataset_folder}: {e}")
            
            # Try to pull from NeuPrint
            self._vprint(f"   ⏳ {dataset}: not found locally, attempting to pull from NeuPrint...")
            pulled_df = self._pull_and_load_dataset(dataset_folder)
            
            if pulled_df is not None and not pulled_df.empty:
                self._vprint(f"   ✓ Pulled {len(pulled_df):,} neurons for {dataset}")
                loaded.append(dataset_folder)
            else:
                # Try fallback via FindNeuronConnection
                self._vprint(f"   ⏳ Trying FindNeuronConnection fallback for {dataset}...")
                fnc_df = self._fetch_neuron_df_via_fnc(dataset_folder)
                if fnc_df is not None and not fnc_df.empty:
                    self._vprint(f"   ✓ Loaded {len(fnc_df):,} neurons via FNC for {dataset}")
                    loaded.append(dataset_folder)
                else:
                    self._vprint(f"   ⚠️ Could not load dataset: {dataset}")
                    self._vprint(f"      This dataset may not be available in NeuPrint.")
                    self._vprint(f"      Neurons from this dataset will be skipped in visualization.")
        
        return loaded
    
    def _load_neuron_df_for_dataset(self, dataset: str) -> Optional[pd.DataFrame]:
        """
        Load neuron DataFrame for a specific dataset.
        
        Parameters
        ----------
        dataset : str
            Dataset folder name (e.g., 'hemibrain_v1_2_1')
            
        Returns
        -------
        pd.DataFrame or None
            Loaded DataFrame, or None if not found.
        """
        if not dataset:
            return None
        # Check cache
        if dataset in self._neuron_dfs:
            return self._neuron_dfs[dataset]
        
        # Try to load from datasets folder. FAFB/BANC local releases must use
        # their own exact table; in particular BANC must never fall back to
        # the FAFB table or a NeuPrint pull.
        if self.datasets_path:
            if is_local_connectome_dataset(dataset):
                dataset_root = resolve_flywire_dataset_dir(
                    Path(self.datasets_path).parent, dataset
                )
            else:
                dataset_root = Path(self.datasets_path) / dataset
            dataset_root = Path(dataset_root) if dataset_root is not None else None
            dataset_names = []
            if dataset_root is not None:
                dataset_names.append(dataset_root.name)
            if dataset not in dataset_names:
                dataset_names.append(dataset)
            neuron_candidates = [
                dataset_root / f"{name}_allneurons_neuron_df.{suffix}"
                for name in dataset_names
                for suffix in ("parquet", "csv")
            ] if dataset_root is not None else []
            neuron_df_path = next(
                (path for path in neuron_candidates if path.exists()),
                None,
            )
            
            if neuron_df_path is not None:
                try:
                    if str(neuron_df_path).lower().endswith('.parquet'):
                        df = pd.read_parquet(neuron_df_path)
                    else:
                        df = pd.read_csv(
                            neuron_df_path,
                            dtype={'bodyId': 'string'},
                            low_memory=False,
                        )
                    if is_local_connectome_dataset(dataset):
                        normalize_flywire_id_columns(df, ['bodyId', 'root_id'])
                    self._neuron_dfs[dataset] = df
                    # Only print loading message if not suppressing
                    if not self._suppress_loading_msgs:
                        self._vprint(f"  ✓ Loaded {len(df):,} neurons from {dataset}")
                    return df
                except Exception as e:
                    self._vprint(f"  ⚠️ Could not load neuron data for {dataset}: {e}")
            elif is_local_connectome_dataset(dataset):
                self._vprint(
                    f"  ⚠️ No local FAFB/BANC neuron table found for {dataset}; "
                    "skipping this dataset."
                )
                return None
            else:
                # Try to pull dataset using statvis.pull_dataset
                return self._pull_and_load_dataset(dataset)
        
        # Try fallback to FindNeuronConnection
        return self._fetch_neuron_df_via_fnc(dataset)
    
    def _pull_and_load_dataset(self, dataset: str) -> Optional[pd.DataFrame]:
        """
        Pull dataset from NeuPrint and load neuron DataFrame.
        
        Uses statvis.pull_dataset() to fetch data when local files don't exist.
        
        Parameters
        ----------
        dataset : str
            Dataset folder name (e.g., 'hemibrain_v1_2_1')
            
        Returns
        -------
        pd.DataFrame or None
            Loaded DataFrame, or None if failed.
        """
        try:
            # Try relative import first (when imported as package)
            try:
                from .statvis import pull_dataset
            except ImportError:
                # Fall back to absolute import (when running from scripts)
                from statvis import pull_dataset
            
            # Convert folder name to dataset format (hemibrain_v1_2_1 -> hemibrain:v1.2.1)
            dataset_name = self._folder_to_dataset_name(dataset)
            
            self._vprint(f"  ⏳ Dataset not found locally, attempting to pull {dataset_name}...")
            
            # Initialize NeuPrint client for this dataset
            try:
                from neuprint import Client, set_default_client, default_client
                
                # Check if there's an existing default client for the same dataset
                try:
                    existing = default_client()
                    if existing.dataset != dataset_name:
                        # Different dataset, need new client
                        existing = None
                except (RuntimeError, AttributeError):
                    existing = None
                
                if existing is None:
                    # Get token using TokenManager
                    token = self.neuprint_token
                    try:
                        from .utils.token_manager import token_manager
                        token = token_manager.get_token('NEUPRINT_TOKEN', token)
                    except ImportError:
                        try:
                            from src.utils.token_manager import token_manager
                            token = token_manager.get_token('NEUPRINT_TOKEN', token)
                        except ImportError:
                            # Fallback to env vars if TokenManager missing
                            if not token:
                                token = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS', os.environ.get('NEUPRINT_TOKEN', ''))
                    
                    if not token:
                        self._vprint(f"")
                        self._vprint(f"  ⚠️  No NeuPrint token found - cannot pull dataset.")
                        self._vprint(f"  ╔══════════════════════════════════════════════════════════════╗")
                        self._vprint(f"  ║  To enable dataset pulling, provide your NeuPrint token:    ║")
                        self._vprint(f"  ║                                                              ║")
                        self._vprint(f"  ║  Option 1: Pass directly to NeuronBridgeFinder:             ║")
                        self._vprint(f"  ║    nbf = NeuronBridgeFinder(neuprint_token='YOUR_TOKEN')    ║")
                        self._vprint(f"  ║                                                              ║")
                        self._vprint(f"  ║  Option 2: Set environment variable:                        ║")
                        self._vprint(f"  ║    export NEUPRINT_TOKEN='YOUR_TOKEN'                       ║")
                        self._vprint(f"  ║                                                              ║")
                        self._vprint(f"  ║  Get your token at:                                         ║")
                        self._vprint(f"  ║  👉 https://neuprint.janelia.org/account                    ║")
                        self._vprint(f"  ╚══════════════════════════════════════════════════════════════╝")
                        self._vprint(f"")
                        self._vprint(f"  ℹ️  Skipping local dataset features (type lookups, specificity metrics).")
                        return None
                    
                    # Parse server URL (remove https:// if present for neuprint client)
                    server = self.neuprint_server
                    if server.startswith('https://'):
                        server = server[8:]
                    if server.startswith('http://'):
                        server = server[7:]
                    
                    client = Client(server, dataset_name, token)
                    set_default_client(client)
                    self._vprint(f"  ✓ Connected to NeuPrint ({dataset_name})")
                    
            except ImportError:
                self._vprint(f"  ⚠️ neuprint package not available. Install with: pip install neuprint-python")
                return None
            except Exception as e:
                self._vprint(f"  ⚠️ Could not connect to NeuPrint: {e}")
                return None
            
            # Create dataset directory
            dataset_dir = os.path.join(self.datasets_path, dataset)
            os.makedirs(dataset_dir, exist_ok=True)
            
            save_path = os.path.join(dataset_dir, f"{dataset}_allneurons")
            
            try:
                pull_dataset(dataset_name, save_path=save_path, omitNoneType=False)
                
                # Load the pulled data
                neuron_df_path = save_path + '_neuron_df.csv'
                if os.path.exists(neuron_df_path):
                    df = pd.read_csv(neuron_df_path, dtype={'bodyId': str}, low_memory=False)
                    self._neuron_dfs[dataset] = df
                    self._vprint(f"  ✓ Pulled and loaded {len(df):,} neurons from {dataset}")
                    return df
            except Exception as e:
                self._vprint(f"  ⚠️ Could not pull dataset {dataset_name}: {e}")
                
            return None
            
        except ImportError:
            self._vprint(f"  ⚠️ statvis module not available for pulling dataset")
            return None
        except Exception as e:
            self._vprint(f"  ⚠️ Error pulling dataset: {e}")
            return None
    
    def _fetch_neuron_df_via_fnc(self, dataset: str) -> Optional[pd.DataFrame]:
        """
        Try to fetch neuron DataFrame via FindNeuronConnection.
        
        This is a fallback when local dataset files don't exist.
        
        Parameters
        ----------
        dataset : str
            Dataset name
            
        Returns
        -------
        pd.DataFrame or None
            Fetched DataFrame, or None if failed.
        """
        try:
            from scripts.FindNeuronConnection import FindNeuronConnection
            
            self._vprint(f"  ⏳ Fetching neuron data for {dataset} via FindNeuronConnection...")
            
            fnc = FindNeuronConnection(dataset=dataset)
            
            # Check if we can get neuron index
            cache_dir = os.path.join(self.datasets_path, '..', 'cache', dataset)
            csv_path = os.path.join(cache_dir, f'{dataset}_allneurons_neuron_df.csv')
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, dtype={'bodyId': str}, low_memory=False)
                self._neuron_dfs[dataset] = df
                self._vprint(f"  ✓ Loaded {len(df):,} neurons via FindNeuronConnection cache")
                return df
            
            return None
            
        except ImportError:
            self._vprint(f"  ⚠️ FindNeuronConnection not available")
            return None
        except Exception as e:
            self._vprint(f"  ⚠️ Could not fetch via FindNeuronConnection: {e}")
            return None
    
    def _enrich_match_with_dataset_info(self, match: Dict[str, Any], em_image) -> Dict[str, Any]:
        """
        Enrich a match dictionary with dataset-specific information.
        
        Parameters
        ----------
        match : dict
            Match dictionary to enrich
        em_image : EMImage
            NeuronBridge EM image object
            
        Returns
        -------
        dict
            Enriched match dictionary
        """
        # Get library name and dataset
        library_name = getattr(em_image, 'libraryName', '')
        dataset_folder = self._get_dataset_from_library(library_name)
        dataset_name = self._get_dataset_name_from_library(library_name)
        
        # Add dataset info to match (use human-readable name, folder for internal lookup)
        match['dataset'] = dataset_name or 'unknown'
        match['dataset_folder'] = dataset_folder or 'unknown'
        match['library'] = library_name
        
        # Get neuron type and instance from EM image if available
        match['neuronType'] = getattr(em_image, 'neuronType', '') or ''
        match['neuronInstance'] = getattr(em_image, 'neuronInstance', '') or ''
        
        # Try to enrich from local dataset
        if dataset_folder:
            neuron_df = self._load_neuron_df_for_dataset(dataset_folder)
            if neuron_df is not None and not neuron_df.empty:
                body_id = str(match.get('bodyId', ''))
                neuron_row = neuron_df[neuron_df['bodyId'] == body_id]
                if not neuron_row.empty:
                    row = neuron_row.iloc[0]
                    # Use local data, fallback to NB data
                    match['type'] = row.get('type', '') or match['neuronType']
                    match['instance'] = row.get('instance', '') or match['neuronInstance']
                    match['status'] = row.get('status', '')
                else:
                    # Use NeuronBridge data
                    match['type'] = match['neuronType']
                    match['instance'] = match['neuronInstance']
                    match['status'] = ''
            else:
                match['type'] = match['neuronType']
                match['instance'] = match['neuronInstance']
                match['status'] = ''
        else:
            match['type'] = match['neuronType']
            match['instance'] = match['neuronInstance']
            match['status'] = ''
        
        return match
    
    def _get_cache_path(self, cache_type: str, identifier: str) -> str:
        """Get the path used by the current versioned Parquet cache."""
        if cache_type == 'id_to_lines':
            return str(self._get_parquet_cache().id_path(identifier))
        raise ValueError(f"Unsupported NeuronBridge cache type: {cache_type}")
    
    def _load_from_cache(self, cache_type: str, identifier: str) -> Optional[pd.DataFrame]:
        """Load a current-version Parquet result table if available."""
        if not self.use_cache:
            return None
        
        if not self._is_cached(cache_type, identifier):
            return None

        cache_path = self._get_cache_path(cache_type, identifier)
        if not os.path.exists(cache_path):
            return None
        try:
            df = pd.read_parquet(cache_path)
            if not self._batch_mode:
                self._vprint(f"  ⏩ Loaded from cache: {cache_path}")
            return df
        except Exception as exc:
            warnings.warn(f"Failed to read NeuronBridge cache {cache_path}: {exc}")
            return None
    
    def _load_from_cache_polars(self, cache_type: str, identifier: str) -> Optional['pl.DataFrame']:
        """Load a current-version Parquet result table as Polars."""
        if not self.use_cache or not HAS_POLARS:
            return None
        
        if not self._is_cached(cache_type, identifier):
            return None

        cache_path = self._get_cache_path(cache_type, identifier)
        if not os.path.exists(cache_path):
            return None
        try:
            return pl.read_parquet(cache_path)
        except Exception as exc:
            warnings.warn(f"Failed to read NeuronBridge cache {cache_path}: {exc}")
            return None
    
    def _load_cached_neurons_bulk_polars(
        self,
        body_info_list: List[Dict[str, Any]],
        match_type: str,
        max_workers: int = 16,
        progress_desc: Optional[str] = None
    ) -> Tuple['pl.DataFrame', List[int]]:
        """
        Load multiple cached neurons in parallel using pure Polars, then concatenate.
        
        This is significantly faster than loading individually because:
        1. Parallel I/O with ThreadPoolExecutor
        2. Polars native concat (no pandas conversion until final output)
        3. Single schema inference pass
        
        Parameters
        ----------
        body_info_list : list of dict
            List of body info dicts with 'bodyId' and 'dataset' keys.
        match_type : str
            Match type for cache key generation.
        max_workers : int
            Number of parallel workers for I/O.
            
        Returns
        -------
        tuple
            (pl.DataFrame with all results concatenated, list of successful body_ids)
        """
        if not HAS_POLARS or not body_info_list:
            return pl.DataFrame(), []
        
        loaded_body_ids = []
        loaded_dfs = []
        
        def load_single(body_info):
            body_id = body_info['bodyId']
            expected_ds = body_info.get('dataset', 'unknown')
            cache_key = self._get_id_to_lines_cache_key(int(body_id), match_type, expected_ds)
            
            df = self._load_from_cache_polars('id_to_lines', cache_key)
            if df is not None and df.height > 0:
                # Add source columns using Polars (no pandas!)
                df = df.with_columns([
                    pl.lit(expected_ds).alias('source_dataset'),
                    pl.lit(body_id).alias('source_bodyId')
                ])
                return body_id, df
            return body_id, None
        
        # Parallel loading with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(load_single, bi) for bi in body_info_list]

            if HAS_TQDM and self.verbose:
                from tqdm import tqdm as tqdm_progress
                pbar_cache = tqdm_progress(
                    as_completed(futures),
                    total=len(futures),
                    desc=progress_desc or "  ⏩ Loading from cache",
                    unit="neuron",
                    bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                    ncols=110,
                    position=0,
                    leave=False,
                    file=sys.stdout,
                )
                for future in pbar_cache:
                    try:
                        body_id, df = future.result()
                        if df is not None:
                            loaded_body_ids.append(body_id)
                            loaded_dfs.append(df)
                        pbar_cache.set_postfix_str(f"{body_id}")
                    except Exception:
                        pass
                pbar_cache.close()
            else:
                for future in as_completed(futures):
                    try:
                        body_id, df = future.result()
                        if df is not None:
                            loaded_body_ids.append(body_id)
                            loaded_dfs.append(df)
                    except Exception:
                        pass
        
        # Concatenate all Polars DataFrames
        if loaded_dfs:
            # Normalize schemas before concat to avoid type mismatches
            # Cast all numeric columns to consistent types
            normalized_dfs = []
            for df in loaded_dfs:
                try:
                    # Cast score to Float64, and any integer columns that might vary
                    cast_exprs = []
                    for col_name in df.columns:
                        col_type = df.schema[col_name]
                        if col_type in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64):
                            # Cast all integers to Int64 for consistency
                            cast_exprs.append(pl.col(col_name).cast(pl.Int64))
                        elif col_type in (pl.Float32, pl.Float64):
                            # Cast all floats to Float64 for consistency
                            cast_exprs.append(pl.col(col_name).cast(pl.Float64))
                        else:
                            cast_exprs.append(pl.col(col_name))
                    
                    if cast_exprs:
                        df = df.select(cast_exprs)
                    normalized_dfs.append(df)
                except Exception:
                    # If casting fails, skip this dataframe
                    pass
            
            if normalized_dfs:
                # Use diagonal concat to handle any remaining schema differences
                combined = pl.concat(normalized_dfs, how='diagonal')
                return combined, loaded_body_ids
        
        return pl.DataFrame(), []
    
    def _save_to_cache(self, cache_type: str, identifier: str, df: pd.DataFrame):
        """Save a directional cache table as compressed Parquet.

        Line-level result tables are deliberately not persisted: they duplicate
        the per-image records and make ``top_n`` part of the cache state.
        """
        if not self.use_cache or df.empty:
            return

        cache_path = self._get_cache_path(cache_type, identifier)
        with self._cache_lock:
            try:
                saved_path = self._get_parquet_cache().save_id(identifier, df)
                # Update in-memory index
                if saved_path:
                    self._add_to_cache_index(cache_type, identifier)
                # Only print cache saves when not in batch mode
                if not self._batch_mode:
                    self._vprint(f"  💾 Saved to cache: {saved_path}")
            except Exception as e:
                warnings.warn(f"Failed to save cache: {e}")
    
    # =========================================================================
    # Image-based Cache System (indexed by image_id)
    # =========================================================================
    # This cache system stores API results per LM image (lm_sample) and match_type
    # allowing reuse regardless of region, top_n, or max_api_images_per_line settings
    #
    # Line-to-Image Mapping:
    # - Tracks which images belong to which line (for each region)
    # - Records cached match types per image
    # - Does NOT mark lines as "complete" since server images can update
    # - Always checks online for new images, uses cache for existing ones
    
    def _get_image_cache_path(self, image_id: str, match_type: str) -> str:
        """Get the new Parquet path for a specific LM image and match type."""
        return str(self._get_parquet_cache().image_path(str(image_id), match_type))
    
    def _get_line_mapping_path(self) -> str:
        """Get the mapping path inside the current versioned cache."""
        return str(self._get_parquet_cache().root / 'line_image_mapping.json')
    
    def _load_line_mapping(self) -> Dict[str, Any]:
        """
        Load the line-to-image mapping file. Thread-safe.
        
        Structure:
        {
            "lines": {
                "VT037867": {
                    "Brain": {
                        "image_ids": ["123", "456", ...],
                        "last_checked": "2024-12-29T10:30:00"
                    },
                    "VNC": {...}
                }
            },
            "images": {
                "123": {
                    "line": "VT037867",
                    "cached_types": ["cds", "pppm"]
                }
            }
        }
        """
        with self._cache_lock:
            return self._read_line_mapping_unlocked()

    def _read_line_mapping_unlocked(self) -> Dict[str, Any]:
        """Read the mapping; the caller must hold ``_cache_lock``."""
        mapping_path = self._get_line_mapping_path()
        if os.path.exists(mapping_path):
            try:
                with open(mapping_path, 'r') as f:
                    mapping = json.load(f)
                if isinstance(mapping, dict):
                    mapping.setdefault("lines", {})
                    mapping.setdefault("images", {})
                    return mapping
            except Exception:
                pass
        return {"lines": {}, "images": {}}
    
    def _save_line_mapping(self, mapping: Dict[str, Any]):
        """Atomically save the line-to-image mapping file. Thread-safe."""
        mapping_path = self._get_line_mapping_path()
        with self._cache_lock:
            self._write_line_mapping_unlocked(mapping)

    def _write_line_mapping_unlocked(self, mapping: Dict[str, Any]):
        """Atomically write the mapping; the caller must hold ``_cache_lock``."""
        mapping_path = self._get_line_mapping_path()
        temporary = f"{mapping_path}.{os.getpid()}.tmp"
        try:
            os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
            with open(temporary, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, indent=2)
            os.replace(temporary, mapping_path)
        except Exception as e:
            try:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            except Exception:
                pass
            warnings.warn(f"Failed to save line mapping: {e}")
    
    def _update_line_mapping(
        self, 
        line_name: str, 
        region: str, 
        image_ids: List[str],
        image_id: Optional[str] = None,
        match_type: Optional[str] = None
    ):
        """
        Update the line-image mapping with new information.
        
        Parameters
        ----------
        line_name : str
            The driver line name
        region : str
            The anatomical region (Brain, VNC, etc.)
        image_ids : list
            List of all image IDs for this line/region (from current API check)
        image_id : str, optional
            Specific image ID to update cached_types for
        match_type : str, optional
            Match type that was just cached for image_id
        """
        # Keep the read-modify-write sequence under one lock so concurrent
        # image workers cannot discard one another's cached-type updates.
        with self._cache_lock:
            mapping = self._read_line_mapping_unlocked()

            # Update lines section
            if line_name not in mapping["lines"]:
                mapping["lines"][line_name] = {}

            from datetime import datetime
            mapping["lines"][line_name][region] = {
                "image_ids": image_ids,
                "last_checked": datetime.now().isoformat(),
                "image_count": len(image_ids)
            }

            # Update images section
            for img_id in image_ids:
                if img_id not in mapping["images"]:
                    mapping["images"][img_id] = {
                        "line": line_name,
                        "cached_types": []
                    }
                else:
                    # Update line info (in case image is shared)
                    mapping["images"][img_id]["line"] = line_name

            # Update cached_types for specific image
            if image_id and match_type:
                if image_id not in mapping["images"]:
                    mapping["images"][image_id] = {
                        "line": line_name,
                        "cached_types": []
                    }
                if match_type not in mapping["images"][image_id]["cached_types"]:
                    mapping["images"][image_id]["cached_types"].append(match_type)

            self._write_line_mapping_unlocked(mapping)
    
    def sync_mapping_from_cache_files(self) -> Dict[str, int]:
        """
        Sync the line-image mapping with existing cache files.
        
        Scans all cache files in image_cache directory and updates the mapping
        with the cached types for each image.
        
        Returns
        -------
        dict
            Statistics: {'images_scanned', 'types_updated'}
        """
        stats = {'images_scanned': 0, 'types_updated': 0}
        
        # The mapping is a small diagnostic/index artifact; Parquet files are
        # the authoritative cache data.
        cache_paths = list(self._get_parquet_cache().image_dir.glob('*.parquet'))
        with self._cache_lock:
            mapping = self._read_line_mapping_unlocked()
            for cache_path in cache_paths:
                if not cache_path.is_file():
                    continue

                # Parse filename: {match_type}_{image_id}.parquet
                parts = cache_path.stem.split('_', 1)
                if len(parts) != 2:
                    continue

                match_type, image_id = parts
                if match_type not in ['cds', 'pppm']:
                    continue

                stats['images_scanned'] += 1

                # Update mapping
                if image_id not in mapping.get("images", {}):
                    mapping["images"][image_id] = {
                        "line": "",  # Unknown line
                        "cached_types": []
                    }

                if match_type not in mapping["images"][image_id].get("cached_types", []):
                    if "cached_types" not in mapping["images"][image_id]:
                        mapping["images"][image_id]["cached_types"] = []
                    mapping["images"][image_id]["cached_types"].append(match_type)
                    stats['types_updated'] += 1

            self._write_line_mapping_unlocked(mapping)
        self._vprint(f"✓ Synced mapping: {stats['images_scanned']} files, {stats['types_updated']} types updated")
        return stats
    
    def _load_image_cache(self, image_id: str, match_type: str) -> Optional[pd.DataFrame]:
        """
        Load cached matches for a specific LM image.
        
        Parameters
        ----------
        image_id : str
            The LM image ID (lm_sample)
        match_type : str
            Match algorithm: 'cds' or 'pppm'
            
        Returns
        -------
        pd.DataFrame or None
            Cached matches DataFrame, or None if not found
        """
        if not self.use_cache:
            return None
        
        cache_path = self._get_image_cache_path(image_id, match_type)
        if not os.path.exists(cache_path):
            return None
        try:
            return self._get_parquet_cache().normalize_image(
                pd.read_parquet(cache_path)
            )
        except Exception as exc:
            warnings.warn(f"Failed to read NeuronBridge image cache {cache_path}: {exc}")
            return None
    
    def _save_image_cache(
        self, 
        image_id: str, 
        match_type: str, 
        matches: List[Dict[str, Any]],
        line_name: Optional[str] = None
    ):
        """
        Save matches for a specific LM image to cache.
        
        Parameters
        ----------
        image_id : str
            The LM image ID (lm_sample)
        match_type : str
            Match algorithm: 'cds', 'pppm', or 'both'
        matches : list
            List of match dictionaries
        line_name : str, optional
            The line name for updating the mapping
        """
        if not self.use_cache or not matches or match_type == 'both':
            return

        try:
            df = pd.DataFrame(matches)
            cache_path = self._get_parquet_cache().save_image(
                str(image_id), match_type, df
            )
            
            # Update mapping with cached type info
            if line_name and cache_path:
                image_id = str(image_id)
                with self._cache_lock:
                    mapping = self._read_line_mapping_unlocked()
                    if image_id not in mapping.get("images", {}):
                        mapping["images"][image_id] = {
                            "line": line_name,
                            "cached_types": []
                        }
                    if match_type not in mapping["images"][image_id].get("cached_types", []):
                        mapping["images"][image_id]["cached_types"].append(match_type)
                    self._write_line_mapping_unlocked(mapping)
        except Exception as e:
            warnings.warn(f"Failed to save image cache: {e}")
    
    def _fetch_matches_from_api(
        self,
        lm_image,
        match_type: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch matches from API for a single match type (cds or pppm).
        
        Parameters
        ----------
        lm_image : LMImage
            The LM image object
        match_type : str
            Match algorithm: 'cds' or 'pppm' (not 'both')
            
        Returns
        -------
        list
            List of match dictionaries
        """
        image_id = getattr(lm_image, 'id', '')
        matches = []
        
        try:
            if match_type == 'cds':
                api_matches = self._retry_with_backoff(
                    self._client.get_cds_matches,
                    lm_image,
                    max_retries=5,
                    initial_delay=1.0
                )
            else:  # pppm
                api_matches = self._retry_with_backoff(
                    self._client.get_ppp_matches,
                    lm_image,
                    max_retries=5,
                    initial_delay=1.0
                )
            
            for match in api_matches:
                if hasattr(match, 'image') and hasattr(match.image, 'type'):
                    if match.image.type == 'EMImage':
                        body_id = self._extract_body_id(match.image)
                        match_dict = {
                            'bodyId': body_id,
                            'score': getattr(match, 'normalizedScore', 0),
                            'image_id': getattr(match.image, 'id', ''),
                            'lm_sample': str(image_id),
                            'match_type': match_type
                        }
                        # Enrich with dataset info
                        match_dict = self._enrich_match_with_dataset_info(match_dict, match.image)
                        matches.append(match_dict)
                        
        except Exception:
            pass  # Return empty matches on error
        
        return matches
    
    def _get_image_matches_cached(
        self, 
        lm_image, 
        match_type: str,
        line_name: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], bool, bool]:
        """
        Get matches for an LM image, using cache if available.
        
        Handles 'both' match_type specially by loading/fetching the two raw
        algorithm tables and combining them in memory.  No derived ``both``
        table is written.
        
        Parameters
        ----------
        lm_image : LMImage
            The LM image object
        match_type : str
            Match algorithm: 'cds', 'pppm', or 'both'
        line_name : str, optional
            Line name for mapping updates
            
        Returns
        -------
        tuple
            (matches_list, from_cache, partial_cache) 
            - matches_list: list of match dicts
            - from_cache: True if ALL data came from cache
            - partial_cache: True if SOME data came from cache (for 'both')
        """
        image_id = getattr(lm_image, 'id', '')
        if not image_id:
            return [], False, False
        
        image_id_str = str(image_id)
        
        # Handle simple cases (cds or pppm)
        if match_type in ['cds', 'pppm']:
            cached_df = self._load_image_cache(image_id_str, match_type)
            if cached_df is not None and not cached_df.empty:
                return cached_df.to_dict('records'), True, False
            
            # Fetch from API
            matches = self._fetch_matches_from_api(lm_image, match_type)
            if matches:
                self._save_image_cache(image_id_str, match_type, matches, line_name)
            return matches, False, False
        
        # Check existing cds and pppm caches
        cds_cache = self._load_image_cache(image_id_str, 'cds')
        pppm_cache = self._load_image_cache(image_id_str, 'pppm')
        
        cds_matches = []
        pppm_matches = []
        cds_from_cache = False
        pppm_from_cache = False
        
        # Get CDS matches (from cache or API)
        if cds_cache is not None and not cds_cache.empty:
            cds_matches = cds_cache.to_dict('records')
            cds_from_cache = True
        else:
            cds_matches = self._fetch_matches_from_api(lm_image, 'cds')
            if cds_matches:
                self._save_image_cache(image_id_str, 'cds', cds_matches, line_name)
        
        # Get PPPM matches (from cache or API)
        if pppm_cache is not None and not pppm_cache.empty:
            pppm_matches = pppm_cache.to_dict('records')
            pppm_from_cache = True
        else:
            pppm_matches = self._fetch_matches_from_api(lm_image, 'pppm')
            if pppm_matches:
                self._save_image_cache(image_id_str, 'pppm', pppm_matches, line_name)
        
        # Combine CDS and PPPM matches
        all_matches = cds_matches + pppm_matches
        
        # Determine cache status
        all_from_cache = cds_from_cache and pppm_from_cache
        partial_cache = cds_from_cache or pppm_from_cache
        
        return all_matches, all_from_cache, partial_cache
    
    def _get_em_images(self, body_id: int) -> List[Any]:
        """Fetch all NeuronBridge EM image records for one body ID."""
        if not self._client:
            return []
        try:
            em_images = self._client.get_em_images(body_id)
            if not em_images:
                return []
            return em_images if isinstance(em_images, list) else list(em_images)
        except Exception:
            return []

    @staticmethod
    def _dataset_identity(dataset: Optional[str]) -> Tuple[str, Optional[str]]:
        """Return a normalized ``(base, version)`` dataset identity.

        NeuronBridge uses both ``male-cns:v1.0`` and published names such as
        ``male-cns:v1.0:12211``.  Folder-style names use underscores.  The
        body ID is removed before parsing, and versions are normalized to
        dotted form so equivalent spellings compare equal.
        """
        if dataset is None:
            return '', None
        value = str(dataset).strip().lower()
        if not value or value in {'unknown', 'none', 'nan'}:
            return '', None

        parts = value.split(':')
        if len(parts) >= 3 and parts[-1].isdigit():
            value = ':'.join(parts[:-1])

        version_match = re.search(r'(?:^|[:_])v?(\d+(?:[._]\d+)*)$', value)
        if version_match:
            version = version_match.group(1).replace('_', '.')
            base = value[:version_match.start()]
        else:
            version = None
            base = value

        base = base.strip(':_').replace(':', '-').replace('_', '-')
        return base, version

    @classmethod
    def _datasets_match(
        cls,
        expected_dataset: Optional[str],
        actual_dataset: Optional[str],
    ) -> bool:
        """Match dataset bases and require an exact version when requested."""
        expected_base, expected_version = cls._dataset_identity(expected_dataset)
        actual_base, actual_version = cls._dataset_identity(actual_dataset)
        if not expected_base:
            return True
        return (
            expected_base == actual_base
            and (expected_version is None or expected_version == actual_version)
        )

    @staticmethod
    def _dataset_name_from_em_image(em_image: Any) -> Optional[str]:
        """Extract a dataset name from an EM image published name."""
        published_name = str(getattr(em_image, 'publishedName', '') or '')
        if not published_name:
            return None
        parts = published_name.split(':')
        if len(parts) >= 3 and parts[-1].isdigit():
            return ':'.join(parts[:-1])
        return ':'.join(parts[:2]) if len(parts) >= 2 else parts[0]

    @classmethod
    def _select_em_image_for_dataset(
        cls,
        em_images: List[Any],
        expected_dataset: Optional[str] = None,
    ) -> Optional[Any]:
        """Select only an image from the requested dataset/version."""
        if not em_images:
            return None
        if not expected_dataset or str(expected_dataset).strip().lower() == 'unknown':
            return em_images[0]
        for em_image in em_images:
            actual_dataset = cls._dataset_name_from_em_image(em_image)
            if cls._datasets_match(expected_dataset, actual_dataset):
                return em_image
        return None

    def _get_em_image_for_dataset(
        self,
        body_id: int,
        expected_dataset: Optional[str] = None,
    ):
        """Return an EM image only when its dataset/version matches."""
        return self._select_em_image_for_dataset(
            self._get_em_images(body_id), expected_dataset
        )

    def _get_body_metadata(
        self,
        body_id: int,
        expected_dataset: Optional[str],
    ) -> Tuple[Optional[str], Optional[Any]]:
        """Fetch body metadata once and return its actual dataset and image."""
        em_images = self._get_em_images(body_id)
        if not em_images:
            return None, None
        selected = self._select_em_image_for_dataset(em_images, expected_dataset)
        actual_image = selected or em_images[0]
        return self._dataset_name_from_em_image(actual_image), selected

    @classmethod
    def _canonical_source_dataset(
        cls,
        expected_dataset: Optional[str],
        actual_dataset: Optional[str],
    ) -> Optional[str]:
        """Stable output spelling for a body's source dataset.

        Fresh fetches stamp NeuronBridge's ``publishedName`` identity
        (``flywire_fafb:v783``) while cache loads stamp the caller's
        spelling (``flywire_FAFB_v783``); the two spellings then split
        ``datasets_labeled`` / ``matched_datasets`` counts.  When the
        fetched identity MATCHES the expected dataset, the caller's
        spelling wins; genuinely different identities pass through.
        """
        if (
            actual_dataset
            and expected_dataset
            and expected_dataset != 'unknown'
            and cls._datasets_match(expected_dataset, actual_dataset)
        ):
            return expected_dataset
        return actual_dataset or expected_dataset

    def _validate_body_id_dataset(
        self,
        body_id: int,
        expected_dataset: str,
    ) -> Optional[str]:
        """Return the actual dataset identity reported for a body ID."""
        if not self._client:
            return expected_dataset
        actual_dataset, _ = self._get_body_metadata(body_id, expected_dataset)
        return actual_dataset
    
    def _validate_body_ids_parallel(
        self,
        body_info_list: List[Dict[str, Any]],
        max_workers: Optional[int] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Validate multiple body IDs in parallel against NeuronBridge API.
        
        This batches all validation API calls to run concurrently, significantly
        speeding up validation for large numbers of uncached neurons.
        
        Parameters
        ----------
        body_info_list : list of dict
            List of body info dicts with 'bodyId' and 'dataset' keys.
        max_workers : int, optional
            Maximum parallel workers. Defaults to self.max_workers.
            
        Returns
        -------
        dict
            Mapping of body_id -> ``{'dataset', 'em_image'}``.  The selected
            image is carried forward so the later match fetch does not repeat
            the metadata request.
        """
        if not body_info_list:
            return {}
        
        workers = max_workers or self.max_workers
        results = {}
        
        def validate_single(body_info):
            body_id = body_info['bodyId']
            expected_ds = body_info.get('dataset', 'unknown')
            actual_ds, em_image = self._get_body_metadata(int(body_id), expected_ds)
            return body_id, {'dataset': actual_ds, 'em_image': em_image}
        
        # Use ThreadPoolExecutor for parallel API calls
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(validate_single, bi): bi for bi in body_info_list}
            for future in as_completed(futures):
                try:
                    body_id, metadata = future.result()
                    results[body_id] = metadata
                except Exception:
                    body_info = futures[future]
                    results[body_info['bodyId']] = {
                        'dataset': body_info.get('dataset', 'unknown'),
                        'em_image': None,
                    }
        
        return results
    
    def _sort_matches_by_rank(
        self, 
        matches: List[Dict[str, Any]], 
        key_field: str = 'line'
    ) -> List[Dict[str, Any]]:
        """
        Sort matches by combined rank when both CDS and PPPM results exist.
        
        For each unique line/bodyId, compute:
        - CDS rank (1-based, by descending score)
        - PPPM rank (1-based, by descending score)
        - Combined rank = CDS_rank + PPPM_rank
        
        Sort by combined rank ascending (lower is better).
        
        Parameters
        ----------
        matches : list
            List of match dictionaries with 'match_type', 'score', and key_field.
        key_field : str
            Field to use as unique identifier ('line' for EM->LM, 'bodyId' for LM->EM).
            
        Returns
        -------
        list
            Sorted list of match dictionaries with added 'cds_rank', 'pppm_rank', 
            'combined_rank' columns.
        """
        if not matches:
            return matches
        
        # Separate CDS and PPPM matches
        cds_matches = [m for m in matches if m.get('match_type') == 'cds']
        pppm_matches = [m for m in matches if m.get('match_type') == 'pppm']
        
        # Sort each by score descending and assign ranks
        cds_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
        pppm_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Build rank dictionaries (key -> rank)
        cds_ranks = {}
        for rank, m in enumerate(cds_matches, 1):
            key = m.get(key_field)
            if key and key not in cds_ranks:
                cds_ranks[key] = rank
        
        pppm_ranks = {}
        for rank, m in enumerate(pppm_matches, 1):
            key = m.get(key_field)
            if key and key not in pppm_ranks:
                pppm_ranks[key] = rank
        
        # Default rank for missing matches (use max+1 or a high number)
        max_cds_rank = len(cds_matches) + 1 if cds_matches else 1
        max_pppm_rank = len(pppm_matches) + 1 if pppm_matches else 1
        
        # Combine matches by key, keeping best score per match_type
        combined = {}
        for m in matches:
            key = m.get(key_field)
            if not key:
                continue
            
            if key not in combined:
                combined[key] = {
                    key_field: key,
                    'library': m.get('library', ''),
                    'image_id': m.get('image_id', ''),
                    'cds_score': 0,
                    'pppm_score': 0,
                    'cds_rank': max_cds_rank,
                    'pppm_rank': max_pppm_rank,
                }
                # Copy other fields for LM->EM matches
                for field in ['bodyId', 'dataset', 'instance', 'type', 'status', 'lm_sample']:
                    if field in m:
                        combined[key][field] = m[field]
            
            # Update scores and ranks
            if m.get('match_type') == 'cds':
                if m.get('score', 0) > combined[key]['cds_score']:
                    combined[key]['cds_score'] = m.get('score', 0)
                combined[key]['cds_rank'] = cds_ranks.get(key, max_cds_rank)
            elif m.get('match_type') == 'pppm':
                if m.get('score', 0) > combined[key]['pppm_score']:
                    combined[key]['pppm_score'] = m.get('score', 0)
                combined[key]['pppm_rank'] = pppm_ranks.get(key, max_pppm_rank)
        
        # Calculate combined rank and sort
        result = []
        for key, data in combined.items():
            data['combined_rank'] = data['cds_rank'] + data['pppm_rank']
            # Use higher score as primary score for display
            data['score'] = max(data['cds_score'], data['pppm_score'])
            data['match_type'] = 'both'
            result.append(data)
        
        # Sort by combined rank ascending
        result.sort(key=lambda x: x['combined_rank'])
        
        return result
    
    def _get_em_matches(
        self, 
        body_id: int, 
        match_type: Optional[str] = None,
        expected_dataset: Optional[str] = None,
        em_image: Any = _EM_IMAGE_NOT_PROVIDED,
        raw: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get LM matches for an EM body ID.
        
        Parameters
        ----------
        body_id : int
            The EM body ID to search for.
        match_type : str, optional
            Match algorithm: 'cds', 'pppm', or 'both'. 
            If None, uses self.match_type. Default: None
        expected_dataset : str, optional
            Expected dataset name (e.g., 'male-cns:v0.9').
            If provided, will filter EM images to match this dataset.
        em_image : object, optional
            A preselected EM image from batched metadata validation.  Passing
            ``None`` explicitly means validation found no matching image and
            avoids a second metadata request.
        raw : bool
            If True, return the CDS/PPPM rows before derived combined-rank
            reduction.  This is used so ``both`` can be reconstructed from
            the two canonical algorithm caches instead of being cached again.
            
        Returns
        -------
        list
            List of match dictionaries, sorted by:
            - 'cds' or 'pppm': score descending
            - 'both': combined rank ascending (sum of CDS rank + PPPM rank)
        """
        # Use class-level match_type if not specified
        if match_type is None:
            match_type = self.match_type
            
        try:
            # Reuse a selected image from batched metadata validation when
            # available; otherwise resolve it here for direct callers.
            if em_image is _EM_IMAGE_NOT_PROVIDED:
                em_image = self._get_em_image_for_dataset(body_id, expected_dataset)
            if em_image is None:
                self._vprint(f"  ⚠️ No EM image found for body ID {body_id}")
                return []
            
            # Check if em_image has required URL for fetching matches
            # Some body IDs exist but don't have valid match URLs in NeuronBridge
            em_files = getattr(em_image, 'files', None)
            if em_files is None:
                self._vprint(f"  ⚠️ No files metadata for body ID {body_id}")
                return []
            
            all_matches = []
            
            cds_failed = False
            pppm_failed = False
            
            # Get CDS matches with retry logic
            if match_type in ['cds', 'both']:
                try:
                    # Check if CDSResults URL exists
                    cds_url = getattr(em_files, 'CDSResults', None)
                    if cds_url is None:
                        cds_failed = True
                    else:
                        cds_matches = self._retry_with_backoff(
                            self._client.get_cds_matches,
                            em_image,
                            max_retries=5,
                            initial_delay=1.0
                        )
                        for match in cds_matches:
                            if hasattr(match, 'image') and hasattr(match.image, 'type'):
                                if match.image.type == 'LMImage':
                                    all_matches.append({
                                        'line': getattr(match.image, 'publishedName', ''),
                                        'library': getattr(match.image, 'libraryName', ''),
                                        'score': getattr(match, 'normalizedScore', 0),
                                        'image_id': getattr(match.image, 'id', ''),
                                        'match_type': 'cds'
                                    })
                except Exception as e:
                    cds_failed = True
                    # Suppress individual errors - will be caught by calling method
            
            # Get PPPM matches with retry logic
            if match_type in ['pppm', 'both']:
                try:
                    # Check if PPPMResults URL exists
                    pppm_url = getattr(em_files, 'PPPMResults', None)
                    if pppm_url is None:
                        pppm_failed = True
                    else:
                        pppm_matches = self._retry_with_backoff(
                            self._client.get_ppp_matches,
                            em_image,
                            max_retries=5,
                            initial_delay=1.0
                        )
                        for match in pppm_matches:
                            if hasattr(match, 'image') and hasattr(match.image, 'type'):
                                if match.image.type == 'LMImage':
                                    all_matches.append({
                                        'line': getattr(match.image, 'publishedName', ''),
                                        'library': getattr(match.image, 'libraryName', ''),
                                        'score': getattr(match, 'normalizedScore', 0),
                                        'image_id': getattr(match.image, 'id', ''),
                                        'match_type': 'pppm'
                                    })
                except Exception as e:
                    pppm_failed = True
                    # Suppress individual errors - will be caught by calling method
            
            # Only warn if both failed when match_type='both'
            if match_type == 'both' and cds_failed and pppm_failed:
                self._vprint(f"  ⚠️ Both CDS and PPPM matches failed for body ID {body_id}")

            if raw:
                return all_matches
            
            # Sort results based on match_type
            if match_type == 'both' and all_matches:
                # Use rank-based sorting for combined results
                all_matches = self._sort_matches_by_rank(all_matches, key_field='line')
            else:
                # Sort by score descending for single match type
                all_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            return all_matches
            
        except Exception as e:
            self._vprint(f"  ⚠️ Error getting matches for body ID {body_id}: {e}")
            return []
    
    def _get_lm_matches(
        self, 
        line_name: str, 
        match_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get EM matches for an LM line name.
        
        Uses image-based caching to store results per LM image, allowing reuse
        regardless of region, top_n, or max_api_images_per_line settings.
        
        Always checks online for current images (since server images can update),
        but uses cached match results when available.
        
        Parameters
        ----------
        line_name : str
            The driver line name to search for.
        match_type : str, optional
            Match algorithm: 'cds', 'pppm', or 'both'. 
            If None, uses self.match_type. Default: None
            
        Returns
        -------
        list
            List of match dictionaries with dataset info, sorted by:
            - 'cds' or 'pppm': score descending
            - 'both': combined rank ascending (sum of CDS rank + PPPM rank)
        """
        # Use class-level match_type if not specified
        if match_type is None:
            match_type = self.match_type
            
        try:
            # Always check online for current images (server images can update)
            lm_images = self._client.get_lm_images(line_name)
            if not lm_images:
                self._vprint(f"  ⚠️ No LM images found for line '{line_name}'")
                return []
            
            # Convert generator to list if needed
            if not isinstance(lm_images, list):
                lm_images = list(lm_images)
            
            # Filter images by region
            lm_images = self._filter_images_by_region(lm_images)
            if not lm_images:
                self._vprint(f"  ⚠️ No LM images for line '{line_name}' in region '{self.region}'")
                return []
            
            # Filter images by match_type availability (pre-scan to skip unavailable)
            lm_images = self._filter_images_by_match_availability(lm_images, match_type)
            if not lm_images:
                self._vprint(f"  ⚠️ No LM images with {match_type.upper()} results for line '{line_name}'")
                return []
            
            # Update line-image mapping with current images from server
            all_image_ids = [str(getattr(img, 'id', '')) for img in lm_images if getattr(img, 'id', '')]
            if all_image_ids:
                self._update_line_mapping(line_name, self.region, all_image_ids)
            
            # Apply max_api_images_per_line limit
            original_count = len(lm_images)
            if self.max_api_images_per_line > 0 and len(lm_images) > self.max_api_images_per_line:
                lm_images = lm_images[:self.max_api_images_per_line]
                self._vprint(f"  ℹ️  Using {len(lm_images)}/{original_count} images (max_api_images_per_line={self.max_api_images_per_line})")
            
            n_images = len(lm_images)
            # Only print verbose message if not in a progress bar context
            if not HAS_TQDM or not self.verbose:
                self._vprint(f"  Found {n_images} LM images for '{line_name}'")
            
            all_matches = []
            cache_hits = 0
            partial_cache_hits = 0
            api_fetches = 0
            errors = 0
            
            # Create progress bar for image processing if we have multiple images and tqdm is available
            show_image_progress = HAS_TQDM and self.verbose and n_images > 1
            
            # Use parallel processing for multiple images when max_workers > 1
            use_parallel = self.max_workers > 1 and n_images > 1
            
            if use_parallel:
                # Parallel processing with ThreadPoolExecutor
                self._suppress_loading_msgs = True
                
                def process_single_image(lm_image):
                    """Worker function to process a single image."""
                    return self._get_image_matches_cached(lm_image, match_type, line_name)
                
                with ThreadPoolExecutor(max_workers=min(self.max_workers, n_images)) as executor:
                    # Submit all tasks
                    futures = {executor.submit(process_single_image, img): img for img in lm_images}
                    
                    # Process results as they complete
                    if show_image_progress:
                        from tqdm import tqdm as tqdm_progress
                        iterator = tqdm_progress(
                            as_completed(futures),
                            total=n_images,
                            desc=f"  🖼️  Processing images",
                            unit="img",
                            leave=False,
                            bar_format='  {desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                            ncols=90,
                            file=sys.stdout,
                        )
                    else:
                        iterator = as_completed(futures)
                    
                    for future in iterator:
                        try:
                            matches, from_cache, partial_cache = future.result()
                            
                            if from_cache:
                                cache_hits += 1
                            elif partial_cache:
                                partial_cache_hits += 1
                            else:
                                api_fetches += 1
                            
                            if matches:
                                all_matches.extend(matches)
                            elif not from_cache and not partial_cache:
                                errors += 1
                        except Exception as e:
                            errors += 1
                
                self._suppress_loading_msgs = False
            else:
                # Sequential processing (original behavior)
                if show_image_progress:
                    from tqdm import tqdm as tqdm_progress
                    # Suppress loading messages while progress bar is active
                    self._suppress_loading_msgs = True
                    image_iterator = tqdm_progress(
                        lm_images,
                        desc=f"  🖼️  Processing images",
                        unit="img",
                        leave=False,
                        bar_format='  {desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                        ncols=90,
                        file=sys.stdout,
                    )
                else:
                    image_iterator = lm_images
                
                for img_idx, lm_image in enumerate(image_iterator, 1):
                    # Use the unified _get_image_matches_cached which handles 'both' internally
                    matches, from_cache, partial_cache = self._get_image_matches_cached(
                        lm_image, match_type, line_name
                    )
                    
                    if from_cache:
                        cache_hits += 1
                    elif partial_cache:
                        partial_cache_hits += 1
                    else:
                        api_fetches += 1
                    
                    if matches:
                        all_matches.extend(matches)
                    elif not from_cache and not partial_cache:
                        errors += 1
                
                # Restore loading messages flag after processing images
                if show_image_progress:
                    self._suppress_loading_msgs = False
            
            # Report cache/API statistics explicitly
            total_lookups = cache_hits + partial_cache_hits + api_fetches
            if total_lookups > 0:
                cache_count = cache_hits + partial_cache_hits
                self._vprint(f"  📊 Image data: {cache_count}/{total_lookups} from cache, {api_fetches}/{total_lookups} API fetches")
            
            # Report error summary if there were failures
            if errors > 0:
                self._vprint(f"  ℹ️  {errors}/{n_images} images had no matches")
            
            # Sort results based on match_type
            if match_type == 'both' and all_matches:
                # Use rank-based sorting for combined results
                unique_matches = self._sort_matches_by_rank(all_matches, key_field='bodyId')
            else:
                # Sort by score descending for single match type
                all_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
                
                # Remove duplicates (same bodyId + dataset), keeping highest score
                seen = set()
                unique_matches = []
                for match in all_matches:
                    key = (match.get('bodyId'), match.get('dataset'))
                    if key not in seen:
                        seen.add(key)
                        unique_matches.append(match)
            
            return unique_matches
            
        except Exception as e:
            self._vprint(f"  ⚠️ Error getting matches for line '{line_name}': {e}")
            return []
    
    def _extract_body_id(self, em_image) -> str:
        """Extract body ID from an EM image object."""
        # Try publishedName first (format: 'hemibrain:v1.2.1:BODY_ID')
        published_name = getattr(em_image, 'publishedName', '')
        if published_name and ':' in published_name:
            parts = published_name.split(':')
            if len(parts) >= 3:
                return parts[-1]
        
        # Try id field
        image_id = getattr(em_image, 'id', '')
        if image_id:
            # Extract digits
            digits = ''.join(c for c in str(image_id) if c.isdigit())
            if digits:
                return digits
        
        return ''
    
    def _find_bodyIds_by_query(
        self, 
        query: Union[str, int, List],
        dataset: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find body IDs matching a query in the neuron DataFrame(s).
        
        Parameters
        ----------
        query : str, int, or list
            - If int: treated as bodyId
            - If str: searched in instance and type columns (supports regex)
            - If list: each item processed individually
        dataset : str, list of str, or None
            Specific dataset(s) to search (e.g., 'hemibrain:v1.2.1', 'male-cns:v0.9').
            Can be a single string or a list of datasets.
            If None, searches all available datasets.
            
        Returns
        -------
        list
            List of dicts with 'bodyId' and 'dataset' keys.
        """
        # Determine which datasets to search
        datasets_to_search = []
        
        if dataset:
            # Handle list of datasets
            dataset_list = [dataset] if isinstance(dataset, str) else list(dataset)
            
            for ds in dataset_list:
                # Map display name to folder name if needed
                dataset_folder = self._dataset_name_to_folder(ds)
                if dataset_folder:
                    datasets_to_search.append(dataset_folder)
                else:
                    datasets_to_search.append(canonical_dataset_name(ds).replace(':', '_').replace('.', '_'))
            # Remove duplicates while preserving order
            datasets_to_search = list(dict.fromkeys(datasets_to_search))
        else:
            # Search all available datasets
            # Get unique dataset folder names
            datasets_to_search = list(set(LIBRARY_TO_DATASET.values()))
        
        if isinstance(query, list):
            all_results = []
            for q in query:
                all_results.extend(self._find_bodyIds_by_query(q, dataset))
            # Remove duplicates (same bodyId + dataset)
            seen = set()
            unique_results = []
            for r in all_results:
                key = (r['bodyId'], r['dataset'])
                if key not in seen:
                    seen.add(key)
                    unique_results.append(r)
            return unique_results

        # NeuronBridge used to maintain a separate case-insensitive exact
        # matcher over only ``type`` and ``instance``.  Route it through the
        # same priority-aware resolver as pathfinding, morphology, skeleton
        # visualization, the inline suggestions, and the neuron-index viewer.
        # This also makes explicit ``name.*``/contains patterns obey the same
        # strict/case-sensitive contract instead of silently broadening them.
        try:
            from src.neuron_search import (
                get_cached_neuron_search,
                resolve_cached_or_dataframe_query,
            )
        except ImportError:  # pragma: no cover - ``src/`` on sys.path imports
            from neuron_search import (
                get_cached_neuron_search,
                resolve_cached_or_dataframe_query,
            )

        shared_results = []
        for ds_folder in datasets_to_search:
            neuron_df = self._load_neuron_df_for_dataset(ds_folder)
            if neuron_df is None or neuron_df.empty:
                continue
            cached_search = get_cached_neuron_search(ds_folder)
            body_ids, _search_info = resolve_cached_or_dataframe_query(
                cached_search,
                neuron_df,
                query,
                search_columns="auto",
            )
            ds_name = self._folder_to_dataset_name(ds_folder)
            for body_id in body_ids:
                shared_results.append({
                    "bodyId": (
                        normalize_flywire_body_id(body_id)
                        if is_local_connectome_dataset(ds_folder)
                        else str(body_id)
                    ),
                    "dataset": ds_name,
                    "dataset_folder": ds_folder,
                })

        deduplicated = []
        seen = set()
        for result in shared_results:
            key = (result["bodyId"], result["dataset"])
            if key not in seen:
                seen.add(key)
                deduplicated.append(result)
        if deduplicated or not isinstance(query, int):
            return deduplicated
        # Preserve the historical NeuronBridge behavior for an integer bodyId
        # that is not present in any local dataset: downstream code displays a
        # useful unknown-dataset result rather than silently dropping it.
        return [{
            "bodyId": str(query),
            "dataset": "unknown",
            "dataset_folder": "",
        }]

    def _dataset_name_to_folder(self, dataset_name: str) -> Optional[str]:
        """Convert display dataset name to folder name."""
        # Try direct lookup via reverse mapping
        for lib, name in LIBRARY_TO_DATASET_NAME.items():
            if name.lower() == dataset_name.lower():
                return LIBRARY_TO_DATASET.get(lib)
        # Try folder names directly
        for lib, folder in LIBRARY_TO_DATASET.items():
            if folder.lower() == canonical_dataset_name(dataset_name.lower()).replace(':', '_').replace('.', '_'):
                return folder
        return None
    
    def _folder_to_dataset_name(self, folder: str) -> str:
        """Convert folder name to display dataset name."""
        # Reverse lookup in LIBRARY_TO_DATASET
        for lib, lib_folder in LIBRARY_TO_DATASET.items():
            if lib_folder == folder:
                return LIBRARY_TO_DATASET_NAME.get(lib, folder)
        return folder
    
    # =========================================================================
    # Public Methods
    # =========================================================================
    
    def _get_id_to_lines_cache_key(self, body_id: int, match_type: str, expected_dataset: Optional[str] = None) -> str:
        """
        Build the cache key for id_to_lines lookups.
        
        Region and max-image settings are intentionally excluded.  The EM
        lookup does not use either setting, so including them only created
        byte-for-byte duplicate files.
        """
        ds_key = expected_dataset if expected_dataset else 'any'
        return f"{body_id}_{match_type}_{ds_key}"

    def id_to_lines(
        self, 
        body_id: int, 
        match_type: str = 'cds',
        expected_dataset: Optional[str] = None,
        _em_image: Any = _EM_IMAGE_NOT_PROVIDED,
    ) -> pd.DataFrame:
        """
        Find driver lines matching a given EM body ID.
        
        Parameters
        ----------
        body_id : int
            The EM body ID to search for.
        match_type : str
            Match algorithm: 'cds', 'pppm', or 'both'. Default: 'cds'
        expected_dataset : str, optional
            Expected dataset name (e.g., 'male-cns:v0.9').
            If provided, will filter EM images to match this dataset.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: line, library, score, image_id, match_type
            Sorted by score descending (cds/pppm) or combined_rank ascending (both)
            
        Example
        -------
        >>> nbf = NeuronBridgeFinder()
        >>> lines = nbf.id_to_lines(636798093)
        >>> print(lines)
        """
        # Validate and normalize match_type
        match_type = self._validate_match_type(match_type)
        
        self._vprint(f"🔍 Searching for lines matching body ID: {body_id}")
        
        # Only the current NeuronBridge-versioned Parquet key is authoritative.
        cache_key = self._get_id_to_lines_cache_key(body_id, match_type, expected_dataset)
        cached = self._load_from_cache('id_to_lines', cache_key)
        if cached is not None:
            return cached

        if match_type == 'both':
            # Keep only raw CDS/PPPM tables on disk.  The combined ranking is
            # deterministic and cheap to derive at query time.
            raw_matches: List[Dict[str, Any]] = []
            missing_types = []
            for algorithm in ('cds', 'pppm'):
                algorithm_key = self._get_id_to_lines_cache_key(
                    body_id, algorithm, expected_dataset
                )
                algorithm_cached = self._load_from_cache('id_to_lines', algorithm_key)
                if algorithm_cached is not None:
                    raw_matches.extend(algorithm_cached.to_dict('records'))
                else:
                    missing_types.append(algorithm)

            for algorithm in missing_types:
                algorithm_matches = self._get_em_matches(
                    body_id,
                    match_type=algorithm,
                    expected_dataset=expected_dataset,
                    em_image=_em_image,
                )
                if algorithm_matches:
                    algorithm_df = pd.DataFrame(algorithm_matches)
                    self._save_to_cache(
                        'id_to_lines',
                        self._get_id_to_lines_cache_key(body_id, algorithm, expected_dataset),
                        algorithm_df,
                    )
                    raw_matches.extend(algorithm_matches)

            matches = self._sort_matches_by_rank(raw_matches, key_field='line')
        else:
            # Fetch from API with expected dataset
            matches = self._get_em_matches(
                body_id,
                match_type=match_type,
                expected_dataset=expected_dataset,
                em_image=_em_image,
            )
        
        if not matches:
            self._vprint(f"  ℹ️ No matches found for body ID {body_id}")
            cols = ['line', 'library', 'score', 'image_id', 'match_type']
            if match_type == 'both':
                cols.extend(['cds_score', 'pppm_score', 'cds_rank', 'pppm_rank', 'combined_rank'])
            return pd.DataFrame(columns=cols)
        
        df = pd.DataFrame(matches)
        
        # Cache only single-algorithm raw rows.  ``both`` is always derived
        # from those two tables and is never persisted.
        if match_type != 'both':
            self._save_to_cache('id_to_lines', cache_key, df)
        
        self._vprint(f"  ✓ Found {len(df)} matches")
        
        return df
    
    def neuron_to_lines(
        self, 
        query: Union[str, int, List],
        dataset: Optional[Union[str, List[str]]] = None,
        match_type: str = 'cds'
    ) -> Dict[str, pd.DataFrame]:
        """
        Find driver lines matching neurons specified by bodyId, instance, or type.
        
        Parameters
        ----------
        query : str, int, or list
            - If int: treated as bodyId
            - If str: searched in instance and type columns (supports regex)
            - If list: each item processed individually
        dataset : str, list of str, or None
            Specific dataset(s) to search (e.g., 'hemibrain:v1.2.1', 'male-cns:v0.9').
            Can be a single string or a list of datasets to search multiple.
            If None, searches all available datasets.
        match_type : str
            Match algorithm: 'cds', 'pppm', or 'both'. Default: 'cds'
            
        Returns
        -------
        dict
            Dictionary mapping bodyId -> DataFrame of matching lines
            All results sorted by score descending (cds/pppm) or combined_rank ascending (both)
            
        Example
        -------
        >>> nbf = NeuronBridgeFinder()
        >>> results = nbf.neuron_to_lines('MBON01')
        >>> for body_id, lines_df in results.items():
        ...     print(f"Body ID {body_id}: {len(lines_df)} matches")
        """
        # Validate and normalize match_type
        match_type = self._validate_match_type(match_type)
        
        self._vprint(f"🔍 Searching for lines matching query: {query}")
        
        # Find matching bodyIds (now returns list of dicts with bodyId and dataset)
        body_info_list = self._find_bodyIds_by_query(query, dataset)
        
        if not body_info_list:
            self._vprint(f"  ⚠️ No neurons found matching query: {query}")
            return {}
        
        self._vprint(f"  Found {len(body_info_list)} neurons to search")
        
        results = {}
        skipped_count = 0
        
        # =====================================================================
        # Phase 1: Partition neurons into cached vs uncached using in-memory index
        # =====================================================================
        cached_neurons = []
        uncached_neurons = []
        
        for body_info in body_info_list:
            body_id = body_info['bodyId']
            expected_ds = body_info.get('dataset', 'unknown')
            cache_key = self._get_id_to_lines_cache_key(int(body_id), match_type, expected_ds)
            
            if self._is_cached('id_to_lines', cache_key):
                cached_neurons.append(body_info)
            else:
                uncached_neurons.append(body_info)
        
        self._vprint(f"  📁 Cache status: {len(cached_neurons)} cached, {len(uncached_neurons)} need API fetch")
        
        # Determine if parallel processing should be used
        use_parallel = self.max_workers > 1 and len(uncached_neurons) > 1
        
        # =====================================================================
        # Pre-validate ALL uncached neurons in parallel (single batch of API calls)
        # =====================================================================
        validation_results = {}  # body_id -> {'dataset', 'em_image'}
        if uncached_neurons:
            self._vprint(f"  🔄 Pre-validating {len(uncached_neurons)} neurons against NeuronBridge API...")
            validation_results = self._validate_body_ids_parallel(
                uncached_neurons, 
                max_workers=min(self.max_workers * 2, 16)  # Use more workers for validation (lightweight API calls)
            )
            self._vprint(f"  ✓ Validation complete")
        
        # Worker function for cached neurons (no API validation needed)
        def load_cached_neuron(body_info):
            """Load a cached neuron - no API calls needed."""
            body_id = body_info['bodyId']
            expected_ds = body_info.get('dataset', 'unknown')
            
            try:
                lines_df = self.id_to_lines(
                    int(body_id), 
                    match_type=match_type,
                    expected_dataset=expected_ds
                )
                if not lines_df.empty:
                    lines_df = lines_df.copy()
                    lines_df['source_dataset'] = expected_ds
                    lines_df['source_bodyId'] = body_id
                return body_id, lines_df, False  # skipped=False
            except Exception as e:
                return body_id, pd.DataFrame(), False
        
        # Worker function for uncached neurons (uses PRE-VALIDATED results - no API call here)
        def fetch_uncached_neuron(body_info):
            """Process an uncached neuron using pre-validated dataset info."""
            body_id = body_info['bodyId']
            expected_ds = body_info.get('dataset', 'unknown')
            
            # Use pre-validated result (no API call needed here!)
            metadata = validation_results.get(
                body_id,
                {'dataset': expected_ds, 'em_image': None},
            )
            actual_ds = metadata.get('dataset') or expected_ds
            em_image = metadata.get('em_image')

            # Check for dataset and version mismatch.
            if (
                actual_ds
                and expected_ds != 'unknown'
                and not self._datasets_match(expected_ds, actual_ds)
            ):
                return body_id, None, True  # skipped=True
            
            try:
                lines_df = self.id_to_lines(
                    int(body_id), 
                    match_type=match_type,
                    expected_dataset=expected_ds,
                    _em_image=em_image,
                )
                if not lines_df.empty:
                    lines_df = lines_df.copy()
                    lines_df['source_dataset'] = self._canonical_source_dataset(expected_ds, actual_ds)
                    lines_df['source_bodyId'] = body_id
                return body_id, lines_df, False  # skipped=False
            except Exception as e:
                return body_id, pd.DataFrame(), False
        
        # Use progress bar for multiple body IDs
        if HAS_TQDM and self.verbose and len(body_info_list) > 1:
            from tqdm import tqdm as tqdm_progress
            
            # Enable batch mode to suppress individual messages
            self._batch_mode = True
            
            # =====================================================================
            # Phase 2: Load cached neurons using BULK Polars loader (fastest path)
            # =====================================================================
            if cached_neurons:
                if HAS_POLARS and len(cached_neurons) > 1:
                    # Use bulk Polars loader for maximum speed
                    self._vprint(
                        f"  ⏩ Loading {len(cached_neurons)} from cache...",
                        force=True
                    )
                    import time
                    t0 = time.perf_counter()
                    
                    combined_pl, loaded_ids = self._load_cached_neurons_bulk_polars(
                        cached_neurons,
                        match_type,
                        max_workers=16,
                        progress_desc=f"  ⏩ Loading {len(cached_neurons)} from cache"
                    )
                    
                    # Convert to per-neuron results dict (still in Polars until final output)
                    if combined_pl.height > 0:
                        # Group by source_bodyId and convert each group to pandas
                        for body_id in loaded_ids:
                            mask = combined_pl.filter(pl.col('source_bodyId') == body_id)
                            results[body_id] = mask.to_pandas()
                    
                    elapsed = time.perf_counter() - t0
                    self._vprint(f"  ✓ Loaded {len(loaded_ids)} neurons ({combined_pl.height:,} rows) in {elapsed:.2f}s", force=True)
                else:
                    # Fallback to parallel loading for single neuron or no Polars
                    cache_workers = min(16, len(cached_neurons))
                    
                    with ThreadPoolExecutor(max_workers=cache_workers) as executor:
                        futures = {executor.submit(load_cached_neuron, bi): bi for bi in cached_neurons}
                        
                        pbar_cache = tqdm_progress(
                            as_completed(futures),
                            total=len(cached_neurons),
                            desc=f"  ⏩ Loading {len(cached_neurons)} from cache",
                            unit="neuron",
                            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                            ncols=110,
                            position=0,
                            leave=False,
                            file=sys.stdout,
                        )
                        
                        for future in pbar_cache:
                            try:
                                body_id, lines_df, skipped = future.result()
                                if skipped:
                                    skipped_count += 1
                                else:
                                    results[body_id] = lines_df if lines_df is not None else pd.DataFrame()
                                pbar_cache.set_postfix_str(f"{body_id}")
                            except Exception as e:
                                body_info = futures[future]
                                results[body_info['bodyId']] = pd.DataFrame()
                        
                        pbar_cache.close()
            
            # =====================================================================
            # Phase 3: Fetch uncached neurons from API (parallel if enabled)
            # =====================================================================
            if uncached_neurons:
                if use_parallel:
                    # Parallel processing with ThreadPoolExecutor
                    self._vprint(f"  🚀 Using parallel processing ({min(self.max_workers, len(uncached_neurons))} workers)", force=True)
                    
                    with ThreadPoolExecutor(max_workers=min(self.max_workers, len(uncached_neurons))) as executor:
                        # Submit all tasks - use fetch_uncached_neuron (with API validation)
                        futures = {executor.submit(fetch_uncached_neuron, bi): bi for bi in uncached_neurons}
                        
                        # Process results with progress bar
                        pbar = tqdm_progress(
                            as_completed(futures),
                            total=len(uncached_neurons),
                            desc=f"  🌐 Fetching {len(uncached_neurons)} from API",
                            unit="neuron",
                            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                            ncols=110,
                            position=0,
                            leave=False,
                            file=sys.stdout,
                        )
                        
                        for future in pbar:
                            try:
                                body_id, lines_df, skipped = future.result()
                                if skipped:
                                    skipped_count += 1
                                else:
                                    results[body_id] = lines_df if lines_df is not None else pd.DataFrame()
                                pbar.set_postfix_str(f"{body_id}")
                            except Exception as e:
                                # Get body_info from futures dict for error handling
                                body_info = futures[future]
                                results[body_info['bodyId']] = pd.DataFrame()
                        
                        pbar.close()
                else:
                    # Sequential processing with progress bar
                    pbar = tqdm_progress(
                        uncached_neurons,
                        desc=f"  🌐 Fetching {len(uncached_neurons)} from API",
                        unit="neuron",
                        bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                        ncols=110,
                        position=0,
                        leave=False,
                        file=sys.stdout,
                    )
                    
                    for body_info in pbar:
                        body_id, lines_df, skipped = fetch_uncached_neuron(body_info)
                        if skipped:
                            skipped_count += 1
                        else:
                            results[body_id] = lines_df if lines_df is not None else pd.DataFrame()
                        pbar.set_postfix_str(f"{body_id}")
                    
                    pbar.close()
            
            self._batch_mode = False
            
            if skipped_count > 0:
                self._vprint(f"  ℹ️  Skipped {skipped_count} body IDs due to dataset mismatch")
        else:
            # Single neuron or no tqdm - use parallel validation for uncached
            # Pre-validate uncached neurons in parallel
            seq_validation_results = {}
            seq_uncached = [bi for bi in body_info_list 
                           if not self._is_cached('id_to_lines', 
                               self._get_id_to_lines_cache_key(int(bi['bodyId']), match_type, bi.get('dataset', 'unknown')))]
            if seq_uncached:
                seq_validation_results = self._validate_body_ids_parallel(seq_uncached)
            
            for i, body_info in enumerate(body_info_list):
                body_id = body_info['bodyId']
                expected_ds = body_info.get('dataset', 'unknown')
                
                # Check if cached - skip validation if so
                cache_key = self._get_id_to_lines_cache_key(int(body_id), match_type, expected_ds)
                is_cached = self._is_cached('id_to_lines', cache_key)
                
                if is_cached:
                    # Cached - just load without API validation
                    try:
                        lines_df = self.id_to_lines(
                            int(body_id), 
                            match_type=match_type,
                            expected_dataset=expected_ds
                        )
                        if not lines_df.empty:
                            lines_df = lines_df.copy()
                            lines_df['source_dataset'] = expected_ds
                            lines_df['source_bodyId'] = body_id
                        results[body_id] = lines_df
                    except Exception as e:
                        self._vprint(f"    ⚠️ Error processing body ID {body_id}: {e}")
                        results[body_id] = pd.DataFrame()
                else:
                    # Uncached - use pre-validated result (no API call here)
                    metadata = seq_validation_results.get(
                        body_id,
                        {'dataset': expected_ds, 'em_image': None},
                    )
                    actual_ds = metadata.get('dataset') or expected_ds
                    em_image = metadata.get('em_image')

                    # Check for dataset and version mismatch.
                    if (
                        actual_ds
                        and expected_ds != 'unknown'
                        and not self._datasets_match(expected_ds, actual_ds)
                    ):
                        self._vprint(f"  ⏭️  Skipping {body_id}: belongs to {actual_ds}, not {expected_ds} (dataset mismatch)")
                        skipped_count += 1
                        continue
                    
                    self._vprint(f"  Processing {i+1}/{len(body_info_list)}: {body_id} ({actual_ds or expected_ds})")
                    try:
                        lines_df = self.id_to_lines(
                            int(body_id), 
                            match_type=match_type,
                            expected_dataset=expected_ds,
                            _em_image=em_image,
                        )
                        if not lines_df.empty:
                            lines_df = lines_df.copy()
                            lines_df['source_dataset'] = self._canonical_source_dataset(expected_ds, actual_ds)
                            lines_df['source_bodyId'] = body_id
                        results[body_id] = lines_df
                    except Exception as e:
                        self._vprint(f"    ⚠️ Error processing body ID {body_id}: {e}")
                        results[body_id] = pd.DataFrame()
            
            if skipped_count > 0:
                self._vprint(f"  ℹ️  Skipped {skipped_count} body IDs due to dataset mismatch")
        
        return results
    
    def line_to_neuron(
        self, 
        line_name: str,
        match_type: str = 'cds',
        top_n: int = -1
    ) -> pd.DataFrame:
        """
        Find EM neurons matching a driver line name.
        
        Parameters
        ----------
        line_name : str
            The driver line name to search for (e.g., 'LH173').
        match_type : str
            Match algorithm: 'cds', 'pppm', or 'both'. Default: 'cds'
        top_n : int
            Maximum number of matches to return. Default: -1 (all matches)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: bodyId, dataset, instance, type, status, 
            score, image_id, lm_sample, match_type, library
            Sorted by score descending (cds/pppm) or combined_rank ascending (both)
            
        Example
        -------
        >>> nbf = NeuronBridgeFinder()
        >>> neurons = nbf.line_to_neuron('LH173')
        >>> print(neurons)
        """
        # Validate and normalize match_type
        match_type = self._validate_match_type(match_type)
        
        # Only print search message if not in batch/progress mode
        # (check if we're being called from within a progress bar loop)
        import inspect
        frame = inspect.currentframe()
        in_progress_context = False
        if frame and frame.f_back:
            # Check if caller's local variables include tqdm iterator
            caller_locals = frame.f_back.f_locals
            in_progress_context = any('tqdm' in str(type(v)).lower() for v in caller_locals.values())
        
        if not in_progress_context:
            self._vprint(f"🔍 Searching for neurons matching line: {line_name}")
        
        # Fetch from the per-image cache (matches are already enriched with
        # dataset info).  A line-level result is deliberately not cached:
        # ``top_n`` is a presentation limit and must never poison a later
        # request for all matches.
        matches = self._get_lm_matches(line_name, match_type=match_type)
        
        if not matches:
            self._vprint(f"  ℹ️ No matches found for line '{line_name}'")
            cols = ['bodyId', 'dataset', 'instance', 'type', 'status', 'score', 
                    'image_id', 'lm_sample', 'match_type', 'library']
            if match_type == 'both':
                cols.extend(['cds_score', 'pppm_score', 'cds_rank', 'pppm_rank', 'combined_rank'])
            return pd.DataFrame(columns=cols)
        
        df = pd.DataFrame(matches)
        
        # Reorder columns - include dataset and rank columns if present
        base_cols = ['bodyId', 'dataset', 'instance', 'type', 'status', 'score', 
                     'image_id', 'lm_sample', 'match_type', 'library']
        rank_cols = ['cds_score', 'pppm_score', 'cds_rank', 'pppm_rank', 'combined_rank']
        cols = [c for c in base_cols if c in df.columns]
        cols.extend([c for c in rank_cols if c in df.columns])
        df = df[cols]
        
        # Apply top_n limit if specified
        if top_n > 0 and len(df) > top_n:
            df = df.head(top_n)
        
        self._vprint(f"  ✓ Found {len(df)} matches")
        
        return df
    
    def save_results(
        self, 
        results: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
        output_path: str,
        include_timestamp: bool = True
    ) -> str:
        """
        Save results to a CSV file.
        
        Parameters
        ----------
        results : DataFrame or dict
            Results from id_to_lines, neuron_to_lines, or line_to_neuron.
        output_path : str
            Output file path.
        include_timestamp : bool
            Whether to include timestamp in filename. Default: True
            
        Returns
        -------
        str
            Path to the saved file.
        """
        if include_timestamp:
            base, ext = os.path.splitext(output_path)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"{base}_{timestamp}{ext}"
        
        if isinstance(results, dict):
            # Combine dict of DataFrames into one
            combined = []
            for body_id, df in results.items():
                if not df.empty:
                    df_copy = df.copy()
                    df_copy.insert(0, 'query_bodyId', body_id)
                    combined.append(df_copy)
            
            if combined:
                final_df = pd.concat(combined, ignore_index=True)
            else:
                final_df = pd.DataFrame()
        else:
            final_df = results
        
        final_df.to_csv(output_path, index=False)
        self._vprint(f"💾 Saved results to: {output_path}")
        
        return output_path
    
    def _save_dataset_categorized_files(
        self,
        neurons_df: pd.DataFrame,
        line_name: str,
        output_path: str,
        verbose: bool = True,
        sort_by: str = 'max_score',
        group_in_by_dataset: bool = True
    ) -> None:
        """
        Save dataset-categorized neuron files and type summary files.

        Creates:
        - {line}_{dataset}_neurons.csv: Neurons for each dataset
        - {line}_{dataset}_types.csv: Type summary with labeled_N and typed_N_in_dataset

        Parameters
        ----------
        neurons_df : pd.DataFrame
            DataFrame with matched neurons (must have 'dataset' column)
        line_name : str
            Driver line name
        output_path : str
            Output directory path
        verbose : bool
            Whether to print progress messages
        group_in_by_dataset : bool
            Group the per-dataset files into a `by_dataset/` subfolder
            (Find EM Neurons run roots; the Compact output-detail pass
            prunes the row-level ones there). Co-Labeling passes False to
            keep its `line_labeled_neurons/` layout unchanged.
        """
        if 'dataset' not in neurons_df.columns:
            return

        # Group by dataset
        for dataset, ds_df in neurons_df.groupby('dataset'):
            # Normalize dataset name for filename (replace : with _)
            ds_filename = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')

            # Per-dataset files are source-data details; Find EM Neurons
            # groups them in one subfolder so the run root carries only the
            # deliverables.
            target_dir = (
                os.path.join(output_path, 'by_dataset')
                if group_in_by_dataset else output_path
            )
            os.makedirs(target_dir, exist_ok=True)

            # Save dataset-specific neurons file
            ds_neurons_file = os.path.join(target_dir, f'{line_name}_{ds_filename}_neurons.csv')
            ds_df.to_csv(ds_neurons_file, index=False)
            if verbose:
                self._vprint(f"   💾 Saved: {ds_neurons_file}")

            # Create type summary
            type_summary = self._create_type_summary(ds_df, dataset, sort_by=sort_by)
            if not type_summary.empty:
                ds_types_file = os.path.join(target_dir, f'{line_name}_{ds_filename}_types.csv')
                type_summary.to_csv(ds_types_file, index=False)
                if verbose:
                    self._vprint(f"   💾 Saved: {ds_types_file}")
        
        # Also create the type-mapped summary CSV
        self._save_type_mapped_csv(neurons_df, line_name, output_path, verbose=verbose, sort_by=sort_by)
    
    def _save_type_mapped_csv(
        self,
        neurons_df: pd.DataFrame,
        line_name: str,
        output_path: str,
        verbose: bool = True,
        sort_by: str = 'max_score'
    ) -> None:
        """
        Create a cross-dataset type-mapped summary CSV.
        
        This file maps types across datasets using CrossDatasetTypeMapper to show
        which types from different datasets correspond to each other.
        
        Creates:
        - {line}_type_mapped.csv: Type mapping with columns for each dataset
        
        Parameters
        ----------
        neurons_df : pd.DataFrame
            DataFrame with matched neurons (must have 'dataset' and 'type' columns)
        line_name : str
            Driver line name
        output_path : str
            Output directory path
        verbose : bool
            Whether to print progress messages
        sort_by : str
            Score metric to sort by: 'max_score', 'median_score', etc.
        """
        if not HAS_TYPE_MAPPER:
            return
        
        if 'dataset' not in neurons_df.columns or 'type' not in neurons_df.columns:
            return
        
        try:
            # Get type mapper instance
            mapper = get_type_mapper()
            if mapper is None:
                return
            
            # Get list of datasets in the data
            datasets = sorted(neurons_df['dataset'].unique())
            if len(datasets) < 2:
                # No need for type mapping with single dataset
                return
            
            # Build a mapping table for all types across datasets
            # First, collect all unique (dataset, type) pairs with their counts and scores
            rows = []
            
            # Group by type and dataset
            for dataset in datasets:
                ds_df = neurons_df[neurons_df['dataset'] == dataset]
                if 'type' not in ds_df.columns:
                    continue
                
                # Create type summary for this dataset
                type_summary = self._create_type_summary(ds_df, dataset, sort_by=sort_by)
                if type_summary.empty:
                    continue
                
                for _, row in type_summary.iterrows():
                    type_name = row['type']
                    if not type_name or pd.isna(type_name):
                        continue
                    
                    # Canonical merge key via the shared validity resolver:
                    # licensed renames merge; a CONFLICTED type stays
                    # dataset-scoped and never groups with another dataset's
                    # same-named rows in the exported summary.
                    try:
                        from comparison.type_resolver import canonical_merge_key
                    except ImportError:
                        from src.comparison.type_resolver import (
                            canonical_merge_key,
                        )
                    canonical = canonical_merge_key(
                        mapper, type_name, dataset).key
                    
                    rows.append({
                        'canonical_type': canonical,
                        'dataset': dataset,
                        'original_type': type_name,
                        'labeled_N': row.get('labeled_N', 0),
                        'max_score': row.get('max_score'),
                        'median_score': row.get('median_score'),
                        'typed_N_in_dataset': row.get('typed_N_in_dataset', 0),
                    })
            
            if not rows:
                return
            
            # Convert to DataFrame and pivot to wide format
            df = pd.DataFrame(rows)
            
            # Create summary grouped by canonical type
            summary_rows = []
            for canonical, group in df.groupby('canonical_type'):
                summary_row = {'canonical_type': canonical}
                total_labeled = 0
                max_max_score = None
                
                for dataset in datasets:
                    ds_group = group[group['dataset'] == dataset]
                    ds_filename = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
                    
                    if len(ds_group) > 0:
                        # Use the first (or only) matching row
                        ds_row = ds_group.iloc[0]
                        summary_row[f'{ds_filename}_type'] = ds_row['original_type']
                        summary_row[f'{ds_filename}_N'] = ds_row['labeled_N']
                        summary_row[f'{ds_filename}_max_score'] = ds_row['max_score']
                        total_labeled += ds_row['labeled_N']
                        if ds_row['max_score'] is not None:
                            if max_max_score is None or ds_row['max_score'] > max_max_score:
                                max_max_score = ds_row['max_score']
                    else:
                        summary_row[f'{ds_filename}_type'] = ''
                        summary_row[f'{ds_filename}_N'] = 0
                        summary_row[f'{ds_filename}_max_score'] = None
                
                summary_row['total_labeled_N'] = total_labeled
                summary_row['best_max_score'] = max_max_score
                summary_rows.append(summary_row)
            
            if not summary_rows:
                return
            
            # Create final DataFrame
            result_df = pd.DataFrame(summary_rows)
            
            # Sort by best_max_score descending, then by total_labeled_N descending
            if 'best_max_score' in result_df.columns:
                result_df = result_df.sort_values(
                    ['best_max_score', 'total_labeled_N'],
                    ascending=[False, False]
                )
            else:
                result_df = result_df.sort_values('total_labeled_N', ascending=False)
            
            # Reorder columns in the user-requested order:
            # male-cns type (canonical), best_max_score, FAFB type, hemibrain type, manc type,
            # then all scores, then all labeled_N counts
            
            # Define dataset order priority for columns
            dataset_order = [
                'male-cns_v0_9',
                'flywire_FAFB_v783',
                'hemibrain_v1_2_1',
                'manc_v1_0',
                'manc_v1_2_1',
                'banc_v626',
            ]
            
            # Get available datasets in the data
            available_ds = []
            for ds_prefix in dataset_order:
                if f'{ds_prefix}_type' in result_df.columns:
                    available_ds.append(ds_prefix)
            
            # Add any other datasets not in the priority list
            for col in result_df.columns:
                if col.endswith('_type') and col != 'canonical_type':
                    ds_prefix = col.replace('_type', '')
                    if ds_prefix not in available_ds:
                        available_ds.append(ds_prefix)
            
            # Build ordered columns:
            # 1. canonical_type (male-cns type)
            # 2. best_max_score
            # 3. Dataset types in order (FAFB, hemibrain, manc, etc.)
            # 4. Dataset max scores in order
            # 5. Dataset labeled_N counts in order
            # 6. total_labeled_N
            ordered_cols = ['canonical_type', 'best_max_score']
            
            # Add type columns (skip male-cns since canonical_type IS the male-cns type)
            for ds_prefix in available_ds:
                if ds_prefix != 'male-cns_v0_9':  # canonical_type already represents male-cns
                    if f'{ds_prefix}_type' in result_df.columns:
                        ordered_cols.append(f'{ds_prefix}_type')
            
            # Add max_score columns
            for ds_prefix in available_ds:
                if f'{ds_prefix}_max_score' in result_df.columns:
                    ordered_cols.append(f'{ds_prefix}_max_score')
            
            # Add labeled_N columns
            for ds_prefix in available_ds:
                if f'{ds_prefix}_N' in result_df.columns:
                    ordered_cols.append(f'{ds_prefix}_N')
            
            # Add total
            ordered_cols.append('total_labeled_N')
            
            # Only include columns that exist
            ordered_cols = [c for c in ordered_cols if c in result_df.columns]
            result_df = result_df[ordered_cols]
            
            # Save the type-mapped CSV
            type_mapped_file = os.path.join(output_path, f'{line_name}_type_mapped.csv')
            result_df.to_csv(type_mapped_file, index=False)
            if verbose:
                self._vprint(f"   💾 Saved: {type_mapped_file}")
            
        except Exception as e:
            if verbose:
                self._vprint(f"   ⚠️ Could not create type-mapped CSV: {e}")
    
    def _create_type_summary(
        self,
        neurons_df: pd.DataFrame,
        dataset: str,
        sort_by: str = 'max_score'
    ) -> pd.DataFrame:
        """
        Create a type summary DataFrame with labeled_N and typed_N_in_dataset.
        
        Parameters
        ----------
        neurons_df : pd.DataFrame
            DataFrame with matched neurons
        dataset : str
            Dataset name for looking up total type counts
        sort_by : str, default 'max_score'
            Score metric to sort by: 'max_score', 'median_score', 'Q3_score', 'Q1_score', 'avg_score'
            
        Returns
        -------
        pd.DataFrame
            Type summary with columns: type, labeled_N, avg_score, typed_N_in_dataset
        """
        # Determine type column - use 'type' if available, else use bodyId
        if 'type' in neurons_df.columns:
            # For untyped neurons, use 'unknown_{bodyId}' format
            neurons_df = neurons_df.copy()
            neurons_df['type_label'] = neurons_df.apply(
                lambda row: f"unknown_{row['bodyId']}" if pd.isna(row['type']) or row['type'] == '' else row['type'],
                axis=1
            )
        else:
            neurons_df = neurons_df.copy()
            neurons_df['type_label'] = neurons_df['bodyId'].apply(lambda x: f"unknown_{x}")
        
        # Group by type and calculate statistics
        score_col = 'score' if 'score' in neurons_df.columns else None
        
        if score_col:
            type_stats = neurons_df.groupby('type_label').agg(
                labeled_N=('bodyId', 'count'),
                max_score=(score_col, 'max'),
                median_score=(score_col, 'median'),
                Q3_score=(score_col, lambda x: x.quantile(0.75)),
                Q1_score=(score_col, lambda x: x.quantile(0.25)),
                avg_score=(score_col, 'mean')
            ).reset_index()
        else:
            type_stats = neurons_df.groupby('type_label').agg(
                labeled_N=('bodyId', 'count')
            ).reset_index()
            type_stats['max_score'] = None
            type_stats['median_score'] = None
            type_stats['Q3_score'] = None
            type_stats['Q1_score'] = None
            type_stats['avg_score'] = None
        
        type_stats = type_stats.rename(columns={'type_label': 'type'})
        
        # Get total count of each type in the dataset
        type_stats['typed_N_in_dataset'] = type_stats['type'].apply(
            lambda t: self._get_type_count_in_dataset(t, dataset)
        )
        
        # Sort by specified metric descending (primary), then by labeled_N descending (secondary)
        # This ensures top-N types are the ones with highest match scores
        if score_col:
            # Use sort_by if it's a valid column, otherwise default to max_score
            sort_column = sort_by if sort_by in type_stats.columns else 'max_score'
            type_stats = type_stats.sort_values([sort_column, 'labeled_N'], ascending=[False, False])
        else:
            type_stats = type_stats.sort_values('labeled_N', ascending=False)
        
        # Reorder columns: max, median, Q3, Q1, avg
        cols = ['type', 'labeled_N', 'typed_N_in_dataset']
        if score_col:
            cols.insert(2, 'max_score')
            cols.insert(3, 'median_score')
            cols.insert(4, 'Q3_score')
            cols.insert(5, 'Q1_score')
            cols.insert(6, 'avg_score')
        type_stats = type_stats[cols]
        
        return type_stats
    
    def _get_type_count_in_dataset(self, type_name: str, dataset: str) -> int:
        """
        Get the total count of a neuron type in the dataset.
        
        Uses _load_neuron_df_for_dataset() which will pull the dataset from
        NeuPrint if not available locally.
        
        Parameters
        ----------
        type_name : str
            Neuron type name (or bodyId if untyped)
        dataset : str
            Dataset name (e.g., 'hemibrain:v1.2.1' or 'male-cns:v0.9')
            
        Returns
        -------
        int
            Total count of the type in the dataset, or 1 if it's a bodyId or lookup fails
        """
        # If type_name is unknown_{bodyId} or looks like a bodyId (numeric), return 1
        if type_name.startswith('unknown_') or type_name.isdigit():
            return 1
        
        # Try to load the dataset neuron index
        try:
            # Convert dataset name to folder format (e.g., 'hemibrain:v1.2.1' -> 'hemibrain_v1_2_1')
            dataset_folder = self._dataset_name_to_folder(dataset)
            
            if dataset_folder is None:
                # Try direct conversion if mapping fails
                dataset_folder = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_').replace('-', '_')
            
            # Use _load_neuron_df_for_dataset which handles loading and pulling if needed
            neuron_df = self._load_neuron_df_for_dataset(dataset_folder)
            
            if neuron_df is not None and 'type' in neuron_df.columns:
                # Count neurons of this type (case-sensitive match)
                count = len(neuron_df[neuron_df['type'] == type_name])
                return count if count > 0 else 1
        except Exception:
            pass
        
        return 1  # Default if lookup fails
    
    def _apply_type_filter(
        self,
        type_list: List[Tuple[int, str]],
        type_filter: Optional[Dict[str, Union[str, List[str]]]]
    ) -> List[Tuple[int, str]]:
        """
        Filter type names based on filter criteria while preserving original ranks.
        
        Filter logic:
        - Multiple values within a key (list): OR logic (match any)
        - Across different keys: AND logic (must match all filter types)
        
        Parameters
        ----------
        type_list : list of (rank, type_name) tuples
            List of types with their original ranks (before filtering)
        type_filter : dict, optional
            Filter criteria with keys: 'contains', 'startswith', 'endswith', 'regex'
            Each value can be a string or list of strings.
            Example: {'contains': ['DN', 'AN'], 'startswith': 'M'}
            
        Returns
        -------
        list of (rank, type_name) tuples
            Filtered list preserving original ranks
        """
        if type_filter is None or not type_filter:
            return type_list
        
        import re
        
        def matches_filter(type_name: str) -> bool:
            """Check if type_name matches all filter criteria."""
            # Each key must be satisfied (AND logic across keys)
            for filter_type, filter_values in type_filter.items():
                # Convert single value to list for uniform processing
                if isinstance(filter_values, str):
                    filter_values = [filter_values]
                
                # OR logic within a key: match any value
                matched = False
                
                if filter_type == 'contains':
                    for val in filter_values:
                        if val in type_name:
                            matched = True
                            break
                elif filter_type == 'startswith':
                    for val in filter_values:
                        if type_name.startswith(val):
                            matched = True
                            break
                elif filter_type == 'endswith':
                    for val in filter_values:
                        if type_name.endswith(val):
                            matched = True
                            break
                elif filter_type == 'regex':
                    for val in filter_values:
                        try:
                            if re.search(val, type_name):
                                matched = True
                                break
                        except re.error:
                            self._vprint(f"   ⚠️  Invalid regex pattern: {val}")
                            continue
                else:
                    # Unknown filter type, skip
                    self._vprint(f"   ⚠️  Unknown filter type: {filter_type}")
                    matched = True  # Don't fail on unknown types
                
                if not matched:
                    return False  # AND logic: all keys must match
            
            return True
        
        # Filter while preserving ranks
        filtered = [(rank, name) for rank, name in type_list if matches_filter(name)]
        
        return filtered
    
    def _get_top_types_fallback(
        self,
        ds_df: pd.DataFrame,
        top_n: Optional[int] = None,
        sort_by: str = 'max_score'
    ) -> List[str]:
        """
        Fallback method to get top N types by score when labeling_info is not available.
        
        Parameters
        ----------
        ds_df : pd.DataFrame
            Dataset-filtered DataFrame with 'type_label' and 'score' columns
        top_n : int, optional
            Number of top types to return. If None, returns all types sorted by score.
        sort_by : str, default 'max_score'
            Score metric to sort by: 'max_score', 'avg_score', 'median_score', 'Q3_score', 'Q1_score'
            
        Returns
        -------
        list
            List of top type names (case-sensitive)
        """
        score_col = 'score' if 'score' in ds_df.columns else None
        
        if score_col:
            # Calculate all score metrics
            type_stats = ds_df.groupby('type_label').agg(
                max_score=(score_col, 'max'),
                median_score=(score_col, 'median'),
                Q3_score=(score_col, lambda x: x.quantile(0.75)),
                Q1_score=(score_col, lambda x: x.quantile(0.25)),
                avg_score=(score_col, 'mean'),
                count=('bodyId', 'count')
            ).reset_index()
            
            # Determine sort column
            if sort_by in type_stats.columns:
                sort_column = sort_by
            else:
                # Default to max_score if invalid sort_by
                sort_column = 'max_score'
            
            type_stats = type_stats.sort_values(sort_column, ascending=False)
            if top_n is not None:
                type_stats = type_stats.head(top_n)
            return type_stats['type_label'].tolist()
        else:
            # Fallback to count-based if no score column
            type_counts = ds_df['type_label'].value_counts()
            if top_n is not None:
                type_counts = type_counts.head(top_n)
            return type_counts.index.tolist()
    
    def _get_type_scores(
        self,
        ds_df: pd.DataFrame,
        sort_by: str = 'max_score'
    ) -> Dict[str, float]:
        """
        Compute score for each type based on the sort_by metric.
        
        Parameters
        ----------
        ds_df : pd.DataFrame
            Dataset-filtered DataFrame with 'type_label' and 'score' columns
        sort_by : str, default 'max_score'
            Score metric: 'max_score', 'avg_score', 'median_score', 'Q3_score', 'Q1_score'
            
        Returns
        -------
        dict
            Dictionary mapping type_label to score value
        """
        score_col = 'score' if 'score' in ds_df.columns else None
        
        if score_col:
            # Calculate the score metric based on sort_by
            if sort_by == 'max_score':
                type_scores = ds_df.groupby('type_label')[score_col].max()
            elif sort_by == 'median_score':
                type_scores = ds_df.groupby('type_label')[score_col].median()
            elif sort_by == 'Q3_score':
                type_scores = ds_df.groupby('type_label')[score_col].quantile(0.75)
            elif sort_by == 'Q1_score':
                type_scores = ds_df.groupby('type_label')[score_col].quantile(0.25)
            elif sort_by == 'avg_score':
                type_scores = ds_df.groupby('type_label')[score_col].mean()
            else:
                # Default to max_score if invalid sort_by
                type_scores = ds_df.groupby('type_label')[score_col].max()
            
            return type_scores.to_dict()
        else:
            # No scores available - return empty dict (will use default alpha)
            return {}
    
    def _compute_layer_alphas(
        self,
        layer_scores: List[float],
        base_alpha: float = 0.2
    ) -> List[float]:
        """
        Compute alpha values for each layer based on relative scores.
        
        Alpha = base_alpha * (score_i / max_score)
        
        Parameters
        ----------
        layer_scores : list
            Score for each layer
        base_alpha : float, default 0.2
            Maximum alpha value for the highest-scoring layer
            
        Returns
        -------
        list
            Alpha value for each layer
        """
        if not layer_scores:
            return []
        
        max_score = max(layer_scores) if layer_scores else 1.0
        if max_score == 0:
            max_score = 1.0  # Avoid division by zero
        
        alphas = [base_alpha * (score / max_score) for score in layer_scores]
        return alphas
    
    def _hex_to_rgba(self, hex_color: str, alpha: float = 1.0) -> Tuple[int, int, int, float]:
        """
        Convert a hex color string to RGBA tuple.
        
        Parameters
        ----------
        hex_color : str
            Hex color string like '#ff0000' or 'ff0000'
        alpha : float, default 1.0
            Alpha value (0.0 to 1.0)
            
        Returns
        -------
        tuple
            RGBA tuple (r, g, b, alpha) with r,g,b as 0-255 integers
        """
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b, alpha)
    
    def _create_neuron_colors_with_alpha(
        self,
        n_layers: int,
        layer_alphas: List[float]
    ) -> List[Tuple[int, int, int, float]]:
        """
        Create neuron colors with explicit alpha values for each layer.
        
        Uses Bokeh's Category10 or Category20 palette depending on layer count.
        
        Parameters
        ----------
        n_layers : int
            Number of layers
        layer_alphas : list
            Alpha value for each layer
            
        Returns
        -------
        list
            List of RGBA tuples, one per layer
        """
        # Choose palette based on number of layers
        if n_layers <= 10:
            base_palette = bokeh.palettes.Category10[10]
        else:
            base_palette = bokeh.palettes.Category20[20]
        
        # Create RGBA colors with per-layer alpha
        colors = []
        for i in range(n_layers):
            hex_color = base_palette[i % len(base_palette)]
            alpha = layer_alphas[i] if i < len(layer_alphas) else 0.2
            rgba = self._hex_to_rgba(hex_color, alpha)
            colors.append(rgba)
        
        return colors
    
    def _visualize_top_types(
        self,
        combined_df: pd.DataFrame,
        top_n: int,
        output_path: str,
        per_dataset: bool = True,
        source_line: str = '',
        visualize_by: str = 'type',
        generate_individual_profiles: Union[bool, List[str]] = None,
        pdf_images_per_page: Tuple[int, int] = (4, 3),
        background_color: str = 'white',
        labeling_info: Optional[pd.DataFrame] = None,
        type_filter: Optional[Dict[str, Union[str, List[str]]]] = None,
        datasets_to_visualize: Union[str, List[str]] = 'all',
        sort_by: str = 'max_score',
        visualization_settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Visualize top N types/bodyIds per dataset using VisualizeSkeleton.
        
        Parameters
        ----------
        combined_df : pd.DataFrame
            Combined DataFrame with all matched neurons
        top_n : int
            Number of top types/bodyIds to visualize per dataset
        output_path : str
            Output directory for visualizations
        per_dataset : bool
            If True, create separate visualization per dataset
        source_line : str
            Source line name for folder naming
        visualize_by : str
            How to organize: 'type' (merge) or 'bodyId' (individual)
        generate_individual_profiles : list of str, bool, or None
            Formats to generate: ['pdf'], ['pptx'], or ['pdf', 'pptx']
            Set to False or None to disable generation
        pdf_images_per_page : tuple
            (columns, rows) for PDF layout
        background_color : str, default 'white'
            Background color for 3D visualization
        labeling_info : pd.DataFrame, optional
            DataFrame with case-sensitive types and dataset column for filtering.
            Columns: type, dataset, {line1_score}, {line2_score}, ...
        type_filter : dict, optional
            Filter neuron types by name pattern. When specified, gets ALL types
            first, applies filter, then takes top N from filtered results.
            Keys: 'contains', 'startswith', 'endswith', 'regex'
            Values: str or list of str
            Logic: OR within same key, AND across keys
            Original ranks are preserved in output labels (r{N} reflects rank
            before filtering, allowing tracking of original ranking position).
        datasets_to_visualize : str or list, default 'all'
            Constrain which datasets to visualize. 'all' for all datasets.
        sort_by : str, default 'max_score'
            Score metric to sort types by when ranking for top N selection:
            - 'max_score': Maximum score among neurons of this type (default)
            - 'median_score': Median score among neurons of this type
            - 'Q3_score': 75th percentile (Q3) score among neurons of this type
            - 'Q1_score': 25th percentile (Q1) score among neurons of this type
            - 'avg_score': Average score among neurons of this type
        """
        try:
            from visualize_skeleton import VisualizeSkeleton
        except ImportError:
            try:
                from .visualize_skeleton import VisualizeSkeleton
            except ImportError:
                self._vprint("⚠️  VisualizeSkeleton not available for visualization")
                return
        
        if 'dataset' not in combined_df.columns:
            self._vprint("⚠️  No dataset column for visualization")
            return
        
        mode_label = 'types' if visualize_by == 'type' else 'bodyIds'
        self._vprint(f"\n🎨 Visualizing top {top_n} {mode_label}...")
        
        # Get all unique datasets from results
        all_datasets = combined_df['dataset'].unique().tolist()
        
        # Apply datasets_to_visualize filter
        if datasets_to_visualize is None or datasets_to_visualize == 'all':
            datasets = all_datasets
        elif isinstance(datasets_to_visualize, str):
            # Single dataset specified
            datasets = [d for d in all_datasets if d == datasets_to_visualize]
            if not datasets:
                self._vprint(f"   ⚠️  Dataset '{datasets_to_visualize}' not found in results")
                self._vprint(f"   Available datasets: {all_datasets}")
                return
        else:
            # List of datasets specified
            datasets = [d for d in all_datasets if d in datasets_to_visualize]
            if not datasets:
                self._vprint(f"   ⚠️  None of the specified datasets found in results")
                self._vprint(f"   Requested: {datasets_to_visualize}")
                self._vprint(f"   Available: {all_datasets}")
                return
            skipped = [d for d in datasets_to_visualize if d not in all_datasets]
            if skipped:
                self._vprint(f"   ⚠️  Skipping unavailable datasets: {skipped}")
        
        if type_filter:
            self._vprint(f"   🔍 Type filter: {type_filter}")
        if datasets_to_visualize not in (None, 'all'):
            self._vprint(f"   📂 Visualizing datasets: {datasets}")
        
        # Pre-load all required datasets (pull from NeuPrint if not available locally)
        self._vprint(f"\n📦 Loading datasets for visualization...")
        loaded_datasets = self._ensure_datasets_loaded(datasets)
        
        if not loaded_datasets:
            self._vprint("   ⚠️  No datasets could be loaded for visualization")
            return
        
        self._vprint(f"   ✓ {len(loaded_datasets)} dataset(s) ready for visualization")
        
        for dataset in datasets:
            ds_df = combined_df[combined_df['dataset'] == dataset].copy()
            
            if visualize_by == 'type':
                # Group by type
                # Create type label (use 'unknown_{bodyId}' for untyped)
                if 'type' in ds_df.columns:
                    ds_df['type_label'] = ds_df.apply(
                        lambda row: f"unknown_{row['bodyId']}" if pd.isna(row['type']) or row['type'] == '' else row['type'],
                        axis=1
                    )
                else:
                    ds_df['type_label'] = ds_df['bodyId'].apply(lambda x: f"unknown_{x}")
                
                # Filter out unknown types for visualization (untyped neurons)
                ds_df_typed = ds_df[~ds_df['type_label'].str.startswith('unknown_')]
                if ds_df_typed.empty:
                    self._vprint(f"   ⚠️  {dataset}: No typed neurons for visualization")
                    continue
                ds_df = ds_df_typed
                
                # Get top N types using labeling_info if available (case-sensitive, properly sorted)
                # Otherwise fallback to score-based ranking using sort_by metric
                # If type_filter is set, get ALL types first, then filter, then take top N
                get_all_types = type_filter is not None and type_filter
                
                if labeling_info is not None and not labeling_info.empty and 'dataset' in labeling_info.columns:
                    # Filter labeling_info for this dataset
                    ds_labeling = labeling_info[labeling_info['dataset'] == dataset].copy()
                    if not ds_labeling.empty:
                        # labeling_info is already sorted by quality (complete types first, then by min_score)
                        if get_all_types:
                            # Get all types for filtering
                            all_types = ds_labeling['type'].tolist()
                        else:
                            # Get only top N types
                            all_types = ds_labeling['type'].head(top_n).tolist()
                        self._vprint(f"   📋 Using labeling_info for {dataset}: {len(all_types)} candidate types")
                    else:
                        # Fallback if no labeling info for this dataset
                        all_types = self._get_top_types_fallback(ds_df, None if get_all_types else top_n, sort_by=sort_by)
                else:
                    # Fallback to score-based ranking
                    all_types = self._get_top_types_fallback(ds_df, None if get_all_types else top_n, sort_by=sort_by)
                
                if not all_types:
                    continue
                
                # Compute type scores for alpha calculation
                type_scores = self._get_type_scores(ds_df, sort_by=sort_by)
                
                # Create list of (original_rank, type_name) tuples
                ranked_types = [(i, name) for i, name in enumerate(all_types, start=1)]
                
                # Apply type filter if specified
                if type_filter:
                    filtered_types = self._apply_type_filter(ranked_types, type_filter)
                    if not filtered_types:
                        self._vprint(f"   ⚠️  {dataset}: No types match filter criteria")
                        continue
                    self._vprint(f"   🔍 Filter applied: {len(filtered_types)}/{len(ranked_types)} types match")
                    # Take top N from filtered results (preserving original ranks)
                    top_ranked_types = filtered_types[:top_n]
                else:
                    top_ranked_types = ranked_types[:top_n]
                
                # Build neuron_layers as nested list (one sublist per type)
                # Use r{rank}_{type}_x{N} format for legend names (rank is ORIGINAL rank before filtering)
                # Also track layer_scores for alpha calculation
                neuron_layers = []
                layer_names = []
                layer_scores = []  # Track score for each layer
                
                for original_rank, type_name in top_ranked_types:
                    # Case-sensitive type matching
                    type_neurons = ds_df[ds_df['type_label'] == type_name]['bodyId'].tolist()
                    # FlyWire IDs remain exact strings. NeuPrint retains its
                    # historical integer body-ID representation.
                    type_neurons = [
                        normalize_flywire_body_id(n)
                        if is_local_connectome_dataset(dataset)
                        else _to_int_bodyid(n)
                        for n in type_neurons
                    ]

                    # Per-type render cap: large types would swamp the scene
                    if len(type_neurons) > TYPE_RENDER_MEMBER_CAP:
                        self._vprint(
                            f"   ⚠️ {dataset} r{original_rank} '{type_name}': "
                            f"{len(type_neurons)} members capped to "
                            f"{TYPE_RENDER_MEMBER_CAP} for visualization")
                        type_neurons = type_neurons[:TYPE_RENDER_MEMBER_CAP]

                    if len(type_neurons) > 0:
                        neuron_layers.append(type_neurons)
                        # Create legend name: r{rank}_{type}_x{N} - using ORIGINAL rank
                        n_neurons = len(type_neurons)
                        layer_names.append(f"r{original_rank}_{type_name}_x{n_neurons}")
                        # Track score for this type (default to 1.0 if not available)
                        layer_scores.append(type_scores.get(type_name, 1.0))
                
                if not neuron_layers:
                    continue
                
                self._vprint(f"   📊 {dataset}: {len(neuron_layers)} types, {sum(len(l) for l in neuron_layers)} neurons")
            
            else:  # visualize_by == 'bodyId'
                # Get top N bodyIds by score, but group by type for visualization
                # Each type becomes a layer, but legend_mode='single' shows individual neurons
                score_col = 'score' if 'score' in ds_df.columns else None
                
                # Create type label (use 'unknown_{bodyId}' for untyped)
                if 'type' in ds_df.columns:
                    ds_df['type_label'] = ds_df.apply(
                        lambda row: f"unknown_{row['bodyId']}" if pd.isna(row['type']) or row['type'] == '' else row['type'],
                        axis=1
                    )
                else:
                    ds_df['type_label'] = ds_df['bodyId'].apply(lambda x: f"unknown_{x}")
                
                # Filter out unknown types for visualization (untyped neurons)
                ds_df_typed = ds_df[~ds_df['type_label'].str.startswith('unknown_')]
                if ds_df_typed.empty:
                    self._vprint(f"   ⚠️  {dataset}: No typed neurons for visualization")
                    continue
                ds_df = ds_df_typed
                
                # Sort by score - get ALL bodyIds first if filter is set, then filter, then limit
                if score_col:
                    ds_df_sorted = ds_df.sort_values(score_col, ascending=False)
                else:
                    ds_df_sorted = ds_df
                
                # Compute type scores for alpha calculation
                type_scores = self._get_type_scores(ds_df, sort_by=sort_by)
                
                # Group ALL bodyIds by type for layer organization
                # Track the minimum rank (best score position) for each type for sorting
                type_to_bodyids = {}
                type_min_rank = {}  # Track minimum rank (best) for each type
                for rank_idx, (_, row) in enumerate(ds_df_sorted.iterrows(), start=1):
                    bodyid = row['bodyId']
                    bodyid_val = (
                        normalize_flywire_body_id(bodyid)
                        if is_local_connectome_dataset(dataset)
                        else _to_int_bodyid(bodyid)
                    )
                    type_label = row['type_label']
                    
                    if type_label not in type_to_bodyids:
                        type_to_bodyids[type_label] = []
                        type_min_rank[type_label] = rank_idx  # First occurrence is minimum rank
                    type_to_bodyids[type_label].append(bodyid_val)
                
                # Sort types by their minimum rank (best ranked type first)
                sorted_types = sorted(type_to_bodyids.keys(), key=lambda t: type_min_rank[t])
                
                # Apply type filter if specified
                if type_filter:
                    # Create list of (rank, type_name) tuples for filtering
                    ranked_types = [(type_min_rank[t], t) for t in sorted_types]
                    filtered_types = self._apply_type_filter(ranked_types, type_filter)
                    if not filtered_types:
                        self._vprint(f"   ⚠️  {dataset}: No types match filter criteria")
                        continue
                    self._vprint(f"   🔍 Filter applied: {len(filtered_types)}/{len(ranked_types)} types match")
                    # Take top N from filtered results, extract type names
                    sorted_types = [t for _, t in filtered_types[:top_n]]
                else:
                    # No filter - just take top N types
                    sorted_types = sorted_types[:top_n]
                
                # Build neuron_layers: one layer per type, containing all bodyIds of that type
                # Use r{rank}_{type}_x{N} format for legend names (rank is ORIGINAL rank before filtering)
                # Also track layer_scores for alpha calculation
                neuron_layers = []
                layer_names = []
                layer_scores = []  # Track score for each layer
                
                for type_label in sorted_types:
                    bodyids = type_to_bodyids[type_label]
                    # Per-type render cap: large types would swamp the scene
                    if len(bodyids) > TYPE_RENDER_MEMBER_CAP:
                        self._vprint(
                            f"   ⚠️ {dataset} '{type_label}': "
                            f"{len(bodyids)} members capped to "
                            f"{TYPE_RENDER_MEMBER_CAP} for visualization")
                        bodyids = bodyids[:TYPE_RENDER_MEMBER_CAP]
                    neuron_layers.append(bodyids)
                    # Layer name: r{rank}_{type}_x{N} where rank is the best (min) rank for this type
                    # This is the ORIGINAL rank from score ranking, preserved after filtering
                    rank = type_min_rank[type_label]
                    n_neurons = len(bodyids)
                    layer_names.append(f"r{rank}_{type_label}_x{n_neurons}")
                    # Track score for this type (default to 1.0 if not available)
                    layer_scores.append(type_scores.get(type_label, 1.0))
                
                if not neuron_layers:
                    continue
                
                self._vprint(f"   📊 {dataset}: {len(neuron_layers)} types ({sum(len(l) for l in neuron_layers)} bodyIds)")
            
            # Verify bodyIds exist in local dataset before attempting visualization
            # This prevents the "No neurons matching" error from NeuPrint
            dataset_folder = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
            local_neuron_df = self._load_neuron_df_for_dataset(dataset_folder)
            
            if local_neuron_df is None or local_neuron_df.empty:
                self._vprint(f"   ⚠️  Skipping visualization for {dataset}: local dataset not available")
                continue
            
            # Check if at least some bodyIds can be found
            # Local dataset is loaded with dtype={'bodyId': str}, so convert to strings for comparison
            sample_ids = [bid for layer in neuron_layers for bid in layer[:3]]  # Sample bodyIds
            
            # Local FlyWire tables use canonical exact strings. Keep the
            # NeuPrint comparison behavior unchanged.
            test_ids = [
                normalize_flywire_body_id(bid)
                if is_local_connectome_dataset(dataset)
                else str(_to_int_bodyid(bid))
                for bid in sample_ids
            ]
            
            found_count = local_neuron_df[local_neuron_df['bodyId'].isin(test_ids)].shape[0]
            
            if found_count == 0:
                # Debug: show what we're looking for vs what exists
                self._vprint(f"   ⚠️  Skipping visualization for {dataset}: bodyIds not found in local dataset")
                self._vprint(f"      Looking for: {test_ids[:3]}")
                sample_local = local_neuron_df['bodyId'].head(3).tolist()
                self._vprint(f"      Sample local bodyIds: {sample_local}")
                self._vprint(f"      (This can happen if NeuronBridge matches come from a different dataset version)")
                continue
            
            try:
                # The shared UI panel uses the dataset-aligned template brain
                # by default.  Keep the same default for programmatic callers.
                viz_settings = dict(visualization_settings or {})
                brain_mesh = viz_settings.get('brain_mesh', 'template')
                pipeline = str(
                    viz_settings.get('neuprint_skeleton_pipeline', 'fine')
                    or 'fine'
                ).strip().lower()
                
                # Set skeleton_mesh_simplification based on dataset when the
                # user leaves the shared control at its default. Analysis
                # renders use more simplification than the dedicated Skeleton
                # tab for FAFB (98% vs 95%); NeuPrint fast keeps its fixed
                # simp90 convention.
                skeleton_simplification = (
                    default_analysis_skeleton_mesh_simplification(
                        dataset, pipeline
                    )
                )
                
                # Legend grouping follows ``visualize_by`` -- except when
                # the shared panel's legend preference is the interactive
                # 'tree' mode: its panel re-groups by this flow's custom
                # layer names (layer > type > bodyId), keeping large result
                # scenes collapsed per layer instead of collapsing into bare
                # type names (the earlier Find Homologs regression).
                requested_legend = str(
                    viz_settings.get('legend_mode', '') or ''
                ).strip().lower()
                if requested_legend == 'tree':
                    legend_mode = 'tree'
                else:
                    legend_mode = (
                        'layer' if visualize_by == 'type' else 'single'
                    )
                
                # Determine whether to show VNC mesh based on region and dataset
                # Show VNC when dataset is manc, male-cns, or region is VNC/All
                is_vnc_dataset = ('manc' in dataset.lower() or 'cns' in dataset.lower() 
                                  or 'vnc' in dataset.lower())
                show_vnc_mesh = viz_settings.get(
                    'vnc_mesh', is_vnc_dataset or self.region in ('VNC', 'All')
                )
                
                # Determine export_views based on region setting
                # - region='VNC': only bottom view (shows VNC from below)
                # - region='All': both front and bottom views (shows brain + VNC)
                # - region='Brain': only front view (standard brain view)
                if self.region == 'VNC':
                    export_views = ['bottom']
                elif self.region == 'All':
                    export_views = ['front', 'bottom']
                else:  # 'Brain' or default
                    export_views = ['front']
                
                # Custom folder name: plot-3d_{dataset_folder} (VisualizeSkeleton prepends 'plot-3d_')
                custom_saveas = dataset_folder
                
                # Compute per-layer alpha based on relative scores.  An
                # explicit palette/opacity in the advanced panel takes
                # precedence over the historical score-based defaults.
                # Alpha = 0.2 * (score_i / max_score) so higher-scoring types are more visible
                layer_alphas = self._compute_layer_alphas(layer_scores, base_alpha=0.2)
                neuron_colors = self._create_neuron_colors_with_alpha(len(neuron_layers), layer_alphas)
                if viz_settings.get('neuron_colors'):
                    neuron_colors = viz_settings['neuron_colors']
                if viz_settings.get('export_views', True):
                    effective_export_views = export_views
                else:
                    effective_export_views = False
                effective_simplification = viz_settings.get(
                    'skeleton_mesh_simplification', skeleton_simplification
                )
                if effective_simplification is None:
                    effective_simplification = skeleton_simplification
                effective_background = viz_settings.get(
                    'background_color', background_color
                )
                default_cache_neurons = (
                    True if is_local_connectome_dataset(dataset)
                    else pipeline not in {
                        'fast', 'direct', 'artistic', 'fine_opt1'
                    }
                )
                viz_kwargs = {
                    'dataset': dataset,
                    'output_dir': output_path,
                    'neuron_layers': neuron_layers,
                    'custom_layer_names': layer_names,
                    'server': self.neuprint_server,
                    'token': self.neuprint_token,
                    'saveas': custom_saveas,
                    'include_timestamp': False,
                    'skip_synapse': viz_settings.get('skip_synapse', True),
                    'neuron_colors': neuron_colors,
                    # NeuronBridge result scenes default to line mode for a
                    # quick overview. Users can opt into tube/fine through
                    # the shared visualization settings or Skeleton tab.
                    'skeleton_mode': viz_settings.get('skeleton_mode', 'line'),
                    'legend_mode': legend_mode,
                    'brain_mesh': brain_mesh,
                    'vnc_mesh': show_vnc_mesh,
                    'export_views': effective_export_views,
                    'skeleton_mesh_simplification': effective_simplification,
                    'roi_mesh_simplification': 0.95,
                    'background_color': effective_background,
                    'cache_neurons': viz_settings.get(
                        'cache_neurons', default_cache_neurons
                    ),
                    'cache_synapses': viz_settings.get('cache_synapses', True),
                    'smooth_skeleton': viz_settings.get('smooth_skeleton', False),
                    'show_soma': viz_settings.get('show_soma', True),
                    'show_connectors': viz_settings.get('show_connectors', False),
                    'show_fig': viz_settings.get('show_fig', False),
                    'export_scale': viz_settings.get('export_scale', 3),
                    'export_method': viz_settings.get('export_method', 'webdriver'),
                    'verbose': 'full',
                }
                for key in (
                    'min_synapse_num', 'synapse_size', 'synapse_mode',
                    'uniform_synapse_size',
                    'synapse_alpha', 'synapse_colors', 'neuron_alpha', 'mesh_roi',
                    'mesh_color', 'mesh_alpha', 'brain_mesh_color',
                    'vnc_mesh_color',
                    'neuprint_skeleton_pipeline',
                    'output_format',
                ):
                    if key in viz_settings:
                        value = viz_settings[key]
                        if key == 'mesh_color' and value == 'auto':
                            continue
                        viz_kwargs[key] = value

                vs = VisualizeSkeleton(
                    **viz_kwargs,
                )
                vs.plot_neurons()
                
                # Generate individual profiles if requested
                # Use region-appropriate views for individual profile generation
                if generate_individual_profiles:
                    # Use same views as export_views for consistency
                    profile_views = effective_export_views or export_views
                    vs.plot_individuals(
                        output_format='png',
                        views=profile_views,
                        scale=viz_settings.get('export_scale', 3),
                        pdf_images_per_page=pdf_images_per_page,
                        pdf_title=f"{source_line} - {dataset}",
                        summary_format=generate_individual_profiles
                    )
                
                self._vprint(f"   ✅ Visualization saved to: {vs.save_folder}")
            except Exception as e:
                self._vprint(f"   ⚠️  Visualization failed for {dataset}: {e}")
    
    # =========================================================================
    # Batch Processing Methods (for simplified script usage)
    # =========================================================================
    
    def find_neurons_batch(
        self,
        line_names: Union[str, List[str]],
        top_n: int = 50,
        match_type: Optional[str] = None,
        output_dir: Optional[str] = None,
        visualize_top_n: int = 0,
        visualize_by: str = 'type',
        visualize_per_dataset: bool = True,
        generate_individual_profiles: Union[bool, List[str]] = None,
        pdf_images_per_page: Tuple[int, int] = (4, 3),
        background_color: str = 'white',
        type_filter: Optional[Dict[str, Union[str, List[str]]]] = None,
        datasets_to_visualize: Union[str, List[str]] = 'all',
        sort_by: str = 'max_score',
        visualization_settings: Optional[Dict[str, Any]] = None,
        min_score: float = 30000.0,
        keep_per_match_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Find EM neurons for multiple driver lines with automatic saving.
        
        This is a convenience method that processes multiple lines, adds source
        information, and optionally saves results to files.
        
        Parameters
        ----------
        line_names : str or list
            Driver line name(s). Can be comma-separated string or list.
        top_n : int
            Maximum matches per line. Default: 50.
        match_type : str, optional
            Match algorithm: 'cds', 'pppm', or 'both'. 
            If None, uses self.match_type. Default: None
        output_dir : str, optional
            Directory to save results. If provided, saves individual and combined CSVs.
        visualize_top_n : int
            Visualize top N types/bodyIds per dataset using 3D skeleton visualization.
            Set to 0 to disable (default). Requires VisualizeSkeleton module.
        visualize_by : str
            How to organize visualization: 'type' or 'bodyId'.
            - 'type': Group neurons by type (legend_mode='layer')
            - 'bodyId': Show individual neurons (legend_mode='single')
            Default: 'type'
        visualize_per_dataset : bool
            If True (default), create separate visualizations per dataset.
            If False, combine all datasets in one visualization.
        generate_individual_profiles : list of str, bool, or None
            Formats to generate: ['pdf'], ['pptx'], or ['pdf', 'pptx']
            Set to False or None to disable generation. Default: None.
        pdf_images_per_page : tuple
            (columns, rows) for PDF layout. Default: (4, 3).
        background_color : str, default 'white'
            Background color for 3D visualization. Can be 'white', 'black',
            or any CSS color (e.g., '#f0f0f0', 'lightgray').
        type_filter : dict, optional
            Filter neuron types for visualization by name pattern.
            Keys: 'contains', 'startswith', 'endswith', 'regex'
            Values: str or list of str patterns
            Multiple patterns within same key use OR logic.
            Multiple keys use AND logic.
            Ranks are preserved from original ranking before filtering.
            Example: {'contains': 'DN', 'startswith': ['IN', 'DN']}
        datasets_to_visualize : str or list, default 'all'
            Constrain which datasets to visualize.
            - 'all' or None: Visualize all datasets found in results
            - List of dataset names: Only visualize specified datasets
            - Single dataset name: Only visualize that dataset
        sort_by : str, default 'max_score'
            Score metric to sort types by when ranking for visualization top N selection:
            - 'max_score': Maximum score among neurons of this type (default)
            - 'median_score': Median score among neurons of this type
            - 'Q3_score': 75th percentile (Q3) score among neurons of this type
            - 'Q1_score': 25th percentile (Q1) score among neurons of this type
            - 'avg_score': Average score among neurons of this type
        min_score : float, default 30000.0
            Score cutoff used for score-based visualization annotations. The
            raw per-line result files always retain the ranked top-N matches,
            including matches below this cutoff.
        keep_per_match_csv : bool, default True
            Output detail: keep the bodyId-level tables ('all_neurons.csv',
            '{line}_neurons.csv', 'by_dataset/*_neurons.csv'). False
            (Compact) removes them once the per-dataset type summaries
            exist; removals are audited in 'cleanup_audit.json'.
            
        Returns
        -------
        pd.DataFrame
            Combined DataFrame with all results, including 'source_line' column.
            
        Output Files
        ------------
        When output_dir is specified:
        - {line}_neurons.csv: All matched neurons for the line
        - {line}_{dataset}_neurons.csv: Neurons categorized by dataset
        - {line}_{dataset}_types.csv: Type summary with labeled_N, max_score, median_score, 
          Q3_score, Q1_score, avg_score, and typed_N_in_dataset
        - all_neurons.csv: Combined results from all searches
        - user_warning_notes.txt: Score-cutoff interpretation notes (when applicable)
            
        Example
        -------
        >>> nbf = NeuronBridgeFinder()
        >>> results = nbf.find_neurons_batch('LH173,VT037867', output_dir='./output')
        >>> # With type filtering
        >>> results = nbf.find_neurons_batch(
        ...     'SS29633',
        ...     visualize_top_n=20,
        ...     type_filter={'contains': 'DN'},
        ...     datasets_to_visualize=['manc:v1.0']
        ... )
        """
        # Use class-level match_type if not specified
        if match_type is None:
            match_type = self.match_type
        else:
            # Validate and normalize match_type using helper
            match_type = self._validate_match_type(match_type)
        
        # Step protocol: the 3D visualization phase is optional (driven by
        # visualize_top_n and an output folder), so the total varies.
        total_steps = 4 if (visualize_top_n > 0 and output_dir) else 3
        self._progress(1, total_steps, "Resolving driver-line queries")
        
        # Parse line names
        if isinstance(line_names, str):
            lines = [ln.strip() for ln in line_names.split(',') if ln.strip()]
        else:
            lines = list(line_names)
        
        if not lines:
            self._vprint("❌ No line names provided")
            return pd.DataFrame()
        
        self._vprint(f"🔍 Finding neurons for {len(lines)} line(s)")
        
        # Create output directory if needed
        output_path = None
        if output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Add line names to folder name
            line_info = '_'.join(lines[:3])  # First 3 lines
            if len(lines) > 3:
                line_info += '_etc'
            # Sanitize folder name (remove special characters)
            line_info = ''.join(c if c.isalnum() or c in '-_' else '_' for c in line_info)
            output_path = os.path.join(output_dir, f'NB-find-neurons_{line_info}_{timestamp}')
            os.makedirs(output_path, exist_ok=True)
            self._vprint(f"   📁 Output folder: {output_path}")
            
            # Save parameters for reproducibility
            self._save_parameters(
                output_path=output_path,
                function_name='find_neurons_batch',
                function_params={
                    'line_names': lines,
                    'top_n': top_n,
                    'match_type': match_type,
                    'visualize_top_n': visualize_top_n,
                    'visualize_by': visualize_by,
                    'visualize_per_dataset': visualize_per_dataset,
                    'generate_individual_profiles': generate_individual_profiles,
                    'pdf_images_per_page': pdf_images_per_page,
                    'type_filter': type_filter,
                    'datasets_to_visualize': datasets_to_visualize,
                    'min_score': min_score,
                    'keep_per_match_csv': keep_per_match_csv,
                }
            )
            self._vprint(f"   💾 Parameters: parameters.json")

            self._save_user_warning_notes(
                output_path,
                [
                    (
                        f"Score cutoff ({min_score:,.0f}) is applied only to score-based "
                        "visualization annotations in Find EM Neurons. Each raw per-line "
                        f"CSV still retains the requested top-N ({top_n}) ranked matches, "
                        "including matches below the cutoff."
                    )
                ] if min_score > 0 else [],
            )
        
        # Process each line
        all_results = []
        self._progress(2, total_steps, "Fetching matching EM neuron records")
        
        # Phase 1: Check cache and identify what needs to be fetched
        if HAS_TQDM and self.verbose and len(lines) > 1:
            from tqdm import tqdm as tqdm_progress
            
            self._vprint("\n📦 Checking cache...")
            cached_lines = []
            uncached_lines = []
            
            cache_pbar = tqdm_progress(
                lines,
                desc="   💾 Loading cache",
                unit="line",
                bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                ncols=110,
                position=0,
                leave=False,
                file=sys.stdout,
            )
            
            for line_name in cache_pbar:
                # Line-level snapshots are no longer authoritative: they
                # duplicate image records and can encode a stale top_n.  The
                # line_to_neuron call below will reuse the per-image Parquet
                # records instead.
                uncached_lines.append(line_name)
                cache_pbar.set_postfix_str(f"↗ {line_name}")
            
            cache_pbar.close()
            
            if cached_lines:
                self._vprint(f"   ✓ Loaded {len(cached_lines)} line result(s) from cache")
            if uncached_lines:
                self._vprint(f"   🌐 Need to fetch {len(uncached_lines)} from API")
            
            # Suppress dataset loading messages during cache processing
            # (datasets will be loaded explicitly before visualization)
            old_suppress = getattr(self, '_suppress_loading_msgs', False)
            self._suppress_loading_msgs = True
            
            # Process cached results first
            for line_name, cached_data in cached_lines:
                cached_data = cached_data.copy()
                cached_data['source_line'] = line_name
                all_results.append(cached_data)
                
                # Save individual cached results if output_path specified
                if output_path:
                    output_file = os.path.join(output_path, f'{line_name}_neurons.csv')
                    cached_data.to_csv(output_file, index=False)
                    
                    # Save dataset-categorized files
                    if 'dataset' in cached_data.columns:
                        self._save_dataset_categorized_files(
                            cached_data, line_name, output_path, verbose=False, sort_by=sort_by
                        )
            
            # Restore loading message setting
            self._suppress_loading_msgs = old_suppress
            
            # Phase 2: Fetch uncached data with parallel processing
            if uncached_lines:
                self._vprint("\n🌐 Fetching new data...")
                
                # Use parallel processing for multiple uncached lines
                use_parallel_lines = self.max_workers > 1 and len(uncached_lines) > 1
                
                if use_parallel_lines:
                    # Parallel fetching with ThreadPoolExecutor
                    self._suppress_loading_msgs = True
                    fetch_results = []
                    fetch_errors = []
                    
                    def fetch_single_line(ln):
                        """Worker function to fetch data for a single line."""
                        try:
                            neurons_df = self.line_to_neuron(
                                ln,
                                top_n=top_n,
                                match_type=match_type
                            )
                            if neurons_df.empty:
                                return (ln, None, None)
                            
                            neurons_df = neurons_df.copy()
                            neurons_df['source_line'] = ln
                            return (ln, neurons_df, None)
                        except Exception as e:
                            return (ln, None, str(e))
                    
                    with ThreadPoolExecutor(max_workers=min(self.max_workers, len(uncached_lines))) as executor:
                        futures = {executor.submit(fetch_single_line, ln): ln for ln in uncached_lines}
                        
                        if HAS_TQDM:
                            iterator = tqdm_progress(
                                as_completed(futures),
                                total=len(uncached_lines),
                                desc="   🔄 Fetching (parallel)",
                                unit="line",
                                bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                                ncols=110,
                                position=0,
                                leave=False,
                                file=sys.stdout,
                            )
                        else:
                            iterator = as_completed(futures)
                        
                        for future in iterator:
                            line_name, neurons_df, error = future.result()
                            if error:
                                fetch_errors.append((line_name, error))
                            elif neurons_df is not None:
                                fetch_results.append((line_name, neurons_df))
                    
                    self._suppress_loading_msgs = False
                    
                    # Process results
                    for line_name, neurons_df in fetch_results:
                        all_results.append(neurons_df)
                        
                        # Save individual results
                        if output_path:
                            output_file = os.path.join(output_path, f'{line_name}_neurons.csv')
                            neurons_df.to_csv(output_file, index=False)
                            
                            # Save dataset-categorized files
                            if 'dataset' in neurons_df.columns:
                                self._save_dataset_categorized_files(
                                    neurons_df, line_name, output_path, verbose=False, sort_by=sort_by
                                )
                    
                    self._vprint(f"   ✓ Fetched {len(fetch_results)}/{len(uncached_lines)} from API (parallel)")
                    if fetch_errors:
                        self._vprint(f"   ⚠️ {len(fetch_errors)} line(s) had errors")
                else:
                    # Sequential fetching (single line or max_workers=1)
                    fetch_pbar = tqdm_progress(
                        uncached_lines,
                        desc="   🔄 Fetching",
                        unit="line",
                        bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                        ncols=110,
                        position=0,
                        leave=False,
                        file=sys.stdout,
                    )
                    
                    for line_name in fetch_pbar:
                        fetch_pbar.set_postfix_str(line_name[:20])
                        
                        try:
                            neurons_df = self.line_to_neuron(
                                line_name,
                                top_n=top_n,
                                match_type=match_type
                            )
                            
                            if neurons_df.empty:
                                continue
                            
                            # Add source line info
                            neurons_df = neurons_df.copy()
                            neurons_df['source_line'] = line_name
                            all_results.append(neurons_df)
                            
                            # Save individual results
                            if output_path:
                                output_file = os.path.join(output_path, f'{line_name}_neurons.csv')
                                neurons_df.to_csv(output_file, index=False)
                                
                                # Save dataset-categorized files
                                if 'dataset' in neurons_df.columns:
                                    self._save_dataset_categorized_files(
                                        neurons_df, line_name, output_path, verbose=False, sort_by=sort_by
                                    )
                            
                        except Exception as e:
                            pass  # Silent in progress mode
                    
                    fetch_pbar.close()
                    self._vprint(f"   ✓ Fetched {len(uncached_lines)} from API")
        
        else:
            # Single line or no progress bar - original behavior
            for idx, line_name in enumerate(lines):
                self._vprint(f"\n📋 Processing: {line_name}")
                
                try:
                    neurons_df = self.line_to_neuron(
                        line_name,
                        top_n=top_n,
                        match_type=match_type
                    )
                    
                    if neurons_df.empty:
                        self._vprint(f"   ⚠️ No matching neurons found")
                        continue
                    
                    # Add source line info
                    neurons_df = neurons_df.copy()
                    neurons_df['source_line'] = line_name
                    all_results.append(neurons_df)
                    
                    self._vprint(f"   ✅ Found {len(neurons_df)} matching neurons")
                    
                    # Show dataset distribution
                    if 'dataset' in neurons_df.columns:
                        datasets = neurons_df['dataset'].value_counts()
                        for ds, count in datasets.items():
                            self._vprint(f"      {ds}: {count}")
                    
                    # Save individual results
                    if output_path:
                        output_file = os.path.join(output_path, f'{line_name}_neurons.csv')
                        neurons_df.to_csv(output_file, index=False)
                        self._vprint(f"   💾 Saved: {output_file}")
                        
                        # Save dataset-categorized files
                        if 'dataset' in neurons_df.columns:
                            self._save_dataset_categorized_files(
                                neurons_df, line_name, output_path, verbose=True, sort_by=sort_by
                            )
                        
                except Exception as e:
                    self._vprint(f"   ❌ Error: {e}")
        
        # Combine results
        if all_results:
            self._progress(3, total_steps, "Aggregating, ranking, and saving neurons")
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Show summary statistics
            self._vprint(f"\n📊 Summary:")
            self._vprint(f"   Total neurons found: {len(combined_df)}")
            self._vprint(f"   From {len(lines)} line(s), {len(all_results)} had matches")
            
            # Dataset distribution
            if 'dataset' in combined_df.columns:
                self._vprint(f"   Dataset distribution:")
                datasets = combined_df['dataset'].value_counts()
                for ds, count in datasets.items():
                    self._vprint(f"      {ds}: {count}")
            
            # Save combined results
            if output_path:
                combined_file = os.path.join(output_path, 'all_neurons.csv')
                combined_df.to_csv(combined_file, index=False)
                self._vprint(f"\n💾 Combined results: {combined_file}")
                
                # Create labeling distribution visualization
                self.visualize_labeling_distribution(
                    data=combined_df,
                    output_path=output_path,
                    score_column='score',
                    label_column='type',
                    title=f"Labeling Score Distribution ({len(lines)} line(s))",
                    show_threshold=min_score if min_score > 0 else None,
                    group_by='source_line' if 'source_line' in combined_df.columns and len(lines) > 1 else None
                )
            
            # Visualize top N types per dataset if requested
            if visualize_top_n > 0 and output_path:
                self._progress(4, 4, "Visualizing top neurons")
                # For multiple lines, visualize each line separately
                if len(lines) > 1:
                    self._vprint(f"\n🎨 Visualizing each line separately (multiple lines detected)...")
                    for line_name in lines:
                        line_df = combined_df[combined_df['source_line'] == line_name]
                        if line_df.empty:
                            continue
                        
                        # Create line-specific output folder
                        line_output_path = os.path.join(output_path, f'viz_{line_name}')
                        os.makedirs(line_output_path, exist_ok=True)
                        
                        self._vprint(f"\n   📍 {line_name}: {len(line_df)} neurons")
                        
                        self._visualize_top_types(
                            combined_df=line_df,
                            top_n=visualize_top_n,
                            output_path=line_output_path,
                            per_dataset=visualize_per_dataset,
                            source_line=line_name,
                            visualize_by=visualize_by,
                            generate_individual_profiles=generate_individual_profiles,
                            pdf_images_per_page=pdf_images_per_page,
                            background_color=background_color,
                            type_filter=type_filter,
                            datasets_to_visualize=datasets_to_visualize,
                            sort_by=sort_by,
                            visualization_settings=visualization_settings,
                        )
                    
                    # Recommend colabeling analysis for multiple lines
                    self._vprint(f"\n" + "="*70)
                    self._vprint(f"💡 TIP: For {len(lines)} driver lines, consider running co-labeling analysis")
                    self._vprint(f"   to find shared neuron types and understand overlap patterns:")
                    self._vprint(f"")
                    self._vprint(f"   >>> nbf.analyze_colabeling(")
                    self._vprint(f"   ...     lines={lines},")
                    self._vprint(f"   ...     output_dir='{output_dir}',")
                    self._vprint(f"   ...     visualize_top_n={visualize_top_n},")
                    self._vprint(f"   ... )")
                    self._vprint(f"="*70)
                else:
                    # Single line - original behavior
                    self._visualize_top_types(
                        combined_df=combined_df,
                        top_n=visualize_top_n,
                        output_path=output_path,
                        per_dataset=visualize_per_dataset,
                        source_line=lines[0],
                        visualize_by=visualize_by,
                        generate_individual_profiles=generate_individual_profiles,
                        pdf_images_per_page=pdf_images_per_page,
                        background_color=background_color,
                        type_filter=type_filter,
                        datasets_to_visualize=datasets_to_visualize,
                        sort_by=sort_by,
                        visualization_settings=visualization_settings,
                    )

            # Output-detail policy (Full keeps everything; Compact drops the
            # bodyId-level match tables once the type aggregates exist).
            if output_path and prune_find_neurons_run is not None:
                try:
                    prune_find_neurons_run(
                        output_path,
                        keep_per_match_csv=keep_per_match_csv,
                    )
                except Exception as exc:
                    self._vprint(f"   ⚠️ Output cleanup skipped: {exc}")

            return combined_df
        
        self._vprint(f"\n⚠️ No neurons found for any of the {len(lines)} line(s)")
        return pd.DataFrame()
    
    def analyze_colabeling(
        self,
        lines: Union[str, List[str]],
        match_type: Optional[str] = None,
        top_n_neurons: int = 50,
        similarity_methods: Union[str, List[str]] = None,
        output_dir: Optional[str] = None,
        generate_report: bool = True,
        visualize: bool = True,
        visualize_top_n: int = 0,
        generate_individual_profiles: Union[bool, List[str]] = None,
        pdf_images_per_page: Tuple[int, int] = (3, 2),
        min_score: float = 30000.0,
        min_type_avg_score: float = 10000.0,
        background_color: str = 'white',
        type_filter: Optional[Dict[str, Union[str, List[str]]]] = None,
        datasets_to_visualize: Union[str, List[str]] = 'all',
        sort_by: str = 'max_score',
        visualize_by: str = 'type',
        visualization_settings: Optional[Dict[str, Any]] = None,
        keep_per_match_csv: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze co-labeling patterns among given driver lines.
        
        This method performs a comprehensive co-labeling analysis to understand
        how different driver lines overlap in their neuron labeling patterns.
        
        Parameters
        ----------
        lines : str or list of str
            Driver line names to analyze. Can be:
            - Single line: 'LH173'
            - Multiple as string: 'LH173,VT037867,SS00731'
            - Multiple as list: ['LH173', 'VT037867', 'SS00731']
        keep_per_match_csv : bool, default True
            Output detail: keep the row-level tables ('line_labeled_neurons/'
            and 'distribution_data_by_neuron.csv'). False (Compact) removes
            them once the similarity matrices/report exist; removals are
            audited in 'cleanup_audit.json'.
        match_type : str, optional
            Match algorithm: 'cds', 'pppm', or 'both'.
            If None, uses self.match_type. Default: None
        top_n_neurons : int
            Number of top neuron matches to consider per line. Default: 50.
        similarity_methods : str or list of str
            Similarity method(s) for co-labeling matrix:
            - 'jaccard': Binary Jaccard similarity (presence/absence of types)
            - 'weighted_jaccard': Jaccard weighted by match scores
            - 'rank_correlation': Spearman correlation of type rankings
            Default: ['jaccard', 'weighted_jaccard']
        output_dir : str, optional
            Directory to save results. If None, returns results without saving.
        generate_report : bool
            Generate a comprehensive HTML/text report. Default: True
        visualize : bool
            Generate heatmap visualizations. Default: True
        visualize_top_n : int
            Visualize top N types per dataset using 3D skeleton. Default: 0 (disabled)
        generate_individual_profiles : list of str, bool, or None
            Formats to generate: ['pdf'], ['pptx'], or ['pdf', 'pptx']
            Set to False or None to disable generation. Default: None
        pdf_images_per_page : tuple
            (columns, rows) for PDF layout. Default: (3, 2)
        min_score : float
            Minimum score for expression-matrix and similarity calculations,
            applied after the ranked top-N matches are retrieved. The raw
            per-line neuron files retain all top-N matches, including rows
            below this cutoff. Default: 30000.0.
        min_type_avg_score : float
            Minimum average score threshold for types in the similarity matrix.
            Types with average score < threshold may be excluded from
            similarity calculations. This does not further filter the
            expression matrix. Default: 10000.0.
        background_color : str, default 'white'
            Background color for 3D visualization. Can be 'white', 'black',
            or any CSS color (e.g., '#f0f0f0', 'lightgray').
        type_filter : dict, optional
            Filter neuron types for visualization by name pattern.
            Uses a dict with filter type as key and pattern(s) as value.
            
            Filter types:
            - 'contains': Match types containing the pattern
            - 'startswith': Match types starting with the pattern
            - 'endswith': Match types ending with the pattern
            - 'regex': Match types matching the regex pattern
            
            Values: str or list of str patterns
            Multiple patterns within same key use OR logic.
            Multiple keys use AND logic.
            The rank r{N} will be preserved from the original ranking before filtering.
            
            Examples:
            - type_filter = {'contains': 'DN'}  # Types containing 'DN'
            - type_filter = {'startswith': ['DN', 'IN']}  # Types starting with 'DN' or 'IN'
            - type_filter = {'endswith': '_R'}  # Types ending with '_R'
            - type_filter = {'regex': r'DN[a-z]\d+'}  # Types matching regex pattern
            - type_filter = {'contains': 'DN', 'endswith': '_R'}  # Both conditions (AND)
            
            Set to None or {} to disable filtering (visualize all top types).
            
            Applies to:
            - 3D skeleton visualization (types shown)
            - expression_matrix_viz.csv and expression_matrix_merged_viz.csv
            - Expression matrix HTML heatmaps
            
            Note: Full data is always saved separately (without filtering).
        datasets_to_visualize : str or list, default 'all'
            Constrain which datasets to visualize for 3D skeleton.
            - 'all' or None: Visualize all datasets found in results
            - List of dataset names: Only visualize specified datasets
            - Single dataset name: Only visualize that dataset
        sort_by : str, default 'max_score'
            Metric to sort types by for visualization ranking:
            - 'max_score': Maximum score among neurons of each type
            - 'median_score': Median score of neurons per type
            - 'Q3_score': 75th percentile (Q3) score among neurons of this type
            - 'Q1_score': 25th percentile (Q1) score among neurons of this type
            - 'avg_score': Average score of neurons per type
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'expression_matrix': pd.DataFrame - Type × Line expression matrix (scores)
              Types are prefixed with dataset abbreviation: {ABBREV}_{type}
              (HEMI=hemibrain, MCNS=male-cns, FAFB=FlyWire FAFB)
            - 'labeling_info': pd.DataFrame - Case-sensitive types with dataset column
            - 'colabeling_matrices': Dict[str, pd.DataFrame] - Similarity matrices per method
            - 'line_neurons': Dict[str, pd.DataFrame] - Neurons per line with scores
            - 'line_summary': pd.DataFrame - Summary stats per line
            - 'report_path': str - Path to generated report (if generate_report=True)
            
        Output Files (when output_dir is provided)
        ------------------------------------------
        - expression_matrix.csv: Full Type × Line matrix with match scores
          (types prefixed with dataset abbreviation: HEMI_, MCNS_, FAFB_)
        - expression_matrix_viz.csv: Truncated/filtered matrix for visualization
        - expression_matrix.html: Interactive heatmap visualization
        - expression_matrix_merged.csv: Full merged matrix (types across datasets)
        - expression_matrix_merged_viz.csv: Truncated/filtered merged matrix
        - expression_matrix_merged.html: Interactive merged heatmap
        - labeling_info.csv: Case-sensitive types with dataset column for per-dataset filtering
        - colabeling_matrix_{method}.csv: Line × Line similarity matrix
        - colabeling_matrix_{method}.html: Interactive heatmap
        - line_labeled_neurons/{line}_neurons.csv: Per-line neuron details
        - line_summary.csv: Summary statistics per line
        - user_warning_notes.txt: Score-cutoff and top-N interpretation notes
        - colabeling_report.html: Comprehensive analysis report
        
        Example
        -------
        >>> nbf = NeuronBridgeFinder()
        >>> results = nbf.analyze_colabeling(
        ...     lines=['LH173', 'VT037867', 'SS00731'],
        ...     output_dir='./colabel_analysis',
        ...     type_filter={'contains': 'DN'},
        ...     background_color='black'
        ... )
        >>> print(results['line_summary'])
        """
        if similarity_methods is None:
            similarity_methods = ['jaccard', 'weighted_jaccard']

        from datetime import datetime
        
        # Parse lines input
        if isinstance(lines, str):
            line_list = [l.strip() for l in lines.split(',') if l.strip()]
        else:
            line_list = list(lines)
        
        if len(line_list) < 2:
            self._vprint("❌ At least 2 lines are required for co-labeling analysis")
            return {}
        
        # Use class-level match_type if not specified
        if match_type is None:
            match_type = self.match_type
        else:
            # Validate and normalize match_type using helper
            match_type = self._validate_match_type(match_type)
            
        # Normalize similarity_methods
        if isinstance(similarity_methods, str):
            similarity_methods = [similarity_methods]
        
        # Validate similarity_methods using helper
        similarity_methods = [self._validate_similarity_method(m) for m in similarity_methods]
        
        self._vprint(f"\n{'='*60}")
        self._vprint(f"🔬 Co-Labeling Analysis")
        self._vprint('='*60)
        self._vprint(f"   Lines: {len(line_list)}")
        self._vprint(f"   Match type: {match_type}")
        self._vprint(f"   Top neurons per line: {top_n_neurons}")
        self._vprint(f"   Similarity methods: {', '.join(similarity_methods)}")
        self._vprint(f"   Min neuron score: {min_score:,.0f}")
        self._vprint(f"   Min type avg score: {min_type_avg_score:,.0f}")
        
        # Step protocol: the report + 3D visualization phase is optional,
        # so the total varies with the run's output flags.
        wants_visuals = bool(generate_report) or (visualize_top_n > 0)
        total_steps = 5 if (wants_visuals and output_dir) else 4
        self._progress(1, total_steps, "Resolving driver lines and datasets")
        
        # Create output directory if needed
        output_path = None
        if output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Create subfolder
            lines_info = '_'.join(line_list[:3])
            if len(line_list) > 3:
                lines_info += f'_etc{len(line_list)}'
            # Sanitize folder name
            lines_info = ''.join(c if c.isalnum() or c in '-_' else '_' for c in lines_info)
            output_path = os.path.join(output_dir, f'NB-colabeling_{lines_info}_{timestamp}')
            os.makedirs(output_path, exist_ok=True)
            self._vprint(f"   📁 Output folder: {output_path}")
            
            # Save parameters for reproducibility
            self._save_parameters(
                output_path=output_path,
                function_name='analyze_colabeling',
                function_params={
                    'lines': line_list,
                    'match_type': match_type,
                    'top_n_neurons': top_n_neurons,
                    'similarity_methods': similarity_methods,
                    'generate_report': generate_report,
                    'visualize': visualize,
                    'visualize_top_n': visualize_top_n,
                    'generate_individual_profiles': generate_individual_profiles,
                    'pdf_images_per_page': pdf_images_per_page,
                    'min_score': min_score,
                    'min_type_avg_score': min_type_avg_score,
                    'background_color': background_color,
                    'type_filter': type_filter,
                    'datasets_to_visualize': datasets_to_visualize,
                    'keep_per_match_csv': keep_per_match_csv,
                }
            )
            self._vprint(f"   💾 Parameters: parameters.json")

            warning_notes = []
            if min_score > 0:
                warning_notes.extend([
                    (
                        f"Score cutoff ({min_score:,.0f}) is applied after the ranked "
                        f"top-N ({top_n_neurons}) matches per line are retrieved. "
                        "The raw line_labeled_neurons/*_neurons.csv files retain all "
                        "top-N matches, including rows below the cutoff."
                    ),
                    (
                        "The score cutoff filters expression_matrix.csv, "
                        "expression_matrix_merged.csv, and labeling_info.csv; "
                        "their aggregate scores and type coverage therefore reflect "
                        "only matches at or above the cutoff."
                    ),
                ])
            if min_type_avg_score > 0:
                warning_notes.append(
                    f"Min Type Avg Score ({min_type_avg_score:,.0f}) is an additional "
                    "filter used for the co-labeling similarity matrices only."
                )
            self._save_user_warning_notes(output_path, warning_notes)
        
        results = {
            'expression_matrix': None,
            'labeling_info': None,
            'colabeling_matrices': {},
            'line_neurons': {},
            'line_summary': None,
            'report_path': None
        }
        
        # Step 1: Fetch neurons for each line and build expression matrix
        self._progress(2, total_steps, "Fetching matches and building the expression matrix")
        self._vprint(f"\n📊 Step 1: Fetching neurons and building expression matrix...")
        
        # Use _calculate_mutual_information which already does what we need
        mi_df, expression_df, line_neurons_dict, labeling_info = self._calculate_mutual_information(
            lines=line_list,
            queried_types=[],  # No specific queried types for pure co-labeling
            match_type=match_type,
            top_n=top_n_neurons,
            output_path=output_path,
            min_score=min_score,
            min_type_avg_score=min_type_avg_score
        )
        
        results['line_neurons'] = line_neurons_dict
        results['labeling_info'] = labeling_info
        
        if expression_df.empty:
            self._vprint("   ⚠️ No expression data found for any lines")
            return results
        
        # Sort expression matrix by co-labeling quality using shared helper
        expression_transposed = self._sort_expression_matrix(
            expression_df, as_types_rows=True, input_format='lines_x_types'
        )
        
        n_lines = len(expression_transposed.columns)
        nonzero_count = (expression_transposed > 0).sum(axis=1)
        n_complete = (nonzero_count == n_lines).sum()
        self._vprint(f"   Sorted {len(expression_transposed)} types: {n_complete} in all lines, {len(expression_transposed) - n_complete} partial")
        
        results['expression_matrix'] = expression_transposed
        
        # Combine all line neurons into a single DataFrame for dataset splitting and visualization
        combined_neurons_df = pd.DataFrame()
        if line_neurons_dict:
            dfs = []
            for line_name, neurons_df in line_neurons_dict.items():
                if not neurons_df.empty:
                    neurons_copy = neurons_df.copy()
                    neurons_copy['source_line'] = line_name
                    dfs.append(neurons_copy)
            if dfs:
                combined_neurons_df = pd.concat(dfs, ignore_index=True)
        
        # Save labeling_info and per-line neuron details
        # Note: expression_matrix.csv is saved in visualize_expression_matrix (with both full and viz versions)
        if output_path:
            # Save labeling_info.csv (case-sensitive types with dataset column)
            if not labeling_info.empty:
                labeling_csv = os.path.join(output_path, 'labeling_info.csv')
                labeling_info.to_csv(labeling_csv, index=False)
                self._vprint(f"   💾 Labeling info: {labeling_csv}")
            
            # Save per-line neuron details (simple version, split by dataset below)
            if line_neurons_dict:
                neurons_dir = os.path.join(output_path, 'line_labeled_neurons')
                os.makedirs(neurons_dir, exist_ok=True)
                
                for line_name, neurons_df in line_neurons_dict.items():
                    if not neurons_df.empty:
                        safe_name = line_name.replace('/', '_').replace('\\', '_')
                        # Save the combined line neurons file
                        neurons_csv = os.path.join(neurons_dir, f'{safe_name}_neurons.csv')
                        neurons_df.to_csv(neurons_csv, index=False)
                        
                        # Split by dataset and save dataset-specific files with type summaries
                        self._save_dataset_categorized_files(neurons_df, safe_name, neurons_dir, verbose=False, sort_by=sort_by, group_in_by_dataset=False)
                
                self._vprint(f"   💾 Line neurons: {neurons_dir}/ ({len(line_neurons_dict)} lines, split by dataset)")
        
        # Step 2: Build co-labeling matrices
        self._progress(3, total_steps, "Building co-labeling matrices")
        self._vprint(f"\n📊 Step 2: Building co-labeling matrices...")
        
        for method in similarity_methods:
            self._vprint(f"   Computing {method} similarity...")
            
            matrix, line_type_sets = self._build_colabeling_matrix(
                lines=line_list,
                match_type=match_type,
                top_n=top_n_neurons,
                similarity_method=method,
                min_score=min_score,
                min_type_avg_score=min_type_avg_score
            )
            
            results['colabeling_matrices'][method] = matrix
            
            # Save matrix CSV
            if output_path:
                csv_filename = f'colabeling_matrix_{method}.csv'
                csv_path = os.path.join(output_path, csv_filename)
                matrix.to_csv(csv_path)
                self._vprint(f"   💾 {csv_filename}")
                
                # Create visualization
                if visualize:
                    method_titles = {
                        'jaccard': 'Co-Labeling Matrix (Binary Jaccard)',
                        'weighted_jaccard': 'Co-Labeling Matrix (Weighted Jaccard)',
                        'rank_correlation': 'Co-Labeling Matrix (Rank Correlation)'
                    }
                    title = method_titles.get(method, f'Co-Labeling Matrix ({method})')
                    html_filename = f'colabeling_matrix_{method}.html'
                    self.visualize_colabeling_matrix(
                        co_labeling_matrix=matrix,
                        output_path=output_path,
                        title=title,
                        color_scale='purple',
                        filename=html_filename
                    )
        
        # Visualize expression matrix (use already sorted expression_transposed.T to get Lines × Types)
        if output_path and visualize:
            # Pass the sorted matrix (transpose back to Lines × Types format expected by visualize_expression_matrix)
            self.visualize_expression_matrix(
                expression_df=expression_transposed.T,
                output_path=output_path,
                queried_types=[],
                title=f"Expression Matrix ({len(line_list)} Lines × Types)",
                type_filter=type_filter
            )
            
            # Create merged dataset version (same types across datasets combined)
            self.visualize_expression_matrix_merged(
                expression_df=expression_transposed.T,
                output_path=output_path,
                queried_types=[],
                title=f"Expression Matrix ({len(line_list)} Lines × Types) - Merged Datasets",
                aggregation='max',  # Use max score across datasets for same type
                type_filter=type_filter
            )
            
            # Visualize labeling distribution (mountain-shaped histogram)
            self.visualize_colabeling_distribution(
                line_neurons_dict=line_neurons_dict,
                output_path=output_path,
                min_score=min_score,
                title=f"Labeling Score Distribution ({len(line_list)} Lines)"
            )
        
        # Step 3: Calculate line summary statistics
        self._progress(4, total_steps, "Computing line statistics and saving summaries")
        self._vprint(f"\n📊 Step 3: Computing line statistics...")
        
        line_summary_data = []
        
        # Get the primary similarity matrix (weighted_jaccard if available)
        primary_matrix = results['colabeling_matrices'].get(
            'weighted_jaccard', 
            results['colabeling_matrices'].get('jaccard', pd.DataFrame())
        )
        
        # Calculate sparsity metrics
        sparsity_scores = {}
        if not primary_matrix.empty:
            sparsity_scores = self._calculate_colabeling_sparsity(primary_matrix)
        
        for line_name in line_list:
            neurons_df = line_neurons_dict.get(line_name, pd.DataFrame())
            
            # Basic stats
            n_neurons = len(neurons_df)
            n_types = neurons_df['type'].nunique() if not neurons_df.empty else 0
            
            # Score statistics
            mean_score = neurons_df['score'].mean() if not neurons_df.empty else 0.0
            max_score = neurons_df['score'].max() if not neurons_df.empty else 0.0
            
            # Half-max score threshold
            half_max_score = max_score / 2.0 if max_score > 0 else 0.0
            
            # n_neurons_HMS: neurons with score >= half_max_score
            n_neurons_HMS = len(neurons_df[neurons_df['score'] >= half_max_score]) if not neurons_df.empty else 0
            
            # n_types_HMS: unique types with score >= half_max_score
            if not neurons_df.empty:
                hms_neurons = neurons_df[neurons_df['score'] >= half_max_score]
                n_types_HMS = hms_neurons['type'].nunique() if not hms_neurons.empty else 0
            else:
                n_types_HMS = 0
            
            # n_neurons_MS: neurons at max score (within 0.1% tolerance)
            if not neurons_df.empty and max_score > 0:
                ms_threshold = max_score * 0.999  # Allow 0.1% tolerance
                n_neurons_MS = len(neurons_df[neurons_df['score'] >= ms_threshold])
            else:
                n_neurons_MS = 0
            
            # n_types_MS: unique types at max score
            if not neurons_df.empty and max_score > 0:
                ms_threshold = max_score * 0.999
                ms_neurons = neurons_df[neurons_df['score'] >= ms_threshold]
                n_types_MS = ms_neurons['type'].nunique() if not ms_neurons.empty else 0
            else:
                n_types_MS = 0
            
            # Qf: Quality factor = max_score / n_types_HMS (higher = more specific/selective)
            Qf = max_score / n_types_HMS if n_types_HMS > 0 else 0.0
            
            # Sparsity metrics
            sparsity_data = sparsity_scores.get(line_name, {})
            
            line_summary_data.append({
                'line': line_name,
                'n_neurons': n_neurons,
                'n_types': n_types,
                'mean_score': round(mean_score, 4),
                'max_score': round(max_score, 4),
                'n_neurons_HMS': n_neurons_HMS,
                'n_types_HMS': n_types_HMS,
                'n_neurons_MS': n_neurons_MS,
                'n_types_MS': n_types_MS,
                'Qf': round(Qf, 2),
                'colabel_sparsity': round(sparsity_data.get('colabel_sparsity', 0), 4),
            })
        
        line_summary = pd.DataFrame(line_summary_data)
        results['line_summary'] = line_summary
        
        # Save line summary
        if output_path:
            summary_csv = os.path.join(output_path, 'line_summary.csv')
            line_summary.to_csv(summary_csv, index=False)
            self._vprint(f"   💾 Line summary: {summary_csv}")
        
        # Step 4: Generate comprehensive report
        if generate_report and output_path:
            self._progress(5, 5, "Generating visualizations and reports")
            self._vprint(f"\n📝 Step 4: Generating analysis report...")
            report_path = self._generate_colabeling_report(
                results=results,
                line_list=line_list,
                match_type=match_type,
                top_n_neurons=top_n_neurons,
                output_path=output_path
            )
            results['report_path'] = report_path
        
        # Step 5: Visualize top N types per dataset (3D skeleton)
        if visualize_top_n > 0 and output_path and not combined_neurons_df.empty:
            self._progress(5, 5, "Generating visualizations and reports")
            self._vprint(f"\n🎨 Step 5: Visualizing top {visualize_top_n} types per dataset...")
            
            # Create a source_line label for folder naming
            lines_label = '_'.join(line_list[:3])
            if len(line_list) > 3:
                lines_label += f'_etc{len(line_list)}'
            
            self._visualize_top_types(
                combined_df=combined_neurons_df,
                top_n=visualize_top_n,
                output_path=output_path,
                per_dataset=True,
                source_line=lines_label,
                visualize_by=visualize_by,
                generate_individual_profiles=generate_individual_profiles,
                pdf_images_per_page=pdf_images_per_page,
                background_color=background_color,
                labeling_info=labeling_info,  # Pass case-sensitive type info for per-dataset filtering
                type_filter=type_filter,
                datasets_to_visualize=datasets_to_visualize,
                sort_by=sort_by,
                visualization_settings=visualization_settings,
            )
        
        # Summary
        self._vprint(f"\n{'='*60}")
        self._vprint(f"✅ Co-Labeling Analysis Complete!")
        self._vprint('='*60)
        self._vprint(f"   Lines analyzed: {len(line_list)}")
        self._vprint(f"   Total unique types: {len(expression_transposed)}")
        self._vprint(f"   Output: {output_path}")

        # Output-detail policy (Full keeps everything; Compact drops
        # line_labeled_neurons/ and the by-neuron distribution data once
        # the matrices/report exist).
        if output_path and prune_colabel_run is not None:
            try:
                prune_colabel_run(
                    output_path,
                    keep_per_match_csv=keep_per_match_csv,
                )
            except Exception as exc:
                self._vprint(f"   ⚠️ Output cleanup skipped: {exc}")

        return results
    
    def _generate_colabeling_report(
        self,
        results: Dict[str, Any],
        line_list: List[str],
        match_type: str,
        top_n_neurons: int,
        output_path: str
    ) -> str:
        """Generate a comprehensive HTML report for co-labeling analysis."""
        from datetime import datetime
        
        report_path = os.path.join(output_path, 'colabeling_report.html')
        
        expression_matrix = results.get('expression_matrix', pd.DataFrame())
        line_summary = results.get('line_summary', pd.DataFrame())
        colabeling_matrices = results.get('colabeling_matrices', {})
        
        # Get top co-labeling pairs
        top_pairs = []
        if 'weighted_jaccard' in colabeling_matrices:
            matrix = colabeling_matrices['weighted_jaccard']
            for i, line_i in enumerate(matrix.index):
                for j, line_j in enumerate(matrix.columns):
                    if i < j:  # Only upper triangle
                        similarity = matrix.iloc[i, j]
                        if similarity > 0.1:  # Only significant pairs
                            top_pairs.append({
                                'line1': line_i,
                                'line2': line_j,
                                'similarity': similarity
                            })
            top_pairs = sorted(top_pairs, key=lambda x: x['similarity'], reverse=True)[:20]
        
        # Find most specific lines (high sparsity)
        most_specific = []
        if not line_summary.empty:
            sorted_by_sparsity = line_summary.sort_values('colabel_sparsity', ascending=False)
            most_specific = sorted_by_sparsity.head(10).to_dict('records')
        
        # Build HTML report
        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Co-Labeling Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        .summary-box {{ background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #e8f4fc; }}
        .highlight {{ background-color: #fff3cd; }}
        .file-link {{ color: #3498db; text-decoration: none; }}
        .file-link:hover {{ text-decoration: underline; }}
        .pair-card {{ background: #e8f6f3; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #1abc9c; }}
        .similarity-bar {{ height: 20px; background: #3498db; border-radius: 4px; }}
        .timestamp {{ color: #95a5a6; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Co-Labeling Analysis Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary-box">
            <div class="metric">
                <div class="metric-value">{len(line_list)}</div>
                <div class="metric-label">Lines Analyzed</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(expression_matrix)}</div>
                <div class="metric-label">Unique Types</div>
            </div>
            <div class="metric">
                <div class="metric-value">{match_type.upper()}</div>
                <div class="metric-label">Match Type</div>
            </div>
            <div class="metric">
                <div class="metric-value">{top_n_neurons}</div>
                <div class="metric-label">Top N Neurons/Line</div>
            </div>
        </div>
        
        <h2>📊 Line Summary</h2>
        <p><strong>Metrics Explanation:</strong></p>
        <ul>
            <li><strong>Neurons</strong>: Total number of neurons matched by this line</li>
            <li><strong>Types</strong>: Total number of unique neuron types labeled</li>
            <li><strong>Max Score</strong>: Highest NeuronBridge match score (50000 = perfect match)</li>
            <li><strong>n_HMS</strong>: Neurons/Types above Half Max Score (score ≥ max_score/2)</li>
            <li><strong>n_MS</strong>: Neurons/Types at Max Score (within 0.1% of max_score)</li>
            <li><strong>Qf</strong>: Quality Factor = max_score / n_types_HMS. Higher = more selective line. A high Qf means high match quality with few high-scoring types.</li>
            <li><strong>Sparsity</strong>: 1 - (fraction of lines with similar pattern). Higher = more unique labeling.</li>
        </ul>
        <table>
            <tr>
                <th>Line</th>
                <th>Neurons</th>
                <th>Types</th>
                <th>Max Score</th>
                <th>n_neurons_HMS</th>
                <th>n_types_HMS</th>
                <th>n_neurons_MS</th>
                <th>n_types_MS</th>
                <th>Qf</th>
                <th>Sparsity</th>
            </tr>
'''
        
        for _, row in line_summary.iterrows():
            html_content += f'''
            <tr>
                <td><strong>{row['line']}</strong></td>
                <td>{row['n_neurons']}</td>
                <td>{row['n_types']}</td>
                <td>{row['max_score']:.0f}</td>
                <td>{row['n_neurons_HMS']}</td>
                <td>{row['n_types_HMS']}</td>
                <td>{row['n_neurons_MS']}</td>
                <td>{row['n_types_MS']}</td>
                <td>{row['Qf']:.0f}</td>
                <td>{row['colabel_sparsity']:.3f}</td>
            </tr>
'''
        
        html_content += '''
        </table>
        
        <h2>🔗 Top Co-Labeling Pairs</h2>
        <p>Lines with highest overlap in labeled neuron types (weighted Jaccard similarity &gt; 0.1):</p>
'''
        
        if top_pairs:
            for pair in top_pairs[:10]:
                bar_width = int(pair['similarity'] * 100)
                html_content += f'''
        <div class="pair-card">
            <strong>{pair['line1']}</strong> ↔ <strong>{pair['line2']}</strong>
            <div style="margin-top: 8px;">
                <div class="similarity-bar" style="width: {bar_width}%;"></div>
            </div>
            <small>Similarity: {pair['similarity']:.3f}</small>
        </div>
'''
        else:
            html_content += '<p><em>No significant co-labeling pairs found (all similarities ≤ 0.1)</em></p>'
        
        html_content += '''
        <h2>🎯 Most Specific Lines</h2>
        <p>Lines with highest sparsity (most unique labeling patterns):</p>
        <table>
            <tr>
                <th>Rank</th>
                <th>Line</th>
                <th>Sparsity</th>
                <th>Quality Factor (Qf)</th>
                <th>Types (HMS)</th>
            </tr>
'''
        
        for i, line_data in enumerate(most_specific[:10], 1):
            html_content += f'''
            <tr>
                <td>{i}</td>
                <td><strong>{line_data['line']}</strong></td>
                <td>{line_data['colabel_sparsity']:.3f}</td>
                <td>{line_data.get('Qf', 0):.2f}</td>
                <td>{line_data.get('n_types_HMS', 0)}</td>
            </tr>
'''
        
        html_content += '''
        </table>
        
        <h2>📈 Visualizations</h2>
        
        <h3>Expression Matrix Heatmap</h3>
        <p>Interactive heatmap showing which neuron types each line labels. Types prefixed with dataset abbreviations (HEMI_, MCNS_, FAFB_).</p>
        <iframe src="expression_matrix.html" width="100%" height="700" frameborder="0" style="border: 1px solid #ddd; border-radius: 8px;"></iframe>
        
        <h3>Expression Matrix (Merged Datasets)</h3>
        <p>Same expression matrix but with types merged across datasets. Same neuron types from different datasets (e.g., MCNS_aMe12, FAFB_aMe12) are combined into a single row (aMe12) using max score.</p>
        <iframe src="expression_matrix_merged.html" width="100%" height="700" frameborder="0" style="border: 1px solid #ddd; border-radius: 8px;"></iframe>
        
        <h3>Co-Labeling Similarity Matrix</h3>
        <p>Pairwise similarity between lines based on their labeled neuron types (weighted Jaccard).</p>
        <iframe src="colabeling_matrix_weighted_jaccard.html" width="100%" height="700" frameborder="0" style="border: 1px solid #ddd; border-radius: 8px;"></iframe>
        
        <h3>Labeling Distribution (by Type)</h3>
        <p>Score distribution for each line aggregated by neuron type. Mountain-shaped with highest scores in center.</p>
        <iframe src="labeling_distribution_by_type.html" width="100%" height="800" frameborder="0" style="border: 1px solid #ddd; border-radius: 8px;"></iframe>
        
        <h3>Labeling Distribution (by Neuron)</h3>
        <p>Score distribution for each line showing individual neuron scores.</p>
        <iframe src="labeling_distribution_by_neuron.html" width="100%" height="800" frameborder="0" style="border: 1px solid #ddd; border-radius: 8px;"></iframe>
        
        <h3>Stacked Labeling Distribution</h3>
        <p>All lines overlaid to show comparative labeling patterns.</p>
        <iframe src="labeling_distribution_stacked.html" width="100%" height="600" frameborder="0" style="border: 1px solid #ddd; border-radius: 8px;"></iframe>
        
        <h2>📁 Output Files</h2>
        <ul>
            <li><a class="file-link" href="expression_matrix.csv">expression_matrix.csv</a> - Type × Line score matrix (types prefixed with dataset: HEMI_, MCNS_, FAFB_)</li>
            <li><a class="file-link" href="expression_matrix_merged.csv">expression_matrix_merged.csv</a> - Type × Line score matrix with types merged across datasets</li>
            <li><a class="file-link" href="labeling_info.csv">labeling_info.csv</a> - Case-sensitive type × Line matrix with dataset column</li>
            <li><a class="file-link" href="expression_matrix.html">expression_matrix.html</a> - Interactive heatmap (per-dataset types)</li>
            <li><a class="file-link" href="expression_matrix_merged.html">expression_matrix_merged.html</a> - Interactive heatmap (merged types)</li>
            <li><a class="file-link" href="colabeling_matrix_weighted_jaccard.csv">colabeling_matrix_weighted_jaccard.csv</a> - Line similarity matrix</li>
            <li><a class="file-link" href="colabeling_matrix_weighted_jaccard.html">colabeling_matrix_weighted_jaccard.html</a> - Interactive heatmap</li>
            <li><a class="file-link" href="labeling_distribution_by_type.html">labeling_distribution_by_type.html</a> - Score distribution by type</li>
            <li><a class="file-link" href="labeling_distribution_by_neuron.html">labeling_distribution_by_neuron.html</a> - Score distribution by neuron</li>
            <li><a class="file-link" href="labeling_distribution_stacked.html">labeling_distribution_stacked.html</a> - Stacked distribution</li>
            <li><a class="file-link" href="distribution_data_by_neuron.csv">distribution_data_by_neuron.csv</a> - Raw neuron data (unfiltered)</li>
            <li><a class="file-link" href="distribution_data_by_type.csv">distribution_data_by_type.csv</a> - Aggregated type data (unfiltered)</li>
            <li><a class="file-link" href="line_labeled_neurons/">line_labeled_neurons/</a> - Per-line neuron details</li>
            <li><a class="file-link" href="line_summary.csv">line_summary.csv</a> - Summary statistics</li>
            <li><a class="file-link" href="parameters.json">parameters.json</a> - Analysis parameters for reproducibility</li>
        </ul>
        
        <h2>📖 Interpretation Guide</h2>
        <h3>Expression Matrix</h3>
        <p>Shows which neuron types each line labels, with values representing NeuronBridge match scores. 
        Types are prefixed with dataset abbreviations (HEMI=hemibrain, MCNS=male-cns, FAFB=FlyWire FAFB).
        Higher scores indicate stronger morphological matches.</p>
        
        <h3>Expression Matrix (Merged)</h3>
        <p>Same data as the expression matrix but with types merged across datasets. For example, 
        MCNS_aMe12 and FAFB_aMe12 are combined into a single "aMe12" row. The merged score uses the 
        maximum value across all datasets. This view is useful when you care about neuron type identity 
        regardless of which EM dataset it comes from.</p>
        
        <h3>Labeling Info</h3>
        <p>Provides case-sensitive neuron types with their source dataset. Use this file for per-dataset
        filtering and to identify which neurons from each dataset match a given driver line.</p>
        
        <h3>Co-Labeling Matrix</h3>
        <p>Shows pairwise similarity between lines based on their labeled neuron types:</p>
        <ul>
            <li><strong>Weighted Jaccard:</strong> Accounts for match scores - lines labeling the same types with similar scores have higher similarity</li>
            <li><strong>Binary Jaccard:</strong> Simple overlap - what proportion of types are labeled by both lines</li>
        </ul>
        
        <h3>Sparsity</h3>
        <p>Measures how unique a line's labeling pattern is. High sparsity (close to 1.0) means the line labels types 
        that few other lines label - useful for identifying highly specific driver lines.</p>
        
    </div>
</body>
</html>
'''
        
        # The report embeds emoji and ↔/≤/≥/× glyphs that the platform default
        # code page (GBK on zh-CN Windows) cannot represent, so pin UTF-8.
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self._vprint(f"   📝 Report: {report_path}")
        
        return report_path

    def find_lines_batch(
        self,
        queries: Union[str, int, List[Union[str, int]], Any],  # str, int, List, or LabelMapper
        dataset: Optional[Union[str, List[str]]] = None,
        match_type: Optional[str] = None,
        sort_by: str = 'max',
        output_dir: Optional[str] = None,
        download_images: Optional[str] = 'flylight',
        download_img_for_top_n_lines: Optional[int] = 30,
        image_formats: Union[str, List[str]] = None,
        image_types: Union[str, List[str]] = 'all',
        max_download_images_per_line: Optional[int] = 12,
        flylight_category: Optional[Union[str, List[str]]] = None,
        organize_by_region: bool = False,
        simple_mode: bool = False,
        pdf_images_per_page: Tuple[int, int] = (3, 2),
        pdf_landscape: bool = True,
        summary_format: Union[str, List[str]] = 'pdf',
        summary_background_color: Union[str, Tuple[int, int, int]] = 'black',
        keep_per_match_csv: bool = True,
        cleanup_source_images: bool = False,
    ) -> pd.DataFrame:
        """
        Find driver lines for multiple EM neurons with automatic saving.
        
        This is a convenience method that processes multiple queries (bodyIds,
        types, or instances), adds source information, optionally saves results,
        and optionally downloads images.
        
        Parameters
        ----------
        queries : str, int, list, or LabelMapper
            Neuron query(s). Can be:
            - bodyId (int)
            - type/instance name (str)
            - comma-separated string of names/ids
            - list of queries
            - LabelMapper object for cross-dataset unified naming (extracts queries
              for each dataset from source_mapping or target_mapping)
        dataset : str, list of str, or None
            Dataset(s) for type/instance lookups (e.g., 'hemibrain:v1.2.1').
            Can be a single string or a list of datasets to search multiple.
            Set to None to search ALL available datasets.
            When using LabelMapper, this is automatically derived from the mapper.
        match_type : str, optional
            Match algorithm: 'cds', 'pppm', or 'both'. 
            If None, uses self.match_type. Default: None
        sort_by : str
            Sorting method for line summary results (case-insensitive):
            - 'max': Sort by weighted_score (agg_mean_score × coverage_ratio —
              rewards lines with strong matches AND broad coverage). Default.
            - 'completeness': Sort by coverage_ratio (prioritizes lines
              labeling ALL queried neurons, regardless of score).
            Default: 'max'
        output_dir : str, optional
            Directory to save results. If provided, saves individual and combined CSVs.
        download_images : str, optional
            Image download source (case-insensitive):
            - 'neuronbridge': Download CDM images from NeuronBridge
            - 'flylight': Download images from FlyLight (S3/HTTP CDN)
            - 'both': Download from both sources
            - None/False: No image download (default)
        download_img_for_top_n_lines : int, optional
            Download images only for top N lines (by aggregate score/rank).
            Default: 30 (None downloads for all lines)
        image_formats : str or list
            File formats to download. For neuronbridge: 'png', 'jpg'.
            For flylight: 'png', 'jpg', 'h5j', 'lsm', 'mp4', 'json', 'all'.
            Default: 'png'
        image_types : str or list
            Image types to download. For neuronbridge: 'cdm', 'mip'.
            For flylight: 'mip', 'cdm', 'aligned', 'translation', 'metadata', 'all'.
            Default: 'all' (download all available image types)
        max_download_images_per_line : int, optional
            Maximum images to download per line. Default: 20
        flylight_category : str or list, optional
            FlyLight collection category for flylight downloads.
            Options: 'GAL4/LEXA', 'SplitGAL4', 'MCFO', 'RawImages', 'All'.
            Default: ['GAL4/LEXA', 'SplitGAL4'] (search these collections)
        organize_by_region : bool
            If True, organize downloaded images into Brain/VNC subfolders
            based on the anatomical region in the filename. Default: False
        simple_mode : bool
            If True, apply filename filtering to reduce download volume:
            - Split-GAL4 collections: only files with '20x' AND 'multichannel' in filename
            - GAL4/LexA collections: only files with 'total' in filename
            Default: False
        pdf_images_per_page : tuple of (int, int)
            (columns, rows) - number of images per page/slide in the summary.
            Default: (3, 2) = 6 images per page/slide
        pdf_landscape : bool
            Use landscape orientation for PDF. Default: True (horizontal A4)
        summary_format : str or list of str
            Format(s) for summary file generation.
            Options: 'pdf', 'pptx', or list like ['pdf', 'pptx']
            Default: 'pdf'
        summary_background_color : str or tuple of (int, int, int)
            Background color for PDF/PPTX summary files. Can be named color 
            ('black', 'white'), hex string ('#000000'), or RGB tuple (0-255).
            Default: 'black' (dark background with white text)
        keep_per_match_csv : bool
            Output detail: keep the bodyId-level per-query match tables
            ('{query}_lines.csv'). Default True (Full). False (Compact)
            removes them once 'line_summary.csv' exists — under the
            default-off match cache they regenerate only by re-running the
            query. Removals are audited in 'cleanup_audit.json'.
        cleanup_source_images : bool
            Output detail: remove the downloaded 'images/' folders after the
            integrated PDF/PPTX contact sheet was written. Default False
            (Full). The images are embedded in the document, so this only
            triggers when the artifact exists.
        
        Notes
        -----
        When `separate_splitgal4=True` is set on the NeuronBridgeFinder instance:
        - Results will include a 'line_type' column ('gal4_lexa' or 'split_gal4')
        - download_img_for_top_n_lines applies separately to each category
        - GAL4/LexA lines: start with VT, R, GMR (e.g., VT037867, R10A06)
        - Split-GAL4 lines: start with SS, LH, MB, IS, OL, LC, etc.
        - Separate summaries saved: gal4_lexa_summary.csv, split_gal4_summary.csv
          (the full per-line match tables are no longer written — they are
          extremely large; line_summary.csv remains the main output)
        
        Weighted Score and Multi-Type Query:
            The weighted_score column prioritizes lines labeling ALL queried neurons:
            
            weighted_score = agg_mean_score × (match_count / total_query_neurons)
            
            - agg_mean_score: Average NeuronBridge match score
            - match_count: Number of unique query neurons this line labels
            - total_query_neurons: Total unique neurons in the query
            - coverage_ratio: match_count / total_query_neurons
            
            All *_summary.csv files are SORTED BY weighted_score (descending).
            This ensures lines labeling ALL queried types rank highest.
            
            ** IMPORTANT: Multi-type query behavior **
            When querying 'aMe12,MBON01', the program finds lines labeling BOTH types.
            If you want lines for DIFFERENT neuron groups, query separately:
            - Query 1: 'aMe12' → best lines for aMe12
            - Query 2: 'MBON01' → best lines for MBON01
        
        Cross-Dataset Scoring (when multiple datasets):
            When searching across multiple datasets, the results include:
            - 'cross_dataset_score': Mean of max scores across datasets
            - 'datasets_labeled': Number of datasets where the line labels the query
            - 'min_score_per_dataset': Minimum score across all datasets
        
        For Specificity/Selectivity Analysis:
            Use analyze_colabeling() method separately to study how specific each
            driver line is to your target neuron types. This provides:
            - Co-labeling matrices showing overlap between lines
            - Expression matrices showing which types each line labels
            - Specificity score distributions
            
        Returns
        -------
        pd.DataFrame
            Combined DataFrame with all results, including:
            - 'source_query': original query string
            - 'source_type': neuron type (for type queries)
            - 'source_bodyId': matching bodyId for type/instance queries
            - 'source_dataset': source dataset
            - For match_type='both': cds_score, pppm_score, cds_rank, pppm_rank, combined_rank
            - For multi-dataset: cross_dataset_score, datasets_labeled, min_score_per_dataset
            
        Example
        -------
        >>> nbf = NeuronBridgeFinder()
        >>> # Basic search
        >>> results = nbf.find_lines_batch('aMe12,MBON01', dataset='hemibrain:v1.2.1')
        >>> # With NeuronBridge images (top 50 lines only)
        >>> results = nbf.find_lines_batch('aMe12', download_images='neuronbridge', 
        ...                                 download_img_for_top_n_lines=50, output_dir='./output')
        >>> # With FlyLight images
        >>> results = nbf.find_lines_batch('aMe12', download_images='flylight', output_dir='./output')
        >>> # With simple mode (reduced download volume)
        >>> results = nbf.find_lines_batch('aMe12', download_images='flylight',  
        ...                                 simple_mode=True, output_dir='./output')
        >>> # Separate GAL4/LexA from Split-GAL4 (set on instance)
        >>> nbf = NeuronBridgeFinder(separate_splitgal4=True)
        >>> results = nbf.find_lines_batch('MBON01', download_img_for_top_n_lines=5)  # 5 GAL4 + 5 Split-GAL4
        >>> # With LabelMapper for cross-dataset unified naming
        >>> from comparison.label_mapper import LabelMapper
        >>> mapper = LabelMapper(source_mapping_file='my_mappings.json')
        >>> results = nbf.find_lines_batch(mapper, output_dir='./output')
        """
        if image_formats is None:
            image_formats = ['png', 'jpg']
        if flylight_category is None:
            flylight_category = ['GAL4/LEXA', 'SplitGAL4']

        # Step protocol: the image download + summary phase is optional
        # (driven by download_images and an output folder), so the total
        # varies between 3 and 4 steps.
        total_steps = 4 if (download_images and output_dir) else 3
        self._progress(1, total_steps, "Resolving EM neuron queries")

        # Handle LabelMapper input - expand to dataset-specific queries
        label_mapper_info = None  # Track LabelMapper for type info
        if HAS_LABELMAPPER and isinstance(queries, LabelMapper):
            label_mapper_info = queries  # Keep reference
            self._vprint("📋 Expanding LabelMapper to cross-dataset queries...")
            
            # Build query list from LabelMapper's source or target mappings
            expanded_queries = []
            mapper_datasets = set()
            
            # Get datasets and standard labels from the mapper
            for role in ['source', 'target', 'intermediate']:
                all_labels = queries.get_all_std_labels(role)
                for std_label in all_labels:
                    ds_list = queries.get_datasets(role)
                    for ds in ds_list:
                        neurons = queries.get_neurons_for_label(std_label, ds, role)
                        if neurons:
                            mapper_datasets.add(ds)
                            # Add as tuples: (neuron_id, dataset, std_label)
                            for neuron in neurons:
                                expanded_queries.append({
                                    'query': neuron,
                                    'dataset': ds,
                                    'std_label': std_label
                                })
            
            # Override dataset parameter with mapper datasets if not specified
            if dataset is None and mapper_datasets:
                dataset = list(mapper_datasets)
                self._vprint(f"   Using datasets from LabelMapper: {dataset}")
            
            # Build query_list from expanded queries - store metadata for later
            query_list = []
            query_metadata = {}  # {query_str: {std_label, dataset}}
            for eq in expanded_queries:
                q = eq['query']
                query_list.append(q)
                # Store metadata for type info enrichment
                key = f"{q}_{eq['dataset']}"
                query_metadata[key] = {
                    'std_label': eq['std_label'],
                    'dataset': eq['dataset']
                }
            
            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for q in query_list:
                if q not in seen:
                    seen.add(q)
                    unique_queries.append(q)
            query_list = unique_queries
            
            self._vprint(f"   Expanded to {len(query_list)} unique queries from LabelMapper")
        elif isinstance(queries, str):
            query_list = [q.strip() for q in queries.split(',') if q.strip()]
            query_metadata = {}
        elif isinstance(queries, int):
            query_list = [queries]
            query_metadata = {}
        else:
            query_list = list(queries)
            query_metadata = {}
        
        # Use class-level match_type if not specified
        if match_type is None:
            match_type = self.match_type
        else:
            # Validate and normalize match_type using helper
            match_type = self._validate_match_type(match_type)
        
        # Validate and normalize sort_by
        sort_by = self._validate_sort_by(sort_by)
        
        if not query_list:
            self._vprint("❌ No queries provided")
            return pd.DataFrame()
        
        # Determine if multi-dataset search
        is_multi_dataset = isinstance(dataset, list) and len(dataset) > 1
        
        self._vprint(f"🔍 Finding lines for {len(query_list)} query(s)")
        if is_multi_dataset:
            self._vprint(f"   📊 Multi-dataset mode: {len(dataset)} datasets")
        
        # Create output directory if needed
        output_path = None
        if output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Add query info to folder name
            query_info = '_'.join(str(q) for q in query_list[:3])  # First 3 queries
            if len(query_list) > 3:
                query_info += '_etc'
            # Sanitize folder name (remove special characters)
            query_info = ''.join(c if c.isalnum() or c in '-_' else '_' for c in query_info)
            ds_abbr = dataset_abbrev(dataset) if isinstance(dataset, str) else 'ALL'
            output_path = os.path.join(output_dir, f'NB-find-lines_{ds_abbr}_{query_info}_{timestamp}')
            os.makedirs(output_path, exist_ok=True)
            self._vprint(f"   📁 Output folder: {output_path}")
            
            # Save parameters for reproducibility
            self._save_parameters(
                output_path=output_path,
                function_name='find_lines_batch',
                function_params={
                    'queries': query_list,
                    'dataset': dataset,
                    'match_type': match_type,
                    'sort_by': sort_by,
                    'download_images': download_images,
                    'download_img_for_top_n_lines': download_img_for_top_n_lines,
                    'image_formats': image_formats,
                    'image_types': image_types,
                    'max_download_images_per_line': max_download_images_per_line,
                    'flylight_category': flylight_category,
                    'organize_by_region': organize_by_region,
                    'simple_mode': simple_mode,
                    'pdf_images_per_page': pdf_images_per_page,
                    'pdf_landscape': pdf_landscape,
                    'keep_per_match_csv': keep_per_match_csv,
                    'cleanup_source_images': cleanup_source_images,
                    'use_cache': self.use_cache,
                }
            )
            self._vprint(f"   💾 Parameters: parameters.json")
        
        # Process each query
        all_results = []
        self._progress(2, total_steps, "Fetching matching driver-line records")
        
        # Phase 1: Check cache and identify what needs to be fetched
        if HAS_TQDM and self.verbose and len(query_list) > 1:
            from tqdm import tqdm as tqdm_progress
            
            self._vprint("\n📦 Checking cache...")
            cached_queries = []
            uncached_queries = []
            
            cache_pbar = tqdm_progress(
                query_list,
                desc="   💾 Loading cache",
                unit="query",
                bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                ncols=110,
                position=0,
                leave=False,
                file=sys.stdout,
            )
            
            for q in cache_pbar:
                # Determine cache key based on query type
                try:
                    body_id = int(q)
                    is_body_id = True
                    cache_key = self._get_id_to_lines_cache_key(
                        body_id, match_type, expected_dataset=None
                    )
                    cached_data = self._load_from_cache('id_to_lines', cache_key)
                    query_name = str(body_id)
                except (ValueError, TypeError):
                    is_body_id = False
                    # For type/instance queries, cache is per bodyId in neuron_to_lines
                    # We can't easily check cache here, so mark as uncached
                    cached_data = None
                    query_name = str(q)
                
                if cached_data is not None and is_body_id:
                    if not cached_data.empty:
                        cached_data = cached_data.copy()
                        cached_data['source_bodyId'] = body_id
                        cached_data['source_query'] = query_name
                        cached_queries.append((q, cached_data))
                        cache_pbar.set_postfix_str(f"✓ {query_name}")
                    else:
                        uncached_queries.append(q)
                else:
                    uncached_queries.append(q)
            
            cache_pbar.close()
            
            if cached_queries:
                self._vprint(f"   ✓ Loaded {len(cached_queries)} from cache")
            if uncached_queries:
                self._vprint(f"   🌐 Need to fetch {len(uncached_queries)} from API")
            
            # Process cached results
            for q, cached_data in cached_queries:
                all_results.append(cached_data)
                
                # Save individual cached results (sorted by score descending)
                if output_path:
                    query_name = str(q)
                    safe_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in query_name)
                    output_file = os.path.join(output_path, f'{safe_name}_lines.csv')
                    # Sort by score descending before saving
                    if 'score' in cached_data.columns:
                        cached_data_sorted = cached_data.sort_values('score', ascending=False)
                    else:
                        cached_data_sorted = cached_data
                    cached_data_sorted.to_csv(output_file, index=False)
            
            # Phase 2: Fetch uncached data
            if uncached_queries:
                self._vprint("\n🌐 Fetching new data...")
                fetch_pbar = tqdm_progress(
                    uncached_queries,
                    desc="   🔄 Fetching",
                    unit="query",
                    bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                    ncols=110,
                    position=0,
                    leave=False,
                    file=sys.stdout,
                )
                
                for q in fetch_pbar:
                    query_name = str(q)
                    fetch_pbar.set_postfix_str(query_name[:20])
                    
                    try:
                        # Check if query is a body ID (integer)
                        try:
                            body_id = int(q)
                            is_body_id = True
                        except (ValueError, TypeError):
                            body_id = None
                            is_body_id = False
                        
                        if is_body_id:
                            # Direct body ID search
                            lines_df = self.id_to_lines(body_id, match_type=match_type)
                            if not lines_df.empty:
                                lines_df = lines_df.copy()
                                lines_df['source_bodyId'] = body_id
                            query_name = str(body_id)
                        else:
                            # Type/instance search
                            results_dict = self.neuron_to_lines(
                                q, dataset=dataset, match_type=match_type
                            )
                            if results_dict:
                                dfs = [df for df in results_dict.values() if not df.empty]
                                lines_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
                            else:
                                lines_df = pd.DataFrame()
                            query_name = str(q)
                        
                        if lines_df.empty:
                            continue
                        
                        # Add source query info
                        lines_df = lines_df.copy()
                        lines_df['source_query'] = query_name
                        all_results.append(lines_df)
                        
                        # Save individual results (sorted by score descending)
                        if output_path:
                            safe_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in query_name)
                            output_file = os.path.join(output_path, f'{safe_name}_lines.csv')
                            # Sort by score descending before saving
                            if 'score' in lines_df.columns:
                                lines_df_sorted = lines_df.sort_values('score', ascending=False)
                            else:
                                lines_df_sorted = lines_df
                            lines_df_sorted.to_csv(output_file, index=False)
                        
                    except Exception as e:
                        pass  # Silent in progress mode
                
                fetch_pbar.close()
                self._vprint(f"   ✓ Fetched {len(uncached_queries)} from API")
        
        else:
            # Single query or no progress bar - original behavior
            for q in query_list:
                self._vprint(f"\n📋 Processing: {q}")
                
                try:
                    # Check if query is a body ID (integer)
                    try:
                        body_id = int(q)
                        is_body_id = True
                    except (ValueError, TypeError):
                        body_id = None
                        is_body_id = False
                    
                    if is_body_id:
                        # Direct body ID search
                        lines_df = self.id_to_lines(body_id, match_type=match_type)
                        if not lines_df.empty:
                            lines_df = lines_df.copy()
                            lines_df['source_bodyId'] = body_id
                        query_name = str(body_id)
                    else:
                        # Type/instance search
                        results_dict = self.neuron_to_lines(
                            q, dataset=dataset, match_type=match_type
                        )
                        if results_dict:
                            dfs = [df for df in results_dict.values() if not df.empty]
                            lines_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
                        else:
                            lines_df = pd.DataFrame()
                        query_name = str(q)
                    
                    if lines_df.empty:
                        self._vprint(f"   ⚠️ No matching lines found")
                        continue
                    
                    # Add source query info
                    lines_df = lines_df.copy()
                    lines_df['source_query'] = query_name
                    all_results.append(lines_df)
                    
                    self._vprint(f"   ✅ Found {len(lines_df)} matching driver lines")
                    
                    # Save individual results (sorted by score descending)
                    if output_path:
                        safe_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in query_name)
                        output_file = os.path.join(output_path, f'{safe_name}_lines.csv')
                        # Sort by score descending before saving
                        if 'score' in lines_df.columns:
                            lines_df_sorted = lines_df.sort_values('score', ascending=False)
                        else:
                            lines_df_sorted = lines_df
                        lines_df_sorted.to_csv(output_file, index=False)
                        self._vprint(f"   💾 Saved: {output_file}")
                        
                except Exception as e:
                    self._vprint(f"   ❌ Error: {e}")
        
        # Combine results - use optimized path for large datasets
        if all_results:
            self._progress(3, total_steps, "Aggregating and ranking line matches")
            combined_df = pd.concat(all_results, ignore_index=True)
            n_rows = len(combined_df)
            
            # For large datasets (>100k rows), use optimized Polars-based processing
            use_polars = HAS_POLARS and n_rows > 100000
            
            if use_polars:
                self._vprint(f"\n⚡ Using Polars for fast processing of {n_rows:,} rows...")
                combined_df, line_stats = self._aggregate_results_polars(
                    combined_df, 
                    match_type=match_type,
                    is_multi_dataset=is_multi_dataset,
                    sort_by=sort_by
                )
            else:
                # Original pandas-based processing for smaller datasets
                combined_df, line_stats = self._aggregate_results_pandas(
                    combined_df,
                    match_type=match_type,
                    is_multi_dataset=is_multi_dataset,
                    sort_by=sort_by
                )
            
            # Save results: only the compact line-level summaries are written
            # (the full per-match tables are extremely large for whole-GAL4-
            # line queries, so all_lines.csv / gal4_lexa_lines.csv /
            # split_gal4_lines.csv are no longer produced).
            if output_path:
                
                # Save line-level aggregate summary
                if 'line' in combined_df.columns and not line_stats.empty:
                    summary_file = os.path.join(output_path, 'line_summary.csv')
                    
                    # Reorder columns: put weighted_score before agg_mean_score
                    # Desired order: line, match_count, weighted_score, coverage_ratio, agg_mean_score, agg_max_score, ...
                    cols = list(line_stats.columns)
                    priority_cols = ['line', 'line_type', 'match_count', 'weighted_score', 'coverage_ratio', 
                                     'agg_mean_score', 'agg_max_score', 'min_score_per_dataset', 
                                     'cross_dataset_score', 'datasets_labeled', 'matched_bodyIds', 
                                     'matched_types', 'matched_datasets']
                    # Build ordered column list: priority columns first (if they exist), then remaining
                    ordered_cols = [col for col in priority_cols if col in cols]
                    remaining_cols = [col for col in cols if col not in ordered_cols]
                    final_cols = ordered_cols + remaining_cols
                    line_stats = line_stats[final_cols]
                    
                    line_stats.to_csv(summary_file, index=False)
                    self._vprint(f"   Summary: {summary_file} ({len(line_stats)} unique lines)")
                    
                    # Save only the per-type summaries (the full per-line
                    # match tables gal4_lexa_lines.csv / split_gal4_lines.csv
                    # are extremely large and are no longer written)
                    if self.separate_splitgal4 and 'line_type' in line_stats.columns:
                        # Split line_stats by line type
                        gal4_lexa_stats = line_stats[line_stats['line_type'] == 'gal4_lexa']
                        split_gal4_stats = line_stats[line_stats['line_type'] == 'split_gal4']
                        
                        # Save GAL4/LexA summary (sorted by score descending)
                        if not gal4_lexa_stats.empty:
                            gal4_summary = os.path.join(output_path, 'gal4_lexa_summary.csv')
                            gal4_lexa_stats.to_csv(gal4_summary, index=False)
                            self._vprint(f"   GAL4/LexA summary: {gal4_summary} ({len(gal4_lexa_stats)} unique)")
                        
                        # Save Split-GAL4 summary (sorted by score descending)
                        if not split_gal4_stats.empty:
                            split_summary = os.path.join(output_path, 'split_gal4_summary.csv')
                            split_gal4_stats.to_csv(split_summary, index=False)
                            self._vprint(f"   Split-GAL4 summary: {split_summary} ({len(split_gal4_stats)} unique)")
            
            # A summary format without image download is a silent no-op:
            # the PDF/PPTX contact sheets are built FROM the downloaded
            # images, so tell the user why no summary file will appear.
            _skip_note = image_summary_skip_note(summary_format, download_images)
            if _skip_note:
                self._vprint(f"\n{_skip_note}")

            # Download images if requested
            if download_images and output_path:
                self._progress(4, 4, "Downloading images and saving outputs")
                # Normalize download_images parameter (case-insensitive)
                download_source = download_images.lower() if isinstance(download_images, str) else None
                
                if download_source in ('neuronbridge', 'flylight', 'both'):
                    # Get line names for image download (use aggregated ranking)
                    if 'line' in combined_df.columns:
                        # Use pre-computed line_stats for ordering
                        all_lines = line_stats['line'].tolist()
                        
                        # Handle separate_splitgal4 mode
                        if self.separate_splitgal4:
                            # Separate lines by type and get top N from each
                            gal4_lines = [l for l in all_lines if self._classify_line_type(l) == 'gal4_lexa']
                            split_lines = [l for l in all_lines if self._classify_line_type(l) == 'split_gal4']
                            
                            if download_img_for_top_n_lines is not None and download_img_for_top_n_lines > 0:
                                gal4_lines = gal4_lines[:download_img_for_top_n_lines]
                                split_lines = split_lines[:download_img_for_top_n_lines]
                                self._vprint(f"\n🖼️  Downloading images (separate mode):")
                                self._vprint(f"      GAL4/LexA: top {len(gal4_lines)} lines")
                                self._vprint(f"      Split-GAL4: top {len(split_lines)} lines")
                            else:
                                self._vprint(f"\n🖼️  Downloading images (separate mode):")
                                self._vprint(f"      GAL4/LexA: {len(gal4_lines)} lines")
                                self._vprint(f"      Split-GAL4: {len(split_lines)} lines")
                            
                            download_lines = gal4_lines + split_lines
                        else:
                            # Normal mode: just apply top_n limit to all lines
                            if download_img_for_top_n_lines is not None and download_img_for_top_n_lines > 0:
                                download_lines = all_lines[:download_img_for_top_n_lines]
                                self._vprint(f"\n🖼️  Downloading images for top {len(download_lines)} lines...")
                            else:
                                download_lines = all_lines
                                self._vprint(f"\n🖼️  Downloading images for {len(download_lines)} lines...")
                    else:
                        download_lines = []
                    
                    if download_lines:
                        images_dir = os.path.join(output_path, 'images')
                        
                        # Download from NeuronBridge
                        nb_files = []
                        if download_source in ('neuronbridge', 'both'):
                            nb_dir = os.path.join(images_dir, 'neuronbridge') if download_source == 'both' else images_dir
                            nb_files = self._download_neuronbridge_images(
                                lines=download_lines,
                                output_dir=nb_dir,
                                formats=image_formats,
                                image_types=image_types,
                                max_files=max_download_images_per_line,
                                verbose=self.verbose
                            )
                        
                        # Download from FlyLight
                        if download_source in ('flylight', 'both'):
                            fl_dir = os.path.join(images_dir, 'flylight') if download_source == 'both' else images_dir
                            flylight_files, lines_without_flylight = self._download_flylight_images_with_category(
                                lines=download_lines,
                                output_dir=fl_dir,
                                formats=image_formats,
                                image_types=image_types,
                                max_files=max_download_images_per_line,
                                category=flylight_category,
                                organize_by_region=organize_by_region,
                                simple_mode=simple_mode,
                                verbose=self.verbose
                            )
                            
                            # Report lines without any FlyLight images
                            if lines_without_flylight:
                                self._vprint(f"\n⚠️  Note: No FlyLight images found for {len(lines_without_flylight)} line(s):")
                                self._vprint(f"   {', '.join(lines_without_flylight)}")
                                self._vprint("   (tried GAL4/LexA → Split-GAL4 → MCFO → RawImages fallbacks)")
                                
                                # Last-resort fallback: NeuronBridge images for
                                # lines the FlyLight sources missed. In 'both'
                                # mode NeuronBridge was already attempted for
                                # every line; retry only lines for which that
                                # first pass produced no image.
                                fallback_lines = []
                                resolved_by_neuronbridge = set()
                                if download_source == 'flylight':
                                    fallback_lines = list(lines_without_flylight)
                                elif download_source == 'both':
                                    nb_lines_with_files = {
                                        os.path.basename(os.path.dirname(path))
                                        for path in nb_files
                                    }
                                    resolved_by_neuronbridge.update(
                                        nb_lines_with_files.intersection(lines_without_flylight)
                                    )
                                    fallback_lines = [
                                        line for line in lines_without_flylight
                                        if line not in nb_lines_with_files
                                    ]

                                if fallback_lines:
                                    self._vprint(f"\n🔄 NeuronBridge fallback for {len(fallback_lines)} line(s) without FlyLight images...")
                                    nb_fb_dir = os.path.join(fl_dir, 'neuronbridge_fallback')
                                    fb_files = self._download_neuronbridge_images(
                                        lines=fallback_lines,
                                        output_dir=nb_fb_dir,
                                        formats=image_formats,
                                        image_types=image_types,
                                        max_files=max_download_images_per_line,
                                        verbose=self.verbose,
                                    )
                                    if fb_files:
                                        resolved_by_neuronbridge.update(
                                            os.path.basename(os.path.dirname(path))
                                            for path in fb_files
                                        )
                                        self._vprint(f"   ✅ NeuronBridge fallback: {len(fb_files)} files for "
                                                     f"{len(set(os.path.basename(os.path.dirname(p)) for p in fb_files))} line(s)")

                                # A line resolved by NeuronBridge is not a
                                # final "no images" failure; retain the
                                # informational FlyLight fallback messages,
                                # but remove the terminal error for that line.
                                if resolved_by_neuronbridge:
                                    self._warning_collector[:] = [
                                        warning for warning in self._warning_collector
                                        if not any(
                                            f"No images found for {line} in" in warning
                                            for line in resolved_by_neuronbridge
                                        )
                                    ]

                        # Print FlyLight warnings only after the NeuronBridge
                        # last-resort has had a chance to run, so the summary
                        # reflects the complete fallback chain.
                        if self._warning_collector:
                            self._print_warning_summary()
                        
                        # Generate PDF/PPTX summary if images were downloaded
                        # from EITHER source. (Previously nested in the FlyLight
                        # branch, so neuronbridge-only runs downloaded images but
                        # never produced the requested summary.)
                        self._generate_image_summaries(
                            images_dir=images_dir,
                            output_path=output_path,
                            download_lines=download_lines,
                            summary_format=summary_format,
                            pdf_images_per_page=pdf_images_per_page,
                            pdf_landscape=pdf_landscape,
                            summary_background_color=summary_background_color,
                        )

            # Output-detail policy (Full keeps everything; Compact drops the
            # bodyId-level match tables once their summary exists and the
            # downloaded images once the PDF/PPTX exists). Audited in
            # cleanup_audit.json.
            try:
                prune_find_lines_run(
                    output_path,
                    keep_per_match_csv=keep_per_match_csv,
                    cleanup_source_images=cleanup_source_images,
                )
            except Exception as exc:
                self._vprint(f"   ⚠️ Output cleanup skipped: {exc}")

            return combined_df
        
        return pd.DataFrame()
    
    def _generate_image_summaries(
        self,
        images_dir: str,
        output_path: str,
        download_lines: List[str],
        summary_format: Union[str, List[str]],
        pdf_images_per_page: tuple = (3, 2),
        pdf_landscape: bool = False,
        summary_background_color: str = 'black',
    ):
        """Generate PDF/PPTX contact sheets from the downloaded images.

        Runs whenever the images directory exists, regardless of whether the
        images came from NeuronBridge, FlyLight, or both.
        """
        if not os.path.exists(images_dir):
            return
        formats = summary_format if isinstance(summary_format, list) else [summary_format]
        # None/'' mean "no summary" (the UI passes None when unchecked)
        formats = [f.lower() for f in formats if f]

        if 'pdf' in formats:
            self._vprint(f"\n📄 Generating PDF summary...")
            pdf_path = create_image_pdf(
                images_dir=images_dir,
                output_pdf=os.path.join(output_path, 'images_summary.pdf'),
                images_per_page=pdf_images_per_page,
                landscape=pdf_landscape,
                line_order=download_lines,
                verbose=self.verbose,
                background_color=summary_background_color
            )
            if pdf_path:
                self._vprint(f"   ✅ PDF saved: {pdf_path}")

        if 'pptx' in formats:
            self._vprint(f"\n📊 Generating PPTX summary...")
            # img2pptx auto-detects font color from background
            pptx_path = img2pptx(
                input_path=images_dir,
                output_pptx=os.path.join(output_path, 'images_summary.pptx'),
                images_per_slide=pdf_images_per_page,
                slide_title='{subfolder}',
                include_subfolders=True,
                group_by_subfolder=True,
                label_position='below',
                background_color=summary_background_color
            ) if HAS_IMG2PPTX else create_image_pptx(
                images_dir=images_dir,
                output_pptx=os.path.join(output_path, 'images_summary.pptx'),
                images_per_slide=pdf_images_per_page,
                line_order=download_lines,
                verbose=self.verbose,
                background_color=summary_background_color
            )
            if pptx_path:
                self._vprint(f"   ✅ PPTX saved: {pptx_path}")

    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear cached results.
        
        Parameters
        ----------
        cache_type : str, optional
            Type of cache to clear: 'id_to_lines', 'image_cache', or None
            (all current-version Parquet data and mapping metadata).
        """
        if not os.path.exists(self.cache_folder):
            return

        root = Path(self.cache_folder)
        files: List[Path] = []
        parquet_root = root / 'parquet'
        if cache_type is None:
            # Only current Parquet cache files and its mapping/manifest are
            # selected, plus known NeuronBridge legacy artifacts so a full
            # reset really leaves no old cache records behind.
            files.extend(path for path in parquet_root.rglob('*.parquet') if path.is_file())
            files.extend(path for path in parquet_root.rglob('manifest.json') if path.is_file())
            files.extend(path for path in parquet_root.rglob('line_image_mapping.json') if path.is_file())
            files.extend(path for path in root.glob('*.csv') if path.is_file())
            files.extend(path for path in root.glob('*.parquet') if path.is_file())
            files.extend(path for path in (root / 'image_cache').glob('*.csv') if path.is_file())
            files.extend(path for path in (root / 'image_cache').glob('line_image_mapping.json') if path.is_file())
        elif cache_type == 'image_cache':
            files.extend(path for path in parquet_root.glob('*/image_cache/*.parquet') if path.is_file())
            files.extend(path for path in parquet_root.glob('*/line_image_mapping.json') if path.is_file())
            files.extend(path for path in (root / 'image_cache').glob('*.csv') if path.is_file())
            files.extend(path for path in (root / 'image_cache').glob('line_image_mapping.json') if path.is_file())
        elif cache_type == 'id_to_lines':
            files.extend(path for path in parquet_root.glob('*/id_to_lines/*.parquet') if path.is_file())
            files.extend(path for path in root.glob('id_to_lines_*.csv') if path.is_file())
            files.extend(path for path in root.glob('id_to_lines_*.parquet') if path.is_file())
        else:
            raise ValueError("cache_type must be 'id_to_lines', 'image_cache', or None")

        for path in files:
            try:
                path.unlink()
            except Exception:
                pass

        self._build_cache_index()
        self._vprint(f"🗑️ Cleared {len(files)} cached files")
    
    # =========================================================================
    # Image Download Methods
    # =========================================================================
    
    def download_line_images(
        self,
        line_names: Union[str, List[str]],
        output_dir: str,
        source: str = 'neuronbridge',
        formats: Union[str, List[str]] = 'png',
        image_types: Union[str, List[str]] = 'cdm',
        max_files: Optional[int] = None,
        verbose: Optional[bool] = None
    ) -> List[str]:
        """
        Download images for driver lines from NeuronBridge or FlyLight.
        
        Parameters
        ----------
        line_names : str or list
            Driver line name(s) to download images for.
            Can be comma-separated string or list.
        output_dir : str
            Directory to save downloaded images.
        source : str
            Image source: 'neuronbridge' for CDM images, 'flylight' for raw images.
            Default: 'neuronbridge'
        formats : str or list
            File formats to download. For neuronbridge: 'png', 'jpg'. 
            For flylight: 'png', 'jpg', 'h5j', 'lsm', 'mp4', 'json', 'all'.
            Default: 'png'
        image_types : str or list
            Image types to download. For neuronbridge: 'cdm', 'mip'.
            For flylight: 'mip', 'cdm', 'aligned', 'translation', 'metadata', 'all'.
            Default: 'cdm'
        max_files : int, optional
            Maximum number of files to download per line. Default: None (no limit)
        verbose : bool, optional
            Override class verbose setting. Default: use class setting.
            
        Returns
        -------
        list
            List of downloaded file paths.
            
        Example
        -------
        >>> nbf = NeuronBridgeFinder()
        >>> files = nbf.download_line_images('LH173', './images', source='neuronbridge')
        >>> files = nbf.download_line_images('VT037867', './images', source='flylight')
        """
        if verbose is None:
            verbose = self.verbose
        
        # Parse line names
        if isinstance(line_names, str):
            lines = [l.strip() for l in line_names.split(',') if l.strip()]
        else:
            lines = list(line_names)
        
        if not lines:
            if verbose:
                print("❌ No line names provided")
            return []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        all_downloaded = []
        
        if source.lower() == 'neuronbridge':
            all_downloaded = self._download_neuronbridge_images(
                lines, output_dir, formats, image_types, max_files, verbose
            )
        elif source.lower() == 'flylight':
            all_downloaded = self._download_flylight_images(
                lines, output_dir, formats, image_types, max_files, verbose
            )
        else:
            if verbose:
                print(f"❌ Unknown source: {source}. Use 'neuronbridge' or 'flylight'.")
        
        return all_downloaded
    
    def _download_neuronbridge_images(
        self,
        lines: List[str],
        output_dir: str,
        formats: Union[str, List[str]],
        image_types: Union[str, List[str]],
        max_files: Optional[int],
        verbose: bool
    ) -> List[str]:
        """Download CDM images from NeuronBridge with progress bar."""
        import urllib.request
        
        downloaded = []
        
        # NeuronBridge CDM URL patterns
        # CDM files are at: https://s3.amazonaws.com/janelia-flylight-color-depth/{path}
        cdm_base_url = "https://s3.amazonaws.com/janelia-flylight-color-depth/"
        
        # Phase 1: Scan all lines to count total files. Every
        # get_lm_images() is a network call, so the scan runs in parallel
        # with a streamed progress bar (previously sequential + silent).
        if verbose:
            print(f"📊 Scanning {len(lines)} lines for NeuronBridge images...")
        
        line_files_map = {}  # line_name -> list of (file_type, file_path)
        total_file_count = 0
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        scan_lock = threading.Lock()
        
        def scan_single_line(line_name: str):
            """Scan one line and return (line_name, filtered (type, path) pairs)."""
            try:
                lm_images = self._client.get_lm_images(line_name)
                if not lm_images:
                    return (line_name, [])
                
                if not isinstance(lm_images, list):
                    lm_images = list(lm_images)
                
                file_paths = []
                for lm_image in lm_images:
                    if max_files and len(file_paths) >= max_files:
                        break
                    
                    files_obj = getattr(lm_image, 'files', None)
                    if not files_obj:
                        continue
                    
                    img_types = image_types if isinstance(image_types, list) else [image_types]
                    
                    for img_type in img_types:
                        if img_type.lower() in ['cdm', 'all']:
                            cdm_path = getattr(files_obj, 'CDM', None)
                            if cdm_path:
                                file_paths.append(('cdm', cdm_path))
                        if img_type.lower() in ['mip', 'all']:
                            mip_path = getattr(files_obj, 'SignalMip', None)
                            if mip_path:
                                file_paths.append(('mip', mip_path))
                            mip_masked = getattr(files_obj, 'SignalMipMasked', None)
                            if mip_masked:
                                file_paths.append(('mip_masked', mip_masked))
                
                # Filter by format
                fmt_list = formats if isinstance(formats, list) else [formats]
                filtered_paths = []
                for file_type, file_path in file_paths:
                    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
                    if 'all' in fmt_list or ext in [f.lstrip('.').lower() for f in fmt_list]:
                        filtered_paths.append((file_type, file_path))
                
                if max_files:
                    filtered_paths = filtered_paths[:max_files]
                
                return (line_name, filtered_paths)
            except Exception:
                return (line_name, [])
        
        scan_pbar = None
        if HAS_TQDM and verbose:
            scan_pbar = tqdm(
                total=len(lines),
                desc="  Scanning",
                unit="line",
                leave=False,
                file=sys.stdout,
            )
        
        scan_workers = min(self.max_workers or os.cpu_count() or 8, max(1, len(lines)))
        with ThreadPoolExecutor(max_workers=scan_workers) as executor:
            future_to_line = {executor.submit(scan_single_line, line): line for line in lines}
            
            for future in as_completed(future_to_line):
                line_name, filtered_paths = future.result()
                with scan_lock:
                    line_files_map[line_name] = filtered_paths
                    total_file_count += len(filtered_paths)
                    scanned_files = total_file_count
                if scan_pbar:
                    scan_pbar.set_postfix(files=scanned_files, refresh=False)
                    scan_pbar.update(1)
        
        if scan_pbar:
            scan_pbar.close()
        
        if total_file_count == 0:
            if verbose:
                print("  ⚠️ No files found for any lines")
            return []
        
        lines_with_files = len([l for l, f in line_files_map.items() if f])
        if verbose:
            print(f"  📦 Found {total_file_count} files across {lines_with_files} lines")
        
        # Phase 2: Download with parallel processing
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        downloaded = []
        downloaded_lock = threading.Lock()
        
        # Build flat list of download tasks
        download_tasks = []
        for line_name in lines:
            if line_name not in line_files_map or not line_files_map[line_name]:
                continue
            line_dir = os.path.join(output_dir, line_name)
            for file_type, file_path in line_files_map[line_name]:
                url = cdm_base_url + file_path
                filename = os.path.basename(file_path)
                local_path = os.path.join(line_dir, filename)
                download_tasks.append((line_name, url, local_path, line_dir))
        
        def download_single_file(task):
            """Download a single file and return (success, local_path, line_name)."""
            line_name, url, local_path, line_dir = task
            try:
                os.makedirs(line_dir, exist_ok=True)
                urllib.request.urlretrieve(url, local_path)
                return (True, local_path, line_name)
            except Exception as e:
                return (False, str(e), line_name)
        
        pbar = None
        if HAS_TQDM and verbose:
            pbar = tqdm(
                total=len(download_tasks),
                desc="  Downloading",
                unit="file",
                file=sys.stdout,
            )
        
        # Use ThreadPoolExecutor for parallel downloads
        # Use all available CPU cores for maximum throughput
        max_workers = min(self.max_workers or os.cpu_count() or 8, len(download_tasks))
        files_downloaded = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(download_single_file, task): task for task in download_tasks}
            
            for future in as_completed(future_to_task):
                success, result, line_name = future.result()
                
                if success:
                    with downloaded_lock:
                        downloaded.append(result)
                    files_downloaded += 1
                    if pbar:
                        pbar.set_postfix(line=line_name[:15], refresh=False)
                        pbar.update(1)
                    elif verbose:
                        print(f"  [{files_downloaded}/{total_file_count}] {line_name}: {os.path.basename(result)}")
                else:
                    if pbar:
                        pbar.update(1)
                    elif verbose:
                        print(f"  ⚠️ Failed to download for {line_name}: {result}")
        
        if pbar:
            pbar.close()
        
        if verbose:
            print(f"  ✅ Downloaded {len(downloaded)}/{total_file_count} files")
        
        return downloaded
    
    def _download_flylight_images(
        self,
        lines: List[str],
        output_dir: str,
        formats: Union[str, List[str]],
        image_types: Union[str, List[str]],
        max_files: Optional[int],
        verbose: bool
    ) -> List[str]:
        """Download images from FlyLight using flylight_downloader module."""
        downloaded, _missing = self._download_flylight_images_with_category(
            lines=lines,
            output_dir=output_dir,
            formats=formats,
            image_types=image_types,
            max_files=max_files,
            category=None,
            organize_by_region=False,
            verbose=verbose
        )
        if verbose:
            self._print_warning_summary()
        return downloaded
    
    def _parse_region_from_filename(self, filename: str, full_key: str = None) -> str:
        """
        Parse anatomical region from FlyLight filename or file key path.
        
        FlyLight filename format: {line}-{date}_{sample}-{sex}-{mag}-{region}-{driver}-...
        Example: SS01015-20131220_31_C3-f-20x-brain-Split_GAL4-JRC2018_Unisex_20x_HR-CDM_1.png
        
        VT GAL4 files have region in the key path:
        Example: VT GAL4/VT037867/brain/filename.jpg
        
        Brain regions include: brain, central, dorsal, optic, protocerebrum, etc.
        VNC regions include: vnc, ventral_nerve_cord, metathoracic, prothoracic, mesothoracic
        
        Returns 'Brain', 'VNC', or 'Other' based on the region field.
        """
        # First check the full key path for region (VT GAL4 files)
        if full_key:
            key_lower = full_key.lower()
            # Check for /brain/ or /vnc/ in the path
            if '/brain/' in key_lower:
                return 'Brain'
            if '/vnc/' in key_lower:
                return 'VNC'
        
        # Split by '-' and try to find region field (MCFO and other formats)
        parts = filename.split('-')
        
        # flweb CGI imagery (flweb.janelia.org view_flew_imagery.cgi) embeds
        # the region in the sample tag: fA01b / fA00b (brain) and fA01v /
        # fA00v (VNC) — e.g. R85D07_AE_01_03-fA01b_C101223_..._total.jpg.
        # Check the joined name (the tag may sit mid-filename, not on a
        # '-' boundary).
        combined_lower = (filename + ' ' + (full_key or '')).lower()
        if 'fa01b' in combined_lower or 'fa00b' in combined_lower:
            return 'Brain'
        if 'fa01v' in combined_lower or 'fa00v' in combined_lower:
            return 'VNC'
        
        # VNC-specific keywords (check first to avoid confusion with 'ventral')
        vnc_keywords = ['vnc', 'ventral_nerve_cord', 'metathoracic', 'prothoracic', 'mesothoracic']
        
        # Brain-specific keywords
        brain_keywords = ['brain', 'central', 'dorsal', 'optic', 'protocerebrum', 
                         'left_optic_lobe', 'right_optic_lobe', 'left_dorsal', 'right_dorsal',
                         'ventral']  # 'ventral' alone (without nerve_cord) means brain ventral
        
        # Search for VNC keywords first (more specific)
        for part in parts:
            part_lower = part.lower()
            for keyword in vnc_keywords:
                if keyword in part_lower:
                    return 'VNC'
        
        # Then search for brain keywords
        for part in parts:
            part_lower = part.lower()
            for keyword in brain_keywords:
                if keyword in part_lower:
                    return 'Brain'
        
        return 'Other'
    
    def _filter_flylight_files_by_region(self, files: List[Any]) -> List[Any]:
        """
        Filter FlyLight files based on anatomical region setting.
        
        Uses filename and key path parsing to determine region (Brain/VNC).
        Only filters when self.region is 'Brain' or 'VNC'.
        
        Parameters
        ----------
        files : list
            List of FlyLightFile objects.
            
        Returns
        -------
        list
            Filtered list of files matching the region criteria.
        """
        if self.region == 'All' or not self.region:
            return files
        
        filtered = []
        for file in files:
            # Get full key and filename
            full_key = file.key if hasattr(file, 'key') else ''
            filename = full_key.split('/')[-1] if full_key else ''
            if not filename and hasattr(file, 'url'):
                filename = file.url.split('/')[-1]
            
            if not filename:
                # If we can't determine filename, include by default
                filtered.append(file)
                continue
            
            # Parse region from filename and full key path
            file_region = self._parse_region_from_filename(filename, full_key)
            
            # Include if matches the desired region
            if file_region == self.region:
                filtered.append(file)
        
        return filtered
    
    def _reorganize_files_by_region(
        self,
        downloaded_files: List[str],
        output_dir: str,
        verbose: bool = False
    ) -> List[str]:
        """
        Reorganize downloaded files into Brain/VNC/Other subfolders.
        
        Original structure: output_dir/Collection/LineName/filename.png
        New structure: output_dir/Brain/LineName/filename.png (or VNC or Other)
        
        Returns list of new file paths.
        """
        import shutil
        from pathlib import Path
        
        reorganized = []
        region_counts = {'Brain': 0, 'VNC': 0, 'Other': 0}
        
        for file_path in downloaded_files:
            file_path = Path(file_path)
            if not file_path.exists():
                continue
            
            filename = file_path.name
            
            # Parse region and line name from filename
            region = self._parse_region_from_filename(filename)
            
            # Extract line name (first field before '-')
            line_name = filename.split('-')[0] if '-' in filename else filename.split('.')[0]
            
            # Create new path: output_dir/Region/LineName/filename
            new_dir = Path(output_dir) / region / line_name
            new_dir.mkdir(parents=True, exist_ok=True)
            new_path = new_dir / filename
            
            try:
                # Move file to new location
                shutil.move(str(file_path), str(new_path))
                reorganized.append(str(new_path))
                region_counts[region] += 1
            except Exception as e:
                # If move fails, try copy
                try:
                    shutil.copy2(str(file_path), str(new_path))
                    file_path.unlink()  # Delete original
                    reorganized.append(str(new_path))
                    region_counts[region] += 1
                except Exception as e2:
                    if verbose:
                        print(f"  ⚠️  Could not reorganize {filename}: {e2}")
                    reorganized.append(str(file_path))  # Keep original
        
        # Clean up empty collection directories
        try:
            for item in Path(output_dir).iterdir():
                if item.is_dir() and item.name not in ('Brain', 'VNC', 'Other'):
                    # Check if empty (or has only empty subdirs)
                    remaining_files = list(item.rglob('*'))
                    remaining_files = [f for f in remaining_files if f.is_file()]
                    if not remaining_files:
                        shutil.rmtree(str(item), ignore_errors=True)
        except Exception:
            pass
        
        if verbose:
            print(f"  📁 Reorganized by region: Brain={region_counts['Brain']}, VNC={region_counts['VNC']}, Other={region_counts['Other']}")
        
        return reorganized
    
    def _download_flylight_images_with_category(
        self,
        lines: List[str],
        output_dir: str,
        formats: Union[str, List[str]],
        image_types: Union[str, List[str]],
        max_files: Optional[int],
        category: Optional[Union[str, List[str]]],
        organize_by_region: bool = False,
        simple_mode: bool = False,
        verbose: bool = False
    ) -> Tuple[List[str], List[str]]:
        """
        Download images from FlyLight with sequential category searching.
        
        This method searches FlyLight collections in priority order, collecting
        images from each category sequentially until ``max_files`` is reached
        for each line. If a line has no images in the specified categories,
        it automatically falls back to MCFO, then RawImages, before the
        caller's final NeuronBridge fallback.
        
        **Category Search Order:**
        
        Categories are searched in the order specified in the ``category`` parameter.
        For example, if ``category=['GAL4/LEXA', 'SplitGAL4']``:
        
        1. First, search 'GAL4/LEXA' collection
        2. If not enough files, search 'SplitGAL4' collection
        3. If still no files found, fallback to 'MCFO' collection
        4. If still no files found, fallback to 'RawImages' collection
        5. If still no files found, the caller tries NeuronBridge images
        
        This sequential approach ensures:
        - Preferred collections are prioritized
        - ``max_files`` limit is respected across all categories
        - Lines without images in primary categories get MCFO and RawImages
          fallbacks in that order
        
        Parameters
        ----------
        lines : list of str
            List of driver line names to download images for.
        output_dir : str
            Directory to save downloaded images.
        formats : str or list of str
            File formats to download: 'png', 'jpg', 'h5j', 'mp4', 'all'.
        image_types : str or list of str
            Image types: 'mip', 'cdm', 'aligned', 'translation', 'all'.
        max_files : int, optional
            Maximum images to download per line. Images are collected from
            categories in order until this limit is reached.
        category : str, list of str, or None
            FlyLight collection category(s) to search, in priority order.
            Options: 'GAL4/LEXA', 'SplitGAL4', 'MCFO', 'RawImages', 'All'.
            If a list, categories are searched sequentially in the given order.
            If None, searches all collections.
            
            **Fallback behavior:** If no images are found for a line in the
            specified categories, MCFO and then RawImages are searched
            automatically. If FlyLight still has no image,
            ``find_lines_batch`` tries NeuronBridge.
        organize_by_region : bool
            If True, organize images into Brain/VNC subfolders.
        simple_mode : bool
            If True, apply filename filtering to reduce download volume:
            - Split-GAL4 collections: only files with '20x' AND 'multichannel'
            - GAL4/LexA collections: only files with 'total' in filename
            - MCFO: all files kept (no filtering)
            
        Returns
        -------
        tuple of (list, list)
            - downloaded_files: List of downloaded file paths
            - lines_without_files: List of line names that had no FlyLight
              files available even after MCFO and RawImages fallbacks
              
        Examples
        --------
        >>> # Search GAL4/LEXA first, then SplitGAL4, with MCFO fallback
        >>> files, missing = finder._download_flylight_images_with_category(
        ...     lines=['SS00731', 'VT000770'],
        ...     output_dir='./images',
        ...     formats=['png', 'jpg'],
        ...     image_types='mip',
        ...     max_files=10,
        ...     category=['GAL4/LEXA', 'SplitGAL4'],  # Priority order
        ...     simple_mode=True
        ... )
        >>> # SS00731 might get images from SplitGAL4
        >>> # VT000770 might get images from MCFO fallback
        """
        try:
            from flylight_downloader import FlyLightDownloader
        except ImportError:
            try:
                # Try relative import
                from .flylight_downloader import FlyLightDownloader
            except ImportError:
                if verbose:
                    print("❌ flylight_downloader module not available")
                return [], lines
        
        def apply_simple_mode_filter(files, collection_name: str):
            """Filter files based on simple_mode rules."""
            if not simple_mode or not files:
                return files
            
            collection_lower = collection_name.lower() if collection_name else ''
            filtered = []
            
            for f in files:
                filename_lower = f.filename.lower()
                
                # Split-GAL4: only include '20x' AND 'multichannel', exclude 'image1'/'image2'
                if 'splitgal4' in collection_lower or 'split-gal4' in collection_lower or 'split_gal4' in collection_lower:
                    if '20x' in filename_lower and 'multichannel' in filename_lower:
                        # Exclude duplicate images (image1, image2, etc.)
                        if 'image1' not in filename_lower and 'image2' not in filename_lower:
                            filtered.append(f)
                # GAL4/LexA: only include 'total'
                elif 'gal4' in collection_lower or 'lexa' in collection_lower:
                    if 'total' in filename_lower:
                        filtered.append(f)
                else:
                    # For other collections (MCFO, etc.), keep all files
                    filtered.append(f)
            
            return filtered
        
        def verify_vt_file_accessible(file) -> tuple:
            """Check if a VT file URL is accessible (HEAD request).
            
            Returns:
                tuple: (is_accessible: bool, error_msg: str or None)
            """
            if file.source != 'http':
                return True, None  # S3 files don't need verification
            try:
                import urllib.request
                req = urllib.request.Request(file.url, method='HEAD')
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return resp.status == 200, None
            except Exception as e:
                # Extract server from URL
                import urllib.parse
                parsed = urllib.parse.urlparse(file.url)
                server = parsed.netloc
                error_msg = f"{server} unreachable: {type(e).__name__}"
                return False, error_msg
        
        def get_files_for_line_sequential(line_name: str, categories: List[str], 
                                          downloader_kwargs: dict, max_files_limit: int,
                                          verify_vt: bool = True) -> List:
            """
            Get files for a line by searching categories sequentially until max_files reached.
            
            Parameters
            ----------
            line_name : str
                The driver line name.
            categories : list of str
                Categories to search in priority order.
            downloader_kwargs : dict
                Base kwargs for FlyLightDownloader.
            max_files_limit : int
                Maximum files to collect.
                
            Returns
            -------
            list
                List of FlyLightFile objects, up to max_files_limit.
            """
            collected_files = []
            
            for cat in categories:
                if len(collected_files) >= max_files_limit:
                    break
                    
                try:
                    # Create downloader for this category
                    dl = FlyLightDownloader(
                        output_dir=downloader_kwargs.get('output_dir', output_dir),
                        collection_category=cat,
                        formats=downloader_kwargs.get('formats', formats),
                        image_types=downloader_kwargs.get('image_types', image_types),
                        automatic_fallback=False,
                        verbose=False
                    )
                    
                    files = dl.get_filtered_files(line_name)
                    if files:
                        collection = files[0].collection if files else cat
                        files = apply_simple_mode_filter(files, collection)
                        
                        # Apply region filter (important for MCFO fallback)
                        files = self._filter_flylight_files_by_region(files)
                        
                        # For VT GAL4 files (HTTP source), verify the server is accessible
                        # This catches cases where flimg.janelia.org is down
                        if verify_vt and files and files[0].source == 'http':
                            is_accessible, error_msg = verify_vt_file_accessible(files[0])
                            if not is_accessible:
                                # Server down - collect warning and skip files for MCFO fallback
                                collection_name = files[0].collection or cat
                                warning_msg = f"⚠️  {collection_name} server not accessible for {line_name}: {error_msg}"
                                self._warning_collector.append(warning_msg)
                                continue
                        
                        # Add files up to the limit
                        remaining = max_files_limit - len(collected_files)
                        collected_files.extend(files[:remaining])
                except Exception:
                    continue
            
            return collected_files
        
        def get_flweb_fallback_files(line_name: str, max_files_limit: int) -> List:
            """Fallback: flweb.janelia.org CGI (view_flew_imagery.cgi).

            The FlyLight Expression Web holds expression-pattern imagery for
            lines that the chosen S3 collections may not cover (e.g. R-lines
            like R85D07 — https://flweb.janelia.org/cgi-bin/view_flew_imagery.cgi?line=R85D07).
            Type filters are lenient (projections / all) because the CGI
            filenames do not carry the S3 'CDM'/'mip' tags; the region and
            simple-mode filters still apply.
            """
            try:
                from flylight_downloader import FlyLightDownloader
                dl = FlyLightDownloader(
                    output_dir=output_dir,
                    collection_category=None,
                    formats='all',
                    image_types='all',
                    verbose=False
                )
                files = dl._get_r_line_files(line_name)
                files = self._filter_flylight_files_by_region(files)
                files = apply_simple_mode_filter(files, 'Gen1 GAL4')
                if max_files_limit:
                    files = files[:max_files_limit]
                return files
            except Exception:
                return []
        
        downloaded = []
        
        # Normalize category to list
        if category is None:
            categories = ['All']
        elif isinstance(category, str):
            categories = [category]
        else:
            categories = list(category)
        
        # Check if MCFO fallback is needed (not already included and not searching all)
        categories_lower = [c.lower() for c in categories]
        needs_mcfo_fallback = ('mcfo' not in categories_lower and 
                               'all' not in categories_lower)
        
        # Separate VT lines from other lines (they need different format settings)
        vt_lines = [l for l in lines if l.upper().startswith('VT')]
        other_lines = [l for l in lines if not l.upper().startswith('VT')]
        
        # Phase 1: Scan all lines to count total files (with sequential category search)
        if verbose:
            cat_str = ' → '.join(categories)
            if needs_mcfo_fallback:
                cat_str += ' → MCFO → RawImages (fallback)'
            if simple_mode:
                print(f"📊 Scanning {len(lines)} lines (simple mode, categories: {cat_str})...")
            else:
                print(f"📊 Scanning {len(lines)} lines (categories: {cat_str})...")
        
        line_files_map = {}  # line_name -> list of FlyLightFile
        max_files_limit = max_files if max_files else 999999
        
        # Determine VT format settings
        vt_formats = formats
        if isinstance(vt_formats, str):
            vt_formats = [vt_formats]
        if 'png' in vt_formats and 'jpg' not in vt_formats:
            vt_formats = list(vt_formats) + ['jpg']
        
        # Thread-safe counters and containers
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        total_file_count = [0]  # Use list for thread-safe mutation
        count_lock = threading.Lock()
        result_lock = threading.Lock()
        warning_lock = threading.Lock()
        
        def scan_single_line(line_name: str) -> tuple:
            """Scan a single line and return (line_name, files)."""
            is_vt = line_name.upper().startswith('VT')
            line_formats = vt_formats if is_vt else formats
            line_image_types = 'all' if is_vt else image_types
            
            try:
                # Sequential category search
                files = get_files_for_line_sequential(
                    line_name, 
                    categories,
                    {'formats': line_formats, 'image_types': line_image_types},
                    max_files_limit
                )
                
                # MCFO fallback if no files found and fallback is enabled
                if not files and needs_mcfo_fallback:
                    with warning_lock:
                        warning_msg = f"ℹ️  No files from {', '.join(categories)} for {line_name}, trying MCFO fallback..."
                        self._warning_collector.append(warning_msg)
                    
                    files = get_files_for_line_sequential(
                        line_name,
                        ['MCFO'],
                        {'formats': line_formats, 'image_types': line_image_types},
                        max_files_limit
                    )
                
                # RawImages collection fallback: raw confocal data on S3.
                if not files:
                    with warning_lock:
                        warning_msg = f"ℹ️  No MCFO files for {line_name}, trying RawImages collection..."
                        self._warning_collector.append(warning_msg)
                    files = get_files_for_line_sequential(
                        line_name,
                        ['RawImages'],
                        {'formats': line_formats, 'image_types': line_image_types},
                        max_files_limit
                    )

                # Legacy flweb CGI fallback: expression-pattern imagery on
                # flweb.janelia.org for lines the selected collections missed
                # (e.g. R-lines like R85D07). The primary GAL4/LexA lookup
                # already checks this endpoint for R-lines; keep this final
                # FlyLight probe for older/atypical CGI pages, after MCFO and
                # RawImages as requested.
                if not files:
                    with warning_lock:
                        warning_msg = f"ℹ️  No RawImages files for {line_name}, trying flweb.janelia.org..."
                        self._warning_collector.append(warning_msg)
                    files = get_flweb_fallback_files(line_name, max_files_limit)
                
                if not files:
                    with warning_lock:
                        warning_msg = f"⚠️  No images found for {line_name} in any FlyLight collection (MCFO/RawImages included)"
                        self._warning_collector.append(warning_msg)
                
                return (line_name, files)
            except Exception:
                return (line_name, [])
        
        # Create a scanning progress bar
        scan_pbar = None
        if HAS_TQDM and verbose:
            scan_pbar = tqdm(
                total=len(lines),
                desc="  Scanning",
                unit="line",
                leave=False,
                file=sys.stdout,
            )
        
        # Use ThreadPoolExecutor for parallel scanning
        # Use the finder's max_workers (capped by the number of lines)
        max_workers = min(self.max_workers or os.cpu_count() or 8, max(1, len(lines)))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all lines for scanning
            future_to_line = {executor.submit(scan_single_line, line): line for line in lines}
            
            for future in as_completed(future_to_line):
                line_name, files = future.result()
                
                with result_lock:
                    line_files_map[line_name] = files
                
                with count_lock:
                    total_file_count[0] += len(files)
                
                if scan_pbar:
                    scan_pbar.set_postfix(files=total_file_count[0], refresh=False)
                    scan_pbar.update(1)
        
        if scan_pbar:
            scan_pbar.close()
        
        total_file_count = total_file_count[0]  # Convert back to int
        
        # Track lines without any FlyLight files
        lines_without_files = [l for l, f in line_files_map.items() if not f]
        
        if total_file_count == 0:
            if verbose:
                print("  ⚠️ No files found for any lines (tried all categories including MCFO)")
            return [], lines
        
        lines_with_files = len([l for l, f in line_files_map.items() if f])
        if verbose:
            print(f"  📦 Found {total_file_count} files across {lines_with_files} lines")
            if lines_without_files:
                print(f"  ⚠️ No files for: {', '.join(lines_without_files[:5])}{'...' if len(lines_without_files) > 5 else ''}")
        
        # Phase 2: Download files
        # Download all lines using pre-scanned files with parallel processing
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        downloaded = []
        downloaded_lock = threading.Lock()
        files_downloaded = [0]
        
        # Create a single progress bar for all files
        pbar = None
        if HAS_TQDM and verbose:
            pbar = tqdm(
                total=total_file_count,
                desc="  Downloading",
                unit="file",
                file=sys.stdout,
            )
        
        def download_line_files(line_info):
            """Download files for a single line."""
            line_name, line_files = line_info
            try:
                # Determine the correct formats for this line
                line_formats = formats
                line_image_types = image_types
                if line_name.upper().startswith('VT'):
                    if isinstance(line_formats, str):
                        line_formats = [line_formats]
                    if 'png' in line_formats and 'jpg' not in line_formats:
                        line_formats = list(line_formats) + ['jpg']
                    line_image_types = 'all'
                
                # Create downloader with parallel support (no verbose to avoid nested progress bars)
                downloader = FlyLightDownloader(
                    output_dir=output_dir,
                    formats=line_formats,
                    image_types=line_image_types,
                    verbose=False,
                    max_workers=min(4, os.cpu_count() or 4)  # Limit per-line workers
                )
                
                dl_files = downloader.download(
                    line_name=line_name,
                    max_files=max_files,
                    flat_structure=True,
                    add_timestamp=False,
                    files=line_files
                )
                return (line_name, [str(f) for f in dl_files], len(line_files))
            except Exception as e:
                return (line_name, [], len(line_files), str(e))
        
        # Prepare download tasks (only lines with files)
        download_tasks = [(ln, line_files_map[ln]) for ln in lines 
                         if ln in line_files_map and line_files_map[ln]]
        
        # Use ThreadPoolExecutor for parallel line downloads
        # Limit workers to avoid too many concurrent connections
        max_line_workers = min(4, len(download_tasks))
        
        if max_line_workers > 1 and len(download_tasks) > 1:
            with ThreadPoolExecutor(max_workers=max_line_workers) as executor:
                future_to_line = {executor.submit(download_line_files, task): task[0] 
                                  for task in download_tasks}
                
                for future in as_completed(future_to_line):
                    result = future.result()
                    line_name = result[0]
                    dl_files = result[1]
                    expected_count = result[2] if len(result) > 2 else len(dl_files)
                    
                    with downloaded_lock:
                        downloaded.extend(dl_files)
                    
                    # Update progress bar
                    if pbar:
                        pbar.set_postfix(line=line_name[:15], refresh=False)
                        pbar.update(expected_count)
        else:
            # Sequential fallback for small number of tasks
            for task in download_tasks:
                result = download_line_files(task)
                line_name = result[0]
                dl_files = result[1]
                expected_count = result[2] if len(result) > 2 else len(dl_files)
                downloaded.extend(dl_files)
                
                if pbar:
                    pbar.set_postfix(line=line_name[:15], refresh=False)
                    pbar.update(expected_count)
        
        if pbar:
            pbar.close()
        
        if verbose:
            print(f"  ✅ Downloaded {len(downloaded)}/{total_file_count} files")
        
        # Reorganize files by anatomical region if requested
        if organize_by_region and downloaded:
            if verbose:
                print("  🔄 Reorganizing files by anatomical region...")
            downloaded = self._reorganize_files_by_region(
                downloaded_files=downloaded,
                output_dir=output_dir,
                verbose=verbose
            )
        
        return downloaded, lines_without_files
    
    def find_lines_batch_with_images(
        self,
        queries: Union[str, int, List],
        dataset: Optional[str] = None,
        top_n: int = -1,
        match_type: str = 'cds',
        output_dir: Optional[str] = None,
        download_images: Union[bool, str] = False,
        image_source: str = 'neuronbridge',
        image_formats: Union[str, List[str]] = 'png',
        image_types: Union[str, List[str]] = 'cdm',
        max_download_images_per_line: Optional[int] = 5
    ) -> pd.DataFrame:
        """
        Find driver lines with optional image download.
        
        .. deprecated::
            This method is deprecated. Use find_lines_batch() with download_images parameter instead.
        
        Parameters
        ----------
        queries : str, int, or list
            Neuron query(s). Can be bodyId, type, instance, or comma-separated string.
        dataset : str, optional
            Dataset for type/instance lookups.
        top_n : int
            Maximum matches per neuron. Default: -1 (all matches)
        match_type : str
            Match algorithm: 'cds', 'pppm', or 'both'. Default: 'cds'
        output_dir : str, optional
            Directory to save results and images.
        download_images : bool or str
            Whether to download images. True uses image_source, or pass source directly.
        image_source : str
            'neuronbridge' for CDM images, 'flylight' for raw confocal. Default: 'neuronbridge'
        image_formats : str or list
            File formats for images. Default: 'png'
        image_types : str or list
            Image types to download. Default: 'cdm'
        max_download_images_per_line : int, optional
            Maximum images to download per line. Default: 5
            
        Returns
        -------
        pd.DataFrame
            Combined results DataFrame.
        """
        import warnings
        warnings.warn(
            "find_lines_batch_with_images() is deprecated. Use find_lines_batch() with download_images parameter.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Convert legacy parameters to new format
        if download_images:
            if isinstance(download_images, bool):
                dl_source = image_source
            else:
                dl_source = download_images
        else:
            dl_source = None
        
        return self.find_lines_batch(
            queries=queries,
            dataset=dataset,
            match_type=match_type,
            output_dir=output_dir,
            download_images=dl_source,
            image_formats=image_formats,
            image_types=image_types,
            max_download_images_per_line=max_download_images_per_line
        )


def _collect_line_images(images_path: Path) -> dict:
    """Collect ``line_name -> [image files]`` from an images directory.

    Handles the three layouts produced by the downloaders:

    - ``images/<LineName>/*.png`` - per-line subdirectories (NeuronBridge)
    - ``images/*.png`` - flat files, grouped by the prefix before ``-``
    - ``images/<source>/...`` - source subfolders (``neuronbridge/``,
      ``flylight/``, created when ``download_images='both'``), scanned one
      level deep and merged into the same map

    Returns a dict of line name -> sorted, de-duplicated list of file paths.
    """
    image_extensions = {'.png', '.jpg', '.jpeg'}
    # Known source subfolders created by download_source='both'.
    source_subdirs = ('neuronbridge', 'flylight')

    def scan_dir(directory: Path, line_images: dict) -> None:
        """Scan one directory: line subdirs whose children are images + flat files."""
        for item in sorted(directory.iterdir()):
            if item.is_dir():
                images = [
                    f for f in item.iterdir() if f.suffix.lower() in image_extensions
                ]
                if images:
                    line_images.setdefault(item.name, []).extend(images)
            elif item.suffix.lower() in image_extensions:
                # Flat files are grouped by the prefix before the first '-'
                # (e.g. VT017647-20x-multichannel.png -> VT017647)
                line_name = item.stem.split('-')[0] if '-' in item.stem else 'Unknown'
                line_images.setdefault(line_name, []).append(item)

    line_images = {}
    for item in sorted(images_path.iterdir()):
        if item.is_dir():
            if item.name in source_subdirs:
                continue  # scanned explicitly below
            direct_images = [
                f for f in item.iterdir() if f.suffix.lower() in image_extensions
            ]
            if direct_images:
                line_images.setdefault(item.name, []).extend(direct_images)
            else:
                # Subfolder that itself contains line dirs (custom layout)
                scan_dir(item, line_images)
        elif item.suffix.lower() in image_extensions:
            line_name = item.stem.split('-')[0] if '-' in item.stem else 'Unknown'
            line_images.setdefault(line_name, []).append(item)

    # Source subfolders from download_source='both'
    for name in source_subdirs:
        sub = images_path / name
        if sub.is_dir():
            scan_dir(sub, line_images)

    # De-duplicate (a file can be reachable through multiple scans) and sort
    for line_name in line_images:
        seen = set()
        unique = []
        for p in line_images[line_name]:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        line_images[line_name] = sorted(unique)
    return line_images


def create_image_pdf(
    images_dir: str,
    output_pdf: Optional[str] = None,
    images_per_page: Tuple[int, int] = (5, 3),
    page_size: str = 'A4',
    landscape: bool = True,
    title_font_size: int = 14,
    margin: float = 0.5,
    line_order: Optional[List[str]] = None,
    verbose: bool = True,
    background_color: Union[str, Tuple[float, float, float]] = None
) -> Optional[str]:
    """
    Create a PDF file from downloaded images, organized by line name.
    
    Each page shows images for one driver line with the line name as title.
    Images are arranged in a grid (default 5 columns x 3 rows = 15 images per page).
    If a line has more images than fit on one page, additional pages are created.
    
    Parameters
    ----------
    images_dir : str
        Directory containing images, organized by line name subdirectories.
        Expected structure: images_dir/LineName/*.png (or .jpg)
    output_pdf : str, optional
        Path for output PDF file. If None, saves to images_dir/images_summary.pdf
    images_per_page : tuple of (int, int)
        (columns, rows) - number of images per page. Default: (5, 3) = 15 images
    page_size : str
        Page size: 'A4', 'Letter', etc. Default: 'A4'
    landscape : bool
        Use landscape orientation. Default: True (horizontal A4)
    title_font_size : int
        Font size for line name title. Default: 14
    margin : float
        Page margin in inches. Default: 0.5
    line_order : list of str, optional
        Ordered list of line names for page ordering. Lines are sorted in this order,
        preserving the ranking used to select top-N lines (e.g., by weighted_score).
        Lines not in this list are appended at the end alphabetically.
        If None, lines are sorted alphabetically.
    verbose : bool
        Print progress messages. Default: True
    background_color : str or tuple, optional
        Background color for PDF pages. Can be named color ('black', 'white'),
        hex string ('#000000'), or RGB tuple (0-1). Default: None (white)
        
    Returns
    -------
    str or None
        Path to the created PDF file, or None if creation failed.
        
    Example
    -------
    >>> create_image_pdf(
    ...     images_dir='/path/to/images',
    ...     output_pdf='/path/to/summary.pdf',
    ...     images_per_page=(5, 3),
    ...     landscape=True,
    ...     background_color='black'
    ... )
    """
    try:
        from reportlab.lib.pagesizes import A4, letter, landscape as rl_landscape
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
        from PIL import Image
        HAS_REPORTLAB = True
    except ImportError:
        HAS_REPORTLAB = False
        
    if not HAS_REPORTLAB:
        if verbose:
            print("⚠️  PDF generation requires reportlab and Pillow.")
            print("   Install with: pip install reportlab Pillow")
        return None
    
    from pathlib import Path
    
    images_path = Path(images_dir)
    if not images_path.exists():
        if verbose:
            print(f"⚠️  Images directory not found: {images_dir}")
        return None
    
    # Find all line directories with images. The helper also scans the
    # source subfolders (neuronbridge/, flylight/) created by
    # download_source='both', so the summary never reports "No images"
    # when the images are nested one level deeper.
    line_images = _collect_line_images(images_path)
    
    if not line_images:
        if verbose:
            print(f"⚠️  No images found in: {images_dir}")
        return None
    
    # Sort images within each line
    for line_name in line_images:
        line_images[line_name] = sorted(line_images[line_name])
    
    # Determine line ordering for PDF pages
    if line_order:
        # Use provided order (preserves ranking like weighted_score)
        # Lines in line_order come first in that order, others appended alphabetically
        ordered_lines = []
        remaining_lines = set(line_images.keys())
        for line in line_order:
            if line in remaining_lines:
                ordered_lines.append(line)
                remaining_lines.remove(line)
        # Append any remaining lines alphabetically
        ordered_lines.extend(sorted(remaining_lines))
    else:
        # Default: alphabetical order
        ordered_lines = sorted(line_images.keys())
    
    # Set output path
    if output_pdf is None:
        output_pdf = images_path / 'images_summary.pdf'
    output_pdf = Path(output_pdf)
    
    # Page setup
    if page_size.upper() == 'A4':
        base_size = A4
    else:
        base_size = letter
    
    if landscape:
        page_width, page_height = rl_landscape(base_size)
    else:
        page_width, page_height = base_size
    
    cols, rows = images_per_page
    margin_pts = margin * inch
    
    # Calculate available space for images
    usable_width = page_width - 2 * margin_pts
    usable_height = page_height - 2 * margin_pts - 30  # 30 pts for title
    
    # Calculate cell size
    cell_width = usable_width / cols
    cell_height = usable_height / rows
    
    # Create PDF
    c = canvas.Canvas(str(output_pdf), pagesize=(page_width, page_height))
    
    total_pages = 0
    total_images = 0
    
    if verbose:
        if line_order:
            print(f"📄 Creating PDF from {len(line_images)} lines (ordered by ranking)...")
        else:
            print(f"📄 Creating PDF from {len(line_images)} lines (alphabetical order)...")
    
    for line_name in ordered_lines:
        images = line_images[line_name]
        # Calculate how many pages needed for this line
        images_per_full_page = cols * rows
        num_pages = (len(images) + images_per_full_page - 1) // images_per_full_page
        
        for page_idx in range(num_pages):
            # Apply background color
            text_color = colors.black
            if background_color:
                try:
                    # Parse color
                    if isinstance(background_color, str):
                        if background_color.startswith('#'):
                            bg_col = colors.HexColor(background_color)
                        else:
                            bg_col = getattr(colors, background_color, colors.white)
                    elif isinstance(background_color, (tuple, list)) and len(background_color) >= 3:
                        bg_col = colors.Color(*background_color[:3])
                    else:
                        bg_col = colors.white
                    
                    c.setFillColor(bg_col)
                    c.rect(0, 0, page_width, page_height, fill=1, stroke=0)
                    
                    # Set text color contrast
                    # Simple luminance check
                    if hasattr(bg_col, 'red'):
                        luminance = bg_col.red*0.299 + bg_col.green*0.587 + bg_col.blue*0.114
                        if luminance < 0.5:
                             text_color = colors.white
                except Exception:
                    pass

            # Draw title
            c.setFillColor(text_color)
            c.setFont("Helvetica-Bold", title_font_size)
            title = line_name
            if num_pages > 1:
                title += f" ({page_idx + 1}/{num_pages})"
            c.drawCentredString(page_width / 2, page_height - margin_pts - 5, title)
            
            # Get images for this page
            start_idx = page_idx * images_per_full_page
            end_idx = min(start_idx + images_per_full_page, len(images))
            page_images = images[start_idx:end_idx]
            
            # Calculate optimal layout for this page
            num_images = len(page_images)
            if num_images < images_per_full_page:
                # Auto-arrange for fewer images
                # Try to fill rows as much as possible
                actual_rows = (num_images + cols - 1) // cols
                if actual_rows < rows:
                    # Recalculate cell height to use more space
                    actual_cell_height = usable_height / actual_rows
                else:
                    actual_cell_height = cell_height
            else:
                actual_rows = rows
                actual_cell_height = cell_height
            
            # Draw images
            for i, img_path in enumerate(page_images):
                row = i // cols
                col = i % cols
                
                # Calculate position (top-left corner of cell)
                x = margin_pts + col * cell_width
                y = page_height - margin_pts - 25 - (row + 1) * actual_cell_height
                
                try:
                    # Open image to get dimensions
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
                        
                        # Calculate scaling to fit in cell with padding
                        padding = 5  # pixels padding
                        max_width = cell_width - 2 * padding
                        max_height = actual_cell_height - 2 * padding
                        
                        scale_w = max_width / img_width
                        scale_h = max_height / img_height
                        scale = min(scale_w, scale_h)
                        
                        draw_width = img_width * scale
                        draw_height = img_height * scale
                        
                        # Center in cell
                        draw_x = x + (cell_width - draw_width) / 2
                        draw_y = y + (actual_cell_height - draw_height) / 2
                        
                        # Draw the image
                        c.drawImage(
                            str(img_path),
                            draw_x, draw_y,
                            width=draw_width,
                            height=draw_height,
                            preserveAspectRatio=True
                        )
                        total_images += 1
                        
                except Exception as e:
                    # Skip problematic images
                    if verbose:
                        print(f"   ⚠️  Could not process: {img_path.name} - {e}")
            
            # Add new page
            c.showPage()
            total_pages += 1
    
    # Save PDF
    c.save()
    
    if verbose:
        print(f"✅ Created PDF: {output_pdf}")
        print(f"   {total_pages} pages, {total_images} images from {len(line_images)} lines")
    
    return str(output_pdf)


def create_image_pptx(
    images_dir: str,
    output_pptx: Optional[str] = None,
    images_per_slide: Tuple[int, int] = (5, 3),
    slide_size: str = 'widescreen',
    title_font_size: int = 24,
    label_font_size: int = 20,
    margin: float = 0.3,
    line_order: Optional[List[str]] = None,
    font_color: Tuple[int, int, int] = (0, 0, 0),
    background_color: Union[str, Tuple[int, int, int]] = None,
    verbose: bool = True
) -> Optional[str]:
    """
    Create a PPTX file from downloaded images, organized by line name.
    
    Each slide shows images for one driver line with the line name as title.
    Images are arranged in a grid (default 5 columns x 3 rows = 15 images per slide).
    If a line has more images than fit on one slide, additional slides are created.
    
    Parameters
    ----------
    images_dir : str
        Directory containing images, organized by line name subdirectories.
        Expected structure: images_dir/LineName/*.png (or .jpg)
    output_pptx : str, optional
        Path for output PPTX file. If None, saves to images_dir/images_summary.pptx
    images_per_slide : tuple of (int, int)
        (columns, rows) - number of images per slide. Default: (5, 3) = 15 images
    slide_size : str
        Slide dimensions: 'widescreen' (16:9), 'standard' (4:3), 'a4'. Default: 'widescreen'
    title_font_size : int
        Font size for line name title. Default: 24
    label_font_size : int
        Font size for image labels. Default: 20
    margin : float
        Slide margin in inches. Default: 0.3
    line_order : list of str, optional
        Ordered list of line names for slide ordering. Lines are sorted in this order,
        preserving the ranking used to select top-N lines (e.g., by weighted_score).
        Lines not in this list are appended at the end alphabetically.
        If None, lines are sorted alphabetically.
    font_color : tuple of (int, int, int), default (0, 0, 0)
        RGB color tuple for label text (r, g, b), each value 0-255. Default is black.
    background_color : str or tuple of (int, int, int), optional
        Background color for slides. Can be named color ('black', 'white'),
        hex string ('#000000'), or RGB tuple (0-255). Default: None (white).
        When set to 'black' or dark color, font_color auto-adjusts to white if not specified.
    verbose : bool
        Print progress messages. Default: True
        
    Returns
    -------
    str or None
        Path to the created PPTX file, or None if creation failed.
    """
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.enum.text import PP_ALIGN
        HAS_PPTX = True
    except ImportError:
        HAS_PPTX = False
    
    if not HAS_PPTX:
        if verbose:
            print("⚠️  PPTX generation requires python-pptx.")
            print("   Install with: pip install python-pptx")
        return None
    
    from pathlib import Path
    from PIL import Image
    
    images_path = Path(images_dir)
    if not images_path.exists():
        if verbose:
            print(f"⚠️  Images directory not found: {images_dir}")
        return None
    
    # Find all line directories with images. The helper also scans the
    # source subfolders (neuronbridge/, flylight/) created by
    # download_source='both', so the summary never reports "No images"
    # when the images are nested one level deeper.
    line_images = _collect_line_images(images_path)
    
    if not line_images:
        if verbose:
            print(f"⚠️  No images found in: {images_dir}")
        return None
    
    # Sort images within each line
    for line_name in line_images:
        line_images[line_name] = sorted(line_images[line_name])
    
    # Determine line ordering
    if line_order:
        ordered_lines = []
        remaining_lines = set(line_images.keys())
        for line in line_order:
            if line in remaining_lines:
                ordered_lines.append(line)
                remaining_lines.remove(line)
        ordered_lines.extend(sorted(remaining_lines))
    else:
        ordered_lines = sorted(line_images.keys())
    
    # Set output path
    if output_pptx is None:
        output_pptx = images_path / 'images_summary.pptx'
    output_pptx = Path(output_pptx)
    
    # Slide setup
    size_presets = {
        'widescreen': (13.333, 7.5),
        'standard': (10, 7.5),
        'a4': (11.69, 8.27),
    }
    slide_width, slide_height = size_presets.get(slide_size, size_presets['widescreen'])
    
    cols, rows = images_per_slide
    title_height = 0.5
    label_height = (label_font_size / 72) * 1.5
    
    # Calculate cell dimensions
    usable_width = slide_width - 2 * margin
    usable_height = slide_height - margin - title_height - margin
    cell_width = usable_width / cols
    cell_height = usable_height / rows
    
    # Create presentation
    prs = Presentation()
    prs.slide_width = Inches(slide_width)
    prs.slide_height = Inches(slide_height)
    blank_layout = prs.slide_layouts[6]
    
    # Parse background color and determine text color
    from pptx.dml.color import RGBColor
    bg_rgb = None
    text_color = font_color
    if background_color:
        try:
            if isinstance(background_color, str):
                if background_color.lower() == 'black':
                    bg_rgb = RGBColor(0, 0, 0)
                elif background_color.lower() == 'white':
                    bg_rgb = RGBColor(255, 255, 255)
                elif background_color.startswith('#'):
                    hex_str = background_color.lstrip('#')
                    r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
                    bg_rgb = RGBColor(r, g, b)
                else:
                    # Try named colors
                    color_map = {'gray': (128, 128, 128), 'grey': (128, 128, 128), 
                                 'darkgray': (64, 64, 64), 'darkgrey': (64, 64, 64)}
                    if background_color.lower() in color_map:
                        r, g, b = color_map[background_color.lower()]
                        bg_rgb = RGBColor(r, g, b)
            elif isinstance(background_color, (tuple, list)) and len(background_color) >= 3:
                bg_rgb = RGBColor(*background_color[:3])
            
            # Auto-adjust text color for dark backgrounds if font_color wasn't explicitly set to non-default
            if bg_rgb and font_color == (0, 0, 0):
                # Calculate luminance
                luminance = (bg_rgb[0]*0.299 + bg_rgb[1]*0.587 + bg_rgb[2]*0.114) / 255
                if luminance < 0.5:
                    text_color = (255, 255, 255)  # White text on dark background
        except Exception:
            pass
    
    total_slides = 0
    total_images = 0
    
    if verbose:
        if line_order:
            print(f"📊 Creating PPTX from {len(line_images)} lines (ordered by ranking)...")
        else:
            print(f"📊 Creating PPTX from {len(line_images)} lines (alphabetical order)...")
    
    for line_name in ordered_lines:
        images = line_images[line_name]
        images_per_full_slide = cols * rows
        num_slides = (len(images) + images_per_full_slide - 1) // images_per_full_slide
        
        for slide_idx in range(num_slides):
            slide = prs.slides.add_slide(blank_layout)
            
            # Apply background color
            if bg_rgb:
                # Set slide background
                background = slide.background
                fill = background.fill
                fill.solid()
                fill.fore_color.rgb = bg_rgb
            
            # Build title
            slide_title = line_name
            if num_slides > 1:
                slide_title += f" ({slide_idx + 1}/{num_slides})"
            
            # Add title
            txBox = slide.shapes.add_textbox(
                Inches(margin),
                Inches(margin / 2),
                Inches(slide_width - 2 * margin),
                Inches(title_height)
            )
            tf = txBox.text_frame
            p = tf.paragraphs[0]
            p.text = slide_title
            p.font.size = Pt(title_font_size)
            p.font.bold = True
            p.font.color.rgb = RGBColor(*text_color)
            p.alignment = PP_ALIGN.CENTER
            
            # Get images for this slide
            start_idx = slide_idx * images_per_full_slide
            end_idx = min(start_idx + images_per_full_slide, len(images))
            slide_images = images[start_idx:end_idx]
            
            content_top = margin + title_height
            
            for i, img_path in enumerate(slide_images):
                row = i // cols
                col = i % cols
                
                cell_left = margin + col * cell_width
                cell_top = content_top + row * cell_height
                
                try:
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
                        
                        max_width = cell_width - 0.1
                        max_height = cell_height - label_height - 0.1
                        
                        scale_w = max_width / (img_width / 96)
                        scale_h = max_height / (img_height / 96)
                        scale_factor = min(scale_w, scale_h)
                        
                        final_width = (img_width / 96) * scale_factor
                        final_height = (img_height / 96) * scale_factor
                        
                        img_left = cell_left + (cell_width - final_width) / 2
                        img_top = cell_top + (cell_height - label_height - final_height) / 2
                        
                        slide.shapes.add_picture(
                            str(img_path),
                            Inches(img_left),
                            Inches(img_top),
                            Inches(final_width),
                            Inches(final_height)
                        )
                        
                        # Add label
                        label = img_path.stem
                        max_chars = int(cell_width * 8)
                        if len(label) > max_chars:
                            label = label[:max_chars-3] + '...'
                        
                        label_box = slide.shapes.add_textbox(
                            Inches(cell_left),
                            Inches(cell_top + cell_height - label_height),
                            Inches(cell_width),
                            Inches(label_height)
                        )
                        tf = label_box.text_frame
                        p = tf.paragraphs[0]
                        p.text = label
                        p.font.size = Pt(label_font_size)
                        p.font.color.rgb = RGBColor(*text_color)
                        p.alignment = PP_ALIGN.CENTER
                        
                        total_images += 1
                        
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Could not process: {img_path.name} - {e}")
            
            total_slides += 1
    
    prs.save(str(output_pptx))
    
    if verbose:
        print(f"✅ Created PPTX: {output_pptx}")
        print(f"   {total_slides} slides, {total_images} images from {len(line_images)} lines")
    
    return str(output_pptx)


# Convenience function for quick usage
def find_lines_for_body(body_id: int, match_type: str = 'cds') -> pd.DataFrame:
    """
    Quick function to find driver lines matching a body ID.
    
    Parameters
    ----------
    body_id : int
        The EM body ID to search for.
    match_type : str
        Match algorithm: 'cds', 'pppm', or 'both'. Default: 'cds'
        
    Returns
    -------
    pd.DataFrame
        DataFrame of matching lines, sorted by score (cds/pppm) or combined_rank (both).
    """
    nbf = NeuronBridgeFinder(verbose=False)
    return nbf.id_to_lines(body_id, match_type=match_type)


def find_neurons_for_line(line_name: str, match_type: str = 'cds') -> pd.DataFrame:
    """
    Quick function to find neurons matching a driver line.
    
    Parameters
    ----------
    line_name : str
        The driver line name to search for.
    match_type : str
        Match algorithm: 'cds', 'pppm', or 'both'. Default: 'cds'
        
    Returns
    -------
    pd.DataFrame
        DataFrame of matching neurons, sorted by score (cds/pppm) or combined_rank (both).
    """
    nbf = NeuronBridgeFinder(verbose=False)
    return nbf.line_to_neuron(line_name, match_type=match_type)


if __name__ == '__main__':
    # Simple test
    print("Testing NeuronBridgeFinder...")
    
    # Initialize
    nbf = NeuronBridgeFinder()
    
    # Test id_to_lines
    print("\n--- Test id_to_lines ---")
    lines = nbf.id_to_lines(636798093)
    print(lines.head(10))
    
    # Test line_to_neuron
    print("\n--- Test line_to_neuron ---")
    neurons = nbf.line_to_neuron('LH173')
    print(neurons.head(10))
