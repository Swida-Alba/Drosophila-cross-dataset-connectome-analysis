"""
Connectivity Profile Extraction Module

This module provides tools for extracting and comparing connectivity profiles
(fingerprints) across neurons and datasets. A connectivity profile captures
the top upstream and downstream partners of a neuron, enabling cross-dataset
verification of neuron type assignments and homolog discovery.

Main components:
- ConnectivityProfile: Dataclass storing a neuron's connectivity fingerprint
- ProfilerConfig: Configuration for profile extraction
- FuzzyMatchConfig: Configuration for fuzzy partner type matching
- ConnectivityProfiler: Main class for extracting profiles from datasets

Example:
    >>> from src.comparison import ConnectivityProfiler, ProfilerConfig
    >>> 
    >>> config = ProfilerConfig(top_k_bodyid=10)
    >>> profiler = ConnectivityProfiler(
    ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    ...     config=config
    ... )
    >>> 
    >>> profile = profiler.get_profile('aMe12', 'hemibrain:v1.2.1')
    >>> print(profile.upstream_partners)
"""

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from ..utils.naming_utils import canonical_dataset_name
except ImportError:  # pragma: no cover - direct package imports
    from utils.naming_utils import canonical_dataset_name

try:
    from ..flywire_ids import (is_local_connectome_dataset,
                               neuprint_dataset_name)
except ImportError:  # pragma: no cover - direct package imports
    from flywire_ids import is_local_connectome_dataset, neuprint_dataset_name


# ============================================================================
# Connectivity Status Classification
# ============================================================================

def progress_bars_disabled(show_progress: bool = True, verbose: bool = True) -> bool:
    """Whether raw tqdm bars should be suppressed.

    The UI runs tools as subprocesses with piped (non-TTY) stdout; every
    tqdm refresh then lands in the log panel as a separate stale line
    ("Loading data: 0%|...|" repeated between each log message).  The
    structured ``[DROCAT][progress]`` step events already drive the UI's
    determinate progress bar, so raw bars are only useful on a real
    terminal.
    """
    if not show_progress or not verbose:
        return True
    try:
        return not sys.stdout.isatty()
    except Exception:
        return True


def _escape_cypher_string_fallback(value):
    """Inline escape fallback (only used when src.utils.api_utils is unavailable)."""
    if not isinstance(value, str):
        return str(value)
    return value.replace('\\', '\\\\').replace("'", "\\'")


class ConnectivityStatus(Enum):
    """
    Hierarchical classification of neuron connectivity profile status.
    
    This enum provides a precise classification of connectivity quality,
    ordered from worst (NONE) to best (COMPLETE):
    
    - NONE: 0 partners in both directions - no valid connectivity profile
    - ORPHAN: Alias for NONE (0 partners both directions) for clarity in reports
    - UNIDIRECTIONAL: 0 partners in one direction, >0 in the other
    - RARE: < 5 partners in either direction - too few for reliable comparison
    - INCOMPLETE: fewer connections than top_k criteria in either direction
    - INCOMPLETE_EXPANSION: has top_k partners but < top_m unique types
    - COMPLETE: has both top_k connections and top_m unique types
    
    The hierarchical order is important:
    1. First check if 0 partners (NONE)
    2. Then check if < 5 partners (RARE) 
    3. Then check if < top_k partners (INCOMPLETE)
    4. Then check if < top_m types (INCOMPLETE_EXPANSION)
    5. Otherwise COMPLETE
    
    Note: NONE and RARE are considered invalid for comparison and should be
    skipped. INCOMPLETE and INCOMPLETE_EXPANSION may produce unreliable results
    but can still be compared with appropriate warnings.
    """
    NONE = "none"  # 0 partners both directions - no connectivity
    ORPHAN = "orphan"  # Alias: both directions 0 partners
    UNIDIRECTIONAL = "unidirectional"  # One direction has 0 partners
    RARE = "rare"  # < 5 partners in either direction - too sparse for reliable comparison
    INCOMPLETE = "incomplete"  # < top_k partners - incomplete profile
    INCOMPLETE_EXPANSION = "incomplete_expansion"  # < top_m types - insufficient type diversity
    COMPLETE = "complete"  # Full profile meeting all criteria
    
    def is_valid_for_comparison(self) -> bool:
        """Check if this status is valid for comparison (not NONE).
        
        Note: RARE and UNIDIRECTIONAL sources are included but should be treated with caution.
        NONE/ORPHAN sources have no partners and cannot be compared at all.
        """
        return self not in {ConnectivityStatus.NONE, ConnectivityStatus.ORPHAN}
    
    def requires_warning(self) -> bool:
        """Check if this status requires a WARNING when used in comparison.
        
        Returns True for RARE sources (< 5 partners) which may produce
        unreliable comparison results.
        """
        return self == ConnectivityStatus.RARE
    
    def is_complete(self) -> bool:
        """Check if this status represents a complete profile."""
        return self == ConnectivityStatus.COMPLETE
    
    @classmethod
    def get_description(cls, status: 'ConnectivityStatus') -> str:
        """Get human-readable description of the status."""
        descriptions = {
            cls.NONE: "No connections (0 partners both directions)",
            cls.ORPHAN: "No connections (0 partners both directions)",
            cls.UNIDIRECTIONAL: "Connections in only one direction",
            cls.RARE: "Rare (<5 partners in either direction)",
            cls.INCOMPLETE: "Incomplete (fewer than top_k partners)",
            cls.INCOMPLETE_EXPANSION: "Incomplete expansion (fewer than top_m unique types)",
            cls.COMPLETE: "Complete (meets all criteria)"
        }
        return descriptions.get(status, "Unknown")


# ============================================================================
# Custom Exceptions
# ============================================================================

class DataNotAvailableError(Exception):
    """
    Raised when connection data is not available for a dataset.
    
    This error provides guidance on how to build the missing cache.
    """
    pass


# ============================================================================
# Module-Level Connection Data Cache
# ============================================================================
# Cache structure: {dataset_key: {'conn_df': DataFrame, 'type_lookup': dict}}
# This is shared across all ConnectivityProfiler instances to avoid repeated
# disk reads when processing multiple neurons/types from the same dataset.

import threading

_PROFILER_CONN_CACHE: Dict[str, Dict[str, Any]] = {}
_PROFILER_CONN_CACHE_LOCK = threading.Lock()
_PROFILER_CACHE_LOGGED: set = set()  # Track which datasets have been logged


def clear_profiler_conn_cache():
    """Clear the module-level connection data cache."""
    global _PROFILER_CONN_CACHE, _PROFILER_CACHE_LOGGED
    with _PROFILER_CONN_CACHE_LOCK:
        _PROFILER_CONN_CACHE.clear()
        _PROFILER_CACHE_LOGGED.clear()


def get_profiler_conn_cache_info() -> Dict[str, Any]:
    """Get info about cached connection data."""
    info = {}
    for dataset, data in _PROFILER_CONN_CACHE.items():
        info[dataset] = {
            'conn_df_rows': len(data.get('conn_df', [])) if data.get('conn_df') is not None else 0,
            'type_lookup_size': len(data.get('type_lookup', {})),
        }
    return info


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class FuzzyMatchConfig:
    """
    Configuration for fuzzy partner type matching across datasets.
    
    Cross-dataset type names may have variations like:
    - T4a_R vs T4a vs T4a_L (laterality suffix)
    - aMe12_1 vs aMe12 (numeric suffix)
    - T4a-like vs T4a (qualitative suffix)
    
    This config controls how partner types are normalized for comparison.
    
    Attributes:
        enabled: Enable fuzzy matching (default: True)
        strip_lr_suffix: Remove _L, _R, _left, _right suffixes
        strip_numeric_suffix: Remove trailing numeric suffixes (_1, _2, etc.)
        strip_like_suffix: Remove -like or _like suffixes
        case_insensitive: Convert to lowercase for matching
        custom_mappings: User-defined type name mappings
    
    Example:
        >>> config = FuzzyMatchConfig(strip_lr_suffix=True)
        >>> normalize_partner_type('T4a_R', config)
        't4a'
    
    Note:
        Default is now DISABLED (exact matching) to prevent grouping of similar types.
        Types like R8, R7, Dm9 will be kept as-is instead of being stripped to 'r', 'dm'.
    """
    enabled: bool = False  # Disabled by default - use exact type matching
    strip_lr_suffix: bool = False  # Don't strip _L/_R suffixes
    strip_numeric_suffix: bool = False  # Don't strip trailing numbers (keeps R8, Dm9 etc.)
    strip_like_suffix: bool = False  # Don't strip -like suffixes
    case_insensitive: bool = False  # Case sensitive matching
    custom_mappings: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProfilerConfig:
    """
    Configuration for ConnectivityProfiler.
    
    Configuration for connectivity profile extraction and comparison.
    
    Key defaults:
    - top_k_bodyid=5: Focus on top 5 strongest connections per direction
    - top_m_type=0: No minimum type requirement (set >0 for type diversity needs)
    - normalize_method='rank': Use rank-based comparison (always used for bodyId comparison)
    
    Hybrid 2-hop profile features:
    - expand_untyped_2hop: Fetch 2-hop typed partners for 1-hop untyped neurons (default: True)
    - top_k_2hop: Max 2-hop partners to fetch per untyped 1-hop neuron (default: 5)
    - If 2-hop doesn't return any typed neuron in top_k, the untyped partner is ignored
    
    Note on removed parameters:
    - max_untyped_fraction: No longer needed - hybrid implementation handles untyped automatically
    - allow_2hop_expansion: Replaced by expand_untyped_2hop
    - use_ranks: Always True for bodyId comparison (proportions not used)
    
    Attributes:
        top_k_bodyid: Top K connections by bodyId (default: 5)
        top_m_type: Minimum unique types to ensure in profile (default: 0 = no expansion)
        min_synapse_threshold: Filter connections with fewer synapses (default: 3)
        include_untyped_partners: Include partners without type annotations (default: False)
        normalize_method: Weight normalization method ('rank' or 'both')
        use_cache: Enable profile caching to disk (default: True)
        fuzzy_match: Fuzzy matching configuration for partner type names
        dynamic_expansion: Dynamically expand K until M types reached (default: True)
        max_expansion_factor: Maximum expansion multiplier for K (default: 5)
        expand_untyped_2hop: Enable 2-hop expansion for untyped 1-hop partners
        top_k_2hop: Number of top 2-hop partners to fetch per untyped neuron
        use_bodyid_for_intra: Keep bodyId for intra-dataset comparison
    
    Example:
        >>> config = ProfilerConfig(
        ...     top_k_bodyid=5,  # Focus on top 5 partners
        ...     top_m_type=0,    # No minimum type requirement
        ...     min_synapse_threshold=3,
        ...     expand_untyped_2hop=True  # Enable 2-hop expansion for untyped
        ... )
    """
    top_k_bodyid: int = 5  # Top K connections by bodyId (default: 5)
    top_m_type: int = 0  # Minimum unique types (default: 0 = no expansion)
    min_synapse_threshold: int = 3
    include_untyped_partners: bool = False
    normalize_method: str = 'rank'  # 'rank' (default, always used for comparison) or 'both'
    use_cache: bool = True
    fuzzy_match: FuzzyMatchConfig = field(default_factory=FuzzyMatchConfig)
    dynamic_expansion: bool = True  # Expand K until M types reached
    max_expansion_factor: int = 5  # Max K expansion multiplier
    # Hybrid 2-hop profile options
    expand_untyped_2hop: bool = True  # Expand untyped 1-hop partners to 2-hop typed partners
    top_k_2hop: int = 5  # Max 2-hop partners per untyped 1-hop neuron
    use_bodyid_for_intra: bool = True  # Keep bodyId for intra-dataset matching


# ============================================================================
# Connectivity Profile Dataclass
# ============================================================================

@dataclass
class ConnectivityProfile:
    """
    Connectivity fingerprint for a neuron or neuron group.
    
    Contains normalized weight distributions and rank information for
    upstream and downstream partners, enabling cross-dataset comparison.
    
    Round 5 additions:
    - unique_types_upstream/downstream: Count of unique types in profile
    - is_sparse: Flag if unique_types < top_m_type after expansion
    - partner_type_mapping: bodyId → type mapping for partners (metadata)
    - top_k_bodyid_used: Actual K used after dynamic expansion
    - top_m_type_target: Target M for unique types
    
    Attributes:
        neuron_id: Neuron type name, bodyId, or list of bodyIds
        dataset: Dataset identifier (e.g., 'hemibrain:v1.2.1')
        upstream_partners: Partner type → actual synapse weight (not normalized)
        downstream_partners: Partner type → actual synapse weight (not normalized)
        upstream_ranks: Partner type → rank (1 = strongest, derived from weights)
        downstream_ranks: Partner type → rank (1 = strongest, derived from weights)
        upstream_top_k: Number of top partners stored
        downstream_top_k: Number of top partners stored
        total_upstream_weight: Raw total weight (pre-normalization)
        total_downstream_weight: Raw total weight
        num_neurons_aggregated: Count of neurons if type-based aggregation
        upstream_weight_variance: Per-partner weight variance (for multi-neuron types)
        downstream_weight_variance: Per-partner weight variance
        untyped_upstream_count: Number of untyped upstream partners excluded
        untyped_downstream_count: Number of untyped downstream partners excluded
        untyped_upstream_weight_fraction: Fraction of weight from untyped partners
        untyped_downstream_weight_fraction: Fraction of weight from untyped partners
        actual_upstream_count: Actual number of partners found
        actual_downstream_count: Actual number of partners found
        is_weak_connectivity: Flag if < 5 partners in either direction
        unique_types_upstream: Count of unique types in upstream profile (Round 5)
        unique_types_downstream: Count of unique types in downstream profile (Round 5)
        is_sparse: Flag if unique_types < top_m_type after expansion (Round 5)
        partner_type_mapping_upstream: bodyId → type mapping for upstream partners (Round 5)
        partner_type_mapping_downstream: bodyId → type mapping for downstream partners (Round 5)
        top_k_bodyid_used: Actual K used after dynamic expansion (Round 5)
        top_m_type_target: Target M for unique types (Round 5)
    """
    neuron_id: Union[str, int, List]
    dataset: str
    # Optional type label for the neuron/profile (set for type-level queries or aggregates)
    neuron_type: Optional[str] = None
    
    # Partner data: partner_type → actual synapse weight (enables easy aggregation)
    # Weights are stored as actual synapse counts (typically < 4096 per pair)
    # This preserves rank structure and allows bodyId→type aggregation via sum
    upstream_partners: Dict[str, float] = field(default_factory=dict)
    downstream_partners: Dict[str, float] = field(default_factory=dict)
    
    # Rank-based representation (derived from weights: higher weight = lower rank)
    upstream_ranks: Dict[str, int] = field(default_factory=dict)
    downstream_ranks: Dict[str, int] = field(default_factory=dict)
    
    # Metadata
    upstream_top_k: int = 10
    downstream_top_k: int = 10
    total_upstream_weight: float = 0.0
    total_downstream_weight: float = 0.0
    num_neurons_aggregated: int = 1
    
    # Variance for multi-neuron types
    upstream_weight_variance: Optional[Dict[str, float]] = None
    downstream_weight_variance: Optional[Dict[str, float]] = None
    
    # Untyped partner statistics
    untyped_upstream_count: int = 0
    untyped_downstream_count: int = 0
    untyped_upstream_weight_fraction: float = 0.0
    untyped_downstream_weight_fraction: float = 0.0
    
    # Sparse profile detection
    actual_upstream_count: int = 0
    actual_downstream_count: int = 0
    is_weak_connectivity: bool = False
    
    # Round 5: Unique type counts and sparse detection
    unique_types_upstream: int = 0
    unique_types_downstream: int = 0
    is_sparse: bool = False  # True if unique_types < top_m_type after expansion
    
    # Round 5: Partner type mapping (bodyId → type) for metadata
    partner_type_mapping_upstream: Optional[Dict[int, str]] = None
    partner_type_mapping_downstream: Optional[Dict[int, str]] = None
    
    # Round 5: Dynamic expansion tracking
    top_k_bodyid_used: int = 20  # Actual K used after expansion
    top_m_type_target: int = 5  # Target M for unique types
    
    # Round 6: Hybrid 2-hop profile for untyped 1-hop partners
    # For untyped 1-hop partners, we store:
    # - bodyId (for intra-dataset comparison)
    # - 2-hop typed partners (for cross-dataset comparison)
    untyped_upstream_bodyids: Optional[Dict[int, float]] = None  # bodyId → weight for untyped 1-hop upstream
    untyped_downstream_bodyids: Optional[Dict[int, float]] = None  # bodyId → weight for untyped 1-hop downstream
    untyped_upstream_2hop: Optional[Dict[int, Dict[str, float]]] = None  # untyped_bodyId → {2hop_type → weight}
    untyped_downstream_2hop: Optional[Dict[int, Dict[str, float]]] = None  # untyped_bodyId → {2hop_type → weight}
    untyped_upstream_2hop_ranks: Optional[Dict[int, Dict[str, int]]] = None  # untyped_bodyId → {2hop_type → rank}
    untyped_downstream_2hop_ranks: Optional[Dict[int, Dict[str, int]]] = None  # untyped_bodyId → {2hop_type → rank}
    
    # Round 7: BodyId-level data for typed partners (for intra-dataset comparison)
    # Stores bodyId → weight for ALL typed 1-hop partners (not aggregated by type)
    typed_upstream_bodyids: Optional[Dict[int, float]] = None  # typed bodyId → weight
    typed_downstream_bodyids: Optional[Dict[int, float]] = None  # typed bodyId → weight
    
    # Minimum partners threshold for weak connectivity warning
    MIN_PARTNERS_THRESHOLD: int = field(default=5, repr=False)
    
    # Connectivity status classification (set in __post_init__)
    _connectivity_status: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Check for weak connectivity and sparse profiles, compute connectivity status."""
        # Compute connectivity status using hierarchical classification
        self._connectivity_status = self._compute_connectivity_status().value
        
        # is_weak_connectivity is now derived from connectivity_status for backward compatibility
        # NONE and RARE are considered "weak"
        status = ConnectivityStatus(self._connectivity_status)
        self.is_weak_connectivity = not status.is_valid_for_comparison()
        
        # Sparse profile check (Round 5) - still tracked separately
        if self.unique_types_upstream > 0 or self.unique_types_downstream > 0:
            if (self.unique_types_upstream < self.top_m_type_target or 
                self.unique_types_downstream < self.top_m_type_target):
                self.is_sparse = True
    
    def _compute_connectivity_status(self) -> ConnectivityStatus:
        """
        Compute connectivity status using hierarchical classification.
        
        The hierarchy is:
        1. NONE: 0 partners in either direction
        2. RARE: < 5 partners in either direction (MIN_PARTNERS_THRESHOLD)
        3. INCOMPLETE: fewer connections than top_k criteria in either direction
        4. INCOMPLETE_EXPANSION: has top_k but < top_m unique types
        5. COMPLETE: meets all criteria
        
        Returns:
            ConnectivityStatus enum value
        """
        # Get partner counts (for bodyId-level profiles, use typed_*_bodyids if available)
        up_count = self.actual_upstream_count
        down_count = self.actual_downstream_count
        
        # Level 1: Check for NONE/ORPHAN (0 partners both directions)
        if up_count == 0 and down_count == 0:
            return ConnectivityStatus.NONE

        # Level 1b: Check for UNIDIRECTIONAL (one direction missing)
        if up_count == 0 or down_count == 0:
            return ConnectivityStatus.UNIDIRECTIONAL
        
        # Level 2: Check for RARE (< 5 partners)
        if up_count < self.MIN_PARTNERS_THRESHOLD or down_count < self.MIN_PARTNERS_THRESHOLD:
            return ConnectivityStatus.RARE
        
        # Level 3: Check for INCOMPLETE (< top_k partners)
        # We check against actual_upstream/downstream_count vs the target top_k
        top_k_target = self.top_k_bodyid_used  # This is the actual K used
        if up_count < top_k_target or down_count < top_k_target:
            return ConnectivityStatus.INCOMPLETE
        
        # Level 4: Check for INCOMPLETE_EXPANSION (< top_m unique types)
        # Only applies if top_m_type_target > 0 (dynamic expansion enabled)
        if self.top_m_type_target > 0:
            up_types = self.unique_types_upstream
            down_types = self.unique_types_downstream
            if up_types < self.top_m_type_target or down_types < self.top_m_type_target:
                return ConnectivityStatus.INCOMPLETE_EXPANSION
        
        # Level 5: COMPLETE
        return ConnectivityStatus.COMPLETE
    
    @property
    def connectivity_status(self) -> ConnectivityStatus:
        """Get the connectivity status enum."""
        if self._connectivity_status is None:
            self._connectivity_status = self._compute_connectivity_status().value
        return ConnectivityStatus(self._connectivity_status)
    
    @property
    def connectivity_status_str(self) -> str:
        """Get the connectivity status as a string."""
        return self.connectivity_status.value
    
    def is_valid_for_comparison(self) -> bool:
        """Check if this profile is valid for comparison (not NONE or RARE)."""
        return self.connectivity_status.is_valid_for_comparison()
    
    def to_dict(self) -> dict:
        """Convert profile to serializable dictionary."""
        # Helper to convert numpy types to Python native types
        def to_native(val):
            if hasattr(val, 'item'):
                return val.item()  # numpy scalar -> Python scalar
            return val
        
        result = {
            'neuron_id': str(self.neuron_id) if not isinstance(self.neuron_id, str) else self.neuron_id,
            'dataset': self.dataset,
            'neuron_type': self.neuron_type,
            'upstream_partners': {str(k): float(v) for k, v in self.upstream_partners.items()},
            'downstream_partners': {str(k): float(v) for k, v in self.downstream_partners.items()},
            'upstream_ranks': {str(k): int(v) for k, v in self.upstream_ranks.items()},
            'downstream_ranks': {str(k): int(v) for k, v in self.downstream_ranks.items()},
            'upstream_top_k': int(self.upstream_top_k),
            'downstream_top_k': int(self.downstream_top_k),
            'total_upstream_weight': to_native(self.total_upstream_weight),
            'total_downstream_weight': to_native(self.total_downstream_weight),
            'num_neurons_aggregated': int(self.num_neurons_aggregated),
            'upstream_weight_variance': to_native(self.upstream_weight_variance) if self.upstream_weight_variance else None,
            'downstream_weight_variance': to_native(self.downstream_weight_variance) if self.downstream_weight_variance else None,
            'untyped_upstream_count': int(self.untyped_upstream_count),
            'untyped_downstream_count': int(self.untyped_downstream_count),
            'untyped_upstream_weight_fraction': float(self.untyped_upstream_weight_fraction),
            'untyped_downstream_weight_fraction': float(self.untyped_downstream_weight_fraction),
            'actual_upstream_count': int(self.actual_upstream_count),
            'actual_downstream_count': int(self.actual_downstream_count),
            'is_weak_connectivity': bool(self.is_weak_connectivity),
            'connectivity_status': self.connectivity_status_str,  # Hierarchical status
            # Round 5 additions
            'unique_types_upstream': int(self.unique_types_upstream),
            'unique_types_downstream': int(self.unique_types_downstream),
            'is_sparse': bool(self.is_sparse),
            'top_k_bodyid_used': int(self.top_k_bodyid_used),
            'top_m_type_target': int(self.top_m_type_target),
        }
        
        # Add partner type mappings - always include for consistent parquet schema
        result['partner_type_mapping_upstream'] = (
            {str(k): str(v) for k, v in self.partner_type_mapping_upstream.items()}
            if self.partner_type_mapping_upstream else {}
        )
        result['partner_type_mapping_downstream'] = (
            {str(k): str(v) for k, v in self.partner_type_mapping_downstream.items()}
            if self.partner_type_mapping_downstream else {}
        )
        
        # Round 6: Add untyped 1-hop bodyIds and 2-hop partner data
        # Always include for consistent parquet schema
        result['untyped_upstream_bodyids'] = (
            {str(k): float(v) for k, v in self.untyped_upstream_bodyids.items()}
            if self.untyped_upstream_bodyids else {}
        )
        result['untyped_downstream_bodyids'] = (
            {str(k): float(v) for k, v in self.untyped_downstream_bodyids.items()}
            if self.untyped_downstream_bodyids else {}
        )
        
        # 2-hop typed partners for untyped 1-hop: {untyped_bodyId: {2hop_type: weight}}
        result['untyped_upstream_2hop'] = (
            {
                str(bid): {str(t): float(w) for t, w in types.items()}
                for bid, types in self.untyped_upstream_2hop.items()
            }
            if self.untyped_upstream_2hop else {}
        )
        result['untyped_downstream_2hop'] = (
            {
                str(bid): {str(t): float(w) for t, w in types.items()}
                for bid, types in self.untyped_downstream_2hop.items()
            }
            if self.untyped_downstream_2hop else {}
        )
        
        # 2-hop ranks
        result['untyped_upstream_2hop_ranks'] = (
            {
                str(bid): {str(t): int(r) for t, r in types.items()}
                for bid, types in self.untyped_upstream_2hop_ranks.items()
            }
            if self.untyped_upstream_2hop_ranks else {}
        )
        result['untyped_downstream_2hop_ranks'] = (
            {
                str(bid): {str(t): int(r) for t, r in types.items()}
                for bid, types in self.untyped_downstream_2hop_ranks.items()
            }
            if self.untyped_downstream_2hop_ranks else {}
        )
        
        # Round 7: Add typed 1-hop bodyId data for intra-dataset comparison
        # Always include these keys (even if empty) to ensure consistent parquet schema
        result['typed_upstream_bodyids'] = (
            {str(k): float(v) for k, v in self.typed_upstream_bodyids.items()}
            if self.typed_upstream_bodyids else {}
        )
        result['typed_downstream_bodyids'] = (
            {str(k): float(v) for k, v in self.typed_downstream_bodyids.items()}
            if self.typed_downstream_bodyids else {}
        )
        
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ConnectivityProfile':
        """Create profile from dictionary."""
        # Remove the non-init field before creating instance
        data = data.copy()
        data.pop('MIN_PARTNERS_THRESHOLD', None)
        
        # Handle neuron_type (optional)
        if 'neuron_type' not in data:
            data['neuron_type'] = None
        
        # Handle connectivity_status - store as _connectivity_status for the instance
        # Will be recomputed in __post_init__ but we preserve it if present
        if 'connectivity_status' in data:
            data['_connectivity_status'] = data.pop('connectivity_status')
        
        # Convert partner_type_mapping keys back to int if present
        if 'partner_type_mapping_upstream' in data and data['partner_type_mapping_upstream']:
            data['partner_type_mapping_upstream'] = {
                int(k): v for k, v in data['partner_type_mapping_upstream'].items()
            }
        if 'partner_type_mapping_downstream' in data and data['partner_type_mapping_downstream']:
            data['partner_type_mapping_downstream'] = {
                int(k): v for k, v in data['partner_type_mapping_downstream'].items()
            }
        
        # Round 6: Convert untyped bodyId keys back to int
        if 'untyped_upstream_bodyids' in data and data['untyped_upstream_bodyids']:
            data['untyped_upstream_bodyids'] = {
                int(k): v for k, v in data['untyped_upstream_bodyids'].items()
            }
        if 'untyped_downstream_bodyids' in data and data['untyped_downstream_bodyids']:
            data['untyped_downstream_bodyids'] = {
                int(k): v for k, v in data['untyped_downstream_bodyids'].items()
            }
        
        # Convert 2-hop data keys back to int
        if 'untyped_upstream_2hop' in data and data['untyped_upstream_2hop']:
            data['untyped_upstream_2hop'] = {
                int(bid): types for bid, types in data['untyped_upstream_2hop'].items()
            }
        if 'untyped_downstream_2hop' in data and data['untyped_downstream_2hop']:
            data['untyped_downstream_2hop'] = {
                int(bid): types for bid, types in data['untyped_downstream_2hop'].items()
            }
        if 'untyped_upstream_2hop_ranks' in data and data['untyped_upstream_2hop_ranks']:
            data['untyped_upstream_2hop_ranks'] = {
                int(bid): types for bid, types in data['untyped_upstream_2hop_ranks'].items()
            }
        if 'untyped_downstream_2hop_ranks' in data and data['untyped_downstream_2hop_ranks']:
            data['untyped_downstream_2hop_ranks'] = {
                int(bid): types for bid, types in data['untyped_downstream_2hop_ranks'].items()
            }
        
        # Round 7: Convert typed bodyId keys back to int
        if 'typed_upstream_bodyids' in data and data['typed_upstream_bodyids']:
            data['typed_upstream_bodyids'] = {
                int(k): v for k, v in data['typed_upstream_bodyids'].items()
            }
        if 'typed_downstream_bodyids' in data and data['typed_downstream_bodyids']:
            data['typed_downstream_bodyids'] = {
                int(k): v for k, v in data['typed_downstream_bodyids'].items()
            }
        
        return cls(**data)
    
    def get_all_partner_types(self) -> set:
        """Get set of all partner types (upstream + downstream)."""
        return set(self.upstream_partners.keys()) | set(self.downstream_partners.keys())
    
    def get_partner_weight(self, partner_type: str, direction: str = 'both') -> float:
        """Get actual synapse weight for a partner type."""
        up = self.upstream_partners.get(partner_type, 0.0)
        down = self.downstream_partners.get(partner_type, 0.0)
        
        if direction == 'upstream':
            return up
        elif direction == 'downstream':
            return down
        else:
            return up + down
    
    def get_proportions(self, direction: str = 'both') -> Dict[str, float]:
        """
        Get normalized proportions from actual weights (on-demand).
        
        This converts actual synapse weights to proportions (sum to 1.0)
        for comparison methods that need normalized values.
        
        Args:
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Dict of partner_type → proportion (0.0 to 1.0)
        """
        if direction == 'upstream':
            partners = self.upstream_partners
        elif direction == 'downstream':
            partners = self.downstream_partners
        else:
            # Combine both directions
            partners = {}
            for k, v in self.upstream_partners.items():
                partners[k] = partners.get(k, 0.0) + v
            for k, v in self.downstream_partners.items():
                partners[k] = partners.get(k, 0.0) + v
        
        total = sum(partners.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in partners.items()}
    
    def get_partner_bodyids(self, partner_type: str, direction: str = 'upstream') -> List[int]:
        """Get bodyIds for a specific partner type (Round 5)."""
        mapping = (self.partner_type_mapping_upstream if direction == 'upstream' 
                   else self.partner_type_mapping_downstream)
        if not mapping:
            return []
        return [bid for bid, ptype in mapping.items() if ptype == partner_type]
    
    def summary(self) -> str:
        """Generate human-readable summary of the profile."""
        status = self.connectivity_status
        status_icon = {
            ConnectivityStatus.NONE: "❌",
            ConnectivityStatus.RARE: "⚠️",
            ConnectivityStatus.INCOMPLETE: "📉",
            ConnectivityStatus.INCOMPLETE_EXPANSION: "📊",
            ConnectivityStatus.COMPLETE: "✅"
        }.get(status, "?")
        
        lines = [
            f"ConnectivityProfile for {self.neuron_id} ({self.dataset})",
            f"  Status: {status_icon} {status.value.upper()} - {ConnectivityStatus.get_description(status)}",
            f"  Neurons aggregated: {self.num_neurons_aggregated}",
            f"  Upstream partners: {self.actual_upstream_count} (top-{self.upstream_top_k} stored, {self.unique_types_upstream} unique types)",
            f"  Downstream partners: {self.actual_downstream_count} (top-{self.downstream_top_k} stored, {self.unique_types_downstream} unique types)",
        ]
        
        # Add specific warnings based on status
        if status == ConnectivityStatus.NONE:
            lines.append("  ❌ NO CONNECTIVITY: Cannot be used for comparison")
        elif status == ConnectivityStatus.RARE:
            lines.append("  ⚠️ RARE CONNECTIVITY: Too few partners (<5) for reliable comparison")
        elif status == ConnectivityStatus.INCOMPLETE:
            lines.append(f"  📉 INCOMPLETE: Fewer than {self.top_k_bodyid_used} partners in one or both directions")
        elif status == ConnectivityStatus.INCOMPLETE_EXPANSION:
            lines.append(f"  📊 INCOMPLETE EXPANSION: Fewer than {self.top_m_type_target} unique types after expansion")
        
        if self.untyped_upstream_count > 0 or self.untyped_downstream_count > 0:
            lines.append(f"  Untyped excluded: {self.untyped_upstream_count} up, {self.untyped_downstream_count} down")
            lines.append(f"  Untyped weight fraction: {self.untyped_upstream_weight_fraction:.1%} up, {self.untyped_downstream_weight_fraction:.1%} down")
            
            # Round 6: Show 2-hop expansion info for untyped partners
            if self.untyped_upstream_bodyids:
                lines.append(f"  Untyped upstream bodyIds (for 2-hop): {list(self.untyped_upstream_bodyids.keys())[:5]}")
            if self.untyped_downstream_bodyids:
                lines.append(f"  Untyped downstream bodyIds (for 2-hop): {list(self.untyped_downstream_bodyids.keys())[:5]}")
            if self.untyped_upstream_2hop:
                n_with_2hop = len([b for b, t in self.untyped_upstream_2hop.items() if t])
                lines.append(f"  Upstream untyped with 2-hop partners: {n_with_2hop}/{len(self.untyped_upstream_bodyids or {})}")
            if self.untyped_downstream_2hop:
                n_with_2hop = len([b for b, t in self.untyped_downstream_2hop.items() if t])
                lines.append(f"  Downstream untyped with 2-hop partners: {n_with_2hop}/{len(self.untyped_downstream_bodyids or {})}")
        
        lines.append(f"  Top upstream: {list(self.upstream_partners.keys())[:5]}")
        lines.append(f"  Top downstream: {list(self.downstream_partners.keys())[:5]}")
        
        # Round 5: Show expansion info
        lines.append(f"  Dynamic expansion: K={self.top_k_bodyid_used} bodyIds, M={self.top_m_type_target} target types")
        
        return "\n".join(lines)
    
    # ========================================================================
    # Vectorization Methods (Round 2)
    # ========================================================================
    
    def to_weight_vector(
        self,
        vocabulary: List[str],
        direction: str = 'both',
        normalize: bool = True
    ) -> np.ndarray:
        """
        Convert connectivity profile to a normalized weight vector.
        
        Args:
            vocabulary: Ordered list of partner types (shared across datasets)
            direction: 'upstream', 'downstream', or 'both'
            normalize: If True, L2-normalize the vector
        
        Returns:
            numpy array of weights, where position i corresponds to vocabulary[i]
            Missing partners have weight 0.
        
        Example:
            >>> vocab = ['L1', 'L2', 'L3', 'Mi1', 'Tm1']
            >>> profile.to_weight_vector(vocab, 'upstream')
            array([0.32, 0.25, 0.18, 0.0, 0.0])
        """
        if direction == 'upstream':
            partners = self.upstream_partners
        elif direction == 'downstream':
            partners = self.downstream_partners
        else:  # both - sum upstream and downstream weights for each partner
            # Sum weights from both directions (consistent with ProfileComparator)
            up_vec = self.to_weight_vector(vocabulary, 'upstream', normalize=False)
            down_vec = self.to_weight_vector(vocabulary, 'downstream', normalize=False)
            combined = up_vec + down_vec
            if normalize and np.linalg.norm(combined) > 0:
                combined = combined / np.linalg.norm(combined)
            return combined
        
        # Build vector for single direction
        vector = np.array([partners.get(p, 0.0) for p in vocabulary])
        
        if normalize and np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)
        
        return vector
    
    def to_rank_vector(
        self,
        vocabulary: List[str],
        direction: str = 'both',
        default_rank: Optional[int] = None
    ) -> np.ndarray:
        """
        Convert connectivity profile to a rank vector.
        
        Args:
            vocabulary: Ordered list of partner types (shared across datasets)
            direction: 'upstream', 'downstream', or 'both'
            default_rank: Rank to assign for missing partners.
                         If None, uses len(profile_partners) + 1
        
        Returns:
            numpy array of ranks, where position i corresponds to vocabulary[i]
            Missing partners get default_rank (low priority).
        
        Example:
            >>> vocab = ['L1', 'L2', 'L3', 'Mi1', 'Tm1']
            >>> profile.to_rank_vector(vocab, 'upstream')
            array([1, 2, 3, 6, 6])  # L1 is rank 1, Mi1/Tm1 missing (default 6)
        """
        if direction == 'upstream':
            ranks = self.upstream_ranks
            k = len(self.upstream_partners)
        elif direction == 'downstream':
            ranks = self.downstream_ranks
            k = len(self.downstream_partners)
        else:  # both - average upstream and downstream ranks for each partner
            up_vec = self.to_rank_vector(vocabulary, 'upstream', default_rank)
            down_vec = self.to_rank_vector(vocabulary, 'downstream', default_rank)
            # Average ranks from both directions
            return (up_vec + down_vec) / 2.0
        
        # Default rank is K+1 (one past the last partner)
        actual_default = default_rank if default_rank is not None else (k + 1)
        
        # Build vector
        vector = np.array([ranks.get(p, actual_default) for p in vocabulary])
        
        return vector
    
    def to_hybrid_rank_vector(
        self,
        typed_vocabulary: List[str],
        bodyid_vocabulary: Optional[List[int]] = None,
        direction: str = 'upstream',
        mode: str = 'cross_dataset',
        default_rank: Optional[int] = None
    ) -> np.ndarray:
        """
        Round 6: Convert connectivity profile to a hybrid rank vector.
        
        For cross-dataset comparison, this creates a vector where:
        - Typed 1-hop partners use their type-based ranks directly
        - Untyped 1-hop partners are expanded to their 2-hop typed partners
        
        For intra-dataset comparison, untyped partners can be compared by bodyId.
        
        Args:
            typed_vocabulary: Ordered list of typed partner types (for cross-dataset)
            bodyid_vocabulary: Ordered list of bodyIds (for intra-dataset, optional)
            direction: 'upstream' or 'downstream'
            mode: 'cross_dataset' (use typed + 2-hop) or 'intra_dataset' (use typed + bodyId)
            default_rank: Rank for missing partners. If None, uses K+1.
        
        Returns:
            numpy array of ranks
        
        For cross_dataset mode:
            - Vocabulary should include both typed 1-hop types AND 2-hop types
            - Untyped 1-hop partners contribute their 2-hop typed partner ranks
        
        For intra_dataset mode:
            - Two vectors: typed_vector (for typed partners) and bodyid_vector (for untyped)
        """
        if direction == 'upstream':
            typed_ranks = self.upstream_ranks
            typed_partners = self.upstream_partners
            untyped_bodyids = self.untyped_upstream_bodyids
            hop2_ranks = self.untyped_upstream_2hop_ranks
        else:
            typed_ranks = self.downstream_ranks
            typed_partners = self.downstream_partners
            untyped_bodyids = self.untyped_downstream_bodyids
            hop2_ranks = self.untyped_downstream_2hop_ranks
        
        k = len(typed_partners) + len(untyped_bodyids or {})
        actual_default = default_rank if default_rank is not None else (k + 1)
        
        if mode == 'cross_dataset':
            # Build vector that includes typed 1-hop + aggregated 2-hop types
            # For each vocabulary entry, check if it's in typed_ranks OR in 2-hop ranks
            
            # Start with typed partner ranks
            combined_ranks = dict(typed_ranks)
            
            # Add 2-hop ranks (for untyped 1-hop partners)
            # If a type appears in multiple untyped partners' 2-hop, use the best (lowest) rank
            if hop2_ranks:
                for untyped_bid, type_ranks in hop2_ranks.items():
                    for ptype, rank in type_ranks.items():
                        # Adjust rank: 2-hop partners should rank after typed 1-hop
                        # Add offset based on untyped bodyId's original rank
                        adjusted_rank = rank + len(typed_partners)
                        if ptype not in combined_ranks or combined_ranks[ptype] > adjusted_rank:
                            combined_ranks[ptype] = adjusted_rank
            
            vector = np.array([combined_ranks.get(p, actual_default) for p in typed_vocabulary])
            return vector
        
        else:  # intra_dataset
            # For intra-dataset, we want separate vectors for typed and untyped
            # Return typed vector; bodyid vector would be separate
            vector = np.array([typed_ranks.get(p, actual_default) for p in typed_vocabulary])
            return vector
    
    def get_untyped_bodyid_ranks(self, direction: str = 'upstream') -> Dict[int, int]:
        """
        Round 6: Get ranks for untyped bodyIds (for intra-dataset comparison).
        
        Returns:
            Dict mapping bodyId → rank based on weight
        """
        if direction == 'upstream':
            bodyids = self.untyped_upstream_bodyids
        else:
            bodyids = self.untyped_downstream_bodyids
        
        if not bodyids:
            return {}
        
        # Rank by weight (highest weight = rank 1)
        sorted_bids = sorted(bodyids.items(), key=lambda x: x[1], reverse=True)
        return {bid: rank + 1 for rank, (bid, _) in enumerate(sorted_bids)}
    
    @staticmethod
    def build_shared_vocabulary(
        profiles: Dict[str, 'ConnectivityProfile'],
        direction: str = 'both',
        min_occurrence: int = 1
    ) -> List[str]:
        """
        Build a shared vocabulary of partner types across multiple profiles.
        
        Args:
            profiles: Dict of {profile_id: ConnectivityProfile}
            direction: 'upstream', 'downstream', or 'both'
            min_occurrence: Minimum number of profiles a partner must appear in
        
        Returns:
            Sorted list of partner types appearing in at least min_occurrence profiles
        
        Example:
            >>> profiles = {'hb_Mi1': profile1, 'fafb_Mi1': profile2}
            >>> vocab = ConnectivityProfile.build_shared_vocabulary(profiles, 'upstream')
            ['L1', 'L2', 'L3', 'L4', 'L5', 'Tm1', 'Tm2']
        """
        partner_counts: Dict[str, int] = {}
        
        for profile in profiles.values():
            if direction in ('upstream', 'both'):
                for partner in profile.upstream_partners.keys():
                    partner_counts[partner] = partner_counts.get(partner, 0) + 1
            if direction in ('downstream', 'both'):
                for partner in profile.downstream_partners.keys():
                    partner_counts[partner] = partner_counts.get(partner, 0) + 1
        
        # Filter by minimum occurrence and sort
        vocabulary = [p for p, count in partner_counts.items() if count >= min_occurrence]
        return sorted(vocabulary)
    
    @staticmethod
    def build_hybrid_vocabulary(
        profiles: Dict[str, 'ConnectivityProfile'],
        direction: str = 'upstream',
        min_occurrence: int = 1,
        include_2hop: bool = True
    ) -> List[str]:
        """
        Round 6: Build vocabulary including both typed 1-hop and 2-hop types.
        
        For cross-dataset comparison with untyped 1-hop partners expanded.
        
        Args:
            profiles: Dict of {profile_id: ConnectivityProfile}
            direction: 'upstream' or 'downstream'
            min_occurrence: Minimum profiles a type must appear in
            include_2hop: Whether to include 2-hop types from untyped partners
        
        Returns:
            Sorted list of all partner types (1-hop typed + 2-hop from untyped)
        """
        partner_counts: Dict[str, int] = {}
        
        for profile in profiles.values():
            # Add typed 1-hop partners
            if direction == 'upstream':
                for partner in profile.upstream_partners.keys():
                    partner_counts[partner] = partner_counts.get(partner, 0) + 1
                
                # Add 2-hop types from untyped partners
                if include_2hop and profile.untyped_upstream_2hop:
                    for untyped_bid, type_weights in profile.untyped_upstream_2hop.items():
                        for ptype in type_weights.keys():
                            partner_counts[ptype] = partner_counts.get(ptype, 0) + 1
            else:
                for partner in profile.downstream_partners.keys():
                    partner_counts[partner] = partner_counts.get(partner, 0) + 1
                
                if include_2hop and profile.untyped_downstream_2hop:
                    for untyped_bid, type_weights in profile.untyped_downstream_2hop.items():
                        for ptype in type_weights.keys():
                            partner_counts[ptype] = partner_counts.get(ptype, 0) + 1
        
        vocabulary = [p for p, count in partner_counts.items() if count >= min_occurrence]
        return sorted(vocabulary)
    
    @staticmethod
    def profiles_to_weight_matrix(
        profiles: Dict[str, 'ConnectivityProfile'],
        vocabulary: Optional[List[str]] = None,
        direction: str = 'both',
        normalize: bool = True
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Convert multiple profiles to a weight matrix.
        
        Args:
            profiles: Dict of {profile_id: ConnectivityProfile}
            vocabulary: Optional shared vocabulary. If None, built from profiles.
            direction: 'upstream', 'downstream', or 'both'
            normalize: L2-normalize each row
        
        Returns:
            Tuple of:
            - matrix: (n_profiles, n_vocabulary) numpy array
            - profile_ids: List of profile IDs (row order)
            - vocabulary: List of partner types (column order)
        
        Example:
            >>> matrix, ids, vocab = ConnectivityProfile.profiles_to_weight_matrix(profiles)
            >>> matrix.shape
            (5, 20)  # 5 profiles, 20 unique partner types
        """
        if vocabulary is None:
            vocabulary = ConnectivityProfile.build_shared_vocabulary(profiles, direction)
        
        profile_ids = list(profiles.keys())
        matrix = np.zeros((len(profile_ids), len(vocabulary)))
        
        for i, pid in enumerate(profile_ids):
            matrix[i] = profiles[pid].to_weight_vector(vocabulary, direction, normalize)
        
        return matrix, profile_ids, vocabulary
    
    @staticmethod
    def profiles_to_rank_matrix(
        profiles: Dict[str, 'ConnectivityProfile'],
        vocabulary: Optional[List[str]] = None,
        direction: str = 'both',
        default_rank: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Convert multiple profiles to a rank matrix.
        
        Args:
            profiles: Dict of {profile_id: ConnectivityProfile}
            vocabulary: Optional shared vocabulary. If None, built from profiles.
            direction: 'upstream', 'downstream', or 'both'
            default_rank: Rank for missing partners. If None, uses profile's K+1.
        
        Returns:
            Tuple of:
            - matrix: (n_profiles, n_vocabulary) numpy array of ranks
            - profile_ids: List of profile IDs (row order)
            - vocabulary: List of partner types (column order)
        
        Example:
            >>> matrix, ids, vocab = ConnectivityProfile.profiles_to_rank_matrix(profiles)
            >>> # Compute pairwise Spearman correlations efficiently
            >>> from scipy.stats import spearmanr
            >>> corr_matrix, _ = spearmanr(matrix, axis=1)
        """
        if vocabulary is None:
            vocabulary = ConnectivityProfile.build_shared_vocabulary(profiles, direction)
        
        profile_ids = list(profiles.keys())
        matrix = np.zeros((len(profile_ids), len(vocabulary)))
        
        for i, pid in enumerate(profile_ids):
            matrix[i] = profiles[pid].to_rank_vector(vocabulary, direction, default_rank)
        
        return matrix, profile_ids, vocabulary
    
    @staticmethod
    def aggregate_bodyid_profiles(
        profiles: List['ConnectivityProfile'],
        neuron_type: str,
        dataset: str
    ) -> 'ConnectivityProfile':
        """
        Aggregate multiple bodyId-level profiles into a single type-level profile.
        
        Since profiles store actual synapse weights (not proportions), aggregation
        is simply summing the weights across all bodyIds. This preserves the rank
        structure naturally - partners with more total synapses get lower ranks.
        
        Args:
            profiles: List of bodyId-level ConnectivityProfile objects
            neuron_type: Name for the aggregated type
            dataset: Dataset identifier
        
        Returns:
            Aggregated ConnectivityProfile representing the entire type
        
        Example:
            >>> # Get profiles for all neurons of type "aMe12"
            >>> bodyid_profiles = [profiler.get_profile(bid, dataset) for bid in body_ids]
            >>> # Aggregate into single type profile
            >>> type_profile = ConnectivityProfile.aggregate_bodyid_profiles(
            ...     bodyid_profiles, 'aMe12', dataset
            ... )
        """
        if not profiles:
            raise ValueError("Cannot aggregate empty profile list")
        
        # Sum weights across all profiles
        upstream_weights: Dict[str, float] = {}
        downstream_weights: Dict[str, float] = {}
        
        total_upstream_weight = 0.0
        total_downstream_weight = 0.0
        untyped_upstream_count = 0
        untyped_downstream_count = 0
        untyped_up_weight = 0.0
        untyped_down_weight = 0.0
        
        for profile in profiles:
            # Sum upstream weights
            for partner, weight in profile.upstream_partners.items():
                upstream_weights[partner] = upstream_weights.get(partner, 0.0) + weight
            
            # Sum downstream weights
            for partner, weight in profile.downstream_partners.items():
                downstream_weights[partner] = downstream_weights.get(partner, 0.0) + weight
            
            # Accumulate totals
            total_upstream_weight += profile.total_upstream_weight
            total_downstream_weight += profile.total_downstream_weight
            untyped_upstream_count += profile.untyped_upstream_count
            untyped_downstream_count += profile.untyped_downstream_count
            untyped_up_weight += profile.untyped_upstream_weight_fraction * profile.total_upstream_weight
            untyped_down_weight += profile.untyped_downstream_weight_fraction * profile.total_downstream_weight
        
        # Compute ranks from aggregated weights
        upstream_ranks = compute_ranks(upstream_weights)
        downstream_ranks = compute_ranks(downstream_weights)
        
        # Compute untyped fractions
        untyped_up_frac = untyped_up_weight / total_upstream_weight if total_upstream_weight > 0 else 0.0
        untyped_down_frac = untyped_down_weight / total_downstream_weight if total_downstream_weight > 0 else 0.0
        
        return ConnectivityProfile(
            neuron_id=neuron_type,
            dataset=dataset,
            neuron_type=neuron_type,
            upstream_partners=upstream_weights,
            downstream_partners=downstream_weights,
            upstream_ranks=upstream_ranks,
            downstream_ranks=downstream_ranks,
            upstream_top_k=profiles[0].upstream_top_k,
            downstream_top_k=profiles[0].downstream_top_k,
            # Provenance must reflect the MEMBERS, not the dataclass
            # defaults: an aggregate claiming top_k_bodyid_used=20 built
            # from k=5 members would pass the tier-1 cache gate and serve
            # silently truncated partner sets.
            top_k_bodyid_used=max(
                (p.top_k_bodyid_used for p in profiles), default=0),
            top_m_type_target=max(
                (p.top_m_type_target for p in profiles), default=0),
            total_upstream_weight=total_upstream_weight,
            total_downstream_weight=total_downstream_weight,
            num_neurons_aggregated=len(profiles),
            untyped_upstream_count=untyped_upstream_count,
            untyped_downstream_count=untyped_downstream_count,
            untyped_upstream_weight_fraction=untyped_up_frac,
            untyped_downstream_weight_fraction=untyped_down_frac,
            actual_upstream_count=len(upstream_weights),
            actual_downstream_count=len(downstream_weights),
            unique_types_upstream=len(upstream_weights),
            unique_types_downstream=len(downstream_weights),
        )


# ============================================================================
# Helper Functions
# ============================================================================

def normalize_partner_type(type_name: str, config: FuzzyMatchConfig) -> str:
    """
    Normalize partner type name for cross-dataset comparison.
    
    Applies fuzzy matching rules to handle naming variations:
    - T4a_R → t4a (strip laterality suffix + lowercase)
    - aMe12_1 → ame12 (strip numeric suffix + lowercase)
    - T4a-like → t4a (strip like suffix)
    
    Args:
        type_name: Original type name
        config: FuzzyMatchConfig with matching rules
    
    Returns:
        Normalized type name
    """
    if not config.enabled or pd.isna(type_name) or type_name is None:
        # NaN is truthy, so it must be caught before the str() fallback,
        # otherwise it would normalize to the literal string 'nan'.
        if type_name is None or (isinstance(type_name, float) and pd.isna(type_name)):
            return ''
        try:
            is_missing = pd.isna(type_name)
        except (TypeError, ValueError):
            is_missing = False
        return '' if is_missing else str(type_name)
    
    result = str(type_name)
    
    # Apply custom mappings first (exact match on original)
    if result in config.custom_mappings:
        result = config.custom_mappings[result]
    
    # Case normalization
    if config.case_insensitive:
        result = result.lower()
    
    # Strip laterality suffixes: _L, _R, _left, _right, -L, -R
    if config.strip_lr_suffix:
        result = re.sub(r'[_-]?([LR]|left|right)$', '', result, flags=re.IGNORECASE)
    
    # Strip numeric suffixes: _1, _2, -1, -2, etc.
    if config.strip_numeric_suffix:
        result = re.sub(r'[_-]?\d+$', '', result)
    
    # Strip "-like" or "_like" suffixes
    if config.strip_like_suffix:
        result = re.sub(r'[_-]?like$', '', result, flags=re.IGNORECASE)
    
    # Apply custom mappings again after normalization (for mapped normalized forms)
    if result in config.custom_mappings:
        result = config.custom_mappings[result]
    
    return result


def compute_ranks(weights: Dict[str, float]) -> Dict[str, int]:
    """
    Compute ranks from weight dictionary.
    
    Args:
        weights: Partner type → weight mapping
    
    Returns:
        Partner type → rank mapping (1 = strongest)
    """
    if not weights:
        return {}
    
    # Sort by weight descending
    sorted_partners = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    # Assign ranks (handle ties by giving same rank)
    ranks = {}
    current_rank = 1
    prev_weight = None
    
    for i, (partner, weight) in enumerate(sorted_partners):
        if prev_weight is not None and weight < prev_weight:
            current_rank = i + 1
        ranks[partner] = current_rank
        prev_weight = weight
    
    return ranks


# ============================================================================
# ConnectivityProfiler Class
# ============================================================================

class ConnectivityProfiler:
    """
    Extract and manage connectivity profiles across datasets.
    
    This class queries neuron connectivity from NeuPrint or local datasets,
    normalizes the data, and produces ConnectivityProfile objects for
    cross-dataset comparison.
    
    Features:
    - Support for neuron types (str), bodyIds (int), or lists of bodyIds
    - Multi-neuron type aggregation with variance computation
    - Configurable filtering (min synapses, exclude untyped)
    - Profile caching to disk
    - Fuzzy partner type matching
    
    Example:
        >>> profiler = ConnectivityProfiler(
        ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
        ...     config=ProfilerConfig(top_k_bodyid=10)
        ... )
        >>> profile = profiler.get_profile('aMe12', 'hemibrain:v1.2.1')
        >>> print(profile.upstream_partners)
    """
    
    def __init__(
        self,
        datasets: List[str],
        client: Optional[Any] = None,
        config: Optional[ProfilerConfig] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize ConnectivityProfiler.
        
        Args:
            datasets: List of dataset identifiers to work with
            client: Optional NeuPrint client (will be created if needed)
            config: ProfilerConfig with extraction settings
            cache_dir: Custom cache directory (default: cache/connectivity_profiles)
            token: NeuPrint API token (used when creating clients)
            verbose: Print progress messages
        """
        self.datasets = datasets
        self.client = client
        self.config = config or ProfilerConfig()
        self.token = token
        self.verbose = verbose
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Use project-level cache folder
            src_dir = Path(__file__).parent.parent
            project_root = src_dir.parent
            self.cache_dir = project_root / 'cache'
        
        if self.config.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for profiles (neuron_id, dataset) -> ConnectivityProfile
        self._memory_cache: Dict[Tuple[str, str], ConnectivityProfile] = {}
        
        # In-memory cache for disk parquet DataFrames: dataset -> DataFrame
        # Avoids repeated disk reads when looking up multiple profiles
        self._disk_cache_df: Dict[str, pd.DataFrame] = {}
        
        # Index for O(1) lookups: dataset -> {neuron_id -> row_index}
        self._disk_cache_index: Dict[str, Dict[str, int]] = {}
        
        # Client cache per dataset
        self._clients: Dict[str, Any] = {}
        # Why a client could NOT be created, per dataset. `ensure_data_available`
        # used to answer "check your credentials" for every failure, including
        # the one that is not a credential problem at all — the server simply
        # does not host that dataset name (measured 2026-09-26, when the live
        # catalog stopped answering `male-cns:v1.0` and lists `manc:*` in its
        # place: a fully local Find Homologs run died at save time and was
        # told to check its token).
        self._client_errors: Dict[str, str] = {}
        
        # Data availability cache: dataset -> True (avoids repeated checks)
        self._data_availability_cache: Dict[str, bool] = {}
        
        # Type normalization cache: dataset -> {original_type -> normalized_type}
        # Pre-computed at connection cache load time for vectorized lookups
        self._type_normalization_cache: Dict[str, Dict[str, str]] = {}
        # Lazily-loaded per-dataset bodyId(str) -> type maps, used to backfill
        # neuron_type on bodyId-queried profiles (the profile builders leave
        # it None). API misses memoize into the same map.
        self._bodyid_type_maps: Dict[str, Dict[str, str]] = {}
        
        # Type -> row-position index for type-based connection queries
        # dataset_safe -> {'conn_id': int, 'pre': {type -> [rows]}, 'post': {...}}
        self._type_row_index: Dict[str, Dict[str, Any]] = {}
        
        # Threading lock for cache writes (prevents corruption during parallel processing)
        import threading
        self._cache_write_lock = threading.Lock()
        
        # Flag to defer cache writes during parallel processing
        self._defer_cache_writes = False
        self._pending_cache_writes: Dict[str, Dict[str, ConnectivityProfile]] = {}
        self._in_progress_bar = False  # Flag to use tqdm.write instead of print
    
    def _log(self, message: str, level: str = 'info'):
        """Print message if verbose mode enabled.
        
        Args:
            message: Message to print
            level: 'info', 'debug', or 'warning'. Debug messages are suppressed.
        """
        if not self.verbose:
            return
        # Only print warnings and key info messages, suppress debug-level messages
        if level == 'debug':
            return  # Suppress verbose loading/joining messages
        
        msg = f"[ConnectivityProfiler] {message}"
        # Use tqdm.write when inside a progress bar to avoid interrupting it
        if getattr(self, '_in_progress_bar', False):
            from tqdm import tqdm
            tqdm.write(msg)
        else:
            print(msg)
    
    def _normalize_types_vectorized(
        self, 
        type_series: pd.Series, 
        config: 'FuzzyMatchConfig'
    ) -> pd.Series:
        """
        Vectorized type normalization using Polars for performance.
        
        Instead of calling normalize_partner_type() per row via .apply(),
        this method:
        1. Gets unique types from the series (much smaller than total rows)
        2. Normalizes only unique types once
        3. Maps back to full series using vectorized replace/map
        
        Performance: ~10-50x faster than .apply() for large DataFrames.
        
        Args:
            type_series: Pandas Series of partner type names
            config: FuzzyMatchConfig with matching rules
        
        Returns:
            Pandas Series of normalized type names
        """
        if type_series.empty:
            return type_series.astype(str)
        
        try:
            import polars as pl
            
            # Get unique types - much smaller than full series
            unique_types = type_series.dropna().unique()
            
            if len(unique_types) == 0:
                return type_series.fillna('').astype(str)
            
            # Build normalization map for unique types only
            # This is O(unique_types) instead of O(total_rows)
            norm_map = {}
            for t in unique_types:
                norm_map[t] = normalize_partner_type(t, config)
            
            # Apply vectorized mapping using Polars (fastest)
            pl_series = pl.Series(type_series.fillna(''))
            
            # Use Polars replace_strict for vectorized mapping
            # (Series.replace(default=) is deprecated since polars 1.0).
            normalized = pl_series.replace_strict(norm_map, default=pl_series).to_pandas()
            
            # CRITICAL: the polars round-trip resets the index to RangeIndex.
            # Callers assign the result back into frames with non-contiguous
            # indexes (e.g. conn_df.iloc[...]); without realigning, pandas
            # aligns by label and silently scrambles the values.
            normalized.index = type_series.index
            
            return normalized
            
        except Exception:
            # Fallback: Use pandas map with pre-computed normalization map
            # (broadened from ImportError: pl.Series() also raises conversion
            # errors on non-string inputs, which must fall back, not crash)
            unique_types = type_series.dropna().unique()
            norm_map = {t: normalize_partner_type(t, config) for t in unique_types}
            
            # Map with fallback for NaN/missing
            return type_series.map(lambda x: norm_map.get(x, str(x) if pd.notna(x) else ''))
    
    def _get_config_hash(self) -> str:
        """Get hash of current configuration for cache invalidation."""
        config_dict = {
            'top_k_bodyid': self.config.top_k_bodyid,
            'min_synapse_threshold': self.config.min_synapse_threshold,
            'include_untyped_partners': self.config.include_untyped_partners,
            'normalize_method': self.config.normalize_method,
            'fuzzy_enabled': self.config.fuzzy_match.enabled,
            'fuzzy_strip_lr': self.config.fuzzy_match.strip_lr_suffix,
            'fuzzy_strip_numeric': self.config.fuzzy_match.strip_numeric_suffix,
        }
        return hashlib.md5(
            json.dumps(config_dict, sort_keys=True).encode()
        ).hexdigest()[:8]
    
    def _get_cache_parquet_path(self, dataset: str) -> Path:
        """Get path to parquet cache file for a dataset."""
        safe_dataset = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
        return self.cache_dir / safe_dataset / 'connectivity_profiles.parquet'
    
    def _get_profile_batch_dir(self, dataset: str) -> Path:
        """Get path to batch directory for per-profile files (interruption-safe)."""
        safe_dataset = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
        return self.cache_dir / safe_dataset / '_profile_batch_files'

    def _quarantine_unreadable_cache(self, main_cache_path: Path,
                                     error: Exception) -> Optional[Path]:
        """Rename an unreadable main cache aside and announce it loudly.

        The unreadable file may hold consolidated rows that exist in no
        batch file, so it is preserved (never overwritten or deleted) for
        manual recovery.  Returns the quarantine path, or ``None`` when the
        rename itself failed (the original stays in place either way).
        """
        stamp = time.strftime('%Y%m%d_%H%M%S')
        quarantined = main_cache_path.with_name(
            f"{main_cache_path.stem}.corrupt-{stamp}"
            f"{main_cache_path.suffix}")
        suffix_n = 1
        while quarantined.exists():
            # Two corruptions inside one wall-clock second must not overwrite
            # the earlier recovery copy (POSIX rename would clobber silently).
            quarantined = main_cache_path.with_name(
                f"{main_cache_path.stem}.corrupt-{stamp}-{suffix_n}"
                f"{main_cache_path.suffix}")
            suffix_n += 1
        try:
            main_cache_path.rename(quarantined)
        except OSError as rename_error:
            self._log(
                f"ERROR: main profile cache {main_cache_path} is unreadable "
                f"({error}) and could not be quarantined ({rename_error}); "
                "leaving it in place - consolidation aborted.")
            return None
        self._log(
            f"ERROR: main profile cache {main_cache_path} was unreadable "
            f"({error}); moved to {quarantined} for recovery. "
            "Consolidation aborted - batch files were kept.")
        return quarantined

    def _save_profile_to_batch_file(self, profile: ConnectivityProfile):
        """
        Save a single profile to its own batch file (interruption-safe).
        
        Similar to connection cache's per-batch saving, this approach:
        - Writes each profile to a separate small file immediately
        - Never loads existing data, just appends new file
        - Survives interruptions (Ctrl+C, OOM, crashes)
        - Files are consolidated later via _consolidate_profile_batch_files()
        
        File naming: {batch_dir}/{neuron_id}.parquet
        """
        batch_dir = self._get_profile_batch_dir(profile.dataset)
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize neuron_id for filename (handle special characters)
        safe_neuron_id = str(profile.neuron_id).replace('/', '_').replace('\\', '_').replace(':', '_')
        batch_file = batch_dir / f"{safe_neuron_id}.parquet"
        
        # Convert profile to row
        row_data = self._profile_to_row(profile)
        df = pd.DataFrame([row_data])
        
        # Atomic write with a reclaimable temp-sibling name (R7-2)
        try:
            from utils.parquet_utils import replace_with_retry, temp_sibling
        except ImportError:
            from .utils.parquet_utils import replace_with_retry, temp_sibling
        temp_file = Path(temp_sibling(str(batch_file), 'profile-batch'))
        try:
            df.to_parquet(temp_file, index=False)
            replace_with_retry(temp_file, batch_file)
        except Exception as e:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            self._log(f"Warning: Could not save profile batch file: {e}")

    def _consolidate_profile_batch_files(self, dataset: str, delete_after: bool = True) -> int:
        """
        Merge all profile batch files into the main connectivity_profiles.parquet.
        
        Uses Polars for memory-efficient consolidation of potentially large caches.
        
        Args:
            dataset: Dataset identifier
            delete_after: If True, delete batch files after successful consolidation
            
        Returns:
            Number of profiles consolidated
        """
        batch_dir = self._get_profile_batch_dir(dataset)
        if not batch_dir.exists():
            return 0
        
        # Find all batch files
        batch_files = sorted(batch_dir.glob('*.parquet'))
        batch_files = [f for f in batch_files if not f.name.endswith('.tmp')]
        
        if not batch_files:
            return 0
        
        self._log(f"Consolidating {len(batch_files)} profile batch files for {dataset}...")
        # Shared by both engine paths below (R7-2 temp-sibling naming);
        # hoisted so the pandas fallback sees the binding too.
        try:
            from utils.parquet_utils import (
                remove_stale_temp_files, remove_stale_temp_files_in_dir,
                replace_with_retry, temp_sibling)
        except ImportError:
            from .utils.parquet_utils import (  # pragma: no cover
                remove_stale_temp_files, remove_stale_temp_files_in_dir,
                replace_with_retry, temp_sibling)
        
        try:
            import polars as pl
            
            # Load existing main cache if exists
            main_cache_path = self._get_cache_parquet_path(dataset)
            all_dfs = []
            
            if main_cache_path.exists():
                try:
                    existing_df = pl.read_parquet(str(main_cache_path))
                    all_dfs.append(existing_df)
                except Exception as e:
                    # The unreadable main cache may hold rows that are in
                    # no batch file: merging batch-only rows and deleting
                    # the batches would silently destroy them.  Quarantine
                    # for recovery and abort without touching the batches.
                    self._quarantine_unreadable_cache(main_cache_path, e)
                    return 0
            
            # Load all batch files
            for bf in batch_files:
                try:
                    df = pl.read_parquet(str(bf))
                    all_dfs.append(df)
                except Exception as e:
                    self._log(f"Warning: Skipping corrupt batch file {bf.name}: {e}")
            
            if not all_dfs:
                return 0
            
            # Normalize schema before concatenation to handle type mismatches
            # neuron_id should be Int64 (bodyIds only - type profiles are NOT cached)
            # Numeric columns should be Float64 for consistency
            
            normalized_dfs = []
            for df in all_dfs:
                # Cast columns for schema compatibility
                cast_exprs = []
                for col in df.columns:
                    if col == 'neuron_id':
                        # neuron_id is always Int64 (bodyId) - type profiles are not cached
                        cast_exprs.append(pl.col(col).cast(pl.Int64))
                    elif df[col].dtype in (pl.Int64, pl.Int32, pl.Int16, pl.Int8):
                        # Cast integers to Float64 for consistency (except neuron_id)
                        cast_exprs.append(pl.col(col).cast(pl.Float64))
                    else:
                        cast_exprs.append(pl.col(col))
                
                if cast_exprs:
                    normalized_dfs.append(df.select(cast_exprs))
                else:
                    normalized_dfs.append(df)
            
            # Concatenate with schema alignment
            combined = pl.concat(normalized_dfs, how='diagonal_relaxed')
            # Keep only the last occurrence of each neuron_id (latest profile)
            combined = combined.unique(subset=['neuron_id'], keep='last')
            
            # Save consolidated cache atomically through the shared
            # temp-sibling scheme (round-7 R7-2: the naked
            # '.parquet.tmp' name leaked a non-reclaimable orphan and its
            # os.replace raced an existing temp into WinError 183).
            # The sweep must name THIS writer's kind: the loader's
            # 'profile-cache' sweep below cannot see a crashed
            # consolidation temp, so without it the heaviest writer leaks
            # a permanent orphan — exactly the R7-2 defect.
            main_cache_path.parent.mkdir(parents=True, exist_ok=True)
            remove_stale_temp_files(main_cache_path, 'profile-consolidation')
            temp_path = Path(temp_sibling(str(main_cache_path),
                                          'profile-consolidation'))
            combined.write_parquet(str(temp_path))
            replace_with_retry(temp_path, main_cache_path)
            
            # Update in-memory cache
            self._disk_cache_df[dataset] = combined.to_pandas()
            self._build_disk_cache_index(dataset)
            
            consolidated_count = len(batch_files)
            
            # Delete batch files if requested
            if delete_after:
                for bf in batch_files:
                    try:
                        bf.unlink()
                    except Exception:
                        pass
                # Remove batch directory if empty.  Reclaim crashed batch
                # temps first: an orphaned '.{neuron}.parquet.profile-batch'
                # sibling would otherwise keep the "empty" dir alive forever.
                remove_stale_temp_files_in_dir(batch_dir, 'profile-batch')
                try:
                    batch_dir.rmdir()
                except Exception:
                    pass
            
            self._log(f"Consolidated {consolidated_count} profiles into main cache ({len(combined)} total)")
            return consolidated_count
            
        except ImportError:
            # Fallback to Pandas
            main_cache_path = self._get_cache_parquet_path(dataset)
            all_dfs = []
            
            if main_cache_path.exists():
                try:
                    all_dfs.append(pd.read_parquet(main_cache_path))
                except Exception as e:
                    # Same protection as the polars path above: never
                    # overwrite/delete around an unreadable main cache.
                    self._quarantine_unreadable_cache(main_cache_path, e)
                    return 0

            for bf in batch_files:
                try:
                    all_dfs.append(pd.read_parquet(bf))
                except Exception as e:
                    self._log(f"Warning: Skipping corrupt batch file {bf.name}: {e}")

            if not all_dfs:
                return 0

            combined = pd.concat(all_dfs, ignore_index=True)
            combined = combined.drop_duplicates(subset=['neuron_id'], keep='last')

            main_cache_path.parent.mkdir(parents=True, exist_ok=True)
            remove_stale_temp_files(main_cache_path, 'profile-consolidation')
            temp_path = Path(temp_sibling(str(main_cache_path),
                                          'profile-consolidation'))
            combined.to_parquet(temp_path, index=False)
            replace_with_retry(temp_path, main_cache_path)
            
            self._disk_cache_df[dataset] = combined
            self._build_disk_cache_index(dataset)
            
            if delete_after:
                for bf in batch_files:
                    try:
                        bf.unlink()
                    except Exception:
                        pass
                # Same pre-rmdir temp reclaim as the polars path above.
                remove_stale_temp_files_in_dir(batch_dir, 'profile-batch')
                try:
                    batch_dir.rmdir()
                except Exception:
                    pass

            return len(batch_files)
    
    def _load_cache_dataframe(self, dataset: str, force_reload: bool = False) -> Optional[pd.DataFrame]:
        """
        Load the cache dataframe for a dataset with in-memory caching.
        
        If batch files exist from interrupted runs, consolidates them first
        into the main connectivity_profiles.parquet file.
        
        First load reads from disk and caches in memory.
        Subsequent loads return the cached DataFrame instantly.
        
        Args:
            dataset: Dataset name
            force_reload: If True, reload from disk even if cached
            
        Returns:
            Cached DataFrame or None if not found
        """
        # Check in-memory cache first
        if not force_reload and dataset in self._disk_cache_df:
            return self._disk_cache_df[dataset]
        
        # Load from disk
        cache_path = self._get_cache_parquet_path(dataset)
        
        # Clean up any leftover temp files from interrupted runs: the
        # current temp-sibling scheme, plus the legacy naked name written
        # before the R7-2 migration.
        try:
            from utils.parquet_utils import remove_stale_temp_files
        except ImportError:
            from .utils.parquet_utils import remove_stale_temp_files
        remove_stale_temp_files(cache_path, 'profile-cache')
        temp_path = cache_path.with_suffix('.parquet.tmp')
        if temp_path.exists():
            try:
                temp_path.unlink()
                self._log(f"Cleaned up incomplete temp file: {temp_path.name}")
            except Exception:
                pass
        
        # Check for batch files from interrupted runs and consolidate them
        batch_dir = self._get_profile_batch_dir(dataset)
        if batch_dir.exists():
            batch_files = [f for f in batch_dir.glob('*.parquet') if not f.name.endswith('.tmp')]
            if batch_files:
                self._log(f"Found {len(batch_files)} profile batch files from previous run, consolidating...")
                self._consolidate_profile_batch_files(dataset, delete_after=True)
        
        # Now load the main cache file (which includes any consolidated batch files)
        if not cache_path.exists():
            return None
        
        try:
            # Try Polars for memory-efficient loading of large caches
            try:
                import polars as pl
                df_pl = pl.read_parquet(str(cache_path))
                df = df_pl.to_pandas()
                del df_pl
            except ImportError:
                df = pd.read_parquet(cache_path)
        except Exception as e:
            # Corrupt parquet: quarantine it aside (the same policy the
            # consolidation paths apply) instead of deleting — the file may
            # hold consolidated rows that exist in no batch file, so
            # destroying it on one bad read loses an expensive cache.
            self._log(f"Warning: Corrupt cache parquet for {dataset}: {e}")
            self._quarantine_unreadable_cache(cache_path, e)
            return None
        
        # Cache in memory
        self._disk_cache_df[dataset] = df
        # Build index for O(1) lookups
        self._build_disk_cache_index(dataset)
        return df
    
    def _build_disk_cache_index(self, dataset: str):
        """Build O(1) lookup index for disk cache DataFrame.
        
        Uses vectorized operations instead of iterrows for 10-100x speedup.
        """
        if dataset not in self._disk_cache_df:
            self._disk_cache_index[dataset] = {}
            return
        
        df = self._disk_cache_df[dataset]
        if 'neuron_id' not in df.columns:
            self._disk_cache_index[dataset] = {}
            return
        
        # Build index using vectorized operations (much faster than iterrows)
        # neuron_id -> row_index
        neuron_ids = df['neuron_id'].astype(str).tolist()
        self._disk_cache_index[dataset] = {
            nid: idx for idx, nid in enumerate(neuron_ids)
        }
    
    def _save_cache_dataframe(self, df: pd.DataFrame, dataset: str):
        """
        Save the cache dataframe for a dataset using atomic write.
        Also updates the in-memory cache.
        
        Uses a write-to-temp-then-rename strategy to ensure the cache file
        is never left in a corrupted state if the process is interrupted
        (Ctrl+C, power cut, system crash, etc.).
        
        Safety guarantees:
        - Ctrl+C during write: temp file may be incomplete, original untouched
        - Power cut during write: temp file may be incomplete, original untouched  
        - Ctrl+C/power cut during rename: atomic on POSIX, either old or new exists
        """
        cache_path = self._get_cache_parquet_path(dataset)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use atomic write: write to temp file, fsync, then rename
        # This ensures the cache is never corrupted by interrupts or power loss
        # R7-2: temp-sibling naming so an interrupted write is reclaimable
        # by remove_stale_temp_files, not a permanent naked .tmp orphan.
        try:
            from utils.parquet_utils import (
                remove_stale_temp_files, replace_with_retry, temp_sibling)
        except ImportError:
            from .utils.parquet_utils import (
                remove_stale_temp_files, replace_with_retry, temp_sibling)
        remove_stale_temp_files(cache_path, 'profile-cache')
        temp_path = Path(temp_sibling(str(cache_path), 'profile-cache'))
        try:
            # Try Polars for faster parquet writing (especially for large caches)
            use_polars = False
            try:
                import polars as pl
                if len(df) > 5000:
                    use_polars = True
            except ImportError:
                pass
            
            # Write to temporary file first
            if use_polars:
                df_pl = pl.from_pandas(df)
                df_pl.write_parquet(str(temp_path))
                del df_pl
            else:
                df.to_parquet(temp_path, index=False)
            
            # Ensure data is flushed to disk before rename (protects against power loss)
            # This is critical: without fsync, data may be in OS buffer when rename happens
            try:
                with open(temp_path, 'r+b') as f:
                    os.fsync(f.fileno())
            except Exception:
                pass  # fsync failure is non-fatal, rename will still work for Ctrl+C
            
            # Atomic swap (atomic_replace keeps the overwrite semantics on
            # Windows too, where Path.rename over an existing file raises;
            # the retry loop rides out a destination briefly held open)
            replace_with_retry(temp_path, cache_path)
            
            # Update in-memory cache
            self._disk_cache_df[dataset] = df
            self._build_disk_cache_index(dataset)
        except Exception as e:
            # Clean up temp file if it exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            self._log(f"Warning: Could not save cache parquet: {e}")
    
    def _load_from_cache(
        self, 
        neuron_id: Union[str, int], 
        dataset: str,
        required_top_k: Optional[int] = None
    ) -> Optional[ConnectivityProfile]:
        """
        Load profile from cache if exists and meets requirements.
        
        Uses a 3-tier cache strategy:
        1. Memory cache (instant): Check _memory_cache dict first
        2. Disk cache index (O(1)): Use _disk_cache_index for fast row lookup
        3. Disk cache load: Only loads parquet once, cached in _disk_cache_df
        
        Option A Cache Strategy:
        - If cached_k >= requested_k: Use cache (slice if needed)
        - If cached_k < requested_k: Return None (need to re-fetch with higher k)
        
        Args:
            neuron_id: Neuron identifier
            dataset: Dataset name
            required_top_k: Minimum top_k required. If None, uses config.top_k_bodyid
        
        Returns:
            Cached profile if available and sufficient, None otherwise
        """
        required_k = required_top_k or self.config.top_k_bodyid
        neuron_id_str = str(neuron_id)
        current_hash = self._get_config_hash()
        
        # Tier 1: Check memory cache first (instant O(1))
        cache_key = (neuron_id_str, dataset)
        if cache_key in self._memory_cache:
            cached = self._memory_cache[cache_key]
            if getattr(cached, '_config_hash', current_hash) != current_hash:
                # Built under a different config (min synapse threshold,
                # untyped/fuzzy toggles) - stale for this reader.
                return None
            # Check if cached profile has sufficient top_k
            if cached.top_k_bodyid_used >= required_k:
                return cached
            # Cached profile has lower k than required - need re-fetch
            return None
        
        # Tier 2 & 3: Check disk cache with in-memory DataFrame
        if not self.config.use_cache:
            return None
        
        # This loads from disk only once, then uses cached DataFrame
        cache_df = self._load_cache_dataframe(dataset)
        if cache_df is None or 'neuron_id' not in cache_df.columns:
            return None
        
        # Use index for O(1) row lookup instead of DataFrame filter
        if dataset in self._disk_cache_index:
            row_idx = self._disk_cache_index[dataset].get(neuron_id_str)
            if row_idx is not None:
                try:
                    # Direct row access by index (O(1))
                    row_data = cache_df.iloc[row_idx]
                    row_hash = row_data.get('config_hash')
                    if row_hash is not None and pd.notna(row_hash) \
                            and str(row_hash) != current_hash:
                        # Persisted under a different config - rebuild
                        # rather than reuse (rows without the column are
                        # legacy and stay accepted).
                        return None
                    profile = self._row_to_profile(row_data)
                    profile._config_hash = (str(row_hash) if row_hash is not None
                                            and pd.notna(row_hash)
                                            else current_hash)
                    
                    # Check if cached profile has sufficient top_k
                    if profile.top_k_bodyid_used >= required_k:
                        self._memory_cache[cache_key] = profile
                        return profile
                    
                    # Cached profile has lower k - return None to trigger re-fetch
                    return None
                except Exception as e:
                    self._log(f"Warning: Could not load cache for {neuron_id}: {e}")
        
        return None
    
    def _row_to_profile(self, row: pd.Series) -> ConnectivityProfile:
        """Convert a parquet row to a ConnectivityProfile."""
        # Parse JSON strings back to dicts
        def parse_json(val):
            if pd.isna(val) or val is None or val == '':
                return None
            if isinstance(val, dict):
                return val
            try:
                return json.loads(val)
            except (TypeError, ValueError) as e:
                # A malformed JSON cell silently drops that partner map;
                # log it so a systematically broken column is discoverable.
                self._log(f"Warning: unparseable JSON profile column "
                          f"value ({str(val)[:60]!r}): {e}")
                return None
        
        def parse_int_key_mapping(val):
            # bodyId -> type mappings lose their int keys in the JSON/CSV
            # round-trip ({'1': 'A'}); restore them like from_dict() does.
            parsed = parse_json(val)
            if not parsed:
                return parsed
            restored = {}
            for k, v in parsed.items():
                try:
                    restored[int(k)] = v
                except (TypeError, ValueError):
                    restored[k] = v
            return restored
        
        return ConnectivityProfile(
            neuron_id=row['neuron_id'],
            dataset=row['dataset'],
            upstream_partners=parse_json(row.get('upstream_partners')) or {},
            downstream_partners=parse_json(row.get('downstream_partners')) or {},
            upstream_ranks=parse_json(row.get('upstream_ranks')) or {},
            downstream_ranks=parse_json(row.get('downstream_ranks')) or {},
            upstream_top_k=int(row.get('upstream_top_k', 10)),
            downstream_top_k=int(row.get('downstream_top_k', 10)),
            total_upstream_weight=float(row.get('total_upstream_weight', 0)),
            total_downstream_weight=float(row.get('total_downstream_weight', 0)),
            num_neurons_aggregated=int(row.get('num_neurons_aggregated', 1)),
            untyped_upstream_count=int(row.get('untyped_upstream_count', 0)),
            untyped_downstream_count=int(row.get('untyped_downstream_count', 0)),
            untyped_upstream_weight_fraction=float(row.get('untyped_upstream_weight_fraction', 0)),
            untyped_downstream_weight_fraction=float(row.get('untyped_downstream_weight_fraction', 0)),
            actual_upstream_count=int(row.get('actual_upstream_count', 0)),
            actual_downstream_count=int(row.get('actual_downstream_count', 0)),
            is_weak_connectivity=bool(row.get('is_weak_connectivity', False)),
            unique_types_upstream=int(row.get('unique_types_upstream', 0)),
            unique_types_downstream=int(row.get('unique_types_downstream', 0)),
            is_sparse=bool(row.get('is_sparse', False)),
            partner_type_mapping_upstream=parse_int_key_mapping(row.get('partner_type_mapping_upstream')),
            partner_type_mapping_downstream=parse_int_key_mapping(row.get('partner_type_mapping_downstream')),
            top_k_bodyid_used=int(row.get('top_k_bodyid_used', 20)),
            top_m_type_target=int(row.get('top_m_type_target', 5)),
            untyped_upstream_bodyids=parse_json(row.get('untyped_upstream_bodyids')),
            untyped_downstream_bodyids=parse_json(row.get('untyped_downstream_bodyids')),
            untyped_upstream_2hop=parse_json(row.get('untyped_upstream_2hop')),
            untyped_downstream_2hop=parse_json(row.get('untyped_downstream_2hop')),
            untyped_upstream_2hop_ranks=parse_json(row.get('untyped_upstream_2hop_ranks')),
            untyped_downstream_2hop_ranks=parse_json(row.get('untyped_downstream_2hop_ranks')),
            # Round 7: typed bodyId-level data
            typed_upstream_bodyids=parse_json(row.get('typed_upstream_bodyids')),
            typed_downstream_bodyids=parse_json(row.get('typed_downstream_bodyids')),
        )
    
    def _profile_to_row(self, profile: ConnectivityProfile) -> dict:
        """Convert a ConnectivityProfile to a flat dict for parquet storage."""
        data = profile.to_dict()
        # Provenance for cache gating (F-XD-002): a row is only reusable
        # when the config that shaped it matches the reader's config.
        data['config_hash'] = self._get_config_hash()
        
        # Convert nested dicts to JSON strings for parquet storage
        # Use empty JSON object '{}' instead of None to avoid parquet struct/non-struct mixing error
        for key in ['upstream_partners', 'downstream_partners', 'upstream_ranks', 'downstream_ranks',
                    'partner_type_mapping_upstream', 'partner_type_mapping_downstream',
                    'untyped_upstream_bodyids', 'untyped_downstream_bodyids',
                    'untyped_upstream_2hop', 'untyped_downstream_2hop',
                    'untyped_upstream_2hop_ranks', 'untyped_downstream_2hop_ranks',
                    'typed_upstream_bodyids', 'typed_downstream_bodyids']:
            if key in data and data[key] is not None:
                data[key] = json.dumps(data[key])
            else:
                # Use empty JSON object string instead of None to ensure consistent column type
                data[key] = '{}'
        
        return data
    
    def _save_to_cache(self, profile: ConnectivityProfile):
        """
        Save profile to cache using per-file approach for interruption safety.
        
        IMPORTANT: Only bodyId-level profiles are saved to disk cache.
        Type-level profiles (neuron_id is a string) are only kept in memory cache
        since they can be quickly rebuilt from bodyId profiles using aggregate_bodyid_profiles().
        
        Uses individual files per profile (like connection batch files) to ensure:
        - Each profile is saved immediately as its own file
        - Interrupted runs don't lose progress
        - Files are consolidated later into main cache
        
        When _defer_cache_writes is True (during parallel processing), writes are queued
        and flushed later via flush_pending_cache_writes().
        """
        # Memory cache (always, no lock needed for dict assignment)
        cache_key = (str(profile.neuron_id), profile.dataset)
        profile._config_hash = self._get_config_hash()
        self._memory_cache[cache_key] = profile
        
        # Only save bodyId-level profiles to disk (neuron_id must be an integer)
        # Type-level profiles are computed on-the-fly and not cached to disk
        is_bodyid_profile = isinstance(profile.neuron_id, (int, np.integer))
        if not is_bodyid_profile:
            # Skip disk cache for type-level profiles
            return
        
        # Disk cache (parquet)
        if not self.config.use_cache:
            return
        
        # If deferred writes enabled (parallel processing), queue for later batch save
        if getattr(self, '_defer_cache_writes', False):
            dataset = profile.dataset
            if dataset not in self._pending_cache_writes:
                self._pending_cache_writes[dataset] = {}
            self._pending_cache_writes[dataset][str(profile.neuron_id)] = profile
            return
        
        # INTERRUPTION-SAFE: Save to individual batch file (no loading of existing data)
        # This approach survives Ctrl+C, OOM, and crashes
        self._save_profile_to_batch_file(profile)
        
        # Also update in-memory index for immediate lookup
        if profile.dataset in self._disk_cache_df:
            # Add to in-memory DataFrame and index
            new_row = self._profile_to_row(profile)
            new_row_df = pd.DataFrame([new_row])
            
            # Remove existing entry if present
            df = self._disk_cache_df[profile.dataset]
            df = df[df['neuron_id'] != str(profile.neuron_id)]
            df = pd.concat([df, new_row_df], ignore_index=True)
            self._disk_cache_df[profile.dataset] = df
            
            # Update index
            self._disk_cache_index[profile.dataset][str(profile.neuron_id)] = len(df) - 1
    
    def flush_pending_cache_writes(self, silent: bool = False):
        """
        Flush all pending cache writes to disk using batch approach.
        
        Called after parallel processing completes to safely write all 
        accumulated profile updates.
        
        IMPORTANT: Only bodyId-level profiles are saved to disk.
        Type-level profiles are kept in memory only.
        
        Args:
            silent: If True, suppress logging messages (for use during progress bars)
        """
        if not getattr(self, '_pending_cache_writes', None):
            return
        
        with self._cache_write_lock:
            for dataset, profiles in self._pending_cache_writes.items():
                if not profiles:
                    continue
                
                # Filter to only bodyId-level profiles (neuron_id must be integer)
                # Type-level profiles stay in memory only
                bodyid_profiles = {
                    k: v for k, v in profiles.items() 
                    if isinstance(v.neuron_id, (int, np.integer))
                }
                
                if not bodyid_profiles:
                    if not silent:
                        self._log(f"Skipped {len(profiles)} type-level profiles (not cached to disk)")
                    continue
                
                # Save all profiles in ONE batch file (much faster than per-profile files)
                # This batch file will be consolidated later
                self._save_profiles_batch_file(bodyid_profiles, dataset)
                
                if not silent:
                    skipped = len(profiles) - len(bodyid_profiles)
                    msg = f"Saved {len(bodyid_profiles)} profiles to batch file for {dataset}"
                    if skipped > 0:
                        msg += f" (skipped {skipped} type-level profiles)"
                    self._log(msg)
            
            # Clear pending writes
            self._pending_cache_writes = {}
    
    def _save_profiles_batch_file(self, profiles: Dict[str, ConnectivityProfile], dataset: str):
        """
        Save multiple profiles to a single batch file (much faster than per-profile).
        
        File naming: {batch_dir}/batch_{timestamp}_{count}.parquet
        """
        if not profiles:
            return
            
        batch_dir = self._get_profile_batch_dir(dataset)
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Create batch filename with timestamp and count
        import time
        timestamp = int(time.time() * 1000)
        batch_file = batch_dir / f"batch_{timestamp}_{len(profiles)}.parquet"
        
        # Convert all profiles to rows
        rows = [self._profile_to_row(p) for p in profiles.values()]
        df = pd.DataFrame(rows)
        
        # Atomic write with a reclaimable temp-sibling name (R7-2)
        try:
            from utils.parquet_utils import replace_with_retry, temp_sibling
        except ImportError:
            from .utils.parquet_utils import replace_with_retry, temp_sibling
        temp_file = Path(temp_sibling(str(batch_file), 'profile-batch'))
        try:
            df.to_parquet(temp_file, index=False)
            replace_with_retry(temp_file, batch_file)
        except Exception as e:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            self._log(f"Warning: Could not save profiles batch file: {e}")
    
    def _save_profiles_to_cache_batch(
        self, 
        profiles: Dict[str, ConnectivityProfile], 
        dataset: str,
        silent: bool = False
    ):
        """
        Save multiple profiles to cache using batch file approach.
        
        All bodyId-level profiles are saved to a single batch file for efficiency.
        Type-level profiles are only kept in memory, not saved to disk.
        Use consolidate_profile_cache() to merge into main cache file.
        
        Args:
            profiles: Dict mapping neuron_id to ConnectivityProfile
            dataset: Dataset identifier
            silent: If True, suppress logging messages
        """
        if not profiles:
            return
        
        # Memory cache update (all profiles, including type-level)
        current_hash = self._get_config_hash()
        for neuron_id, profile in profiles.items():
            cache_key = (str(neuron_id), dataset)
            profile._config_hash = current_hash
            self._memory_cache[cache_key] = profile
        
        # Disk cache using batch file approach (one file for all profiles)
        if not self.config.use_cache:
            return
        
        # Filter to only bodyId-level profiles for disk cache
        bodyid_profiles = {
            k: v for k, v in profiles.items() 
            if isinstance(v.neuron_id, (int, np.integer))
        }
        
        if not bodyid_profiles:
            return
        
        # Save all profiles to ONE batch file (much faster)
        self._save_profiles_batch_file(bodyid_profiles, dataset)
        
        if not silent:
            self._log(f"Saved {len(bodyid_profiles)} profiles to batch file")
    
    def consolidate_profile_cache(self, dataset: str = None):
        """
        Consolidate profile batch files into main cache file.
        
        Call this after profile building to merge individual profile files
        into the main connectivity_profiles.parquet. This improves subsequent
        load times.
        
        Args:
            dataset: Dataset to consolidate. If None, consolidates the
                datasets already loaded into memory THIS SESSION (batch
                dirs of never-loaded datasets are skipped — pass each
                dataset explicitly to consolidate those).
        """
        if dataset:
            count = self._consolidate_profile_batch_files(dataset)
            if count > 0:
                self._log(f"Consolidated {count} profiles for {dataset}")
        else:
            # Consolidate the session's loaded datasets
            for ds in list(self._disk_cache_df.keys()):
                count = self._consolidate_profile_batch_files(ds)
                if count > 0:
                    self._log(f"Consolidated {count} profiles for {ds}")

    def read_connectivity_profile_cache(
        self,
        dataset: str,
        neuron_types: Optional[List[str]] = None,
        min_top_k: Optional[int] = None
    ) -> Dict[str, ConnectivityProfile]:
        """
        Read pre-built connectivity profiles from cache (parquet format).
        
        This method loads cached profiles from disk for efficient batch access.
        Uses the 3-tier cache system for O(1) lookups when neuron_types are specified.
        
        Args:
            dataset: Dataset identifier (e.g., 'hemibrain_v1_2_1', 'flywire_FAFB_v783')
            neuron_types: Specific neuron types to load. If None, loads all cached profiles.
            min_top_k: Minimum top_k required. Profiles with lower top_k are skipped.
        
        Returns:
            Dict mapping neuron_type/id to ConnectivityProfile
        """
        self._log(f"Reading connectivity profile cache for {dataset}...")
        
        # Load cache DataFrame (uses in-memory cache if available)
        cache_df = self._load_cache_dataframe(dataset)
        
        if cache_df is None or cache_df.empty:
            cache_path = self._get_cache_parquet_path(dataset)
            self._log(f"Cache not found: {cache_path}")
            return {}
        
        profiles = {}
        loaded = 0
        skipped_k = 0
        failed = 0
        
        self._log(f"Found {len(cache_df)} profiles in cache")
        
        # Use index for O(1) lookups if specific neuron_types requested
        if neuron_types is not None and dataset in self._disk_cache_index:
            # Fast path: Use index for specific types
            for neuron_id in neuron_types:
                neuron_id_str = str(neuron_id)
                row_idx = self._disk_cache_index[dataset].get(neuron_id_str)
                
                if row_idx is None:
                    continue
                
                try:
                    row = cache_df.iloc[row_idx]
                    profile = self._fill_neuron_type(
                        self._row_to_profile(row), dataset)

                    # Check min_top_k requirement
                    if min_top_k is not None and profile.top_k_bodyid_used < min_top_k:
                        skipped_k += 1
                        continue
                    
                    profiles[neuron_id_str] = profile
                    loaded += 1
                    
                    # Add to memory cache (stamped with the config the row
                    # was persisted under, so tier-1 gating can re-check)
                    profile._config_hash = (str(row.get('config_hash'))
                                            if row.get('config_hash') is not None
                                            and pd.notna(row.get('config_hash'))
                                            else self._get_config_hash())
                    cache_key = (neuron_id_str, dataset)
                    self._memory_cache[cache_key] = profile
                    
                except Exception as e:
                    self._log(f"Warning: Failed to load profile for {neuron_id}: {e}")
                    failed += 1
        else:
            # Slow path: Iterate all rows (only when loading ALL profiles)
            for idx, row in cache_df.iterrows():
                try:
                    neuron_id = row['neuron_id']

                    # Convert row to profile
                    profile = self._fill_neuron_type(
                        self._row_to_profile(row), dataset)

                    # Check min_top_k requirement
                    if min_top_k is not None:
                        if profile.top_k_bodyid_used < min_top_k:
                            skipped_k += 1
                            continue
                    
                    profiles[neuron_id] = profile
                    loaded += 1
                    
                    # Also add to memory cache (stamped with the row's
                    # config hash for tier-1 re-checking)
                    profile._config_hash = (str(row.get('config_hash'))
                                            if row.get('config_hash') is not None
                                            and pd.notna(row.get('config_hash'))
                                            else self._get_config_hash())
                    cache_key = (str(neuron_id), dataset)
                    self._memory_cache[cache_key] = profile
                    
                except Exception as e:
                    self._log(f"Warning: Failed to load profile from row {idx}: {e}")
                    failed += 1
        
        self._log(f"Loaded {loaded} profiles from cache "
                  f"(skipped {skipped_k} with insufficient top_k, {failed} failed)")
        
        return profiles
    
    def get_cache_stats(self, dataset: str) -> Dict[str, Any]:
        """
        Get statistics about cached profiles for a dataset.
        
        Args:
            dataset: Dataset identifier
        
        Returns:
            Dict with cache statistics:
            - 'total_profiles': Number of cached profiles
            - 'total_size_mb': Total cache size in MB
            - 'neuron_types': List of cached neuron types
            - 'top_k_distribution': Dict of top_k_used -> count
            - 'cache_modified': Last modification time of cache file
        """
        cache_path = self._get_cache_parquet_path(dataset)
        
        if not cache_path.exists():
            return {
                'total_profiles': 0,
                'total_size_mb': 0,
                'neuron_types': [],
                'top_k_distribution': {},
                'cache_modified': None
            }
        
        cache_df = self._load_cache_dataframe(dataset)
        
        if cache_df is None or cache_df.empty:
            return {
                'total_profiles': 0,
                'total_size_mb': 0,
                'neuron_types': [],
                'top_k_distribution': {},
                'cache_modified': None
            }
        
        total_size = cache_path.stat().st_size
        neuron_types = cache_df['neuron_id'].tolist() if 'neuron_id' in cache_df.columns else []
        
        # Get top_k distribution
        top_k_dist = {}
        if 'top_k_bodyid_used' in cache_df.columns:
            for top_k in cache_df['top_k_bodyid_used'].dropna():
                top_k = int(top_k)
                top_k_dist[top_k] = top_k_dist.get(top_k, 0) + 1
        
        from datetime import datetime
        cache_mtime = cache_path.stat().st_mtime
        
        return {
            'total_profiles': len(cache_df),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'neuron_types': neuron_types,
            'top_k_distribution': top_k_dist,
            'cache_modified': datetime.fromtimestamp(cache_mtime).isoformat()
        }
    
    def ensure_data_available(self, dataset: str, raise_on_missing: bool = True) -> bool:
        """
        Ensure connection data is available for a dataset.
        
        This method checks if the required connection cache files exist for
        building connectivity profiles. For FAFB and standalone BANC local releases,
        it checks the datasets/ folder for merged_connections files.
        
        Args:
            dataset: Dataset identifier (e.g., 'flywire_FAFB_v783', 'hemibrain:v1.2.1')
            raise_on_missing: If True, raise DataNotAvailableError when data is missing
            
        Returns:
            True if data is available, False otherwise
            
        Raises:
            DataNotAvailableError: If raise_on_missing=True and data is not found
            
        Example:
            >>> profiler.ensure_data_available('flywire_FAFB_v783')
            True  # Data exists
            
            >>> profiler.ensure_data_available('some_new_dataset')
            DataNotAvailableError: Connection data not found for 'some_new_dataset'.
            Please run: python src/build_connection_cache.py some_new_dataset
        """
        is_local = is_local_connectome_dataset(dataset)
        
        if is_local:
            # Check for local connection files
            safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
            src_dir = Path(__file__).parent.parent
            project_root = src_dir.parent
            datasets_folder = project_root / 'datasets'
            dataset_path = datasets_folder / safe_name
            
            # Check for connection files
            conn_files = [
                dataset_path / f'{safe_name}_merged_connections.parquet',
                dataset_path / f'{safe_name}_merged_connections.csv',
                dataset_path / f'{safe_name}_connections.parquet',
                dataset_path / f'{safe_name}_connections.csv',
                dataset_path / 'connections.parquet',
                dataset_path / 'connections.csv',
            ]
            
            data_exists = any(f.exists() for f in conn_files)
            
            if not data_exists and raise_on_missing:
                raise DataNotAvailableError(
                    f"Connection data not found for '{dataset}'.\n\n"
                    f"Expected location: {dataset_path}/\n"
                    f"Expected files: {safe_name}_merged_connections.parquet or .csv\n\n"
                    f"To build the connection cache, run:\n"
                    f"  python src/build_connection_cache.py {dataset}\n\n"
                    f"Or ensure the dataset files exist in the datasets/ folder."
                )
            
            return data_exists
        else:
            # For NeuPrint datasets, check if we can create a client
            try:
                client = self._get_client_for_dataset(dataset)
                if client is None:
                    if raise_on_missing:
                        reason = self._client_errors.get(dataset)
                        if reason:
                            # name what the server actually said rather than
                            # guessing at credentials
                            raise DataNotAvailableError(
                                f"Cannot connect to NeuPrint for '{dataset}'.\n\n"
                                f"The server refused this dataset: {reason}\n\n"
                                "If this release is analysed from local tables, "
                                "the dataset NAME is the problem (NeuPrint's list "
                                "no longer matches it), not the token; "
                                "otherwise check NEUPRINT_APPLICATION_CREDENTIALS "
                                "and that the token may access it."
                            )
                        raise DataNotAvailableError(
                            f"Cannot connect to NeuPrint for '{dataset}'.\n\n"
                            f"Please ensure:\n"
                            f"  1. NEUPRINT_APPLICATION_CREDENTIALS environment variable is set\n"
                            f"  2. The token has access to dataset '{dataset}'\n"
                            f"  3. NeuPrint server (neuprint.janelia.org) is accessible"
                        )
                    return False
                return True
            except Exception as e:
                if raise_on_missing:
                    raise DataNotAvailableError(
                        f"Failed to verify data availability for '{dataset}': {e}"
                    )
                return False
    
    def get_data_status(self, datasets: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get status of connection data for multiple datasets.
        
        Args:
            datasets: List of datasets to check. If None, uses self.datasets.
            
        Returns:
            Dict mapping dataset -> status info:
                - 'available': bool
                - 'type': 'local' or 'neuprint'
                - 'path': Path to data file (for local)
                - 'rows': Number of connection rows (if available)
                - 'error': Error message (if not available)
        """
        if datasets is None:
            datasets = self.datasets
        
        status = {}
        for dataset in datasets:
            is_local = is_local_connectome_dataset(dataset)
            
            try:
                available = self.ensure_data_available(dataset, raise_on_missing=False)
                
                if is_local:
                    safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
                    src_dir = Path(__file__).parent.parent
                    project_root = src_dir.parent
                    dataset_path = project_root / 'datasets' / safe_name
                    
                    # Find the actual file
                    conn_file = None
                    for fname in [f'{safe_name}_merged_connections.parquet',
                                  f'{safe_name}_merged_connections.csv',
                                  f'{safe_name}_connections.parquet',
                                  'connections.parquet']:
                        f = dataset_path / fname
                        if f.exists():
                            conn_file = f
                            break
                    
                    rows = None
                    if conn_file and available:
                        try:
                            conn_df = self._get_cached_conn_df(dataset)
                            if conn_df is not None:
                                rows = len(conn_df)
                        except Exception:
                            pass
                    
                    status[dataset] = {
                        'available': available,
                        'type': 'local',
                        'path': str(conn_file) if conn_file else str(dataset_path),
                        'rows': rows,
                        'error': None if available else f"No connection file found in {dataset_path}"
                    }
                else:
                    status[dataset] = {
                        'available': available,
                        'type': 'neuprint',
                        'path': 'neuprint.janelia.org',
                        'rows': None,  # Would need API call to count
                        'error': None if available else "Cannot connect to NeuPrint"
                    }
            except Exception as e:
                status[dataset] = {
                    'available': False,
                    'type': 'local' if is_local else 'neuprint',
                    'path': None,
                    'rows': None,
                    'error': str(e)
                }
        
        return status

    def _normalize_neuprint_dataset_name(self, dataset: str) -> str:
        """Convert a dataset folder name to its NeuPrint dataset name.

        Thin delegate to the shared boundary helper (``flywire_ids``), which
        also refuses to translate FAFB / standalone BANC:
        ``hemibrain_v1_2_1`` -> ``hemibrain:v1.2.1``, ``male-cns_v0_9`` ->
        ``male-cns:v0.9``.
        """
        return neuprint_dataset_name(dataset)

    def _get_client_for_dataset(self, dataset: str) -> Optional[Any]:
        """Get or create NeuPrint client for a dataset."""
        if dataset in self._clients:
            return self._clients[dataset]
        
        # FAFB and standalone BANC use local release tables and no NeuPrint
        # client.  Keep the source decision centralized.
        if is_local_connectome_dataset(dataset):
            # Local dataset - no client needed
            self._clients[dataset] = None
            return None
        
        # Create NeuPrint client
        if self.client is not None:
            self._clients[dataset] = self.client
            return self.client
        
        try:
            from neuprint import Client
            
            # Use provided token or fall back to the env/config chain
            try:
                from utils.token_manager import token_manager
                resolved = token_manager.get_neuprint_token()
            except ImportError:
                resolved = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS', '')
            token = self.token or resolved or ''
            if not token:
                self._log(f"Warning: No NeuPrint token found for {dataset}")
            
            # Normalize dataset name for NeuPrint API
            neuprint_dataset = self._normalize_neuprint_dataset_name(dataset)
            client = Client('neuprint.janelia.org', dataset=neuprint_dataset, token=token)
            self._clients[dataset] = client
            return client
        except Exception as e:
            self._client_errors[dataset] = str(e)
            self._log(f"Warning: Could not create client for {dataset}: {e}")
            return None
    
    def _join_type_info_from_neuron_df(
        self,
        conn_df: pd.DataFrame,
        dataset_path: Path,
        safe_name: str
    ) -> pd.DataFrame:
        """
        Join type information from neuron_df to connection DataFrame.
        
        This is needed for FAFB/BANC local releases where merged_connections
        files don't include type columns. Type info must be looked up from
        the neuron_df file.
        
        Args:
            conn_df: Connection DataFrame with bodyId_pre, bodyId_post columns
            dataset_path: Path to dataset folder
            safe_name: Sanitized dataset name
        
        Returns:
            conn_df with type_pre and type_post columns added
        """
        # Try to load neuron_df
        neuron_files = [
            dataset_path / f'{safe_name}_allneurons_neuron_df.parquet',
            dataset_path / f'{safe_name}_allneurons_neuron_df.csv',
            dataset_path / f'{safe_name}_neuron_df.parquet',
            dataset_path / f'{safe_name}_neuron_df.csv',
            dataset_path / 'neuron_df.parquet',
            dataset_path / 'neuron_df.csv',
        ]
        
        neuron_df = None
        for neuron_file in neuron_files:
            if neuron_file.exists():
                try:
                    if str(neuron_file).endswith('.parquet'):
                        neuron_df = pd.read_parquet(neuron_file)
                    else:
                        neuron_df = pd.read_csv(neuron_file, low_memory=False)
                    self._log(f"Loaded neuron info from {neuron_file}", level='debug')
                    break
                except Exception as e:
                    self._log(f"Warning: Could not load {neuron_file}: {e}")
        
        if neuron_df is None or neuron_df.empty:
            self._log("Warning: No neuron_df found to join type information")
            return conn_df
        
        # Determine type column name in neuron_df
        # IMPORTANT: 'type' should be checked BEFORE 'cell_type' because:
        # - FAFB/FlyWire 'type' has specific types like 'aMe12', 'PPL101'
        # - FAFB/FlyWire 'cell_type' has generic categories like 't5_neuron', 'transmedullary'
        type_col = None
        for col in ['type', 'Type', 'cellType', 'cell_type']:
            if col in neuron_df.columns:
                type_col = col
                break
        
        if type_col is None:
            self._log("Warning: No type column found in neuron_df")
            return conn_df
        
        # Standardize bodyId column
        if 'bodyId' not in neuron_df.columns:
            for col in ['root_id', 'pt_root_id', 'body_id']:
                if col in neuron_df.columns:
                    neuron_df = neuron_df.rename(columns={col: 'bodyId'})
                    break
        
        if 'bodyId' not in neuron_df.columns:
            self._log("Warning: No bodyId column found in neuron_df")
            return conn_df
        
        # Create type lookup - ensure bodyId types match.
        # IMPORTANT: conn_df may be the shared _FNC_CACHE frame and neuron_df
        # may come from statvis' module cache - mutating either in place would
        # silently flip dtypes/add columns for every other consumer. Work on
        # copies.
        neuron_df = neuron_df.copy()
        conn_df = conn_df.copy()
        neuron_df['bodyId'] = neuron_df['bodyId'].astype(str)
        type_lookup = neuron_df.set_index('bodyId')[type_col].to_dict()
        
        # Ensure connection bodyIds are strings too
        conn_df['bodyId_pre'] = conn_df['bodyId_pre'].astype(str)
        conn_df['bodyId_post'] = conn_df['bodyId_post'].astype(str)
        
        # Map types
        conn_df['type_pre'] = conn_df['bodyId_pre'].map(type_lookup)
        conn_df['type_post'] = conn_df['bodyId_post'].map(type_lookup)
        
        # Log stats
        typed_pre = conn_df['type_pre'].notna().sum()
        typed_post = conn_df['type_post'].notna().sum()
        total = len(conn_df)
        self._log(f"Joined type info: {typed_pre}/{total} pre-types, {typed_post}/{total} post-types", level='debug')
        
        return conn_df
    
    def _query_connections_neuprint(
        self,
        neuron: Union[str, int, List],
        dataset: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Query upstream and downstream connections from NeuPrint.
        
        Returns:
            Tuple of (upstream_df, downstream_df)
            Each has columns: partner_bodyId, partner_type, weight
        """
        client = self._get_client_for_dataset(dataset)
        if client is None:
            self._log(f"ERROR: No NeuPrint client available for {dataset}. "
                     f"Set NEUPRINT_APPLICATION_CREDENTIALS env variable or provide token.")
            return pd.DataFrame(), pd.DataFrame()
        
        min_syn = self.config.min_synapse_threshold
        
        # Import API utilities for Cypher escaping and timeout
        try:
            from src.utils.api_utils import escape_cypher_string, api_call_with_retry
        except ImportError:
            # Fallback: inline escape function
            escape_cypher_string = _escape_cypher_string_fallback
            api_call_with_retry = None
        
        # Build neuron condition with proper escaping for special characters
        if isinstance(neuron, str):
            escaped_neuron = escape_cypher_string(neuron)
            # Type-based query with regex support
            if '.*' in neuron or '*' in neuron:
                neuron_cond = f"n.type =~ '{escaped_neuron}'"
            else:
                neuron_cond = f"n.type = '{escaped_neuron}'"
        elif isinstance(neuron, int):
            neuron_cond = f"n.bodyId = {neuron}"
        elif isinstance(neuron, list):
            bodyids_str = ', '.join(str(b) for b in neuron)
            neuron_cond = f"n.bodyId IN [{bodyids_str}]"
        else:
            raise ValueError(f"Unsupported neuron type: {type(neuron)}")
        
        # Query upstream (inputs to the neuron)
        upstream_query = f"""
        MATCH (pre:Neuron)-[c:ConnectsTo]->(n:Neuron)
        WHERE {neuron_cond} AND c.weight >= {min_syn}
        RETURN pre.bodyId AS partner_bodyId, pre.type AS partner_type, 
               n.bodyId AS neuron_bodyId, c.weight AS weight
        """
        
        # Query downstream (outputs from the neuron)
        downstream_query = f"""
        MATCH (n:Neuron)-[c:ConnectsTo]->(post:Neuron)
        WHERE {neuron_cond} AND c.weight >= {min_syn}
        RETURN post.bodyId AS partner_bodyId, post.type AS partner_type,
               n.bodyId AS neuron_bodyId, c.weight AS weight
        """
        
        try:
            # Use timeout wrapper if available
            if api_call_with_retry is not None:
                upstream_df = api_call_with_retry(
                    lambda: client.fetch_custom(upstream_query),
                    timeout=60.0,
                    max_retries=2,
                    description=f"Upstream query for {neuron}",
                    verbose=False
                )
                downstream_df = api_call_with_retry(
                    lambda: client.fetch_custom(downstream_query),
                    timeout=60.0,
                    max_retries=2,
                    description=f"Downstream query for {neuron}",
                    verbose=False
                )
            else:
                upstream_df = client.fetch_custom(upstream_query)
                downstream_df = client.fetch_custom(downstream_query)
            return upstream_df, downstream_df
        except Exception as e:
            self._log(f"Warning: Query failed for {neuron} in {dataset}: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def _get_cached_conn_df(self, dataset: str) -> Optional[pd.DataFrame]:
        """
        Get connection DataFrame from cache, checking FNC's cache first.
        
        Priority:
        1. Check profiler's own cache (fastest)
        2. Check FNC's module-level cache (already indexed)
        3. Load from disk if needed
        
        Thread-safe: Uses module-level lock to prevent race conditions.
        
        Returns:
            Cached and preprocessed connection DataFrame, or None if not available
        """
        global _PROFILER_CONN_CACHE, _PROFILER_CONN_CACHE_LOCK, _PROFILER_CACHE_LOGGED
        
        # Build cache key
        safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
        
        # Quick check without lock (for already cached data in profiler cache)
        if safe_name in _PROFILER_CONN_CACHE and 'conn_df' in _PROFILER_CONN_CACHE[safe_name]:
            cached_df = _PROFILER_CONN_CACHE[safe_name]['conn_df']
            if cached_df is not None:
                return cached_df
        
        # Check FNC's module-level cache (set by coana.py after build_connection_cache)
        try:
            from coana import _FNC_CACHE
            if safe_name in _FNC_CACHE and 'conn_df' in _FNC_CACHE[safe_name]:
                fnc_df = _FNC_CACHE[safe_name]['conn_df']
                fnc_index = _FNC_CACHE[safe_name].get('conn_index', {})
                
                # Handle both Polars and pandas DataFrames from FNC cache
                # FNC may store Polars DFs for efficiency, but profiler expects pandas
                fnc_is_empty = False
                if fnc_df is not None:
                    try:
                        import polars as pl
                        if isinstance(fnc_df, pl.DataFrame):
                            fnc_is_empty = fnc_df.is_empty()
                            if not fnc_is_empty:
                                # Convert Polars to pandas for profiler compatibility
                                fnc_df = fnc_df.to_pandas()
                        else:
                            fnc_is_empty = fnc_df.empty
                    except ImportError:
                        fnc_is_empty = fnc_df.empty if hasattr(fnc_df, 'empty') else len(fnc_df) == 0
                else:
                    fnc_is_empty = True
                
                if fnc_df is not None and not fnc_is_empty:
                    # Use FNC's cache - it's already loaded and indexed
                    # Store reference in profiler cache too
                    if safe_name not in _PROFILER_CONN_CACHE:
                        _PROFILER_CONN_CACHE[safe_name] = {}
                    
                    # Need to ensure type columns exist and build post index
                    conn_df = fnc_df
                    
                    # Check if type columns exist, if not join from neuron_df
                    if 'type_pre' not in conn_df.columns or 'type_post' not in conn_df.columns:
                        src_dir = Path(__file__).parent.parent
                        project_root = src_dir.parent
                        dataset_path = project_root / 'datasets' / safe_name
                        conn_df = self._join_type_info_from_neuron_df(conn_df, dataset_path, safe_name)
                    
                    _PROFILER_CONN_CACHE[safe_name]['conn_df'] = conn_df
                    # Use FNC's pre index
                    _PROFILER_CONN_CACHE[safe_name]['bodyid_pre_index'] = fnc_index
                    
                    # Use FNC's post index if available, otherwise build it
                    fnc_post_index = _FNC_CACHE[safe_name].get('conn_index_post')
                    if fnc_post_index:
                        _PROFILER_CONN_CACHE[safe_name]['bodyid_post_index'] = fnc_post_index
                    elif 'bodyid_post_index' not in _PROFILER_CONN_CACHE[safe_name]:
                        # Build bodyId_post index only if not available from FNC
                        # This happens once per dataset, log it so user knows what's happening
                        n_rows = len(conn_df)
                        if n_rows > 100000:
                            self._log(f"Building post index for {dataset} ({n_rows:,} rows)...")
                        post_index = {}
                        if 'bodyId_post' in conn_df.columns:
                            post_col = conn_df['bodyId_post'].values
                            for idx in range(len(post_col)):
                                post_key = str(post_col[idx])
                                if post_key not in post_index:
                                    post_index[post_key] = []
                                post_index[post_key].append(idx)
                        _PROFILER_CONN_CACHE[safe_name]['bodyid_post_index'] = post_index
                        if n_rows > 100000:
                            self._log(f"Post index built: {len(post_index):,} unique downstream neurons")
                    
                    if safe_name not in _PROFILER_CACHE_LOGGED:
                        _PROFILER_CACHE_LOGGED.add(safe_name)
                        self._log(f"Using FNC cache for {dataset} ({len(conn_df):,} rows, indexed)")
                    
                    return conn_df
        except ImportError:
            pass
        
        # Acquire lock for loading from disk
        with _PROFILER_CONN_CACHE_LOCK:
            # Double-check after acquiring lock (another thread may have loaded)
            if safe_name in _PROFILER_CONN_CACHE and 'conn_df' in _PROFILER_CONN_CACHE[safe_name]:
                cached_df = _PROFILER_CONN_CACHE[safe_name]['conn_df']
                if cached_df is not None:
                    return cached_df
            
            # Load from disk (inside lock to prevent duplicate loading)
            src_dir = Path(__file__).parent.parent
            project_root = src_dir.parent
            datasets_folder = project_root / 'datasets'
            cache_folder = project_root / 'cache'
            dataset_path = datasets_folder / safe_name
            cache_path = cache_folder / safe_name
        
            # Try to load connections file - check multiple naming conventions
            # Priority: 1) datasets/ folder (pre-processed data)
            #           2) cache/ folder (built by FNC.build_connection_cache)
            conn_files = [
                # datasets/ folder (for pre-processed datasets like FlyWire)
                dataset_path / f'{safe_name}_merged_connections.parquet',
                dataset_path / f'{safe_name}_merged_connections.csv',
                dataset_path / f'{safe_name}_connections.parquet',
                dataset_path / f'{safe_name}_connections.csv',
                dataset_path / 'connections.parquet',
                dataset_path / 'connections.csv',
                # cache/ folder (for NeuPrint datasets built via build_connection_cache)
                cache_path / 'connections.parquet',
            ]
            
            conn_df = None
            for conn_file in conn_files:
                if conn_file.exists():
                    try:
                        if str(conn_file).endswith('.parquet'):
                            # Try Polars for large connection files
                            try:
                                import polars as pl
                                conn_df_pl = pl.read_parquet(str(conn_file))
                                conn_df = conn_df_pl.to_pandas()
                                del conn_df_pl
                            except ImportError:
                                conn_df = pd.read_parquet(conn_file)
                        else:
                            conn_df = pd.read_csv(conn_file)
                        break
                    except Exception as e:
                        self._log(f"Warning: Could not load {conn_file}: {e}")
            
            if conn_df is None or conn_df.empty:
                # Cache the None result to avoid repeated disk checks
                if safe_name not in _PROFILER_CONN_CACHE:
                    _PROFILER_CONN_CACHE[safe_name] = {}
                _PROFILER_CONN_CACHE[safe_name]['conn_df'] = None
                return None
            
            # Standardize column names
            col_mapping = {
                'pre_pt_root_id': 'bodyId_pre',
                'post_pt_root_id': 'bodyId_post',
                'pre_type': 'type_pre',
                'post_type': 'type_post',
                'syn_count': 'weight',
            }
            conn_df = conn_df.rename(columns={k: v for k, v in col_mapping.items() if k in conn_df.columns})
            
            # If type columns missing, try to join from neuron_df
            if 'type_pre' not in conn_df.columns or 'type_post' not in conn_df.columns:
                conn_df = self._join_type_info_from_neuron_df(conn_df, dataset_path, safe_name)
            
            # Ensure required columns exist
            if 'weight' not in conn_df.columns:
                if 'syn_count' in conn_df.columns:
                    conn_df['weight'] = conn_df['syn_count']
                else:
                    conn_df['weight'] = 1
            
            # Build O(1) lookup indexes for fast querying
            # This dramatically speeds up profile building
            bodyid_pre_index = {}  # bodyId -> list of row indices where it's pre
            bodyid_post_index = {}  # bodyId -> list of row indices where it's post
            
            n_rows = len(conn_df)
            if n_rows > 100000:
                self._log(f"Building connection indexes for {dataset} ({n_rows:,} rows)...")
            
            if 'bodyId_pre' in conn_df.columns and 'bodyId_post' in conn_df.columns:
                # Try Polars for faster index building (2-3x faster for large datasets)
                try:
                    import polars as pl
                    
                    # Build indexes using Polars group_by (much faster than Python loop)
                    df_pl = pl.DataFrame({
                        'bodyId_pre': conn_df['bodyId_pre'].astype(str).values,
                        'bodyId_post': conn_df['bodyId_post'].astype(str).values,
                        'idx': range(n_rows)
                    })
                    
                    # Group by pre and collect indices using iter_rows for efficiency
                    pre_result = df_pl.group_by('bodyId_pre').agg(pl.col('idx'))
                    bodyid_pre_index = {row[0]: row[1] for row in pre_result.iter_rows()}
                    
                    # Group by post and collect indices
                    post_result = df_pl.group_by('bodyId_post').agg(pl.col('idx'))
                    bodyid_post_index = {row[0]: row[1] for row in post_result.iter_rows()}
                    
                    del df_pl, pre_result, post_result
                    
                except ImportError:
                    # Fallback to optimized Python with defaultdict
                    from collections import defaultdict
                    bodyid_pre_index = defaultdict(list)
                    bodyid_post_index = defaultdict(list)
                    pre_col = conn_df['bodyId_pre'].values
                    post_col = conn_df['bodyId_post'].values
                    for idx in range(len(pre_col)):
                        bodyid_pre_index[str(pre_col[idx])].append(idx)
                        bodyid_post_index[str(post_col[idx])].append(idx)
                    # Convert to regular dict
                    bodyid_pre_index = dict(bodyid_pre_index)
                    bodyid_post_index = dict(bodyid_post_index)
            
            if n_rows > 100000:
                self._log(f"Indexes built: {len(bodyid_pre_index):,} upstream, {len(bodyid_post_index):,} downstream neurons")
            
            # Cache the preprocessed DataFrame and indexes
            if safe_name not in _PROFILER_CONN_CACHE:
                _PROFILER_CONN_CACHE[safe_name] = {}
            _PROFILER_CONN_CACHE[safe_name]['conn_df'] = conn_df
            _PROFILER_CONN_CACHE[safe_name]['bodyid_pre_index'] = bodyid_pre_index
            _PROFILER_CONN_CACHE[safe_name]['bodyid_post_index'] = bodyid_post_index
            
            # Log only once per dataset (track in module-level set)
            if safe_name not in _PROFILER_CACHE_LOGGED:
                _PROFILER_CACHE_LOGGED.add(safe_name)
                self._log(f"Cached connection data for {dataset} ({len(conn_df):,} rows, indexed)")
            
            return conn_df
    
    def _query_connections_local(
        self,
        neuron: Union[str, int, List],
        dataset: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Query upstream and downstream connections from local dataset files.
        
        Uses module-level cache with pre-built indexes for O(1) lookups.
        
        Returns:
            Tuple of (upstream_df, downstream_df)
        """
        global _PROFILER_CONN_CACHE
        
        # Get cached connection DataFrame
        conn_df = self._get_cached_conn_df(dataset)
        
        if conn_df is None or conn_df.empty:
            self._log(f"Warning: No connection data found for {dataset}")
            return pd.DataFrame(), pd.DataFrame()
        
        min_syn = self.config.min_synapse_threshold
        safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
        
        # Get pre-built indexes for O(1) lookup
        cache_entry = _PROFILER_CONN_CACHE.get(safe_name, {})
        bodyid_pre_index = cache_entry.get('bodyid_pre_index', {})
        bodyid_post_index = cache_entry.get('bodyid_post_index', {})
        
        # For bodyId queries (int), use O(1) index lookup
        if isinstance(neuron, int):
            neuron_str = str(neuron)
            
            # Upstream: connections where this neuron is post (inputs to neuron)
            up_indices = bodyid_post_index.get(neuron_str, [])
            if up_indices:
                upstream = conn_df.iloc[up_indices].copy()
                if min_syn > 0:
                    upstream = upstream[upstream['weight'] >= min_syn]
                if not upstream.empty:
                    upstream = upstream.rename(columns={
                        'bodyId_pre': 'partner_bodyId',
                        'type_pre': 'partner_type',
                        'bodyId_post': 'neuron_bodyId',
                    })
                    upstream = upstream[['partner_bodyId', 'partner_type', 'neuron_bodyId', 'weight']]
            else:
                upstream = pd.DataFrame()
            
            # Downstream: connections where this neuron is pre (outputs from neuron)
            down_indices = bodyid_pre_index.get(neuron_str, [])
            if down_indices:
                downstream = conn_df.iloc[down_indices].copy()
                if min_syn > 0:
                    downstream = downstream[downstream['weight'] >= min_syn]
                if not downstream.empty:
                    downstream = downstream.rename(columns={
                        'bodyId_post': 'partner_bodyId',
                        'type_post': 'partner_type',
                        'bodyId_pre': 'neuron_bodyId',
                    })
                    downstream = downstream[['partner_bodyId', 'partner_type', 'neuron_bodyId', 'weight']]
            else:
                downstream = pd.DataFrame()
            
            return upstream, downstream
        
        # For list of bodyIds, use index lookup for each
        elif isinstance(neuron, list):
            up_indices = []
            down_indices = []
            for n in neuron:
                n_str = str(n)
                up_indices.extend(bodyid_post_index.get(n_str, []))
                down_indices.extend(bodyid_pre_index.get(n_str, []))
            
            if up_indices:
                upstream = conn_df.iloc[up_indices].copy()
                if min_syn > 0:
                    upstream = upstream[upstream['weight'] >= min_syn]
                if not upstream.empty:
                    upstream = upstream.rename(columns={
                        'bodyId_pre': 'partner_bodyId',
                        'type_pre': 'partner_type',
                        'bodyId_post': 'neuron_bodyId',
                    })
                    upstream = upstream[['partner_bodyId', 'partner_type', 'neuron_bodyId', 'weight']]
            else:
                upstream = pd.DataFrame()
            
            if down_indices:
                downstream = conn_df.iloc[down_indices].copy()
                if min_syn > 0:
                    downstream = downstream[downstream['weight'] >= min_syn]
                if not downstream.empty:
                    downstream = downstream.rename(columns={
                        'bodyId_post': 'partner_bodyId',
                        'type_post': 'partner_type',
                        'bodyId_pre': 'neuron_bodyId',
                    })
                    downstream = downstream[['partner_bodyId', 'partner_type', 'neuron_bodyId', 'weight']]
            else:
                downstream = pd.DataFrame()
            
            return upstream, downstream
        
        # For type-based queries (string), resolve types via the lazily built
        # type->row index instead of full-table regex scans per query.
        elif isinstance(neuron, str):
            if 'type_pre' not in conn_df.columns:
                return pd.DataFrame(), pd.DataFrame()
            
            type_idx = self._get_type_row_index(conn_df, safe_name)
            
            if '.*' in neuron or '*' in neuron:
                import re
                pattern = neuron.replace('.*', '.*').replace('*', '.*')
                rx = re.compile(pattern)
                # Upstream of typed neurons: their type lives in type_post
                up_types = [t for t in type_idx['post'] if rx.match(t)]
                down_types = [t for t in type_idx['pre'] if rx.match(t)]
            else:
                up_types = [neuron] if neuron in type_idx['post'] else []
                down_types = [neuron] if neuron in type_idx['pre'] else []
            
            up_indices = [i for t in up_types for i in type_idx['post'][t]]
            down_indices = [i for t in down_types for i in type_idx['pre'][t]]
            
            if up_indices:
                upstream = conn_df.iloc[up_indices].copy()
                if min_syn > 0:
                    upstream = upstream[upstream['weight'] >= min_syn]
                if not upstream.empty:
                    upstream = upstream.rename(columns={
                        'bodyId_pre': 'partner_bodyId',
                        'type_pre': 'partner_type',
                        'bodyId_post': 'neuron_bodyId',
                    })
                    upstream = upstream[['partner_bodyId', 'partner_type', 'neuron_bodyId', 'weight']]
            else:
                upstream = pd.DataFrame()
            
            if down_indices:
                downstream = conn_df.iloc[down_indices].copy()
                if min_syn > 0:
                    downstream = downstream[downstream['weight'] >= min_syn]
                if not downstream.empty:
                    downstream = downstream.rename(columns={
                        'bodyId_post': 'partner_bodyId',
                        'type_post': 'partner_type',
                        'bodyId_pre': 'neuron_bodyId',
                    })
                    downstream = downstream[['partner_bodyId', 'partner_type', 'neuron_bodyId', 'weight']]
            else:
                downstream = pd.DataFrame()
            
            return upstream, downstream
        
        else:
            raise ValueError(f"Unsupported neuron type: {type(neuron)}")
    
    def _get_type_row_index(self, conn_df: pd.DataFrame, safe_name: str) -> Dict[str, Dict[str, List[int]]]:
        """
        Lazily build (and cache) type -> row-position maps for the connection
        table, used by type-based queries.
        
        Type queries previously scanned the FULL table with a regex per query
        (O(rows) pandas str.match on millions of rows, repeated for every
        neuron type in a profiling run). With the index, a type query costs
        O(types) lookup + O(matching rows) iloc, built once per dataset.
        """
        cached = self._type_row_index.get(safe_name)
        # Rebuild if the underlying frame object changed (cache reload)
        if cached is not None and cached.get('conn_id') == id(conn_df):
            return cached
        
        pre_map: Dict[str, List[int]] = {}
        post_map: Dict[str, List[int]] = {}
        if 'type_pre' in conn_df.columns and 'type_post' in conn_df.columns:
            positions = pd.Series(np.arange(len(conn_df)))
            for t, idx_arr in positions.groupby(conn_df['type_pre'].astype(str)).indices.items():
                pre_map[t] = idx_arr.tolist()
            for t, idx_arr in positions.groupby(conn_df['type_post'].astype(str)).indices.items():
                post_map[t] = idx_arr.tolist()
        
        entry = {'conn_id': id(conn_df), 'pre': pre_map, 'post': post_map}
        self._type_row_index[safe_name] = entry
        return entry
    
    def _fetch_2hop_partners(
        self,
        untyped_bodyids: List[int],
        dataset: str,
        direction: str
    ) -> Dict[int, Tuple[Dict[str, float], Dict[str, int]]]:
        """
        Round 6: Fetch 2-hop typed partners for untyped 1-hop partners.
        
        For each untyped 1-hop partner, query their connections and return
        only typed 2-hop partners with their weights and ranks.
        
        Uses batched processing with Polars for efficiency when there are
        many untyped partners (common in datasets with incomplete type annotations).
        
        Args:
            untyped_bodyids: List of untyped 1-hop partner bodyIds
            dataset: Dataset identifier
            direction: 'upstream' or 'downstream' - indicates the direction from original neuron
                       For upstream untyped partners, we get their upstream (2-hop upstream)
                       For downstream untyped partners, we get their downstream (2-hop downstream)
        
        Returns:
            Dict mapping untyped_bodyId → (weights_dict, ranks_dict)
            where weights_dict = {typed_type → normalized_weight}
            and ranks_dict = {typed_type → rank}
        """
        global _PROFILER_CONN_CACHE
        
        if not untyped_bodyids:
            return {}
        
        # Use cached connection data
        conn_df = self._get_cached_conn_df(dataset)
        
        if conn_df is None or conn_df.empty:
            return {}
        
        # Get pre-built indexes for O(1) lookup per bodyId
        safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
        cache_entry = _PROFILER_CONN_CACHE.get(safe_name, {})
        bodyid_pre_index = cache_entry.get('bodyid_pre_index', {})
        bodyid_post_index = cache_entry.get('bodyid_post_index', {})
        
        min_syn = self.config.min_synapse_threshold
        top_k_2hop = self.config.top_k_2hop
        
        # Determine index and columns based on direction
        if direction == 'upstream':
            index_to_use = bodyid_post_index
            partner_type_col = 'type_pre'
            source_bid_col = 'bodyId_post'
        else:
            index_to_use = bodyid_pre_index
            partner_type_col = 'type_post'
            source_bid_col = 'bodyId_pre'
        
        # Check column existence once
        if partner_type_col not in conn_df.columns:
            return {bid: ({}, {}) for bid in untyped_bodyids}
        
        # === BATCHED APPROACH: Collect all indices at once ===
        # This is much faster than processing one bodyId at a time
        all_indices = []
        bid_to_indices = {}
        for untyped_bid in untyped_bodyids:
            bid_str = str(untyped_bid)
            indices = index_to_use.get(bid_str, [])
            if indices:
                bid_to_indices[untyped_bid] = (len(all_indices), len(all_indices) + len(indices))
                all_indices.extend(indices)
        
        # If no indices found, return empty results
        if not all_indices:
            return {bid: ({}, {}) for bid in untyped_bodyids}
        
        # Single iloc for all indices (much faster than multiple small ilocs)
        all_hop2_df = conn_df.iloc[all_indices]
        
        # Apply min_syn filter once
        if min_syn > 0:
            all_hop2_df = all_hop2_df[all_hop2_df['weight'] >= min_syn]
        
        # Filter to only typed 2-hop partners once
        typed_mask = all_hop2_df[partner_type_col].notna() & (all_hop2_df[partner_type_col].astype(str).str.strip() != '')
        all_hop2_df = all_hop2_df[typed_mask]
        
        if all_hop2_df.empty:
            return {bid: ({}, {}) for bid in untyped_bodyids}
        
        # Apply fuzzy matching once using vectorized method
        if self.config.fuzzy_match.enabled:
            all_hop2_df = all_hop2_df.copy()
            all_hop2_df['partner_type_normalized'] = self._normalize_types_vectorized(
                all_hop2_df[partner_type_col], self.config.fuzzy_match
            )
        else:
            all_hop2_df = all_hop2_df.copy()
            all_hop2_df['partner_type_normalized'] = all_hop2_df[partner_type_col].astype(str)
        
        # Process results per untyped bodyId
        results = {}
        
        try:
            import polars as pl
            
            # Convert entire filtered DataFrame to Polars once
            pl_df = pl.from_pandas(all_hop2_df[[source_bid_col, 'partner_type_normalized', 'weight']])
            
            # Ensure consistent type for bodyId column - cast to Int64 to match untyped_bid
            # This fixes "cannot compare string with numeric type (i32)" error
            cast_failed = False
            try:
                pl_df = pl_df.with_columns(pl.col(source_bid_col).cast(pl.Int64))
            except Exception:
                # Keep as-is and compare with strings below
                cast_failed = True
            
            # Single grouped top-k aggregation for ALL bodyIds at once.
            # The previous implementation ran pl_df.filter(...) per untyped
            # bodyId - a full scan of the table for each one (O(B x rows)).
            # group_by + per-group sort/head computes the same per-bodyId
            # top-k-then-aggregate result in one pass.
            per_bid = (
                pl_df
                .sort([source_bid_col, 'weight'], descending=[False, True])
                .group_by(source_bid_col, maintain_order=True)
                .agg(
                    pl.col('partner_type_normalized')
                    .sort_by(pl.col('weight'), descending=True)
                    .head(top_k_2hop)
                    .alias('top_types'),
                    pl.col('weight')
                    .sort_by(pl.col('weight'), descending=True)
                    .head(top_k_2hop)
                    .alias('top_weights'),
                )
            )
            per_bid_map = dict(zip(
                per_bid[source_bid_col].to_list(),
                zip(per_bid['top_types'].to_list(), per_bid['top_weights'].to_list()),
            ))
            
            # Process each untyped bodyId
            for untyped_bid in untyped_bodyids:
                if untyped_bid not in bid_to_indices:
                    results[untyped_bid] = ({}, {})
                    continue
                
                try:
                    key = str(untyped_bid) if cast_failed else int(untyped_bid)
                except (ValueError, TypeError):
                    key = str(untyped_bid)
                entry = per_bid_map.get(key)
                if entry is None and not cast_failed:
                    # Defensive: retry with string key if int() succeeded but
                    # the column somehow retained string values.
                    entry = per_bid_map.get(str(untyped_bid))
                if entry is None:
                    results[untyped_bid] = ({}, {})
                    continue
                
                top_types, top_weights = entry
                
                # Aggregate the top-k rows by type
                agg = {}
                for t, w in zip(top_types, top_weights):
                    agg[t] = agg.get(t, 0.0) + float(w)
                
                # Normalize weights
                total_weight = sum(agg.values())
                if total_weight > 0:
                    weights_dict = {t: w / total_weight for t, w in agg.items()}
                else:
                    weights_dict = {}
                
                # Compute ranks
                ranks_dict = compute_ranks(weights_dict)
                results[untyped_bid] = (weights_dict, ranks_dict)
                
        except ImportError:
            # Fallback to Pandas - still batched but slightly slower
            for untyped_bid in untyped_bodyids:
                bid_str = str(untyped_bid)
                if untyped_bid not in bid_to_indices:
                    results[untyped_bid] = ({}, {})
                    continue
                
                # Filter for this bodyId - handle type mismatch between column and bid
                try:
                    # Try numeric comparison first
                    mask = all_hop2_df[source_bid_col] == int(untyped_bid)
                except (ValueError, TypeError):
                    # Fallback to string comparison
                    mask = all_hop2_df[source_bid_col].astype(str) == str(untyped_bid)
                hop2_df = all_hop2_df[mask]
                
                if hop2_df.empty:
                    results[untyped_bid] = ({}, {})
                    continue
                
                # Take top-k by weight
                hop2_df = hop2_df.nlargest(top_k_2hop, 'weight')
                
                # Aggregate by type using vectorized operations
                aggregated = hop2_df.groupby('partner_type_normalized')['weight'].sum()
                total_weight = aggregated.sum()
                
                if total_weight > 0:
                    weights_dict = (aggregated / total_weight).to_dict()
                else:
                    weights_dict = {}
                
                ranks_dict = compute_ranks(weights_dict)
                results[untyped_bid] = (weights_dict, ranks_dict)
        
        # Fill in missing bodyIds
        for untyped_bid in untyped_bodyids:
            if untyped_bid not in results:
                results[untyped_bid] = ({}, {})
        
        return results
    
    def _process_connections(
        self,
        conn_df: pd.DataFrame,
        direction: str,
        top_k: int
    ) -> Tuple[Dict[str, float], Dict[str, int], int, float, float, int, int, Dict[int, str], int, Dict[int, float], Dict[int, float]]:
        """
        Process connection DataFrame into normalized weights and ranks.
        
        Round 5: Added dynamic expansion to ensure minimum unique types,
        and bodyId→type mapping for metadata.
        
        Round 6: Also returns untyped partner bodyIds with their weights
        for 2-hop expansion.
        
        Round 7: Also returns typed partner bodyIds with their weights
        for intra-dataset bodyId-level comparison.
        
        Args:
            conn_df: DataFrame with partner_type, weight columns
            direction: 'upstream' or 'downstream' (for logging)
            top_k: Number of top partners to keep
        
        Returns:
            Tuple of (partners_dict, ranks_dict, untyped_count, 
                     untyped_weight_fraction, total_weight, actual_count,
                     unique_types, type_mapping, k_used, untyped_bodyids, typed_bodyids)
            where untyped_bodyids = {bodyId → weight} for untyped 1-hop partners
            and typed_bodyids = {bodyId → weight} for typed 1-hop partners
        """
        empty_result = ({}, {}, 0, 0.0, 0.0, 0, 0, {}, top_k, {}, {})
        
        if conn_df.empty:
            return empty_result
        
        # Ensure partner_type column exists
        if 'partner_type' not in conn_df.columns:
            return empty_result
        
        # Track untyped partners
        untyped_mask = conn_df['partner_type'].isna() | (conn_df['partner_type'].astype(str).str.strip() == '')
        untyped_count = untyped_mask.sum()
        untyped_weight = conn_df.loc[untyped_mask, 'weight'].sum() if untyped_count > 0 else 0.0
        total_weight = conn_df['weight'].sum()
        
        # Round 6: Extract untyped partner bodyIds and weights for 2-hop expansion
        untyped_bodyids_dict = {}
        if self.config.expand_untyped_2hop and 'partner_bodyId' in conn_df.columns:
            untyped_df = conn_df[untyped_mask].copy()
            if not untyped_df.empty:
                # Sort by weight and take top-k untyped
                untyped_df = untyped_df.nlargest(self.config.top_k_bodyid, 'weight')
                # Use vectorized extraction instead of iterrows
                try:
                    bids = untyped_df['partner_bodyId'].astype(int).tolist()
                    weights = untyped_df['weight'].astype(float).tolist()
                    untyped_bodyids_dict = dict(zip(bids, weights))
                except (ValueError, TypeError):
                    # Fallback for edge cases
                    for _, row in untyped_df.iterrows():
                        try:
                            bid = int(row['partner_bodyId'])
                            w = float(row['weight'])
                            untyped_bodyids_dict[bid] = w
                        except (ValueError, TypeError):
                            pass
        
        # Filter untyped if configured
        if not self.config.include_untyped_partners:
            conn_df = conn_df[~untyped_mask].copy()
        else:
            # Use 'untyped' as collective type for untyped partners (Round 5)
            conn_df = conn_df.copy()
            conn_df.loc[untyped_mask, 'partner_type'] = 'untyped'
        
        if conn_df.empty:
            # Round 6: Return with untyped bodyids even if no typed partners.
            # NOTE: must return all 11 elements (including typed_bodyids_dict)
            # - callers unpack 11 values, and returning 10 here used to raise
            # ValueError, silently dropping every neuron with only-untyped
            # partners from batch profiling.
            return ({}, {}, untyped_count, untyped_weight / total_weight if total_weight > 0 else 0.0, 
                    total_weight, 0, 0, {}, top_k, untyped_bodyids_dict, {})
        
        # Apply fuzzy matching to partner types using vectorized lookup
        if self.config.fuzzy_match.enabled:
            conn_df = conn_df.copy()
            # Use vectorized type normalization via pre-computed cache or Polars
            conn_df['partner_type_normalized'] = self._normalize_types_vectorized(
                conn_df['partner_type'], self.config.fuzzy_match
            )
        else:
            conn_df['partner_type_normalized'] = conn_df['partner_type'].astype(str)
        
        # Round 5: Dynamic expansion to ensure minimum unique types
        k_used = top_k
        top_m = self.config.top_m_type
        max_k = top_k * self.config.max_expansion_factor
        
        if self.config.dynamic_expansion:
            # Sort by weight first (reset index: positional math below relies
            # on a clean RangeIndex)
            conn_df = conn_df.sort_values('weight', ascending=False, ignore_index=True)
            
            # Single-pass expansion: the old loop re-ran head(k).nunique()
            # for k += 5 steps (quadratic work). The position where the
            # top_m-th distinct type first appears is computed once with a
            # cumulative unique count - identical selection semantics.
            n_rows = len(conn_df)
            is_first_occ = ~conn_df['partner_type_normalized'].duplicated(keep='first')
            cum_unique = is_first_occ.cumsum()
            total_unique = int(cum_unique.iloc[-1]) if n_rows > 0 else 0
            
            if total_unique >= top_m:
                hit_positions = cum_unique[cum_unique >= top_m].index
                first_hit_pos = conn_df.index.get_loc(hit_positions[0])
                # k_hit = smallest head size reaching top_m unique types.
                k_hit = first_hit_pos + 1
                # Replicate the old +5 stepping exactly (same final k), but
                # arithmetically - the old loop re-ran head(k).nunique() on
                # every step, which was the quadratic cost.
                k_used = top_k
                while k_used < k_hit and k_used < max_k and k_used < n_rows:
                    k_used += 5
            else:
                # Fewer unique types than top_m anywhere: keep as many rows as
                # the expansion budget allows (matches previous behavior).
                k_used = min(max_k, n_rows)
            
            conn_df = conn_df.head(k_used)
        else:
            # No dynamic expansion - just sort and take top_k
            conn_df = conn_df.sort_values('weight', ascending=False).head(top_k)
        
        # === POLARS OPTIMIZATION: Aggregate by normalized partner type ===
        # Use Polars for faster groupby aggregation (2-3x speedup)
        try:
            import polars as pl
            
            # Convert to Polars for fast aggregation
            pl_df = pl.from_pandas(conn_df[['partner_type_normalized', 'weight']])
            
            # Group by and aggregate - Polars is significantly faster
            aggregated_pl = (
                pl_df
                .group_by('partner_type_normalized')
                .agg(pl.col('weight').sum())
                .sort('weight', descending=True)
            )
            
            actual_count = len(aggregated_pl)
            unique_types = actual_count
            
            # Convert to dict directly (faster than iterrows)
            partners_dict = dict(zip(
                aggregated_pl['partner_type_normalized'].to_list(),
                aggregated_pl['weight'].to_list()
            ))
            
        except ImportError:
            # Fallback to Pandas if Polars not available
            aggregated = conn_df.groupby('partner_type_normalized').agg({
                'weight': 'sum'
            }).reset_index()
            
            actual_count = len(aggregated)
            unique_types = actual_count
            
            # Sort by weight
            aggregated = aggregated.sort_values('weight', ascending=False)
            
            # Store actual synapse weights (not normalized)
            partners_dict = {}
            for _, row in aggregated.iterrows():
                partners_dict[row['partner_type_normalized']] = float(row['weight'])
        
        # Compute ranks (derived from weights: higher weight = lower rank number)
        ranks_dict = compute_ranks(partners_dict)
        
        # === POLARS OPTIMIZATION: Build bodyId → type mapping ===
        # Round 5: Build bodyId → type mapping
        # Round 7: Build typed bodyId → weight mapping for intra-dataset comparison
        type_mapping = {}
        typed_bodyids_dict = {}
        if 'partner_bodyId' in conn_df.columns:
            try:
                import polars as pl
                
                # Filter and convert in one pass using Polars
                # Only include non-untyped, non-null entries
                pl_map_df = pl.from_pandas(
                    conn_df[['partner_bodyId', 'partner_type_normalized', 'weight']]
                )
                
                # Filter out untyped and null
                valid_df = pl_map_df.filter(
                    (pl.col('partner_type_normalized').is_not_null()) &
                    (pl.col('partner_type_normalized') != 'untyped') &
                    (pl.col('partner_bodyId').is_not_null())
                )
                
                if len(valid_df) > 0:
                    # Extract to lists for dict construction (faster than iterrows)
                    bodyids = valid_df['partner_bodyId'].cast(pl.Int64).to_list()
                    types = valid_df['partner_type_normalized'].cast(pl.Utf8).to_list()
                    weights = valid_df['weight'].cast(pl.Float64).to_list()
                    
                    type_mapping = dict(zip(bodyids, types))
                    typed_bodyids_dict = dict(zip(bodyids, weights))
                    
            except (ImportError, Exception):
                # Fallback to Pandas iterrows (slower)
                for _, row in conn_df.iterrows():
                    bid = row.get('partner_bodyId')
                    ptype = row.get('partner_type_normalized')
                    weight = row.get('weight', 0.0)
                    if bid is not None and ptype is not None and ptype != 'untyped':
                        try:
                            bid_int = int(bid)
                            type_mapping[bid_int] = str(ptype)
                            typed_bodyids_dict[bid_int] = float(weight)
                        except (ValueError, TypeError):
                            pass
        
        return (
            partners_dict,
            ranks_dict,
            untyped_count,
            untyped_weight / total_weight if total_weight > 0 else 0.0,
            total_weight,
            actual_count,
            unique_types,
            type_mapping,
            k_used,
            untyped_bodyids_dict,  # Round 6: untyped bodyId → weight
            typed_bodyids_dict,    # Round 7: typed bodyId → weight
        )
    
    def _bodyid_type_map(self, dataset: str) -> Dict[str, str]:
        """bodyId(str) -> type name for one dataset, loaded once per instance.

        Feeds the neuron_type backfill for bodyId-queried profiles (the
        profile builders leave it None). Reads the dataset's neuron table,
        then the neuron index — the same local sources the type lookups
        use; an empty map means no local table (NeuPrint-API-only dataset),
        where misses fall back to per-id `get_types_for_bodyids` calls
        memoized into this map.
        """
        cached = self._bodyid_type_maps.get(dataset)
        if cached is not None:
            return cached
        from pathlib import Path
        safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
        project_root = Path(__file__).parent.parent.parent
        candidates = [
            project_root / 'datasets' / safe_name
            / f'{safe_name}_allneurons_neuron_df.parquet',
            project_root / 'datasets' / safe_name
            / f'{safe_name}_allneurons_neuron_df.csv',
            project_root / 'neuron_indexes' / safe_name / 'neuron_index.parquet',
        ]
        lookup: Dict[str, str] = {}
        for path in candidates:
            if not path.exists():
                continue
            try:
                if path.suffix.lower() == '.parquet':
                    columns = None
                    try:
                        import pyarrow.parquet as _pq
                        available = set(_pq.read_schema(path).names)
                        columns = [c for c in ('bodyId', 'type', 'cellType',
                                               'cell_type') if c in available]
                        if not {'type', 'cellType', 'cell_type'} & set(columns):
                            continue
                    except Exception:
                        columns = None
                    df = pd.read_parquet(path, columns=columns)
                else:
                    df = pd.read_csv(path, low_memory=False)
            except Exception:
                continue
            if 'bodyId' not in df.columns:
                continue
            type_col = next((c for c in ('type', 'cellType', 'cell_type')
                             if c in df.columns), None)
            if type_col is None:
                continue
            lookup = dict(zip(df['bodyId'].astype(str),
                              df[type_col].fillna('').astype(str)))
            break
        self._bodyid_type_maps[dataset] = lookup
        return lookup

    def _fill_neuron_type(self, profile: ConnectivityProfile,
                          dataset: str) -> ConnectivityProfile:
        """Backfill ``neuron_type`` on bodyId-queried profiles in place.

        BodyId profiles are built with neuron_type=None (the builders only
        set it for type-name queries); every consumer previously had to
        re-resolve the name itself. The fill runs after cache retrieval and
        after building, so freshly built profiles persist the name into the
        profile disk cache too. Returns *profile* unchanged on any miss.
        """
        if profile is None or profile.neuron_type:
            return profile
        bid = getattr(profile, 'neuron_id', None)
        if bid is None or isinstance(bid, (list, tuple, dict)):
            return profile
        text = str(bid).strip()
        if not text.isdigit():
            return profile
        type_map = self._bodyid_type_map(dataset)
        ntype = type_map.get(text)
        if not ntype:
            try:
                resolved = self.get_types_for_bodyids([int(text)], dataset) or {}
                ntype = resolved.get(int(text))
                if ntype:
                    # Memoize the (possibly API-backed) resolution so the
                    # same bodyId never resolves twice per process.
                    type_map[text] = str(ntype)
            except Exception:
                ntype = None
        if ntype:
            profile.neuron_type = str(ntype)
        return profile

    def get_profile(
        self,
        neuron: Union[str, int, List],
        dataset: str,
        force_refresh: bool = False
    ) -> ConnectivityProfile:
        """
        Extract connectivity profile for a neuron or neuron group.
        
        Round 5: Now includes dynamic expansion to ensure minimum unique types,
        and stores bodyId→type mapping metadata.
        
        Args:
            neuron: Neuron type (str), bodyId (int), or list of bodyIds
            dataset: Dataset identifier (e.g., 'hemibrain:v1.2.1')
            force_refresh: Bypass cache if True
        
        Returns:
            ConnectivityProfile with both proportion and rank representations
            
        Raises:
            DataNotAvailableError: If connection data is not available for the dataset
        
        Example:
            >>> profile = profiler.get_profile('aMe12', 'hemibrain:v1.2.1')
            >>> print(profile.upstream_partners)
        """
        # Check cache first
        if not force_refresh:
            cached = self._load_from_cache(neuron, dataset)
            if cached is not None:
                self._log(f"Loaded from cache: {neuron} in {dataset}", level='debug')
                return self._fill_neuron_type(cached, dataset)
        
        self._log(f"Extracting profile for {neuron} in {dataset}", level='debug')
        
        # Ensure connection data is available before querying (cached check)
        # Only check once per dataset per session to avoid repeated overhead
        if dataset not in self._data_availability_cache:
            self.ensure_data_available(dataset, raise_on_missing=True)
            self._data_availability_cache[dataset] = True
        
        # Query connections - ALWAYS try local cache first (much faster)
        # Local cache includes: FAFB/BANC local releases AND NeuPrint datasets
        # with pre-built connection cache from FNC.build_connection_cache()
        upstream_df, downstream_df = self._query_connections_local(neuron, dataset)
        
        # Fall back to NeuPrint API only if local data not available
        if upstream_df.empty and downstream_df.empty:
            if not is_local_connectome_dataset(dataset):
                # Try NeuPrint API as fallback
                upstream_df, downstream_df = self._query_connections_neuprint(neuron, dataset)
        
        # Count neurons aggregated
        neurons_aggregated = 1
        if isinstance(neuron, list):
            neurons_aggregated = len(neuron)
        elif isinstance(neuron, str) and not upstream_df.empty:
            # Type-based query - count unique bodyIds
            if 'neuron_bodyId' in upstream_df.columns:
                neurons_aggregated = upstream_df['neuron_bodyId'].nunique()
            elif not downstream_df.empty and 'neuron_bodyId' in downstream_df.columns:
                neurons_aggregated = downstream_df['neuron_bodyId'].nunique()
        
        # Process upstream (Round 5/6/7: expanded return tuple with untyped and typed bodyids)
        (up_partners, up_ranks, up_untyped_count, up_untyped_frac, 
         up_total_weight, up_actual_count, up_unique_types,
         up_type_mapping, up_k_used, up_untyped_bodyids, up_typed_bodyids) = self._process_connections(
            upstream_df, 'upstream', self.config.top_k_bodyid
        )
        
        # Process downstream (Round 5/6/7: expanded return tuple with untyped and typed bodyids)
        (down_partners, down_ranks, down_untyped_count, down_untyped_frac,
         down_total_weight, down_actual_count, down_unique_types,
         down_type_mapping, down_k_used, down_untyped_bodyids, down_typed_bodyids) = self._process_connections(
            downstream_df, 'downstream', self.config.top_k_bodyid
        )
        
        # Round 6: Fetch 2-hop partners for untyped 1-hop partners
        up_2hop_weights = None
        up_2hop_ranks = None
        down_2hop_weights = None
        down_2hop_ranks = None
        
        if self.config.expand_untyped_2hop:
            # Fetch 2-hop for upstream untyped partners
            if up_untyped_bodyids:
                up_2hop_data = self._fetch_2hop_partners(
                    list(up_untyped_bodyids.keys()), dataset, 'upstream'
                )
                if up_2hop_data:
                    up_2hop_weights = {bid: data[0] for bid, data in up_2hop_data.items() if data[0]}
                    up_2hop_ranks = {bid: data[1] for bid, data in up_2hop_data.items() if data[1]}
            
            # Fetch 2-hop for downstream untyped partners
            if down_untyped_bodyids:
                down_2hop_data = self._fetch_2hop_partners(
                    list(down_untyped_bodyids.keys()), dataset, 'downstream'
                )
                if down_2hop_data:
                    down_2hop_weights = {bid: data[0] for bid, data in down_2hop_data.items() if data[0]}
                    down_2hop_ranks = {bid: data[1] for bid, data in down_2hop_data.items() if data[1]}
        
        # Create profile with Round 5, Round 6, and Round 7 fields
        profile_type = neuron if isinstance(neuron, str) else None
        profile = ConnectivityProfile(
            neuron_id=neuron,
            dataset=dataset,
            neuron_type=profile_type,
            upstream_partners=up_partners,
            downstream_partners=down_partners,
            upstream_ranks=up_ranks,
            downstream_ranks=down_ranks,
            upstream_top_k=self.config.top_k_bodyid,
            downstream_top_k=self.config.top_k_bodyid,
            total_upstream_weight=up_total_weight,
            total_downstream_weight=down_total_weight,
            num_neurons_aggregated=neurons_aggregated,
            untyped_upstream_count=up_untyped_count,
            untyped_downstream_count=down_untyped_count,
            untyped_upstream_weight_fraction=up_untyped_frac,
            untyped_downstream_weight_fraction=down_untyped_frac,
            actual_upstream_count=up_actual_count,
            actual_downstream_count=down_actual_count,
            # Round 5 fields
            unique_types_upstream=up_unique_types,
            unique_types_downstream=down_unique_types,
            partner_type_mapping_upstream=up_type_mapping if up_type_mapping else None,
            partner_type_mapping_downstream=down_type_mapping if down_type_mapping else None,
            top_k_bodyid_used=max(up_k_used, down_k_used),
            top_m_type_target=self.config.top_m_type,
            # Round 6 fields: 2-hop expansion for untyped partners
            untyped_upstream_bodyids=up_untyped_bodyids if up_untyped_bodyids else None,
            untyped_downstream_bodyids=down_untyped_bodyids if down_untyped_bodyids else None,
            untyped_upstream_2hop=up_2hop_weights if up_2hop_weights else None,
            untyped_downstream_2hop=down_2hop_weights if down_2hop_weights else None,
            untyped_upstream_2hop_ranks=up_2hop_ranks if up_2hop_ranks else None,
            untyped_downstream_2hop_ranks=down_2hop_ranks if down_2hop_ranks else None,
            # Round 7 fields: typed bodyId-level data for intra-dataset comparison
            typed_upstream_bodyids=up_typed_bodyids if up_typed_bodyids else None,
            typed_downstream_bodyids=down_typed_bodyids if down_typed_bodyids else None,
        )
        
        # Fill the type name for bodyId queries, then persist (the name
        # rides along into the profile disk cache).
        profile = self._fill_neuron_type(profile, dataset)

        # Save to cache
        self._save_to_cache(profile)
        
        self._log(f"Profile extracted: {up_actual_count} upstream ({up_unique_types} types), "
                  f"{down_actual_count} downstream ({down_unique_types} types)", level='debug')
        
        return profile
    
    def get_profiles_batch(
        self,
        neurons: List[Union[str, int]],
        dataset: str,
        force_refresh: bool = False,
        skip_profile_cache: bool = False,
        show_progress: bool = True
    ) -> Dict[Union[str, int], ConnectivityProfile]:
        """
        Batch extraction for multiple neurons.
        
        More efficient than calling get_profile repeatedly for large queries.
        Uses the connection cache with O(1) index lookups.
        
        Args:
            neurons: List of neuron types or bodyIds
            dataset: Dataset identifier
            force_refresh: Bypass cache if True
            skip_profile_cache: Skip individual profile disk caching (faster for large batches)
            show_progress: Show progress bar
        
        Returns:
            Dict mapping neuron identifier to ConnectivityProfile
        """
        results = {}
        
        # Ensure connection cache is loaded first
        self._get_cached_conn_df(dataset)
        
        for neuron in tqdm(neurons, desc="Building profiles from cache",
                          disable=progress_bars_disabled(show_progress, getattr(self, 'verbose', True)), unit="neuron"):
            try:
                if skip_profile_cache:
                    # Build profile directly from connection cache (no disk cache check)
                    profile = self._build_profile_from_cache_direct(neuron, dataset)
                else:
                    profile = self.get_profile(neuron, dataset, force_refresh)
                if profile is not None:
                    results[neuron] = profile
            except Exception as e:
                self._log(f"Warning: Failed to extract profile for {neuron}: {e}")
        
        return results
    
    def _build_profile_from_cache_direct(
        self,
        neuron: Union[str, int],
        dataset: str
    ) -> Optional[ConnectivityProfile]:
        """
        Build profile directly from connection cache without disk caching.
        
        This is faster for large batch operations where individual profile
        caching would create too much overhead.
        
        Args:
            neuron: Neuron bodyId (int) or type (str)
            dataset: Dataset identifier
        
        Returns:
            ConnectivityProfile or None
        """
        # Query connections from cache
        upstream_df, downstream_df = self._query_connections_local(neuron, dataset)
        
        # If no local data, try NeuPrint as fallback (for non-local datasets)
        if upstream_df.empty and downstream_df.empty:
            if not is_local_connectome_dataset(dataset):
                upstream_df, downstream_df = self._query_connections_neuprint(neuron, dataset)
        
        if upstream_df.empty and downstream_df.empty:
            return None
        
        # Count neurons aggregated
        neurons_aggregated = 1
        if isinstance(neuron, list):
            neurons_aggregated = len(neuron)
        elif isinstance(neuron, str) and not upstream_df.empty:
            if 'neuron_bodyId' in upstream_df.columns:
                neurons_aggregated = upstream_df['neuron_bodyId'].nunique()
            elif not downstream_df.empty and 'neuron_bodyId' in downstream_df.columns:
                neurons_aggregated = downstream_df['neuron_bodyId'].nunique()
        
        # Process connections (simplified - no 2-hop expansion for batch mode)
        (up_partners, up_ranks, up_untyped_count, up_untyped_frac, 
         up_total_weight, up_actual_count, up_unique_types,
         up_type_mapping, up_k_used, up_untyped_bodyids, up_typed_bodyids) = self._process_connections(
            upstream_df, 'upstream', self.config.top_k_bodyid
        )
        
        (down_partners, down_ranks, down_untyped_count, down_untyped_frac,
         down_total_weight, down_actual_count, down_unique_types,
         down_type_mapping, down_k_used, down_untyped_bodyids, down_typed_bodyids) = self._process_connections(
            downstream_df, 'downstream', self.config.top_k_bodyid
        )
        
        # Create profile (without 2-hop data for efficiency)
        profile_type = neuron if isinstance(neuron, str) else None
        profile = ConnectivityProfile(
            neuron_id=neuron,
            dataset=dataset,
            neuron_type=profile_type,
            upstream_partners=up_partners,
            downstream_partners=down_partners,
            upstream_ranks=up_ranks,
            downstream_ranks=down_ranks,
            upstream_top_k=self.config.top_k_bodyid,
            downstream_top_k=self.config.top_k_bodyid,
            total_upstream_weight=up_total_weight,
            total_downstream_weight=down_total_weight,
            num_neurons_aggregated=neurons_aggregated,
            untyped_upstream_count=up_untyped_count,
            untyped_downstream_count=down_untyped_count,
            untyped_upstream_weight_fraction=up_untyped_frac,
            untyped_downstream_weight_fraction=down_untyped_frac,
            actual_upstream_count=up_actual_count,
            actual_downstream_count=down_actual_count,
            unique_types_upstream=up_unique_types,
            unique_types_downstream=down_unique_types,
            partner_type_mapping_upstream=up_type_mapping if up_type_mapping else None,
            partner_type_mapping_downstream=down_type_mapping if down_type_mapping else None,
            top_k_bodyid_used=max(up_k_used, down_k_used),
            top_m_type_target=self.config.top_m_type,
            typed_upstream_bodyids=up_typed_bodyids if up_typed_bodyids else None,
            typed_downstream_bodyids=down_typed_bodyids if down_typed_bodyids else None,
        )

        return self._fill_neuron_type(profile, dataset)

    def get_profiles_for_type_across_datasets(
        self,
        neuron_type: str,
        datasets: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> Dict[str, ConnectivityProfile]:
        """
        Get profiles for a neuron type across multiple datasets.
        
        Useful for cross-dataset verification of type assignments.
        
        Args:
            neuron_type: Type name to query
            datasets: List of datasets (defaults to self.datasets)
            force_refresh: Bypass cache if True
        
        Returns:
            Dict mapping dataset name to ConnectivityProfile
        """
        datasets = datasets or self.datasets
        
        results = {}
        for dataset in datasets:
            try:
                profile = self.get_profile(neuron_type, dataset, force_refresh)
                results[dataset] = profile
            except Exception as e:
                self._log(f"Warning: Failed to extract profile for {neuron_type} in {dataset}: {e}")
        
        return results
    
    def _has_local_table(self, dataset: str) -> bool:
        """Whether a repo-local neuron table exists for *dataset*.

        Used by the offline-first type lookups: a NeuPrint release that
        also ships a local table (e.g. male-cns) resolves locally instead
        of requiring the API.
        """
        safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
        project_root = Path(__file__).parent.parent.parent
        dataset_path = project_root / 'datasets' / safe_name
        for name in (f'{safe_name}_allneurons_neuron_df.parquet',
                     f'{safe_name}_allneurons_neuron_df.csv',
                     f'{safe_name}_neurons.parquet',
                     f'{safe_name}_neurons.csv'):
            if (dataset_path / name).exists():
                return True
        return False

    def get_bodyids_for_type(
        self,
        neuron_type: str,
        dataset: str
    ) -> List[int]:
        """
        Get all bodyIds for a neuron type in a dataset.

        Offline-first: a repo-local neuron table wins when the dataset has
        one (FAFB/BANC always; male-cns and other NeuPrint releases ship
        one too), so type lookups keep working when the NeuPrint API is
        unreachable. On a local miss, FAFB/BANC (complete local datasets)
        return [] — but a partial NeuPrint-release export falls back to
        the API instead of shadowing it.

        Args:
            neuron_type: Type name
            dataset: Dataset identifier

        Returns:
            List of bodyIds
        """
        # Offline-first resolution: prefer a repo-local neuron table when
        # the dataset has one (FAFB/BANC always; male-cns and other
        # NeuPrint releases ship one too).  This keeps type lookups working
        # when the NeuPrint API is unreachable, and the API path below is
        # kept only for datasets with no local table.
        local_df = self._load_local_neuron_frame(dataset)
        has_local_table = local_df is not None and not local_df.empty
        natively_local = is_local_connectome_dataset(dataset)
        if natively_local or has_local_table:
            # Local dataset - resolve the query with the shared prioritized
            # column search used by the connection tabs (bodyId -> type ->
            # instance -> other *Type fields such as cell_type -> taxonomy).
            # The first column with a hit owns the query, so a name stored
            # only in cell_type (e.g. FAFB's circadian_clock) still resolves.
            df = local_df
            body_ids = []
            if df is not None and not df.empty:
                try:
                    from ..neuron_search import resolve_dataframe_query
                except ImportError:
                    from neuron_search import resolve_dataframe_query
                try:
                    body_ids, _info = resolve_dataframe_query(
                        df, neuron_type, search_columns='auto'
                    )
                except Exception as e:
                    self._log(f"Warning: Could not resolve '{neuron_type}' in {dataset}: {e}")

            if body_ids:
                resolved = pd.to_numeric(pd.Series(body_ids), errors='coerce').dropna()
                return resolved.astype(int).tolist()
            if natively_local:
                # FAFB/BANC are complete local datasets — a miss is a real
                # miss; there is nothing better to consult.
                return []
            # A partial local export (e.g. a hemibrain snapshot missing the
            # type entirely) must not shadow a reachable API below — fall
            # through instead of silently returning [].

        # NeuPrint query (also the fallback when a non-native local export
        # missed and the API is configured)
        client = self._get_client_for_dataset(dataset)
        if client is None:
            return []

        try:
            # Import API utilities for Cypher escaping
            try:
                from src.utils.api_utils import escape_cypher_string
            except ImportError:
                escape_cypher_string = _escape_cypher_string_fallback

            escaped_type = escape_cypher_string(neuron_type)
            query = f"""
            MATCH (n:Neuron)
            WHERE n.type = '{escaped_type}'
            RETURN n.bodyId AS bodyId
            """
            result = client.fetch_custom(query)
            return result['bodyId'].astype(int).tolist()
        except Exception as e:
            self._log(f"Warning: Could not get bodyIds for {neuron_type} in {dataset}: {e}")
            return []
    
    def _load_local_neuron_frame(self, dataset: str) -> Optional[pd.DataFrame]:
        """Load (and memoize) the local neuron table for FAFB or BANC.

        The frame is cached per file path and mtime so repeated type
        lookups (one per query item) do not re-read the full table.
        Returns ``None`` when no neuron table exists for the dataset.
        """
        src_dir = Path(__file__).parent.parent
        project_root = src_dir.parent
        datasets_folder = project_root / 'datasets'
        safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
        dataset_path = datasets_folder / safe_name

        neurons_files = [
            dataset_path / f'{safe_name}_allneurons_neuron_df.parquet',
            dataset_path / f'{safe_name}_allneurons_neuron_df.csv',
            dataset_path / f'{safe_name}_neurons.parquet',
            dataset_path / f'{safe_name}_neurons.csv',
            dataset_path / 'neurons.parquet',
            dataset_path / 'neurons.csv',
        ]
        neurons_file = next((p for p in neurons_files if p.exists()), None)
        if neurons_file is None:
            return None

        cache = getattr(self, '_local_neuron_frames', None)
        if cache is None:
            cache = {}
            self._local_neuron_frames = cache
        try:
            mtime = neurons_file.stat().st_mtime
        except OSError:
            mtime = None
        cached = cache.get(str(neurons_file))
        if cached is not None and cached[0] == mtime:
            return cached[1]

        try:
            if str(neurons_file).endswith('.parquet'):
                df = pd.read_parquet(neurons_file)
            else:
                df = pd.read_csv(neurons_file)
        except Exception as e:
            self._log(f"Warning: Could not load neurons from {neurons_file}: {e}")
            return None

        cache[str(neurons_file)] = (mtime, df)
        return df
    
    def list_types(
        self,
        pattern: Optional[str] = None,
        dataset: Optional[str] = None
    ) -> List[str]:
        """
        List neuron types in a dataset, optionally filtered by a regex pattern.
        
        The pattern is matched with ``re.match`` semantics (anchored at the
        start, free at the end) so it accepts the same patterns the UI name
        filter produces ('aMe.*', '.*aMe.*', '.*aMe'). A None/empty pattern
        returns ALL types. The full type list is cached per dataset.
        
        Args:
            pattern: Regex pattern; None/empty returns all types.
            dataset: Dataset identifier (defaults to the first configured).
        
        Returns:
            Sorted list of matching type names.
        """
        dataset = dataset or (self.datasets[0] if self.datasets else None)
        if dataset is None:
            return []
        
        cache = getattr(self, '_all_types_cache', None)
        if cache is None:
            cache = {}
            self._all_types_cache = cache
        
        if dataset not in cache:
            cache[dataset] = self._load_all_types(dataset)
        
        all_types = cache[dataset]
        if not pattern:
            return list(all_types)
        
        try:
            return [t for t in all_types if re.match(pattern, t)]
        except re.error:
            self._log(f"Warning: invalid type pattern '{pattern}' — treating as literal")
            return [t for t in all_types if t == pattern]
    
    def _load_all_types(self, dataset: str) -> List[str]:
        """Load the full sorted list of neuron types for a dataset."""
        if is_local_connectome_dataset(dataset):
            # Local dataset - load neurons file
            src_dir = Path(__file__).parent.parent
            project_root = src_dir.parent
            datasets_folder = project_root / 'datasets'
            safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
            dataset_path = datasets_folder / safe_name
            
            neurons_files = [
                dataset_path / f'{safe_name}_allneurons_neuron_df.parquet',
                dataset_path / f'{safe_name}_allneurons_neuron_df.csv',
                dataset_path / f'{safe_name}_neurons.parquet',
                dataset_path / f'{safe_name}_neurons.csv',
                dataset_path / 'neurons.parquet',
                dataset_path / 'neurons.csv',
            ]
            
            for neurons_file in neurons_files:
                if neurons_file.exists():
                    try:
                        if str(neurons_file).endswith('.parquet'):
                            df = pd.read_parquet(neurons_file)
                        else:
                            df = pd.read_csv(neurons_file)
                        
                        # Find type column
                        type_col = None
                        for col in ['type', 'cell_type', 'celltype']:
                            if col in df.columns:
                                type_col = col
                                break
                        
                        if type_col is not None:
                            types = df[type_col].dropna().astype(str).unique().tolist()
                            return sorted(types)
                    except Exception as e:
                        self._log(f"Warning: Could not load types from {neurons_file}: {e}")
            
            return []
        
        # NeuPrint query: fetch DISTINCT types once, filter in Python so the
        # regex semantics match re.match exactly (Cypher =~ is full-match).
        client = self._get_client_for_dataset(dataset)
        if client is None:
            return []
        
        try:
            query = "MATCH (n:Neuron) WHERE n.type IS NOT NULL RETURN DISTINCT n.type AS type"
            result = client.fetch_custom(query)
            if result is None or result.empty or 'type' not in result.columns:
                return []
            return sorted(str(t) for t in result['type'].dropna().unique().tolist())
        except Exception as e:
            self._log(f"Warning: Could not list types for {dataset}: {e}")
            return []
    
    def get_type_for_bodyid(
        self,
        bodyid: int,
        dataset: str
    ) -> Optional[str]:
        """
        Get the type name for a given bodyId in a dataset.
        
        Args:
            bodyid: The bodyId to look up
            dataset: Dataset identifier
        
        Returns:
            Type name as string, or None if not found
        """
        if is_local_connectome_dataset(dataset):
            # Local dataset - load neurons file
            src_dir = Path(__file__).parent.parent
            project_root = src_dir.parent
            datasets_folder = project_root / 'datasets'
            safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
            dataset_path = datasets_folder / safe_name
            
            neurons_files = [
                dataset_path / f'{safe_name}_allneurons_neuron_df.parquet',
                dataset_path / f'{safe_name}_allneurons_neuron_df.csv',
                dataset_path / f'{safe_name}_neurons.parquet',
                dataset_path / f'{safe_name}_neurons.csv',
                dataset_path / 'neurons.parquet',
                dataset_path / 'neurons.csv',
            ]
            
            for neurons_file in neurons_files:
                if neurons_file.exists():
                    try:
                        if str(neurons_file).endswith('.parquet'):
                            df = pd.read_parquet(neurons_file)
                        else:
                            df = pd.read_csv(neurons_file)
                        
                        # Find bodyId column
                        bid_col = None
                        for col in ['bodyId', 'pt_root_id', 'root_id', 'body_id']:
                            if col in df.columns:
                                bid_col = col
                                break
                        
                        # Find type column
                        type_col = None
                        for col in ['type', 'cell_type', 'celltype']:
                            if col in df.columns:
                                type_col = col
                                break
                        
                        if bid_col and type_col:
                            mask = df[bid_col] == bodyid
                            matches = df.loc[mask, type_col]
                            if not matches.empty:
                                return str(matches.iloc[0])
                    except Exception as e:
                        pass
            
            return None
        
        else:
            # NeuPrint query
            client = self._get_client_for_dataset(dataset)
            if client is None:
                return None
            
            try:
                query = f"""
                MATCH (n:Neuron)
                WHERE n.bodyId = {bodyid}
                RETURN n.type AS type
                """
                result = client.fetch_custom(query)
                if not result.empty and result['type'].iloc[0]:
                    return str(result['type'].iloc[0])
                return None
            except Exception as e:
                return None

    def get_types_for_bodyids(
        self,
        bodyids: List[int],
        dataset: str
    ) -> Dict[int, Optional[str]]:
        """
        Get type names for multiple bodyIds efficiently (batch operation).
        
        This is much more efficient than calling get_type_for_bodyid multiple times
        as it performs a single query/file read for all bodyIds.
        
        Offline-first: a repo-local neuron table wins when the dataset has
        one (FAFB/BANC always; male-cns and other NeuPrint releases ship
        one too), so type labels stay available when the NeuPrint API is
        unreachable.
        
        Args:
            bodyids: List of bodyIds to look up
            dataset: Dataset identifier
        
        Returns:
            Dictionary mapping bodyId -> type name (or None if not found)
        """
        if not bodyids:
            return {}

        # the untyped-sentinel vocabulary, shared by both lookup paths below
        try:
            from utils.label_utils import UntypedLabelPolicy
        except ImportError:  # direct src/ execution
            from src.utils.label_utils import UntypedLabelPolicy

        result_map = {}
        # Offline-first: prefer a repo-local neuron table when the dataset
        # has one (FAFB/BANC always; male-cns and other NeuPrint releases
        # ship one too).  Keeps type labels available when the NeuPrint API
        # is unreachable; the API path below is for datasets with no table.
        if is_local_connectome_dataset(dataset) or self._has_local_table(
                dataset):
            # Local dataset - load neurons file once
            src_dir = Path(__file__).parent.parent
            project_root = src_dir.parent
            datasets_folder = project_root / 'datasets'
            safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
            dataset_path = datasets_folder / safe_name
            
            neurons_files = [
                dataset_path / f'{safe_name}_allneurons_neuron_df.parquet',
                dataset_path / f'{safe_name}_allneurons_neuron_df.csv',
                dataset_path / f'{safe_name}_neurons.parquet',
                dataset_path / f'{safe_name}_neurons.csv',
                dataset_path / 'neurons.parquet',
                dataset_path / 'neurons.csv',
            ]
            
            for neurons_file in neurons_files:
                if neurons_file.exists():
                    try:
                        if str(neurons_file).endswith('.parquet'):
                            df = pd.read_parquet(neurons_file)
                        else:
                            df = pd.read_csv(neurons_file)
                        
                        # Find bodyId column
                        bid_col = None
                        for col in ['bodyId', 'pt_root_id', 'root_id', 'body_id']:
                            if col in df.columns:
                                bid_col = col
                                break
                        
                        # Find type column
                        type_col = None
                        for col in ['type', 'cell_type', 'celltype']:
                            if col in df.columns:
                                type_col = col
                                break
                        
                        if bid_col and type_col:
                            # Create lookup dictionary
                            df[bid_col] = df[bid_col].astype(str)
                            type_lookup = df.set_index(bid_col)[type_col].to_dict()

                            # Map all bodyIds.  A missing annotation reaches
                            # the table as pandas NaN, whose str() is the
                            # literal 'nan' — return None for it, matching
                            # the docstring contract (untyped consumers'
                            # `or` guards cannot catch the truthy string).
                            for bid in bodyids:
                                bid_str = str(bid)
                                if bid_str in type_lookup:
                                    label = str(type_lookup[bid_str])
                                    result_map[bid] = (
                                        None if UntypedLabelPolicy.is_untyped(
                                            label) else label)
                                else:
                                    result_map[bid] = None
                            return result_map
                    except Exception as e:
                        # Round-7 K4a: a read that RAISES (lock, truncated
                        # file, dtype surprise) used to vanish here, so a
                        # present-but-unreadable table looked identical to
                        # "no table" and every caller saw an all-None
                        # lookup with no trace of why.
                        self._log(f"Warning: could not read neuron table "
                                  f"{neurons_file} for type lookup: {e}")

            # File not found - return all None
            self._log(f"type lookup: no readable local neuron table for "
                      f"{dataset} under {dataset_path} — returning None "
                      f"for {len(bodyids)} bodyIds")
            for bid in bodyids:
                result_map[bid] = None
            return result_map
        
        else:
            # NeuPrint query - batch query
            client = self._get_client_for_dataset(dataset)
            if client is None:
                for bid in bodyids:
                    result_map[bid] = None
                return result_map
            
            try:
                # Query all bodyIds at once
                bodyids_str = ', '.join(str(b) for b in bodyids)
                query = f"""
                MATCH (n:Neuron)
                WHERE n.bodyId IN [{bodyids_str}]
                RETURN n.bodyId AS bodyId, n.type AS type
                """
                df = client.fetch_custom(query)
                
                # Build lookup from results
                if not df.empty:
                    for _, row in df.iterrows():
                        bid = int(row['bodyId'])
                        ntype = row['type']
                        # a missing type property reaches the DataFrame as
                        # pandas NaN, which is TRUTHY — guard it through the
                        # same policy as the local-table path above, or it
                        # stringifies to the literal 'nan'
                        result_map[bid] = (
                            None if not str(ntype or '').strip()
                            or UntypedLabelPolicy.is_untyped(str(ntype))
                            else str(ntype))
                
                # Fill in any missing bodyIds with None
                for bid in bodyids:
                    if bid not in result_map:
                        result_map[bid] = None
                
                return result_map
            except Exception as e:
                # On error, return all None
                for bid in bodyids:
                    result_map[bid] = None
                return result_map

    def get_types_for_label(
        self,
        label: str,
        dataset: str,
        search_columns: Optional[List[str]] = None,
    ) -> Dict[str, List[int]]:
        """Resolve a coarse taxonomy label into the real neuron types it maps to.

        ``label`` may be a cell class, cell type, subclass or another taxonomy
        value rather than a concrete neuron type.  The local neuron table is
        searched across the identity/taxonomy columns; the matching rows are
        grouped by their fine-grained ``type`` column so a query such as a cell
        class that covers many real types expands into one entry per type
        (mirroring the network tab's multi-column query resolution).

        Returns ``{real_type: [bodyIds]}``; empty when nothing matches or the
        dataset has no local table (NeuPrint falls back to the single-column
        lookup in ``get_bodyids_for_type``).
        """
        if not is_local_connectome_dataset(dataset):
            return {}

        df = self._load_local_neuron_frame(dataset)
        if df is None or df.empty:
            return {}

        bid_col = None
        for col in ['bodyId', 'pt_root_id', 'root_id', 'body_id']:
            if col in df.columns:
                bid_col = col
                break
        if bid_col is None:
            return {}

        # The fine-grained type column the returned rows are grouped by.
        type_col = None
        for col in ['type', 'Type', 'cellType', 'cell_type']:
            if col in df.columns:
                type_col = col
                break
        if type_col is None:
            return {}

        taxonomy_cols = search_columns or [
            'type', 'Type', 'cellType', 'cell_type', 'celltype',
            'cell_class', 'sub_class', 'subclass',
            'super_class', 'superclass', 'class',
            'flywireType', 'hemibrainType', 'mancType',
        ]
        present = [c for c in taxonomy_cols if c in df.columns]
        if not present:
            return {}

        target = str(label).strip()
        mask = pd.Series(False, index=df.index)
        for col in present:
            try:
                col_values = df[col].astype(str).str.strip()
            except Exception:
                continue
            mask = mask | (col_values == target)
        if not mask.any():
            return {}

        matched = df.loc[mask]
        result: Dict[str, List[int]] = {}
        for _, row in matched.iterrows():
            raw_type = row[type_col]
            tname = str(raw_type).strip() if pd.notna(raw_type) else ''
            if not tname:
                continue
            raw_bid = row[bid_col]
            try:
                bid = int(raw_bid)
            except (TypeError, ValueError):
                bid = raw_bid
            result.setdefault(tname, []).append(bid)
        return result

    def get_type_profile_from_bodyids(
        self,
        neuron_type: str,
        dataset: str,
        force_refresh: bool = False
    ) -> ConnectivityProfile:
        """
        Get type-level profile by fetching individual bodyId profiles and aggregating.
        
        This is the recommended method for type-level profiles as it:
        1. Fetches each bodyId's profile separately (ensuring bodyId-level caching)
        2. Aggregates using weight summation (proper for actual synapse weights)
        3. Derives ranks from aggregated weights
        
        Args:
            neuron_type: Type name
            dataset: Dataset identifier
            force_refresh: Bypass cache if True
        
        Returns:
            Aggregated ConnectivityProfile for the type
        
        Example:
            >>> type_profile = profiler.get_type_profile_from_bodyids('aMe12', 'hemibrain:v1.2.1')
            >>> print(type_profile.num_neurons_aggregated)  # Number of bodyIds aggregated
        """
        # Get all bodyIds for this type
        bodyids = self.get_bodyids_for_type(neuron_type, dataset)
        
        if not bodyids:
            self._log(f"No bodyIds found for {neuron_type} in {dataset}")
            # Return empty profile
            return ConnectivityProfile(
                neuron_id=neuron_type,
                dataset=dataset,
                upstream_partners={},
                downstream_partners={},
                upstream_ranks={},
                downstream_ranks={},
            )
        
        self._log(f"Fetching profiles for {len(bodyids)} bodyIds of {neuron_type} in {dataset}")
        
        # Fetch individual profiles
        bodyid_profiles = []
        for bid in bodyids:
            try:
                profile = self.get_profile(bid, dataset, force_refresh)
                if profile is not None:
                    bodyid_profiles.append(profile)
            except Exception as e:
                self._log(f"Warning: Could not get profile for bodyId {bid}: {e}")
        
        if not bodyid_profiles:
            self._log(f"No profiles extracted for {neuron_type}")
            return ConnectivityProfile(
                neuron_id=neuron_type,
                dataset=dataset,
                upstream_partners={},
                downstream_partners={},
                upstream_ranks={},
                downstream_ranks={},
            )
        
        # Aggregate using the static method
        return ConnectivityProfile.aggregate_bodyid_profiles(
            bodyid_profiles, neuron_type, dataset
        )

    def get_vectorized_profiles(
        self,
        neuron_types: List[str],
        datasets: Optional[List[str]] = None,
        direction: str = 'both',
        vector_type: str = 'weight',
        normalize: bool = True,
        min_occurrence: int = 1
    ) -> Dict[str, Any]:
        """
        Get vectorized profiles for multiple neurons across datasets.
        
        This method provides efficient batch vectorization with a shared
        vocabulary, suitable for batch comparisons and clustering.
        
        Args:
            neuron_types: List of neuron types to vectorize
            datasets: List of datasets (default: all configured datasets)
            direction: 'upstream', 'downstream', or 'both'
            vector_type: 'weight' for normalized weights, 'rank' for ranks
            normalize: L2-normalize weight vectors (ignored for rank)
            min_occurrence: Minimum profiles a partner must appear in
        
        Returns:
            Dict with keys:
            - 'matrix': (n_profiles, n_features) numpy array
            - 'profile_ids': List of profile IDs in row order
            - 'vocabulary': List of partner types in column order
            - 'profiles': Dict of {profile_id: ConnectivityProfile}
        
        Example:
            >>> result = profiler.get_vectorized_profiles(
            ...     ['Mi1', 'Tm1', 'L1'],
            ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
            ...     vector_type='weight'
            ... )
            >>> result['matrix'].shape
            (6, 25)  # 3 types x 2 datasets = 6 profiles, 25 partners
        """
        datasets = datasets or self.datasets
        
        # Collect all profiles
        profiles = {}
        for neuron_type in neuron_types:
            for dataset in datasets:
                try:
                    profile = self.get_profile(neuron_type, dataset)
                    profile_id = f"{dataset}:{neuron_type}"
                    profiles[profile_id] = profile
                except Exception as e:
                    self._log(f"Warning: Could not get profile for {neuron_type} in {dataset}: {e}")
        
        if not profiles:
            return {
                'matrix': np.empty((0, 0)),
                'profile_ids': [],
                'vocabulary': [],
                'profiles': {}
            }
        
        # Build shared vocabulary
        vocabulary = ConnectivityProfile.build_shared_vocabulary(
            profiles, direction, min_occurrence
        )
        
        # Build matrix
        if vector_type == 'rank':
            matrix, profile_ids, vocab = ConnectivityProfile.profiles_to_rank_matrix(
                profiles, vocabulary, direction
            )
        else:  # weight
            matrix, profile_ids, vocab = ConnectivityProfile.profiles_to_weight_matrix(
                profiles, vocabulary, direction, normalize
            )
        
        return {
            'matrix': matrix,
            'profile_ids': profile_ids,
            'vocabulary': vocab,
            'profiles': profiles
        }
    
    def save_vectorized_profiles(
        self,
        output_path: Union[str, Path],
        neuron_types: List[str],
        datasets: Optional[List[str]] = None,
        direction: str = 'both',
        vector_type: str = 'weight'
    ) -> Path:
        """
        Save vectorized profiles to a numpy archive for fast loading.
        
        Args:
            output_path: Path to save the .npz file
            neuron_types: List of neuron types to vectorize
            datasets: List of datasets (default: all configured datasets)
            direction: 'upstream', 'downstream', or 'both'
            vector_type: 'weight' or 'rank'
        
        Returns:
            Path to the saved file
        
        Example:
            >>> path = profiler.save_vectorized_profiles(
            ...     'cache/vectorized_profiles.npz',
            ...     ['Mi1', 'Tm1', 'L1']
            ... )
        """
        result = self.get_vectorized_profiles(
            neuron_types, datasets, direction, vector_type
        )
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            output_path,
            matrix=result['matrix'],
            profile_ids=result['profile_ids'],
            vocabulary=result['vocabulary'],
            vector_type=vector_type,
            direction=direction
        )
        
        self._log(f"Saved vectorized profiles to {output_path}")
        return output_path
    
    @staticmethod
    def load_vectorized_profiles(input_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load vectorized profiles from a numpy archive.
        
        Args:
            input_path: Path to the .npz file
        
        Returns:
            Dict with matrix, profile_ids, vocabulary, vector_type, direction
        """
        data = np.load(input_path, allow_pickle=True)
        return {
            'matrix': data['matrix'],
            'profile_ids': list(data['profile_ids']),
            'vocabulary': list(data['vocabulary']),
            'vector_type': str(data['vector_type']),
            'direction': str(data['direction'])
        }
    
    def clear_cache(self, dataset: Optional[str] = None):
        """
        Clear cached profiles.
        
        Args:
            dataset: Clear only this dataset's cache (None = clear all)
        """
        # Clear memory cache
        if dataset:
            keys_to_remove = [k for k in self._memory_cache if k[1] == dataset]
            for key in keys_to_remove:
                del self._memory_cache[key]
        else:
            self._memory_cache.clear()
        
        # Clear disk cache
        if self.config.use_cache:
            if dataset:
                safe_dataset = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
                dataset_cache_dir = self.cache_dir / safe_dataset
                if dataset_cache_dir.exists():
                    import shutil
                    shutil.rmtree(dataset_cache_dir)
                    self._log(f"Cleared disk cache for {dataset}")
            else:
                if self.cache_dir.exists():
                    import shutil
                    shutil.rmtree(self.cache_dir)
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    self._log("Cleared all disk cache")
    
    def get_available_types(self, dataset: str) -> Optional[List[str]]:
        """
        Get list of all neuron types available in a dataset.
        
        This method queries the dataset to find all unique neuron types
        that can be used with get_profile().
        
        Args:
            dataset: Dataset identifier (e.g., 'hemibrain_v1_2_1', 'flywire_FAFB_v783')
        
        Returns:
            List of neuron type names, or None if dataset not found
        
        Example:
            >>> profiler = ConnectivityProfiler(datasets=['hemibrain_v1_2_1'])
            >>> types = profiler.get_available_types('hemibrain_v1_2_1')
            >>> print(f"Found {len(types)} types: {types[:10]}...")
        """
        dataset_lower = dataset.lower()

        # Optic-lobe/male-cns retain their legacy local-table behavior; FAFB
        # and BANC are resolved by the shared exact-release predicate.
        is_local = is_local_connectome_dataset(dataset) or any(
            x in dataset_lower for x in ['optic', 'male'])
        
        if is_local:
            return self._get_local_dataset_types(dataset)
        else:
            return self._get_neuprint_dataset_types(dataset)
    
    def _get_local_dataset_types(self, dataset: str) -> Optional[List[str]]:
        """Get types from local dataset files."""
        # Sanitize dataset name
        safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
        
        # Find dataset folder
        src_dir = Path(__file__).parent.parent
        project_root = src_dir.parent
        
        # Try multiple possible paths
        possible_paths = [
            project_root / 'datasets' / safe_name,
            project_root / 'datasets' / dataset,
            project_root / 'cache' / safe_name,
        ]
        
        dataset_path = None
        for path in possible_paths:
            if path.exists():
                dataset_path = path
                break
        
        if dataset_path is None:
            self._log(f"Dataset folder not found for {dataset}")
            return None
        
        # Try to load neuron_df to get types
        neuron_files = [
            dataset_path / f'{safe_name}_allneurons_neuron_df.parquet',
            dataset_path / f'{safe_name}_allneurons_neuron_df.csv',
            dataset_path / f'{safe_name}_neuron_df.parquet',
            dataset_path / f'{safe_name}_neuron_df.csv',
            dataset_path / 'neuron_df.parquet',
            dataset_path / 'neuron_df.csv',
        ]
        
        neuron_df = None
        for neuron_file in neuron_files:
            if neuron_file.exists():
                try:
                    if str(neuron_file).endswith('.parquet'):
                        neuron_df = pd.read_parquet(neuron_file)
                    else:
                        neuron_df = pd.read_csv(neuron_file)
                    self._log(f"Loaded types from {neuron_file.name}")
                    break
                except Exception as e:
                    self._log(f"Warning: Could not load {neuron_file}: {e}")
        
        if neuron_df is None or neuron_df.empty:
            self._log(f"No neuron_df found for {dataset}")
            return None
        
        # Find type column
        type_col = None
        for col in ['type', 'Type', 'cellType', 'cell_type']:
            if col in neuron_df.columns:
                type_col = col
                break
        
        if type_col is None:
            self._log(f"No type column found in neuron_df for {dataset}")
            return None
        
        # Get unique non-null types
        types = neuron_df[type_col].dropna().unique().tolist()
        types = [str(t) for t in types if t and str(t).strip()]
        types = sorted(set(types))
        
        self._log(f"Found {len(types)} unique types in {dataset}")
        return types
    
    def _get_neuprint_dataset_types(self, dataset: str) -> Optional[List[str]]:
        """Get types from NeuPrint dataset."""
        try:
            client = self._get_client_for_dataset(dataset)
            if client is None:
                return None
            
            from neuprint import fetch_neurons
            
            # Fetch all neurons with type info
            # Note: This can be slow for large datasets
            neurons, _ = fetch_neurons(None)
            
            if neurons is None or neurons.empty:
                return None
            
            # Get type column
            type_col = None
            for col in ['type', 'Type', 'cellType']:
                if col in neurons.columns:
                    type_col = col
                    break
            
            if type_col is None:
                return None
            
            types = neurons[type_col].dropna().unique().tolist()
            types = [str(t) for t in types if t and str(t).strip()]
            types = sorted(set(types))
            
            self._log(f"Found {len(types)} unique types in {dataset}")
            return types
            
        except Exception as e:
            self._log(f"Error getting types from NeuPrint: {e}")
            return None
    
    # =========================================================================
    # Round 7: Cache Building and Homolog Finding Methods
    # =========================================================================
    
    def build_connectivity_profile_cache(
        self,
        dataset: str,
        neuron_types: Optional[List[str]] = None,
        top_k_bodyid: int = 10,
        top_m_type: int = 5,
        expand_untyped_2hop: bool = True,
        force_refresh: bool = False,
        max_neurons: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, ConnectivityProfile]:
        """
        Build connectivity profile cache for all neurons in a dataset.
        
        This method efficiently pre-builds profiles for all (or selected) neuron
        types in a dataset, storing them in the cache for fast lookup during
        homolog finding operations.
        
        Args:
            dataset: Dataset identifier (e.g., 'hemibrain_v1_2_1', 'flywire_FAFB_v783')
            neuron_types: Specific neuron types to cache. If None, caches all types.
            top_k_bodyid: Number of top partners by bodyId to store
            top_m_type: Minimum unique types to ensure via dynamic expansion
            expand_untyped_2hop: Whether to fetch 2-hop profiles for untyped partners
            force_refresh: If True, rebuild cache even if profiles exist
            max_neurons: Maximum number of neurons to cache (for testing)
            progress_callback: Optional callback(current, total, type_name) for progress
        
        Returns:
            Dict mapping neuron_type to ConnectivityProfile
        
        Example:
            >>> profiler = ConnectivityProfiler()
            >>> profiles = profiler.build_connectivity_profile_cache(
            ...     'hemibrain_v1_2_1',
            ...     top_k_bodyid=10,
            ...     expand_untyped_2hop=True
            ... )
            >>> print(f"Cached {len(profiles)} profiles")
        """
        self._log(f"Building connectivity profile cache for {dataset}...")
        
        # Update config for this operation (ProfilerConfig has no
        # cache_dir/verbose fields — passing them raised TypeError on
        # every call of this documented entry point)
        original_config = self.config
        self.config = ProfilerConfig(
            top_k_bodyid=top_k_bodyid,
            top_m_type=top_m_type,
            expand_untyped_2hop=expand_untyped_2hop,
            top_k_2hop=self.config.top_k_2hop,
            use_bodyid_for_intra=self.config.use_bodyid_for_intra,
            use_cache=True,
        )

        # Get neuron types if not specified
        if neuron_types is None:
            neuron_types = self.list_types(dataset=dataset)
            if neuron_types is None:
                self._log(f"Could not retrieve types for {dataset}")
                self.config = original_config
                return {}
        
        # Apply max_neurons limit if specified
        if max_neurons is not None and len(neuron_types) > max_neurons:
            neuron_types = neuron_types[:max_neurons]
            self._log(f"Limited to first {max_neurons} neuron types")
        
        total = len(neuron_types)
        self._log(f"Building profiles for {total} neuron types...")
        
        profiles = {}
        failed = []
        batch_size = 50  # Save to parquet every N profiles
        batch_profiles = {}
        
        # Temporarily disable per-profile cache saving (we'll batch save)
        save_per_profile = self.config.use_cache
        
        for i, neuron_type in enumerate(neuron_types):
            try:
                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, total, neuron_type)
                
                # Log progress
                if (i + 1) % 50 == 0 or i == 0:
                    self._log(f"Progress: {i + 1}/{total} ({100*(i+1)/total:.1f}%)")
                
                # Get profile (check cache first unless force_refresh)
                if not force_refresh:
                    cached = self._load_from_cache(neuron_type, dataset)
                    if cached is not None:
                        profiles[neuron_type] = cached
                        continue
                
                # Temporarily disable per-profile cache save; restored in
                # finally — a swallowed profile failure used to leave
                # use_cache False for every remaining profile AND the final
                # batch flush, silently writing nothing to disk.
                self.config.use_cache = False
                try:
                    profile = self.get_profile(
                        neuron_type, dataset, force_refresh=True)
                finally:
                    self.config.use_cache = save_per_profile

                profiles[neuron_type] = profile
                batch_profiles[neuron_type] = profile
                
                # Batch save every N profiles
                if len(batch_profiles) >= batch_size:
                    self._save_profiles_to_cache_batch(batch_profiles, dataset)
                    batch_profiles = {}
                
            except Exception as e:
                self._log(f"Warning: Failed to build profile for {neuron_type}: {e}")
                failed.append(neuron_type)
        
        # Save remaining batch
        if batch_profiles:
            self._save_profiles_to_cache_batch(batch_profiles, dataset)
        
        # Restore original config
        self.config = original_config
        
        self._log(f"Cache build complete: {len(profiles)} profiles built, "
                  f"{len(failed)} failed")
        
        if failed:
            self._log(f"Failed types: {failed[:10]}{'...' if len(failed) > 10 else ''}")
        
        return profiles
    
    def find_homologs_loose(
        self,
        query_type: str,
        query_dataset: str,
        target_dataset: str,
        direction: str = 'both',
        top_n: int = 10,
        use_ranks: bool = True
    ) -> pd.DataFrame:
        """
        Find potential homologs using loose matching (Jaccard similarity).
        
        This mode compares the typed connectivity profiles at the type level,
        computing both intersection-based and union-based Jaccard similarity.
        
        Similarity metrics:
        - Jaccard Intersection: |A ∩ B| / |A ∩ B| = overlap quality
        - Jaccard Union: |A ∩ B| / |A ∪ B| = standard Jaccard
        - Weighted variants: Use connection weights instead of binary sets
        
        Args:
            query_type: The neuron type to find homologs for
            query_dataset: Dataset containing the query neuron
            target_dataset: Dataset to search for homologs
            direction: 'upstream', 'downstream', or 'both'
            top_n: Number of top candidates to return
            use_ranks: If True, compare ranks (ordinal); if False, compare weights
        
        Returns:
            DataFrame with columns:
            - target_type: Candidate homolog type
            - jaccard_intersection: Intersection-based Jaccard
            - jaccard_union: Union-based Jaccard
            - weighted_jaccard: Weight-adjusted Jaccard
            - common_partners: Number of shared partner types
            - direction: Which direction was compared
        
        Example:
            >>> results = profiler.find_homologs_loose(
            ...     'aMe12', 'hemibrain_v1_2_1', 
            ...     'flywire_FAFB_v783',
            ...     direction='both'
            ... )
            >>> print(results.head())
        """
        self._log(f"Finding loose homologs for {query_type} from {query_dataset} "
                  f"in {target_dataset}...")
        
        # Get query profile
        query_profile = self.get_profile(query_type, query_dataset)
        
        # Get all types in target dataset
        target_types = self.list_types(dataset=target_dataset)
        if not target_types:
            self._log(f"No types found in {target_dataset}")
            return pd.DataFrame()
        
        results = []
        
        for target_type in target_types:
            try:
                target_profile = self.get_profile(target_type, target_dataset)
                
                # Compute similarities
                scores = self._compute_jaccard_similarity(
                    query_profile, target_profile, direction, use_ranks
                )
                
                if scores:
                    for score_dict in scores:
                        score_dict['target_type'] = target_type
                        results.append(score_dict)
                        
            except Exception as e:
                # Skip types that fail
                continue
        
        if not results:
            return pd.DataFrame()
        
        # Create results DataFrame
        df = pd.DataFrame(results)
        
        # Sort by combined score (average of intersection and union Jaccard)
        if 'jaccard_union' in df.columns:
            df['combined_score'] = (df['jaccard_intersection'] + df['jaccard_union']) / 2
            df = df.sort_values('combined_score', ascending=False)
        
        return df.head(top_n)
    
    def _compute_jaccard_similarity(
        self,
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str,
        use_ranks: bool
    ) -> List[Dict]:
        """
        Compute Jaccard similarity between two profiles.
        
        Returns list of dicts with scores for each direction.
        """
        results = []
        
        directions = []
        if direction in ['both', 'upstream']:
            directions.append('upstream')
        if direction in ['both', 'downstream']:
            directions.append('downstream')
        
        for dir_name in directions:
            if use_ranks:
                partners_a = getattr(profile_a, f'{dir_name}_ranks', {}) or {}
                partners_b = getattr(profile_b, f'{dir_name}_ranks', {}) or {}
            else:
                partners_a = getattr(profile_a, f'{dir_name}_partners', {}) or {}
                partners_b = getattr(profile_b, f'{dir_name}_partners', {}) or {}
            
            if not partners_a or not partners_b:
                continue
            
            # Get partner type sets
            set_a = set(partners_a.keys())
            set_b = set(partners_b.keys())
            
            intersection = set_a & set_b
            union = set_a | set_b
            
            if not union:
                continue
            
            # Jaccard metrics
            jaccard_intersection = len(intersection) / len(intersection) if intersection else 0.0
            jaccard_union = len(intersection) / len(union) if union else 0.0
            
            # Weighted Jaccard: sum of min weights for intersection / sum of max weights for union
            weighted_intersection = 0.0
            weighted_union = 0.0
            
            for partner in union:
                val_a = partners_a.get(partner, 0.0)
                val_b = partners_b.get(partner, 0.0)
                weighted_intersection += min(val_a, val_b)
                weighted_union += max(val_a, val_b)
            
            weighted_jaccard = weighted_intersection / weighted_union if weighted_union > 0 else 0.0
            
            results.append({
                'direction': dir_name,
                'jaccard_intersection': jaccard_intersection,
                'jaccard_union': jaccard_union,
                'weighted_jaccard': weighted_jaccard,
                'common_partners': len(intersection),
                'total_partners_query': len(set_a),
                'total_partners_target': len(set_b),
            })
        
        return results
    
    def find_homologs_strict(
        self,
        query_type: str,
        query_dataset: str,
        target_dataset: str,
        direction: str = 'both',
        top_n: int = 10,
        min_common_partners: int = 3
    ) -> pd.DataFrame:
        """
        Find potential homologs using strict matching (per-bodyId profiles).
        
        This mode compares profiles at the individual neuron level, computing:
        1. Rank correlation between matching partner types
        2. Per-bodyId Jaccard similarities
        3. 2-hop profile matching for untyped partners (if available)
        
        This is more stringent than loose matching and works well when:
        - You want to match individual neurons, not just types
        - The datasets have good type coverage
        - 2-hop expansion data is available
        
        Args:
            query_type: The neuron type to find homologs for
            query_dataset: Dataset containing the query neuron
            target_dataset: Dataset to search for homologs
            direction: 'upstream', 'downstream', or 'both'
            top_n: Number of top candidates to return
            min_common_partners: Minimum shared partners required
        
        Returns:
            DataFrame with columns:
            - target_type: Candidate homolog type
            - rank_correlation: Spearman correlation of partner ranks
            - jaccard_typed: Jaccard on typed partners only
            - jaccard_2hop: Jaccard including 2-hop expanded partners
            - combined_score: Weighted combination of metrics
            - direction: Which direction was compared
        
        Example:
            >>> results = profiler.find_homologs_strict(
            ...     'aMe12', 'hemibrain_v1_2_1',
            ...     'flywire_FAFB_v783',
            ...     direction='both',
            ...     min_common_partners=3
            ... )
        """
        self._log(f"Finding strict homologs for {query_type} from {query_dataset} "
                  f"in {target_dataset}...")
        
        from scipy.stats import spearmanr
        
        # Get query profile
        query_profile = self.get_profile(query_type, query_dataset)
        
        # Get all types in target dataset
        target_types = self.list_types(dataset=target_dataset)
        if not target_types:
            self._log(f"No types found in {target_dataset}")
            return pd.DataFrame()
        
        results = []
        
        for target_type in target_types:
            try:
                target_profile = self.get_profile(target_type, target_dataset)
                
                # Compute strict similarity
                scores = self._compute_strict_similarity(
                    query_profile, target_profile, direction, min_common_partners
                )
                
                if scores:
                    for score_dict in scores:
                        score_dict['target_type'] = target_type
                        results.append(score_dict)
                        
            except Exception as e:
                continue
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Sort by combined score
        if 'combined_score' in df.columns:
            df = df.sort_values('combined_score', ascending=False)
        elif 'rank_correlation' in df.columns:
            df = df.sort_values('rank_correlation', ascending=False)
        
        return df.head(top_n)
    
    def _compute_strict_similarity(
        self,
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str,
        min_common: int
    ) -> List[Dict]:
        """
        Compute strict similarity using rank correlation and 2-hop matching.
        """
        from scipy.stats import spearmanr
        
        results = []
        
        directions = []
        if direction in ['both', 'upstream']:
            directions.append('upstream')
        if direction in ['both', 'downstream']:
            directions.append('downstream')
        
        for dir_name in directions:
            # Get typed partner ranks
            ranks_a = getattr(profile_a, f'{dir_name}_ranks', {}) or {}
            ranks_b = getattr(profile_b, f'{dir_name}_ranks', {}) or {}
            
            if not ranks_a or not ranks_b:
                continue
            
            # Find common partners
            set_a = set(ranks_a.keys())
            set_b = set(ranks_b.keys())
            common = set_a & set_b
            
            if len(common) < min_common:
                continue
            
            # Compute rank correlation on common partners
            ranks_list_a = [ranks_a[p] for p in common]
            ranks_list_b = [ranks_b[p] for p in common]
            
            try:
                corr, p_val = spearmanr(ranks_list_a, ranks_list_b)
                if np.isnan(corr):
                    corr = 0.0
            except Exception:
                corr = 0.0
                p_val = 1.0
            
            # Jaccard on typed partners
            union = set_a | set_b
            jaccard_typed = len(common) / len(union) if union else 0.0
            
            # 2-hop Jaccard (if available)
            jaccard_2hop = 0.0
            twohhop_a = getattr(profile_a, f'untyped_{dir_name}_2hop', {}) or {}
            twohop_b = getattr(profile_b, f'untyped_{dir_name}_2hop', {}) or {}
            
            if twohhop_a and twohop_b:
                # Flatten 2-hop partners
                all_2hop_a = set()
                all_2hop_b = set()
                
                for bid, partners in twohhop_a.items():
                    if isinstance(partners, dict):
                        all_2hop_a.update(partners.keys())
                
                for bid, partners in twohop_b.items():
                    if isinstance(partners, dict):
                        all_2hop_b.update(partners.keys())
                
                if all_2hop_a and all_2hop_b:
                    common_2hop = all_2hop_a & all_2hop_b
                    union_2hop = all_2hop_a | all_2hop_b
                    jaccard_2hop = len(common_2hop) / len(union_2hop) if union_2hop else 0.0
            
            # Combined score: weighted average
            # Weight rank correlation highest, then typed Jaccard, then 2-hop
            combined_score = 0.5 * max(0, corr) + 0.3 * jaccard_typed + 0.2 * jaccard_2hop
            
            results.append({
                'direction': dir_name,
                'rank_correlation': corr,
                'rank_correlation_pval': p_val,
                'jaccard_typed': jaccard_typed,
                'jaccard_2hop': jaccard_2hop,
                'combined_score': combined_score,
                'common_partners': len(common),
                'total_partners_query': len(set_a),
                'total_partners_target': len(set_b),
            })
        
        return results
    
    def get_hybrid_profile_vector(
        self,
        neuron_type: str,
        dataset: str,
        direction: str = 'both'
    ) -> Dict[str, Dict]:
        """
        Get hybrid 1-hop/2-hop profile as a dict (not vectorized).
        
        For typed 1-hop partners: stores the partner type and weight
        For untyped 1-hop partners: stores bodyId with 2-hop type profile
        
        This is useful for intra-dataset comparisons where you want to
        preserve bodyId information, or for cross-dataset comparisons
        where 2-hop profiles help disambiguate untyped partners.
        
        Args:
            neuron_type: Neuron type to get profile for
            dataset: Dataset identifier
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Dict with structure:
            {
                'upstream': {
                    'typed': {partner_type: weight, ...},
                    'typed_ranks': {partner_type: rank, ...},
                    'untyped_2hop': {
                        bodyId: {2hop_type: weight, ...},
                        ...
                    }
                },
                'downstream': {...}
            }
        
        Example:
            >>> hybrid = profiler.get_hybrid_profile_vector('aMe12', 'hemibrain_v1_2_1')
            >>> print(hybrid['upstream']['typed'])
        """
        profile = self.get_profile(neuron_type, dataset)
        
        result = {}
        
        directions = []
        if direction in ['both', 'upstream']:
            directions.append('upstream')
        if direction in ['both', 'downstream']:
            directions.append('downstream')
        
        for dir_name in directions:
            typed_partners = getattr(profile, f'{dir_name}_partners', {}) or {}
            typed_ranks = getattr(profile, f'{dir_name}_ranks', {}) or {}
            untyped_2hop = getattr(profile, f'untyped_{dir_name}_2hop', {}) or {}
            
            result[dir_name] = {
                'typed': typed_partners,
                'typed_ranks': typed_ranks,
                'untyped_2hop': untyped_2hop,
            }
        
        return result
