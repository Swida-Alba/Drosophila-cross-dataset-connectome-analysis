"""
Profile Comparison Module

This module provides tools for comparing connectivity profiles across neurons
and datasets using various similarity metrics.

Main components:
- ProfileComparator: Class with methods for computing similarity scores
- ComparisonResult: Dataclass storing comparison results with all metrics

Supported Metrics:
- Jaccard similarity: Set-based overlap of partner types
- Cosine similarity: Weight vector similarity
- Rank correlation: Spearman correlation of partner rankings
- Overlap fraction: Asymmetric overlap measures
- Combined score: Weighted combination of all metrics

Example:
    >>> from src.comparison import ProfileComparator
    >>> 
    >>> similarity = ProfileComparator.weighted_cosine_similarity(
    ...     profile_a, profile_b, direction='both'
    ... )
    >>> print(f"Cosine similarity: {similarity:.3f}")
"""

from dataclasses import dataclass
from itertools import combinations
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Any, TYPE_CHECKING
from datetime import datetime
import os
import time
import warnings
import math
import heapq
import json
import re
try:
    from ..utils.naming_utils import dataset_abbrev, make_unique_dataset_labels
except ImportError:
    try:
        from utils.naming_utils import dataset_abbrev, make_unique_dataset_labels
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

        def make_unique_dataset_labels(datasets, labels=None):
            return [dataset_abbrev(d) for d in (datasets or [])]

import numpy as np
import pandas as pd
from scipy import stats

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .connectivity_profiler import ConnectivityProfile, ConnectivityProfiler, ProfilerConfig, progress_bars_disabled
from .cross_dataset_type_mapper import CrossDatasetTypeMapper, get_type_mapper

try:
    from ..utils.naming_utils import canonical_dataset_name
except ImportError:  # pragma: no cover - direct package imports
    from utils.naming_utils import canonical_dataset_name
try:
    from ..visualization_options import default_analysis_skeleton_mesh_simplification
except ImportError:
    from visualization_options import default_analysis_skeleton_mesh_simplification

try:
    from ..flywire_ids import (
        is_banc_dataset,
        is_fafb_dataset,
        is_local_connectome_dataset,
    )
except ImportError:
    from flywire_ids import (
        is_banc_dataset,
        is_fafb_dataset,
        is_local_connectome_dataset,
    )

try:
    from ..utils.label_utils import build_body_id_label
except ImportError:  # pragma: no cover - direct src/ execution
    try:
        from utils.label_utils import build_body_id_label
    except ImportError:  # last resort: bare '{bodyId}_{type}' labels
        build_body_id_label = None

try:
    from . import report_kit
except ImportError:  # pragma: no cover - direct src/ execution
    import report_kit

if TYPE_CHECKING:
    from .connectivity_profiler import ConnectivityStatus


# ============================================================================
# Default Score Weights
# ============================================================================

# Simplified comparison metrics:
# - Jaccard: Set-based overlap of partner types (0-1)
# - Rank correlation: Spearman correlation normalized to 0-1 using (x+1)/2
# These two metrics work well for both bodyId-level and type-level comparison
DEFAULT_SCORE_WEIGHTS = {
    'jaccard': 0.50,
    'rank': 0.50
}


# ============================================================================
# Comparison Result Dataclass
# ============================================================================

_OUTPUT_DROP_COLS = ('rank_corr', 'rank_union_raw', 'avg_rank_corr')


#: The per-source VISUALIZATION pick falls back along this chain when the
#: run's ``similarity_metric`` column is absent — Jaccard first, matching the
#: shipped default sort (BENCHMARK_RESULTS.md §7.1).  Hard-coding rank_union
#: here used to make the rendered set disagree with the exported ordering of
#: the same run (plan-tmvev-jaccard-primary-bodyid-ranking S3).
_VIS_METRIC_FALLBACK = ('jaccard', 'rank_union', 'cosine', 'rank_corr')


def pick_visualization_metric(metric, columns) -> Optional[str]:
    """Which column the per-source top-N visualization pick sorts by: the
    run's own ``similarity_metric`` when the frame carries it, else the first
    of :data:`_VIS_METRIC_FALLBACK` that is present."""
    for name in ((metric,) if metric else ()) + _VIS_METRIC_FALLBACK:
        if name in columns:
            return name
    return None


def _drop_rank_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Strip the poorly-ranking metric columns from user-facing output CSVs.

    rank_corr (shared-partner Spearman) is retained internally for the sort
    fallback and the type-mean derivation, but it is a poor ranker (see the
    benchmark) and is not shown in the exported result tables. rank_union_raw
    is an internal alias for the same signal and is likewise dropped.
    """
    drop = [c for c in _OUTPUT_DROP_COLS if c in df.columns]
    return df.drop(columns=drop) if drop else df


def auto_mapping_result_metadata(
    requested: bool,
    type_mapper: Optional[CrossDatasetTypeMapper],
    snapshot: Optional[object] = None,
    resolution_counts: Optional[Dict[str, int]] = None,
    partner_resolution_counts: Optional[Dict[str, int]] = None,
    raw_fallback_used: bool = False,
) -> Dict[str, Any]:
    """The standard auto-type-mapping metadata block for saved results.

    Records requested vs active mapping, the active mapper source/version,
    any load error, per-status resolution counts, and whether raw-name
    fallback occurred — so a saved result can always answer whether it was
    produced under a healthy v1.0 mapper.

    ``resolution_counts`` is the primary metric on the
    ``'unique_type_resolutions'`` basis.  ``partner_resolution_counts`` is
    a distinct occurrence-basis metric (contributor-type occurrences inside
    canonicalized profiles) written under its own key.
    """
    from .type_resolver import auto_mapping_metadata, get_mapper_snapshot

    if snapshot is None and (type_mapper is not None or requested):
        snapshot = get_mapper_snapshot(type_mapper, requested=requested)
    if snapshot is None:
        return {
            'auto_type_mapping_requested': requested,
            'auto_type_mapping_active': False,
            'auto_type_mapping_status': 'disabled',
            'mapping_policy_version': '1',
            'mapping_resolution_counts_by_status': dict(
                resolution_counts or {}),
            'mapping_resolution_counts_basis': 'unique_type_resolutions',
            'mapping_partner_type_resolutions_by_status': dict(
                partner_resolution_counts or {}),
            'raw_fallback_used': bool(raw_fallback_used),
        }
    return auto_mapping_metadata(
        snapshot,
        resolution_counts=resolution_counts,
        raw_fallback_used=raw_fallback_used,
        partner_resolution_counts=partner_resolution_counts)


@dataclass
class ComparisonResult:
    """
    Complete comparison result with all metrics.
    
    Attributes:
        profile_a_id: Identifier of first profile
        profile_b_id: Identifier of second profile
        dataset_a: Dataset of first profile
        dataset_b: Dataset of second profile
        direction: Direction compared ('upstream', 'downstream', 'both')
        jaccard: Jaccard similarity score
        cosine: Cosine similarity score
        rank_correlation: Spearman rank correlation
        overlap_a_in_b: Fraction of A's partners found in B (|A ∩ B| / |A|)
        overlap_b_in_a: Fraction of B's partners found in A (|A ∩ B| / |B|)
        weak_connectivity_a: Flag if profile A has weak connectivity
        weak_connectivity_b: Flag if profile B has weak connectivity
    """
    profile_a_id: str
    profile_b_id: str
    dataset_a: str
    dataset_b: str
    direction: str
    
    # Individual metrics
    jaccard: float
    cosine: float
    rank_correlation: float
    overlap_a_in_b: float  # |A ∩ B| / |A| - fraction of A's partners found in B
    overlap_b_in_a: float  # |A ∩ B| / |B| - fraction of B's partners found in A
    
    # Flags
    rank_union: float = np.nan
    weak_connectivity_a: bool = False
    weak_connectivity_b: bool = False
    notes: str = ""  # Additional notes (e.g., "constant_array" when rank correlation is NaN)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for CSV/JSON export."""
        result = {
            'profile_a': self.profile_a_id,
            'profile_b': self.profile_b_id,
            'dataset_a': self.dataset_a,
            'dataset_b': self.dataset_b,
            'direction': self.direction,
            'jaccard': round(self.jaccard, 4),
            'cosine': round(self.cosine, 4),
            'rank_corr': round(self.rank_correlation, 4) if not np.isnan(self.rank_correlation) else np.nan,
            'rank_union': round(self.rank_union, 4) if not np.isnan(self.rank_union) else np.nan,
            'overlap_a_in_b': round(self.overlap_a_in_b, 4),
            'overlap_b_in_a': round(self.overlap_b_in_a, 4),
            'weak_connectivity_warning': self.weak_connectivity_a or self.weak_connectivity_b
        }
        if self.notes:
            result['notes'] = self.notes
        return result
    
    def summary(self) -> str:
        """Generate a human-readable summary string."""
        # Handle NaN in rank correlation display
        rank_corr_str = f"{self.rank_correlation:.3f}" if not np.isnan(self.rank_correlation) else "NaN"
        
        lines = [
            f"Comparison: {self.profile_a_id} ({self.dataset_a}) vs {self.profile_b_id} ({self.dataset_b})",
            f"  Direction: {self.direction}",
            f"  Metrics: Jaccard={self.jaccard:.3f}, Cosine={self.cosine:.3f}, RankCorr={rank_corr_str}",
            f"  Overlap: A_in_B={self.overlap_a_in_b:.3f}, B_in_A={self.overlap_b_in_a:.3f}"
        ]
        if self.weak_connectivity_a or self.weak_connectivity_b:
            lines.append("  ⚠️ Weak connectivity warning")
        if self.notes:
            lines.append(f"  ℹ️ {self.notes}")
        return "\n".join(lines)


# ============================================================================
# Profile Comparator Class
# ============================================================================

class ProfileComparator:
    """
    Compare connectivity profiles using various similarity metrics.
    
    This class provides static methods for computing different similarity
    measures between ConnectivityProfile objects, enabling cross-dataset
    verification and homolog discovery.
    
    Example:
        >>> comparator = ProfileComparator()
        >>> result = comparator.compare_profiles(profile_a, profile_b)
        >>> print(f"Jaccard: {result.jaccard:.3f}, rank_corr: {result.rank_correlation:.3f}")
    """
    
    @staticmethod
    def _get_partner_sets(
        profile: ConnectivityProfile,
        direction: str
    ) -> set:
        """Get set of partner types for a given direction."""
        if direction == 'upstream':
            return set(profile.upstream_partners.keys())
        elif direction == 'downstream':
            return set(profile.downstream_partners.keys())
        else:  # both
            return (set(profile.upstream_partners.keys()) | 
                    set(profile.downstream_partners.keys()))
    
    @staticmethod
    def _get_weight_vector(
        profile: ConnectivityProfile,
        partners: List[str],
        direction: str
    ) -> np.ndarray:
        """
        Get weight vector for specified partners.
        
        Args:
            profile: ConnectivityProfile to extract weights from
            partners: Ordered list of partner types
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            numpy array of weights (0 if partner not present)
        """
        weights = []
        
        for partner in partners:
            if direction == 'upstream':
                w = profile.upstream_partners.get(partner, 0.0)
            elif direction == 'downstream':
                w = profile.downstream_partners.get(partner, 0.0)
            else:  # both - sum of upstream and downstream
                w = (profile.upstream_partners.get(partner, 0.0) +
                     profile.downstream_partners.get(partner, 0.0))
            weights.append(w)
        
        return np.array(weights)
    
    @staticmethod
    def jaccard_similarity(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both'
    ) -> float:
        """
        Jaccard similarity of partner type sets (binary presence).
        
        Formula: |A ∩ B| / |A ∪ B|
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Float in [0, 1], where 1 = identical partner sets
        
        Notes:
            - Ignores weights, purely set-based
            - Good for "do they connect to the same types?"
            - Returns 0 if both sets are empty
        """
        set_a = ProfileComparator._get_partner_sets(profile_a, direction)
        set_b = ProfileComparator._get_partner_sets(profile_b, direction)
        
        if not set_a and not set_b:
            return 0.0
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def weighted_cosine_similarity(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both',
        use_proportions: bool = True
    ) -> float:
        """
        Cosine similarity of weight vectors.
        
        Formula: (A · B) / (||A|| × ||B||)
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
            use_proportions: If True, use normalized proportions (default)
        
        Returns:
            Float in [-1, 1] (typically [0, 1] for non-negative weights)
        
        Notes:
            - Considers relative importance of each partner
            - Robust to total weight differences when using proportions
            - Returns 0 if either vector is all zeros
        """
        # Get union of all partner types
        set_a = ProfileComparator._get_partner_sets(profile_a, direction)
        set_b = ProfileComparator._get_partner_sets(profile_b, direction)
        all_partners = sorted(set_a | set_b)
        
        if not all_partners:
            return 0.0
        
        # Get weight vectors
        vec_a = ProfileComparator._get_weight_vector(profile_a, all_partners, direction)
        vec_b = ProfileComparator._get_weight_vector(profile_b, all_partners, direction)
        
        # Compute cosine similarity
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        dot_product = np.dot(vec_a, vec_b)
        cosine_sim = dot_product / (norm_a * norm_b)
        
        return float(cosine_sim)
    
    @staticmethod
    def rank_correlation(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both',
        method: str = 'spearman',
        use_all_partners: bool = False
    ) -> float:
        """
        Spearman or Kendall rank correlation of partner rankings.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
            method: 'spearman' or 'kendall'
            use_all_partners: If True, use all partners from both profiles with
                             default low ranks (K+1/M+1) for missing partners.
                             If False (default), only consider shared partners.
        
        Returns:
            Float in [-1, 1], where 1 = identical rankings.
            Returns NaN if insufficient partners (< 3).
        
        Notes:
            - When use_all_partners=False (default), focuses on shared partners only,
              avoiding the problem where many unique partners dilute the correlation
            - When use_all_partners=True, captures overall similarity more comprehensively
              by penalizing missing partners with low default ranks
            - Handles ties appropriately
        """
        # Get ranks for each direction
        if direction == 'upstream':
            ranks_a = profile_a.upstream_ranks
            ranks_b = profile_b.upstream_ranks
        elif direction == 'downstream':
            ranks_a = profile_a.downstream_ranks
            ranks_b = profile_b.downstream_ranks
        else:  # both - average of upstream and downstream correlations
            # Compute upstream and downstream correlations separately, then average
            # This prevents the issue where partners appearing in different directions
            # in different datasets cause artificially low combined correlation
            upstream_corr = ProfileComparator.rank_correlation(
                profile_a, profile_b, direction='upstream', method=method,
                use_all_partners=use_all_partners
            )
            downstream_corr = ProfileComparator.rank_correlation(
                profile_a, profile_b, direction='downstream', method=method,
                use_all_partners=use_all_partners
            )
            
            # Handle NaN cases - use the valid one, or NaN if both are NaN
            if np.isnan(upstream_corr) and np.isnan(downstream_corr):
                return np.nan
            elif np.isnan(upstream_corr):
                return downstream_corr
            elif np.isnan(downstream_corr):
                return upstream_corr
            else:
                return (upstream_corr + downstream_corr) / 2
        
        if use_all_partners:
            # Use union of all partners with default ranks for missing
            all_partners = set(ranks_a.keys()) | set(ranks_b.keys())
            
            if len(all_partners) < 3:
                return np.nan
            
            # Default ranks: K+1 for missing in A, M+1 for missing in B
            k_a = len(ranks_a)  # Number of partners in A
            k_b = len(ranks_b)  # Number of partners in B
            default_rank_a = k_a + 1  # Default rank for partners missing in A
            default_rank_b = k_b + 1  # Default rank for partners missing in B
            
            # Build rank arrays for all partners
            # Use str() key for consistent sorting (may have mixed int/str types)
            all_list = sorted(all_partners, key=lambda x: str(x))
            rank_array_a = [ranks_a.get(p, default_rank_a) for p in all_list]
            rank_array_b = [ranks_b.get(p, default_rank_b) for p in all_list]
        else:
            # Shared partners mode with expansion if needed
            shared = set(ranks_a.keys()) & set(ranks_b.keys())
            all_partners = set(ranks_a.keys()) | set(ranks_b.keys())
            
            min_partners = 5  # Minimum partners needed for meaningful correlation
            
            if len(shared) >= min_partners:
                # Enough shared partners, use them directly
                partner_list = sorted(shared, key=lambda x: str(x))
                rank_array_a = [ranks_a[p] for p in partner_list]
                rank_array_b = [ranks_b[p] for p in partner_list]
            elif len(all_partners) < 2:
                # Not enough total partners
                return np.nan
            else:
                # Expand to min_partners by adding top-ranked partners from union
                # Priority: shared first, then lowest-ranked (most important) from each side
                
                # Start with shared partners
                selected = set(shared)
                
                # Get non-shared partners sorted by their rank (lower = more important)
                non_shared_a = [(p, ranks_a[p]) for p in ranks_a.keys() if p not in shared]
                non_shared_b = [(p, ranks_b[p]) for p in ranks_b.keys() if p not in shared]
                
                # Sort by rank (ascending = most important first)
                non_shared_a.sort(key=lambda x: x[1])
                non_shared_b.sort(key=lambda x: x[1])
                
                # Alternate adding from A and B to balance expansion
                idx_a, idx_b = 0, 0
                while len(selected) < min_partners and (idx_a < len(non_shared_a) or idx_b < len(non_shared_b)):
                    # Add from A
                    if idx_a < len(non_shared_a) and len(selected) < min_partners:
                        selected.add(non_shared_a[idx_a][0])
                        idx_a += 1
                    # Add from B
                    if idx_b < len(non_shared_b) and len(selected) < min_partners:
                        selected.add(non_shared_b[idx_b][0])
                        idx_b += 1
                
                if len(selected) < 2:
                    return np.nan
                
                # Build rank arrays with default ranks for missing partners
                k_a = len(ranks_a)
                k_b = len(ranks_b)
                default_rank_a = k_a + 1
                default_rank_b = k_b + 1
                
                partner_list = sorted(selected)
                rank_array_a = [ranks_a.get(p, default_rank_a) for p in partner_list]
                rank_array_b = [ranks_b.get(p, default_rank_b) for p in partner_list]
        
        # Check for constant arrays before computing correlation
        # (correlation is undefined when one array has no variance)
        arr_a = np.array(rank_array_a)
        arr_b = np.array(rank_array_b)
        if np.std(arr_a) == 0 or np.std(arr_b) == 0:
            # Return NaN and let caller handle it (marked as 'constant_array' in results)
            return np.nan
        
        # Compute correlation (suppress ConstantInputWarning just in case of edge cases)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='An input array is constant')
            if method == 'kendall':
                corr, _ = stats.kendalltau(rank_array_a, rank_array_b)
            else:  # spearman
                corr, _ = stats.spearmanr(rank_array_a, rank_array_b)
        
        # Handle NaN (can occur with constant arrays)
        if np.isnan(corr):
            return np.nan
        
        return float(corr)
    
    @staticmethod
    def overlap_fraction(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both'
    ) -> Tuple[float, float]:
        """
        Fraction of top-K partners that overlap.
        
        Returns:
            Tuple of (overlap_fraction_a, overlap_fraction_b)
            - overlap_fraction_a: |A ∩ B| / |A|
            - overlap_fraction_b: |A ∩ B| / |B|
        
        Notes:
            - Asymmetric measure (A→B may differ from B→A)
            - Useful when one profile has more specific annotations
            - Returns (0, 0) if either profile is empty
        """
        set_a = ProfileComparator._get_partner_sets(profile_a, direction)
        set_b = ProfileComparator._get_partner_sets(profile_b, direction)
        
        if not set_a or not set_b:
            return 0.0, 0.0
        
        intersection = len(set_a & set_b)
        
        overlap_a = intersection / len(set_a) if set_a else 0.0
        overlap_b = intersection / len(set_b) if set_b else 0.0
        
        return overlap_a, overlap_b
    
    @staticmethod
    def combined_score(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        weights: Optional[Dict[str, float]] = None,
        direction: str = 'both'
    ) -> Dict[str, float]:
        """
        Combined similarity metric calculator (Jaccard + rank based).
        
        Only the raw (unnormalized) metrics are returned: Jaccard and the two
        Spearman rank correlations (shared and union).  The normalized 0-1
        rank variants and the derived 'combined' score are no longer computed.
        The ``weights`` argument is retained only for API compatibility and has
        no effect.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            weights: Custom weight dictionary (optional, inert)
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Dict with scores:
            - 'jaccard': Jaccard similarity (0-1)
            - 'rank': Original rank correlation (-1 to 1)
            - 'cosine': Cosine similarity
            - 'overlap_a_in_b', 'overlap_b_in_a': Overlap fractions
        """
        if weights is None:
            weights = DEFAULT_SCORE_WEIGHTS
        
        # Compute individual metrics
        jaccard = ProfileComparator.jaccard_similarity(profile_a, profile_b, direction)
        cosine = ProfileComparator.weighted_cosine_similarity(profile_a, profile_b, direction)
        rank_corr = ProfileComparator.rank_correlation(profile_a, profile_b, direction)
        overlap_a, overlap_b = ProfileComparator.overlap_fraction(profile_a, profile_b, direction)
        
        # Weighted Jaccard: sum(min(w_a, w_b)) / sum(max(w_a, w_b)) over the
        # partner-type union (same semantics as the homolog scorer)
        set_a = ProfileComparator._get_partner_sets(profile_a, direction)
        set_b = ProfileComparator._get_partner_sets(profile_b, direction)
        w_partners = sorted(set_a | set_b)
        if w_partners:
            w_vec_a = ProfileComparator._get_weight_vector(profile_a, w_partners, direction)
            w_vec_b = ProfileComparator._get_weight_vector(profile_b, w_partners, direction)
            w_intersection = float(np.sum(np.minimum(w_vec_a, w_vec_b)))
            w_union = float(np.sum(np.maximum(w_vec_a, w_vec_b)))
            weighted_jaccard = w_intersection / w_union if w_union > 0 else 0.0
        else:
            weighted_jaccard = 0.0
        
        # Rank correlation on the UNION of partner types (missing = 0.0) —
        # same semantics as the homolog scorer's rank_union
        if len(w_partners) >= 3:
            u_vec_a = ProfileComparator._get_weight_vector(profile_a, w_partners, direction)
            u_vec_b = ProfileComparator._get_weight_vector(profile_b, w_partners, direction)
            if len(set(u_vec_a)) > 1 and len(set(u_vec_b)) > 1:
                try:
                    from scipy.stats import spearmanr
                    rank_union, _ = spearmanr(u_vec_a, u_vec_b)
                    rank_union = float(rank_union) if not np.isnan(rank_union) else np.nan
                except Exception:
                    rank_union = np.nan
            else:
                rank_union = np.nan
        else:
            rank_union = np.nan

        # Keep overlap values for backward compatibility
        overlap_avg = (overlap_a + overlap_b) / 2

        return {
            'jaccard': jaccard,
            'weighted_jaccard': weighted_jaccard,
            'cosine': cosine,
            'rank': rank_corr,  # Original [-1, 1]
            'rank_union': rank_union,  # Union-based rank, original [-1, 1]
            'overlap_a_in_b': overlap_a,
            'overlap_b_in_a': overlap_b,
            'overlap_avg': overlap_avg
        }
    
    @staticmethod
    def _get_all_bodyids(
        profile: ConnectivityProfile,
        direction: str
    ) -> Dict[int, float]:
        """
        Get all partner bodyIds (typed + untyped) for bodyId-level comparison.
        
        Args:
            profile: ConnectivityProfile
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Dict[bodyId, weight] for all partners
        """
        all_bodyids: Dict[int, float] = {}
        
        if direction in ('upstream', 'both'):
            # Add typed upstream bodyIds
            if profile.typed_upstream_bodyids:
                for bid, weight in profile.typed_upstream_bodyids.items():
                    if weight is not None:
                        all_bodyids[bid] = all_bodyids.get(bid, 0.0) + float(weight)
            # Add untyped upstream bodyIds
            if profile.untyped_upstream_bodyids:
                for bid, weight in profile.untyped_upstream_bodyids.items():
                    if weight is not None:
                        all_bodyids[bid] = all_bodyids.get(bid, 0.0) + float(weight)
        
        if direction in ('downstream', 'both'):
            # Add typed downstream bodyIds
            if profile.typed_downstream_bodyids:
                for bid, weight in profile.typed_downstream_bodyids.items():
                    if weight is not None:
                        all_bodyids[bid] = all_bodyids.get(bid, 0.0) + float(weight)
            # Add untyped downstream bodyIds
            if profile.untyped_downstream_bodyids:
                for bid, weight in profile.untyped_downstream_bodyids.items():
                    if weight is not None:
                        all_bodyids[bid] = all_bodyids.get(bid, 0.0) + float(weight)
        
        return all_bodyids
    
    @staticmethod
    def _get_expanded_types(
        profile: ConnectivityProfile,
        direction: str,
        prefix_2hop: bool = True,
        top_n_for_2hop_check: int = 5
    ) -> Dict[str, float]:
        """
        Get expanded type profile including 2-hop expansion for untyped partners.
        
        IMPORTANT: 2-hop expansion is ONLY used when the top-N connections are ALL untyped.
        This prevents diluting the similarity signal with indirect connections when
        direct typed connections are available.
        
        For cross-dataset comparison:
        - If top-N connections include ANY typed partner: use only 1-hop typed partners
        - If top-N connections are ALL untyped: expand to their 2-hop typed partners
          with "2hop:" prefix (e.g., "2hop:Mi1") to distinguish from direct partners
        
        The 2-hop prefix ensures that:
        1. Direct partners and indirect partners are tracked separately
        2. A type appearing as both 1-hop and 2-hop creates two distinct entries
        3. Comparisons can match 1-hop↔1-hop and 2-hop↔2-hop connections
        
        Args:
            profile: ConnectivityProfile
            direction: 'upstream', 'downstream', or 'both'
            prefix_2hop: If True, prefix 2-hop types with "2hop:" (default: True)
            top_n_for_2hop_check: Only use 2-hop if top N connections are all untyped (default: 5)
        
        Returns:
            Dict[type, weight] with 2-hop types prefixed to distinguish from 1-hop
        """
        expanded_types: Dict[str, float] = {}
        hop2_prefix = "2hop:" if prefix_2hop else ""
        
        if direction in ('upstream', 'both'):
            # Check if top-N upstream are all untyped
            # Get typed partner weights
            typed_weights = [(ptype, weight) for ptype, weight in profile.upstream_partners.items()
                           if ptype and ptype != 'untyped']
            # Get untyped partner weights
            untyped_weights = []
            if profile.untyped_upstream_bodyids:
                untyped_weights = list(profile.untyped_upstream_bodyids.values())
            
            # Combine and sort by weight to check top-N
            all_weights = [(w, True) for _, w in typed_weights] + [(w, False) for w in untyped_weights]
            all_weights.sort(key=lambda x: x[0], reverse=True)
            top_n = all_weights[:top_n_for_2hop_check]
            
            # Only use 2-hop if ALL top-N are untyped (is_typed=False)
            use_2hop_upstream = len(top_n) > 0 and all(not is_typed for _, is_typed in top_n)
            
            # Add 1-hop typed partners (no prefix) - always add these
            for ptype, weight in profile.upstream_partners.items():
                if ptype and ptype != 'untyped':
                    expanded_types[ptype] = expanded_types.get(ptype, 0.0) + weight
            
            # Add 2-hop types for untyped upstream partners (with prefix) - only if top-N all untyped
            if use_2hop_upstream and profile.untyped_upstream_2hop:
                for untyped_bid, hop2_types in profile.untyped_upstream_2hop.items():
                    # Weight the 2-hop types by the 1-hop connection weight
                    untyped_weight = 1.0
                    if profile.untyped_upstream_bodyids and untyped_bid in profile.untyped_upstream_bodyids:
                        untyped_weight = profile.untyped_upstream_bodyids[untyped_bid]
                    
                    # Normalize 2-hop weights and scale by 1-hop weight
                    hop2_total = sum(hop2_types.values()) if hop2_types else 1.0
                    for hop2_type, hop2_weight in hop2_types.items():
                        if hop2_type and hop2_type != 'untyped':
                            # Scale by relative 2-hop weight and 1-hop weight
                            scaled_weight = (hop2_weight / hop2_total) * untyped_weight if hop2_total > 0 else 0
                            prefixed_type = f"{hop2_prefix}{hop2_type}"
                            expanded_types[prefixed_type] = expanded_types.get(prefixed_type, 0.0) + scaled_weight
        
        if direction in ('downstream', 'both'):
            # Check if top-N downstream are all untyped
            typed_weights = [(ptype, weight) for ptype, weight in profile.downstream_partners.items()
                           if ptype and ptype != 'untyped']
            untyped_weights = []
            if profile.untyped_downstream_bodyids:
                untyped_weights = list(profile.untyped_downstream_bodyids.values())
            
            all_weights = [(w, True) for _, w in typed_weights] + [(w, False) for w in untyped_weights]
            all_weights.sort(key=lambda x: x[0], reverse=True)
            top_n = all_weights[:top_n_for_2hop_check]
            
            # Only use 2-hop if ALL top-N are untyped
            use_2hop_downstream = len(top_n) > 0 and all(not is_typed for _, is_typed in top_n)
            
            # Add 1-hop typed partners (no prefix) - always add these
            for ptype, weight in profile.downstream_partners.items():
                if ptype and ptype != 'untyped':
                    expanded_types[ptype] = expanded_types.get(ptype, 0.0) + weight
            
            # Add 2-hop types for untyped downstream partners (with prefix) - only if top-N all untyped
            if use_2hop_downstream and profile.untyped_downstream_2hop:
                for untyped_bid, hop2_types in profile.untyped_downstream_2hop.items():
                    # Weight the 2-hop types by the 1-hop connection weight
                    untyped_weight = 1.0
                    if profile.untyped_downstream_bodyids and untyped_bid in profile.untyped_downstream_bodyids:
                        untyped_weight = profile.untyped_downstream_bodyids[untyped_bid]
                    
                    # Normalize 2-hop weights and scale by 1-hop weight
                    hop2_total = sum(hop2_types.values()) if hop2_types else 1.0
                    for hop2_type, hop2_weight in hop2_types.items():
                        if hop2_type and hop2_type != 'untyped':
                            scaled_weight = (hop2_weight / hop2_total) * untyped_weight if hop2_total > 0 else 0
                            prefixed_type = f"{hop2_prefix}{hop2_type}"
                            expanded_types[prefixed_type] = expanded_types.get(prefixed_type, 0.0) + scaled_weight
        
        return expanded_types
    
    @staticmethod
    def _get_expanded_types_standardized(
        profile: ConnectivityProfile,
        direction: str,
        type_mapper: Optional[CrossDatasetTypeMapper] = None,
        prefix_2hop: bool = True,
        top_n_for_2hop_check: int = 5
    ) -> Dict[str, float]:
        """
        Get expanded type profile with standardized (canonical) type names.
        
        This wraps _get_expanded_types and optionally standardizes type names
        to their male-cns canonical names for cross-dataset comparison.
        
        When comparing profiles from different datasets, partner types like
        'MTe07' (FAFB) and 'MeVPLo2' (male-cns) should be recognized as the
        same type. This method maps all types to their canonical names.
        
        Args:
            profile: ConnectivityProfile
            direction: 'upstream', 'downstream', or 'both'
            type_mapper: Optional CrossDatasetTypeMapper for standardization.
                        If None, returns un-standardized types.
            prefix_2hop: If True, prefix 2-hop types with "2hop:" (default: True)
            top_n_for_2hop_check: Only use 2-hop if top N connections are all untyped (default: 5)
        
        Returns:
            Dict[canonical_type, weight] with standardized type names
        """
        # Get raw expanded types
        expanded = ProfileComparator._get_expanded_types(
            profile, direction, prefix_2hop, top_n_for_2hop_check
        )
        
        # If no mapper, return as-is
        if type_mapper is None:
            return expanded

        # Standardize to canonical names
        return type_mapper.standardize_partner_types(expanded, profile.dataset)

    @staticmethod
    def _get_expanded_types_resolved(
        profile: ConnectivityProfile,
        direction: str,
        type_mapper: Optional[CrossDatasetTypeMapper] = None,
        prefix_2hop: bool = True,
        top_n_for_2hop_check: int = 5,
        decision_cache: Optional[Dict] = None,
        allow_evidence_only: bool = False,
        status_counts_out: Optional[Counter] = None,
    ) -> Dict[str, float]:
        """Expanded type profile canonicalized with the panel's VALIDITY
        policy (shared resolver, ``comparison.type_resolver``).

        Unlike ``_get_expanded_types_standardized`` (which applies raw
        canonicalization only), this records which mapping status produced
        each key: conflicts are excluded outright (never compared by raw
        same-name), valid splits distribute their weight evenly across
        every licensed target, and unmapped types keep their raw name as
        the counted long-tail fallback.  Returned dict is
        ``{canonical_type: weight}``, same contract as the raw variant.
        Per-type statuses are aggregated into ``status_counts_out`` when
        given, for the result metadata block.
        """
        expanded = ProfileComparator._get_expanded_types(
            profile, direction, prefix_2hop, top_n_for_2hop_check)
        if type_mapper is None:
            return expanded
        from .type_resolver import expand_profile_types
        expansion = expand_profile_types(
            type_mapper, expanded, profile.dataset, None,
            decision_cache=decision_cache,
            allow_evidence_only=allow_evidence_only)
        if status_counts_out is not None:
            for status, count in expansion.status_counts.items():
                status_counts_out[status] += count
            if expansion.fallback_used:
                status_counts_out['_fallback_profiles'] += 1
        return expansion.canonical

    @staticmethod
    def standardize_profile(
        profile: 'ConnectivityProfile',
        dataset: str,
        type_mapper: Optional['CrossDatasetTypeMapper'],
    ) -> 'ConnectivityProfile':
        """Return a copy of ``profile`` whose partner dicts are canonical.

        Both directions are standardized with
        ``standardize_partner_types`` (``{canonical_type: weight}``) and
        the rank caches rebuilt, so any two standardized profiles from
        different datasets live in the same comparison namespace.  Used by
        the type-level pooling path and the mapping-aware vector prefilter.
        """
        if type_mapper is None:
            return profile
        from .connectivity_profiler import ConnectivityProfile, compute_ranks
        up = type_mapper.standardize_partner_types(
            dict(profile.upstream_partners), dataset)
        down = type_mapper.standardize_partner_types(
            dict(profile.downstream_partners), dataset)
        return ConnectivityProfile(
            neuron_id=profile.neuron_id, dataset=profile.dataset,
            upstream_partners=up, downstream_partners=down,
            upstream_ranks=compute_ranks(up),
            downstream_ranks=compute_ranks(down),
            upstream_top_k=len(up), downstream_top_k=len(down),
            total_upstream_weight=profile.total_upstream_weight,
            total_downstream_weight=profile.total_downstream_weight,
            untyped_upstream_weight_fraction=(
                profile.untyped_upstream_weight_fraction),
            untyped_downstream_weight_fraction=(
                profile.untyped_downstream_weight_fraction),
            actual_upstream_count=len(up),
            actual_downstream_count=len(down),
            unique_types_upstream=len(up),
            unique_types_downstream=len(down),
        )
    
    @staticmethod
    def bodyid_jaccard(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both'
    ) -> float:
        """
        Jaccard similarity using partner bodyIds directly (for intra-dataset).
        
        Uses all partner bodyIds (typed + untyped) without type aggregation.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Float in [0, 1], where 1 = identical partner bodyId sets
        """
        bodyids_a = set(ProfileComparator._get_all_bodyids(profile_a, direction).keys())
        bodyids_b = set(ProfileComparator._get_all_bodyids(profile_b, direction).keys())
        
        if not bodyids_a and not bodyids_b:
            return 0.0
        
        intersection = len(bodyids_a & bodyids_b)
        union = len(bodyids_a | bodyids_b)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def bodyid_rank_correlation(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both'
    ) -> float:
        """
        Rank correlation using partner bodyIds directly (for intra-dataset).
        
        Uses weight-based ranks on shared bodyIds.
        
        Args:
            profile_a: First connectivity profile  
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Float in [-1, 1], where 1 = identical rankings
        """
        from scipy.stats import spearmanr
        
        bodyids_a = ProfileComparator._get_all_bodyids(profile_a, direction)
        bodyids_b = ProfileComparator._get_all_bodyids(profile_b, direction)
        
        # Find shared bodyIds
        shared = set(bodyids_a.keys()) & set(bodyids_b.keys())
        
        if len(shared) < 3:
            return np.nan
        
        # Build weight vectors for shared bodyIds
        shared_list = sorted(shared)
        weights_a = [bodyids_a[bid] for bid in shared_list]
        weights_b = [bodyids_b[bid] for bid in shared_list]
        
        # Check for constant arrays (would cause ConstantInputWarning)
        if len(set(weights_a)) <= 1 or len(set(weights_b)) <= 1:
            return np.nan
        
        # Compute Spearman correlation
        try:
            corr, _ = spearmanr(weights_a, weights_b)
            return float(corr) if not np.isnan(corr) else np.nan
        except Exception:
            return np.nan
    
    @staticmethod
    def bodyid_rank_correlation_union(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both'
    ) -> float:
        """
        Rank correlation using UNION of partner bodyIds (for intra-dataset).
        
        Unlike the shared-based version, this uses ALL bodyIds from both profiles.
        Missing partners are assigned weight 0.
        
        Args:
            profile_a: First connectivity profile  
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Float in [-1, 1], where 1 = identical rankings
        """
        from scipy.stats import spearmanr
        
        bodyids_a = ProfileComparator._get_all_bodyids(profile_a, direction)
        bodyids_b = ProfileComparator._get_all_bodyids(profile_b, direction)
        
        # Use UNION of all bodyIds (missing = weight 0)
        all_bodyids = set(bodyids_a.keys()) | set(bodyids_b.keys())
        
        if len(all_bodyids) < 3:
            return np.nan
        
        # Build weight vectors for ALL bodyIds (0 if missing)
        # Convert to string for consistent sorting (bodyIds may be mixed int/str)
        bodyids_list = sorted(all_bodyids, key=lambda x: str(x))
        weights_a = [bodyids_a.get(bid, 0.0) for bid in bodyids_list]
        weights_b = [bodyids_b.get(bid, 0.0) for bid in bodyids_list]
        
        # Check for constant arrays (would cause ConstantInputWarning)
        if len(set(weights_a)) <= 1 or len(set(weights_b)) <= 1:
            return np.nan
        
        # Compute Spearman correlation
        try:
            corr, _ = spearmanr(weights_a, weights_b)
            return float(corr) if not np.isnan(corr) else np.nan
        except Exception:
            return np.nan
    
    @staticmethod
    def expanded_type_rank_correlation_union(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both'
    ) -> float:
        """
        Rank correlation using UNION of expanded types (for cross-dataset).
        
        Unlike the shared-based version, this uses ALL types from both profiles.
        Missing types are assigned weight 0.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Float in [-1, 1], where 1 = identical rankings
        """
        from scipy.stats import spearmanr
        
        types_a = ProfileComparator._get_expanded_types(profile_a, direction)
        types_b = ProfileComparator._get_expanded_types(profile_b, direction)
        
        # Use UNION of all types (missing = weight 0)
        all_types = set(types_a.keys()) | set(types_b.keys())
        
        if len(all_types) < 3:
            return np.nan
        
        # Build weight vectors for ALL types (0 if missing)
        # Use str() key for safety (types should already be strings)
        types_list = sorted(all_types, key=lambda x: str(x))
        weights_a = [types_a.get(t, 0.0) for t in types_list]
        weights_b = [types_b.get(t, 0.0) for t in types_list]
        
        # Check for constant arrays (would cause ConstantInputWarning)
        if len(set(weights_a)) <= 1 or len(set(weights_b)) <= 1:
            return np.nan
        
        # Compute Spearman correlation
        try:
            corr, _ = spearmanr(weights_a, weights_b)
            return float(corr) if not np.isnan(corr) else np.nan
        except Exception:
            return np.nan
    
    @staticmethod
    def expanded_type_jaccard(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both'
    ) -> float:
        """
        Jaccard similarity using expanded types (for cross-dataset).
        
        Includes 2-hop expansion for untyped 1-hop partners.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Float in [0, 1], where 1 = identical partner type sets
        """
        types_a = set(ProfileComparator._get_expanded_types(profile_a, direction).keys())
        types_b = set(ProfileComparator._get_expanded_types(profile_b, direction).keys())
        
        if not types_a and not types_b:
            return 0.0
        
        intersection = len(types_a & types_b)
        union = len(types_a | types_b)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def expanded_type_rank_correlation(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both'
    ) -> float:
        """
        Rank correlation using expanded types (for cross-dataset).
        
        Includes 2-hop expansion for untyped 1-hop partners.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Float in [-1, 1], where 1 = identical rankings
        """
        from scipy.stats import spearmanr
        
        types_a = ProfileComparator._get_expanded_types(profile_a, direction)
        types_b = ProfileComparator._get_expanded_types(profile_b, direction)
        
        # Find shared types
        shared = set(types_a.keys()) & set(types_b.keys())
        
        if len(shared) < 3:
            return np.nan
        
        # Build weight vectors for shared types
        shared_list = sorted(shared)
        weights_a = [types_a[t] for t in shared_list]
        weights_b = [types_b[t] for t in shared_list]
        
        # Check for constant arrays (would cause ConstantInputWarning)
        if len(set(weights_a)) <= 1 or len(set(weights_b)) <= 1:
            return np.nan
        
        # Compute Spearman correlation
        try:
            corr, _ = spearmanr(weights_a, weights_b)
            return float(corr) if not np.isnan(corr) else np.nan
        except Exception:
            return np.nan
    
    @staticmethod
    def combined_score_intra_dataset(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        weights: Optional[Dict[str, float]] = None,
        direction: str = 'both'
    ) -> Dict[str, float]:
        """
        Combined similarity score for intra-dataset (same dataset) comparison.
        
        Uses bodyId-level comparison directly:
        - All partner bodyIds (typed + untyped) are compared
        - No type aggregation or 2-hop expansion
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            weights: Custom weight dictionary (optional)
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Dict with scores (same format as combined_score)
        """
        if weights is None:
            weights = DEFAULT_SCORE_WEIGHTS
        
        # Check for empty profiles (0 partners = no valid connectivity profile)
        bodyids_a = ProfileComparator._get_all_bodyids(profile_a, direction)
        bodyids_b = ProfileComparator._get_all_bodyids(profile_b, direction)
        
        # If either profile has 0 partners, return NaN for all metrics
        if len(bodyids_a) == 0 or len(bodyids_b) == 0:
            return {
                'jaccard': np.nan,
                'cosine': np.nan,
                'rank': np.nan,
                'rank_union': np.nan,
                'overlap_a_in_b': np.nan,
                'overlap_b_in_a': np.nan,
                'overlap_avg': np.nan,
                'shared_type_count': 0,
                'union_type_count': len(bodyids_a) + len(bodyids_b)  # 0 for both empty
            }
        
        # Compute bodyId-level metrics
        jaccard = ProfileComparator.bodyid_jaccard(profile_a, profile_b, direction)
        rank_corr = ProfileComparator.bodyid_rank_correlation(profile_a, profile_b, direction)
        rank_union = ProfileComparator.bodyid_rank_correlation_union(profile_a, profile_b, direction)
        
        # Cosine similarity using bodyId weights (bodyids_a/b already computed above)
        # Ensure all bodyIds are strings for sorting
        all_bodyids = sorted(set(str(k) for k in bodyids_a.keys()) | set(str(k) for k in bodyids_b.keys()))
        
        # Re-map bodyids_a/b to string keys for lookup
        bodyids_a_str = {str(k): v for k, v in bodyids_a.items()}
        bodyids_b_str = {str(k): v for k, v in bodyids_b.items()}
        
        shared_bodyids = set(bodyids_a_str.keys()) & set(bodyids_b_str.keys())
        
        if all_bodyids:
            vec_a = np.array([bodyids_a_str.get(bid, 0.0) for bid in all_bodyids])
            vec_b = np.array([bodyids_b_str.get(bid, 0.0) for bid in all_bodyids])
            norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
            cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0
        else:
            cosine = 0.0
        
        # Weighted Jaccard over the bodyId union (same semantics as the
        # homolog scorer's weighted_jaccard)
        if all_bodyids:
            w_intersection = float(np.sum(np.minimum(vec_a, vec_b)))
            w_union = float(np.sum(np.maximum(vec_a, vec_b)))
            weighted_jaccard = w_intersection / w_union if w_union > 0 else 0.0
        else:
            weighted_jaccard = 0.0
        
        return {
            'jaccard': jaccard,
            'weighted_jaccard': weighted_jaccard,
            'cosine': cosine,
            'rank': rank_corr,
            'rank_union': rank_union,
            'overlap_a_in_b': 0.0,  # Not computed for bodyId-level
            'overlap_b_in_a': 0.0,
            'overlap_avg': 0.0,
            'shared_type_count': len(shared_bodyids),  # For bodyId comparison, this is shared bodyIds
            'union_type_count': len(all_bodyids)  # Total unique bodyIds in union
        }
    
    @staticmethod
    def combined_score_cross_dataset(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        weights: Optional[Dict[str, float]] = None,
        direction: str = 'both'
    ) -> Dict[str, float]:
        """
        Combined similarity score for cross-dataset comparison.
        
        Uses type-based comparison with 2-hop expansion:
        - 1-hop typed partners are compared directly
        - 1-hop untyped partners are expanded to their 2-hop typed partners
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            weights: Custom weight dictionary (optional)
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Dict with scores (same format as combined_score)
        """
        if weights is None:
            weights = DEFAULT_SCORE_WEIGHTS
        
        # Get expanded types first for empty profile check
        types_a = ProfileComparator._get_expanded_types(profile_a, direction)
        types_b = ProfileComparator._get_expanded_types(profile_b, direction)
        
        # If either profile has 0 partners, return NaN for all metrics
        if len(types_a) == 0 or len(types_b) == 0:
            return {
                'jaccard': np.nan,
                'cosine': np.nan,
                'rank': np.nan,
                'rank_union': np.nan,
                'overlap_a_in_b': np.nan,
                'overlap_b_in_a': np.nan,
                'overlap_avg': np.nan,
                'shared_type_count': 0,
                'union_type_count': len(types_a) + len(types_b)  # 0 for both empty
            }
        
        # Compute expanded type metrics
        jaccard = ProfileComparator.expanded_type_jaccard(profile_a, profile_b, direction)
        rank_corr = ProfileComparator.expanded_type_rank_correlation(profile_a, profile_b, direction)
        rank_union = ProfileComparator.expanded_type_rank_correlation_union(profile_a, profile_b, direction)
        
        # Cosine similarity using expanded type weights (types_a/b already computed above)
        all_types = sorted(set(types_a.keys()) | set(types_b.keys()))
        shared_types = set(types_a.keys()) & set(types_b.keys())
        
        if all_types:
            vec_a = np.array([types_a.get(t, 0.0) for t in all_types])
            vec_b = np.array([types_b.get(t, 0.0) for t in all_types])
            norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
            cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0
        else:
            cosine = 0.0
        
        return {
            'jaccard': jaccard,
            'cosine': cosine,
            'rank': rank_corr,
            'rank_union': rank_union,
            'overlap_a_in_b': 0.0,  # Not computed for expanded types
            'overlap_b_in_a': 0.0,
            'overlap_avg': 0.0,
            'shared_type_count': len(shared_types),  # Number of shared types for rank correlation
            'union_type_count': len(all_types)  # Total unique types in union
        }

    
    @staticmethod
    def batch_compare_cross_dataset(
        source_profile: ConnectivityProfile,
        target_profiles: Dict[int, ConnectivityProfile],
        candidate_map: Dict[int, int],
        direction: str = 'both',
        type_mapper: Optional[CrossDatasetTypeMapper] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch compare one source profile against many targets efficiently.
        
        Optimized for speed by:
        1. Pre-computing source expanded types once
        2. Using vectorized operations where possible
        3. Avoiding repeated function calls
        
        Args:
            source_profile: Source connectivity profile
            target_profiles: Dict mapping target_bid -> ConnectivityProfile
            candidate_map: Dict mapping target_bid -> shared_count (adjacency score)
            direction: 'upstream', 'downstream', or 'both'
            type_mapper: Optional CrossDatasetTypeMapper for standardizing partner types.
                        When comparing across datasets, this maps types like 'MTe07' (FAFB)
                        to their canonical male-cns names like 'MeVPLo2'.
        
        Returns:
            List of score dicts for each valid comparison
        """
        from scipy.stats import spearmanr
        
        results = []
        
        # Pre-compute source expanded types ONCE
        # Use standardized types if type_mapper is provided.  The scorer
        # core intentionally uses raw canonicalization (cheap, per-type
        # dicts): validity is enforced where candidates and endpoints enter
        # the pipeline (_expand_partner_types_to_target,
        # _compute_same_type_candidates, _get_expanded_types_resolved), so
        # conflicted partner names cannot create matches here.
        if type_mapper is not None:
            source_types = ProfileComparator._get_expanded_types_standardized(
                source_profile, direction, type_mapper
            )
        else:
            source_types = ProfileComparator._get_expanded_types(source_profile, direction)
        source_type_set = set(source_types.keys())
        source_type_count = len(source_types)
        
        if source_type_count == 0:
            return results
        
        # Process each target
        for target_bid, shared_count in candidate_map.items():
            target_profile = target_profiles.get(target_bid)
            if target_profile is None:
                continue
            
            # Get target expanded types (standardized if mapper provided)
            if type_mapper is not None:
                target_types = ProfileComparator._get_expanded_types_standardized(
                    target_profile, direction, type_mapper
                )
            else:
                target_types = ProfileComparator._get_expanded_types(target_profile, direction)
            target_type_set = set(target_types.keys())
            target_type_count = len(target_types)
            
            if target_type_count == 0:
                results.append({
                    'target_bid': target_bid,
                    'shared_count': shared_count,
                    'jaccard': np.nan,
                    'cosine': np.nan,
                    'rank': np.nan,
                    'rank_union': np.nan,
                    'shared_type_count': 0,
                    'union_type_count': source_type_count,
                    'target_type_count': 0
                })
                continue
            
            # Compute Jaccard (set-based, fast)
            intersection = source_type_set & target_type_set
            union = source_type_set | target_type_set
            jaccard = len(intersection) / len(union) if union else 0.0
            
            # Compute rank correlation on shared types
            shared_types = sorted(intersection)
            if len(shared_types) >= 3:
                weights_a = [source_types[t] for t in shared_types]
                weights_b = [target_types[t] for t in shared_types]
                
                # Check for constant arrays
                if len(set(weights_a)) > 1 and len(set(weights_b)) > 1:
                    try:
                        rank_corr, _ = spearmanr(weights_a, weights_b)
                        rank_corr = float(rank_corr) if not np.isnan(rank_corr) else np.nan
                    except Exception:
                        rank_corr = np.nan
                else:
                    rank_corr = np.nan
            else:
                rank_corr = np.nan
            
            # Compute rank correlation on UNION of types
            all_types = sorted(union)
            if len(all_types) >= 3:
                weights_a_union = [source_types.get(t, 0.0) for t in all_types]
                weights_b_union = [target_types.get(t, 0.0) for t in all_types]
                
                if len(set(weights_a_union)) > 1 and len(set(weights_b_union)) > 1:
                    try:
                        rank_union, _ = spearmanr(weights_a_union, weights_b_union)
                        rank_union = float(rank_union) if not np.isnan(rank_union) else np.nan
                    except Exception:
                        rank_union = np.nan
                else:
                    rank_union = np.nan
            else:
                rank_union = np.nan
            
            # Compute cosine similarity
            if all_types:
                vec_a = np.array([source_types.get(t, 0.0) for t in all_types])
                vec_b = np.array([target_types.get(t, 0.0) for t in all_types])
                norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
                cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0
            else:
                cosine = 0.0
            
            # Weighted Jaccard: sum(min(w_a, w_b)) / sum(max(w_a, w_b)) over
            # the union — same semantics as the profiling scorer
            if all_types:
                w_intersection = float(np.sum(np.minimum(vec_a, vec_b)))
                w_union = float(np.sum(np.maximum(vec_a, vec_b)))
                weighted_jaccard = w_intersection / w_union if w_union > 0 else 0.0
            else:
                weighted_jaccard = 0.0
            
            results.append({
                'target_bid': target_bid,
                'shared_count': shared_count,
                'jaccard': jaccard,
                'weighted_jaccard': weighted_jaccard,
                'cosine': cosine,
                'rank': rank_corr,
                'rank_union': rank_union,
                'shared_type_count': len(shared_types),
                'union_type_count': len(all_types),
                'target_type_count': target_type_count
            })

        return results

    @staticmethod
    def build_pooled_type_profiles(
        conn_df: pd.DataFrame,
        bid_to_labels: Dict[Any, Any],
        dataset: str = '',
        min_weight: Optional[int] = None,
        normalize_types: Optional[Callable[[pd.Series], pd.Series]] = None,
    ) -> Dict[str, ConnectivityProfile]:
        """
        Pool ALL adjacencies of each label's member neurons into type-level
        profiles (no per-bodyId top-k truncation).

        For every label L in ``bid_to_labels`` the upstream profile sums the
        weights of all connections whose postsynaptic neuron is a member of L
        (grouped by presynaptic partner type), and symmetrically for the
        downstream profile. This is the type-level analogue of a bodyId
        profile: one row per connection, aggregated by partner type, with the
        'untyped' bucket kept (the cross-dataset scorer drops it).

        Args:
            conn_df: Connection frame with bodyId_pre, bodyId_post, weight and
                type_pre, type_post columns (raw type names; fuzzy-normalized
                here when ``normalize_types`` is given).
            bid_to_labels: bodyId -> label (str) or list of labels. A bodyId
                may belong to several labels (e.g. overlapping custom groups);
                its connections then contribute to every one of them.
            dataset: Dataset identifier stored on the profiles.
            min_weight: Minimum synapse weight to include (the profiler's
                min_synapse_threshold; None keeps everything).
            normalize_types: Optional vectorized normalizer applied to the
                partner-type columns (the profiler's
                ``_normalize_types_vectorized``).

        Returns:
            Dict mapping label -> pooled ConnectivityProfile (neuron_id is the
            label string, so these are never written to the bodyId profile
            disk cache).
        """
        from .connectivity_profiler import ConnectivityProfile, compute_ranks

        if conn_df is None or conn_df.empty or not bid_to_labels:
            return {}

        df = conn_df
        if min_weight is not None:
            df = df[df['weight'] >= min_weight]
        if df.empty:
            return {}

        # Normalize the lookup keys so int64 and string bodyId columns both
        # resolve (FlyWire frames carry string bodyIds, NeuPrint frames ints).
        lut: Dict[Any, Any] = {}
        overlaps = False
        for key, value in bid_to_labels.items():
            labels = value if isinstance(value, (list, tuple, set)) else [value]
            labels = [lbl for lbl in labels if lbl is not None and lbl == lbl
                      and str(lbl).strip() != '']
            if not labels:
                continue
            if len(labels) > 1:
                overlaps = True
            lut[key] = labels[0] if len(labels) == 1 else list(labels)
            try:
                int_key = int(key)
                if int_key != key:
                    lut[int_key] = lut[key]
            except (TypeError, ValueError):
                pass
            str_key = str(key)
            if str_key not in lut:
                lut[str_key] = lut[key]

        def partner_types(col: pd.Series) -> pd.Series:
            types = col.where(col.notna(), '').astype(str).str.strip()
            types = types.where(types != '', 'untyped')
            if normalize_types is not None:
                types = normalize_types(types)
            return types

        # Positional frame (numpy columns -> no index-alignment pitfalls),
        # then explode label lists once each: a row whose pre AND post neurons
        # are both multi-label contributes to every label combination.
        work = pd.DataFrame({
            'lbl_post': df['bodyId_post'].map(lut).to_numpy(),
            'lbl_pre': df['bodyId_pre'].map(lut).to_numpy(),
            'pt_pre': partner_types(df['type_pre']).to_numpy(),
            'pt_post': partner_types(df['type_post']).to_numpy(),
            'weight': df['weight'].to_numpy(dtype=np.float64),
        })
        work = work[work['lbl_post'].notna() | work['lbl_pre'].notna()]
        if work.empty:
            return {}
        if overlaps:
            work = work.explode('lbl_post').explode('lbl_pre')
            work = work[work['lbl_post'].notna() | work['lbl_pre'].notna()]
        if work.empty:
            return {}

        def finish(label: str, up: Dict[str, float], down: Dict[str, float]) -> ConnectivityProfile:
            up = {t: float(w) for t, w in up.items() if w > 0}
            down = {t: float(w) for t, w in down.items() if w > 0}
            total_up = float(sum(up.values()))
            total_down = float(sum(down.values()))
            untyped_up = up.get('untyped', 0.0)
            untyped_down = down.get('untyped', 0.0)
            return ConnectivityProfile(
                neuron_id=label,
                dataset=dataset,
                upstream_partners=up,
                downstream_partners=down,
                upstream_ranks=compute_ranks(up),
                downstream_ranks=compute_ranks(down),
                upstream_top_k=len(up),
                downstream_top_k=len(down),
                total_upstream_weight=total_up,
                total_downstream_weight=total_down,
                num_neurons_aggregated=1,
                untyped_upstream_count=1 if untyped_up > 0 else 0,
                untyped_downstream_count=1 if untyped_down > 0 else 0,
                untyped_upstream_weight_fraction=(
                    untyped_up / total_up if total_up > 0 else 0.0),
                untyped_downstream_weight_fraction=(
                    untyped_down / total_down if total_down > 0 else 0.0),
                actual_upstream_count=len(up),
                actual_downstream_count=len(down),
                is_weak_connectivity=(len(up) + len(down) - ('untyped' in up)
                                      - ('untyped' in down)) < 5,
                unique_types_upstream=len(up),
                unique_types_downstream=len(down),
                top_k_bodyid_used=0,   # sentinel: pooled profiles are untruncated
                top_m_type_target=0,
            )

        profiles: Dict[str, ConnectivityProfile] = {}
        up_agg = (work[work['lbl_post'].notna()]
                  .groupby(['lbl_post', 'pt_pre'])['weight'].sum())
        down_agg = (work[work['lbl_pre'].notna()]
                    .groupby(['lbl_pre', 'pt_post'])['weight'].sum())
        labels = set(work['lbl_post'].dropna()) | set(work['lbl_pre'].dropna())
        for label in labels:
            up = up_agg.get(label)
            down = down_agg.get(label)
            profiles[str(label)] = finish(
                str(label),
                {} if up is None else up.to_dict(),
                {} if down is None else down.to_dict(),
            )
        return profiles

    @staticmethod
    def compare_profiles(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both',
        weights: Optional[Dict[str, float]] = None,
        score_weights: Optional[Dict[str, float]] = None,
    ) -> ComparisonResult:
        """
        Full comparison between two profiles.
        
        Computes all metrics and returns a ComparisonResult object.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
            weights: Custom score weights (optional)
            score_weights: Alias for weights (compatibility)
        
        Returns:
            ComparisonResult with all metrics
        """
        merged_weights = weights if weights is not None else score_weights
        scores = ProfileComparator.combined_score(profile_a, profile_b, weights=merged_weights, direction=direction)
        rank_union = ProfileComparator.expanded_type_rank_correlation_union(profile_a, profile_b, direction)
        
        # Add note if rank correlation is NaN (due to constant arrays)
        notes = ""
        if np.isnan(scores['rank']):
            notes = "Rank correlation undefined (constant input array)"
        
        return ComparisonResult(
            profile_a_id=str(profile_a.neuron_id),
            profile_b_id=str(profile_b.neuron_id),
            dataset_a=profile_a.dataset,
            dataset_b=profile_b.dataset,
            direction=direction,
            jaccard=scores['jaccard'],
            cosine=scores['cosine'],
            rank_correlation=scores['rank'],
            rank_union=rank_union,
            overlap_a_in_b=scores['overlap_a_in_b'],
            overlap_b_in_a=scores['overlap_b_in_a'],
            weak_connectivity_a=profile_a.is_weak_connectivity,
            weak_connectivity_b=profile_b.is_weak_connectivity,
            notes=notes
        )
    
    @staticmethod
    def compare_profiles_simple(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both'
    ) -> Dict[str, float]:
        """
        Simple comparison between two profiles, returning a dictionary.
        
        This is a lightweight version of compare_profiles() that returns
        a dictionary instead of a ComparisonResult object. Useful for
        bulk comparisons where object creation overhead matters.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Dict with keys: jaccard, cosine, rank_correlation, shared_count
        """
        scores = ProfileComparator.combined_score(profile_a, profile_b, direction=direction)
        
        # Count shared partners
        if direction == 'upstream':
            shared = set(profile_a.upstream_partners.keys()) & set(profile_b.upstream_partners.keys())
        elif direction == 'downstream':
            shared = set(profile_a.downstream_partners.keys()) & set(profile_b.downstream_partners.keys())
        else:
            shared_up = set(profile_a.upstream_partners.keys()) & set(profile_b.upstream_partners.keys())
            shared_down = set(profile_a.downstream_partners.keys()) & set(profile_b.downstream_partners.keys())
            shared = shared_up | shared_down
        
        return {
            'jaccard': scores['jaccard'],
            'cosine': scores['cosine'],
            'rank_correlation': scores['rank'],
            'shared_count': len(shared)
        }
    
    @staticmethod
    def get_partner_overlap_details(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'upstream'
    ) -> pd.DataFrame:
        """
        Detailed breakdown of partner overlap.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream' or 'downstream'
        
        Returns DataFrame:
            | partner_type | in_a | in_b | weight_a | weight_b | rank_a | rank_b |
            |--------------|------|------|----------|----------|--------|--------|
            | Mi1          | True | True | 0.25     | 0.22     | 1      | 2      |
            | Tm3          | True | True | 0.18     | 0.20     | 2      | 1      |
            | L2           | True | False| 0.12     | NaN      | 3      | NaN    |
        """
        if direction == 'upstream':
            partners_a = profile_a.upstream_partners
            partners_b = profile_b.upstream_partners
            ranks_a = profile_a.upstream_ranks
            ranks_b = profile_b.upstream_ranks
        else:
            partners_a = profile_a.downstream_partners
            partners_b = profile_b.downstream_partners
            ranks_a = profile_a.downstream_ranks
            ranks_b = profile_b.downstream_ranks
        
        # Get all partner types
        all_partners = set(partners_a.keys()) | set(partners_b.keys())
        
        if not all_partners:
            # No partners on either side: return an empty detail table with
            # the documented schema (skip sorting, which fails columnless).
            return pd.DataFrame(columns=[
                'partner_type', 'in_a', 'in_b',
                'weight_a', 'weight_b', 'rank_a', 'rank_b', 'status'
            ])
        
        rows = []
        for partner in sorted(all_partners, key=lambda x: str(x)):
            in_a = partner in partners_a
            in_b = partner in partners_b
            
            rows.append({
                'partner_type': partner,
                'in_a': in_a,
                'in_b': in_b,
                'weight_a': partners_a.get(partner, np.nan),
                'weight_b': partners_b.get(partner, np.nan),
                'rank_a': ranks_a.get(partner, np.nan),
                'rank_b': ranks_b.get(partner, np.nan),
                'status': 'shared' if (in_a and in_b) else ('a_only' if in_a else 'b_only')
            })
        
        df = pd.DataFrame(rows)
        
        # Sort by status (shared first) then by average rank
        df['sort_key'] = df.apply(
            lambda r: (0 if r['status'] == 'shared' else 1, 
                       min(r['rank_a'] if not np.isnan(r['rank_a']) else 999,
                           r['rank_b'] if not np.isnan(r['rank_b']) else 999)),
            axis=1
        )
        df = df.sort_values('sort_key').drop(columns=['sort_key'])
        
        return df
    
    @staticmethod
    def compare_multiple_profiles(
        profiles: Dict[str, ConnectivityProfile],
        direction: str = 'both',
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Pairwise comparison of multiple profiles.
        
        Args:
            profiles: Dict mapping identifier to profile
            direction: 'upstream', 'downstream', or 'both'
            weights: Custom score weights (optional)
        
        Returns:
            DataFrame with pairwise comparison results
        """
        results = []
        keys = list(profiles.keys())
        
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                key_a, key_b = keys[i], keys[j]
                profile_a, profile_b = profiles[key_a], profiles[key_b]
                
                result = ProfileComparator.compare_profiles(
                    profile_a, profile_b, direction, weights
                )
                
                result_dict = result.to_dict()
                result_dict['pair'] = f"{key_a} vs {key_b}"
                results.append(result_dict)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def direct_comparison(
        neurons_a: Union[str, int, List[Union[str, int]]],
        neurons_b: Union[str, int, List[Union[str, int]]],
        dataset_a: Optional[str] = None,
        dataset_b: Optional[str] = None,
        # Alternative names to align with callers (e.g., HomologFinder)
        source_dataset: Optional[str] = None,
        target_dataset: Optional[str] = None,
        profiler: Optional[Any] = None,
        direction: str = 'both',
        comparison_mode: str = 'loose',
        label_mapper: Optional[Any] = None,
        score_weights: Optional[Dict[str, float]] = None,
        top_k: int = 25,
        top_m: int = 5,
        min_synapse_threshold: int = 3,
        include_untyped_partners: bool = True,
        min_common_partners: Optional[int] = None,
        same_label_only: bool = False,
        type_mapper: Optional[Any] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Direct comparison of specific neurons between datasets.
        
        This method provides a unified interface for comparing neurons by:
        - Type name (string)
        - BodyId (integer)
        - LabelMapper (cross-dataset name resolution)
        
        Args:
            neurons_a: Neuron(s) in dataset_a (type names, bodyIds, or list)
            neurons_b: Neuron(s) in dataset_b (type names, bodyIds, or list)
                       If None, uses neurons_a with label_mapper for name resolution
            dataset_a / source_dataset: First dataset identifier
            dataset_b / target_dataset: Second dataset identifier
            profiler: ConnectivityProfiler instance (created if None)
            direction: 'upstream', 'downstream', or 'both'
            comparison_mode: 'loose' (type-level) or 'strict' (bodyId-level). Legacy
                             values 'type' -> 'loose', 'bodyid' -> 'strict'.
            label_mapper: Optional LabelMapper for cross-dataset type name resolution
            score_weights: Custom weights {'jaccard': 0.5, 'rank': 0.5}
            top_k: Number of top partners per direction
            top_m: Minimum unique partner types
            min_synapse_threshold: Minimum synapse count
            include_untyped_partners: Include partners without type
            min_common_partners: Minimum shared partners when using strict/type path
            verbose: Print progress messages
        
        Returns:
            Dict with keys:
            - 'results': DataFrame with pairwise comparison results
            - 'profiles_a': Dict of profiles from dataset_a
            - 'profiles_b': Dict of profiles from dataset_b
            - 'type_summary': Aggregated per-type statistics (strict/type path)
            - 'bodyid_results': BodyId-level pairs (strict/type path)
            - 'summary': Summary statistics
            - 'comparison_mode': Mode used ('loose' or 'strict')
        
        Example:
            >>> # Compare by type name
            >>> results = ProfileComparator.direct_comparison(
            ...     'aMe12', 'aMe12',
            ...     'flywire_FAFB_v783', 'male-cns:v0.9'
            ... )
            >>> 
            >>> # Compare by bodyId
            >>> results = ProfileComparator.direct_comparison(
            ...     [720575940609627403, 720575940622446106],
            ...     [12517, 13190],
            ...     'flywire_FAFB_v783', 'male-cns:v0.9',
            ...     comparison_mode='strict'
            ... )
        """
        from .connectivity_profiler import ConnectivityProfiler, ProfilerConfig
        
        def _log(msg):
            if verbose:
                print(f"[DirectComparison] {msg}")

        # Resolve dataset names, honoring both legacy and new parameter names
        dataset_a = dataset_a or source_dataset
        dataset_b = dataset_b or target_dataset
        if not dataset_a or not dataset_b:
            raise ValueError("Both dataset_a/source_dataset and dataset_b/target_dataset must be provided")
        
        # Normalize inputs to lists
        if not isinstance(neurons_a, list):
            neurons_a = [neurons_a]
        if neurons_b is None:
            # Use neurons_a with label_mapper for name resolution
            neurons_b = neurons_a
        if not isinstance(neurons_b, list):
            neurons_b = [neurons_b]

        # Determine whether inputs are bodyIds (ints) or type names (str)
        def _is_bodyid(x: Any) -> bool:
            return isinstance(x, int)

        all_bodyid_input = all(_is_bodyid(x) for x in neurons_a + neurons_b)

        # Normalize comparison mode
        mode = (comparison_mode or 'loose').lower()
        if mode == 'type':
            mode = 'loose'
        if mode == 'bodyid':
            mode = 'strict'
        if all_bodyid_input:
            mode = 'strict'
        
        # Create profiler if not provided
        if profiler is None:
            config = ProfilerConfig(
                top_k_bodyid=top_k,
                top_m_type=top_m,
                min_synapse_threshold=min_synapse_threshold,
                include_untyped_partners=include_untyped_partners,
                use_cache=True
            )
            profiler = ConnectivityProfiler(
                datasets=[dataset_a, dataset_b],
                config=config,
                verbose=verbose
            )
        
        # Ensure connection caches are complete and pre-warm profiles
        # This mimics HomologFinder.find_homologs_fast progress
        def _ensure_cache_and_prewarm(dataset_name):
            if not dataset_name:
                return
                
            # 1. Ensure connection cache is complete (using FindNeuronConnection)
            try:
                # Try relative import first
                try:
                    from ..coana import FindNeuronConnection
                except ImportError:
                    from coana import FindNeuronConnection
            except ImportError:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from coana import FindNeuronConnection
            
            if verbose:
                print(f"[DirectComparison] Ensuring connection cache for {dataset_name}...")
            
            # Use token from profiler if available
            token = getattr(profiler, 'token', None)
            
            fnc = FindNeuronConnection(
                dataset=dataset_name,
                token=token,
                use_cache=True,
                verbose_mode='simple',
                simple_fetch=True
            )
            # Build connection cache
            fnc.build_connection_cache(batch_size=1000, quiet=not verbose)
            
            # Cleanup FNC to free memory
            fnc._conn_df_cache = None
            del fnc
            import gc
            gc.collect()
            
            # 2. Pre-warm profile cache
            if verbose:
                print(f"[DirectComparison] Pre-warming profile cache for {dataset_name}...")
            try:
                profiler.consolidate_profile_cache(dataset_name)
            except Exception:
                pass
            try:
                profiler._load_cache_dataframe(dataset_name)
            except Exception:
                pass
            try:
                profiler._get_cached_conn_df(dataset_name)
            except Exception:
                pass

        # Run for both datasets
        _ensure_cache_and_prewarm(dataset_a)
        if dataset_b != dataset_a:
            _ensure_cache_and_prewarm(dataset_b)

        weights = score_weights or DEFAULT_SCORE_WEIGHTS
        
        # Resolve neuron names via label_mapper if provided
        resolved_a = []
        resolved_b = []
        for na in neurons_a:
            if label_mapper and isinstance(na, str):
                mapped = label_mapper.get_mapped_label(na, dataset_a)
                resolved_a.append(mapped if mapped else na)
            else:
                resolved_a.append(na)
        
        for nb in neurons_b:
            if label_mapper and isinstance(nb, str):
                mapped = label_mapper.get_mapped_label(nb, dataset_b)
                resolved_b.append(mapped if mapped else nb)
            else:
                resolved_b.append(nb)
        _log(f"Comparing {len(resolved_a)} neuron(s) from {dataset_a} vs {len(resolved_b)} from {dataset_b} (mode={mode})")

        # Strict mode on type names: delegate to bodyId core for per-bodyId
        # coverage — but only WITHIN one dataset. The bodyId core scores
        # partners by bodyId identity (combined_score_intra_dataset), a key
        # that is only meaningful inside one connectome's ID space; across
        # two datasets, colliding integer bodyIds would count unrelated
        # neurons as shared partners (F-XD-001), so cross-dataset strict
        # falls through to the type-level standardized comparison below.
        same_dataset = (canonical_dataset_name(str(dataset_a))
                        == canonical_dataset_name(str(dataset_b)))
        if (mode == 'strict' and not all_bodyid_input and same_dataset
                and hasattr(ProfileComparator, 'compare_types_bodyid_core')):
            strict_types = sorted(set(str(t) for t in resolved_a) & set(str(t) for t in resolved_b))
            if strict_types:
                bodyid_df, type_summary = ProfileComparator.compare_types_bodyid_core(
                    profiler=profiler,
                    neuron_types=strict_types,
                    dataset_a=dataset_a,
                    dataset_b=dataset_b,
                    direction=direction,
                    min_common_partners=min_common_partners or 0,
                    score_weights=weights,
                    verbose=verbose,
                )

                rows = []
                for _, row in type_summary.iterrows():
                    rank_corr = row.get('avg_rank_corr', np.nan)
                    rank_union = row.get('avg_rank_union', np.nan)
                    rows.append({
                        'neuron_a': row['neuron_type'],
                        'neuron_b': row['neuron_type'],
                        'type_a': row['neuron_type'],
                        'type_b': row['neuron_type'],
                        'dataset_a': dataset_a,
                        'dataset_b': dataset_b,
                        'direction': direction,
                        'rank_corr': rank_corr,
                        'rank_union': rank_union,
                        'jaccard': row.get('avg_jaccard', np.nan),
                        'cosine': row.get('avg_cosine', np.nan),
                        'shared_type_count': np.nan,
                        'union_type_count': np.nan,
                        'is_same_type': True,
                        'n_source_bodyIds': row.get('n_source_bodyIds', np.nan),
                        'n_target_bodyIds': row.get('n_target_bodyIds', np.nan),
                    })

                results_df = pd.DataFrame(rows)
                summary = {
                    'n_types': len(strict_types),
                    'avg_rank_corr': results_df['rank_corr'].mean() if not results_df.empty else np.nan,
                    'avg_jaccard': results_df['jaccard'].mean() if not results_df.empty else np.nan,
                }

                return {
                    'results': (results_df.sort_values(
                        ['jaccard', 'rank_union'], ascending=[False, False],
                        na_position='last')
                        if not results_df.empty else results_df),
                    'profiles_a': {},
                    'profiles_b': {},
                    'type_summary': type_summary,
                    'bodyid_results': bodyid_df,
                    'summary': summary,
                    'comparison_mode': mode,
                }

        if mode == 'strict' and not all_bodyid_input and not same_dataset:
            _log("Strict mode across two datasets: per-bodyId scoring is not "
                 "defined across ID spaces — comparing at the type level "
                 "with standardized partner names instead.")

        # Build profiles for loose mode or bodyId-driven strict comparisons
        profiles_a: Dict[str, ConnectivityProfile] = {}
        profiles_b: Dict[str, ConnectivityProfile] = {}

        _log(f"Building profiles for dataset_a ({dataset_a})...")
        for neuron in resolved_a:
            try:
                profile = profiler.get_profile(neuron, dataset_a)
                if profile is not None:
                    key = str(neuron)
                    profiles_a[key] = profile
                else:
                    _log(f"  Warning: No profile found for {neuron} in {dataset_a}")
            except Exception as e:
                _log(f"  Error getting profile for {neuron}: {e}")

        _log(f"Building profiles for dataset_b ({dataset_b})...")
        for neuron in resolved_b:
            try:
                profile = profiler.get_profile(neuron, dataset_b)
                if profile is not None:
                    key = str(neuron)
                    profiles_b[key] = profile
                else:
                    _log(f"  Warning: No profile found for {neuron} in {dataset_b}")
            except Exception as e:
                _log(f"  Error getting profile for {neuron}: {e}")

        if not profiles_a or not profiles_b:
            _log("Warning: Insufficient profiles for comparison")
            return {
                'results': pd.DataFrame(),
                'profiles_a': profiles_a,
                'profiles_b': profiles_b,
                'summary': {},
                'comparison_mode': mode
            }

        # Pairwise comparison
        _log(f"Computing pairwise comparisons ({len(profiles_a)} x {len(profiles_b)})...")
        rows = []

        # Prepare candidate map for batch comparison (all profiles_b are candidates)
        # Use 0 for shared_count as we don't have adjacency info here
        candidate_map = {k: 0 for k in profiles_b.keys()}

        for key_a, profile_a in profiles_a.items():
            # Use batch comparison for efficiency and consistency with HomologFinder
            # This ensures we use the exact same metric calculation logic
            batch_results = ProfileComparator.batch_compare_cross_dataset(
                source_profile=profile_a,
                target_profiles=profiles_b,
                candidate_map=candidate_map,
                direction=direction,
                type_mapper=type_mapper
            )
            
            # Handle case where source profile is empty (batch_results will be empty)
            if not batch_results:
                # Check if source is empty
                source_types = ProfileComparator._get_expanded_types(profile_a, direction)
                if len(source_types) == 0:
                    # Add NaN rows for all targets to indicate invalid comparison
                    for key_b, profile_b in profiles_b.items():
                        type_a = getattr(profile_a, 'neuron_type', None) or key_a
                        type_b = getattr(profile_b, 'neuron_type', None) or key_b
                        
                        if same_label_only and type_a != type_b:
                            continue
                            
                        rows.append({
                            'neuron_a': key_a,
                            'neuron_b': key_b,
                            'type_a': type_a,
                            'type_b': type_b,
                            'dataset_a': dataset_a,
                            'dataset_b': dataset_b,
                            'direction': direction,
                            'rank_corr': np.nan,
                            'rank_union': np.nan,
                            'jaccard': np.nan,
                            'cosine': np.nan,
                            'shared_type_count': 0,
                            'union_type_count': 0,
                            'is_same_type': type_a == type_b
                        })
                continue

            for res in batch_results:
                key_b = res['target_bid']
                profile_b = profiles_b[key_b]
                
                # Determine if same type (for reference)
                type_a = getattr(profile_a, 'neuron_type', None) or key_a
                type_b = getattr(profile_b, 'neuron_type', None) or key_b
                if same_label_only and type_a != type_b:
                    continue
                is_same_type = type_a == type_b

                rank_corr = res['rank']
                rank_union = res['rank_union']
                jaccard = res['jaccard']

                rows.append({
                    'neuron_a': key_a,
                    'neuron_b': key_b,
                    'type_a': type_a,
                    'type_b': type_b,
                    'dataset_a': dataset_a,
                    'dataset_b': dataset_b,
                    'direction': direction,
                    'rank_corr': rank_corr,
                    'rank_union': rank_union,
                    'jaccard': jaccard,
                    'cosine': res['cosine'],
                    'shared_type_count': res['shared_type_count'],
                    'union_type_count': res['union_type_count'],
                    'is_same_type': is_same_type
                })

        results_df = pd.DataFrame(rows)
        if not results_df.empty:
            # bodyId-level ordering convention (plan-tmvev-jaccard-primary-
            # bodyid-ranking, user 2026-09-20): Jaccard first with rank_union
            # as the tie-break, the same chain TM VEV ranks on
            results_df = results_df.sort_values(
                ['jaccard', 'rank_union'], ascending=[False, False],
                na_position='last')

        # Build summary
        summary = {}
        if not results_df.empty:
            summary = {
                'n_comparisons': len(results_df),
                'n_neurons_a': len(profiles_a),
                'n_neurons_b': len(profiles_b),
                'avg_rank_corr': results_df['rank_corr'].mean(),
                'avg_jaccard': results_df['jaccard'].mean(),
                'same_type_matches': results_df['is_same_type'].sum(),
            }
            _log(f"Comparison complete: {summary['n_comparisons']} pairs, avg_rank_corr={summary['avg_rank_corr']:.3f}")

        return {
            'results': results_df,
            'profiles_a': profiles_a,
            'profiles_b': profiles_b,
            'summary': summary,
            'comparison_mode': mode
        }
    
    @staticmethod
    def build_similarity_matrix(
        profiles: Dict[str, ConnectivityProfile],
        metric: str = 'rank',
        direction: str = 'both'
    ) -> pd.DataFrame:
        """
        Build similarity matrix for multiple profiles.
        
        Args:
            profiles: Dict mapping identifier to profile
            metric: 'jaccard', 'cosine', or 'rank'
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            DataFrame with similarity matrix (symmetric)
        """
        keys = list(profiles.keys())
        n = len(keys)
        
        # Initialize matrix
        matrix = np.zeros((n, n))
        
        for i in range(n):
            matrix[i, i] = 1.0  # Self-similarity
            for j in range(i + 1, n):
                profile_a = profiles[keys[i]]
                profile_b = profiles[keys[j]]
                
                if metric == 'jaccard':
                    sim = ProfileComparator.jaccard_similarity(profile_a, profile_b, direction)
                elif metric == 'cosine':
                    sim = ProfileComparator.weighted_cosine_similarity(profile_a, profile_b, direction)
                elif metric == 'rank':
                    sim = ProfileComparator.rank_correlation(profile_a, profile_b, direction)
                else:
                    sim = ProfileComparator.rank_correlation(profile_a, profile_b, direction)
                
                matrix[i, j] = sim
                matrix[j, i] = sim
        
        return pd.DataFrame(matrix, index=keys, columns=keys)
    
    # ========================================================================
    # Inter-Type Rank Correlation (Cross-Dataset and Intra-Dataset)
    # ========================================================================
    
    @staticmethod
    def compute_inter_type_rank_correlation_matrix(
        profiles_by_dataset: Dict[str, Dict[str, ConnectivityProfile]],
        metric: str = 'rank',
        direction: str = 'both'
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute inter-type rank correlation matrices.
        
        Builds similarity matrices comparing connectivity profiles between 
        different neuron types, both within datasets and across datasets.
        
        Args:
            profiles_by_dataset: Dict of {dataset: {type_name: ConnectivityProfile}}
            metric: Similarity metric to use ('rank', 'jaccard', 'cosine')
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Dict containing:
            - 'cross_dataset': DataFrame with cross-dataset pairwise similarities
            - 'intra_dataset': Dict of {dataset: DataFrame} with intra-dataset matrices
            - 'all_types_matrix': Combined matrix of all types across all datasets
        """
        results = {
            'cross_dataset': None,
            'intra_dataset': {},
            'all_types_matrix': None
        }
        
        if not profiles_by_dataset:
            return results
        
        # 1. Build intra-dataset matrices
        for dataset, profiles in profiles_by_dataset.items():
            if len(profiles) >= 2:
                matrix = ProfileComparator.build_similarity_matrix(profiles, metric=metric, direction=direction)
                results['intra_dataset'][dataset] = matrix
        
        # 2. Build cross-dataset inter-type comparison
        # Collect all profiles with dataset-prefixed keys
        all_profiles: Dict[str, ConnectivityProfile] = {}
        for dataset, profiles in profiles_by_dataset.items():
            for type_name, profile in profiles.items():
                key = f"{dataset}:{type_name}"
                all_profiles[key] = profile
        
        if len(all_profiles) >= 2:
            # Build combined matrix
            results['all_types_matrix'] = ProfileComparator.build_similarity_matrix(
                all_profiles, metric=metric, direction=direction
            )
            
            # Build cross-dataset specific comparisons
            cross_dataset_rows = []
            datasets = list(profiles_by_dataset.keys())
            
            for i, ds1 in enumerate(datasets):
                for ds2 in datasets[i+1:]:
                    # Compare each type in ds1 with each type in ds2
                    for type1, profile1 in profiles_by_dataset[ds1].items():
                        for type2, profile2 in profiles_by_dataset[ds2].items():
                            if metric == 'rank':
                                sim = ProfileComparator.rank_correlation(profile1, profile2, direction)
                            elif metric == 'jaccard':
                                sim = ProfileComparator.jaccard_similarity(profile1, profile2, direction)
                            elif metric == 'cosine':
                                sim = ProfileComparator.weighted_cosine_similarity(profile1, profile2, direction)
                            else:
                                sim = ProfileComparator.rank_correlation(profile1, profile2, direction)
                            
                            cross_dataset_rows.append({
                                'dataset_1': ds1,
                                'type_1': type1,
                                'dataset_2': ds2,
                                'type_2': type2,
                                'similarity': sim,
                                'same_type': type1 == type2
                            })
            
            if cross_dataset_rows:
                results['cross_dataset'] = pd.DataFrame(cross_dataset_rows)
        
        return results


# ============================================================================
# NeuPrint Dataset Availability
# ============================================================================

# Cached per-server dataset listing: server -> (timestamp, dataset list).
_NEUPRINT_AVAILABLE_CACHE: Dict[str, Tuple[float, List[str]]] = {}


def fetch_neuprint_available_datasets(
    server: str = 'https://neuprint.janelia.org',
    token: Optional[str] = None,
    max_age_seconds: float = 300.0,
) -> Optional[List[str]]:
    """Return the datasets hosted by a NeuPrint server, minus hidden ones.

    The server's ``/api/dbmeta/datasets`` endpoint also lists hidden
    datasets (e.g. ``banc:v888``) that are not queryable through the API;
    those are the only entries excluded.  Returns ``None`` when the listing
    cannot be obtained (no token, network failure), so callers can fall
    back to attempting the requested dataset directly.
    """
    now = time.time()
    cached = _NEUPRINT_AVAILABLE_CACHE.get(server)
    if cached is not None and (now - cached[0]) < max_age_seconds:
        return list(cached[1])

    if not token:
        try:
            from ..utils.token_manager import token_manager
        except ImportError:
            try:
                from utils.token_manager import token_manager
            except ImportError:
                token_manager = None
        if token_manager is not None:
            token = token_manager.get_neuprint_token()
    if not token:
        return None

    try:
        import requests
        response = requests.get(
            f"{server}/api/dbmeta/datasets",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-type": "application/json",
            },
            timeout=15,
        )
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict):
                available = sorted(
                    name for name, meta in data.items()
                    if not (isinstance(meta, dict)
                            and str(meta.get("hidden", "")).lower() == "true")
                )
                # Only successful listings are cached: a transient network
                # failure must not poison later calls within the TTL.
                _NEUPRINT_AVAILABLE_CACHE[server] = (now, available)
                return list(available)
    except Exception:
        pass

    return None


class HomologFinder:
    """
    A class for finding homologs of neurons across datasets using connectivity profiles.
    
    This class provides a unified interface for homolog discovery, supporting:
    - Query by type name or bodyId
    - Search across multiple datasets
    - Novel homolog discovery within the same dataset
    - BodyId-level comparisons with automatic type-level aggregation
    - Fast finding via adjacency expansion for candidate discovery
    - Optional 3D skeleton visualization of top candidates
    
    Profile Rules (consistent with ConnectivityProfiler):
        - top_k_bodyid: Top K partners per direction (default: 15)
        - top_m_type: Minimum unique types to ensure via dynamic expansion (default: 5)
        - Dynamic expansion: If fewer than top_m types after top_k, expand K until M types reached
        - max_expansion_factor: Maximum K = top_k * max_expansion_factor (default: 5)
        - Profiles use the ConnectivityProfiler's 3-tier cache system
    
    Output (always both levels):
        - bodyid_results.csv: BodyId-level comparisons (sorted by source_bodyId, rank_corr)
        - type_summary.csv: Type-mean aggregated FROM bodyId-level results
        - type_level_results.csv: True type-level ranking (pooled type profiles)
        - homolog_results.csv: Legacy format (sorted by rank_corr only)
    
    Visualization:
        - Optional 3D skeleton visualization of top-n candidates
        - Uses VisualizeSkeleton module for interactive HTML + PNG export
    
    Cache Integration:
        - Uses ConnectivityProfiler's get_profile() for cache-aware profile building
        - Profiles are stored in 3-tier cache: memory → disk-index → disk-df (parquet)
        - Fast mode builds profiles from pre-aggregated connection data
    
    Example:
        >>> from src.comparison import HomologFinder
        >>> 
        >>> # Initialize finder with visualization enabled
        >>> finder = HomologFinder(
        ...     top_k=15,
        ...     top_m=5,
        ...     visualize_skeleton=True,  # Enable 3D visualization
        ...     visualize_top_n=5         # Visualize top 5 candidates
        ... )
        >>> 
        >>> # Find homologs with automatic visualization
        >>> results = finder.find_homologs_fast(
        ...     query='Mi1',
        ...     source_dataset='hemibrain_v1_2_1',
        ...     target_dataset='hemibrain_v1_2_1',
        ...     output_dir='./results'
        ... )
        >>> 
        >>> # Results include both bodyId-level and type-level files
        >>> # Plus visualization/ folder with 3D plots
    """

    def __init__(
        self,
        source: Optional[Union[str, int]] = None,
        source_dataset: Optional[str] = None,
        target_dataset: Optional[str] = None,
        output_dir: Optional[str] = None,
        saveas: Optional[str] = None,
        top_n: int = 100,
        top_k: int = 25,
        top_m: int = 5,
        min_synapse_threshold: int = 3,
        include_untyped_partners: bool = True,
        use_cache: bool = True,
        visualize_skeleton: bool = False,
        visualize_top_n: int = 5,
        visualization_settings: Optional[Dict[str, Any]] = None,
        similarity_metric: Union[str, Dict[str, float]] = 'jaccard',
        score_weights: Optional[Dict[str, float]] = None,
        verbose: bool = True,
        token: str = '',
        vector_prefiltering: bool = False,
        min_shared_partners: int = 2,
        vector_prune_fraction: float = 0.05,
        prefilter_rank_metric: str = 'cosine',
        use_auto_type_mapping: bool = True,
        ensure_cache_complete: bool = False,
        output_folder_prefix: str = 'homologs',
    ):
        """
        Initialize HomologFinder with configuration and default parameters.
        
        Module-Level Defaults:
            These attributes are used as defaults by find_homologs() and 
            find_homologs_fast() when not provided as method arguments:
            - source: Query neuron (type name or bodyId)
            - source_dataset: Source dataset
            - target_dataset: Target dataset to search
            - output_dir: Directory to save results
            - saveas: Custom folder name for results
        
        Profile Construction Rules:
            - top_k: Top K partners by weight per direction
            - top_m: Minimum unique partner types to ensure
            - Dynamic expansion: If top_k yields < top_m types, expand K
            - Maximum K = top_k * 5 (max_expansion_factor)
        
        Visualization:
            - visualize_skeleton: If True, generate 3D skeleton plots for top matches
            - visualize_top_n: Number of top candidates to visualize
        
        Args:
            source: Default query neuron (type name or bodyId)
            source_dataset: Default source dataset
            target_dataset: Default target dataset
            output_dir: Default directory to save results
            saveas: Default folder name (auto-generates timestamp if None)
            top_n: Maximum number of candidates to return per source neuron (default: 20)
            top_k: Number of top partners to include per direction (default: 15)
            top_m: Minimum unique partner types to ensure via expansion (default: 5)
            min_synapse_threshold: Minimum synapse count for connections
            include_untyped_partners: Include partners without type annotations
            use_cache: Enable profile caching (uses ConnectivityProfiler cache)
            visualize_skeleton: Generate 3D skeleton visualizations (default: False)
            visualize_top_n: Number of top candidates to visualize (default: 5)
            similarity_metric: Metric for sorting top-N candidates:
                - str: One of 'rank_union', 'rank_corr', 'jaccard', 'cosine'
                - dict: Accepted for backward compatibility; no longer selects a
                  weighted 'combined' score (the comparator falls back to rank_corr).
            score_weights: (Deprecated) Retained for backward compatibility only.
            verbose: Print progress messages
            token: API token for NeuPrint (if empty, FNC auto-handles from env vars)
            vector_prefiltering: Keep only the most similar candidates before
                full scoring (default False).
            min_shared_partners: Minimum shared partners for adjacency-expansion
                candidate discovery (default: 2). Lower = looser search
                (1 = any single shared partner makes a candidate).
            vector_prune_fraction: Fraction of cosine-positive candidates kept by
                vector pre-filtering (default: 0.05 = top 5%).
                1.0 keeps ALL cosine-positive candidates (loosest search).
            prefilter_rank_metric: Ranking key used by the pre-filter prune:
                'cosine' (default) keeps candidates with the highest weighted
                profile cosine; 'adjacency' ranks by the shared partner-type
                count (legacy behaviour).
            use_auto_type_mapping: Enable automatic type mapping for cross-dataset
                comparison (default: True). When enabled, partner types resolve
                through the shared validity-aware resolver (comparison.type_resolver):
                licensed renames and valid splits map, conflicts fail closed (never
                matched by raw same-name), and unmapped types keep their raw name as
                the counted long-tail fallback. This allows proper matching of types
                with different names across datasets (e.g., 'MTe07' in FAFB → 'MeVPLo2'
                in male-cns). For intra-dataset comparison, original types are always
                used. Each search writes auto_type_mapping.json with the mapper
                provenance and per-status resolution counts.
            ensure_cache_complete: If True, build/complete the FULL dataset connection
                cache before searching (fetches connections for every uncached neuron).
                This can take hours on first use with a new dataset. Default False:
                the search fetches only the connections it needs (recommended for
                the UI and normal first-time use).
            output_folder_prefix: Prefix for auto-generated result folders.
                The Homolog Finding tab uses ``homologs``; Similarity's
                connectivity mode passes ``similar-connectivity``. Custom
                ``saveas`` values are unaffected.
        
        Example:
            >>> # Set up finder with visualization
            >>> finder = HomologFinder(
            ...     source='Mi1',
            ...     source_dataset='hemibrain:v1.2.1',
            ...     target_dataset='hemibrain:v1.2.1',
            ...     output_dir='./results',
            ...     visualize_skeleton=True,
            ...     visualize_top_n=5
            ... )
            >>> 
            >>> # Run search - always saves both bodyId and type-level results
            >>> results = finder.find_homologs_fast()
        """
        from .connectivity_profiler import ConnectivityProfiler, ProfilerConfig
        
        # Module-level default parameters
        self.source = source
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.token = token  # API token passed to FNC
        self.top_n = top_n
        
        # Output directory - use default if not specified
        # Default: <project root>/local_data/homolog_finding/
        if output_dir is None:
            self.output_dir = str(
                Path(__file__).resolve().parents[2]
                / 'local_data'
                / 'homolog_finding'
            )
        else:
            self.output_dir = output_dir
        # Normalize empty strings to None so the auto-generated per-run folder
        # (homologs_..._timestamp) is used when no custom name is given.
        self.saveas = saveas or None
        self.output_folder_prefix = (
            str(output_folder_prefix or 'homologs').strip() or 'homologs'
        )
        
        # Visualization settings
        self.visualize_skeleton = visualize_skeleton
        self.visualize_top_n = visualize_top_n
        self.visualization_settings = dict(visualization_settings or {})
        # Programmatic callers that do not use the UI still get the same
        # dataset-aligned template brain default as the analysis tabs.
        self.visualization_settings.setdefault('brain_mesh', 'native')
        self.vector_prefiltering = vector_prefiltering

        # Loose-search knobs: candidate discovery requires at least
        # min_shared_partners shared partners; vector pre-filtering keeps the
        # top vector_prune_fraction of candidates among the cosine-positive
        # set, ranked by prefilter_rank_metric ('cosine' = weighted profile
        # cosine, 'adjacency' = legacy shared partner-type count).
        self.min_shared_partners = min_shared_partners
        self.vector_prune_fraction = vector_prune_fraction
        metric_norm = str(prefilter_rank_metric or 'cosine').strip().lower()
        if metric_norm not in {'cosine', 'adjacency'}:
            raise ValueError(
                f"prefilter_rank_metric must be 'cosine' or 'adjacency', got "
                f"{prefilter_rank_metric!r}"
            )
        self.prefilter_rank_metric = metric_norm

        # Auto type mapping for cross-dataset comparison
        # When enabled, partner types resolve via the shared validity-aware
        # resolver (conflicts fail closed, splits expand, unmapped fall back to raw)
        # This allows proper matching of types like 'MTe07' (FAFB) ↔ 'MeVPLo2' (male-cns)
        self.use_auto_type_mapping = use_auto_type_mapping
        self._type_mapper: Optional[CrossDatasetTypeMapper] = None
        # Per-run mapping-resolution statuses (candidate expansion), exported
        # with auto_type_mapping.json and consumed between saves: each saved
        # run snapshots and clears the counter, so its counts are its own.
        self._mapping_status_counts: Dict[str, int] = {}
        # One run-scoped mapper snapshot (lazy; reset per search) and a
        # unique-resolution dedupe set for candidate-expansion counting.
        self._mapper_snapshot = None
        self._counted_expansion_resolutions: set = set()
        
        # Similarity metric for sorting.  Escalation can still pass a dict for
        # backward compatibility, but it no longer selects a 'combined' score;
        # the comparator therefore falls back to the raw rank_corr sort.
        if isinstance(similarity_metric, dict):
            self.similarity_metric = 'rank_corr'
        else:
            self.similarity_metric = similarity_metric
        self.score_weights = score_weights if score_weights else DEFAULT_SCORE_WEIGHTS
        
        # Profiler configuration
        self.verbose = verbose
        self.use_cache = use_cache
        self.ensure_cache_complete = ensure_cache_complete
        self.min_synapse_threshold = min_synapse_threshold
        
        config = ProfilerConfig(
            top_k_bodyid=top_k,
            top_m_type=top_m,
            min_synapse_threshold=min_synapse_threshold,
            include_untyped_partners=include_untyped_partners,
            use_cache=use_cache
        )
        
        self.profiler = ConnectivityProfiler(
            datasets=[],  # Empty - datasets determined at query time
            config=config,
            verbose=verbose
        )
        
        # Cache references to profiler's caches
        # We use the profiler's cache system directly for consistency
        
        # Cache for available types per dataset
        self._types_cache: Dict[str, List[str]] = {}
        
        # Cache for bodyIds per type per dataset
        self._bodyids_by_type_cache: Dict[str, Dict[str, List[int]]] = {}
        
        # Cache for loaded connection DataFrames
        self._conn_cache: Dict[str, pd.DataFrame] = {}
        
        # Cache for FindNeuronConnection instances per dataset
        self._fnc_cache: Dict[str, Any] = {}
        self._in_progress_bar = False  # Track if we're inside a progress bar context
        self._batch_size = 1000  # Batch size for profile building with intermediate saves
        
        # Initialize clients for datasets
        self.clients = {}
        
        # Helper to get or create client
        def get_client(dataset):
            if not dataset: return None
            if dataset in self.clients: return self.clients[dataset]
            
            # FAFB and standalone BANC live in local release stores, not on
            # the NeuPrint server.  Skip client creation instead of
            # surfacing a misleading "dataset does not exist on the neuprint
            # server" warning.
            if is_local_connectome_dataset(dataset):
                if self.verbose:
                    source = "BANC public release" if is_banc_dataset(dataset) else "FAFB local release"
                    print(
                        f"[HomologFinder] {dataset} uses the {source}; "
                        "no NeuPrint client needed"
                    )
                return None
            
            try:
                from neuprint import Client
            except ImportError:
                if self.verbose:
                    print(
                        f"[HomologFinder] Warning: neuprint package is not "
                        f"installed; skipping client for {dataset}"
                    )
                return None
            
            # Pre-screen against the datasets the NeuPrint server actually
            # hosts.  Only server-hidden entries (e.g. banc:v888) are
            # excluded from that listing; unknown datasets get a short
            # warning instead of the client constructor's long error dump.
            available = fetch_neuprint_available_datasets(token=self.token)
            if available is not None and dataset not in available:
                if self.verbose:
                    print(
                        f"[HomologFinder] Warning: Dataset '{dataset}' does not "
                        f"exist on the NeuPrint server. "
                        f"Available datasets: {available}"
                    )
                return None
            
            try:
                # Try to use token if provided
                if self.token:
                    c = Client('https://neuprint.janelia.org', dataset=dataset, token=self.token)
                else:
                    # Try environment or default
                    c = Client('https://neuprint.janelia.org', dataset=dataset)
                c.fetch_version()
                self.clients[dataset] = c
                return c
            except Exception as e:
                if self.verbose:
                    # Drop the server's long "Available datasets: [...]" dump
                    # from constructor errors so the warning stays readable.
                    msg = str(e).split('Available datasets:')[0].strip()
                    print(f"[HomologFinder] Warning: Could not initialize client for {dataset}: {msg}")
                return None

        # Initialize clients for source and target
        if self.source_dataset:
            get_client(self.source_dataset)
        if self.target_dataset and self.target_dataset != self.source_dataset:
            get_client(self.target_dataset)
            
        # Set self.client to source client for backward compatibility / default usage
        self.client = self.clients.get(self.source_dataset) or self.clients.get(self.target_dataset)
    
    def _log(self, msg: str):
        """Print message if verbose. Uses tqdm.write() if inside a progress bar."""
        if self.verbose:
            if self._in_progress_bar:
                tqdm.write(f"[HomologFinder] {msg}")
            else:
                print(f"[HomologFinder] {msg}")

    def _progress(self, step: int, total: int, label: str = ""):
        """Emit a structured step-progress event consumed by the web UI.

        The line is a control event (determinate bar + step label in the
        results panel), not log output. Uses tqdm.write() inside a progress
        bar so the bar is redrawn after the event.
        """
        if self.verbose:
            msg = f"[DROCAT][progress] {int(step)}/{int(total)} {label}".rstrip()
            if self._in_progress_bar:
                tqdm.write(msg)
            else:
                print(msg, flush=True)

    def _get_type_mapper_for_comparison(self, is_cross_dataset: bool) -> Optional[CrossDatasetTypeMapper]:
        """
        Get the type mapper for cross-dataset comparison if enabled.
        
        Returns None if:
        - use_auto_type_mapping is False
        - is_cross_dataset is False (intra-dataset uses original types/bodyIds)
        
        Returns:
            CrossDatasetTypeMapper if enabled for cross-dataset, None otherwise.
        """
        if not self.use_auto_type_mapping:
            return None
        
        if not is_cross_dataset:
            # Intra-dataset comparison: use original types and bodyIds
            return None
        
        # Cross-dataset: get or create type mapper
        if self._type_mapper is None:
            self._type_mapper = get_type_mapper()
            if self._type_mapper._loaded:
                self._log(f"Using auto type mapping for cross-dataset comparison")
                self._log(f"  Partner types resolve via the shared validity-aware resolver (conflicts fail closed, splits expand, unmapped fall back to raw)")
        
        return self._type_mapper if self._type_mapper._loaded else None

    def _compute_same_type_candidates(
        self,
        query_type_names,
        target_type_lookup: Dict[int, str],
        type_mapper: Optional[CrossDatasetTypeMapper],
        target_dataset: str,
        source_dataset: Optional[str] = None,
    ) -> set:
        """Target bodyIds whose (canonical) type matches the query type name(s).

        Cross-dataset candidate discovery is connectivity-based, so neurons of
        the queried type itself can be pruned by vector prefiltering when the
        two datasets label their partner types differently (the adjacency
        score counts shared partner TYPES).  Those same-name homologs are the
        primary answer to a named-type search, so they are re-added to the
        candidate pool and exempted from prefilter pruning.

        ``standardize_partner_types`` returns ``{canonical_type: weight}``;
        a canonical-keyed dict cannot recover which raw target types
        produced each key, so each raw target type is canonicalized
        individually here.  A type counts as "same" when its canonical
        form equals a query name (or the query's own canonical form, so
        target->source queries resolve too).  When ``source_dataset`` is
        given, conflicted target types fail closed: they must not re-enter
        the pool through their raw same-name (the BANC CB1011 class).
        """
        names = {str(n).strip() for n in query_type_names if n and str(n).strip()}
        names.discard('')
        if not names:
            return set()

        raw_types = sorted({
            str(t) for t in target_type_lookup.values()
            if t and not pd.isna(t) and str(t).strip()
        })
        matched_types: set = {t for t in raw_types if t in names}
        if type_mapper is not None:
            def _canon(name: str, ds: Optional[str] = None) -> str:
                try:
                    canonical = type_mapper.get_canonical_type(name, ds)
                except Exception:
                    canonical = None
                return canonical or name

            canon_queries = {_canon(q) for q in names} | names
            matched_types.update(
                t for t in raw_types
                if _canon(t, target_dataset) in canon_queries)

            # Fail closed on conflicts (raw same-name or mapped canon both
            # count above; only the explicit conflict verdict removes).
            if matched_types and source_dataset is not None:
                matched_types = {
                    t for t in matched_types
                    if not self._is_conflicted_target_type(
                        type_mapper, t, target_dataset, source_dataset)
                }

        if not matched_types:
            return set()
        return {
            int(bid) for bid, t in target_type_lookup.items() if t in matched_types
        }

    @staticmethod
    def _is_conflicted_target_type(
        type_mapper: 'CrossDatasetTypeMapper',
        target_type: str,
        target_dataset: str,
        source_dataset: str,
    ) -> bool:
        """True when ``target_type`` carries a conflict verdict toward the
        query namespace (scoped decision; failure stays permissive)."""
        try:
            decision = type_mapper.get_mapping_decision(
                target_type, target_dataset, source_dataset,
                include_bridges=False)
        except Exception:
            return False
        return decision is not None and decision.get('status') == 'conflict'

    def _expand_partner_types_to_target(
        self,
        source_types,
        source_dataset: str,
        target_dataset: str,
        type_mapper: Optional['CrossDatasetTypeMapper'],
        cache: Optional[Dict[str, Tuple[str, ...]]] = None,
        status_counts_out: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Tuple[str, ...]]:
        """Map source-local partner types to target-local expansion names.

        Uses the shared validity resolver (``comparison.type_resolver``):
        unique renames and bridge-derived targets, EVERY branch of a valid
        split, nothing for conflicts (fail closed), and the raw name
        itself for unmapped types (the counted long-tail fallback).
        Without a mapper every type expands to itself.

        One run-scoped ``self._mapper_snapshot`` is reused across calls (it
        is created lazily and reset per search), so repeated expansions
        share the base mapper load state and decision cache.

        Each UNIQUE ``(source_dataset, target_dataset, type)`` resolution
        is counted once into ``status_counts_out`` when given (deduped
        across calls via ``self._counted_expansion_resolutions``), feeding
        the run's ``auto_type_mapping.json`` metadata and its
        ``raw_fallback_used`` flag.
        """
        if type_mapper is None:
            return {t: (t,) for t in source_types}
        from .type_resolver import (
            expansion_targets, get_mapper_snapshot, resolve_valid_targets,
        )
        if self._mapper_snapshot is None:
            self._mapper_snapshot = get_mapper_snapshot(type_mapper)
        snapshot = self._mapper_snapshot
        expansion: Dict[str, Tuple[str, ...]] = {}
        for t in source_types:
            hit = cache.get(t) if cache is not None else None
            if hit is None:
                res = resolve_valid_targets(
                    type_mapper, t, source_dataset, target_dataset,
                    snapshot=snapshot)
                hit = expansion_targets(res)
                if status_counts_out is not None:
                    resolution_key = (
                        str(source_dataset), str(target_dataset), str(t))
                    if resolution_key not in self._counted_expansion_resolutions:
                        self._counted_expansion_resolutions.add(resolution_key)
                        status = res.status
                        status_counts_out[status] = (
                            status_counts_out.get(status, 0) + 1)
                if cache is not None:
                    cache[t] = hit
            expansion[t] = hit
        return expansion

    def _compute_type_level_results(
        self,
        source_types: List[str],
        target_dataset: str,
        similarity_metric: str = 'jaccard',
        type_mapper: Optional['CrossDatasetTypeMapper'] = None,
        source_pooled: Optional[Dict[str, 'ConnectivityProfile']] = None,
        target_pooled: Optional[Dict[str, 'ConnectivityProfile']] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        True type-level homolog ranking: pooled all-adjacency type profiles.

        Every source type's profile pools ALL adjacencies of its member
        neurons (no top-k truncation); target type profiles are pooled the
        same way and ranked per source type with the same scorer the bodyId
        level uses (``batch_compare_cross_dataset``), partner types
        standardized through the cross-dataset type mapper.

        Args:
            source_types: Resolved source type names.
            target_dataset: Dataset whose types are ranked.
            similarity_metric: Metric used for the ``rank`` column (all five
                metrics are always computed).
            type_mapper: CrossDatasetTypeMapper for partner-type
                standardization and same-type detection.
            source_pooled: Pre-built pooled source profiles (label ->
                ConnectivityProfile). Built on demand when omitted.
            target_pooled: Pre-built pooled target profiles for ALL typed
                targets of ``target_dataset``. Built on demand when omitted.

        Returns:
            DataFrame: source_type, target_type, is_same_type, target_dataset,
            jaccard, weighted_jaccard, cosine, rank_corr, rank_union, rank
            (1-based rank of the target type under ``similarity_metric``,
            NaN last).
        """
        normalize = lambda s: self.profiler._normalize_types_vectorized(
            s, self.profiler.config.fuzzy_match)  # noqa: E731
        # Captured before the pre-standardization block nulls type_mapper
        # (the canonical() closure below must keep seeing the real mapper).
        mapper_for_same = type_mapper
        # NOTE: called unbound with the finder's own profiler config; the
        # normalizer is stateless beyond that config.

        if source_pooled is None:
            conn_s = self._load_connection_cache(self.source_dataset)
            if conn_s is None:
                return pd.DataFrame()
            bid_to_label: Dict[Any, Any] = {}
            for st in source_types:
                for bid in self.get_bodyids_for_type(st, self.source_dataset):
                    bid_to_label[int(bid)] = st
            source_pooled = ProfileComparator.build_pooled_type_profiles(
                conn_s, bid_to_label, dataset=self.source_dataset,
                min_weight=self.min_synapse_threshold,
                normalize_types=normalize)
        if target_pooled is None:
            conn_t = self._load_connection_cache(target_dataset)
            if conn_t is None:
                return pd.DataFrame()
            target_type_lookup = self._build_bodyid_type_lookup(conn_t, target_dataset)
            target_pooled = ProfileComparator.build_pooled_type_profiles(
                conn_t, dict(target_type_lookup), dataset=target_dataset,
                min_weight=self.min_synapse_threshold,
                normalize_types=normalize)

        # Pre-standardize each profile ONCE (partner types -> canonical names).
        # Passing type_mapper=None to the scorer after this is equivalent to
        # standardizing per pair, but avoids re-standardizing every target
        # profile for every source (O(sources x targets) Python-loop calls).
        if type_mapper is not None:
            source_pooled = {k: ProfileComparator.standardize_profile(
                v, self.source_dataset, type_mapper)
                for k, v in source_pooled.items()}
            target_pooled = {k: ProfileComparator.standardize_profile(
                v, target_dataset, type_mapper)
                for k, v in target_pooled.items()}
            type_mapper = None

        target_names = sorted(target_pooled.keys())
        target_profiles = {i: target_pooled[name] for i, name in enumerate(target_names)}
        candidate_map = {i: 0 for i in range(len(target_names))}

        # Same-type marking via the shared resolver's merge keys: licensed
        # renames match (MTe07 == MeVPLo2), while a CONFLICTED type's
        # dataset-scoped key can never equal the query's key, so a
        # raw-same-name conflict is never marked as the same type.
        merge_key_cache: dict = {}

        def canonical(name: str, ds: str) -> str:
            if mapper_for_same is None:
                return name
            try:
                from .type_resolver import canonical_merge_key
                return canonical_merge_key(
                    mapper_for_same, name, ds, cache=merge_key_cache).key
            except Exception:
                return name

        rows = []
        for source_type in sorted(source_types):
            src_profile = source_pooled.get(source_type)
            if src_profile is None:
                continue
            src_canon = canonical(source_type, self.source_dataset or '')
            scores = ProfileComparator.batch_compare_cross_dataset(
                src_profile, target_profiles, candidate_map, 'both',
                type_mapper=type_mapper)
            for score in scores:
                t_name = target_names[score['target_bid']]
                rows.append({
                    'source_type': source_type,
                    'target_type': t_name,
                    'is_same_type': canonical(t_name, target_dataset) == src_canon,
                    'target_dataset': target_dataset,
                    'jaccard': score['jaccard'],
                    'weighted_jaccard': score.get('weighted_jaccard', np.nan),
                    'cosine': score['cosine'],
                    'rank_corr': score['rank'],
                    'rank_union': score['rank_union'],
                })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        sort_col = (similarity_metric if similarity_metric in df.columns
                    else 'jaccard')
        df = df.sort_values(['source_type', sort_col], ascending=[True, False],
                            na_position='last').reset_index(drop=True)
        df['rank'] = df.groupby('source_type').cumcount() + 1
        return df

    # ------------------------------------------------------------------
    # Shared bodyId-level type comparison core (for ComparisonAnalyzer)
    # ------------------------------------------------------------------
    @staticmethod
    def compare_types_bodyid_core(
        profiler: 'ConnectivityProfiler',
        neuron_types: List[str],
        dataset_a: str,
        dataset_b: str,
        direction: str = 'both',
        min_common_partners: int = 3,
        score_weights: Optional[Dict[str, float]] = None,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compare types via bodyId-level pairs then aggregate to type level.

        Returns
        -------
        bodyid_df : pd.DataFrame
            Per-pair results with columns: neuron_type, source_bodyId, target_bodyId,
            jaccard, cosine, rank_corr, rank_union.
        type_summary : pd.DataFrame
            Aggregated per-type averages (avg_* columns) plus counts.
        """
        results = []
        type_iter = neuron_types
        if verbose:
            type_iter = tqdm(neuron_types, desc=f"{dataset_a} vs {dataset_b} types", unit="type",
                             disable=progress_bars_disabled(True, verbose))

        for neuron_type in type_iter:
            try:
                bodyids_a = profiler.get_bodyids_for_type(neuron_type, dataset_a) or []
                bodyids_b = profiler.get_bodyids_for_type(neuron_type, dataset_b) or []
            except Exception:
                bodyids_a, bodyids_b = [], []

            if verbose:
                print(f"[ProfileComparator] Comparing type '{neuron_type}' ({len(bodyids_a)} vs {len(bodyids_b)} bodyIds)")

            for bid_a in bodyids_a:
                profile_a = profiler.get_profile(bid_a, dataset_a)
                if profile_a is None:
                    continue
                for bid_b in bodyids_b:
                    profile_b = profiler.get_profile(bid_b, dataset_b)
                    if profile_b is None:
                        continue

                    scores = ProfileComparator.combined_score_intra_dataset(
                        profile_a,
                        profile_b,
                        weights=score_weights or DEFAULT_SCORE_WEIGHTS,
                        direction=direction,
                    )

                    # Enforce minimum shared partners if requested
                    overlap_count = 0
                    try:
                        # Same partner source the combined score uses (typed +
                        # untyped bodyIds), so the filter matches the metric.
                        # Note: do NOT reuse the loop variable names here.
                        partner_set_a = ProfileComparator._get_all_bodyids(profile_a, direction)
                        partner_set_b = ProfileComparator._get_all_bodyids(profile_b, direction)
                        overlap_count = len(set(partner_set_a) & set(partner_set_b))
                    except Exception:
                        pass

                    if min_common_partners and overlap_count < min_common_partners:
                        scores['rank'] = np.nan
                        scores['rank_union'] = np.nan

                    results.append({
                        'neuron_type': neuron_type,
                        'source_bodyId': bid_a,
                        'target_bodyId': bid_b,
                        'jaccard': scores.get('jaccard'),
                        'cosine': scores.get('cosine'),
                        'rank_corr': scores.get('rank'),
                        'rank_union': scores.get('rank_union'),
                        'overlap_partners': overlap_count,
                    })

        bodyid_df = pd.DataFrame(results)

        if bodyid_df.empty:
            return bodyid_df, pd.DataFrame()

        # Aggregate to type-level averages
        agg_dict = {
            'jaccard': 'mean',
            'cosine': 'mean',
            'rank_corr': 'mean',
            'rank_union': 'mean',
            'source_bodyId': 'nunique',
            'target_bodyId': 'nunique',
        }
        type_summary = bodyid_df.groupby('neuron_type').agg(agg_dict).rename(columns={
            'source_bodyId': 'n_source_bodyIds',
            'target_bodyId': 'n_target_bodyIds',
            'jaccard': 'avg_jaccard',
            'cosine': 'avg_cosine',
            'rank_corr': 'avg_rank_corr',
            'rank_union': 'avg_rank_union',
        }).reset_index()

        return bodyid_df, type_summary


    # ------------------------------------------------------------------------------
    # Compatibility shim: ensure ProfileComparator has compare_types_bodyid_core
    # ------------------------------------------------------------------------------
    # Some entry points (e.g., ComparisonAnalyzer) rely on ProfileComparator exposing
    # a bodyId-level type comparison core. In case this attribute is missing in older
    # environments, attach a fallback implementation here.
    if not hasattr(ProfileComparator, 'compare_types_bodyid_core'):

        def _compare_types_bodyid_core(
            profiler: 'ConnectivityProfiler',
            neuron_types: List[str],
            dataset_a: str,
            dataset_b: str,
            direction: str = 'both',
            min_common_partners: int = 3,
            score_weights: Optional[Dict[str, float]] = None,
            verbose: bool = True,
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
            """Fallback bodyId-level comparison then aggregate to type level."""
            results = []

            for neuron_type in neuron_types:
                try:
                    bodyids_a = profiler.get_bodyids_for_type(neuron_type, dataset_a) or []
                    bodyids_b = profiler.get_bodyids_for_type(neuron_type, dataset_b) or []
                except Exception:
                    bodyids_a, bodyids_b = [], []

                if verbose:
                    print(
                        f"[ProfileComparator] Comparing type '{neuron_type}' "
                        f"({len(bodyids_a)} vs {len(bodyids_b)} bodyIds)"
                    )

                for bid_a in bodyids_a:
                    profile_a = profiler.get_profile(bid_a, dataset_a)
                    if profile_a is None:
                        continue
                    for bid_b in bodyids_b:
                        profile_b = profiler.get_profile(bid_b, dataset_b)
                        if profile_b is None:
                            continue

                        scores = ProfileComparator.combined_score_intra_dataset(
                            profile_a,
                            profile_b,
                            weights=score_weights or DEFAULT_SCORE_WEIGHTS,
                            direction=direction,
                        )

                        # Enforce minimum shared partners if requested
                        overlap_count = 0
                        try:
                            # Same partner source the combined score uses
                            # (typed + untyped bodyIds).
                            # Note: do NOT reuse the loop variable names here.
                            partner_set_a = ProfileComparator._get_all_bodyids(profile_a, direction)
                            partner_set_b = ProfileComparator._get_all_bodyids(profile_b, direction)
                            overlap_count = len(set(partner_set_a) & set(partner_set_b))
                        except Exception:
                            pass

                        if min_common_partners and overlap_count < min_common_partners:
                            scores['rank'] = np.nan
                            scores['rank_union'] = np.nan

                        results.append(
                            {
                                'neuron_type': neuron_type,
                                'source_bodyId': bid_a,
                                'target_bodyId': bid_b,
                                'jaccard': scores.get('jaccard'),
                                'cosine': scores.get('cosine'),
                                'rank_corr': scores.get('rank'),
                                'rank_union': scores.get('rank_union'),
                                'overlap_partners': overlap_count,
                            }
                        )

            bodyid_df = pd.DataFrame(results)

            if bodyid_df.empty:
                return bodyid_df, pd.DataFrame()

            agg_dict = {
                'jaccard': 'mean',
                'cosine': 'mean',
                'rank_corr': 'mean',
                'rank_union': 'mean',
                'source_bodyId': 'nunique',
                'target_bodyId': 'nunique',
            }

            type_summary = (
                bodyid_df.groupby('neuron_type')
                .agg(agg_dict)
                .rename(
                    columns={
                        'source_bodyId': 'n_source_bodyIds',
                        'target_bodyId': 'n_target_bodyIds',
                        'jaccard': 'avg_jaccard',
                        'cosine': 'avg_cosine',
                        'rank_corr': 'avg_rank_corr',
                        'rank_union': 'avg_rank_union',
                    }
                )
                .reset_index()
            )

            return bodyid_df, type_summary

        ProfileComparator.compare_types_bodyid_core = staticmethod(_compare_types_bodyid_core)  # type: ignore[attr-defined]
    
    def direct_comparison(
        self,
        neurons_a: Union[str, int, List[Union[str, int]]],
        neurons_b: Optional[Union[str, int, List[Union[str, int]]]] = None,
        dataset_a: Optional[str] = None,
        dataset_b: Optional[str] = None,
        direction: str = 'both',
        comparison_mode: str = 'type',
        label_mapper: Optional[Any] = None,
        output_dir: Optional[str] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Direct comparison of specific neurons between datasets.
        
        This is a convenience wrapper around ProfileComparator.direct_comparison()
        that uses the finder's pre-configured profiler and settings.
        
        Args:
            neurons_a: Neuron(s) to compare from dataset_a (type names or bodyIds)
            neurons_b: Neuron(s) to compare from dataset_b. If None and label_mapper
                       is provided, uses neurons_a with name resolution.
            dataset_a: First dataset (default: self.source_dataset)
            dataset_b: Second dataset (default: self.target_dataset)
            direction: 'upstream', 'downstream', or 'both'
            comparison_mode: 'type' (aggregate by type) or 'bodyid' (individual neurons)
            label_mapper: Optional LabelMapper for cross-dataset type name resolution
            output_dir: Directory to save results (default: self.output_dir)
            save_results: If True, save results to CSV
        
        Returns:
            Dict with 'results' DataFrame and 'summary' statistics
        
        Example:
            >>> finder = HomologFinder(source_dataset='flywire_FAFB_v783',
            ...                        target_dataset='male-cns:v0.9')
            >>> results = finder.direct_comparison('aMe12', 'aMe12')
            >>> print(results['summary'])
        """
        # Use instance defaults if not specified
        ds_a = dataset_a or self.source_dataset
        ds_b = dataset_b or self.target_dataset
        out_dir = output_dir or self.output_dir
        
        if not ds_a or not ds_b:
            raise ValueError("Both dataset_a and dataset_b must be specified")
        
        # Call ProfileComparator.direct_comparison
        results = ProfileComparator.direct_comparison(
            neurons_a=neurons_a,
            neurons_b=neurons_b,
            dataset_a=ds_a,
            dataset_b=ds_b,
            profiler=self.profiler,
            direction=direction,
            comparison_mode=comparison_mode,
            label_mapper=label_mapper,
            score_weights=self.score_weights,
            top_k=self.profiler.config.top_k_bodyid,
            top_m=self.profiler.config.top_m_type,
            min_synapse_threshold=self.min_synapse_threshold,
            include_untyped_partners=self.profiler.config.include_untyped_partners,
            verbose=self.verbose
        )
        
        # Save results if requested
        if save_results and not results['results'].empty:
            os.makedirs(out_dir, exist_ok=True)
            
            # Generate filename
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            neurons_str = str(neurons_a[0] if isinstance(neurons_a, list) else neurons_a)
            safe_neurons = neurons_str.replace('/', '_').replace(':', '_')[:20]
            filename = f'direct_comparison_{safe_neurons}_{timestamp}.csv'
            
            filepath = os.path.join(out_dir, filename)
            results['results'].to_csv(filepath, index=False)
            self._log(f"Saved results to: {filepath}")
            results['output_file'] = filepath

            # Save parameters to JSON
            import json
            params = {
                'neurons_a': neurons_a,
                'neurons_b': neurons_b,
                'dataset_a': ds_a,
                'dataset_b': ds_b,
                'direction': direction,
                'comparison_mode': comparison_mode,
                'top_k': self.profiler.config.top_k_bodyid,
                'top_m': self.profiler.config.top_m_type,
                'min_synapse_threshold': self.min_synapse_threshold,
                'include_untyped_partners': self.profiler.config.include_untyped_partners,
                'score_weights': self.score_weights,
                'timestamp': timestamp
            }
            params.update(auto_mapping_result_metadata(
                requested=getattr(self, 'use_auto_type_mapping', True),
                type_mapper=getattr(self, '_type_mapper', None),
            ))
            json_filename = f'direct_comparison_{safe_neurons}_{timestamp}_params.json'
            json_filepath = os.path.join(out_dir, json_filename)
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2, default=str)
            self._log(f"Saved parameters to: {json_filepath}")
        
        return results

    def _get_target_bodyids_and_types(self, dataset: str) -> Tuple[List[int], Dict[int, str]]:
        """
        Get all bodyIds and their types from a dataset WITHOUT loading full connection DataFrame.
        
        Uses neuron_index.parquet which is much smaller than connections.parquet.
        Falls back to datasets/ neuron_df if index doesn't exist.
        
        Args:
            dataset: Dataset identifier
            
        Returns:
            Tuple of (list of bodyIds, dict mapping bodyId -> type)
        """
        from pathlib import Path
        import gc
        
        # Try to use Polars for memory-efficient loading
        try:
            import polars as pl
            use_polars = True
        except ImportError:
            use_polars = False
        
        safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
        src_dir = Path(__file__).parent.parent
        project_root = src_dir.parent
        
        bodyids = []
        type_lookup = {}
        coarse_labels = self._coarse_type_labels(dataset)

        def _lookup_add(bid, typ) -> None:
            if bid is None:
                return
            value = ''
            if pd.notna(typ) and str(typ).strip():
                value = '' if str(typ).strip().lower() in coarse_labels \
                    else str(typ)
            type_lookup[int(bid)] = value
        
        # Try neuron_index.parquet first (smallest file)
        index_path = project_root / 'neuron_indexes' / safe_name / 'neuron_index.parquet'
        if index_path.exists():
            try:
                if use_polars:
                    index_df_pl = pl.read_parquet(str(index_path))
                    index_df = index_df_pl.to_pandas()
                    del index_df_pl
                else:
                    index_df = pd.read_parquet(index_path)
                if 'bodyId' in index_df.columns:
                    # Convert to int for bodyIds
                    index_df['bodyId'] = pd.to_numeric(index_df['bodyId'], errors='coerce')
                    bodyids = index_df['bodyId'].dropna().astype(int).tolist()
                    
                    if 'type' in index_df.columns:
                        for _, row in index_df.iterrows():
                            bid = int(row['bodyId']) if pd.notna(row['bodyId']) else None
                            if bid is not None:
                                _lookup_add(bid, row['type'])
                    
                    self._log(f"Loaded {len(bodyids)} bodyIds from neuron_index")
                    del index_df
                    gc.collect()
                    return bodyids, type_lookup
            except Exception as e:
                self._log(f"Warning: Could not read neuron_index: {e}")
        
        # Fall back to datasets/ neuron_df
        neuron_df_paths = [
            project_root / 'datasets' / safe_name / f'{safe_name}_allneurons_neuron_df.parquet',
            project_root / 'datasets' / safe_name / f'{safe_name}_allneurons_neuron_df.csv'
        ]
        
        for path in neuron_df_paths:
            if path.exists():
                try:
                    if path.suffix == '.parquet':
                        if use_polars:
                            neuron_df_pl = pl.read_parquet(str(path))
                            neuron_df = neuron_df_pl.to_pandas()
                            del neuron_df_pl
                        else:
                            neuron_df = pd.read_parquet(path)
                    else:
                        neuron_df = pd.read_csv(path)
                    
                    if 'bodyId' in neuron_df.columns:
                        neuron_df['bodyId'] = pd.to_numeric(neuron_df['bodyId'], errors='coerce')
                        bodyids = neuron_df['bodyId'].dropna().astype(int).tolist()
                        
                        type_col = 'type' if 'type' in neuron_df.columns else None
                        if type_col:
                            for _, row in neuron_df.iterrows():
                                bid = int(row['bodyId']) if pd.notna(row['bodyId']) else None
                                if bid is not None:
                                    _lookup_add(bid, row[type_col])
                        
                        self._log(f"Loaded {len(bodyids)} bodyIds from {path.name}")
                        del neuron_df
                        gc.collect()
                        return bodyids, type_lookup
                except Exception as e:
                    self._log(f"Warning: Could not read {path}: {e}")
        
        self._log(f"Warning: No neuron list found for {dataset}")
        return bodyids, type_lookup
    
    def _build_profiles_memory_safe(
        self,
        bodyids: List[int],
        dataset: str,
        show_progress: bool = True,
        label: str = "neurons"
    ) -> Dict[int, 'ConnectivityProfile']:
        """
        Build connectivity profiles, then release connection data to free memory.
        
        This method builds profiles with indexed connections for O(1) lookups,
        then releases the connection DataFrame after building to allow loading
        another dataset's connections without OOM.
        
        Memory-efficient cross-dataset workflow:
        1. Load source connections → build source profiles → release
        2. Load target connections → build target profiles → release
        3. Compare profiles (only profiles in memory)
        
        Args:
            bodyids: List of bodyIds to build profiles for
            dataset: Dataset identifier
            show_progress: Show progress bars
            label: Label for progress bar
            
        Returns:
            Dict mapping bodyId -> ConnectivityProfile
        """
        import gc
        
        # Step 1: Optionally ensure connection cache is complete.
        # Full-dataset completion is opt-in; normal runs use existing cache
        # entries and fetch missing neurons on demand via FindNeuronConnection.
        if self.ensure_cache_complete:
            self._log(f"Ensuring connection data for {dataset}...")
            cache_ready = self._ensure_connection_cache_complete(dataset)
            if not cache_ready:
                self._log(f"Warning: Connection cache may be incomplete for {dataset}")
        else:
            self._log(f"Using existing connection cache for {dataset} (ensure_cache_complete=False).")
        
        # Step 2: Build profiles using existing batch method (indexed)
        # This loads connections, builds indexes, and processes efficiently
        profiles = self._build_profiles_batch(
            bodyids,
            dataset,
            show_progress=show_progress,
            label=label
        )
        
        # Step 3: Release connection data to free memory for next dataset
        self._log(f"Releasing connection data for {dataset}...")
        
        # Clear profiler's connection cache
        try:
            from .connectivity_profiler import _PROFILER_CONN_CACHE
            safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
            if safe_name in _PROFILER_CONN_CACHE:
                _PROFILER_CONN_CACHE[safe_name] = {}
        except Exception:
            pass
        
        # Clear FNC's module-level cache
        try:
            from coana import _FNC_CACHE
            safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
            if safe_name in _FNC_CACHE:
                if 'conn_df' in _FNC_CACHE[safe_name]:
                    _FNC_CACHE[safe_name]['conn_df'] = None
                # Keep indexes info but clear the data
                _FNC_CACHE[safe_name] = {}
        except Exception:
            pass
        
        # Clear our local cache
        if dataset in self._conn_cache:
            del self._conn_cache[dataset]
        
        # Force garbage collection
        gc.collect()
        
        return profiles
    
    def _build_profiles_batch(
        self,
        bodyids: List[int],
        dataset: str,
        show_progress: bool = True,
        label: str = "target"
    ) -> Dict[int, 'ConnectivityProfile']:
        """
        Build connectivity profiles for a list of bodyIds with proper cache management.
        
        This method:
        1. Loads cached profiles first (checking top-k/top-m requirements)
        2. Identifies which profiles need to be built
        3. Builds profiles in batches with intermediate cache saves
        
        Args:
            bodyids: List of bodyIds to build profiles for
            dataset: Dataset identifier
            show_progress: Show progress bars
            label: Label for progress bar (e.g., "target", "source")
            
        Returns:
            Dict mapping bodyId -> ConnectivityProfile
        """
        import os
        
        profiles: Dict[int, 'ConnectivityProfile'] = {}
        required_top_k = self.profiler.config.top_k_bodyid
        
        # Note: Connection cache completeness is ensured by caller (find_homologs/find_homologs_fast)
        # before calling this method, so we don't need to check again here.

        # Merge any pending batch files first so cached profiles are visible
        try:
            self.profiler.consolidate_profile_cache(dataset)
        except Exception:
            pass
        
        # Step 1: Pre-warm caches to avoid repeated disk reads. A broken
        # cache is exactly what these failures would reveal — log them, do
        # not drop them silently (F-XD-006).
        self._log(f"Pre-loading connection cache for {dataset}...")
        try:
            self.profiler._get_cached_conn_df(dataset)
            self._log(f"Connection cache ready for {dataset}")
        except Exception as e:
            self._log(f"WARNING: connection cache pre-warm failed for "
                      f"{dataset}: {e!r}")
        try:
            self.profiler._load_cache_dataframe(dataset)
        except Exception as e:
            self._log(f"WARNING: profile cache pre-load failed for "
                      f"{dataset}: {e!r}")
        
        # Step 2: Check which profiles are already cached with sufficient top-k
        cached_count = 0
        need_build: List[int] = []
        
        self._in_progress_bar = True
        self.profiler._in_progress_bar = True  # Tell profiler to use tqdm.write
        try:
            for bid in tqdm(
                bodyids,
                desc=f"Loading cached {label} profiles",
                disable=progress_bars_disabled(show_progress, self.verbose),
                leave=True
            ):
                cached = self.profiler._load_from_cache(bid, dataset, required_top_k=required_top_k)
                if cached is not None:
                    profiles[bid] = cached
                    cached_count += 1
                else:
                    need_build.append(bid)
        finally:
            self._in_progress_bar = False
            self.profiler._in_progress_bar = False
        
        if cached_count > 0:
            self._log(f"Loaded {cached_count} {label} profiles from cache (top_k>={required_top_k})")
        
        if not need_build:
            self._log(f"All {len(bodyids)} {label} profiles found in cache")
            return profiles
        
        # Step 3: Build profiles for remaining bodyIds
        n_to_build = len(need_build)
        self._log(f"Building {n_to_build} {label} profiles (batch_size={self._batch_size})")
        
        # Check data availability before building
        try:
            self.profiler.ensure_data_available(dataset, raise_on_missing=True)
        except Exception as e:
            self._log(f"WARNING: Data availability check failed: {e}")
        
        # Flag to track if we were interrupted
        interrupted = False
        
        # Sequential execution with batch saves
        self._log(f"Caching to disk every {self._batch_size} profiles")
        batch_count = 0
        built_in_batch = 0
        total_built = 0
        
        # Enable deferred cache writes for batch saving
        self.profiler._defer_cache_writes = True
        
        self._in_progress_bar = True
        self.profiler._in_progress_bar = True  # Tell profiler to use tqdm.write
        try:
            pbar = tqdm(
                need_build,
                desc=f"Building {label} profiles",
                disable=progress_bars_disabled(show_progress, self.verbose),
                leave=True
            )
            
            build_failures: List[str] = []
            first_build_error: Optional[BaseException] = None
            for bid in pbar:
                try:
                    profile = self.profiler.get_profile(bid, dataset)
                    if profile is not None:
                        profiles[bid] = profile
                        built_in_batch += 1
                        total_built += 1
                    # profile None = the neuron has no data (a normal
                    # outcome), NOT a build failure; only exceptions count.
                    
                    # Save cache at batch boundaries (silently)
                    if built_in_batch >= self._batch_size:
                        batch_count += 1
                        # Flush pending cache writes silently
                        self.profiler._defer_cache_writes = False
                        try:
                            self.profiler.flush_pending_cache_writes(silent=True)
                        except Exception:
                            pass
                        # Re-enable deferred writes for next batch
                        self.profiler._defer_cache_writes = True
                        built_in_batch = 0
                except Exception as e:
                    build_failures.append(f"{bid}: {e!r}")
                    if first_build_error is None:
                        first_build_error = e
                    
            pbar.close()
            if build_failures:
                shown = '; '.join(build_failures[:5])
                more = f" (+{len(build_failures) - 5} more)" \
                    if len(build_failures) > 5 else ''
                self._log(f"WARNING: {len(build_failures)}/{n_to_build} "
                          f"{label} profiles failed to build: {shown}{more}")
            if total_built == 0 and build_failures and n_to_build > 0:
                # Every single build failed — an empty/partial result here
                # would read as 'no homologs' instead of 'broken run'.
                raise RuntimeError(
                    f"profile build failed for every requested bodyId of "
                    f"{dataset} ({len(build_failures)} failures); first "
                    f"error: {first_build_error!r}") from first_build_error
        except KeyboardInterrupt:
            interrupted = True
            self._log(f"\n⚠️  Interrupted! Saving {len(profiles)} profiles built so far...")
        finally:
            self._in_progress_bar = False
            self.profiler._in_progress_bar = False
            # Final flush (silently) - save what we have
            self.profiler._defer_cache_writes = False
            try:
                self.profiler.flush_pending_cache_writes(silent=True)
            except Exception:
                pass
            
            # Consolidate batch files into main cache file (if not interrupted)
            # Even when total_built == 0, consolidation merges any prior pending batches
            if not interrupted:
                try:
                    self.profiler.consolidate_profile_cache(dataset)
                except Exception:
                    pass
        
        if interrupted:
            self._log(f"Saved {len(profiles)} {label} profiles before interrupt")
            raise KeyboardInterrupt("Profile building interrupted by user")
        
        self._log(f"Built {total_built} {label} profiles")
        return profiles
    
    def ensure_data_available(self, datasets: List[str] = None, raise_on_missing: bool = True) -> Dict[str, bool]:
        """
        Ensure connection data is available for datasets before running homolog search.
        
        This method checks if connection data exists for all required datasets.
        Call this before find_homologs() or find_homologs_fast() to get clear
        error messages if data is missing.
        
        Args:
            datasets: List of datasets to check. If None, checks source and target datasets.
            raise_on_missing: If True, raise DataNotAvailableError on first missing dataset.
            
        Returns:
            Dict mapping dataset -> availability (True/False)
            
        Raises:
            DataNotAvailableError: If raise_on_missing=True and any data is missing
            
        Example:
            >>> finder = HomologFinder(source_dataset='flywire_FAFB_v783', ...)
            >>> finder.ensure_data_available()  # Checks both source and target
            {'flywire_FAFB_v783': True, 'hemibrain:v1.2.1': True}
        """
        if datasets is None:
            datasets = []
            if self.source_dataset:
                datasets.append(self.source_dataset)
            if self.target_dataset and self.target_dataset != self.source_dataset:
                datasets.append(self.target_dataset)
        
        availability = {}
        for dataset in datasets:
            try:
                self.profiler.ensure_data_available(dataset, raise_on_missing=raise_on_missing)
                availability[dataset] = True
            except Exception as e:
                availability[dataset] = False
                if raise_on_missing:
                    raise
        
        return availability
    
    def get_data_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed status of connection data for configured datasets.
        
        Returns:
            Dict with status info for each dataset (available, type, path, rows, error)
            
        Example:
            >>> finder.get_data_status()
            {'hemibrain:v1.2.1': {'available': True, 'type': 'neuprint', ...}}
        """
        datasets = []
        if self.source_dataset:
            datasets.append(self.source_dataset)
        if self.target_dataset and self.target_dataset != self.source_dataset:
            datasets.append(self.target_dataset)
        
        return self.profiler.get_data_status(datasets)
    
    def _get_fnc_for_dataset(self, dataset: str) -> Optional[Any]:
        """
        Get or create a FindNeuronConnection instance for a dataset.
        
        Uses FindNeuronConnection module for fetching neuron info instead of
        direct database queries. This provides consistent data access patterns.
        
        Args:
            dataset: Dataset identifier (e.g., 'hemibrain:v1.2.1')
            
        Returns:
            FindNeuronConnection instance or None if not available
        """
        if dataset in self._fnc_cache:
            return self._fnc_cache[dataset]
        
        try:
            from ..coana import FindNeuronConnection
            
            # Determine the exact dataset source: BANC is its own standalone
            # source ('banc'), FAFB is 'flywire' for compatibility, everything else
            # is NeuPrint (§I of the integration plan — BANC must never
            # present itself as a NeuPrint client).
            is_fafb = is_fafb_dataset(dataset)
            is_banc = is_banc_dataset(dataset)
            if is_banc:
                client_type = 'banc'
            elif is_fafb:
                client_type = 'flywire'
            else:
                client_type = 'neuprint'
            
            # Create FNC instance with minimal settings
            # Pass token from HomologFinder; FNC handles env vars if empty
            fnc = FindNeuronConnection(
                dataset=dataset,
                client_type=client_type,
                max_interlayer=-1,  # No connection fetching, just neuron info
                verbose_mode='silent',
                token=getattr(self, 'token', '')
            )
            
            self._fnc_cache[dataset] = fnc
            return fnc
            
        except Exception as e:
            self._log(f"Warning: Could not create FindNeuronConnection for {dataset}: {e}")
            return None
    
    def get_neuron_info(
        self,
        neurons: List[Union[str, int]],
        dataset: str,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get neuron information using the FindNeuronConnection module.
        
        This method provides consistent data access by delegating to FNC,
        which handles caching, local dataset files, and API fallback.
        
        Args:
            neurons: List of bodyIds (int) or type names (str)
            dataset: Dataset identifier
            columns: Specific columns to return (optional)
            
        Returns:
            DataFrame with neuron information
        """
        fnc = self._get_fnc_for_dataset(dataset)
        if fnc is None:
            return pd.DataFrame()
        
        try:
            # Determine if input is bodyIds or types
            if neurons and isinstance(neurons[0], int):
                # BodyId query
                return fnc._fetch_neurons_local_or_api(neurons, columns=columns)
            else:
                # Type query
                return fnc._fetch_neurons_by_types(list(neurons), columns=columns)
        except Exception as e:
            self._log(f"Warning: Could not fetch neuron info via FNC: {e}")
            return pd.DataFrame()
    
    def get_profile(
        self, 
        query: Union[str, int], 
        dataset: str,
        force_refresh: bool = False
    ) -> Optional['ConnectivityProfile']:
        """
        Get connectivity profile for a neuron, using the profiler's cache system.
        
        This delegates to ConnectivityProfiler.get_profile() which uses the 3-tier
        cache system (memory → disk-index → disk-df).
        
        Args:
            query: Neuron type (str) or bodyId (int)
            dataset: Dataset identifier
            force_refresh: Bypass cache if True
            
        Returns:
            ConnectivityProfile or None if not found
        """
        return self.profiler.get_profile(query, dataset, force_refresh=force_refresh)
    
    def get_profiles_batch(
        self,
        neurons: List[Union[str, int]],
        dataset: str,
        force_refresh: bool = False,
        show_progress: bool = True
    ) -> Dict[Union[str, int], 'ConnectivityProfile']:
        """
        Get connectivity profiles for multiple neurons efficiently.
        
        Uses the profiler's batch method with deferred cache writes for
        parallel-safe operation.
        
        Args:
            neurons: List of neuron types or bodyIds
            dataset: Dataset identifier
            force_refresh: Bypass cache if True
            show_progress: Show progress bar
            
        Returns:
            Dict mapping neuron identifier to ConnectivityProfile
        """
        profiles = {}
        
        # Enable deferred writes for batch operation
        self.profiler._defer_cache_writes = True
        self.profiler._pending_cache_writes = {}
        
        try:
            iterator = tqdm(
                neurons,
                desc=f"Building profiles for {dataset}",
                disable=progress_bars_disabled(show_progress, self.verbose),
                leave=False
            )
            
            for neuron in iterator:
                try:
                    profile = self.profiler.get_profile(neuron, dataset, force_refresh)
                    if profile is not None:
                        profiles[neuron] = profile
                except Exception as e:
                    self._log(f"Warning: Failed for {neuron}: {e}")
                    
        finally:
            # Flush all pending writes at once
            self.profiler._defer_cache_writes = False
            self.profiler.flush_pending_cache_writes()
        
        return profiles

    def _prewarm_profile_cache(self, dataset: str) -> None:
        """Warm up profile cache/index so first access avoids cold disk reads."""
        try:
            self.profiler.consolidate_profile_cache(dataset)
        except Exception:
            pass
        try:
            self.profiler._load_cache_dataframe(dataset)
        except Exception:
            pass
        try:
            self.profiler._get_cached_conn_df(dataset)
        except Exception:
            pass
    
    def get_available_types(self, dataset: str) -> List[str]:
        """Get available types in a dataset, with caching."""
        if dataset not in self._types_cache:
            types = self.profiler.get_available_types(dataset)
            self._types_cache[dataset] = types or []
        return self._types_cache[dataset]
    
    def get_bodyids_for_type(self, neuron_type: str, dataset: str) -> List[int]:
        """
        Get all bodyIds for a neuron type in a dataset.
        
        Uses caching for repeated lookups.
        
        Args:
            neuron_type: Type name
            dataset: Dataset identifier
            
        Returns:
            List of bodyIds
        """
        if dataset not in self._bodyids_by_type_cache:
            self._bodyids_by_type_cache[dataset] = {}
        
        if neuron_type not in self._bodyids_by_type_cache[dataset]:
            bodyids = self.profiler.get_bodyids_for_type(neuron_type, dataset)
            self._bodyids_by_type_cache[dataset][neuron_type] = bodyids or []
        
        return self._bodyids_by_type_cache[dataset][neuron_type]
    
    def _resolve_queries_to_types(
        self,
        queries: List[Union[str, int]],
        source_dataset: str,
    ) -> List[Dict[str, Any]]:
        """Expand a set of raw queries into per-real-type search units.

        Each returned unit drives one independent homolog search so results can
        be grouped by query type. Handles:

        - bodyId items -> one unit with a single bodyId.
        - Coarse taxonomy / higher-category labels (cell class, subclass, or
          another string column that maps to several real types) -> one unit
          per real type, mirroring the network/connectivity tabs' query
          expansion via ``get_types_for_label``.
        - Plain exact type names -> one unit holding all bodyIds of the type.
        - Unresolvable items -> a placeholder unit with no bodyIds (skipped by
          the caller with a warning).

        Returns an ordered list of ``{'query', 'type', 'bodyids'}`` dicts.
        """
        units: List[Dict[str, Any]] = []
        label_resolver = getattr(self.profiler, 'get_types_for_label', None)

        for item in queries:
            is_bodyid = isinstance(item, int) or (
                isinstance(item, str) and str(item).isdigit()
            )
            if is_bodyid:
                units.append({'query': str(item), 'type': None, 'bodyids': [int(item)]})
                continue

            raw = str(item).strip()
            if not raw:
                continue

            # 1) Coarse-taxonomy / higher-category expansion.  get_types_for_label
            #    groups matching rows by their real neuron type, so a cell class /
            #    cell type label maps to several real types.  This must run before
            #    the union lookup in get_bodyids_for_type, which would otherwise
            #    collapse the whole label into a single unit.
            if label_resolver is not None:
                try:
                    label_groups = label_resolver(raw, source_dataset)
                except Exception:
                    label_groups = {}
                if label_groups:
                    for real_type, ids in label_groups.items():
                        bodyids = [
                            int(bid) if isinstance(bid, str) and bid.isdigit() else bid
                            for bid in ids
                        ]
                        if bodyids:
                            units.append({
                                'query': raw, 'type': real_type, 'bodyids': bodyids,
                            })
                    continue

            # 2) Exact type: all bodyIds of the type form one unit.
            bodyids = self.get_bodyids_for_type(raw, source_dataset)
            if bodyids:
                units.append({'query': raw, 'type': raw, 'bodyids': bodyids})
            else:
                # 3) Unresolvable: placeholder, skipped by the caller.
                units.append({'query': raw, 'type': raw, 'bodyids': []})

        return units
    
    def find_homologs(
        self,
        source: Optional[Union[str, int]] = None,
        source_dataset: Optional[str] = None,
        target_dataset: Optional[str] = None,
        top_n: Optional[int] = None,
        metric: str = 'rank_corr',
        direction: str = 'both',
        min_score: float = 0.0,
        show_progress: bool = True,
        output_dir: Optional[str] = None,
        saveas: Optional[str] = None,
        include_partner_details: bool = True,
        top_n_details: int = 10,
        run_shuffle_test: bool = False,
        n_shuffles: int = 100,
        shuffle_seed: Optional[int] = None,
        visualize_skeleton: Optional[bool] = None,
        visualize_top_n: Optional[int] = None,
        vector_prune_fraction: Optional[float] = None,
        extra_readme_params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Comprehensive homolog discovery - searches the ENTIRE target dataset.
        
        This method builds connectivity profiles for ALL neurons (typed + untyped)
        in the target dataset, then compares them with source neurons. This is the
        most thorough search but can be slow for large datasets.
        
        Note: Untyped neurons are only ignored in 2-hop profiles during profile
        building, but they are included as candidates in this comprehensive search.
        
        For faster search using adjacency expansion, use find_homologs_fast().
        
        Algorithm:
            1. Build profiles for source neurons (by type or bodyId)
            2. Get ALL bodyIds in target dataset (typed + untyped)
            3. Build profiles for ALL target bodyIds with batch saves
            4. Compare source profiles against all target profiles
            5. Return top-n matches sorted by rank_corr
        
        Uses module-level defaults (self.source, self.source_dataset, etc.) 
        when parameters are not provided.
        
        Args:
            source: Neuron identifier - bodyId (int) or type name (str).
                   Uses self.source if not provided.
            source_dataset: Dataset containing the query neuron.
                           Uses self.source_dataset if not provided.
            target_dataset: Target dataset to search for homologs.
                           Uses self.target_dataset if not provided.
            top_n: Number of top candidates to return per source neuron
            metric: Similarity metric ('rank_corr', 'jaccard', 'cosine')
            direction: Connection direction ('upstream', 'downstream', 'both')
            min_score: Minimum similarity score to include
            show_progress: Show progress bar
            output_dir: Directory to save results. Uses self.output_dir if not provided.
            saveas: Custom folder name. Uses self.saveas if not provided.
            include_partner_details: Include per-match partner overlap CSVs (when saving)
            top_n_details: Number of top matches to include detailed profiles for
            run_shuffle_test: If True, run random control test to validate results
            n_shuffles: Number of shuffle iterations for control test (default: 100)
            shuffle_seed: Random seed for reproducibility of shuffle test
            visualize_skeleton: Generate 3D visualizations. Uses self.visualize_skeleton if None.
            visualize_top_n: Number of top candidates to visualize. Uses self.visualize_top_n if None.
        
        Returns:
            DataFrame with columns:
                - source_bodyId: Source neuron bodyId
                - source_type: Source neuron type
                - target_bodyId: Target neuron bodyId
                - target_type: Target neuron type (empty string for untyped)
                - target_dataset: Target dataset
                - jaccard: Jaccard similarity
                - rank_corr: Rank correlation (primary metric for sorting)
                - is_same_type: True if type name matches query
                
            Results are sorted by source_bodyId, then rank_corr (descending).
            
            If run_shuffle_test=True, additional columns are added:
                - shuffle_p_value: P-value from shuffle test
                - shuffle_z_score: Z-score from shuffle test
                - shuffle_significant: True if p < 0.05
        """
        # Per-search mapper state: a fresh snapshot (never a stale load
        # state on a long-lived finder) and a fresh resolution-count basis,
        # so this search's auto_type_mapping.json owns exactly its own
        # candidate-expansion resolutions.
        self._mapper_snapshot = None
        self._counted_expansion_resolutions = set()
        self._mapping_status_counts = {}

        # Use module-level defaults where not provided
        query = source if source is not None else self.source
        source_dataset = source_dataset if source_dataset is not None else self.source_dataset
        target_dataset = target_dataset if target_dataset is not None else self.target_dataset
        output_dir = output_dir if output_dir is not None else self.output_dir
        saveas = saveas if saveas is not None else self.saveas
        visualize_skeleton = visualize_skeleton if visualize_skeleton is not None else self.visualize_skeleton
        visualize_top_n = visualize_top_n if visualize_top_n is not None else self.visualize_top_n
        top_n = top_n if top_n is not None else self.top_n
        vector_prune_fraction = vector_prune_fraction if vector_prune_fraction is not None else self.vector_prune_fraction
        
        # Validate required parameters
        if query is None:
            raise ValueError("No source neuron specified. Set 'source' parameter or self.source.")
        if source_dataset is None:
            raise ValueError("No source_dataset specified. Set 'source_dataset' parameter or self.source_dataset.")
        if target_dataset is None:
            raise ValueError("No target_dataset specified. Set 'target_dataset' parameter or self.target_dataset.")
        
        # Determine if query is bodyId or type
        is_bodyid = isinstance(query, int) or (isinstance(query, str) and query.isdigit())
        query_label = f"bodyId:{query}" if is_bodyid else query
        is_cross_dataset = (source_dataset != target_dataset)
        
        self._log(f"Comprehensive homolog search: {query_label} from {source_dataset} → {target_dataset}")
        
        # =====================================================================
        # Step 0: Ensure connection caches are complete for both datasets
        # =====================================================================
        # MEMORY-EFFICIENT APPROACH:
        # 1. Build source profiles (load connections only if needed, then release)
        # 2. Build target profiles (load connections only if needed, then release)
        # 3. Compare profiles (only profiles in memory, not connection DataFrames)
        # =====================================================================
        
        # =====================================================================
        # Step 1: Get source neurons and build their profiles
        # =====================================================================
        if not is_bodyid:
            # Type query: get all bodyIds for this type
            source_bodyids = self.get_bodyids_for_type(str(query), source_dataset)
            if not source_bodyids:
                self._log(f"ERROR: No bodyIds found for type '{query}' in {source_dataset}")
                return pd.DataFrame()
            self._log(f"Found {len(source_bodyids)} source neurons for type '{query}'")
        else:
            source_bodyids = [int(query)]
            self._log(f"Source bodyId: {query}")
        
        # Build source profiles using memory-efficient method
        self._progress(1, 4, "Building source profiles")
        self._log(f"Building source profiles for {len(source_bodyids)} neurons...")
        source_profiles = self._build_profiles_memory_safe(
            source_bodyids, 
            source_dataset, 
            show_progress=show_progress,
            label="source"
        )
        
        if not source_profiles:
            self._log(f"ERROR: Could not build profiles for any source neurons")
            return pd.DataFrame()
        
        source_type = str(query) if not is_bodyid else None
        
        # =====================================================================
        # Step 2: Get target bodyIds and build their profiles
        # Memory is now free from source connection data
        # =====================================================================
        # Get target bodyIds from neuron index (doesn't require loading full connections)
        all_target_bodyids, target_type_lookup = self._get_target_bodyids_and_types(target_dataset)
        
        if not all_target_bodyids:
            self._log(f"ERROR: Could not get target bodyIds for {target_dataset}")
            return pd.DataFrame()
        
        # Count typed vs untyped
        typed_count = sum(1 for bid in all_target_bodyids 
                         if target_type_lookup.get(bid, '') and 
                         not pd.isna(target_type_lookup.get(bid, '')) and
                         target_type_lookup.get(bid, '') != '')
        untyped_count = len(all_target_bodyids) - typed_count
        
        self._log(f"Found {len(all_target_bodyids)} neurons in target dataset ({typed_count} typed, {untyped_count} untyped)")
        
        # Build target profiles using memory-efficient method
        self._progress(2, 4, "Building target profiles")
        target_profiles = self._build_profiles_memory_safe(
            all_target_bodyids,
            target_dataset,
            show_progress=show_progress,
            label="target"
        )
        
        self._log(f"Built {len(target_profiles)} target profiles")
        
        # =====================================================================
        # Step 4: Compare source profiles against target profiles (shared core)
        # =====================================================================
        # Build status maps
        from .connectivity_profiler import ConnectivityStatus
        source_status_map = {bid: profile.connectivity_status for bid, profile in source_profiles.items()}
        source_status_counts = {s.value: 0 for s in ConnectivityStatus}
        for status in source_status_map.values():
            source_status_counts[status.value] += 1

        # Candidate map: comprehensive search includes every target bodyId
        candidate_map: Dict[int, Dict[int, int]] = {}
        for source_bid in source_profiles.keys():
            candidate_map[source_bid] = {bid: 1 for bid in target_profiles.keys() if (is_cross_dataset or bid != source_bid)}

        # Build source_type_lookup. BodyId queries build profiles without a
        # neuron_type (the profile builder never resolves it), so resolve the
        # missing source types once here instead of exporting the raw
        # bodyId as the source's type.
        _missing_type_bids = [bid for bid, profile in source_profiles.items()
                              if not profile.neuron_type]
        _resolved_source_types: Dict[int, Optional[str]] = {}
        if _missing_type_bids:
            _resolved_source_types = self.profiler.get_types_for_bodyids(
                _missing_type_bids, source_dataset) or {}
        source_type_lookup = {
            bid: (profile.neuron_type
                  or _resolved_source_types.get(bid)
                  or str(query))
            for bid, profile in source_profiles.items()}

        # Get type mapper for cross-dataset comparison
        type_mapper = self._get_type_mapper_for_comparison(is_cross_dataset)

        # Run shared comparison core
        self._progress(3, 4, "Comparing & scoring candidates")
        results_df, intra_type_df, skipped_sources, warned_sources, source_status_map, target_status_map, target_status_counts = self._compare_candidates_core(
            source_bodyids=list(source_profiles.keys()),
            source_profiles_cache=source_profiles,
            source_status_map=source_status_map,
            target_profiles_cache=target_profiles,
            target_type_lookup=target_type_lookup,
            source_type_lookup=source_type_lookup,
            candidate_map=candidate_map,
            is_cross_dataset=is_cross_dataset,
            target_dataset=target_dataset,
            show_progress=show_progress,
            similarity_metric=metric,
            top_n=top_n,
            min_score=min_score,
            include_intra_type=not is_cross_dataset,
            vector_prefiltering=self.vector_prefiltering,
            vector_prune_fraction=vector_prune_fraction,
            type_mapper=type_mapper
        )

        if results_df.empty:
            self._log("No candidates found")
            return pd.DataFrame()

        total_skipped = len(skipped_sources['none'])
        total_warned = len(warned_sources['rare_or_uni'])
        if total_skipped > 0:
            self._log(f"Skipped {total_skipped} source neurons with no partners (NONE/ORPHAN): {skipped_sources['none'][:5]}{'...' if total_skipped>5 else ''}")
        if total_warned > 0:
            self._log(f"WARNING: {total_warned} source neurons are sparse/unidirectional (RARE/UNIDIRECTIONAL)")
        
        self._log(f"Found {len(results_df)} bodyId-level matches")
        
        # Run shuffle test if requested
        shuffle_stats = None
        if run_shuffle_test and not results_df.empty:
            self._log(f"\nRunning shuffle test with {n_shuffles} iterations...")
            shuffle_stats = self.run_random_control_test(
                source=query,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                n_shuffles=n_shuffles,
                top_n=top_n,
                seed=shuffle_seed,
                show_progress=show_progress
            )
            
            if shuffle_stats and 'p_value' in shuffle_stats:
                results_df['shuffle_p_value'] = shuffle_stats['p_value']
                results_df['shuffle_z_score'] = shuffle_stats['z_score']
                results_df['shuffle_effect_size'] = shuffle_stats['effect_size']
                results_df['shuffle_significant'] = shuffle_stats['is_significant']
                self._log(shuffle_stats.get('summary', ''))
        
        # True type-level results (pooled all-adjacency type profiles). The
        # comprehensive path does not keep the connection frames in memory, so
        # the helper reloads them from the connection cache.
        type_level_df = None
        try:
            source_types_resolved = sorted(
                {str(t) for t in results_df['source_type'].dropna().unique()}
            ) if not results_df.empty else []
            if source_types_resolved:
                type_level_df = self._compute_type_level_results(
                    source_types_resolved, target_dataset,
                    similarity_metric=self.similarity_metric,
                    type_mapper=self._get_type_mapper_for_comparison(
                        source_dataset != target_dataset),
                    show_progress=show_progress)
        except Exception as e:
            self._log(f"Type-level results skipped: {e}")

        # Save results if output_dir is provided
        self._progress(4, 4, "Saving results")
        if output_dir is not None:
            save_result = self._save_homolog_results_internal(
                results_df=results_df,
                query=query,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                output_dir=output_dir,
                saveas=saveas,
                direction=direction,
                include_partner_details=include_partner_details,
                top_n_details=top_n_details,
                params={
                    'query': query,
                    'source_dataset': source_dataset,
                    'target_dataset': target_dataset,
                    'top_n': top_n,
                    'metric': metric,
                    'direction': direction,
                    'min_score': min_score,
                    'method': 'find_homologs (comprehensive)',
                    'run_shuffle_test': run_shuffle_test,
                    'n_shuffles': n_shuffles if run_shuffle_test else None,
                    **(extra_readme_params or {}),
                },
                shuffle_stats=shuffle_stats,
                visualize_skeleton=visualize_skeleton,
                visualize_top_n=visualize_top_n,
                similarity_metric=self.similarity_metric
            )
        else:
            # Always save - use default output_dir
            save_result = self._save_homolog_results_internal(
                results_df=results_df,
                query=query,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                output_dir=self.output_dir,
                saveas=saveas,
                direction=direction,
                include_partner_details=include_partner_details,
                top_n_details=top_n_details,
                params={
                    'query': query,
                    'source_dataset': source_dataset,
                    'target_dataset': target_dataset,
                    'top_n': top_n,
                    'metric': metric,
                    'direction': direction,
                    'min_score': min_score,
                    'method': 'find_homologs (comprehensive)',
                    'run_shuffle_test': run_shuffle_test,
                    'n_shuffles': n_shuffles if run_shuffle_test else None,
                    **(extra_readme_params or {}),
                },
                shuffle_stats=shuffle_stats,
                visualize_skeleton=visualize_skeleton,
                visualize_top_n=visualize_top_n,
                similarity_metric=self.similarity_metric,
                type_level_df=type_level_df,
                top_n=top_n
            )
        
        return results_df
    
    def find_novel_homologs(
        self,
        query: Union[str, int],
        dataset: str,
        top_n: int = 20,
        min_score: float = 0.3
    ) -> pd.DataFrame:
        """
        Find potential novel homologs within the same dataset.
        
        Shortcut for find_homologs() with target_dataset = source_dataset.
        
        Args:
            query: Neuron identifier
            dataset: Dataset to search
            top_n: Number of top candidates
            min_score: Minimum similarity score
        
        Returns:
            DataFrame with similar types/neurons (excluding self)
        """
        return self.find_homologs(
            query=query,
            source_dataset=dataset,
            target_dataset=dataset,
            top_n=top_n,
            min_score=min_score
        )
    
    def get_partner_overlap(
        self,
        query: Union[str, int],
        source_dataset: str,
        target_type: str,
        target_dataset: str,
        direction: str = 'upstream'
    ) -> pd.DataFrame:
        """
        Get detailed partner overlap between query and a candidate.
        
        Args:
            query: Source neuron identifier
            source_dataset: Source dataset
            target_type: Target type to compare
            target_dataset: Target dataset
            direction: 'upstream' or 'downstream'
        
        Returns:
            DataFrame with partner overlap details
        """
        source_profile = self.get_profile(query, source_dataset)
        target_profile = self.get_profile(target_type, target_dataset)
        
        if source_profile is None or target_profile is None:
            return pd.DataFrame()
        
        return ProfileComparator.get_partner_overlap_details(
            source_profile, target_profile, direction=direction
        )
    
    # =========================================================================
    # Fast Homolog Finding with Connection Cache & Adjacency Expansion
    # =========================================================================
    
    def _get_connection_cache_path(self, dataset: str) -> 'Path':
        """Get path to existing connection cache."""
        from pathlib import Path
        safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
        src_dir = Path(__file__).parent.parent
        project_root = src_dir.parent
        return project_root / 'cache' / safe_name / 'connections.parquet'
    
    def _get_neuron_index_path(self, dataset: str) -> 'Path':
        """Get path to the app-owned neuron index for type mapping."""
        from pathlib import Path
        safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
        src_dir = Path(__file__).parent.parent
        project_root = src_dir.parent
        return project_root / 'neuron_indexes' / safe_name / 'neuron_index.parquet'
    
    def _load_connection_cache(self, dataset: str, auto_build: bool = True) -> Optional[pd.DataFrame]:
        """
        Load complete connections from cache with type information.
        
        If the cache doesn't exist and auto_build=True, attempts to build it
        automatically by fetching from the API or local dataset files.
        
        Also verifies cache completeness by checking against neuron_df if available.
        
        Uses internal caching to avoid reloading the same dataset multiple times.
        Prioritizes FNC's module-level cache to avoid memory duplication.
        
        Type mapping priority:
        1. datasets/{dataset}/neuron_df.parquet (authoritative source)
        2. neuron_indexes/{dataset}/neuron_index.parquet (fallback)
        
        Args:
            dataset: Dataset identifier
            auto_build: If True, automatically build cache if missing
        
        Returns:
            DataFrame with columns: bodyId_pre, bodyId_post, weight, type_pre, type_post
            or None if cache not available and cannot be built
        """
        # Check internal cache first
        if dataset in self._conn_cache:
            return self._conn_cache[dataset]
        
        from pathlib import Path
        safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
        src_dir = Path(__file__).parent.parent
        project_root = src_dir.parent
        
        # Check FNC's module-level cache first (avoids loading from disk twice)
        try:
            from coana import _FNC_CACHE
            if safe_name in _FNC_CACHE and 'conn_df' in _FNC_CACHE[safe_name]:
                fnc_df = _FNC_CACHE[safe_name]['conn_df']
                
                # Handle both Polars and pandas DataFrames from FNC cache
                fnc_is_empty = False
                if fnc_df is not None:
                    try:
                        import polars as pl
                        if isinstance(fnc_df, pl.DataFrame):
                            fnc_is_empty = fnc_df.is_empty()
                            if not fnc_is_empty:
                                # Convert Polars to pandas for compatibility
                                fnc_df = fnc_df.to_pandas()
                        else:
                            fnc_is_empty = fnc_df.empty
                    except ImportError:
                        fnc_is_empty = fnc_df.empty if hasattr(fnc_df, 'empty') else len(fnc_df) == 0
                else:
                    fnc_is_empty = True
                
                if fnc_df is not None and not fnc_is_empty:
                    self._log(f"Using FNC cache for {dataset} ({len(fnc_df):,} connections)")
                    # FNC cache may not have type columns. The cached frame is
                    # SHARED with FNC and the connectivity profiler, so it must
                    # not be mutated in place (to_numeric here would flip the
                    # bodyId dtypes for every other consumer). Enrich a copy.
                    if 'type_pre' not in fnc_df.columns or 'type_post' not in fnc_df.columns:
                        index_path = self._get_neuron_index_path(dataset)
                        type_map = self._load_type_mapping(dataset, project_root, safe_name, index_path)
                        if type_map:
                            fnc_df = fnc_df.copy()
                            # Convert bodyIds to numeric on the copy only
                            fnc_df['bodyId_pre'] = pd.to_numeric(fnc_df['bodyId_pre'], errors='coerce')
                            fnc_df['bodyId_post'] = pd.to_numeric(fnc_df['bodyId_post'], errors='coerce')
                            type_map_int = {int(k) if str(k).isdigit() else k: v for k, v in type_map.items()}
                            fnc_df['type_pre'] = fnc_df['bodyId_pre'].map(type_map_int)
                            fnc_df['type_post'] = fnc_df['bodyId_post'].map(type_map_int)
                    # Don't store in self._conn_cache - just reference FNC's cache
                    return fnc_df
        except ImportError:
            pass
        
        cache_path = self._get_connection_cache_path(dataset)
        index_path = self._get_neuron_index_path(dataset)
        
        # Check if cache exists, if not try to build it
        if not cache_path.exists():
            self._log(f"Connection cache not found: {cache_path}")
            
            if auto_build:
                self._log(f"Attempting to build connection cache for {dataset}...")
                if self._build_connection_cache_for_dataset(dataset, cache_path, project_root, safe_name):
                    self._log(f"Successfully built connection cache for {dataset}")
                else:
                    self._log(f"ERROR: Could not build connection cache for {dataset}")
                    return None
            else:
                return None
        
        # Verify cache exists now
        if not cache_path.exists():
            return None
        
        # Load connections using Polars for memory efficiency, then convert to pandas
        self._log(f"Loading connection cache for {dataset}...")
        try:
            import polars as pl
            # Use Polars for memory-efficient loading of large caches
            conn_df_pl = pl.read_parquet(cache_path)
            conn_df = conn_df_pl.to_pandas()
            del conn_df_pl
            self._log(f"Loaded {len(conn_df):,} connections from cache (via Polars)")
        except ImportError:
            # Fallback to pandas if polars not available
            conn_df = pd.read_parquet(cache_path)
            self._log(f"Loaded {len(conn_df):,} connections from cache")
        
        # Note: Cache is incremental - build_connection_cache always adds missing neurons
        # No need for threshold-based completeness checking since cache grows over time
        
        # Load type mapping
        self._log(f"Loading type mapping for {dataset}...")
        type_map = self._load_type_mapping(dataset, project_root, safe_name, index_path)
        
        if type_map is None:
            self._log(f"No neuron type mapping available for {dataset}")
            return None
        
        # Ensure bodyId columns are numeric for proper type mapping
        try:
            conn_df['bodyId_pre'] = pd.to_numeric(conn_df['bodyId_pre'], errors='coerce')
            conn_df['bodyId_post'] = pd.to_numeric(conn_df['bodyId_post'], errors='coerce')
        except Exception:
            pass  # Keep original types if conversion fails
        
        # Convert type_map keys to match bodyId dtype if needed
        if conn_df['bodyId_pre'].dtype in ['int64', 'float64']:
            self._log(f"Converting type map keys ({len(type_map):,} entries)...")
            type_map = {int(k) if isinstance(k, (int, float, str)) and str(k).isdigit() else k: v 
                        for k, v in type_map.items()}
        
        # Add type columns
        self._log(f"Mapping types to connections ({len(conn_df):,} rows)...")
        conn_df['type_pre'] = conn_df['bodyId_pre'].map(type_map)
        conn_df['type_post'] = conn_df['bodyId_post'].map(type_map)
        
        # Log mapping success
        mapped_pre = conn_df['type_pre'].notna().sum()
        mapped_post = conn_df['type_post'].notna().sum()
        self._log(f"Loaded {len(conn_df):,} connections from cache")
        self._log(f"Type mapping: {mapped_pre:,}/{len(conn_df):,} pre, {mapped_post:,}/{len(conn_df):,} post")
        
        # NOTE: Do NOT store in self._conn_cache to avoid memory issues
        # For large datasets (20M+ rows), keeping in memory causes OOM
        # The caller should use the returned DataFrame and discard when done
        # self._conn_cache[dataset] = conn_df  # Disabled to save memory
        
        return conn_df
    
    def _load_type_mapping(self, dataset: str, project_root: Path, safe_name: str, index_path: Path) -> Optional[Dict]:
        """Load neuron type mapping from available sources."""
        type_map = None
        
        # Priority 1: datasets/{dataset}/neuron_df.parquet
        datasets_neuron_parquet = project_root / 'datasets' / safe_name / f'{safe_name}_allneurons_neuron_df.parquet'
        if datasets_neuron_parquet.exists():
            neuron_df = pd.read_parquet(datasets_neuron_parquet)
            if 'type' in neuron_df.columns:
                type_map = neuron_df.set_index('bodyId')['type'].to_dict()
                self._log(f"Loaded type mapping from {datasets_neuron_parquet.name}")
        
        # Priority 2: datasets/{dataset}/neuron_df.csv (fallback for CSV files)
        if type_map is None:
            datasets_neuron_csv = project_root / 'datasets' / safe_name / f'{safe_name}_allneurons_neuron_df.csv'
            if datasets_neuron_csv.exists():
                neuron_df = pd.read_csv(datasets_neuron_csv, low_memory=False)
                if 'type' in neuron_df.columns:
                    type_map = neuron_df.set_index('bodyId')['type'].to_dict()
                    self._log(f"Loaded type mapping from {datasets_neuron_csv.name}")
        
        # Priority 3: neuron_indexes/{dataset}/neuron_index.parquet
        if type_map is None and index_path.exists():
            neuron_df = pd.read_parquet(index_path)
            if 'type' in neuron_df.columns:
                type_map = neuron_df.set_index('bodyId')['type'].to_dict()
                self._log(f"Loaded type mapping from neuron_index.parquet ({len(type_map)} neurons)")
        
        return type_map
    
    def _build_connection_cache_for_dataset(self, dataset: str, cache_path: Path, 
                                             project_root: Path, safe_name: str) -> bool:
        """
        Build connection cache using FindNeuronConnection.
        
        This method uses FindNeuronConnection.build_connection_cache() which:
        - Handles both NeuPrint datasets and local datasets
        - Builds cache incrementally (only fetches missing neurons)
        - Builds cache/{dataset}/connections.parquet AND
          neuron_indexes/{dataset}/neuron_index.parquet
        - Uses O(1) indexed lookups for fast cache queries
        
        Cache Hierarchy:
        ---------------
        Level 0: datasets/{dataset}/*_neuron_df - Authoritative neuron list
        Level 1: neuron_indexes/{dataset}/neuron_index.parquet - Neuron metadata index
        Level 2: cache/{dataset}/connections.parquet - Connection data
        
        Returns:
            True if cache was built successfully, False otherwise
        """
        try:
            # Import FindNeuronConnection
            try:
                from .coana import FindNeuronConnection
            except ImportError:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from coana import FindNeuronConnection
            
            self._log(f"Building connection cache for {dataset}...")
            
            # Get token via the env/config chain (needed for NeuPrint datasets)
            try:
                from utils.token_manager import token_manager
                token = token_manager.get_neuprint_token()
            except ImportError:
                import os
                token = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS') \
                    or os.environ.get('NEUPRINT_TOKEN')
            
            # Initialize FindNeuronConnection with cache enabled
            fnc = FindNeuronConnection(
                dataset=dataset,
                token=token,  # Will be None for local datasets, which is fine
                use_cache=True,
                verbose_mode='normal'
            )
            
            # Build cache incrementally - only fetches missing neurons
            result = fnc.build_connection_cache(batch_size=1000, quiet=False)
            
            total_connections = result.get('total_connections', 0)
            newly_cached = result.get('newly_cached', 0)
            already_cached = result.get('already_cached', 0)
            
            if total_connections > 0 or already_cached > 0:
                self._log(f"Cache ready: {total_connections:,} connections")
                if newly_cached > 0:
                    self._log(f"  Newly cached: {newly_cached:,} neurons")
                if already_cached > 0:
                    self._log(f"  Already cached: {already_cached:,} neurons")
                return True
            else:
                self._log("WARNING: No connections found - dataset may be empty or inaccessible")
                return False
                
        except Exception as e:
            self._log(f"ERROR: Failed to build connection cache: {e}")
            import traceback
            self._log(f"Details: {traceback.format_exc()}", level='debug')
            return False
    
    def _ensure_connection_cache_complete(self, dataset: str) -> bool:
        """
        Ensure connection cache is complete for the entire dataset.
        
        This method calls FNC's build_connection_cache which:
        - Checks which neurons are already cached (O(1) lookup)
        - Only fetches missing neurons (incremental)
        - Uses progress bar for long operations
        
        Cache Hierarchy (must be built in order):
        -----------------------------------------
        Level 0: datasets/{dataset}/*_neuron_df - Authoritative neuron list  
        Level 1: neuron_indexes/{dataset}/neuron_index.parquet - Neuron metadata index
        Level 2: cache/{dataset}/connections.parquet - Connection data
        Level 3: Connectivity profiles (built after this check)
        
        Args:
            dataset: Dataset identifier
            
        Returns:
            True if cache is ready, False otherwise
        """
        try:
            # Import FindNeuronConnection - try relative first (src.coana), then absolute
            try:
                from ..coana import FindNeuronConnection
            except ImportError:
                try:
                    from coana import FindNeuronConnection
                except ImportError:
                    import sys
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from coana import FindNeuronConnection
            
            # Initialize FindNeuronConnection
            # Use 'simple' mode to show loading progress, not 'silent' which hides everything
            # Use simple_fetch=True to reduce memory usage during bulk fetch
            # Token is passed from HomologFinder; FNC auto-handles env vars if empty
            fnc = FindNeuronConnection(
                dataset=dataset,
                token=self.token,
                use_cache=True,
                verbose_mode='simple',  # Show loading progress (not 'silent')
                simple_fetch=True
            )
            
            # Pass the progress bar flag so FNC uses tqdm.write if needed
            fnc._in_progress_bar = getattr(self, '_in_progress_bar', False)
            
            # Build connection cache - this is incremental (skips already cached neurons)
            # If cache is already complete, this returns quickly
            # Use quiet=False to show progress for large fetches
            # Use smaller batch size (50) to reduce memory pressure during fetch
            result = fnc.build_connection_cache(batch_size=1000, quiet=False)
            
            total_neurons = result.get('total_neurons', 0)
            already_cached = result.get('already_cached', 0)
            newly_cached = result.get('newly_cached', 0)
            total_connections = result.get('total_connections', 0)
            
            # Clear FNC's in-memory cache to free memory
            # Data is already saved to disk, we'll load only what we need later
            fnc._conn_df_cache = None
            fnc._conn_index = {}
            fnc._conn_index_post = {}
            
            # Also clear the module-level cache for this dataset
            # This prevents accumulating large DataFrames across datasets
            safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
            
            # Try clearing from all possible module locations to handle import variations
            modules_to_check = ['coana', 'src.coana', 'src.comparison.coana']
            for mod_name in modules_to_check:
                try:
                    import sys
                    if mod_name in sys.modules:
                        mod = sys.modules[mod_name]
                        if hasattr(mod, '_FNC_CACHE'):
                            if safe_name in mod._FNC_CACHE:
                                mod._FNC_CACHE[safe_name] = {}
                except Exception:
                    pass
            
            del fnc
            import gc
            gc.collect()
            
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                mem = process.memory_info().rss / 1024 / 1024
                self._log(f"Memory after cache check for {dataset}: {mem:.1f} MB")
            except Exception:
                pass
            
            if total_neurons > 0:
                if newly_cached == 0:
                    self._log(f"Connection cache complete: {already_cached:,} neurons, {total_connections:,} connections")
                else:
                    self._log(f"Connection cache updated: +{newly_cached:,} neurons (total: {total_connections:,} connections)")
                return True
            else:
                self._log("WARNING: No neurons found in dataset")
                return False
            
        except Exception as e:
            self._log(f"WARNING: Could not ensure connection cache: {e}")
            # Continue anyway - may work with partial data
            return False
    
    def _build_type_aggregates(
        self,
        conn_df: pd.DataFrame,
        min_weight: int = 1,
        show_progress: bool = True
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        Pre-aggregate connection data into type-level dictionaries.
        
        This is the key optimization for find_homologs_fast - we aggregate
        once and then look up types in O(1) instead of O(N) per type.
        
        Args:
            conn_df: Connection DataFrame with type columns
            min_weight: Minimum synapse weight
            show_progress: Show progress bar
            
        Returns:
            Tuple of (upstream_by_type, downstream_by_type)
            Each is Dict[type -> Dict[partner_type -> total_weight]]
        """
        if show_progress and self.verbose:
            self._log("Pre-aggregating connection data by type...")
        
        # Filter by weight
        filtered = conn_df[conn_df['weight'] >= min_weight]
        
        # Filter to only typed connections
        typed = filtered[
            filtered['type_pre'].notna() & 
            filtered['type_post'].notna() &
            (filtered['type_pre'].astype(str).str.strip() != '') &
            (filtered['type_post'].astype(str).str.strip() != '')
        ]

        upstream_by_type: Dict[str, Dict[str, float]] = {}
        downstream_by_type: Dict[str, Dict[str, float]] = {}

        # Use Polars aggregation when available (notably faster for large caches)
        use_polars = False
        try:
            import polars as pl
            typed_pl = pl.from_pandas(typed[['type_pre', 'type_post', 'weight']])
            use_polars = True
        except Exception:
            typed_pl = None
            use_polars = False

        if use_polars and typed_pl is not None:
            upstream_agg = typed_pl.group_by(['type_post', 'type_pre']).agg(pl.col('weight').sum())
            for row in upstream_agg.iter_rows(named=True):
                post_type = row['type_post']
                pre_type = row['type_pre']
                weight = row['weight']
                upstream_by_type.setdefault(post_type, {})[pre_type] = weight

            downstream_agg = typed_pl.group_by(['type_pre', 'type_post']).agg(pl.col('weight').sum())
            for row in downstream_agg.iter_rows(named=True):
                pre_type = row['type_pre']
                post_type = row['type_post']
                weight = row['weight']
                downstream_by_type.setdefault(pre_type, {})[post_type] = weight
        else:
            # Pandas fallback
            upstream_agg = typed.groupby(['type_post', 'type_pre'])['weight'].sum()
            for (post_type, pre_type), weight in upstream_agg.items():
                upstream_by_type.setdefault(post_type, {})[pre_type] = weight

            downstream_agg = typed.groupby(['type_pre', 'type_post'])['weight'].sum()
            for (pre_type, post_type), weight in downstream_agg.items():
                downstream_by_type.setdefault(pre_type, {})[post_type] = weight
        
        if show_progress and self.verbose:
            self._log(f"Aggregated {len(upstream_by_type):,} types with upstream, "
                      f"{len(downstream_by_type):,} with downstream connections")
        
        return upstream_by_type, downstream_by_type
    
    def _build_bodyid_aggregates(
        self,
        conn_df: pd.DataFrame,
        min_weight: int = 1,
        show_progress: bool = True
    ) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
        """
        Pre-aggregate connection data into bodyId-level dictionaries.
        
        Similar to _build_type_aggregates but uses bodyId as primary key
        and partner types as values. Used for bodyId-level comparisons.
        
        Args:
            conn_df: Connection DataFrame with type columns
            min_weight: Minimum synapse weight
            show_progress: Show progress bar
            
        Returns:
            Tuple of (upstream_by_bodyid, downstream_by_bodyid)
            Each is Dict[bodyId -> Dict[partner_type -> total_weight]]
        """
        if show_progress and self.verbose:
            self._log("Pre-aggregating connection data by bodyId...")
        
        # Filter by weight
        filtered = conn_df[conn_df['weight'] >= min_weight]

        upstream_by_bodyid: Dict[int, Dict[str, float]] = {}
        downstream_by_bodyid: Dict[int, Dict[str, float]] = {}

        use_polars = False
        try:
            import polars as pl
            filtered_pl = pl.from_pandas(filtered[['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post', 'weight']])
            use_polars = True
        except Exception:
            filtered_pl = None
            use_polars = False

        if use_polars and filtered_pl is not None:
            upstream_agg = filtered_pl.group_by(['bodyId_post', 'type_pre']).agg(pl.col('weight').sum())
            for row in upstream_agg.iter_rows(named=True):
                post_bodyid = row['bodyId_post']
                pre_type = row['type_pre']
                weight = row['weight']
                if pre_type is None or str(pre_type).strip() == '':
                    continue
                upstream_by_bodyid.setdefault(post_bodyid, {})[pre_type] = weight

            downstream_agg = filtered_pl.group_by(['bodyId_pre', 'type_post']).agg(pl.col('weight').sum())
            for row in downstream_agg.iter_rows(named=True):
                pre_bodyid = row['bodyId_pre']
                post_type = row['type_post']
                weight = row['weight']
                if post_type is None or str(post_type).strip() == '':
                    continue
                downstream_by_bodyid.setdefault(pre_bodyid, {})[post_type] = weight
        else:
            # Pandas fallback
            upstream_agg = filtered.groupby(['bodyId_post', 'type_pre'])['weight'].sum()
            for (post_bodyid, pre_type), weight in upstream_agg.items():
                if pd.isna(pre_type) or str(pre_type).strip() == '':
                    continue
                upstream_by_bodyid.setdefault(post_bodyid, {})[pre_type] = weight

            downstream_agg = filtered.groupby(['bodyId_pre', 'type_post'])['weight'].sum()
            for (pre_bodyid, post_type), weight in downstream_agg.items():
                if pd.isna(post_type) or str(post_type).strip() == '':
                    continue
                downstream_by_bodyid.setdefault(pre_bodyid, {})[post_type] = weight
        
        if show_progress and self.verbose:
            self._log(f"Aggregated {len(upstream_by_bodyid):,} bodyIds with upstream, "
                      f"{len(downstream_by_bodyid):,} with downstream connections")
        
        return upstream_by_bodyid, downstream_by_bodyid

    def _compare_candidates_core(
        self,
        source_bodyids: List[int],
        source_profiles_cache: Dict[int, 'ConnectivityProfile'],
        source_status_map: Dict[int, 'ConnectivityStatus'],
        target_profiles_cache: Dict[int, 'ConnectivityProfile'],
        target_type_lookup: Dict[int, str],
        source_type_lookup: Dict[int, str],
        candidate_map: Dict[int, Dict[int, int]],
        is_cross_dataset: bool,
        target_dataset: str,
        show_progress: bool,
        similarity_metric: str,
        top_n: int,
        min_score: Optional[float] = None,
        include_intra_type: bool = True,
        vector_prefiltering: bool = False,
        vector_prune_fraction: float = 0.05,
        type_mapper: Optional[CrossDatasetTypeMapper] = None,
        guaranteed_candidates: Optional[Dict[int, set]] = None,
        prefilter_rank_metric: str = 'cosine'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[int]], Dict[str, List[int]], Dict[int, 'ConnectivityStatus'], Dict[int, 'ConnectivityStatus'], Dict[str, int]]: # 
        """
        Shared bodyId comparison core used by find_homologs and find_homologs_fast.

        Args:
            type_mapper: Optional CrossDatasetTypeMapper for standardizing partner types
                        during cross-dataset comparison. When provided, partner types
                        are mapped to their canonical (male-cns) names before comparison.
            guaranteed_candidates: Optional per-source sets of target bodyIds that
                        must be compared even when vector prefiltering would prune
                        them (used for same-name homologs of the queried type).
            prefilter_rank_metric: Ranking key for the prune step when vector
                        prefiltering is enabled: 'cosine' keeps the highest
                        weighted-profile cosine candidates; 'adjacency' ranks
                        by the shared partner-type count.
        """
        from .connectivity_profiler import ConnectivityStatus

        target_status_counts: Dict[str, int] = {s.value: 0 for s in ConnectivityStatus}
        target_status_map: Dict[int, ConnectivityStatus] = {}

        for bid, profile in target_profiles_cache.items():
            status = profile.connectivity_status
            target_status_counts[status.value] += 1
            target_status_map[bid] = status

        all_results = []
        skipped_sources: Dict[str, List[int]] = {'none': []}
        warned_sources: Dict[str, List[int]] = {'rare_or_uni': []}
        # Per-run standardized profile caches for the mapping-aware vector
        # prefilter: one canonical conversion per profile per run, not one
        # per source x target pair.
        std_source_cache: Dict[int, 'ConnectivityProfile'] = {}
        std_target_cache: Dict[int, 'ConnectivityProfile'] = {}

        if vector_prefiltering and self.verbose:
            prune_label = (
                f"top {vector_prune_fraction * 100:.0f}% by adjacency"
                if vector_prune_fraction < 1.0
                else "all cosine-positive"
            )
            self._log(
                f"Vector prefiltering enabled: keep {prune_label}, then cosine>0 before full scoring"
            )

        # Set progress bar flag so _log uses tqdm.write
        was_in_progress = self._in_progress_bar
        self._in_progress_bar = True

        # Comparison loop
        for source_bid in tqdm(
            source_bodyids,
            desc="Comparing source neurons",
            disable=progress_bars_disabled(show_progress, self.verbose),
            leave=True
        ):
            source_profile = source_profiles_cache.get(source_bid)
            if source_profile is None:
                continue

            source_status = source_status_map.get(source_bid, ConnectivityStatus.NONE)

            if not source_status.is_valid_for_comparison():
                skipped_sources['none'].append(source_bid)
                continue

            if source_status in {ConnectivityStatus.RARE, ConnectivityStatus.UNIDIRECTIONAL}:
                warned_sources['rare_or_uni'].append(source_bid)

            source_type_str = source_type_lookup.get(source_bid, str(source_bid))

            source_partner_count = len(ProfileComparator._get_expanded_types(source_profile, 'both')) if is_cross_dataset else len(ProfileComparator._get_all_bodyids(source_profile, 'both'))

            source_candidates = candidate_map.get(source_bid, {})

            if vector_prefiltering and source_candidates:
                total_candidates = len(source_candidates)

                # Prefilter on the SAME standardized vectors the scorer uses
                # (batch_compare_cross_dataset standardizes both sides into
                # the canonical namespace): a candidate whose partner types
                # are renamed across datasets shows a raw-name cosine near
                # zero and would be dropped before the mapping-aware scorer
                # ever saw it.
                pre_source = source_profile
                if is_cross_dataset and type_mapper is not None:
                    pre_source = std_source_cache.get(source_bid)
                    if pre_source is None:
                        pre_source = ProfileComparator.standardize_profile(
                            source_profile, source_profile.dataset, type_mapper)
                        std_source_cache[source_bid] = pre_source

                # First filter by cosine > 0 (quick rejection of dissimilar vectors)
                cos_filtered: Dict[int, Tuple[int, float]] = {}
                for target_bid, shared_count in source_candidates.items():
                    target_profile = target_profiles_cache.get(target_bid)
                    if target_profile is None:
                        continue
                    pre_target = target_profile
                    if is_cross_dataset and type_mapper is not None:
                        pre_target = std_target_cache.get(target_bid)
                        if pre_target is None:
                            pre_target = ProfileComparator.standardize_profile(
                                target_profile, target_dataset, type_mapper)
                            std_target_cache[target_bid] = pre_target
                    cos_val = ProfileComparator.weighted_cosine_similarity(
                        pre_source, pre_target, 'both')
                    if cos_val > 0:
                        cos_filtered[target_bid] = (shared_count, cos_val)

                cos_kept = len(cos_filtered)

                # Then keep the top fraction of TOTAL candidates among the
                # cosine-positive set. The ranking key is selectable:
                # 'cosine' ranks by the weighted profile cosine itself
                # (weight-aware, robust to hub types that inflate the plain
                # shared-type count); 'adjacency' keeps the legacy behaviour
                # of ranking by the shared partner-type count. Pruning is a
                # fraction >= 1.0 -> disabled (loosest search): every
                # cosine-positive candidate is kept.
                if cos_kept > 0 and vector_prune_fraction < 1.0:
                    top_k = max(1, math.ceil(total_candidates * vector_prune_fraction))
                    rank_slot = 1 if prefilter_rank_metric == 'cosine' else 0
                    top_candidates = dict(heapq.nlargest(
                        top_k, cos_filtered.items(), key=lambda x: x[1][rank_slot]))
                    source_candidates = {bid: shared for bid, (shared, _) in top_candidates.items()}
                elif cos_kept > 0:
                    source_candidates = {bid: shared for bid, (shared, _) in cos_filtered.items()}
                else:
                    source_candidates = {}

                if self.verbose and show_progress:
                    keep_label = (
                        f"top {vector_prune_fraction * 100:.0f}% of total by "
                        f"{prefilter_rank_metric}"
                        if vector_prune_fraction < 1.0
                        else "all"
                    )
                    self._log(
                        f"Prefiltered {total_candidates}→{cos_kept} (cos>0) →{len(source_candidates)} ({keep_label}) for source {source_bid}"
                    )

            # Same-name homologs (e.g. FAFB l-LNv for a male-cns l-LNv query)
            # must never be lost to prefilter pruning: adjacency scores count
            # shared partner TYPES, which across datasets can be far sparser
            # than true similarity.  Re-add them with their original score.
            guaranteed = (guaranteed_candidates or {}).get(source_bid)
            if guaranteed:
                base_scores = candidate_map.get(source_bid, {})
                merged = dict(source_candidates)
                restored = 0
                for bid in guaranteed:
                    if bid not in merged and bid in target_profiles_cache:
                        merged[bid] = int(base_scores.get(bid, 0))
                        restored += 1
                if restored:
                    source_candidates = merged
                    if self.verbose:
                        self._log(
                            f"Restored {restored} same-type candidates skipped by "
                            f"prefiltering for source {source_bid}"
                        )

            if not source_candidates:
                continue

            if is_cross_dataset:
                batch_scores = ProfileComparator.batch_compare_cross_dataset(
                    source_profile, target_profiles_cache, source_candidates, 'both',
                    type_mapper=type_mapper
                )

                for score_dict in batch_scores:
                    target_bid = score_dict['target_bid']
                    target_status = target_status_map.get(target_bid, ConnectivityStatus.NONE)
                    target_type_str = target_type_lookup.get(target_bid, '')

                    rank_corr_value = score_dict['rank']
                    rank_union_raw = score_dict['rank_union']

                    # Optional score filter
                    if min_score is not None and (pd.isna(rank_corr_value) or rank_corr_value < min_score):
                        continue

                    all_results.append({
                        'source_bodyId': source_bid,
                        'source_type': source_type_str,
                        'target_bodyId': target_bid,
                        'target_type': target_type_str,
                        'target_dataset': target_dataset,
                        'adjacency_score': int(score_dict.get('shared_count', 0)),
                        'shared_type_count': score_dict['shared_type_count'],
                        'union_type_count': score_dict['union_type_count'],
                        'rank_corr': rank_corr_value,
                        # rank_union is the RAW union-based Spearman correlation
                        # (sign meaningful, 0 = no monotonic relation)
                        'rank_union': rank_union_raw,
                        'jaccard': score_dict['jaccard'],
                        'weighted_jaccard': score_dict.get('weighted_jaccard', 0.0),
                        'cosine': score_dict['cosine'],
                        'is_same_type': source_type_str == target_type_str if target_type_str else False,
                        'is_same_dataset': not is_cross_dataset,
                        'source_status': source_status.value,
                        'target_status': target_status.value,
                        'weak_source': not source_status.is_valid_for_comparison(),
                        'weak_target': not target_status.is_valid_for_comparison(),
                        'source_partner_count': source_partner_count,
                        'target_partner_count': score_dict['target_type_count']
                    })
            else:
                for target_bid, shared_count in source_candidates.items():
                    target_profile = target_profiles_cache.get(target_bid)
                    if target_profile is None:
                        continue

                    target_status = target_status_map.get(target_bid, ConnectivityStatus.NONE)
                    scores = ProfileComparator.combined_score_intra_dataset(source_profile, target_profile)

                    rank_corr_value = scores['rank']
                    rank_union_raw = scores.get('rank_union', np.nan)

                    if min_score is not None and (pd.isna(rank_corr_value) or rank_corr_value < min_score):
                        continue

                    target_type_str = target_type_lookup.get(target_bid, '')
                    target_partner_count = len(ProfileComparator._get_all_bodyids(target_profile, 'both'))

                    all_results.append({
                        'source_bodyId': source_bid,
                        'source_type': source_type_str,
                        'target_bodyId': target_bid,
                        'target_type': target_type_str,
                        'target_dataset': target_dataset,
                        'adjacency_score': int(shared_count),
                        'shared_type_count': scores.get('shared_type_count', 0),
                        'union_type_count': scores.get('union_type_count', 0),
                        'rank_corr': rank_corr_value,
                        'rank_union': rank_union_raw,
                        'jaccard': scores['jaccard'],
                        'weighted_jaccard': scores.get('weighted_jaccard', 0.0),
                        'cosine': scores['cosine'],
                        'is_same_type': source_type_str == target_type_str if target_type_str else False,
                        'is_same_dataset': not is_cross_dataset,
                        'source_status': source_status.value,
                        'target_status': target_status.value,
                        'weak_source': not source_status.is_valid_for_comparison(),
                        'weak_target': not target_status.is_valid_for_comparison(),
                        'source_partner_count': source_partner_count,
                        'target_partner_count': target_partner_count
                    })

        # Build intra-type comparison if requested
        intra_type_df = pd.DataFrame()
        if include_intra_type and len(source_bodyids) > 1:
            intra_type_results = []
            source_bid_list = sorted(source_bodyids)
            for i, bid_a in enumerate(source_bid_list):
                profile_a = source_profiles_cache.get(bid_a)
                if profile_a is None:
                    continue
                for bid_b in source_bid_list[i+1:]:
                    profile_b = source_profiles_cache.get(bid_b)
                    if profile_b is None:
                        continue

                    scores = ProfileComparator.combined_score_intra_dataset(profile_a, profile_b)
                    rank_corr_value = scores['rank']
                    rank_union_raw = scores.get('rank_union', np.nan)

                    source_partner_count = len(ProfileComparator._get_all_bodyids(profile_a, 'both'))
                    target_partner_count = len(ProfileComparator._get_all_bodyids(profile_b, 'both'))

                    status_a = source_status_map.get(bid_a, ConnectivityStatus.NONE)
                    status_b = source_status_map.get(bid_b, ConnectivityStatus.NONE)

                    intra_type_results.append({
                        'source_bodyId': bid_a,
                        'source_type': source_type_lookup.get(bid_a, str(bid_a)),
                        'target_bodyId': bid_b,
                        'target_type': source_type_lookup.get(bid_b, str(bid_b)),
                        'target_dataset': target_dataset,
                        'adjacency_score': 0,
                        'shared_type_count': scores.get('shared_type_count', 0),
                        'union_type_count': scores.get('union_type_count', 0),
                        'rank_corr': rank_corr_value,
                        'rank_union': rank_union_raw,
                        'jaccard': scores['jaccard'],
                        'weighted_jaccard': scores.get('weighted_jaccard', 0.0),
                        'cosine': scores['cosine'],
                        'is_same_type': True,
                        'is_same_dataset': not is_cross_dataset,
                        'source_status': status_a.value,
                        'target_status': status_b.value,
                        'weak_source': not status_a.is_valid_for_comparison(),
                        'weak_target': not status_b.is_valid_for_comparison(),
                        'source_partner_count': source_partner_count,
                        'target_partner_count': target_partner_count
                    })

            if intra_type_results:
                intra_type_df = pd.DataFrame(intra_type_results)

        results_df = pd.DataFrame(all_results)

        # Sorting and top-N trimming
        if not results_df.empty:
            sort_col = similarity_metric if similarity_metric in results_df.columns else 'jaccard'
            results_df = results_df.sort_values(['source_bodyId', sort_col], ascending=[True, False], na_position='last')
            if top_n > 0:
                results_df = results_df.groupby('source_bodyId').head(top_n).reset_index(drop=True)

        self._in_progress_bar = was_in_progress
        return (
            results_df,
            intra_type_df,
            skipped_sources,
            warned_sources,
            source_status_map,
            target_status_map,
            target_status_counts,
        )
    
    def _build_bodyid_type_lookup(
        self,
        conn_df: pd.DataFrame,
        dataset: Optional[str] = None,
    ) -> Dict[int, str]:
        """
        Build a bodyId to type lookup dictionary from connection DataFrame.

        Args:
            conn_df: Connection DataFrame with bodyId_pre, bodyId_post, type_pre, type_post columns
            dataset: Optional dataset identifier. When given, type values that
                are coarse class labels (some datasets backfill the ``type``
                column of untyped neurons with the row's class / super-class,
                e.g. FAFB v783 writing ``optic_lobes``) are skipped so they
                cannot leak into candidate expansion or result rows.

        Returns:
            Dict mapping bodyId (int) to type (str)
        """
        coarse = self._coarse_type_labels(dataset) if dataset else frozenset()
        type_lookup: Dict[int, str] = {}

        def _add(bid, typ) -> None:
            if pd.notna(bid) and pd.notna(typ) and str(typ).strip():
                if coarse and str(typ).strip().lower() in coarse:
                    return
                try:
                    type_lookup[int(bid)] = str(typ)
                except (ValueError, TypeError):
                    pass

        # Get types from pre neurons
        for _, row in conn_df[['bodyId_pre', 'type_pre']].drop_duplicates().iterrows():
            _add(row['bodyId_pre'], row['type_pre'])

        # Get types from post neurons (may add new entries or override)
        for _, row in conn_df[['bodyId_post', 'type_post']].drop_duplicates().iterrows():
            _add(row['bodyId_post'], row['type_post'])

        return type_lookup

    # Class columns consulted when deciding whether a `type` value is a
    # real cell-type annotation or a coarse category backfilled by the
    # dataset's neuron table.
    _CLASS_LABEL_COLUMNS = ('cell_class', 'class', 'super_class', 'subclass')
    _coarse_type_labels_cache: Dict[str, frozenset] = {}

    def _coarse_type_labels(self, dataset: Optional[str],
                            project_root: Optional[str] = None) -> frozenset:
        """Class-level labels the dataset's neuron table writes into its
        ``type`` column for untyped neurons.

        A ``type`` value is considered coarse when somewhere in the table a
        row carries it as ``type`` while the same row's class column holds
        the same value (e.g. FAFB v783 ``type='optic_lobes'`` with
        ``cell_class='optic_lobes'``). Results are cached per dataset;
        unknown datasets return an empty set (nothing is blanked).
        """
        if not dataset:
            return frozenset()
        key = str(dataset)
        root_key = str(project_root or '')
        cache_key = f'{root_key}|{key}'
        cached = self._coarse_type_labels_cache.get(cache_key)
        if cached is not None:
            return cached

        labels: set = set()
        safe_name = canonical_dataset_name(key).replace(':', '_').replace('.', '_')
        root = Path(project_root) if project_root \
            else Path(__file__).parent.parent.parent
        candidates = [
            root / 'neuron_indexes' / safe_name / 'neuron_index.parquet',
            root / 'datasets' / safe_name / f'{safe_name}_allneurons_neuron_df.parquet',
            root / 'datasets' / safe_name / f'{safe_name}_allneurons_neuron_df.csv',
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                if path.suffix == '.parquet':
                    frame = pd.read_parquet(path)
                else:
                    frame = pd.read_csv(path, low_memory=False)
            except Exception:
                continue
            if 'type' not in frame.columns:
                continue
            type_vals = frame['type'].astype('string').str.strip().str.lower()
            for cls_col in self._CLASS_LABEL_COLUMNS:
                if cls_col not in frame.columns:
                    continue
                cls_vals = frame[cls_col].astype('string').str.strip().str.lower()
                backfilled = type_vals.notna() & (type_vals == cls_vals)
                labels.update(type_vals[backfilled].dropna().tolist())
            break
        coarse = frozenset(labels)
        self._coarse_type_labels_cache[cache_key] = coarse
        return coarse

    
    def _build_profile_from_bodyid_aggregates(
        self,
        bodyid: int,
        dataset: str,
        upstream_by_bodyid: Dict[int, Dict[str, float]],
        downstream_by_bodyid: Dict[int, Dict[str, float]],
        top_k: Optional[int] = None,
        top_m: Optional[int] = None
    ) -> Optional['ConnectivityProfile']:
        """
        DEPRECATED: Use self.profiler.get_profile(bodyid, dataset) instead.
        
        This method builds profiles from pre-aggregated data WITHOUT the 1-hop/2-hop
        hybrid approach. All profile building should now go through ConnectivityProfiler
        which provides proper 2-hop expansion for untyped 1-hop partners.
        
        Args:
            bodyid: BodyId of the neuron
            dataset: Dataset name
            upstream_by_bodyid: Pre-aggregated upstream connections by bodyId
            downstream_by_bodyid: Pre-aggregated downstream connections by bodyId
            top_k: Top K partners to keep per direction
            top_m: Minimum unique types to ensure
            
        Returns:
            ConnectivityProfile or None if no connections found
            
        .. deprecated::
            Use `self.profiler.get_profile(bodyid, dataset)` for proper 1-hop/2-hop
            hybrid profile building with 2-hop expansion for untyped neurons.
        """
        import warnings
        warnings.warn(
            "_build_profile_from_bodyid_aggregates is deprecated. "
            "Use self.profiler.get_profile(bodyid, dataset) for proper 1-hop/2-hop hybrid profiles.",
            DeprecationWarning,
            stacklevel=2
        )
        from .connectivity_profiler import ConnectivityProfile
        
        # Get defaults from profiler config
        if top_k is None:
            top_k = self.profiler.config.top_k_bodyid
        if top_m is None:
            top_m = self.profiler.config.top_m_type
        
        max_expansion_factor = self.profiler.config.max_expansion_factor
        dynamic_expansion = self.profiler.config.dynamic_expansion
        
        upstream_all = upstream_by_bodyid.get(bodyid, {})
        downstream_all = downstream_by_bodyid.get(bodyid, {})
        
        if not upstream_all and not downstream_all:
            return None
        
        def apply_top_k_m(partners: Dict[str, float], top_k: int, top_m: int) -> Tuple[Dict[str, float], int]:
            """Apply top-k filtering with optional top-m expansion."""
            if not partners:
                return {}, top_k
            
            # Sort by weight descending
            sorted_partners = sorted(partners.items(), key=lambda x: -x[1])
            
            k_used = top_k
            max_k = top_k * max_expansion_factor
            
            if dynamic_expansion and top_m > 0:
                selected = dict(sorted_partners[:k_used])
                unique_types = len(selected)
                
                while unique_types < top_m and k_used < max_k and k_used < len(sorted_partners):
                    k_used += 5
                    selected = dict(sorted_partners[:k_used])
                    unique_types = len(selected)
                
                return selected, k_used
            else:
                return dict(sorted_partners[:top_k]), top_k
        
        upstream_partners, up_k_used = apply_top_k_m(upstream_all, top_k, top_m)
        downstream_partners, down_k_used = apply_top_k_m(downstream_all, top_k, top_m)
        
        up_total = sum(upstream_partners.values()) if upstream_partners else 0.0
        down_total = sum(downstream_partners.values()) if downstream_partners else 0.0
        
        upstream_ranked = sorted(upstream_partners.items(), key=lambda x: -x[1])
        downstream_ranked = sorted(downstream_partners.items(), key=lambda x: -x[1])
        
        upstream_ranks = {k: i+1 for i, (k, _) in enumerate(upstream_ranked)}
        downstream_ranks = {k: i+1 for i, (k, _) in enumerate(downstream_ranked)}
        
        is_weak = len(upstream_partners) < 5 or len(downstream_partners) < 5
        is_sparse = (
            (top_m > 0 and len(upstream_partners) < top_m) or
            (top_m > 0 and len(downstream_partners) < top_m)
        )
        
        return ConnectivityProfile(
            neuron_id=bodyid,
            dataset=dataset,
            upstream_partners=upstream_partners,
            downstream_partners=downstream_partners,
            upstream_ranks=upstream_ranks,
            downstream_ranks=downstream_ranks,
            upstream_top_k=up_k_used,
            downstream_top_k=down_k_used,
            total_upstream_weight=up_total,
            total_downstream_weight=down_total,
            actual_upstream_count=len(upstream_partners),
            actual_downstream_count=len(downstream_partners),
            is_weak_connectivity=is_weak,
            unique_types_upstream=len(upstream_partners),
            unique_types_downstream=len(downstream_partners),
            is_sparse=is_sparse,
            top_k_bodyid_used=max(up_k_used, down_k_used),
            top_m_type_target=top_m,
        )
    
    def find_homologs_fast(
        self,
        source: Optional[Union[str, int]] = None,
        source_dataset: Optional[str] = None,
        target_dataset: Optional[str] = None,
        top_n: Optional[int] = None,
        min_shared_partners: Optional[int] = None,
        vector_prune_fraction: Optional[float] = None,
        prefilter_rank_metric: Optional[str] = None,
        min_weight: int = 3,
        show_progress: bool = True,
        output_dir: Optional[str] = None,
        saveas: Optional[str] = None,
        include_partner_details: bool = True,
        top_n_details: int = 10,
        run_shuffle_test: bool = False,
        n_shuffles: int = 100,
        shuffle_seed: Optional[int] = None,
        visualize_skeleton: Optional[bool] = None,
        visualize_top_n: Optional[int] = None,
        similarity_metric: Optional[str] = None,
        extra_readme_params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Fast homolog discovery via adjacency expansion.
        
        This method uses adjacency expansion to efficiently find candidates:
        
        Cross-Dataset Algorithm (type-based expansion):
            1. Get source neuron's upstream types (A) and downstream types (B)
            2. Find target bodyIds with types in A → set A' (same types as source's upstream)
            3. Find target bodyIds with types in B → set B' (same types as source's downstream)
            4. Get A' neurons' downstream types → set C (upstream's downstream)
            5. Get B' neurons' upstream types → set D (downstream's upstream)
            6. Build profiles for candidates in A' ∪ B' ∪ C ∪ D
            7. Compare with source profiles
        
        Same-Dataset Algorithm (shared-partner expansion):
            1. Find bodyIds that share upstream or downstream partners with source
            2. Build profiles for those candidates
            3. Compare with source profiles
        
        Profile Building (via ConnectivityProfiler):
            - top_k: Top K partners per direction by synapse weight
            - top_m: Minimum unique types via dynamic K expansion
            - expand_untyped_2hop: 2-hop typed partners for untyped 1-hop neurons
        
        Uses module-level defaults (self.source, self.source_dataset, etc.) 
        when parameters are not provided.
        
        For comprehensive search of the ENTIRE target dataset, use find_homologs().
        
        Args:
            source: Neuron type name (str) or bodyId (int).
                   Uses self.source if not provided.
            source_dataset: Source dataset. Uses self.source_dataset if not provided.
            target_dataset: Target dataset to search. Uses self.target_dataset if not provided.
            top_n: Maximum candidates to return per source neuron (default: 20)
            min_shared_partners: Minimum shared partners to be a candidate
                (default: 2). Lower = looser search (1 = any shared partner).
            vector_prune_fraction: Fraction of cosine-positive candidates kept by
                vector pre-filtering (default: 0.05 = top 5%).
                1.0 keeps ALL cosine-positive candidates (loosest search).
            prefilter_rank_metric: Ranking key for the pre-filter prune
                ('cosine' = highest weighted profile cosine, default;
                'adjacency' = legacy shared partner-type count).
            min_weight: Minimum synapse weight for candidate discovery (default: 3)
            show_progress: Show progress bar
            output_dir: Directory to save results. Uses self.output_dir if not provided.
            saveas: Custom folder name. Uses self.saveas if not provided.
            include_partner_details: Include per-match partner overlap CSVs (when saving)
            top_n_details: Number of top matches to include detailed profiles for
            run_shuffle_test: If True, run random control test to validate results
            n_shuffles: Number of shuffle iterations for control test (default: 100)
            shuffle_seed: Random seed for reproducibility of shuffle test
            visualize_skeleton: Generate 3D visualizations. Uses self.visualize_skeleton if None.
            visualize_top_n: Number of top candidates to visualize. Uses self.visualize_top_n if None.
            similarity_metric: Metric for sorting top-N candidates. Options:
                - 'rank_corr': Raw Spearman rank correlation on shared partners only
                - 'rank_union': Raw Spearman rank correlation on union of all partners (missing=0)
                - 'jaccard': Jaccard similarity
                - 'cosine': Cosine similarity
        
        Returns:
            DataFrame with columns (bodyId-to-bodyId comparison):
                - source_bodyId: Source neuron bodyId
                - source_type: Source neuron type
                - target_bodyId: Target neuron bodyId
                - target_type: Target neuron type
                - target_dataset: Target dataset
                - adjacency_score: Number of shared partners found during candidate discovery
                - shared_type_count: Number of types used for rank_corr calculation
                - union_type_count: Total unique types/bodyIds in union
                - rank_corr: Raw Spearman correlation [-1,1] (shared partners only)
                - rank_union: Raw Spearman correlation [-1,1] on the partner
                  union (missing = 0); sign is meaningful, 0 = no relation
                - jaccard: Jaccard similarity
                - cosine: Cosine similarity
                - is_same_type: True if source and target types match
            
            Results are sorted by source_bodyId then selected similarity_metric (descending).
            
            If run_shuffle_test=True, additional columns are added:
                - shuffle_p_value: P-value from shuffle test
                - shuffle_z_score: Z-score from shuffle test
                - shuffle_significant: True if p < 0.05
        
        Output Files (always saved when output_dir is set):
            - bodyid_results.csv: BodyId-level comparisons (sorted by source_bodyId, metric)
            - intra_type_results.csv: Intra-type self-comparisons (when source is a type)
            - type_summary.csv: Type-mean aggregated FROM bodyId-level results
        - type_level_results.csv: True type-level ranking (pooled type profiles)
            - homolog_results.csv: Legacy format (sorted by metric only)
            - visualization/: 3D skeleton plots (if visualize_skeleton=True)
        
        Note:
            - All profile building uses ConnectivityProfiler's 1-hop/2-hop hybrid approach
            - Always compares individual bodyIds for comprehensive results
            - Uses adjacency expansion to find candidates with shared partners
            - For comprehensive search without candidate filtering, use find_homologs()
        """
        # Per-search mapper state: a fresh snapshot (never a stale load
        # state on a long-lived finder) and a fresh resolution-count basis,
        # so this search's auto_type_mapping.json owns exactly its own
        # candidate-expansion resolutions.
        self._mapper_snapshot = None
        self._counted_expansion_resolutions = set()
        self._mapping_status_counts = {}

        # Use module-level defaults where not provided
        query = source if source is not None else self.source
        source_dataset = source_dataset if source_dataset is not None else self.source_dataset
        target_dataset = target_dataset if target_dataset is not None else self.target_dataset
        output_dir = output_dir if output_dir is not None else self.output_dir
        saveas = saveas if saveas is not None else self.saveas
        visualize_skeleton = visualize_skeleton if visualize_skeleton is not None else self.visualize_skeleton
        visualize_top_n = visualize_top_n if visualize_top_n is not None else self.visualize_top_n
        similarity_metric = similarity_metric if similarity_metric is not None else self.similarity_metric
        top_n = top_n if top_n is not None else self.top_n
        min_shared_partners = min_shared_partners if min_shared_partners is not None else self.min_shared_partners
        vector_prune_fraction = vector_prune_fraction if vector_prune_fraction is not None else self.vector_prune_fraction
        prefilter_rank_metric = (
            str(prefilter_rank_metric or self.prefilter_rank_metric).strip().lower()
        )

        # Validate required parameters
        if query is None:
            raise ValueError("No source neuron specified. Set 'source' parameter or self.source.")
        if source_dataset is None:
            raise ValueError("No source_dataset specified. Set 'source_dataset' parameter or self.source_dataset.")
        if target_dataset is None:
            raise ValueError("No target_dataset specified. Set 'target_dataset' parameter or self.target_dataset.")
        
        is_bodyid = isinstance(query, int) or (isinstance(query, str) and str(query).isdigit())
        query_label = f"bodyId:{query}" if is_bodyid else query
        is_cross_dataset = (source_dataset != target_dataset)
        
        self._log(f"Fast homolog search: {query_label} from {source_dataset} → {target_dataset}")
        
        # Step 0: Optionally ensure connection caches are complete BEFORE loading.
        # This is opt-in because completing a full dataset cache can fetch
        # connections for 100k+ neurons (hours on first use). By default the
        # search fetches only the connections it needs via FindNeuronConnection.
        if self.ensure_cache_complete:
            self._log("Ensuring connection caches are complete...")
            self._ensure_connection_cache_complete(source_dataset)
            if is_cross_dataset:
                self._log(f"Checking target dataset: {target_dataset}...")
                self._ensure_connection_cache_complete(target_dataset)
            self._log("Connection caches verified.")
        else:
            self._log("Skipping full-dataset cache build (ensure_cache_complete=False).")

        # Step 0.5: Pre-warm profile caches so subsequent loads avoid cold reads
        self._log("Pre-warming profile caches...")
        self._prewarm_profile_cache(source_dataset)
        if is_cross_dataset:
            self._prewarm_profile_cache(target_dataset)
        
        # Pooled all-adjacency type profiles for the true type-level view
        # (built during the loading block, before the frames are released).
        source_pooled_profiles: Dict[str, 'ConnectivityProfile'] = {}
        target_pooled_profiles: Dict[str, 'ConnectivityProfile'] = {}

        # Step 1: Load connection caches
        self._progress(1, 6, "Loading connection data")
        self._log("Loading and processing connection data...")
        self._in_progress_bar = True
        try:
            with tqdm(total=4, desc="Loading data", disable=progress_bars_disabled(show_progress, self.verbose), leave=True) as pbar:
                # Load source cache (already complete from Step 0)
                source_conn = self._load_connection_cache(source_dataset, auto_build=False)
                if source_conn is None:
                    self._log(f"ERROR: Could not load connection cache for {source_dataset}")
                    return pd.DataFrame()
                pbar.update(1)
                pbar.set_description("Aggregating source")
                
                # Pre-aggregate source (type-level)
                source_up, source_down = self._build_type_aggregates(source_conn, min_weight, show_progress=False)
                
                # Build bodyId-level aggregates and lookups up-front so both type and bodyId
                # queries can reuse the shared comparison core.
                source_bodyid_up, source_bodyid_down = self._build_bodyid_aggregates(
                    source_conn, min_weight, show_progress=False
                )
                source_type_lookup = self._build_bodyid_type_lookup(source_conn, source_dataset)

                # Pooled all-adjacency type profiles for the query's source
                # type(s) (true type-level view; built before source_conn is
                # released, costs one groupby over the connection frame).
                source_pooled_profiles: Dict[str, 'ConnectivityProfile'] = {}
                try:
                    if is_bodyid:
                        _src_bids = [int(query)]
                    else:
                        _src_bids = self.get_bodyids_for_type(str(query), source_dataset)
                    _bid_to_label = {}
                    for _b in _src_bids:
                        _t = source_type_lookup.get(int(_b))
                        if _t:
                            _bid_to_label[int(_b)] = _t
                    source_pooled_profiles = ProfileComparator.build_pooled_type_profiles(
                        source_conn, _bid_to_label, dataset=source_dataset,
                        min_weight=self.min_synapse_threshold,
                        normalize_types=lambda s: self.profiler._normalize_types_vectorized(
                            s, self.profiler.config.fuzzy_match))
                    self._log(f"Pooled type profiles built for "
                              f"{len(source_pooled_profiles)} source type(s)")
                except Exception as e:
                    self._log(f"Type-level source pooling skipped: {e}")

                # WORKFLOW OPTIMIZATION: Build source profiles NOW while source connections are in memory
                # This avoids reloading source connections later when building profiles
                # Get source bodyIds for the query type
                if not is_bodyid:
                    source_bodyids_early = self.get_bodyids_for_type(str(query), source_dataset)
                    if source_bodyids_early:
                        # Set progress bar flag for profiler to use tqdm.write
                        self.profiler._in_progress_bar = True
                        
                        # Pre-load connection cache into profiler BEFORE logging (avoids log interruption)
                        try:
                            self.profiler._get_cached_conn_df(source_dataset)
                        except Exception:
                            pass
                        
                        # Consolidate any existing batch files first
                        try:
                            self.profiler.consolidate_profile_cache(source_dataset)
                        except Exception:
                            pass
                        
                        self._progress(2, 6, "Building source profiles")
                        tqdm.write(f"[HomologFinder] Pre-building {len(source_bodyids_early)} source profiles while connections are loaded...")
                        
                        # Build source profiles with deferred writes
                        self.profiler._defer_cache_writes = True
                        source_profiles_early: Dict[int, Any] = {}
                        
                        for bid in source_bodyids_early:
                            try:
                                profile = self.profiler.get_profile(bid, source_dataset)
                                if profile is not None:
                                    source_profiles_early[bid] = profile
                            except Exception:
                                pass
                        
                        # Flush cache writes
                        self.profiler._defer_cache_writes = False
                        try:
                            self.profiler.flush_pending_cache_writes(silent=True)
                        except Exception:
                            pass
                        
                        # Consolidate batch files into main cache
                        try:
                            self.profiler.consolidate_profile_cache(source_dataset)
                        except Exception:
                            pass
                        
                        tqdm.write(f"[HomologFinder] Pre-built {len(source_profiles_early)} source profiles")
                        self.profiler._in_progress_bar = False
                pbar.update(1)
                
                # Release source before loading target to avoid OOM
                if is_cross_dataset:
                    self._log("Releasing source DataFrame before loading target...")
                    del source_conn
                    
                    # Also clear FNC's cache for source dataset to free memory
                    source_safe_name = canonical_dataset_name(source_dataset).replace(':', '_').replace('.', '_')
                    try:
                        from coana import _FNC_CACHE
                        if source_safe_name in _FNC_CACHE:
                            if 'conn_df' in _FNC_CACHE[source_safe_name]:
                                _FNC_CACHE[source_safe_name]['conn_df'] = None
                            self._log(f"Cleared FNC cache for {source_dataset}")
                    except ImportError:
                        pass
                    
                    import gc
                    gc.collect()
                    
                    pbar.set_description("Loading target")
                    target_conn = self._load_connection_cache(target_dataset, auto_build=False)
                    if target_conn is None:
                        self._log(f"ERROR: Could not load connection cache for {target_dataset}")
                        return pd.DataFrame()
                    pbar.update(1)
                    pbar.set_description("Aggregating target")
                    target_up, target_down = self._build_type_aggregates(target_conn, min_weight, show_progress=False)
                    
                    # Build target bodyId aggregates and lookups for shared-core comparison
                    target_bodyid_up, target_bodyid_down = self._build_bodyid_aggregates(
                        target_conn, min_weight, show_progress=False
                    )
                    target_type_lookup = self._build_bodyid_type_lookup(target_conn, target_dataset)

                    # Pooled all-adjacency type profiles for ALL typed target
                    # types (one groupby over the connection frame).
                    try:
                        target_pooled_profiles = ProfileComparator.build_pooled_type_profiles(
                            target_conn, dict(target_type_lookup),
                            dataset=target_dataset,
                            min_weight=self.min_synapse_threshold,
                            normalize_types=lambda s: self.profiler._normalize_types_vectorized(
                                s, self.profiler.config.fuzzy_match))
                        self._log(f"Pooled type profiles built for "
                                  f"{len(target_pooled_profiles)} target types")
                    except Exception as e:
                        target_pooled_profiles = {}
                        self._log(f"Type-level target pooling skipped: {e}")
                    
                    # Release target after building aggregates
                    del target_conn
                    gc.collect()
                    pbar.update(1)
                else:
                    # Same dataset - reuse source aggregates
                    target_conn = None  # Don't need a separate reference
                    target_up, target_down = source_up, source_down
                    target_bodyid_up, target_bodyid_down = source_bodyid_up, source_bodyid_down
                    target_type_lookup = source_type_lookup
                    pbar.update(2)

                    try:
                        target_pooled_profiles = ProfileComparator.build_pooled_type_profiles(
                            source_conn, dict(target_type_lookup),
                            dataset=target_dataset,
                            min_weight=self.min_synapse_threshold,
                            normalize_types=lambda s: self.profiler._normalize_types_vectorized(
                                s, self.profiler.config.fuzzy_match))
                        self._log(f"Pooled type profiles built for "
                                  f"{len(target_pooled_profiles)} target types")
                    except Exception as e:
                        target_pooled_profiles = {}
                        self._log(f"Type-level target pooling skipped: {e}")
        finally:
            self._in_progress_bar = False
        
        # =====================================================================
        # BodyId-to-bodyId comparison (always used for comprehensive results)
        # Type-level aggregation is done when saving results
        # =====================================================================
        if not is_bodyid:
            # Type query: Get all bodyIds for this type
            source_bodyids = self.get_bodyids_for_type(str(query), source_dataset)
            if not source_bodyids:
                self._log(f"ERROR: No bodyIds found for type '{query}' in {source_dataset}")
                return pd.DataFrame()
            self._log(f"Running bodyId-to-bodyId comparison for {len(source_bodyids)} neurons in type '{query}'")
            
            # Aggregates already built above during loading phase
            # Memory already freed - source_conn and target_conn no longer exist
            if is_cross_dataset:
                # source_type_lookup already set
                pass
            else:
                source_type_lookup = target_type_lookup
            
            # Use pre-built source profiles (built earlier while source connections were loaded)
            # This avoids reloading source connections
            from .connectivity_profiler import ConnectivityStatus
            source_profiles_cache: Dict[int, 'ConnectivityProfile'] = {}
            source_status_counts: Dict[str, int] = {s.value: 0 for s in ConnectivityStatus}
            source_status_map: Dict[int, ConnectivityStatus] = {}
            
            # Check if profiles were pre-built during the loading phase
            if 'source_profiles_early' in dir() and source_profiles_early:
                self._log(f"Using {len(source_profiles_early)} pre-built source profiles")
                source_profiles_cache = source_profiles_early
                for bid, profile in source_profiles_cache.items():
                    status = profile.connectivity_status
                    source_status_counts[status.value] += 1
                    source_status_map[bid] = status
            else:
                # Fallback: Build profiles now (may require reloading connection data)
                self._progress(2, 6, "Building source profiles")
                self._log("Building source profiles via ConnectivityProfiler (1-hop/2-hop hybrid)")
                
                # Consolidate any existing batch files first
                try:
                    self.profiler.consolidate_profile_cache(source_dataset)
                except Exception:
                    pass
                
                # Pre-load connection cache BEFORE starting progress bar
                self._log(f"Pre-loading connection cache for {source_dataset}...")
                try:
                    self.profiler._get_cached_conn_df(source_dataset)
                except Exception:
                    pass
                
                # Enable deferred cache writes for batch saving
                self.profiler._defer_cache_writes = True
                self._in_progress_bar = True
                self.profiler._in_progress_bar = True
                
                try:
                    pbar = tqdm(
                        source_bodyids, 
                        desc="Building source profiles",
                        disable=progress_bars_disabled(show_progress, self.verbose),
                        leave=False
                    )
                    
                    for bid in pbar:
                        try:
                            profile = self.profiler.get_profile(bid, source_dataset)
                            if profile is not None:
                                source_profiles_cache[bid] = profile
                                status = profile.connectivity_status
                                source_status_counts[status.value] += 1
                                source_status_map[bid] = status
                        except Exception as e:
                            tqdm.write(f"[HomologFinder] Warning: Could not build profile for source {bid}: {e}")
                    
                    pbar.close()
                finally:
                    self._in_progress_bar = False
                    self.profiler._in_progress_bar = False
                    self.profiler._defer_cache_writes = False
                    try:
                        self.profiler.flush_pending_cache_writes(silent=True)
                    except Exception:
                        pass
                    # Consolidate batch files into main cache
                    try:
                        self.profiler.consolidate_profile_cache(source_dataset)
                    except Exception:
                        pass
            
            # Log connectivity status breakdown
            status_summary = ", ".join([f"{k.upper()}: {v}" for k, v in source_status_counts.items() if v > 0])
            self._log(f"Source connectivity status breakdown: {status_summary}")
            
            # Identify sources to skip (NONE/ORPHAN) or warn (RARE/UNIDIRECTIONAL)
            none_sources = [bid for bid, status in source_status_map.items() 
                           if status in {ConnectivityStatus.NONE, ConnectivityStatus.ORPHAN}]
            warn_sources = [bid for bid, status in source_status_map.items() 
                           if status in {ConnectivityStatus.RARE, ConnectivityStatus.UNIDIRECTIONAL}]
            if none_sources:
                self._log(f"⚠️ {len(none_sources)} source neurons have no partners (NONE/ORPHAN) and will be skipped")
            if warn_sources:
                self._log(f"⚠️ WARNING: {len(warn_sources)} source neurons are sparse or unidirectional (RARE/UNIDIRECTIONAL) - results may be unreliable")
            
            all_results = []
            # Track which target bodyIds we need profiles for
            all_candidate_bodyids: set = set()
            candidate_map: Dict[int, Dict[int, int]] = {}  # source_bid -> {target_bid: shared_count}
            guaranteed_candidates: Dict[int, set] = {}  # source_bid -> never-prune same-type bids
            
            # For both cross-dataset and same-dataset: use adjacency expansion
            # Cross-dataset: find target neurons by TYPE matching (source partner types → target bodyIds)
            # Same-dataset: find target neurons by shared partners

            self._progress(3, 6, "Discovering candidate neurons (adjacency expansion)")
            if is_cross_dataset:
                # Cross-dataset adjacency expansion using TYPE matching:
                # Step 1: Get upstream types (A) and downstream types (B) from source neurons
                # Step 2: Find target bodyIds with those types (set A', B') - includes untyped
                # Step 3: Get A' downstream (C) and B' upstream (D) - includes untyped
                # Step 4: Candidate set = A' ∪ B' ∪ C ∪ D (includes both typed and untyped)
                # Note: Untyped neurons are only ignored in 2-hop profiles, not in candidate set
                
                # Collect all unique upstream/downstream types from source neurons
                # Include all source neurons (even weak ones) for comprehensive type collection
                self._log("Collecting source partner types...")
                all_upstream_types: set = set()
                all_downstream_types: set = set()
                
                for source_bid in source_bodyids:
                    source_profile = source_profiles_cache.get(source_bid)
                    if source_profile is None:
                        continue
                    all_upstream_types.update(source_profile.upstream_partners.keys())
                    all_downstream_types.update(source_profile.downstream_partners.keys())
                
                # Remove untyped placeholders from partner types
                all_upstream_types = {t for t in all_upstream_types if t and not pd.isna(t) and t != ''}
                all_downstream_types = {t for t in all_downstream_types if t and not pd.isna(t) and t != ''}
                
                self._log(f"Source partner types: {len(all_upstream_types)} upstream, {len(all_downstream_types)} downstream")
                
                # Build reverse lookup: type -> bodyIds in target dataset (typed neurons only)
                self._log("Building type-to-bodyId lookup for target dataset...")
                target_type_to_bodyids: Dict[str, set] = {}
                for bid, t in target_type_lookup.items():
                    if t and not pd.isna(t) and t != '':
                        if t not in target_type_to_bodyids:
                            target_type_to_bodyids[t] = set()
                        target_type_to_bodyids[t].add(bid)
                
                # Get all bodyIds in target dataset (for including untyped in C and D)
                all_target_bodyids = set(target_bodyid_up.keys()) | set(target_bodyid_down.keys())
                self._log(f"Target dataset: {len(all_target_bodyids)} total bodyIds")
                
                # Mapping-aware partner-type expansion: each source partner
                # type is resolved to the target-local names the shared
                # validity resolver licenses — unique renames and
                # bridge-derived targets, and EVERY branch of a valid split.
                # A conflict expands to nothing (fail closed: a conflicted
                # partner type must not match its raw same-name in the
                # target); an unmapped type keeps its raw name as the
                # explicitly-counted long-tail fallback.  The adjacency
                # scoring below reuses the SAME expanded sets, so
                # partner-type overlap is computed in one namespace instead
                # of comparing raw source names against raw target names.
                fast_mapper = self._get_type_mapper_for_comparison(True)
                fast_expansion_cache: Dict[str, Tuple[str, ...]] = {}
                expansion_up = self._expand_partner_types_to_target(
                    all_upstream_types, source_dataset, target_dataset,
                    fast_mapper, cache=fast_expansion_cache,
                    status_counts_out=self._mapping_status_counts)
                expansion_down = self._expand_partner_types_to_target(
                    all_downstream_types, source_dataset, target_dataset,
                    fast_mapper, cache=fast_expansion_cache,
                    status_counts_out=self._mapping_status_counts)
                expanded_upstream_names: set = {
                    name for names in expansion_up.values() for name in names}
                expanded_downstream_names: set = {
                    name for names in expansion_down.values() for name in names}

                # Set A': Target bodyIds with types matching source's upstream partners
                self._log("Computing Set A' (type-matched upstream)...")
                set_a_prime: set = set()
                for expanded in expansion_up.values():
                    for target_name in expanded:
                        set_a_prime.update(
                            target_type_to_bodyids.get(target_name, set()))

                # Set B': Target bodyIds with types matching source's downstream partners
                self._log("Computing Set B' (type-matched downstream)...")
                set_b_prime: set = set()
                for expanded in expansion_down.values():
                    for target_name in expanded:
                        set_b_prime.update(
                            target_type_to_bodyids.get(target_name, set()))

                self._log(f"Type-matched targets: A'={len(set_a_prime)}, B'={len(set_b_prime)}")
                
                # Precompute partner type sets per bodyId for fast union
                downstream_sets_by_bodyid = {bid: set(partners.keys()) for bid, partners in target_bodyid_down.items()}
                upstream_sets_by_bodyid = {bid: set(partners.keys()) for bid, partners in target_bodyid_up.items()}

                # Set C: Downstream of A' (upstream's downstream)
                self._log("Computing Set C (downstream of A') via type unions...")
                downstream_types_c: set = set()
                for bid in set_a_prime:
                    downstream_types_c.update(downstream_sets_by_bodyid.get(bid, set()))
                set_c: set = set()
                for partner_type in downstream_types_c:
                    if partner_type and not pd.isna(partner_type) and partner_type != '':
                        set_c.update(target_type_to_bodyids.get(partner_type, set()))

                # Set D: Upstream of B' (downstream's upstream)
                self._log("Computing Set D (upstream of B') via type unions...")
                upstream_types_d: set = set()
                for bid in set_b_prime:
                    upstream_types_d.update(upstream_sets_by_bodyid.get(bid, set()))
                set_d: set = set()
                for partner_type in upstream_types_d:
                    if partner_type and not pd.isna(partner_type) and partner_type != '':
                        set_d.update(target_type_to_bodyids.get(partner_type, set()))
                
                # Candidate set = A' ∪ B' ∪ C ∪ D (includes both typed and untyped)
                all_candidate_bodyids = set_a_prime | set_b_prime | set_c | set_d
                
                # Count typed vs untyped
                typed_candidates = {bid for bid in all_candidate_bodyids 
                                   if target_type_lookup.get(bid, '') and 
                                   not pd.isna(target_type_lookup.get(bid, '')) and
                                   target_type_lookup.get(bid, '') != ''}
                untyped_candidates = all_candidate_bodyids - typed_candidates
                
                self._log(f"Cross-dataset candidates: A'={len(set_a_prime)}, B'={len(set_b_prime)}, C={len(set_c)}, D={len(set_d)} → {len(all_candidate_bodyids)} unique ({len(typed_candidates)} typed, {len(untyped_candidates)} untyped)")

                # Same-named homologs bypass candidate pruning: target neurons
                # of the queried type are guaranteed comparison candidates even
                # if their cross-dataset adjacency score would lose to hub
                # types under vector prefiltering.
                guaranteed_candidates: Dict[int, set] = {}
                same_type_bids: set = set()
                if is_cross_dataset:
                    same_type_bids = self._compute_same_type_candidates(
                        [str(query)],
                        target_type_lookup,
                        self._get_type_mapper_for_comparison(is_cross_dataset),
                        target_dataset,
                        source_dataset=source_dataset,
                    )
                    if same_type_bids:
                        newly_added = len(same_type_bids - all_candidate_bodyids)
                        all_candidate_bodyids |= same_type_bids
                        self._log(
                            f"Guaranteed inclusion of {len(same_type_bids)} "
                            f"same-type targets for '{query}' "
                            f"({newly_added} outside adjacency expansion)"
                        )
                    guaranteed_candidates = {
                        source_bid: same_type_bids for source_bid in source_bodyids
                    }

                # For cross-dataset, all source neurons compare against same candidate pool
                # Compute adjacency_score as the count of partner-type overlaps per candidate
                # (shared upstream/downstream partner types relative to the
                # source partner sets, expanded into the target namespace)
                for source_bid in source_bodyids:
                    candidate_scores: Dict[int, int] = {}
                    for bid in all_candidate_bodyids:
                        shared_downstream = len(downstream_sets_by_bodyid.get(bid, set()) & expanded_upstream_names)
                        shared_upstream = len(upstream_sets_by_bodyid.get(bid, set()) & expanded_downstream_names)
                        score = shared_downstream + shared_upstream
                        candidate_scores[bid] = score
                    candidate_map[source_bid] = candidate_scores
            else:
                # Same-dataset: use adjacency expansion (shared partner types)
                # Include all source neurons, even with weak connectivity
                # Precompute partner-type → bodyId indexes for fast lookups
                from collections import defaultdict
                upstream_partner_to_bodyids: Dict[str, set] = defaultdict(set)
                downstream_partner_to_bodyids: Dict[str, set] = defaultdict(set)
                for bid, partners in target_bodyid_up.items():
                    for ptype in partners.keys():
                        if ptype and not pd.isna(ptype) and ptype != '':
                            upstream_partner_to_bodyids[ptype].add(bid)
                for bid, partners in target_bodyid_down.items():
                    for ptype in partners.keys():
                        if ptype and not pd.isna(ptype) and ptype != '':
                            downstream_partner_to_bodyids[ptype].add(bid)

                for source_bid in tqdm(
                    source_bodyids,
                    desc=f"Finding candidates for {query} neurons",
                    disable=progress_bars_disabled(show_progress, self.verbose),
                    leave=False
                ):
                    # Get source profile for candidate finding
                    source_profile = source_profiles_cache.get(source_bid)
                    
                    if source_profile is None:
                        continue
                    
                    # Get candidate bodyIds via adjacency expansion
                    # Find bodyIds that share partners with this source bodyId
                    upstream_types = set(source_profile.upstream_partners.keys())
                    downstream_types = set(source_profile.downstream_partners.keys())
                    
                    # Find target bodyIds that share upstream/downstream partners
                    candidate_bodyids: Dict[int, int] = {}  # bodyId -> shared partner count
                    
                    # Get bodyIds receiving from same upstream types
                    for up_type in upstream_types:
                        for target_bid in upstream_partner_to_bodyids.get(up_type, set()):
                            if target_bid == source_bid:
                                continue
                            candidate_bodyids[target_bid] = candidate_bodyids.get(target_bid, 0) + 1
                    
                    # Get bodyIds sending to same downstream types
                    for down_type in downstream_types:
                        for target_bid in downstream_partner_to_bodyids.get(down_type, set()):
                            if target_bid == source_bid:
                                continue
                            candidate_bodyids[target_bid] = candidate_bodyids.get(target_bid, 0) + 1
                    
                    # Filter by minimum shared partners
                    candidate_bodyids = {k: v for k, v in candidate_bodyids.items() if v >= min_shared_partners}
                    
                    candidate_map[source_bid] = candidate_bodyids
                    all_candidate_bodyids.update(candidate_bodyids.keys())
            
            # Build target profiles using the batch method with cache management
            target_profiles_cache: Dict[int, 'ConnectivityProfile'] = {}
            target_status_counts: Dict[str, int] = {s.value: 0 for s in ConnectivityStatus}
            target_status_map: Dict[int, ConnectivityStatus] = {}  # bodyId -> status
            
            if all_candidate_bodyids:
                self._progress(4, 6, "Building target profiles")
                target_profiles_cache = self._build_profiles_batch(
                    list(all_candidate_bodyids),
                    target_dataset,
                    show_progress=show_progress,
                    label="target"
                )
                # Track target connectivity status
                for bid, profile in target_profiles_cache.items():
                    status = profile.connectivity_status
                    target_status_counts[status.value] += 1
                    target_status_map[bid] = status
                
                # Log target status breakdown
                status_summary = ", ".join([f"{k.upper()}: {v}" for k, v in target_status_counts.items() if v > 0])
                self._log(f"Target connectivity status breakdown: {status_summary}")
            
            # Get type mapper for cross-dataset comparison
            type_mapper = self._get_type_mapper_for_comparison(is_cross_dataset)
            
            # Shared comparison core
            self._progress(5, 6, "Comparing & scoring candidates")
            results_df, intra_type_df, skipped_sources, warned_sources, source_status_map, target_status_map, target_status_counts = self._compare_candidates_core(
                source_bodyids=source_bodyids,
                source_profiles_cache=source_profiles_cache,
                source_status_map=source_status_map,
                target_profiles_cache=target_profiles_cache,
                target_type_lookup=target_type_lookup,
                source_type_lookup=source_type_lookup,
                candidate_map=candidate_map,
                is_cross_dataset=is_cross_dataset,
                target_dataset=target_dataset,
                show_progress=show_progress,
                similarity_metric=similarity_metric,
                top_n=top_n,
                min_score=None,
                include_intra_type=not is_cross_dataset,
                vector_prefiltering=self.vector_prefiltering,
            vector_prune_fraction=vector_prune_fraction,
                type_mapper=type_mapper,
                guaranteed_candidates=guaranteed_candidates,
                prefilter_rank_metric=prefilter_rank_metric,
            )

            # Log skipped and warned sources by status
            total_skipped = len(skipped_sources['none'])
            total_warned = len(warned_sources['rare_or_uni'])
            if total_skipped > 0:
                self._log(f"Skipped {total_skipped} source neurons with no partners (NONE/ORPHAN):")
                self._log(f"  - {skipped_sources['none'][:5]}{'...' if len(skipped_sources['none']) > 5 else ''}")
            if total_warned > 0:
                self._log(f"WARNING: {total_warned} source neurons are sparse/unidirectional (RARE/UNIDIRECTIONAL) - results may be unreliable:")
                self._log(f"  - {warned_sources['rare_or_uni'][:5]}{'...' if len(warned_sources['rare_or_uni']) > 5 else ''}")

            if results_df.empty:
                self._log("No valid candidates after profile comparison")
                return pd.DataFrame()

            self._log(f"Found {len(results_df)} bodyId-level matches across {len(source_bodyids)} source neurons")

            # Build source status summary for JSON export
            source_status_summary = {
                'query': str(query),
                'source_dataset': source_dataset,
                'total_source_neurons': len(source_bodyids),
                'status_breakdown': source_status_counts,
                'skipped_sources': {
                    'none': skipped_sources['none'],
                    'total': len(skipped_sources['none'])
                },
                'warned_sources': {
                    'rare_or_uni': warned_sources['rare_or_uni'],
                    'total': len(warned_sources['rare_or_uni'])
                },
                'included_for_comparison': len(source_bodyids) - len(skipped_sources['none']),
                'status_hierarchy': {
                    'none': 'No connections (0 partners) - SKIPPED entirely',
                    'rare_or_uni': 'Rare (<5 partners) or unidirectional - INCLUDED with WARNING',
                    'incomplete': 'Fewer than top_k partners - included with Warning',
                    'incomplete_expansion': 'Fewer than top_m unique types - included with Warning',
                    'complete': 'Full profile meeting all criteria'
                }
            }
            
            # Save results - always save (use default output_dir if not specified)
            save_output_dir = output_dir if output_dir is not None else self.output_dir
            # True type-level results: pooled all-adjacency type profiles
            # scored against every typed target type (cheap; ~1-2 s/source).
            type_level_df = None
            try:
                if source_pooled_profiles and target_pooled_profiles:
                    type_level_df = self._compute_type_level_results(
                        sorted(source_pooled_profiles.keys()), target_dataset,
                        similarity_metric=similarity_metric,
                        type_mapper=type_mapper,
                        source_pooled=source_pooled_profiles,
                        target_pooled=target_pooled_profiles,
                        show_progress=show_progress)
                    self._log(f"Type-level results computed for "
                              f"{type_level_df['source_type'].nunique() if not type_level_df.empty else 0} source type(s)")
            except Exception as e:
                self._log(f"Type-level results skipped: {e}")

            self._progress(6, 6, "Saving results")
            self._save_homolog_results_internal(
                results_df=results_df,
                query=query,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                output_dir=save_output_dir,
                saveas=saveas,
                direction='both',
                include_partner_details=include_partner_details,
                top_n_details=top_n_details,
                params={
                    'query': query,
                    'source_dataset': source_dataset,
                    'target_dataset': target_dataset,
                    'top_n': top_n,
                    'min_shared_partners': min_shared_partners,
                    'min_weight': min_weight,
                    'method': 'find_homologs_fast',
                    'source_bodyids_count': len(source_bodyids),
                    'profile_method': '1-hop/2-hop hybrid via ConnectivityProfiler',
                    'type_level': 'pooled all-adjacency type profiles',
                    **(extra_readme_params or {}),
                },
                visualize_skeleton=visualize_skeleton,
                visualize_top_n=visualize_top_n,
                type_level_df=type_level_df,
                top_n=top_n,
                intra_type_df=intra_type_df,  # Pass intra-type results for saving
                similarity_metric=self.similarity_metric,
                source_status_summary=source_status_summary
            )
            
            return results_df
        
        # =====================================================================
        # BodyId query: single neuron comparison (also used for single-bodyId type)
        # =====================================================================
        
        from .connectivity_profiler import ConnectivityStatus

        # Build source profile using ConnectivityProfiler (1-hop/2-hop hybrid)
        self._progress(2, 6, "Building source profile")
        self._log("Building source profile via ConnectivityProfiler (1-hop/2-hop hybrid)")
        source_profile = self.profiler.get_profile(query, source_dataset)
        if source_profile is None:
            self._log(f"ERROR: No connections found for {query_label}")
            return pd.DataFrame()

        source_bid = int(query)
        source_bodyids = [source_bid]

        upstream_types = set(source_profile.upstream_partners.keys())
        downstream_types = set(source_profile.downstream_partners.keys())

        self._log(f"Source profile: {len(upstream_types)} upstream, {len(downstream_types)} downstream types")

        # Build source caches for shared comparison core
        source_profiles_cache: Dict[int, 'ConnectivityProfile'] = {source_bid: source_profile}
        source_status_map: Dict[int, ConnectivityStatus] = {source_bid: source_profile.connectivity_status}
        source_status_counts: Dict[str, int] = {s.value: 0 for s in ConnectivityStatus}
        source_status_counts[source_profile.connectivity_status.value] += 1
        source_type_lookup_single = {
            source_bid: source_type_lookup.get(source_bid, source_profile.neuron_type or str(query))
        }

        # Step 3: Find candidates via adjacency expansion (2-hop neighbors, bodyId-level)
        self._progress(3, 6, "Discovering candidate neurons (adjacency expansion)")
        self._log("Finding candidate field via adjacency expansion (bodyId-level)...")

        candidate_map: Dict[int, Dict[int, int]] = {}
        all_candidate_bodyids: set = set()
        guaranteed_candidates: Dict[int, set] = {}

        if is_cross_dataset:
            # Cross-dataset: use partner TYPE overlap to expand candidates, then map to bodyIds
            self._log("Collecting source partner types for cross-dataset expansion...")
            all_upstream_types = {t for t in upstream_types if t and not pd.isna(t) and t != ''}
            all_downstream_types = {t for t in downstream_types if t and not pd.isna(t) and t != ''}

            self._log(f"Source partner types: {len(all_upstream_types)} upstream, {len(all_downstream_types)} downstream")

            # Build reverse lookup: type -> bodyIds in target dataset (typed neurons only)
            self._log("Building type-to-bodyId lookup for target dataset...")
            target_type_to_bodyids: Dict[str, set] = {}
            for bid, t in target_type_lookup.items():
                if t and not pd.isna(t) and t != '':
                    target_type_to_bodyids.setdefault(t, set()).add(bid)

            # Get all bodyIds in target dataset (for including untyped in C and D)
            all_target_bodyids = set(target_bodyid_up.keys()) | set(target_bodyid_down.keys())
            self._log(f"Target dataset: {len(all_target_bodyids)} total bodyIds")

            # Mapping-aware partner-type expansion (same policy as the
            # type-query branch): renames/bridges resolve to their target
            # names, valid splits expand to every branch, conflicts fail
            # closed, unmapped types keep their raw name.
            bid_mapper = self._get_type_mapper_for_comparison(True)
            expansion_up = self._expand_partner_types_to_target(
                all_upstream_types, source_dataset, target_dataset, bid_mapper,
                status_counts_out=self._mapping_status_counts)
            expansion_down = self._expand_partner_types_to_target(
                all_downstream_types, source_dataset, target_dataset,
                bid_mapper, status_counts_out=self._mapping_status_counts)
            expanded_upstream_names: set = {
                name for names in expansion_up.values() for name in names}
            expanded_downstream_names: set = {
                name for names in expansion_down.values() for name in names}

            # Set A': Target bodyIds with types matching source's upstream partners
            self._log("Computing Set A' (type-matched upstream)...")
            set_a_prime: set = set()
            for expanded in expansion_up.values():
                for target_name in expanded:
                    set_a_prime.update(
                        target_type_to_bodyids.get(target_name, set()))

            # Set B': Target bodyIds with types matching source's downstream partners
            self._log("Computing Set B' (type-matched downstream)...")
            set_b_prime: set = set()
            for expanded in expansion_down.values():
                for target_name in expanded:
                    set_b_prime.update(
                        target_type_to_bodyids.get(target_name, set()))

            # Precompute partner type sets per bodyId for fast union
            downstream_sets_by_bodyid = {bid: set(partners.keys()) for bid, partners in target_bodyid_down.items()}
            upstream_sets_by_bodyid = {bid: set(partners.keys()) for bid, partners in target_bodyid_up.items()}

            # Set C: Downstream of A' (upstream's downstream)
            self._log("Computing Set C (downstream of A') via type unions...")
            downstream_types_c: set = set()
            for bid in set_a_prime:
                downstream_types_c.update(downstream_sets_by_bodyid.get(bid, set()))
            set_c: set = set()
            for partner_type in downstream_types_c:
                if partner_type and not pd.isna(partner_type) and partner_type != '':
                    set_c.update(target_type_to_bodyids.get(partner_type, set()))

            # Set D: Upstream of B' (downstream's upstream)
            self._log("Computing Set D (upstream of B') via type unions...")
            upstream_types_d: set = set()
            for bid in set_b_prime:
                upstream_types_d.update(upstream_sets_by_bodyid.get(bid, set()))
            set_d: set = set()
            for partner_type in upstream_types_d:
                if partner_type and not pd.isna(partner_type) and partner_type != '':
                    set_d.update(target_type_to_bodyids.get(partner_type, set()))

            # Candidate set = A' ∪ B' ∪ C ∪ D (includes both typed and untyped)
            all_candidate_bodyids = set_a_prime | set_b_prime | set_c | set_d

            # Count typed vs untyped
            typed_candidates = {bid for bid in all_candidate_bodyids 
                               if target_type_lookup.get(bid, '') and 
                               not pd.isna(target_type_lookup.get(bid, '')) and
                               target_type_lookup.get(bid, '') != ''}
            untyped_candidates = all_candidate_bodyids - typed_candidates

            self._log(f"Cross-dataset candidates: A'={len(set_a_prime)}, B'={len(set_b_prime)}, C={len(set_c)}, D={len(set_d)} → {len(all_candidate_bodyids)} unique ({len(typed_candidates)} typed, {len(untyped_candidates)} untyped)")

            candidate_scores: Dict[int, int] = {}
            for bid in all_candidate_bodyids:
                shared_downstream = len(downstream_sets_by_bodyid.get(bid, set()) & expanded_upstream_names)
                shared_upstream = len(upstream_sets_by_bodyid.get(bid, set()) & expanded_downstream_names)
                candidate_scores[bid] = shared_downstream + shared_upstream
            candidate_map[source_bid] = candidate_scores

            # Same-named homologs bypass candidate pruning (see the type-query
            # branch): the query bodyId's own type is a guaranteed candidate.
            self_guaranteed: set = set()
            if is_cross_dataset:
                self_guaranteed = self._compute_same_type_candidates(
                    [getattr(source_profile, 'neuron_type', '') or ''],
                    target_type_lookup,
                    self._get_type_mapper_for_comparison(is_cross_dataset),
                    target_dataset,
                    source_dataset=source_dataset,
                )
                if self_guaranteed:
                    newly_added = len(self_guaranteed - all_candidate_bodyids)
                    all_candidate_bodyids |= self_guaranteed
                    for bid in self_guaranteed - candidate_scores.keys():
                        shared_downstream = len(downstream_sets_by_bodyid.get(bid, set()) & expanded_upstream_names)
                        shared_upstream = len(upstream_sets_by_bodyid.get(bid, set()) & expanded_downstream_names)
                        candidate_scores[bid] = shared_downstream + shared_upstream
                    self._log(
                        f"Guaranteed inclusion of {len(self_guaranteed)} "
                        f"same-type targets for '{source_profile.neuron_type}' "
                        f"({newly_added} outside adjacency expansion)"
                    )
            guaranteed_candidates = {source_bid: self_guaranteed}
        else:
            # Same-dataset: adjacency expansion using shared partner types
            from collections import defaultdict
            upstream_partner_to_bodyids: Dict[str, set] = defaultdict(set)
            downstream_partner_to_bodyids: Dict[str, set] = defaultdict(set)
            for bid, partners in target_bodyid_up.items():
                for ptype in partners.keys():
                    if ptype and not pd.isna(ptype) and ptype != '':
                        upstream_partner_to_bodyids[ptype].add(bid)
            for bid, partners in target_bodyid_down.items():
                for ptype in partners.keys():
                    if ptype and not pd.isna(ptype) and ptype != '':
                        downstream_partner_to_bodyids[ptype].add(bid)

            candidate_bodyids: Dict[int, int] = {}

            # Get bodyIds receiving from same upstream types
            for up_type in upstream_types:
                for target_bid in upstream_partner_to_bodyids.get(up_type, set()):
                    if target_bid == source_bid:
                        continue
                    candidate_bodyids[target_bid] = candidate_bodyids.get(target_bid, 0) + 1

            # Get bodyIds sending to same downstream types
            for down_type in downstream_types:
                for target_bid in downstream_partner_to_bodyids.get(down_type, set()):
                    if target_bid == source_bid:
                        continue
                    candidate_bodyids[target_bid] = candidate_bodyids.get(target_bid, 0) + 1

            # Filter by minimum shared partners
            candidate_bodyids = {k: v for k, v in candidate_bodyids.items() if v >= min_shared_partners}

            candidate_map[source_bid] = candidate_bodyids
            all_candidate_bodyids.update(candidate_bodyids.keys())

        if not all_candidate_bodyids:
            self._log("No candidates found after adjacency expansion")
            return pd.DataFrame()

        # Step 4: Build profiles for all candidates using ConnectivityProfiler (1-hop/2-hop hybrid)
        self._progress(4, 6, "Building target profiles")
        target_profiles_cache: Dict[int, 'ConnectivityProfile'] = {}
        target_status_counts: Dict[str, int] = {s.value: 0 for s in ConnectivityStatus}
        target_status_map: Dict[int, ConnectivityStatus] = {}

        target_profiles_cache = self._build_profiles_batch(
            list(all_candidate_bodyids),
            target_dataset,
            show_progress=show_progress,
            label="target"
        )
        for bid, profile in target_profiles_cache.items():
            status = profile.connectivity_status
            target_status_counts[status.value] += 1
            target_status_map[bid] = status

        if target_status_counts:
            status_summary = ", ".join([f"{k.upper()}: {v}" for k, v in target_status_counts.items() if v > 0])
            self._log(f"Target connectivity status breakdown: {status_summary}")

        # Get type mapper for cross-dataset comparison
        type_mapper = self._get_type_mapper_for_comparison(is_cross_dataset)
        
        # Step 5: Compare via shared core
        self._progress(5, 6, "Comparing & scoring candidates")
        results_df, intra_type_df, skipped_sources, warned_sources, source_status_map, target_status_map, target_status_counts = self._compare_candidates_core(
            source_bodyids=source_bodyids,
            source_profiles_cache=source_profiles_cache,
            source_status_map=source_status_map,
            target_profiles_cache=target_profiles_cache,
            target_type_lookup=target_type_lookup,
            source_type_lookup=source_type_lookup_single,
            candidate_map=candidate_map,
            is_cross_dataset=is_cross_dataset,
            target_dataset=target_dataset,
            show_progress=show_progress,
            similarity_metric=similarity_metric,
            top_n=top_n,
            min_score=None,
            include_intra_type=False,
            vector_prefiltering=self.vector_prefiltering,
        vector_prune_fraction=vector_prune_fraction,
            type_mapper=type_mapper,
            guaranteed_candidates=guaranteed_candidates,
            prefilter_rank_metric=prefilter_rank_metric,
        )

        # Log skipped and warned sources by status
        total_skipped = len(skipped_sources['none'])
        total_warned = len(warned_sources['rare_or_uni'])
        if total_skipped > 0:
            self._log(f"Skipped {total_skipped} source neurons with no partners (NONE/ORPHAN):")
            self._log(f"  - {skipped_sources['none'][:5]}{'...' if len(skipped_sources['none']) > 5 else ''}")
        if total_warned > 0:
            self._log(f"WARNING: {total_warned} source neurons are sparse/unidirectional (RARE/UNIDIRECTIONAL) - results may be unreliable:")
            self._log(f"  - {warned_sources['rare_or_uni'][:5]}{'...' if len(warned_sources['rare_or_uni']) > 5 else ''}")

        if results_df.empty:
            self._log("No valid candidates after profile comparison")
            return pd.DataFrame()

        self._log(f"Found {len(results_df)} bodyId-level matches")

        # Run shuffle test if requested (before saving so we include results)
        shuffle_stats = None
        if run_shuffle_test:
            self._log(f"Running shuffle control test with {n_shuffles} iterations...")

            query_profile = self.profiler.get_profile(
                query,
                source_dataset
            )

            if query_profile is None:
                self._log("Warning: Could not get query profile for shuffle test")
            else:
                shuffle_stats = self.run_random_control_test(
                    source=query,
                    source_dataset=source_dataset,
                    target_dataset=target_dataset,
                    n_shuffles=n_shuffles,
                    top_n=top_n,
                    seed=shuffle_seed,
                    show_progress=show_progress
                )

                if shuffle_stats and 'p_value' in shuffle_stats:
                    results_df['shuffle_p_value'] = shuffle_stats['p_value']
                    results_df['shuffle_z_score'] = shuffle_stats['z_score']
                    results_df['shuffle_effect_size'] = shuffle_stats['effect_size']
                    results_df['shuffle_significant'] = shuffle_stats.get('is_significant', 
                                                                         shuffle_stats['p_value'] < 0.05)

                self._log(f"Shuffle test: p={shuffle_stats['p_value']:.4f}, "
                         f"z={shuffle_stats['z_score']:.2f}, d={shuffle_stats['effect_size']:.2f}")

        # Save results - always save (use default output_dir if not specified)
        save_output_dir = output_dir if output_dir is not None else self.output_dir

        # Build source status summary for JSON export
        source_status_summary = {
            'query': str(query),
            'source_dataset': source_dataset,
            'total_source_neurons': len(source_bodyids),
            'status_breakdown': source_status_counts,
            'skipped_sources': {
                'none': skipped_sources['none'],
                'total': len(skipped_sources['none'])
            },
            'warned_sources': {
                'rare_or_uni': warned_sources['rare_or_uni'],
                'total': len(warned_sources['rare_or_uni'])
            },
            'included_for_comparison': len(source_bodyids) - len(skipped_sources['none']),
            'status_hierarchy': {
                'none': 'No connections (0 partners) - SKIPPED entirely',
                'rare_or_uni': 'Rare (<5 partners) or unidirectional - INCLUDED with WARNING',
                'incomplete': 'Fewer than top_k partners - included with Warning',
                'incomplete_expansion': 'Fewer than top_m unique types - included with Warning',
                'complete': 'Full profile meeting all criteria'
            }
        }

        # True type-level results: pooled all-adjacency type profiles scored
        # against every typed target type (cheap; ~1-2 s/source).
        type_level_df = None
        try:
            if source_pooled_profiles and target_pooled_profiles:
                type_level_df = self._compute_type_level_results(
                    sorted(source_pooled_profiles.keys()), target_dataset,
                    similarity_metric=similarity_metric,
                    type_mapper=type_mapper,
                    source_pooled=source_pooled_profiles,
                    target_pooled=target_pooled_profiles,
                    show_progress=show_progress)
                self._log(f"Type-level results computed for "
                          f"{type_level_df['source_type'].nunique() if not type_level_df.empty else 0} source type(s)")
        except Exception as e:
            self._log(f"Type-level results skipped: {e}")

        self._progress(6, 6, "Saving results")
        self._save_homolog_results_internal(
            results_df=results_df,
            query=query,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            output_dir=save_output_dir,
            saveas=saveas,
            direction='both',
            include_partner_details=include_partner_details,
            top_n_details=top_n_details,
            params={
                'query': query,
                'source_dataset': source_dataset,
                'target_dataset': target_dataset,
                'top_n': top_n,
                'min_shared_partners': min_shared_partners,
                'min_weight': min_weight,
                'method': 'find_homologs_fast',
                'source_bodyids_count': len(source_bodyids),
                'profile_method': '1-hop/2-hop hybrid via ConnectivityProfiler',
                **(extra_readme_params or {}),
            },
            shuffle_stats=shuffle_stats,
            visualize_skeleton=visualize_skeleton,
            visualize_top_n=visualize_top_n,
            similarity_metric=self.similarity_metric,
            type_level_df=type_level_df,
            top_n=top_n,
            intra_type_df=intra_type_df,
            source_status_summary=source_status_summary
        )

        return results_df

    def find_homologs_multi(
        self,
        source: Optional[Union[str, int, List[Union[str, int]]]] = None,
        source_dataset: Optional[str] = None,
        target_dataset: Optional[str] = None,
        top_n: Optional[int] = None,
        min_shared_partners: Optional[int] = None,
        vector_prune_fraction: Optional[float] = None,
        prefilter_rank_metric: Optional[str] = None,
        min_weight: int = 3,
        show_progress: bool = True,
        output_dir: Optional[str] = None,
        saveas: Optional[str] = None,
        include_partner_details: bool = True,
        top_n_details: int = 10,
        run_shuffle_test: bool = False,
        n_shuffles: int = 100,
        shuffle_seed: Optional[int] = None,
        visualize_skeleton: Optional[bool] = None,
        visualize_top_n: Optional[int] = None,
        similarity_metric: Optional[str] = None,
        use_fast: bool = True,
        morph_qualify: bool = False,
        morph_null_k: int = 200,
        morph_bar_offset: float = 0.0,
        morph_mode: str = 'null',
        morph_level: int = 95,
    ) -> pd.DataFrame:
        """Run homolog discovery for several types / higher-category labels and
        aggregate all results into one combined output folder grouped by query
        type.

        This is the multi-query entry point used by the Homolog Finding tab.
        Each input item is expanded into its real types (coarse taxonomy labels
        expand into one search per real type), every resolved type is searched
        with the same logic as ``find_homologs_fast``, and the results are
        written to a single output folder::

            <output_dir>/<combined_name>/
                README.txt
                results/
                    bodyid_results.csv   # all rows, grouped per query_type
                    type_summary.csv     # per (query_type, source_type, target_type)
                    homolog_results.csv  # legacy, sorted by metric
                by_type/<query_type>/    # full per-type output incl. visualization/

        A single resolved query type skips the by_type/ nesting: the full
        per-type output (results/, profiles/, overlaps/, visualization/) is
        written directly into <combined_name>/.

        Morph qualification (``morph_qualify=True``, cross- or
        intra-dataset, requires visualization): after the connectivity
        ranking, the visualized top-N candidates are scored against the
        transformed query with the production vector_v2 scorer (no NBLAST
        cross-dataset) and gated by a per-query null bar — the p95 of
        ``morph_null_k`` seeded random target neurons plus
        ``morph_bar_offset``. Candidates below the bar are excluded from
        the rendered scenes; the results tables keep every row and gain
        ``morph_v2`` / ``morph_null_p95`` / ``morph_z`` /
        ``morph_qualified`` on the scored rows, plus a
        ``results/morph_qualification.json`` provenance summary. With the
        option off, output is byte-identical to the classic behavior.

        Args: same as ``find_homologs_fast`` except ``source`` may be a list.

        Returns: the combined bodyId-level results DataFrame (with a
        ``query_type`` column), or an empty DataFrame when nothing resolves.
        """
        query = source if source is not None else self.source
        source_dataset = source_dataset if source_dataset is not None else self.source_dataset
        target_dataset = target_dataset if target_dataset is not None else self.target_dataset
        output_dir = output_dir if output_dir is not None else self.output_dir
        saveas = saveas if saveas is not None else self.saveas
        visualize_skeleton = visualize_skeleton if visualize_skeleton is not None else self.visualize_skeleton
        visualize_top_n = visualize_top_n if visualize_top_n is not None else self.visualize_top_n
        similarity_metric = similarity_metric if similarity_metric is not None else self.similarity_metric
        top_n = top_n if top_n is not None else self.top_n
        min_shared_partners = min_shared_partners if min_shared_partners is not None else self.min_shared_partners
        vector_prune_fraction = vector_prune_fraction if vector_prune_fraction is not None else self.vector_prune_fraction

        if source_dataset is None:
            raise ValueError("No source_dataset specified. Set 'source_dataset' parameter or self.source_dataset.")
        if target_dataset is None:
            raise ValueError("No target_dataset specified. Set 'target_dataset' parameter or self.target_dataset.")

        # Normalize to a list of queries so a single coarse label also expands.
        queries = query if isinstance(query, (list, tuple)) else [query]
        if not queries:
            raise ValueError("No source neuron(s) specified. Set 'source' parameter or self.source.")

        self._log(f"Multi-type homolog search: {len(queries)} query item(s) from {source_dataset} → {target_dataset}")

        units = self._resolve_queries_to_types(list(queries), source_dataset)
        if not units:
            self._log("ERROR: No query item could be resolved")
            return pd.DataFrame()

        # Compute the single combined folder name once.
        if saveas:
            combined_name = saveas
        else:
            safe_parts = []
            for unit in units:
                label = str(unit['type'] if unit['type'] is not None else unit['query'])
                safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")
                if safe:
                    safe_parts.append(safe)
            safe_queries = "_".join(safe_parts)[:60] or "multi"
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            combined_name = (
                f"{self.output_folder_prefix}_{dataset_abbrev(source_dataset)}"
                f"_to_{dataset_abbrev(target_dataset)}_{safe_queries}_{timestamp}"
            )

        combined_path = Path(output_dir) / combined_name
        self._log(f"📁 Output folder: {combined_path}")

        # A single resolved query type writes its full per-type output
        # (results/, profiles/, overlaps/, visualization/) directly into the
        # run folder — the by_type/<type>/ nesting only pays off when two or
        # more types share one run.
        single_type_output = len(units) == 1

        all_dfs: List[pd.DataFrame] = []
        # (idx, query_type, real_type, per_source, per_type_output_path, unit_df)
        fetched: List[Tuple[int, str, Optional[str], Union[str, int], Path, pd.DataFrame]] = []
        skipped = 0

        # ---- Phase 1: fetch homologs for EVERY resolved type (no visualization).
        # Per-type visualization is deferred until every query type has been
        # fetched, so all connection/profile pulls are batched before the
        # (potentially slow) render phase instead of interleaving
        # fetch→viz→fetch→viz per input type.
        for idx, unit in enumerate(units):
            query_type = str(unit['query'])
            real_type = unit['type']
            if not unit['bodyids']:
                skipped += 1
                self._log(f"WARNING: Could not resolve query '{query_type}' to any neurons; skipping")
                continue

            self._log(f"--- type {idx + 1}/{len(units)}: {query_type} ---")
            safe_sub = re.sub(
                r"[^A-Za-z0-9._-]+", "_",
                str(real_type if real_type is not None else query_type),
            ).strip("_")
            safe_sub = safe_sub[:60] or f"type_{idx + 1}"
            per_type_saveas = (
                combined_name if single_type_output
                else f"{combined_name}/by_type/{safe_sub}"
            )
            per_type_output_path = Path(output_dir) / per_type_saveas
            per_source = real_type if real_type is not None else query_type

            if use_fast:
                unit_df = self.find_homologs_fast(
                    source=per_source,
                    source_dataset=source_dataset,
                    target_dataset=target_dataset,
                    top_n=top_n,
                    min_shared_partners=min_shared_partners,
                    vector_prune_fraction=vector_prune_fraction,
                    prefilter_rank_metric=prefilter_rank_metric,
                    min_weight=min_weight,
                    show_progress=show_progress,
                    output_dir=output_dir,
                    saveas=per_type_saveas,
                    include_partner_details=include_partner_details,
                    top_n_details=top_n_details,
                    run_shuffle_test=run_shuffle_test,
                    n_shuffles=n_shuffles,
                    shuffle_seed=shuffle_seed,
                    visualize_skeleton=False,  # deferred to phase 2
                    visualize_top_n=visualize_top_n,
                    similarity_metric=similarity_metric,
                    extra_readme_params=self._morph_readme_params(
                        morph_qualify, morph_mode, morph_level,
                        morph_bar_offset, morph_null_k),
                )
            else:
                unit_df = self.find_homologs(
                    source=per_source,
                    source_dataset=source_dataset,
                    target_dataset=target_dataset,
                    top_n=top_n,
                    metric=similarity_metric,
                    direction='both',
                    min_score=0.0,
                    show_progress=show_progress,
                    output_dir=output_dir,
                    saveas=per_type_saveas,
                    include_partner_details=include_partner_details,
                    top_n_details=top_n_details,
                    run_shuffle_test=run_shuffle_test,
                    n_shuffles=n_shuffles,
                    shuffle_seed=shuffle_seed,
                    visualize_skeleton=False,  # deferred to phase 2
                    visualize_top_n=visualize_top_n,
                    vector_prune_fraction=vector_prune_fraction,
                    extra_readme_params=self._morph_readme_params(
                        morph_qualify, morph_mode, morph_level,
                        morph_bar_offset, morph_null_k),
                )
            if unit_df is not None and not unit_df.empty:
                fetched.append((idx, query_type, real_type, per_source, per_type_output_path, unit_df))
                tagged = unit_df.copy()
                tagged['query_type'] = query_type
                all_dfs.append(tagged)

        if skipped == len(units):
            self._log("ERROR: None of the queries could be resolved to neurons")
            return pd.DataFrame()

        if not all_dfs:
            self._log("No homolog results produced for any query type")
            return pd.DataFrame()

        # ---- Morph qualification (optional). Pool the would-be visualized
        # pairs of EVERY resolved type into ONE scoring call so the target
        # transforms, vectorizations, and the shared null sample are
        # computed once per run (plan-morph-qualification-find-homolog.md).
        # With the option off this block is skipped entirely — output stays
        # byte-identical.
        morph_state = None
        if morph_qualify and visualize_skeleton and fetched:
            # Works cross-dataset (bridging transform) and same-dataset
            # (identity chain); target-scope refusals surface through the
            # exception path below as a logged "continuing without it".
            try:
                morph_state = self._run_morph_qualification(
                fetched, source_dataset, target_dataset,
                visualize_top_n=visualize_top_n,
                null_k=morph_null_k,
                bar_offset=morph_bar_offset,
                mode=morph_mode, level=morph_level)
            except Exception as exc:
                self._log(f"Morph qualification failed "
                          f"(continuing without it): {exc}")
            if morph_state is not None:
                self._write_morph_qualification_summary(
                    combined_path / 'results' / 'morph_qualification.json',
                    morph_state, source_dataset, target_dataset)

        # ---- Phase 2: visualizations per resolved type, after ALL fetching.
        # The per-type save (phase 1) already wrote results/type_summary.csv,
        # so the renderer reuses it to stay identical to the inline path.
        if visualize_skeleton and fetched:
            self._log(f"Fetch complete; generating visualizations for {len(fetched)} resolved type(s)...")
            for (idx, query_type, real_type, per_source, per_type_output_path, unit_df) in fetched:
                self._log(f"--- visualize type {idx + 1}: {query_type} ---")
                viz_dir = per_type_output_path / 'visualization'
                type_summary = None
                summary_path = per_type_output_path / 'results' / 'type_summary.csv'
                if summary_path.exists():
                    try:
                        type_summary = pd.read_csv(summary_path)
                    except Exception:
                        type_summary = None
                type_level_df = None
                tl_path = per_type_output_path / 'results' / 'type_level_results.csv'
                if tl_path.exists():
                    try:
                        type_level_df = pd.read_csv(tl_path)
                    except Exception:
                        type_level_df = None
                rendered_types = self._visualize_homolog_candidates(
                    results_df=unit_df,
                    query=per_source,
                    source_dataset=source_dataset,
                    target_dataset=target_dataset,
                    visualization_dir=viz_dir,
                    top_n=visualize_top_n,
                    files_saved=[],
                    type_summary=type_summary,
                    type_level_df=type_level_df,
                    morph_qualification=morph_state,
                ) or []
                # The phase-1 save predates the scenes; mirror the inline
                # path and fold the rendered-type flags into the CSV so
                # `visualized` / `visualization_rank` exist in multi runs.
                if type_summary is not None and rendered_types:
                    ts_col = ('target_type'
                              if 'target_type' in type_summary.columns else
                              'target'
                              if 'target' in type_summary.columns else None)
                    if ts_col:
                        rendered_text = [str(t) for t in rendered_types]
                        rank_map = {t: i + 1 for i, t
                                    in enumerate(rendered_text)}
                        type_summary['visualized'] = (
                            type_summary[ts_col].astype(str)
                            .isin(rendered_text))
                        type_summary['visualization_rank'] = (
                            type_summary[ts_col].astype(str).map(rank_map))
                        _drop_rank_cols(type_summary).to_csv(
                            summary_path, index=False)
                        self._log("Updated: results/type_summary.csv "
                                  "(visualized flags)")

        combined_df = pd.concat(all_dfs, ignore_index=True)

        if morph_state is not None:
            combined_df = self._merge_morph_columns(
                combined_df, morph_state)

        # Sort grouped by query_type, then the selected similarity metric.
        sort_metric = similarity_metric if similarity_metric in combined_df.columns else 'jaccard'
        if sort_metric not in combined_df.columns:
            for fallback in ['jaccard', 'rank_union', 'cosine', 'rank_corr']:
                if fallback in combined_df.columns:
                    sort_metric = fallback
                    break
        combined_df = combined_df.sort_values(
            ['query_type', sort_metric], ascending=[True, False], na_position='last'
        )

        if single_type_output:
            # The per-type save already wrote everything to the run folder
            # root; writing the combined duplicates on top would only
            # overwrite identical data with an extra query_type column.
            self._log(
                "Single query type: results written directly to the run "
                "folder (no by_type/ nesting)"
            )
            if morph_state is not None:
                self._rewrite_bodyid_csv_with_morph(
                    combined_path / 'results' / 'bodyid_results.csv',
                    morph_state)
        else:
            self._save_multi_results(
                combined_df=combined_df,
                combined_path=combined_path,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                # A coarse label expands into one unit per real type, all
                # sharing the same 'query' label; dedupe so README/
                # type_summary list each query once.
                queries=list(dict.fromkeys(str(u['query']) for u in units)),
                similarity_metric=similarity_metric,
            )

        return combined_df

    def _import_morph_cross_dataset(self):
        try:
            from comparison import morph_cross_dataset
        except ImportError:
            import morph_cross_dataset
        return morph_cross_dataset

    @staticmethod
    def _morph_readme_params(morph_qualify, morph_mode, morph_level,
                             morph_bar_offset, morph_null_k):
        """Morph-qualification parameters for the per-type README record."""
        return {
            'morph_qualify': bool(morph_qualify),
            'morph_mode': morph_mode,
            'morph_level': morph_level,
            'morph_bar_offset': morph_bar_offset,
            'morph_null_k': morph_null_k,
        }

    def _run_morph_qualification(self, fetched, source_dataset,
                                 target_dataset, visualize_top_n,
                                 null_k, bar_offset,
                                 mode='null', level=95):
        """Pooled morph-qualification pass over the visualized top-N.

        Scores exactly the pairs the visualization would render (same
        selection logic) plus one shared seeded null sample per run;
        returns the MorphQualification used for scene exclusion and CSV
        annotation, or None when nothing usable was scored.
        """
        mcd = self._import_morph_cross_dataset()
        pairs = []
        source_types = {}
        # Same-dataset scene rule: rows whose target is a query neuron are
        # rendered as the query layer, never as candidates — the pair
        # selection must exclude them exactly like the scene does.
        same_dataset = str(source_dataset) == str(target_dataset)
        for (_idx, _qtype, _real, _per_source, _path, unit_df) in fetched:
            exclude = None
            if (same_dataset and unit_df is not None
                    and 'source_bodyId' in getattr(unit_df, 'columns', [])):
                exclude = set()
                for value in unit_df['source_bodyId'].dropna().unique():
                    try:
                        if not pd.isna(value):
                            exclude.add(int(value))
                    except (TypeError, ValueError):
                        continue
                exclude = exclude or None
            pairs.extend(mcd.select_visualized_pairs(
                unit_df, visualize_top_n, exclude_targets=exclude))
            if mode == 'mapping_ref' and unit_df is not None \
                    and 'source_type' in getattr(unit_df, 'columns', []):
                for _idx, row in unit_df.iterrows():
                    try:
                        src = int(row['source_bodyId'])
                    except (TypeError, ValueError, KeyError):
                        continue
                    stype = row.get('source_type')
                    if src and stype and pd.notna(stype):
                        source_types.setdefault(src, str(stype))
        if not pairs:
            self._log("Morph qualification: no visualized pairs found")
            return None
        self._log(f"Morph qualification: scoring {len(pairs)} visualized "
                  f"pair(s) against a shared null sample (vector_v2, no "
                  f"NBLAST)...")
        state = mcd.qualify_visualized_pairs(
            source_dataset, target_dataset, pairs, null_k=null_k,
            bar_offset=bar_offset, mode=mode, level=level,
            source_types=source_types, log=self._log)
        for warning in state.warnings:
            self._log(f"[morph-qualify] {warning}")
        if not state.active:
            return None
        return state

    def _merge_morph_columns(self, combined_df, morph_state):
        mcd = self._import_morph_cross_dataset()
        combined_df = mcd.merge_morph_columns(combined_df, morph_state)
        self._log("Annotated bodyId results with morph-qualification "
                  "columns (visualized rows only)")
        return combined_df

    def _rewrite_bodyid_csv_with_morph(self, csv_path, morph_state):
        """Fold qualification columns into the phase-1 bodyId CSV.

        Single-type runs write their results during phase 1 — before the
        qualification pass runs — so the existing CSV is re-read and
        extended in place, preserving row order and formatting.
        """
        mcd = self._import_morph_cross_dataset()
        csv_path = Path(csv_path)
        if not csv_path.exists():
            return
        try:
            df = pd.read_csv(csv_path)
            df = mcd.merge_morph_columns(df, morph_state)
            df.to_csv(csv_path, index=False)
            self._log("Updated: results/bodyid_results.csv "
                      "(morph-qualification columns, visualized rows only)")
        except Exception as exc:
            self._log(f"Warning: could not add morph columns to "
                      f"{csv_path.name}: {exc}")

    def _write_morph_qualification_summary(self, json_path, morph_state,
                                           source_dataset, target_dataset):
        """Persist the qualification provenance (bars, pairs, exclusions)."""
        mcd = self._import_morph_cross_dataset()
        try:
            payload = {
                'source_dataset': source_dataset,
                'target_dataset': target_dataset,
                'null_k': morph_state.null_k,
                'bar_offset': morph_state.bar_offset,
                'mode': morph_state.mode,
                'level': morph_state.level,
                'bar_definition': (
                    f'morph_v2 >= null_p{morph_state.level} + bar_offset'
                    if morph_state.mode == 'null' else
                    f'mapping_ref floors v3: native pool floor (>=2 refs) '
                    f'binding, else Track-A B_b - Δ, else '
                    f'null_p{morph_state.level} + bar_offset'),
                'scorer': 'vector_v2 (production block-weighted whitened '
                          'cosine; no NBLAST cross-dataset)',
                'per_source': {
                    str(bid): {**stats, 'bar': morph_state.bar(bid)}
                    for bid, stats in morph_state.null_stats.items()},
                'scored_pairs': [
                    {'source_bodyId': src, 'target_bodyId': tgt,
                     'morph_v2': morph_state.scores[(src, tgt)],
                     'qualified': morph_state.is_qualified(src, tgt)}
                    for src, tgt in sorted(morph_state.scores)],
                'warnings': list(morph_state.warnings),
            }
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
            self._log("Saved: results/morph_qualification.json")
        except Exception as exc:
            self._log(f"Warning: could not write morph-qualification "
                      f"summary: {exc}")

    def _attach_instance_columns(self, df: pd.DataFrame,
                                 source_dataset: str,
                                 target_dataset: str) -> pd.DataFrame:
        """Add ``source_instance`` / ``target_instance`` next to the types.

        Instance names follow the tree-legend export convention and resolve
        from the row's dataset (the ``target_dataset`` column wins on
        cross-dataset runs). Missing metadata leaves the column empty
        instead of failing the export.
        """
        if df is None or df.empty:
            return df
        if ('source_type' in df.columns
                and 'source_instance' not in df.columns):
            if 'source_bodyId' in df.columns:
                instances = df['source_bodyId'].map(
                    lambda v: _body_id_instance_name(source_dataset, v))
            else:
                instances = pd.Series([''] * len(df), index=df.index)
            df.insert(df.columns.get_loc('source_type') + 1,
                      'source_instance', instances)
        if ('target_type' in df.columns
                and 'target_instance' not in df.columns):
            if 'target_bodyId' in df.columns:
                def _target_instance(row):
                    ds = str(row.get('target_dataset') or ''
                             ).strip() or str(target_dataset or ''
                                              ).strip() or source_dataset
                    return _body_id_instance_name(ds, row['target_bodyId'])
                instances = df.apply(_target_instance, axis=1)
            else:
                instances = pd.Series([''] * len(df), index=df.index)
            df.insert(df.columns.get_loc('target_type') + 1,
                      'target_instance', instances)
        return df

    def _save_multi_results(
        self,
        combined_df: pd.DataFrame,
        combined_path: Path,
        source_dataset: str,
        target_dataset: str,
        queries: List[str],
        similarity_metric: str,
    ) -> None:
        """Write the combined, per-query-type aggregated outputs into one folder."""
        results_dir = combined_path / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)

        numeric_cols = [
            'rank_corr', 'rank_union', 'rank_union_raw', 'jaccard', 'cosine',
            'adjacency_score', 'shared_type_count', 'union_type_count',
        ]
        rounded = combined_df.copy()
        for col in numeric_cols:
            if col in rounded.columns:
                rounded[col] = rounded[col].round(4)
        rounded = self._attach_instance_columns(
            rounded, source_dataset, target_dataset)

        # 1) homolog_results.csv (legacy, sorted by the selected metric).
        sort_metric = similarity_metric if similarity_metric in rounded.columns else 'jaccard'
        if sort_metric not in rounded.columns:
            for fallback in ['jaccard', 'rank_union', 'cosine', 'rank_corr']:
                if fallback in rounded.columns:
                    sort_metric = fallback
                    break
        if sort_metric in rounded.columns:
            results_by_score = rounded.sort_values(sort_metric, ascending=False, na_position='last')
        else:
            results_by_score = rounded  # no metric column available; keep frame order
        _drop_rank_cols(results_by_score).to_csv(results_dir / 'homolog_results.csv', index=False)

        # 2) bodyid_results.csv (grouped per query_type).
        bodyid_cols = [
            'query_type', 'source_bodyId', 'source_type', 'source_instance',
            'target_bodyId', 'target_type', 'target_instance', 'rank_union',
            'jaccard', 'cosine', 'adjacency_score', 'shared_type_count',
            'union_type_count', 'is_same_type', 'is_same_dataset',
            'source_status', 'target_status', 'weak_source', 'weak_target',
            'source_partner_count', 'target_partner_count',
            # Morph qualification (present only when the option ran):
            'morph_v2', 'morph_null_p95', 'morph_z', 'morph_qualified',
        ]
        available = [c for c in bodyid_cols if c in rounded.columns]
        bodyid_df = rounded[available].copy()
        _drop_rank_cols(bodyid_df).to_csv(results_dir / 'bodyid_results.csv', index=False)

        # 3) type_summary.csv aggregated per (query_type, source_type, target_type).
        valid = (
            rounded[rounded['rank_corr'].notna()].copy()
            if 'rank_corr' in rounded.columns else rounded.copy()
        )
        type_summary = pd.DataFrame()
        if not valid.empty and {'query_type', 'source_type'} <= set(valid.columns):
            target_col = 'target_type' if 'target_type' in valid.columns else 'target'
            agg_dict = {}
            if 'rank_corr' in valid.columns:
                agg_dict['rank_corr'] = ['mean', 'count']
            for m in ['rank_union', 'jaccard', 'cosine', 'adjacency_score',
                      'shared_type_count', 'union_type_count']:
                if m in valid.columns:
                    agg_dict[m] = 'mean'
            group_cols = ['query_type', 'source_type', target_col]
            type_summary = valid.groupby(group_cols).agg(agg_dict).round(4)
            type_summary.columns = [
                '_'.join(col).strip() if isinstance(col, tuple) else col
                for col in type_summary.columns.values
            ]
            type_summary = type_summary.reset_index()
            rename_map = {
                'rank_corr_mean': 'avg_rank_corr',
                'rank_corr_count': 'n_bodyid_comparisons',
                'jaccard_mean': 'avg_jaccard',
                'rank_union_mean': 'avg_rank_union',
                'cosine_mean': 'avg_cosine',
                'adjacency_score_mean': 'avg_adjacency_score',
                'shared_type_count_mean': 'avg_shared_type_count',
                'union_type_count_mean': 'avg_union_type_count',
            }
            type_summary = type_summary.rename(columns=rename_map)
            sort_by = f'avg_{sort_metric}' if f'avg_{sort_metric}' in type_summary.columns else 'avg_rank_corr'
            type_summary = type_summary.sort_values(sort_by, ascending=False, na_position='last')
            type_summary.insert(0, 'source_dataset', source_dataset)
            type_summary.insert(0, 'query', ','.join(queries))
        _drop_rank_cols(type_summary).to_csv(results_dir / 'type_summary.csv', index=False)

        # 3b) type_level_results.csv — pooled all-adjacency type-level
        # ranking, collected from the per-type output folders.
        tl_frames = []
        for per_dir in sorted(combined_path.glob('by_type/*')):
            tl_path = per_dir / 'results' / 'type_level_results.csv'
            if tl_path.exists():
                try:
                    tl_df = pd.read_csv(tl_path)
                    tl_df.insert(0, 'query_type', per_dir.name)
                    tl_frames.append(tl_df)
                except Exception as e:
                    self._log(f"Warning: could not read {tl_path}: {e}")
        if tl_frames:
            tl_combined = pd.concat(tl_frames, ignore_index=True)
            tl_combined.insert(0, 'source_dataset', source_dataset)
            _drop_rank_cols(tl_combined).to_csv(results_dir / 'type_level_results.csv', index=False)
        self._log("Saved: results/type_level_results.csv (pooled type-level ranking)"
                  if tl_frames else "No type-level results collected from per-type folders")

        # 4) README.txt summarizing all query types.
        readme = combined_path / 'README.txt'
        with open(readme, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("  HOMOLOG FINDING RESULTS (MULTI-QUERY)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Query Types: {', '.join(queries)}\n")
            f.write(f"Source Dataset: {source_dataset}\n")
            f.write(f"Target Dataset: {target_dataset}\n")
            f.write(f"Total matches: {len(combined_df)}\n\n")
            f.write("  OUTPUT STRUCTURE\n")
            f.write("  results/\n")
            f.write("    ├── bodyid_results.csv  Combined results grouped per query_type\n")
            f.write("    ├── type_summary.csv    Aggregated per (query_type, source_type, target_type)\n")
            f.write("    ├── type_level_results.csv  Pooled all-adjacency type-level ranking\n")
            f.write("    └── homolog_results.csv Legacy format sorted by metric\n")
            f.write("  by_type/<query_type>/     Full per-type output (results, visualization)\n")
        # 5) Auto-type-mapping provenance for the COMBINED folder (the
        # per-type sub-runs each write their own auto_type_mapping.json
        # under by_type/<query_type>/; this root-level file records the
        # run-level mapper state so the combined output is self-describing).
        try:
            mapping_meta = auto_mapping_result_metadata(
                requested=self.use_auto_type_mapping,
                type_mapper=self._type_mapper,
            )
            mapping_meta.update({
                'queries': list(queries),
                'source_dataset': source_dataset,
                'target_dataset': target_dataset,
                'per_type_metadata': 'by_type/<query_type>/auto_type_mapping.json',
            })
            (combined_path / 'auto_type_mapping.json').write_text(
                json.dumps(mapping_meta, indent=2, default=str), encoding='utf-8')
        except Exception as exc:
            self._log(f"Warning: could not write combined mapping metadata: {exc}")

        self._log(f"Saved combined multi-query results to {combined_path}")

    def find_homologs_intra_dataset(
        self,
        query: Union[str, int],
        dataset: str,
        search_untyped: bool = False,
        top_n: int = 20,
        min_weight: int = 3,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Find homologs within the same dataset.
        
        For intra-dataset search, useful for:
        - Finding untyped neurons similar to a typed query
        - Finding sub-populations within a type
        - Finding neurons with similar connectivity profiles
        
        Args:
            query: Neuron type (str) or bodyId (int)
            dataset: Dataset to search
            search_untyped: If True, search untyped neurons; else search typed
            top_n: Number of top results
            min_weight: Minimum synapse weight
            show_progress: Show progress bar
        
        Returns:
            DataFrame with matches
        """
        # Use find_homologs with same source/target dataset
        return self.find_homologs(
            query=query,
            source_dataset=dataset,
            target_dataset=dataset,
            top_n=top_n,
            show_progress=show_progress
        )

    def _save_homolog_results_internal(
        self,
        results_df: pd.DataFrame,
        query: Union[str, int],
        source_dataset: str,
        target_dataset: str,
        output_dir: str,
        saveas: Optional[str],
        direction: str,
        include_partner_details: bool,
        top_n_details: int,
        params: Dict[str, Any],
        shuffle_stats: Optional[Dict[str, Any]] = None,
        visualize_skeleton: bool = False,
        visualize_top_n: int = 5,
        intra_type_df: Optional[pd.DataFrame] = None,
        type_level_df: Optional[pd.DataFrame] = None,
        top_n: int = 100,
        similarity_metric: str = 'jaccard',
        source_status_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Internal method to save homolog finding results with organized folder structure.
        
        Called by find_homologs() and find_homologs_fast() when output_dir is provided.
        
        Output folder structure:
            {output_dir}/{saveas or timestamp}/
            ├── README.txt                    # Parameters, summary, and column descriptions
            ├── results/                      # Main results
            │   ├── homolog_results.csv       # Full results with all columns (sorted by metric)
            │   ├── type_summary.csv          # Type-mean from bodyId level
            │   ├── type_level_results.csv   # Pooled type-level ranking
            │   ├── bodyid_results.csv        # BodyId-level results (sorted by source, metric)
            │   ├── intra_type_results.csv    # Intra-type comparisons (if type query)
            │   ├── shuffle_test.json         # Shuffle test statistics (if run)
            │   └── source_status_summary.json # Source neuron connectivity status breakdown
            ├── profiles/                     # Connectivity profiles
            │   ├── query/                    # Query neuron profiles
            │   │   ├── {query}.csv           # Type-level profile
            │   │   └── source_bodyids.csv    # Source bodyId summary (if bodyId-level)
            │   └── matches/                  # Top match profiles  
            │       ├── top_target_bodyids.csv # Target bodyId summary (if bodyId-level)
            │       ├── {match_type1}.csv     # Type-level profile for top match types
            │       └── {match_type2}.csv
            ├── overlaps/                     # Partner overlap details
            │   ├── {query}_vs_{match1}.csv
            │   └── {query}_vs_{match2}.csv
            └── visualization/                # 3D skeleton plots (if enabled)
                ├── {type}_{bodyId}.html      # Interactive HTML per match
                └── ...
        
        Args:
            results_df: Results DataFrame
            query: Original query neuron
            source_dataset: Source dataset
            target_dataset: Target dataset
            output_dir: Base output directory
            saveas: Custom folder name (None = auto-generate with timestamp)
            direction: 'upstream', 'downstream', or 'both' for overlap details
            include_partner_details: Include per-match partner overlap CSVs
            top_n_details: Number of top matches to include detailed profiles for
            params: Dict of input parameters to save
            shuffle_stats: Optional shuffle test statistics dict to save
            visualize_skeleton: If True, generate 3D skeleton visualizations
            visualize_top_n: Number of top matches to visualize
            intra_type_df: Optional DataFrame with intra-type comparison results
            similarity_metric: The metric to use for sorting results 
                             ('rank_union', 'rank_corr', 'jaccard', 'cosine')
            source_status_summary: Optional dict with source neuron connectivity status breakdown
        
        Returns:
            Dict with output_dir, files_saved, query_profile, match_profiles
        """
        import os
        import json
        from pathlib import Path
        from datetime import datetime
        
        # Check for bodyId-level columns
        has_bodyid_cols = 'source_bodyId' in results_df.columns if not results_df.empty else False
        
        # Generate folder name with timestamp if not provided.
        # Treat empty strings like None: the UI sends saveas="" when the
        # field is blank, which must still auto-generate the per-run folder
        # (otherwise results would be dumped straight into output_dir).
        if not saveas:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_query = str(query).replace('/', '_').replace(':', '_').replace('*', '_')
            folder_name = (
                f"{self.output_folder_prefix}_{dataset_abbrev(source_dataset)}"
                f"_to_{dataset_abbrev(target_dataset)}"
                f"_{safe_query}_{timestamp}"
            )
        else:
            folder_name = saveas
        
        # Create organized folder structure
        output_path = Path(output_dir) / folder_name
        results_dir = output_path / 'results'
        profiles_dir = output_path / 'profiles'
        query_profiles_dir = profiles_dir / 'query'
        match_profiles_dir = profiles_dir / 'matches'
        overlaps_dir = output_path / 'overlaps'
        visualization_dir = output_path / 'visualization'
        
        # Create all directories
        dirs_to_create = [output_path, results_dir, query_profiles_dir, match_profiles_dir, overlaps_dir]
        if visualize_skeleton:
            dirs_to_create.append(visualization_dir)
        self._log(f"📁 Output folder: {output_path}")
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        files_saved = []

        # Auto-type-mapping provenance (plan Workstream E): requested vs
        # active mapper, source table, version, load error — so a saved
        # run states whether it was produced under a healthy v1.0 mapper.
        try:
            # Per-run semantics: this saved run owns the resolutions counted
            # since the last save; `raw_fallback_used` is MEASURED (counted
            # unmapped candidate-expansion resolutions), not a constant.
            status_snapshot = dict(getattr(self, '_mapping_status_counts', {}) or {})
            self._mapping_status_counts = {}
            mapping_meta = auto_mapping_result_metadata(
                requested=self.use_auto_type_mapping,
                type_mapper=self._type_mapper,
                resolution_counts=status_snapshot,
                raw_fallback_used=bool(status_snapshot.get('unmapped')),
            )
            mapping_meta.update({
                'query': query,
                'source_dataset': source_dataset,
                'target_dataset': target_dataset,
            })
            mapping_path = output_path / 'auto_type_mapping.json'
            mapping_path.write_text(
                json.dumps(mapping_meta, indent=2, default=str),
                encoding='utf-8')
            files_saved.append(str(mapping_path))
        except Exception:
            pass

        self._log(f"Saving homolog results to {output_path}")
        
        # 1. Save README.txt with parameters, summary, and shuffle test results
        readme_file = output_path / 'README.txt'
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("  HOMOLOG FINDING RESULTS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Query: {query}\n")
            f.write(f"Source Dataset: {source_dataset}\n")
            f.write(f"Target Dataset: {target_dataset}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("  INPUT PARAMETERS\n")
            f.write("-" * 70 + "\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("  RESULTS SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Total matches: {len(results_df)}\n")
            
            # BodyId-level stats
            if has_bodyid_cols:
                unique_sources = results_df['source_bodyId'].nunique() if 'source_bodyId' in results_df.columns else 0
                unique_targets = results_df['target_bodyId'].nunique() if 'target_bodyId' in results_df.columns else 0
                f.write(f"  Unique source bodyIds: {unique_sources}\n")
                f.write(f"  Unique target bodyIds: {unique_targets}\n")
            
            if not results_df.empty and 'rank_union' in results_df.columns:
                f.write(f"  Best rank_union: {results_df['rank_union'].max():.4f}\n")
                f.write(f"  Mean rank_union: {results_df['rank_union'].mean():.4f}\n")
                f.write(f"  Median rank_union: {results_df['rank_union'].median():.4f}\n")
            if not results_df.empty and 'jaccard' in results_df.columns:
                f.write(f"  Best jaccard: {results_df['jaccard'].max():.4f}\n")
                f.write(f"  Mean jaccard: {results_df['jaccard'].mean():.4f}\n")
            
            # Shuffle test results section
            if shuffle_stats:
                f.write("\n" + "-" * 70 + "\n")
                f.write("  SHUFFLE TEST RESULTS (Statistical Validation)\n")
                f.write("-" * 70 + "\n")
                f.write(f"  Number of shuffles: {shuffle_stats.get('n_shuffles', 'N/A')}\n")
                f.write(f"  Observed score: {shuffle_stats.get('observed_score', 0):.4f}\n")
                f.write(f"  Mean shuffled score: {shuffle_stats.get('mean_shuffled', 0):.4f}\n")
                f.write(f"  Std shuffled score: {shuffle_stats.get('std_shuffled', 0):.4f}\n")
                f.write(f"  P-value: {shuffle_stats.get('p_value', 1.0):.4f}\n")
                f.write(f"  Z-score: {shuffle_stats.get('z_score', 0):.2f}\n")
                f.write(f"  Effect size (Cohen's d): {shuffle_stats.get('effect_size', 0):.2f}\n")
                f.write(f"  Interpretation: {shuffle_stats.get('interpretation', 'N/A')}\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("  FOLDER STRUCTURE\n")
            f.write("-" * 70 + "\n")
            f.write("  results/\n")
            f.write("    ├── bodyid_results.csv     BodyId-level comparisons (sorted by source_bodyId, rank_union)\n")
            f.write("    ├── type_summary.csv       Type-level aggregated summary\n")
            f.write("    ├── type_level_results.csv Pooled type-level homolog ranking (top N per source type)\n")
            f.write("    ├── homolog_results.csv    Full results sorted by the run's metric (legacy format)\n")
            f.write("    ├── source_status_summary.json  Source neuron connectivity status breakdown\n")
            if shuffle_stats:
                f.write("    └── shuffle_test.json      Shuffle test statistics\n")
            f.write("  profiles/\n")
            f.write("    ├── query/                 Query neuron connectivity profile\n")
            f.write("    └── matches/               Top match connectivity profiles\n")
            f.write("  overlaps/\n")
            f.write("    └── *_vs_*.csv             Partner overlap details per match\n")
            if visualize_skeleton:
                f.write("  visualization/\n")
                f.write("    └── *.html, *.png          3D skeleton plots of top matches\n")
            
            # Add interpretation guide
            f.write("\n" + "-" * 70 + "\n")
            f.write("  INTERPRETATION GUIDE\n")
            f.write("-" * 70 + "\n")
            f.write("  rank_corr: Raw Spearman correlation on SHARED partners (-1 to 1)\n")
            f.write("            (kept internal; not exported)\n")
            f.write("  rank_union: Raw Spearman correlation on the partner union (-1 to 1)\n")
            f.write("  jaccard: Jaccard similarity of partner sets (0-1, higher=better)\n")
            f.write("  cosine: Cosine similarity of weight vectors (0-1, higher=better)\n")
            f.write("  adjacency_score: Number of shared partners found during candidate discovery\n")
            f.write("                   (from 2-hop adjacency expansion, NOT from profile comparison)\n")
            f.write("  shared_type_count: Number of types/bodyIds used for rank_corr calculation\n")
            f.write("                     (from actual profile comparison)\n")
            f.write("  union_type_count: Total unique types/bodyIds in both profiles\n")
            f.write("  target_type_members: Member count of the candidate type in the target\n")
            f.write("               dataset; very large counts (hundreds+) indicate coarse or\n")
            f.write("               hemilineage-scale annotations (e.g. Mi15) - interpret with care\n")
            if shuffle_stats:
                f.write("\n  Shuffle Test Interpretation:\n")
                f.write("    P-value < 0.05: Result is statistically significant\n")
                f.write("    Z-score > 2: Observed score is 2+ std deviations above random\n")
                f.write("    Effect size > 0.8: Large effect (meaningful difference)\n")
        
        files_saved.append('README.txt')
        self._log("Saved: README.txt")
        
        # 2. Save bodyId-level results (always saved, sorted by source_bodyId then rank_corr)
        # This is the foundational comparison data showing each source-target bodyId pair
        if not results_df.empty:
            # Instance names alongside the type columns (tree-legend export
            # convention) for every downstream bodyId-level table.
            results_df = self._attach_instance_columns(
                results_df, source_dataset, target_dataset)
            # Round numeric columns for better readability
            numeric_cols = ['rank_corr', 'rank_union', 'rank_union_raw', 'jaccard', 'cosine']
            results_rounded = results_df.copy()
            for col in numeric_cols:
                if col in results_rounded.columns:
                    results_rounded[col] = results_rounded[col].round(4)
            
            # Determine sorting columns based on available data
            # Use the user-selected similarity_metric for sorting
            sort_metric = similarity_metric if similarity_metric in results_rounded.columns else 'jaccard'
            if sort_metric not in results_rounded.columns:
                # Fallback to any available metric
                for fallback in ['jaccard', 'rank_union', 'cosine', 'rank_corr']:
                    if fallback in results_rounded.columns:
                        sort_metric = fallback
                        break
            if sort_metric not in results_rounded.columns:
                sort_metric = None  # fixture has no metric column; skip metric sorting
            
            if 'source_bodyId' in results_rounded.columns:
                # BodyId-level data: sort by source_bodyId, then similarity_metric
                bodyid_sort_cols = (['source_bodyId', sort_metric] if sort_metric
                                    else ['source_bodyId'])
                bodyid_sorted = results_rounded.sort_values(
                    bodyid_sort_cols,
                    ascending=([True, False] if sort_metric else [True]),
                    na_position='last'
                )
                
                # Save bodyId-level results with clear column order
                # Include source_status/target_status to indicate connectivity quality
                # Include weak_source/weak_target columns (legacy) to indicate neurons with <5 partners
                # Include source/target_partner_count to diagnose comparison (bodyIds for intra, types for cross)
                # shared_type_count shows how many shared types were used for rank correlation
                # union_type_count shows total unique types in union (for rank_union)
                bodyid_cols = ['source_bodyId', 'source_type', 'source_instance',
                              'target_bodyId', 'target_type', 'target_instance',
                              'rank_union',
                              'jaccard', 'cosine',
                              'adjacency_score', 'shared_type_count', 'union_type_count',
                              'is_same_type', 'is_same_dataset',
                              'source_status', 'target_status',  # Hierarchical status
                              'weak_source', 'weak_target',  # Legacy flags
                              'source_partner_count', 'target_partner_count']
                available_bodyid_cols = [c for c in bodyid_cols if c in bodyid_sorted.columns]
                bodyid_df = bodyid_sorted[available_bodyid_cols].copy()
                _drop_rank_cols(bodyid_df).to_csv(results_dir / 'bodyid_results.csv', index=False)
                files_saved.append('results/bodyid_results.csv')
                self._log(f"Saved: results/bodyid_results.csv (sorted by source_bodyId, {sort_metric})")
            else:
                # Type-level only data (source_neuron instead of source_bodyId)
                if sort_metric:
                    bodyid_sorted = results_rounded.sort_values(sort_metric, ascending=False, na_position='last')
                else:
                    bodyid_sorted = results_rounded
            
            # Also save as homolog_results.csv (legacy format, sorted by similarity_metric)
            if sort_metric:
                results_by_score = results_rounded.sort_values(sort_metric, ascending=False, na_position='last')
            else:
                results_by_score = results_rounded
            _drop_rank_cols(results_by_score).to_csv(results_dir / 'homolog_results.csv', index=False)
            files_saved.append('results/homolog_results.csv')
            self._log(f"Saved: results/homolog_results.csv (sorted by {sort_metric})")
        
        # 2b. Save intra-type comparison results if provided
        if intra_type_df is not None and not intra_type_df.empty:
            # Round numeric columns
            intra_type_rounded = intra_type_df.copy()
            for col in ['rank_corr', 'rank_union', 'rank_union_raw', 'jaccard', 'cosine']:
                if col in intra_type_rounded.columns:
                    intra_type_rounded[col] = intra_type_rounded[col].round(4)
            
            # Sort by rank_union (descending) to show most similar pairs first
            sort_col = next((c for c in ['jaccard', 'rank_union', 'rank_corr', 'cosine']
                             if c in intra_type_rounded.columns), None)
            intra_type_sorted = (intra_type_rounded.sort_values(sort_col, ascending=False)
                                 if sort_col else intra_type_rounded)
            _drop_rank_cols(intra_type_sorted).to_csv(results_dir / 'intra_type_results.csv', index=False)
            files_saved.append('results/intra_type_results.csv')
            self._log(f"Saved: results/intra_type_results.csv ({len(intra_type_sorted)} pairs)")
        
        # 3. Save shuffle test results as JSON
        if shuffle_stats:
            shuffle_file = results_dir / 'shuffle_test.json'
            with open(shuffle_file, 'w', encoding='utf-8') as f:
                # Convert to JSON-serializable format, excluding DataFrames and complex objects
                serializable_stats = {}
                exclude_keys = ['real_results', 'shuffled_results']  # These are DataFrames
                
                for k, v in shuffle_stats.items():
                    if k in exclude_keys:
                        continue  # Skip DataFrames
                    if v is None:
                        serializable_stats[k] = None
                    elif isinstance(v, pd.DataFrame):
                        continue  # Skip DataFrames
                    elif hasattr(v, 'item'):  # numpy scalar
                        serializable_stats[k] = v.item()
                    elif isinstance(v, np.ndarray):
                        serializable_stats[k] = v.tolist()
                    elif isinstance(v, list):
                        # Handle lists that might contain numpy types
                        serializable_stats[k] = [
                            x.item() if hasattr(x, 'item') else x 
                            for x in v 
                            if not isinstance(x, pd.DataFrame)
                        ]
                    elif isinstance(v, (str, int, float, bool)):
                        serializable_stats[k] = v
                    else:
                        # Try to convert to string for other types
                        try:
                            serializable_stats[k] = str(v)
                        except Exception:
                            pass
                
                json.dump(serializable_stats, f, indent=2)
            files_saved.append('results/shuffle_test.json')
            self._log("Saved: results/shuffle_test.json")
        
        # 3b. Save source status summary as JSON
        # This provides detailed breakdown of source neuron connectivity status
        if source_status_summary is not None:
            status_summary_file = results_dir / 'source_status_summary.json'
            with open(status_summary_file, 'w', encoding='utf-8') as f:
                json.dump(source_status_summary, f, indent=2)
            files_saved.append('results/source_status_summary.json')
            self._log("Saved: results/source_status_summary.json")
        
        # 4. Get and save query profile
        # For bodyId-level comparisons, save both type-level and individual bodyId profiles
        query_profile = self.get_profile(query, source_dataset)
        query_profile_data = None
        
        if query_profile is not None:
            query_profile_data = self._profile_to_dataframe(query_profile, source_dataset)
            safe_query = str(query).replace('/', '_').replace(':', '_')
            query_profile_data.to_csv(query_profiles_dir / f'{safe_query}.csv', index=False)
            files_saved.append(f'profiles/query/{safe_query}.csv')
            self._log(f"Saved: profiles/query/{safe_query}.csv (type-level)")
        
        # If bodyId-level comparison, also save source bodyId summary
        if has_bodyid_cols and not results_df.empty:
            # Create summary of source bodyIds with their status and connectivity
            source_cols = ['source_bodyId', 'source_type', 'source_instance',
                           'source_status', 'source_partner_count']
            available_source_cols = [c for c in source_cols if c in results_df.columns]
            source_summary = results_df[available_source_cols].drop_duplicates()
            source_summary = source_summary.sort_values('source_bodyId')
            source_summary.to_csv(query_profiles_dir / 'source_bodyids.csv', index=False)
            files_saved.append('profiles/query/source_bodyids.csv')
            self._log(f"Saved: profiles/query/source_bodyids.csv ({len(source_summary)} source neurons)")
        
        # 5. Get and save top match profiles + partner overlaps
        match_profiles = {}
        
        if not results_df.empty and include_partner_details:
            top_matches = results_df.head(top_n_details)
            
            # For bodyId-level comparisons, save target bodyId summary for top matches
            if has_bodyid_cols:
                target_cols = ['target_bodyId', 'target_type', 'target_instance',
                              'target_status', 'target_partner_count',
                              'rank_union', 'jaccard']
                available_target_cols = [c for c in target_cols if c in top_matches.columns]
                target_summary = top_matches[available_target_cols].drop_duplicates(subset=['target_bodyId'])
                if 'jaccard' in target_summary.columns:
                    target_summary = target_summary.sort_values('jaccard', ascending=False, na_position='last')
                target_summary.to_csv(match_profiles_dir / 'top_target_bodyids.csv', index=False)
                files_saved.append('profiles/matches/top_target_bodyids.csv')
                self._log(f"Saved: profiles/matches/top_target_bodyids.csv ({len(target_summary)} top targets)")
            
            for _, row in top_matches.iterrows():
                target_type = row.get('target_type', row.get('target', None))
                if target_type is None or pd.isna(target_type) or str(target_type) == '':
                    continue
                
                try:
                    # Get match profile (type-level)
                    match_profile = self.get_profile(target_type, target_dataset)
                    if match_profile is None:
                        continue
                    
                    # Save match profile to matches subfolder
                    match_data = self._profile_to_dataframe(match_profile, target_dataset)
                    safe_type = str(target_type).replace('/', '_').replace(':', '_')
                    match_data.to_csv(match_profiles_dir / f'{safe_type}.csv', index=False)
                    files_saved.append(f'profiles/matches/{safe_type}.csv')
                    match_profiles[target_type] = match_data
                    
                    # Get partner overlap details and save to overlaps subfolder
                    if query_profile is not None:
                        overlap_dfs = []
                        
                        for dir_name in ['upstream', 'downstream'] if direction == 'both' else [direction]:
                            overlap = ProfileComparator.get_partner_overlap_details(
                                query_profile, match_profile, dir_name
                            )
                            if not overlap.empty:
                                overlap['direction'] = dir_name
                                overlap_dfs.append(overlap)
                        
                        if overlap_dfs:
                            combined_overlap = pd.concat(overlap_dfs, ignore_index=True)
                            safe_query_name = str(query).replace('/', '_').replace(':', '_')
                            overlap_file = f'{safe_query_name}_vs_{safe_type}.csv'
                            combined_overlap.to_csv(overlaps_dir / overlap_file, index=False)
                            files_saved.append(f'overlaps/{overlap_file}')
                
                except Exception as e:
                    self._log(f"Warning: Could not save details for {target_type}: {e}")
        
        # 6. Save type-level summary (always generated by aggregating results)
        # This aggregates bodyId-level comparisons to show type-level patterns
        type_summary = pd.DataFrame()
        if not results_df.empty:
            # Determine the target column name (could be target_type or target)
            target_col = 'target_type' if 'target_type' in results_df.columns else 'target'
            
            if has_bodyid_cols and target_col in results_df.columns:
                # Filter out rows with NaN rank_corr (from weak sources with 0 partners)
                # These have no valid connectivity profile and shouldn't be included in type summary
                valid_results = results_df[results_df['rank_corr'].notna()].copy()
                # Also drop targets that never resolved to a real cell type
                # (untyped neurons / coarse class labels skipped by the type
                # lookup): they cannot form a meaningful type-level group.
                if target_col in valid_results.columns:
                    target_vals = valid_results[target_col].astype(str).str.strip()
                    valid_results = valid_results[
                        valid_results[target_col].notna()
                        & target_vals.ne('')
                        & target_vals.str.lower().ne('nan')]
                
                if valid_results.empty:
                    self._log("Warning: No valid comparisons for type summary (all sources have invalid status)")
                    # Create empty type summary with proper columns (averages only)
                    type_summary = pd.DataFrame(columns=[
                        'query', 'source_dataset', 'target_dataset', 'source_type', target_col,
                        'avg_rank_corr', 'avg_jaccard', 'avg_rank_union',
                        'avg_cosine', 'avg_adjacency_score', 'n_bodyid_comparisons',
                        'n_complete_sources', 'n_incomplete_sources'
                    ])
                else:
                    # Aggregate bodyId results to type level
                    # Group by both source_type and target_type to show pairwise type comparisons
                    if 'source_type' in valid_results.columns:
                        group_cols = ['source_type', target_col]
                    else:
                        group_cols = [target_col]
                    
                    # Aggregate with averages only for all similarity metrics
                    agg_dict = {'rank_corr': ['mean', 'count']}
                    if 'jaccard' in valid_results.columns:
                        agg_dict['jaccard'] = 'mean'
                    if 'rank_union' in valid_results.columns:
                        agg_dict['rank_union'] = 'mean'
                    if 'cosine' in valid_results.columns:
                        agg_dict['cosine'] = 'mean'
                    if 'adjacency_score' in valid_results.columns:
                        agg_dict['adjacency_score'] = 'mean'
                    if 'shared_type_count' in valid_results.columns:
                        agg_dict['shared_type_count'] = 'mean'
                    if 'union_type_count' in valid_results.columns:
                        agg_dict['union_type_count'] = 'mean'
                    
                    # Add connectivity status aggregation
                    if 'source_status' in valid_results.columns:
                        # Count sources by status type
                        agg_dict['source_status'] = lambda x: (x == 'complete').sum()
                    
                    type_summary = valid_results.groupby(group_cols).agg(agg_dict).round(4)
                    type_summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in type_summary.columns.values]
                    type_summary = type_summary.reset_index()
                    
                    # Rename columns - use avg_ prefix for all metrics (no best_/std_)
                    rename_map = {
                        'rank_corr_mean': 'avg_rank_corr',
                        'rank_corr_count': 'n_bodyid_comparisons',
                        'jaccard_mean': 'avg_jaccard',
                        'rank_union_mean': 'avg_rank_union',
                        'cosine_mean': 'avg_cosine',
                        'adjacency_score_mean': 'avg_adjacency_score',
                        'shared_type_count_mean': 'avg_shared_type_count',
                        'union_type_count_mean': 'avg_union_type_count'
                    }
                    
                    # Handle the lambda aggregation column name
                    for col in type_summary.columns:
                        if 'source_status' in col and 'lambda' in col.lower():
                            rename_map[col] = 'n_complete_sources'
                    
                    type_summary = type_summary.rename(columns=rename_map)
                    
                    # If we have source_status, also count incomplete sources
                    if 'n_complete_sources' in type_summary.columns:
                        type_summary['n_incomplete_sources'] = type_summary['n_bodyid_comparisons'] - type_summary['n_complete_sources']
                    
                    # Sort by the average of the selected similarity_metric
                    sort_by_col = f'avg_{sort_metric}' if f'avg_{sort_metric}' in type_summary.columns else 'avg_jaccard'
                    type_summary = type_summary.sort_values(
                        sort_by_col, ascending=False, na_position='last'
                    ).reset_index(drop=True)

                    type_summary.insert(0, 'query', query)
                    type_summary.insert(1, 'source_dataset', source_dataset)
                    type_summary.insert(2, 'target_dataset', target_dataset)
            elif target_col in results_df.columns:
                # Already type-level, just format
                # Filter out NaN rows here too
                valid_results = results_df[results_df['rank_corr'].notna()].copy() if 'rank_corr' in results_df.columns else results_df.copy()
                pairwise_cols = [target_col, 'rank_corr', 'rank_union', 'jaccard', 
                                'adjacency_score', 'is_same_type']
                available_cols = [c for c in pairwise_cols if c in valid_results.columns]
                type_summary = valid_results[available_cols].drop_duplicates().copy()
                type_summary = type_summary.sort_values(
                    'jaccard', ascending=False, na_position='last'
                ).reset_index(drop=True)
                type_summary.insert(0, 'query', query)
                type_summary.insert(1, 'source_dataset', source_dataset)
                type_summary.insert(2, 'target_dataset', target_dataset)
            else:
                # Fallback: just save as-is
                type_summary = results_df.copy()
                type_summary.insert(0, 'query', query)
                type_summary.insert(1, 'source_dataset', source_dataset)
                type_summary.insert(2, 'target_dataset', target_dataset)
            

        # 6b. True type-level results: pooled all-adjacency type profiles
        # (source type vs every typed target type; NOT derived from the
        # bodyId-level scores).
        if type_level_df is not None and not type_level_df.empty:
            tl_cols = ['source_type', 'target_type', 'is_same_type',
                       'target_dataset', 'jaccard', 'weighted_jaccard',
                       'cosine', 'rank_union', 'rank']
            tl_out = type_level_df[[c for c in tl_cols if c in type_level_df.columns]]
            tl_out = tl_out.round(6)
            # Apply the run's top-N cap per source type (by type-level rank).
            if top_n and top_n > 0 and 'rank' in tl_out.columns:
                tl_out = (tl_out.sort_values(['source_type', 'rank'],
                                             na_position='last')
                          .groupby('source_type', as_index=False)
                          .head(int(top_n)))
            # Membership of each candidate type in the target dataset, so
            # very large (coarse / hemilineage-scale) annotations are
            # visible instead of silent.
            try:
                if 'target_type' in tl_out.columns:
                    member_counts = {
                        str(t): len(self.get_bodyids_for_type(str(t), target_dataset))
                        for t in tl_out['target_type'].dropna().astype(str).unique()
                    }
                    tl_out.insert(
                        tl_out.columns.get_loc('target_type') + 1,
                        'target_type_members',
                        tl_out['target_type'].astype(str).map(member_counts))
            except Exception as e:
                self._log(f"Type member counts skipped: {e}")
            _drop_rank_cols(tl_out).to_csv(results_dir / 'type_level_results.csv', index=False)
            files_saved.append('results/type_level_results.csv')
            self._log(f"Saved: results/type_level_results.csv "
                      f"({len(tl_out)} type-level rows, top {top_n} per "
                      f"source type)")

        # 7. Generate 3D skeleton visualizations if enabled. type_summary.csv
        # is written AFTER the scenes so `visualized` reflects the types
        # actually rendered rather than the table's own row order.
        rendered_types: List[str] = []
        if visualize_skeleton and not results_df.empty:
            rendered_types = self._visualize_homolog_candidates(
                results_df=results_df,
                query=query,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                visualization_dir=visualization_dir,
                top_n=visualize_top_n,
                files_saved=files_saved,
                type_summary=type_summary,
                type_level_df=type_level_df,
            ) or []

        if type_summary.columns.size:
            if rendered_types:
                ts_type_col = ('target_type' if 'target_type' in type_summary.columns
                               else 'target' if 'target' in type_summary.columns
                               else None)
            else:
                ts_type_col = None
            if ts_type_col:
                rendered_text = [str(t) for t in rendered_types]
                rank_map = {t: i + 1 for i, t in enumerate(rendered_text)}
                type_summary['visualized'] = (
                    type_summary[ts_type_col].astype(str).isin(rendered_text))
                type_summary['visualization_rank'] = (
                    type_summary[ts_type_col].astype(str).map(rank_map))
            _drop_rank_cols(type_summary).to_csv(results_dir / 'type_summary.csv', index=False)
            files_saved.append('results/type_summary.csv')
            self._log("Saved: results/type_summary.csv (type-mean aggregated FROM bodyId-level results)")

        # 8. Auto type mapping caveats: user_warning_notes.txt at the run
        # folder root (rendered by the run guide) whenever auto type mapping
        # expanded the queried type name or hit an N-to-1 / 1-to-N mapping.
        if self.use_auto_type_mapping:
            try:
                is_cross_dataset = str(source_dataset) != str(target_dataset)
                type_mapper = self._get_type_mapper_for_comparison(is_cross_dataset)
                mapping_notes = (
                    type_mapper.build_user_warning_notes(
                        [query], [source_dataset, target_dataset])
                    if type_mapper is not None else []
                )
            except Exception as e:
                mapping_notes = []
                self._log(f"Warning: could not build auto type mapping notes: {e}")
            if mapping_notes:
                try:
                    notes_path = output_path / 'user_warning_notes.txt'
                    with open(notes_path, 'w', encoding='utf-8') as f:
                        f.write('User warning notes\n')
                        f.write('==================\n\n')
                        f.write('Auto type mapping changed how the queried type was '
                                'matched across datasets:\n\n')
                        for note in mapping_notes:
                            f.write(f'- {note}\n')
                    files_saved.append('user_warning_notes.txt')
                    self._log("⚠️ Auto type mapping expanded or merged the queried "
                              "type - see user_warning_notes.txt and double check")
                except Exception as e:
                    self._log(f"Warning: could not write user_warning_notes.txt: {e}")

        self._log(f"Saved {len(files_saved)} files to {output_path}")
        
        return {
            'output_dir': str(output_path),
            'files_saved': files_saved,
            'query_profile': query_profile_data,
            'match_profiles': match_profiles
        }
    
    def _visualize_homolog_candidates(
        self,
        results_df: pd.DataFrame,
        query: Union[str, int],
        source_dataset: str,
        target_dataset: str,
        visualization_dir: 'Path',
        top_n: int = 5,
        files_saved: List[str] = None,
        type_summary: Optional[pd.DataFrame] = None,
        type_level_df: Optional[pd.DataFrame] = None,
        individual_profiles: bool = False,
        morph_qualification=None,
    ) -> List[str]:
        """
        Generate 3D skeleton visualizations for top homolog candidates.

        Optimized pipeline using neuron_layers and plot_individuals() for efficient
        batch visualization instead of creating separate VisualizeSkeleton instances.

        Creates three types of visualizations:
        1. bodyId_level/: The query (when source and target datasets match)
           followed by the top target bodyIds, with independent profiles via
           plot_individuals()
        2. type_level/: The query (when source and target datasets match)
           followed by target types with independent profiles per type via
           plot_individuals()
        3. source_neurons/: The source/query neurons plotted in their source
           dataset (also used as the cross-dataset reference scene)

        Returns the ordered list of target types that were actually rendered
        (bodyId-level scene order first, then type-level-only types), so the
        caller can flag ``visualized`` rows in type_summary truthfully.
        
        Uses VisualizeSkeleton module to create interactive HTML + PNG exports.
        The plot_individuals() method efficiently generates separate visualizations
        for each layer by toggling visibility rather than re-fetching data.
        
        Args:
            results_df: Results DataFrame with homolog matches
            query: Query neuron identifier
            source_dataset: Source dataset
            target_dataset: Target dataset
            visualization_dir: Directory to save visualization files
            top_n: Number of top candidates to visualize
            files_saved: List to append saved file names to
            type_summary: Optional type-level summary DataFrame
        """
        try:
            # VisualizeSkeleton is a standalone module.  Importing it from
            # ``coana`` used to fail because coana.py exposes the CLI helpers
            # but does not re-export this class; the failure was swallowed and
            # left the homolog run without any candidate visualizations.
            import sys
            from pathlib import Path
            
            # Ensure src is in path
            src_dir = Path(__file__).parent.parent
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            
            from visualize_skeleton import VisualizeSkeleton
            from neuprint import set_default_client

            self._log(f"Generating 3D visualizations for top {top_n} candidates...")
            # Target types actually rendered, in scene order (bodyId-level
            # scene first, then the type-level scene); returned to the caller
            # so type_summary's `visualized` flag can be set truthfully.
            rendered_types: List[str] = []

            # Use original dataset names for VisualizeSkeleton to ensure correct NeuPrint connection
            vis_source_dataset = source_dataset
            vis_target_dataset = target_dataset
            
            # Create subdirectories for organized output
            bodyid_dir = visualization_dir / 'bodyId_level'
            type_dir = visualization_dir / 'type_level'
            source_dir = visualization_dir / 'source_neurons'

            def _visualizer_kwargs(defaults: Optional[Dict[str, Any]] = None, **required):
                return self._homolog_visualizer_kwargs(defaults, **required)
            
            bodyid_dir.mkdir(parents=True, exist_ok=True)
            type_dir.mkdir(parents=True, exist_ok=True)
            source_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine target column names
            target_col = 'target_type' if 'target_type' in results_df.columns else 'target'
            target_bodyid_col = 'target_bodyId' if 'target_bodyId' in results_df.columns else None

            def _body_ids(values):
                """Normalize a sequence of bodyId-like values for rendering."""
                ids = []
                for value in values or []:
                    try:
                        if pd.isna(value):
                            continue
                        ids.append(int(value))
                    except (TypeError, ValueError):
                        continue
                return list(dict.fromkeys(ids))

            # The source/query is rendered as a reference layer.  For a type
            # query, result rows normally carry every source bodyId; when a
            # caller supplies a type-only result frame, resolve the members
            # from the same source lookup used by the search.
            query_bodyids = []
            if 'source_bodyId' in results_df.columns:
                query_bodyids = _body_ids(results_df['source_bodyId'].dropna().unique().tolist())
            if not query_bodyids and str(query).strip().isdigit():
                query_bodyids = [int(str(query).strip())]
            if not query_bodyids:
                try:
                    query_bodyids = _body_ids(
                        self.get_bodyids_for_type(str(query), vis_source_dataset)
                    )
                except Exception:
                    query_bodyids = []
            query_bodyid_set = set(query_bodyids)
            same_dataset = source_dataset == target_dataset
            safe_query = ''.join(
                char if char.isalnum() or char in '._-' else '_'
                for char in str(query)
            ).strip('_') or 'neuron'
            query_layer_name = f'query_{safe_query}_x{len(query_bodyids)}'

            # Cross-dataset: the query source is also rendered INSIDE the
            # target brain as a transformed overlay layer, labeled
            # query_transformed_{neuron name} (naming spec). Interactive tree
            # legends use the source row's native type for these overlays.
            query_overlay = None          # per-neuron entries (bodyId level)
            query_overlay_grouped = None  # single grouped layer (type level)
            if not same_dataset and query_bodyids:
                try:
                    query_overlay, query_overlay_grouped = (
                        self._build_query_overlay_layer(
                            vis_source_dataset, vis_target_dataset,
                            query_bodyids, safe_query,
                        ))
                except Exception as e:
                    self._log(f'Query overlay skipped: {e}')

            def _is_query_bodyid(value):
                try:
                    return not pd.isna(value) and int(value) in query_bodyid_set
                except (TypeError, ValueError):
                    return False

            # A same-dataset search can render the query and its matches in a
            # single scene. Cross-dataset searches keep separate scenes because
            # VisualizeSkeleton accepts one dataset per instance; the source
            # reference is still rendered below under source_neurons/.
            candidate_results = results_df
            if same_dataset and target_bodyid_col and query_bodyid_set:
                candidate_results = results_df.loc[
                    ~results_df[target_bodyid_col].map(
                        _is_query_bodyid
                    )
                ]

            # Get matches to visualize; for bodyId-level use union of per-source top-N targets
            if top_n and top_n > 0:
                if target_bodyid_col:
                    if 'source_bodyId' in candidate_results.columns:
                        per_source_matches = []
                        for _, group in candidate_results.groupby('source_bodyId'):
                            # the run's own sort metric, not a hard-coded
                            # rank_union — the picture must agree with the
                            # CSV the same run exported
                            _vc = pick_visualization_metric(
                                getattr(self, 'similarity_metric', None),
                                group.columns)
                            sorted_group = (
                                group.sort_values(_vc, ascending=False,
                                                  na_position='last')
                                if _vc else group)
                            per_source_matches.append(sorted_group.head(top_n))

                        if per_source_matches:
                            top_matches = pd.concat(per_source_matches, ignore_index=True)
                        else:
                            top_matches = results_df.iloc[0:0]
                    else:
                        if 'rank_corr' in results_df.columns:
                            top_matches = candidate_results.nlargest(top_n, 'rank_corr')
                        else:
                            top_matches = candidate_results.head(top_n)

                    top_matches = top_matches.drop_duplicates(subset=[target_bodyid_col], keep='first')
                else:
                    top_matches = candidate_results.head(top_n)
            else:
                top_matches = candidate_results

            # Morph qualification (optional): exclude pairs below the null
            # bar from the RENDERED set only — the results tables keep
            # every row (plan-morph-qualification-find-homolog.md).
            morph_failed_types: set = set()
            if morph_qualification is not None:
                mcd = self._import_morph_cross_dataset()
                top_matches, morph_excluded = (
                    mcd.filter_qualified_top_matches(
                        top_matches, morph_qualification))
                if len(morph_excluded):
                    morph_failed_types = {
                        str(v) for v in
                        morph_excluded[target_col].dropna().tolist()
                        if str(v)} if target_col in morph_excluded.columns \
                        else set()
                    preview = ", ".join(
                        f"{int(r['source_bodyId'])}->{int(r['target_bodyId'])}"
                        for _i, r in morph_excluded.iterrows()
                        if pd.notna(r.get('source_bodyId'))
                        and pd.notna(r.get('target_bodyId')))
                    more = (" …" if len(morph_excluded) > 10 else "")
                    self._log(
                        f"  Morph qualification: {len(morph_excluded)} "
                        "visualized candidate(s) below the null bar, "
                        f"excluded from the scene: {preview}{more}")
                    self._log(
                        "  (Rows stay in results/bodyid_results.csv with "
                        "morph_qualified = False.)")
            
            # =====================================================================
            # 1. BodyId-level visualizations: Batch all bodyIds as separate layers
            #    Using plot_individuals() for efficient per-neuron exports
            # =====================================================================
            if target_bodyid_col and (
                not top_matches.empty or (same_dataset and bool(query_bodyids))
            ):
                self._log(f"  Creating bodyId-level visualizations (batch mode with plot_individuals)...")
                if target_col in top_matches.columns:
                    for value in top_matches[target_col].dropna().tolist():
                        text = str(value)
                        if text and text not in rendered_types:
                            rendered_types.append(text)
                
                # Collect all bodyIds as separate layers (one bodyId per layer)
                bodyid_layers = []
                bodyid_layer_names = []

                if same_dataset and query_bodyids:
                    bodyid_layers.append(query_bodyids)
                    bodyid_layer_names.append(query_layer_name)
                
                for idx, row in top_matches.iterrows():
                    target_type = row.get(target_col, '')
                    target_bodyid = row.get(target_bodyid_col)
                    
                    if not target_bodyid:
                        continue
                    
                    try:
                        bodyid_layers.append(int(target_bodyid))
                        # Custom layer name: type_bodyid for clear identification
                        safe_name = f"{target_type}_{target_bodyid}".replace('/', '_').replace(':', '_').replace('*', '_')
                        bodyid_layer_names.append(safe_name)
                    except (ValueError, TypeError):
                        continue
                
                if bodyid_layers:
                    try:
                        # Get correct client for target dataset
                        target_client = self.clients.get(vis_target_dataset)
                        if target_client:
                            set_default_client(target_client)
                        
                        # Create single VisualizeSkeleton with all bodyIds as separate layers
                        # legend_mode='layer' gives each bodyId its own call-site
                        # group name; an interactive tree legend keeps every
                        # group expandable down to its bodyId-level rows.
                        vs_bodyid = VisualizeSkeleton(**_visualizer_kwargs(
                            {
                                'show_fig': False,
                                'export_views': False,
                                'brain_mesh': 'native',
                                'neuron_alpha': 0.2,
                                'legend_mode': 'layer',
                                'verbose': 'simple',
                                'skip_synapse': True,
                            },
                            dataset=vis_target_dataset,
                            neuron_layers=bodyid_layers,
                            custom_layer_names=bodyid_layer_names,
                            custom_neurons=query_overlay,
                            output_dir=str(bodyid_dir),
                            client=target_client,
                        ))
                        
                        # Plot all neurons together first (required for plot_individuals)
                        vs_bodyid.plot_neurons()

                        # Individual per-neuron exports are disabled by
                        # default (opt in via individual_profiles=True).
                        if individual_profiles:
                            profiles_out = vs_bodyid.plot_individuals(
                                output_format=['png', 'html'],
                                views=['front'],
                                summary_format=['pdf'],  # Generate PDF summary
                                neuron_alpha=0.2,
                            )

                            if files_saved is not None:
                                # Record what was actually produced instead of
                                # assuming exactly one file per requested layer name.
                                if profiles_out and Path(profiles_out).is_dir():
                                    for profile_path in sorted(
                                            Path(profiles_out).iterdir()):
                                        if profile_path.is_file():
                                            files_saved.append(
                                                'visualization/bodyId_level/'
                                                f'individual_profiles/{profile_path.name}'
                                            )
                                for summary_path in sorted(
                                        bodyid_dir.glob('individual_profiles*.pdf')):
                                    files_saved.append(
                                        f'visualization/bodyId_level/{summary_path.name}'
                                    )

                        self._log(
                            f"    Saved: bodyId_level/ ({len(bodyid_layers)} layers, "
                            "query + top matches)"
                        )

                    except Exception as e:
                        self._log(f"    Warning: BodyId batch visualization failed: {e}, falling back to individual mode...")
                        # Fallback to individual visualization (original method)
                        self._visualize_bodyids_individual(
                            top_matches, target_col, target_bodyid_col, 
                            vis_target_dataset, bodyid_dir, files_saved,
                            reference_bodyids=query_bodyids if same_dataset else None,
                            reference_name=query_layer_name,
                            custom_overlay=query_overlay,
                        )
            
            # =====================================================================
            # 2. Type-level visualizations: Batch all types as separate layers
            #    Each layer contains all bodyIds of that type, using plot_individuals()
            # =====================================================================
            query_types = set()
            if 'source_type' in results_df.columns and query_bodyid_set:
                source_rows = (
                    results_df[results_df['source_bodyId'].map(_is_query_bodyid)]
                    if 'source_bodyId' in results_df.columns
                    else results_df.iloc[0:0]
                )
                query_types = {
                    str(value) for value in source_rows['source_type'].dropna().tolist()
                    if str(value)
                }
            if not query_types and not str(query).strip().isdigit():
                query_types.add(str(query))

            # Prefer the pooled all-adjacency type-level results (true
            # type-level ranking) over the bodyId-mean aggregation view; fall
            # back to type_summary (bodyId-mean) and then to the bodyId-level
            # top matches.
            type_values = []
            if type_level_df is not None and not type_level_df.empty and 'target_type' in type_level_df.columns:
                tl = type_level_df
                if 'rank' in tl.columns:
                    tl = tl.sort_values(['source_type', 'rank'],
                                        ascending=[True, True], na_position='last')
                # Filter to the source type(s) actually queried.
                qsrc = query_types or {str(query)}
                if 'source_type' in tl.columns:
                    tl = tl[tl['source_type'].isin(qsrc)]
                type_values = tl['target_type'].dropna().unique().tolist()
            elif type_summary is not None and not type_summary.empty:
                ts = type_summary
                type_col_name = 'target_type' if 'target_type' in ts.columns else ('target' if 'target' in ts.columns else None)
                if type_col_name:
                    type_values = ts[type_col_name].dropna().tolist()
                else:
                    type_values = []
            else:
                type_values = top_matches[target_col].dropna().tolist()

            unique_types = []
            seen_types = set()
            for value in type_values:
                target_type = str(value)
                if not target_type or target_type in seen_types:
                    continue
                # A type-only same-dataset result cannot distinguish query
                # members from the matching type, so leave that type to the
                # explicit query layer. BodyId-level rows are filtered below
                # and may still render non-query members of the type.
                if (
                    same_dataset and not target_bodyid_col
                    and target_type in query_types
                ):
                    continue
                seen_types.add(target_type)
                unique_types.append(target_type)
                if top_n and top_n > 0 and len(unique_types) >= top_n:
                    break

            # Morph qualification (optional): a type whose every SCORED
            # member failed the bar has no qualified representation —
            # exclude it from the type-level scene. Types that were never
            # part of the visualized set (no qualification data) render.
            if morph_failed_types and unique_types:
                dropped_types = [t for t in unique_types
                                 if str(t) in morph_failed_types]
                if dropped_types:
                    unique_types = [t for t in unique_types
                                    if str(t) not in morph_failed_types]
                    self._log(
                        f"  Morph qualification: {len(dropped_types)} "
                        "type(s) without a qualified member excluded from "
                        "the type-level scene.")

            type_layers = []
            type_layer_names = []
            if same_dataset and query_bodyids:
                type_layers.append(
                    query_bodyids if len(query_bodyids) > 1 else query_bodyids[0]
                )
                type_layer_names.append(query_layer_name)

            if unique_types:
                self._log(
                    f"  Creating type-level visualizations for {len(unique_types)} "
                    "result types plus the query (batch mode with plot_individuals)..."
                )
                
                # Build type layers: each layer is a list of bodyIds for that type (grouped)
                for target_type in unique_types:
                    if not target_type:
                        continue
                    
                    # Get all bodyIds for this type from results
                    type_rows = candidate_results[
                        candidate_results[target_col] == target_type
                    ]
                    
                    # Resolve the type's FULL membership from the target
                    # dataset (pooled type-level results are type-level, so
                    # every member neuron belongs in the layer, not only the
                    # candidates that happened to appear in results_df).
                    # Cap the rendered members per type (mirrors the
                    # find-similar visualization's TYPE_RENDER_MEMBER_CAP).
                    try:
                        from morphology import TYPE_RENDER_MEMBER_CAP
                    except Exception:
                        TYPE_RENDER_MEMBER_CAP = 20
                    try:
                        members_for_type = self.get_bodyids_for_type(
                            str(target_type), vis_target_dataset)
                    except Exception:
                        members_for_type = []
                    if len(members_for_type) > TYPE_RENDER_MEMBER_CAP:
                        self._log(
                            f"    Warning: type '{target_type}' has "
                            f"{len(members_for_type)} members; rendering the "
                            f"first {TYPE_RENDER_MEMBER_CAP} "
                            f"(per-type render cap)")
                        members_for_type = members_for_type[:TYPE_RENDER_MEMBER_CAP]
                    if members_for_type:
                        bodyids_for_type = members_for_type
                    elif target_bodyid_col:
                        bodyids_for_type = _body_ids(
                            type_rows[target_bodyid_col].dropna().unique().tolist()
                        )
                    else:
                        bodyids_for_type = []

                    if bodyids_for_type:
                        # Multiple bodyIds for a type share one layer.
                        type_layers.append(
                            bodyids_for_type
                            if len(bodyids_for_type) > 1
                            else bodyids_for_type[0]
                        )
                        safe_name = str(target_type).replace('/', '_').replace(':', '_').replace('*', '_')
                        type_layer_names.append(safe_name)
                        if str(target_type) not in rendered_types:
                            rendered_types.append(str(target_type))
                    else:
                        continue
                
            if type_layers:
                try:
                    # Get correct client for target dataset
                    target_client = self.clients.get(vis_target_dataset)
                    if target_client:
                        set_default_client(target_client)

                    # Create single VisualizeSkeleton with all types as separate layers
                    vs_type = VisualizeSkeleton(**_visualizer_kwargs(
                        {
                            'show_fig': False,
                            'export_views': False,
                            'brain_mesh': 'native',
                            'neuron_alpha': 0.2,
                            'legend_mode': 'layer',
                            'verbose': 'simple',
                            'skip_synapse': True,
                        },
                        dataset=vis_target_dataset,
                        neuron_layers=type_layers,
                        custom_layer_names=type_layer_names,
                        custom_neurons=query_overlay_grouped,
                        output_dir=str(type_dir),
                        client=target_client,
                    ))

                    # Plot all types together first
                    vs_type.plot_neurons()

                    # Individual per-type exports are disabled by default
                    # (opt in via individual_profiles=True).
                    if individual_profiles:
                        profiles_out = vs_type.plot_individuals(
                            output_format=['png', 'html'],
                            views=['front'],
                            summary_format=['pdf'],
                            neuron_alpha=0.2,
                        )

                        if files_saved is not None:
                            # Record what was actually produced instead of
                            # assuming exactly one file per requested layer name.
                            if profiles_out and Path(profiles_out).is_dir():
                                for profile_path in sorted(
                                        Path(profiles_out).iterdir()):
                                    if profile_path.is_file():
                                        files_saved.append(
                                            'visualization/type_level/'
                                            f'individual_profiles/{profile_path.name}'
                                        )
                            for summary_path in sorted(
                                    type_dir.glob('individual_profiles*.pdf')):
                                files_saved.append(
                                    f'visualization/type_level/{summary_path.name}'
                                )

                    self._log(
                        f"    Saved: type_level/ ({len(type_layers)} layers, "
                        "query + top types)"
                    )

                except Exception as e:
                    self._log(f"    Warning: Type batch visualization failed: {e}, falling back to individual mode...")
                    # Fallback to individual visualization
                    self._visualize_types_individual(
                        unique_types, results_df, target_col, target_bodyid_col,
                        vis_target_dataset, type_dir, files_saved,
                        reference_bodyids=query_bodyids if same_dataset else None,
                        reference_name=query_layer_name,
                        custom_overlay=query_overlay_grouped,
                    )
            
            # =====================================================================
            # 3. Source neurons visualization: All source neurons together
            # =====================================================================
            self._log(f"  Creating source neurons visualization...")
            
            source_bodyids = query_bodyids or []
            if not source_bodyids and 'source_bodyId' in results_df.columns:
                source_bodyids = _body_ids(
                    results_df['source_bodyId'].dropna().unique().tolist()
                )
            if not source_bodyids and str(query).strip().isdigit():
                source_bodyids = [int(str(query).strip())]
            if not source_bodyids:
                source_bodyids = [query]
            
            if source_bodyids:
                safe_name = f"source_{query}".replace('/', '_').replace(':', '_').replace('*', '_')

                # The transformed query is already layer 1 of the
                # bodyId-level and type-level scenes (query_transformed_*).
                # The standalone source scene is a FALLBACK only: rendered in
                # the ORIGINAL source dataset (untransformed), and skipped
                # when the overlay is available.
                if query_overlay:
                    self._log(
                        "    Skipping standalone source_neurons scene: the "
                        "query is rendered in the target scenes as "
                        "query_transformed overlay"
                    )
                else:
                    # Fallback (no overlay available): render in the source
                    # dataset's own template.
                        try:
                            source_layers = [int(bid) for bid in source_bodyids]
                        except (ValueError, TypeError):
                            source_layers = [str(bid) for bid in source_bodyids]

                        source_client = self.clients.get(vis_source_dataset)
                        if source_client:
                            set_default_client(source_client)

                        # Separate layers enable plot_individuals when multiple
                        if len(source_layers) > 1:
                            source_layer_names = [f"source_{bid}" for bid in source_bodyids]
                            vs_source = VisualizeSkeleton(**_visualizer_kwargs(
                                {
                                    'show_fig': False,
                                    'export_views': True,
                                    'brain_mesh': 'native',
                                    'legend_mode': 'layer',
                                    'neuron_alpha': 0.2,
                                    'verbose': 'simple',
                                    'skip_synapse': True,
                                },
                                dataset=vis_source_dataset,
                                neuron_layers=source_layers,
                                custom_layer_names=source_layer_names,
                                saveas=safe_name,
                                output_dir=str(source_dir),
                                client=source_client,
                            ))
                            vs_source.plot_neurons()

                            # Individual exports: disabled by default.
                            if individual_profiles:
                                profiles_out = vs_source.plot_individuals(
                                    output_format=['png', 'html'],
                                    views=['front'],
                                    summary_format=['pdf'],
                                    neuron_alpha=0.2,
                                )

                                if files_saved is not None:
                                    files_saved.append(f'visualization/source_neurons/{safe_name}.html')
                                    files_saved.append(f'visualization/source_neurons/{safe_name}.png')
                                    if profiles_out and Path(profiles_out).is_dir():
                                        for profile_path in sorted(
                                                Path(profiles_out).iterdir()):
                                            if profile_path.is_file():
                                                files_saved.append(
                                                'visualization/source_neurons/'
                                                f'individual_profiles/{profile_path.name}'
                                            )
                                for summary_path in sorted(
                                        source_dir.glob('individual_profiles*.pdf')):
                                    files_saved.append(
                                        f'visualization/source_neurons/{summary_path.name}'
                                    )

                            self._log(f"    Saved: source_neurons/{safe_name}.html "
                                      f"({len(source_layers)} neurons)")
                        else:
                            vs_source = VisualizeSkeleton(**_visualizer_kwargs(
                                {
                                    'show_fig': False,
                                    'export_views': True,
                                    'brain_mesh': 'native',
                                    'legend_mode': 'single',
                                    'neuron_alpha': 0.2,
                                    'verbose': 'simple',
                                    'skip_synapse': True,
                                },
                                dataset=vis_source_dataset,
                                neuron_layers=source_layers,
                                saveas=safe_name,
                                output_dir=str(source_dir),
                                client=source_client,
                            ))
                            vs_source.plot_neurons()

                            if files_saved is not None:
                                files_saved.append(f'visualization/source_neurons/{safe_name}.html')
                                files_saved.append(f'visualization/source_neurons/{safe_name}.png')

                            self._log(f"    Saved: source_neurons/{safe_name}.html (1 neuron)")
        except ImportError as e:
            self._log(f"Warning: Could not import VisualizeSkeleton for visualization: {e}")
        except Exception as e:
            self._log(f"Warning: Visualization failed: {e}")
        return rendered_types

    def _homolog_visualizer_kwargs(
        self,
        defaults: Optional[Dict[str, Any]] = None,
        **required,
    ) -> Dict[str, Any]:
        """Build renderer kwargs for the individual-visualization fallbacks."""
        options = dict(defaults or {})
        settings = dict(self.visualization_settings or {})
        # Rendering semantics owned by a call site must win over user
        # preferences: the homolog outputs promise one legend entry per layer
        # named ``{type}_{bodyId}``, which requires the site's explicit
        # legend_mode instead of the app-wide default ('type'). The
        # interactive 'tree' preference IS honoured: its panel re-groups by
        # the call site's custom layer names, keeping candidates collapsed
        # per layer.
        if 'legend_mode' in options and settings.get('legend_mode') != 'tree':
            settings.pop('legend_mode', None)
        options.update(settings)
        options.pop('visualize_top_n', None)
        options.pop('visualize_by', None)
        options.pop('use_default_simplification', None)
        # Homolog-result visualizations default to line mode; the dedicated
        # Visualization > Skeleton workflow remains the high-quality tube
        # entry point unless the caller explicitly opts into tube/fine here.
        options.setdefault('skeleton_mode', 'line')
        pipeline = str(
            options.get('neuprint_skeleton_pipeline', 'fine') or 'fine'
        ).strip().lower()
        dataset_name = str(options.get('dataset', '') or '').strip().lower()
        options.setdefault(
            'cache_neurons', (
                True if is_local_connectome_dataset(dataset_name)
                else pipeline not in {
                    'fast', 'direct', 'artistic', 'fine_opt1'
                }
            )
        )
        options.update(required)
        if options.get('skeleton_mesh_simplification') is None:
            options['skeleton_mesh_simplification'] = (
                default_analysis_skeleton_mesh_simplification(
                    options.get('dataset'), pipeline
                )
            )
        if options.get('mesh_color') == 'auto':
            options.pop('mesh_color', None)
        return options
    
    def _build_query_overlay_layer(
        self,
        source_dataset: str,
        target_dataset: str,
        query_bodyids: List[int],
        query_label: str,
    ) -> Optional[List[Any]]:
        """Cross-dataset overlay layer: the query source rendered in the
        target brain.

        Fetches the source skeletons (source dataset's NeuPrint client),
        transforms them into the target dataset's native coordinate space via
        the flybrains bridge registry (male-cns -> FAFB resolves to the same
        offline affine + landmark-TPS bridge the ROI mesh pipeline uses), and
        returns two lists ready for VisualizeSkeleton's ``custom_neurons``:
        a per-neuron variant (each transformed source neuron its own layer,
        ``query_transformed_{label}_{bodyId}`` — for bodyId-level scenes)
        and a grouped variant (one ``query_transformed_{label}`` layer with
        every neuron — for type-level scenes). Returns (None, None) when
        there is nothing to overlay (same space, no bodyIds, or
        fetch/transform unavailable). Each transformed neuron retains its
        source dataset marker, so the interactive tree legend can use the
        source row's native ``type`` label rather than a mapped crosswalk
        value such as ``flywireType``.
        """
        if not query_bodyids:
            return None
        try:
            import navis
            import navis.interfaces.neuprint as neu
            from visualize_skeleton import (
                dataset_native_space, dataset_render_space,
                transform_neurons_to_space,
            )

            # Source skeletons live in the dataset's NATIVE space; the
            # overlay must land in the target scene's RENDER space (the
            # template-mode _get_template_info target). For male-cns the
            # native space is raw voxels (JRCFIB2022Mraw) while the scene
            # renders in JRCFIB2022M nanometers — transforming into the
            # native space would shrink the neuron ~8-45x (units mismatch).
            source_space = dataset_native_space(source_dataset)
            target_space = dataset_render_space(target_dataset)
            if source_space == target_space:
                return None

            # Source skeletons WITHOUT depending on self.clients: UI
            # subprocess runs may have no token-bound client at all.
            # FAFB and standalone BANC sources never have a NeuPrint client.
            # FAFB resolves through its local bundle/CAVE compatibility path;
            # BANC resolves through the public-release SWC path. NeuPrint
            # sources use the raw cache plus a NeuPrint fetch fallback.
            project_root = str(Path(__file__).resolve().parents[2])

            fetched = []
            if is_banc_dataset(source_dataset):
                # BANC ships no CAVE/NeuPrint skeleton product: resolve
                # through the shared batch fetch, whose BANC branch uses
                # the official public-bucket SWCs (fetch_banc_swc).
                from morphology import fetch_skeletons_on_demand_batch
                resolved = fetch_skeletons_on_demand_batch(
                    source_dataset, [int(b) for b in query_bodyids],
                    project_root=project_root, persist=True)
                fetched = [resolved[int(b)] for b in query_bodyids
                           if int(b) in resolved
                           and resolved[int(b)] is not None]
            elif is_fafb_dataset(source_dataset):
                from morphology import load_local_release_skeletons
                resolved = load_local_release_skeletons(
                    source_dataset, query_bodyids,
                    project_root=project_root, log=self._log)
                fetched = [resolved[int(b)] for b in query_bodyids
                           if int(b) in resolved]
            else:
                try:
                    from morphology import find_similar_raw_cache
                    raw_cache = find_similar_raw_cache(
                        source_dataset, project_root=project_root,
                        verbose=False)
                except Exception:
                    raw_cache = None

                missing = []
                for bid in query_bodyids:
                    try:
                        n = (raw_cache.load_skeleton(int(bid))
                             if raw_cache else None)
                    except Exception:
                        n = None
                    if n is not None:
                        fetched.append(n)
                    else:
                        missing.append(int(bid))

                if missing:
                    client = self.clients.get(source_dataset)
                    if client is None:
                        try:
                            from utils.token_manager import token_manager
                            from neuprint import Client as _Client
                            _tk = (token_manager.get_neuprint_token()
                                   or self.token)
                            if _tk:
                                client = _Client(
                                    'https://neuprint.janelia.org',
                                    dataset=source_dataset, token=_tk)
                        except Exception:
                            client = None
                    if client is not None:
                        try:
                            neurons = neu.fetch_skeletons(
                                missing, client=client, with_synapses=False)
                            fetched.extend(
                                n for n in (neurons or []) if n is not None)
                        except Exception as e:
                            self._log(
                                f'Query overlay: NeuPrint fallback fetch '
                                f'failed for {len(missing)} skeleton(s): {e}')
                    else:
                        self._log(
                            f'Query overlay: {len(missing)} source '
                            f'skeleton(s) not in the local raw cache and no '
                            f'NeuPrint client available')

            neurons = navis.NeuronList([n for n in fetched if n is not None])
            if neurons is None or len(neurons) == 0:
                self._log('Query overlay skipped: source skeletons '
                          'unavailable')
                return None

            transformed = transform_neurons_to_space(
                neurons, source_space, target_space, validate_bounds=True,
                verbose=True)
            if not transformed:
                self._log('Query overlay skipped: no neuron survived the '
                          f'{source_space} -> {target_space} transform')
                return None
            safe = ''.join(
                ch if ch.isalnum() or ch in '._-' else '_'
                for ch in str(query_label)
            ).strip('_') or 'neuron'

            # Per-neuron entries (single-neuron legend per transformed
            # source, mirroring the found-results layers) plus one grouped
            # entry for type-level scenes. Each transformed neuron carries
            # its source dataset so downstream exports (neuron_info.csv) can
            # resolve its metadata from the right table.
            for n in transformed:
                try:
                    n._drocat_source_dataset = source_dataset
                except Exception:
                    pass
            per_neuron = []
            for n in transformed:
                try:
                    bid = int(getattr(n, 'id', None)
                              or getattr(n, 'bodyId', None) or 0)
                except (TypeError, ValueError):
                    bid = 0
                nm = (f'query_transformed_{safe}_{bid}'
                      if str(bid) != str(query_label).strip()
                      else f'query_transformed_{bid}')
                per_neuron.append((nm, [n]))
            grouped = [n for _, ns in per_neuron for n in ns]
            self._log(
                f'Query overlay ready: query_transformed_{safe} '
                f'({len(transformed)} neuron(s), '
                f'{source_space} -> {target_space})')
            return per_neuron, [(f'query_transformed_{safe}', grouped)]
        except Exception as e:
            self._log(f'Query overlay skipped: {e}')
            return None

    def _visualize_bodyids_individual(
        self,
        top_matches: pd.DataFrame,
        target_col: str,
        target_bodyid_col: str,
        vis_target_dataset: str,
        bodyid_dir: 'Path',
        files_saved: List[str],
        reference_bodyids: Optional[List[int]] = None,
        reference_name: str = 'query',
        custom_overlay: Optional[List[Any]] = None,
        individual_profiles: bool = False,
    ):
        """Fallback method: visualize bodyIds individually (original approach)."""
        from visualize_skeleton import VisualizeSkeleton
        from neuprint import set_default_client
        
        target_client = self.clients.get(vis_target_dataset)
        if target_client:
            set_default_client(target_client)

        if reference_bodyids:
            try:
                vs = VisualizeSkeleton(**self._homolog_visualizer_kwargs(
                    {
                        'show_fig': False,
                        'brain_mesh': 'native',
                        'neuron_alpha': 0.2,
                        'legend_mode': 'layer',
                        'verbose': 'simple',
                    },
                    dataset=vis_target_dataset,
                    neuron_layers=[reference_bodyids],
                    custom_layer_names=[reference_name],
                    saveas=reference_name,
                    output_dir=str(bodyid_dir),
                    client=target_client,
                ))
                vs.plot_neurons()
                if files_saved is not None:
                    files_saved.append(f'visualization/bodyId_level/{reference_name}.html')
                    files_saved.append(f'visualization/bodyId_level/{reference_name}.png')
                self._log(
                    f"    Saved: bodyId_level/{reference_name}.html "
                    "(query reference)"
                )
            except Exception as e:
                self._log(f"    Warning: Could not visualize query reference: {e}")
        
        for idx, row in top_matches.iterrows():
            target_type = row.get(target_col, '')
            target_bodyid = row.get(target_bodyid_col)
            rank_corr = row.get('rank_corr', 0)
            
            if not target_bodyid:
                continue
            
            layers = [int(target_bodyid)]
            safe_name = f"{target_type}_{target_bodyid}".replace('/', '_').replace(':', '_').replace('*', '_')
            
            try:
                vs = VisualizeSkeleton(**self._homolog_visualizer_kwargs(
                    {
                        'show_fig': False,
                        'brain_mesh': 'native',
                        'neuron_alpha': 0.2,
                        'legend_mode': 'layer',
                        'verbose': 'simple',
                    },
                    dataset=vis_target_dataset,
                    neuron_layers=layers,
                    custom_layer_names=[safe_name],
                    custom_neurons=custom_overlay,
                    saveas=safe_name,
                    output_dir=str(bodyid_dir),
                    client=target_client,
                ))
                vs.plot_neurons()
                
                if files_saved is not None:
                    files_saved.append(f'visualization/bodyId_level/{safe_name}.html')
                    files_saved.append(f'visualization/bodyId_level/{safe_name}.png')
                
                self._log(f"    Saved: bodyId_level/{safe_name}.html (rank_corr={rank_corr:.3f})")
                
            except Exception as e:
                self._log(f"    Warning: Could not visualize bodyId {target_bodyid}: {e}")
    
    def _visualize_types_individual(
        self,
        unique_types: list,
        results_df: pd.DataFrame,
        target_col: str,
        target_bodyid_col: str,
        vis_target_dataset: str,
        type_dir: 'Path',
        files_saved: List[str],
        reference_bodyids: Optional[List[int]] = None,
        reference_name: str = 'query',
        custom_overlay: Optional[List[Any]] = None,
        individual_profiles: bool = False,
    ):
        """Fallback method: visualize types individually (original approach)."""
        from visualize_skeleton import VisualizeSkeleton
        from neuprint import set_default_client
        
        target_client = self.clients.get(vis_target_dataset)
        if target_client:
            set_default_client(target_client)

        if reference_bodyids:
            try:
                vs = VisualizeSkeleton(**self._homolog_visualizer_kwargs(
                    {
                        'show_fig': False,
                        'brain_mesh': 'native',
                        'neuron_alpha': 0.2,
                        'legend_mode': 'layer',
                        'verbose': 'simple',
                    },
                    dataset=vis_target_dataset,
                    neuron_layers=[reference_bodyids],
                    custom_layer_names=[reference_name],
                    saveas=reference_name,
                    output_dir=str(type_dir),
                    client=target_client,
                ))
                vs.plot_neurons()
                if files_saved is not None:
                    files_saved.append(f'visualization/type_level/{reference_name}.html')
                    files_saved.append(f'visualization/type_level/{reference_name}.png')
                self._log(
                    f"    Saved: type_level/{reference_name}.html "
                    "(query reference)"
                )
            except Exception as e:
                self._log(f"    Warning: Could not visualize query reference: {e}")
        
        for target_type in unique_types:
            if not target_type:
                continue
            
            type_rows = results_df[results_df[target_col] == target_type]
            best_rank_corr = type_rows['rank_corr'].max() if 'rank_corr' in type_rows.columns else 0
            
            bodyid_layers = []
            if target_bodyid_col:
                bodyids_for_type = type_rows[target_bodyid_col].dropna().unique().tolist()
                if bodyids_for_type:
                    try:
                        bodyid_layers = [int(bid) for bid in bodyids_for_type]
                    except (ValueError, TypeError):
                        bodyid_layers = [str(bid) for bid in bodyids_for_type]

            if bodyid_layers:
                layers = [bodyid_layers] if len(bodyid_layers) > 1 else bodyid_layers
            else:
                layers = [str(target_type)]
            
            safe_name = f"{target_type}".replace('/', '_').replace(':', '_').replace('*', '_')
            
            try:
                vs = VisualizeSkeleton(**self._homolog_visualizer_kwargs(
                    {
                        'show_fig': False,
                        'brain_mesh': 'native',
                        'verbose': 'simple',
                    },
                    dataset=vis_target_dataset,
                    neuron_layers=layers,
                    saveas=safe_name,
                    output_dir=str(type_dir),
                    client=target_client,
                ))
                vs.plot_neurons()
                
                if files_saved is not None:
                    files_saved.append(f'visualization/type_level/{safe_name}.html')
                    files_saved.append(f'visualization/type_level/{safe_name}.png')
                
                bodyid_note = f", bodyIds={len(bodyid_layers)}" if bodyid_layers else ""
                self._log(f"    Saved: type_level/{safe_name}.html (best_rank_corr={best_rank_corr:.3f}{bodyid_note})")
                
            except Exception as e:
                self._log(f"    Warning: Could not visualize type {target_type}: {e}")
    
    def _profile_to_dataframe(
        self,
        profile: 'ConnectivityProfile',
        dataset: str
    ) -> pd.DataFrame:
        """
        Convert a ConnectivityProfile to a DataFrame with partners, weights, and ranks.
        
        Args:
            profile: ConnectivityProfile to convert
            dataset: Dataset identifier for context
        
        Returns:
            DataFrame with columns: neuron_type, dataset, direction, partner_type, weight, rank
        """
        rows = []
        neuron_id = str(profile.neuron_id)
        
        # Export upstream partners (sorted by rank)
        for partner_type, weight in sorted(
            profile.upstream_partners.items(),
            key=lambda x: profile.upstream_ranks.get(x[0], 999)
        ):
            rank = profile.upstream_ranks.get(partner_type, None)
            rows.append({
                'neuron_type': neuron_id,
                'dataset': dataset,
                'direction': 'upstream',
                'partner_type': partner_type,
                'weight': weight,
                'rank': rank
            })
        
        # Export downstream partners (sorted by rank)
        for partner_type, weight in sorted(
            profile.downstream_partners.items(),
            key=lambda x: profile.downstream_ranks.get(x[0], 999)
        ):
            rank = profile.downstream_ranks.get(partner_type, None)
            rows.append({
                'neuron_type': neuron_id,
                'dataset': dataset,
                'direction': 'downstream',
                'partner_type': partner_type,
                'weight': weight,
                'rank': rank
            })
        
        return pd.DataFrame(rows)

    # =========================================================================
    # Random Control Test
    # =========================================================================
    
    def shuffle_profile(
        self,
        profile: 'ConnectivityProfile',
        seed: Optional[int] = None
    ) -> 'ConnectivityProfile':
        """
        Create a shuffled version of a connectivity profile for control testing.
        
        Shuffles the partner types while keeping the same number of partners and
        weight distribution. This creates a randomized profile that maintains
        the statistical properties but loses the biological meaning.
        
        The shuffle randomly reassigns weights to different partner types from
        the same dataset's type pool, preserving:
        - Total number of upstream/downstream partners
        - Weight distribution (same set of weights, different assignment)
        - Rank structure (maintains the same ranking pattern)
        
        Args:
            profile: The original ConnectivityProfile to shuffle
            seed: Random seed for reproducibility (optional)
        
        Returns:
            A new ConnectivityProfile with shuffled partner assignments
        
        Example:
            >>> finder = HomologFinder(...)
            >>> profile = finder.profiler.get_profile('Mi1', 'hemibrain:v1.2.1')
            >>> shuffled = finder.shuffle_profile(profile, seed=42)
            >>> # Compare results with shuffled vs original profile
        """
        from .connectivity_profiler import ConnectivityProfile
        
        if seed is not None:
            np.random.seed(seed)
        
        # Get all available types from the dataset for shuffling
        # We use the type pool from the dataset's connection cache
        dataset_key = canonical_dataset_name(profile.dataset).replace(':', '_').replace('.', '_')
        type_pool = self._get_type_pool(profile.dataset)
        
        if not type_pool:
            warnings.warn(f"No type pool available for {profile.dataset}, using profile types only")
            # Fall back to shuffling within existing types
            type_pool = list(set(list(profile.upstream_partners.keys()) + 
                               list(profile.downstream_partners.keys())))
        
        # Shuffle upstream partners
        upstream_weights = list(profile.upstream_partners.values())
        n_upstream = len(upstream_weights)
        if n_upstream > 0 and len(type_pool) >= n_upstream:
            shuffled_upstream_types = list(np.random.choice(
                type_pool, size=n_upstream, replace=False
            ))
        else:
            shuffled_upstream_types = list(np.random.permutation(
                list(profile.upstream_partners.keys())
            ))
        
        shuffled_upstream_partners = dict(zip(shuffled_upstream_types, upstream_weights))
        
        # Compute ranks for shuffled upstream
        sorted_upstream = sorted(shuffled_upstream_partners.items(), key=lambda x: -x[1])
        shuffled_upstream_ranks = {t: i+1 for i, (t, _) in enumerate(sorted_upstream)}
        
        # Shuffle downstream partners
        downstream_weights = list(profile.downstream_partners.values())
        n_downstream = len(downstream_weights)
        if n_downstream > 0 and len(type_pool) >= n_downstream:
            shuffled_downstream_types = list(np.random.choice(
                type_pool, size=n_downstream, replace=False
            ))
        else:
            shuffled_downstream_types = list(np.random.permutation(
                list(profile.downstream_partners.keys())
            ))
        
        shuffled_downstream_partners = dict(zip(shuffled_downstream_types, downstream_weights))
        
        # Compute ranks for shuffled downstream
        sorted_downstream = sorted(shuffled_downstream_partners.items(), key=lambda x: -x[1])
        shuffled_downstream_ranks = {t: i+1 for i, (t, _) in enumerate(sorted_downstream)}
        
        # Create new profile with shuffled data
        shuffled_profile = ConnectivityProfile(
            neuron_id=f"{profile.neuron_id}_shuffled",
            dataset=profile.dataset,
            upstream_partners=shuffled_upstream_partners,
            downstream_partners=shuffled_downstream_partners,
            upstream_ranks=shuffled_upstream_ranks,
            downstream_ranks=shuffled_downstream_ranks,
            upstream_top_k=profile.upstream_top_k,
            downstream_top_k=profile.downstream_top_k,
            total_upstream_weight=profile.total_upstream_weight,
            total_downstream_weight=profile.total_downstream_weight,
            num_neurons_aggregated=profile.num_neurons_aggregated,
            actual_upstream_count=profile.actual_upstream_count,
            actual_downstream_count=profile.actual_downstream_count,
            is_weak_connectivity=profile.is_weak_connectivity,
            unique_types_upstream=len(shuffled_upstream_partners),
            unique_types_downstream=len(shuffled_downstream_partners),
        )
        
        return shuffled_profile
    
    def _get_type_pool(self, dataset: str) -> List[str]:
        """
        Get all available neuron types from a dataset for shuffling.
        
        Args:
            dataset: Dataset identifier
        
        Returns:
            List of all unique type names in the dataset
        """
        # Try to get types from the profiler's connection cache
        dataset_key = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
        
        # Check profiler cache first
        from .connectivity_profiler import _PROFILER_CONN_CACHE
        if dataset_key in _PROFILER_CONN_CACHE:
            type_lookup = _PROFILER_CONN_CACHE[dataset_key].get('type_lookup', {})
            return list(set(type_lookup.values()))
        
        # Try neuron index from the profiler
        try:
            if hasattr(self.profiler, '_load_neuron_index'):
                neuron_idx = self.profiler._load_neuron_index(dataset)
                if neuron_idx is not None and 'type' in neuron_idx.columns:
                    return list(neuron_idx['type'].dropna().unique())
        except Exception:
            pass
        
        # Fall back to connection data
        try:
            conn_data = self.profiler._load_connection_data(dataset)
            if conn_data is not None:
                types = set()
                if 'type_pre' in conn_data.columns:
                    types.update(conn_data['type_pre'].dropna().unique())
                if 'type_post' in conn_data.columns:
                    types.update(conn_data['type_post'].dropna().unique())
                return list(types)
        except Exception:
            pass
        
        return []
    
    def run_random_control_test(
        self,
        source: Union[str, int],
        source_dataset: str,
        target_dataset: str,
        n_shuffles: int = 100,
        top_n: int = 20,
        seed: Optional[int] = None,
        show_progress: bool = True,
        output_dir: Optional[str] = None,
        saveas: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a random control test for homolog finding.
        
        This method performs homolog finding on both the real profile and multiple
        shuffled profiles to assess whether the homolog finding results are
        statistically significant. If the real profile produces significantly
        higher similarity scores than shuffled profiles, the homolog finding
        is considered meaningful.
        
        Algorithm:
        1. Build the real connectivity profile for the source neuron
        2. Run homolog finding on the real profile → get real_scores
        3. For n_shuffles iterations:
           a. Shuffle the profile (randomize partner type assignments)
           b. Run homolog finding on shuffled profile → get shuffled_scores
        4. Compare real_scores vs distribution of shuffled_scores
        5. Compute p-values and effect sizes
        
        Args:
            source: Source neuron (type name or bodyId)
            source_dataset: Dataset of source neuron
            target_dataset: Dataset to search for homologs
            n_shuffles: Number of shuffle iterations (default: 100)
            top_n: Number of top homolog candidates to consider
            seed: Random seed for reproducibility
            show_progress: Whether to show progress bar
            output_dir: Directory to save results (optional)
            saveas: Custom name for output folder (optional)
        
        Returns:
            Dict containing:
                - real_results: DataFrame of homolog finding results with real profile
                - shuffled_results: List of DataFrames from shuffled profiles
                - real_mean_score: Mean similarity score for real profile
                - shuffled_mean_scores: List of mean scores for each shuffle
                - p_value: Proportion of shuffles with higher mean score than real
                - z_score: Z-score of real score vs shuffle distribution
                - effect_size: Cohen's d effect size
                - is_significant: Whether result is significant (p < 0.05)
                - summary: Human-readable summary string
        
        Example:
            >>> finder = HomologFinder(
            ...     source_dataset='hemibrain:v1.2.1',
            ...     target_dataset='male-cns:v0.9'
            ... )
            >>> results = finder.run_random_control_test(
            ...     source='Mi1',
            ...     source_dataset='hemibrain:v1.2.1',
            ...     target_dataset='male-cns:v0.9',
            ...     n_shuffles=100,
            ...     seed=42
            ... )
            >>> print(results['summary'])
            >>> if results['is_significant']:
            ...     print("Homolog finding is meaningful!")
        """
        from datetime import datetime
        from pathlib import Path
        
        if seed is not None:
            np.random.seed(seed)
        
        self._log(f"\n{'='*60}")
        self._log(f"Running Random Control Test for: {source}")
        self._log(f"Source dataset: {source_dataset}")
        self._log(f"Target dataset: {target_dataset}")
        self._log(f"Number of shuffles: {n_shuffles}")
        self._log(f"{'='*60}\n")
        
        # 1. Get the real profile
        self._log("Step 1: Building real connectivity profile...")
        try:
            real_profile = self.profiler.get_profile(source, source_dataset)
        except Exception as e:
            self._log(f"Error building profile: {e}")
            return {'error': str(e), 'is_significant': False}
        
        # 2. Run homolog finding with real profile
        self._log("Step 2: Running homolog finding with real profile...")
        real_results = self.find_homologs_fast(
            source=source,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            top_n=top_n,
            show_progress=False,
        )
        
        if real_results.empty:
            self._log("No homologs found with real profile")
            return {
                'real_results': real_results,
                'shuffled_results': [],
                'real_mean_score': 0.0,
                'shuffled_mean_scores': [],
                'p_value': 1.0,
                'z_score': 0.0,
                'effect_size': 0.0,
                'is_significant': False,
                'summary': "No homologs found with real profile."
            }
        
        # Get mean rank_corr from real results (use normalized version)
        score_col = 'rank_corr' if 'rank_corr' in real_results.columns else 'jaccard'
        real_mean_score = real_results[score_col].mean()
        real_max_score = real_results[score_col].max()
        
        self._log(f"Real profile - Mean {score_col}: {real_mean_score:.4f}, Max: {real_max_score:.4f}")
        
        # 3. Run homolog finding with shuffled profiles
        self._log(f"\nStep 3: Running {n_shuffles} shuffle iterations...")
        shuffled_mean_scores = []
        shuffled_max_scores = []
        shuffled_results_list = []
        
        iterator = range(n_shuffles)
        if show_progress:
            try:
                iterator = tqdm(iterator, desc="Shuffle iterations",
                                disable=progress_bars_disabled(show_progress, self.verbose))
            except Exception:
                pass
        
        for i in iterator:
            # Create shuffled profile
            shuffle_seed = seed + i if seed is not None else None
            shuffled_profile = self.shuffle_profile(real_profile, seed=shuffle_seed)
            
            # Run homolog finding with shuffled profile
            # We need to temporarily replace the profile in the comparison
            shuffled_df = self._run_homolog_with_profile(
                shuffled_profile,
                target_dataset,
                top_n
            )
            
            if not shuffled_df.empty and score_col in shuffled_df.columns:
                shuffled_mean_scores.append(shuffled_df[score_col].mean())
                shuffled_max_scores.append(shuffled_df[score_col].max())
                shuffled_results_list.append(shuffled_df)
            else:
                shuffled_mean_scores.append(0.0)
                shuffled_max_scores.append(0.0)
        
        # 4. Compute statistics
        self._log("\nStep 4: Computing statistics...")
        shuffled_mean_scores = np.array(shuffled_mean_scores)
        shuffled_max_scores = np.array(shuffled_max_scores)
        
        # P-value: proportion of shuffles with mean score >= real score
        p_value = np.mean(shuffled_mean_scores >= real_mean_score)
        p_value_max = np.mean(shuffled_max_scores >= real_max_score)
        
        # Z-score
        if np.std(shuffled_mean_scores) > 0:
            z_score = (real_mean_score - np.mean(shuffled_mean_scores)) / np.std(shuffled_mean_scores)
        else:
            z_score = 0.0
        
        # Cohen's d effect size
        if np.std(shuffled_mean_scores) > 0:
            effect_size = (real_mean_score - np.mean(shuffled_mean_scores)) / np.std(shuffled_mean_scores)
        else:
            effect_size = 0.0
        
        is_significant = p_value < 0.05
        
        # Create summary
        summary_lines = [
            f"\n{'='*60}",
            "RANDOM CONTROL TEST RESULTS",
            f"{'='*60}",
            f"Source neuron: {source}",
            f"Source dataset: {source_dataset}",
            f"Target dataset: {target_dataset}",
            f"Number of shuffles: {n_shuffles}",
            f"Metric used: {score_col}",
            "",
            "SCORES:",
            f"  Real profile mean {score_col}: {real_mean_score:.4f}",
            f"  Real profile max {score_col}: {real_max_score:.4f}",
            f"  Shuffled mean (avg): {np.mean(shuffled_mean_scores):.4f} ± {np.std(shuffled_mean_scores):.4f}",
            f"  Shuffled max (avg): {np.mean(shuffled_max_scores):.4f} ± {np.std(shuffled_max_scores):.4f}",
            "",
            "STATISTICS:",
            f"  P-value (mean): {p_value:.4f} {'*' if p_value < 0.05 else ''} {'**' if p_value < 0.01 else ''} {'***' if p_value < 0.001 else ''}",
            f"  P-value (max): {p_value_max:.4f}",
            f"  Z-score: {z_score:.4f}",
            f"  Effect size (Cohen's d): {effect_size:.4f}",
            "",
            "INTERPRETATION:",
        ]
        
        if is_significant:
            if effect_size > 0.8:
                summary_lines.append("  ✓ SIGNIFICANT with LARGE effect size")
                summary_lines.append("  → Homolog finding results are highly meaningful")
            elif effect_size > 0.5:
                summary_lines.append("  ✓ SIGNIFICANT with MEDIUM effect size")
                summary_lines.append("  → Homolog finding results are meaningful")
            else:
                summary_lines.append("  ✓ SIGNIFICANT with SMALL effect size")
                summary_lines.append("  → Homolog finding results may be meaningful, interpret with caution")
        else:
            summary_lines.append("  ✗ NOT SIGNIFICANT")
            summary_lines.append("  → Homolog finding results may be due to chance")
            summary_lines.append("  → Consider using different parameters or more shuffles")
        
        summary_lines.append(f"{'='*60}")
        summary = "\n".join(summary_lines)
        
        self._log(summary)
        
        # 5. Save results if output_dir specified
        results = {
            'real_results': real_results,
            'shuffled_results': shuffled_results_list,
            'real_mean_score': float(real_mean_score),
            'real_max_score': float(real_max_score),
            'shuffled_mean_scores': list(shuffled_mean_scores),
            'shuffled_max_scores': list(shuffled_max_scores),
            'p_value': float(p_value),
            'p_value_max': float(p_value_max),
            'z_score': float(z_score),
            'effect_size': float(effect_size),
            'is_significant': bool(is_significant),
            'summary': summary,
            'n_shuffles': n_shuffles,
            'score_metric': score_col,
        }
        
        if output_dir:
            output_path = Path(output_dir)
            if saveas:
                output_path = output_path / saveas
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = output_path / f"control_test_{source}_{timestamp}"
            
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save real results
            real_results.to_csv(output_path / 'real_results.csv', index=False)
            
            # Save statistics
            stats_df = pd.DataFrame({
                'metric': ['real_mean_score', 'real_max_score', 'shuffled_mean_avg', 
                          'shuffled_mean_std', 'p_value', 'p_value_max', 'z_score', 
                          'effect_size', 'is_significant', 'n_shuffles'],
                'value': [real_mean_score, real_max_score, np.mean(shuffled_mean_scores),
                         np.std(shuffled_mean_scores), p_value, p_value_max, z_score,
                         effect_size, is_significant, n_shuffles]
            })
            stats_df.to_csv(output_path / 'control_test_stats.csv', index=False)
            
            # Save shuffled score distribution
            shuffle_dist_df = pd.DataFrame({
                'iteration': range(n_shuffles),
                'mean_score': shuffled_mean_scores,
                'max_score': shuffled_max_scores
            })
            shuffle_dist_df.to_csv(output_path / 'shuffled_score_distribution.csv', index=False)
            
            # Save summary
            # The summary carries ± → ✓ ✗, none of which exist in the GBK
            # code page used by default on zh-CN Windows hosts; without an
            # explicit encoding the write raises UnicodeEncodeError there.
            with open(output_path / 'summary.txt', 'w', encoding='utf-8') as f:
                f.write(summary)
            
            self._log(f"\nResults saved to: {output_path}")
            results['output_path'] = str(output_path)
        
        return results
    
    def _run_homolog_with_profile(
        self,
        profile: 'ConnectivityProfile',
        target_dataset: str,
        top_n: int
    ) -> pd.DataFrame:
        """
        Run homolog finding using a pre-built profile (for shuffle testing).
        
        This is an internal method that compares a given profile against
        all types in the target dataset.
        
        Args:
            profile: The connectivity profile to compare
            target_dataset: Dataset to search for homologs
            top_n: Number of top results to return
        
        Returns:
            DataFrame with homolog results
        """
        # Get target types from the target dataset
        target_types = self._get_type_pool(target_dataset)
        if not target_types:
            return pd.DataFrame()
        
        results = []
        
        for target_type in target_types:
            try:
                target_profile = self.profiler.get_profile(target_type, target_dataset)
                if target_profile is None:
                    continue
                
                # Compare profiles
                comparison = ProfileComparator.compare_profiles_simple(
                    profile, target_profile, direction='both'
                )
                
                results.append({
                    'target_type': target_type,
                    'target_dataset': target_dataset,
                    'rank_corr': comparison.get('rank_correlation', 0.0),
                    'jaccard': comparison.get('jaccard', 0.0),
                    'adjacency_score': comparison.get('shared_count', 0),
                })
            except Exception:
                continue
        
        if not results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        
        # Normalize rank correlation to 0-1 scale
        if 'rank_corr' in results_df.columns:
            results_df['rank_corr'] = (results_df['rank_corr'] + 1) / 2
        
        # Sort by rank_union and return top_n
        if 'jaccard' in results_df.columns:
            results_df = results_df.sort_values('jaccard', ascending=False).head(top_n)
        else:
            results_df = results_df.head(top_n)
        
        return results_df


# ============================================================================
# Connectivity Profile Comparer Class
# ============================================================================

# ============================================================================
# BodyId display labels (instance / type+hemisphere)
# ============================================================================
#
# BodyId-level exports label every row as '{bodyId}_{instance}' (NeuPrint-style
# datasets) or '{bodyId}_{type}_{L|R}' (FAFB/BANC), mirroring the
# visualization tree legend (``VisualizeSkeleton._tree_neuron_label``). The
# identity maps come from the dataset neuron table, then the neuron index —
# the same source priority as type mapping. Missing metadata degrades to the
# resolved comparison type label, then the bare bodyId (the pre-existing
# behavior), so label generation can never fail a run.

_BODY_ID_LABEL_MAPS_CACHE: Dict[str, Tuple[dict, dict, dict]] = {}


def _load_body_id_label_maps(dataset: str) -> Tuple[dict, dict, dict]:
    """(bodyId -> type, bodyId -> instance, bodyId -> side) for one dataset."""
    key = str(dataset or "")
    cached = _BODY_ID_LABEL_MAPS_CACHE.get(key)
    if cached is not None:
        return cached

    from pathlib import Path as _Path
    safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
    project_root = _Path(__file__).parent.parent.parent
    candidates = [
        project_root / 'datasets' / safe_name
        / f'{safe_name}_allneurons_neuron_df.parquet',
        project_root / 'datasets' / safe_name
        / f'{safe_name}_allneurons_neuron_df.csv',
        project_root / 'neuron_indexes' / safe_name / 'neuron_index.parquet',
    ]
    wanted = ('bodyId', 'type', 'instance', 'somaSide', 'hemisphere')
    type_map: dict = {}
    instance_map: dict = {}
    side_map: dict = {}
    for path in candidates:
        if type_map and instance_map and side_map:
            break
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == '.parquet':
                try:
                    import pyarrow.parquet as _pq
                    available = set(_pq.read_schema(path).names)
                    columns = [c for c in wanted if c in available]
                except Exception:
                    columns = None
                df = pd.read_parquet(path, columns=columns)
            else:
                df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if 'bodyId' not in df.columns:
            continue
        ids = df['bodyId']
        if 'type' in df.columns and not type_map:
            type_map = dict(zip(ids, df['type'].fillna('').astype(str)))
        if 'instance' in df.columns and not instance_map:
            instance_map = dict(zip(ids, df['instance'].fillna('').astype(str)))
        side_col = next(
            (c for c in ('somaSide', 'hemisphere') if c in df.columns), None)
        if side_col and not side_map:
            side_map = dict(zip(ids, df[side_col].fillna('').astype(str)))

    maps = (type_map, instance_map, side_map)
    _BODY_ID_LABEL_MAPS_CACHE[key] = maps
    return maps


def _body_id_display_label(dataset: str, bid, fallback_type: str = None) -> str:
    """Tree-legend display label for one bodyId (never fails)."""
    bid_text = str(bid).strip()
    if build_body_id_label is None:
        label = (f"{bid_text}_{fallback_type}" if fallback_type else bid_text)
        return label
    type_map, instance_map, side_map = _load_body_id_label_maps(dataset)
    try:
        return build_body_id_label(
            dataset, bid, type_map, instance_map, side_map,
            fallback_type=fallback_type)
    except Exception:
        return (f"{bid_text}_{fallback_type}" if fallback_type else bid_text)


def _body_id_instance_name(dataset: str, bid) -> str:
    """Instance name for one bodyId from the dataset maps ('' when unknown)."""
    _, instance_map, _ = _load_body_id_label_maps(dataset)
    for key in (bid, str(bid)):
        if key in instance_map:
            return str(instance_map.get(key) or "").strip()
    text = str(bid).strip()
    if text.isdigit():
        try:
            return str(instance_map.get(int(text)) or "").strip()
        except (TypeError, ValueError):
            return ""
    return ""


def _bodyid_profile_stem(bid, type_label) -> str:
    """Filename stem for one bodyId connectivity profile.

    At bodyid aggregation the type label already reads
    ``'{bodyId}_{instance/type}'``; avoid the doubled
    ``'{bid}_{bid}_...'`` stem by prefixing only when needed.
    """
    bid_text = str(bid)
    label = str(type_label)
    if label.startswith(f"{bid_text}_"):
        return label
    return f"{bid_text}_{label}"


class ConnectivityProfileComparer:
    """
    Compare connectivity profiles within or across datasets.
    
    This class provides functionality for connectivity profile comparison,
    supporting both intra-dataset and cross-dataset comparisons with
    interactive heatmap visualization.
    
    Key Features:
        - Intra-dataset comparison: Compare neurons within a single dataset
        - Cross-dataset comparison: Compare neurons across different datasets (N×M matrix)
        - Flexible query input: simple list, nested list with custom names, or CSV file
        - Dict query format for cross-dataset: {'datasetA': [types], 'datasetB': [types]}
        - Auto type mapping: Standardizes partner types across datasets (can be disabled)
        - Mean-pooled aggregation for type-level profiles
        - Outputs ALL metrics: jaccard, cosine, rank_corr, rank_corr_union
        - Separate heatmap files for EACH metric (not just one with switching)
        - Separate upstream/downstream and combined analysis
        - Interactive heatmap via VisualizePath with native Ward clustering
        - Auto-generated output folder: profiling_{query_name}_{timestamp}
        - Saves individual and aggregated connectivity profiles
    
    Query Input Formats:
        1. Single dataset (intra-dataset comparison):
           - Simple list: ['Mi1', 'Tm3', 720575940610453042]
           - Nested list with custom names: [['DN1p', ['DN1pA', 'DN1pB']], ...]
           - CSV file via group_map_csv
           
        2. Cross-dataset comparison (dict format):
           - Dict with dataset keys: {'male-cns:v0.9': ['Mi1', 'aMe12'], 
                                       'flywire_FAFB_v783': ['Mi1', 'aMe12']}
           - Generates N×M similarity matrix between neurons from different datasets
           - Auto type mapping enabled by default to standardize partner types
    
    Aggregation Strategies:
        - 'bodyid': Compare individual bodyId profiles directly
        - 'type': Aggregate profiles by neuron type using mean pooling
        - 'custom': Compare user-defined custom groups (LabelMapper preset,
          nested-list groups or group_map_csv)
    
    Pattern Expansion:
        - aggregation_level='type': a query item that matches no exact type
          and looks like a regex pattern ('aMe.*', '.*DN.*') expands into
          its matched types, each becoming an INDEPENDENT row. The same
          pattern under 'custom' aggregation is taken literally as one group.
    
    Custom Groups (aggregation_level='custom'):
        - LabelMapper preset JSON (custom_mapping_file): groups come from
          the preset's source_mapping side — each custom_label/std_label
          row is one group, and its members for the profiling dataset fill
          the row
        - Nested list query: [['Group1', [id1, id2]], ...]
        - group_map_csv: columns 'group' and 'id_type_instance'
        - A flat query under 'custom' keeps each item as one single-member
          group (no pattern expansion)
    
    Auto Type Mapping (Cross-Dataset):
        When comparing across datasets, partner types may have different names
        (e.g., 'MTe07' in FAFB vs 'MeVPLo2' in male-cns). Auto type mapping
        standardizes partner types to their canonical (male-cns) names before
        comparison. This can be disabled via use_auto_type_mapping=False.
    
    Output Structure:
        {output_dir}/profiling_{query_name}_{timestamp}/
        ├── parameters.json
        ├── README.txt
        ├── results/
        │   ├── similarity_jaccard_{direction}.csv
        │   ├── similarity_cosine_{direction}.csv
        │   ├── similarity_rank_corr_{direction}.csv
        │   └── similarity_rank_corr_union_{direction}.csv
        ├── profiles/
        │   ├── profiles_summary.json
        │   ├── individual/          # Raw profiles per bodyId
        │   └── aggregated/          # Type-aggregated profiles
        └── visualization/
            ├── heatmap_{direction}_jaccard.html
            ├── heatmap_{direction}_cosine.html
            ├── heatmap_{direction}_rank_corr.html
            └── heatmap_{direction}_rank_corr_union.html
    
    Example:
        >>> # Intra-dataset: Simple list format
        >>> comparer = ConnectivityProfileComparer(
        ...     query=['Mi1', 'Tm3', 'aMe12'],
        ...     dataset='male-cns:v0.9',
        ...     aggregation_level='type',
        ...     output_dir='./results'
        ... )
        >>> results = comparer.run()
        
        >>> # Cross-dataset: Dict format (generates N×M matrix)
        >>> comparer = ConnectivityProfileComparer(
        ...     query={
        ...         'male-cns:v0.9': ['Mi1', 'aMe12', 'MeVPLo2'],
        ...         'flywire_FAFB_v783': ['Mi1', 'aMe12', 'MTe07'],
        ...     },
        ...     use_auto_type_mapping=True,  # Standardize partner types
        ...     output_dir='./results'
        ... )
        >>> # Note: dataset=None when using dict query
        
        >>> # Nested list with custom group names
        >>> comparer = ConnectivityProfileComparer(
        ...     query=[['Clock Neurons', ['DN1p', 'DN2']], 
        ...            ['Visual', ['Mi1', 'Tm3']]],
        ...     dataset='male-cns:v0.9'
        ... )
        
        >>> # Using CSV file for group mapping
        >>> comparer = ConnectivityProfileComparer(
        ...     query=[],  # Will be overridden by CSV
        ...     dataset='male-cns:v0.9',
        ...     group_map_csv='my_groups.csv'
        ... )
    """
    
    # Keep the report ordering aligned with the matrices generated by the
    # profiler. The report deliberately keeps this order stable so readers
    # can compare cards across datasets.
    _REPORT_METRICS = (
        'jaccard', 'weighted_jaccard', 'cosine',
        'rank_corr', 'rank_corr_union',
    )
    _REPORT_DIRECTIONS = ('overall', 'upstream', 'downstream')

    # Use explicit report scales so zero is always white. Positive-only
    # similarity metrics run from white to red; signed rank metrics run from
    # blue through white to red. Both live in report_kit so the morphology
    # comparison renders with the same conventions.
    _REPORT_POSITIVE_COLORSCALE = report_kit.REPORT_POSITIVE_COLORSCALE
    _REPORT_DIVERGING_COLORSCALE = report_kit.REPORT_DIVERGING_COLORSCALE

    def _report_metric_styles(self) -> Dict[str, 'report_kit.MetricStyle']:
        """Per-metric report presentation, in _REPORT_METRICS card order."""
        return {
            metric: report_kit.metric_style(
                metric, self._metric_display_name(metric))
            for metric in self._REPORT_METRICS
        }

    def __init__(
        self,
        query: Union[str, int, List[Union[str, int, List]], Dict[str, List]],
        dataset: Optional[str] = None,
        datasets: Optional[List[str]] = None,
        aggregation_level: str = 'type',
        top_k: int = 25,
        top_m: int = 5,
        min_synapse_threshold: int = 3,
        direction: str = 'both',
        output_dir: Optional[str] = None,
        generate_heatmaps: bool = True,
        show_figures: bool = False,
        verbose: bool = True,
        use_cache: bool = True,
        group_map_csv: Optional[str] = None,
        custom_mapping_file: Optional[str] = None,
        skip_bodyId_level: Union[bool, str] = 'auto',
        use_auto_type_mapping: bool = True,
        ensure_cache_complete: bool = False,
    ):
        """
        Initialize ConnectivityProfileComparer.
        
        Args:
            query: Neuron query - supports multiple formats:
                - Single type/bodyId: 'Mi1' or 720575940610453042
                - List of types/bodyIds: ['Mi1', 'Tm3', 'aMe12']
                - Patterns with wildcards: ['Mi.*', 'Tm.*']
                - Nested list with custom group names (like VisualizeSkeleton):
                  [['Group1', [bodyId1, bodyId2]], ['Group2', [bodyId3, 'type1']]]
                  Each group is ['GroupName', [list_of_ids_or_types]]
                - Dict for cross-dataset comparison (generates N×M matrix):
                  {'male-cns:v0.9': ['Mi1', 'aMe12'], 'flywire_FAFB_v783': ['Mi1']}
                  When using dict format, the `dataset` parameter should be None.
            dataset: Dataset identifier (e.g., 'male-cns:v0.9').
                For intra-dataset comparison, this is required.
                For cross-dataset comparison (dict query), this should be None.
                If dict query is provided but dataset is not None, a warning is issued.
            datasets: Optional list of dataset identifiers for MULTI-DATASET
                profiling (2+): the same query is profiled in every dataset
                (names mapped per dataset via the type mapper), intra-dataset
                matrices are computed per dataset, and the same queried
                neuron is compared ACROSS datasets with the homolog finding
                backend algorithm. When provided, `dataset` is ignored.
            aggregation_level: 'bodyid', 'type' or 'custom'
                - 'type' (default): each matched neuron type is one row;
                  pattern items like 'aMe.*' expand to independent types
                - 'bodyid': every individual neuron is one row
                  (labels {bodyId}_{type})
                - 'custom': rows are user-defined custom groups (from a
                  LabelMapper preset, nested-list query or group_map_csv);
                  a flat query keeps each item as its own group
            top_k: Top K partners per direction (default: 15)
            top_m: Minimum unique types to ensure (default: 5)
            min_synapse_threshold: Minimum synapses for connections
            direction: 'upstream', 'downstream', or 'both'
            output_dir: Parent directory to save results (subfolder auto-generated)
            generate_heatmaps: Generate interactive heatmap visualizations
            show_figures: Open visualizations in browser
            verbose: Print progress messages
            use_cache: Enable profile caching
            group_map_csv: Path to CSV file for group mapping (like VisualizeSkeleton's layer_map_csv)
                CSV format: columns 'group' and 'id_type_instance'
                - 'group': custom group name (neurons with same group value are grouped)
                - 'id_type_instance': neuron identifier (bodyId, type, or instance name)
                When provided, this overrides `query` parameter.
            custom_mapping_file: Path to a LabelMapper preset JSON (the format
                exported by the Settings tab's mapping presets, passed straight to
                LabelMapper(overall_mapping_json=...)). Custom groups are read from
                the preset's source_mapping side: each custom_label/std_label row is
                one group, and its members for the profiling dataset fill the row.
                When provided, this overrides `query` and forces
                aggregation_level='custom' (rows = groups).
                JSON format: {"source_mapping": {
                    "custom_label": ["grp1", "grp2"],
                    "hemibrain:v1.2.1": [["aMe12", "aMe12_R"], ["aMe12_L"]]}}
            skip_bodyId_level: Whether to skip bodyId-level computations (Steps 3 & 4).
                - False: Always compute bodyId-level matrices
                - True: Always skip bodyId-level matrices  
                - 'auto': Skip if n_bodyId_profiles > 1000 (default)
            Note: Cross-dataset comparison always skips bodyId-level matrices;
                multi-dataset profiling can compute bodyId-level intra-dataset
                matrices for each dataset.
            use_auto_type_mapping: Enable automatic type name mapping for cross-dataset
                comparison (default: True). When enabled, partner types resolve through
                the shared validity-aware resolver (comparison.type_resolver):
                licensed renames and valid splits map to canonical (male-cns)
                names, conflicts fail closed, and unmapped types keep their raw
                name as the counted long-tail fallback. This allows proper matching
                of types with different names across datasets (e.g., 'MTe07' in FAFB
                → 'MeVPLo2' in male-cns). Has no effect for intra-dataset comparison.
            ensure_cache_complete: If True, build/complete the FULL dataset connection
                cache before profiling (fetches connections for every uncached neuron).
                This can take hours on first use with a new dataset. Default False:
                profiling uses the connections already cached (recommended for the UI).
        """
        self.group_map_csv = group_map_csv
        self.use_auto_type_mapping = use_auto_type_mapping
        self._type_mapper: Optional[CrossDatasetTypeMapper] = None
        # One mapper snapshot + per-run resolution bookkeeping for the whole
        # comparison (Workstreams D/E/F): every query mapping and profile
        # expansion records its status so results can distinguish a real
        # zero similarity from an unmapped, conflicted, or unavailable run.
        self._mapper_snapshot = None
        self.mapping_resolution_record: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.mapping_resolution_counts: Dict[str, int] = {}
        self.mapping_raw_fallback_used = False
        # Unique-resolution dedupe for query mapping: one count per distinct
        # (dataset, item), so repeated/overlapping resolution paths do not
        # double-count.
        self._counted_query_resolutions: set = set()
        self._expansion_decision_cache: Dict = {}
        self._expansion_status_counts: Counter = Counter()
        # Canonical merge-key cache for anchor keys (_canonical_label).
        self._merge_key_cache: Dict = {}

        # Detect cross-dataset mode (dict query format)
        self.is_cross_dataset = isinstance(query, dict)
        
        # Multi-dataset profiling mode (2+ datasets, one shared query)
        self.is_multi_dataset = datasets is not None and len(datasets) >= 2
        
        # Normalize aggregation level (accepts the UI label 'custom group')
        self.aggregation_level = {
            'type': 'type', 'bodyid': 'bodyid',
            'custom': 'custom', 'custom group': 'custom',
        }.get(str(aggregation_level).lower(), 'type')
        self.custom_mapping_file = custom_mapping_file
        # Set early: the query/mapping parsing below logs via self._log
        self.verbose = verbose
        
        if self.is_cross_dataset:
            # Cross-dataset comparison mode
            # Query format: {'datasetA': [types], 'datasetB': [types]}
            if dataset is not None:
                warnings.warn(
                    f"Cross-dataset comparison detected (dict query). "
                    f"The 'dataset' parameter ('{dataset}') is ignored. "
                    f"Datasets are specified in the query dict.",
                    UserWarning
                )
            
            # Extract datasets and queries from dict
            self._cross_dataset_query = query  # Store original dict
            self.datasets = list(query.keys())
            
            if len(self.datasets) < 2:
                raise ValueError(
                    "Cross-dataset comparison requires at least 2 datasets in query dict. "
                    f"Got: {self.datasets}"
                )
            
            # For cross-dataset, we use the first dataset as "primary" for internal bookkeeping
            self.dataset = self.datasets[0]
            
            # Flatten query for logging
            self.query = []
            self._custom_group_names = None
            for ds, neurons in query.items():
                for n in neurons:
                    self.query.append(f"{ds}:{n}")
            
            # Cross-dataset always skips bodyId-level (profiles from different datasets)
            if skip_bodyId_level != True:
                self._log_pending = f"Cross-dataset comparison: bodyId-level matrices skipped (profiles from different datasets)"
            self.skip_bodyId_level = True
            
        elif self.is_multi_dataset:
            # Multi-dataset profiling mode: the SAME query is resolved in every
            # dataset (type names mapped per dataset) and compared both within
            # each dataset (intra) and across datasets (inter, same neuron).
            self._cross_dataset_query = None
            self.datasets = [str(ds) for ds in datasets]
            self.dataset = self.datasets[0]
            self._log_pending = None
            
            # Parse group_map_csv / LabelMapper preset / nested query as usual
            if group_map_csv is not None:
                self.query, self._custom_group_names = self._parse_group_map_csv(group_map_csv)
            elif custom_mapping_file is not None:
                self.query, self._custom_group_names = self._load_custom_groups_from_mapping(
                    custom_mapping_file, self.dataset
                )
                if self.aggregation_level != 'custom':
                    self._log(f"custom_mapping_file forces aggregation_level='custom' "
                              f"(was '{self.aggregation_level}')")
                    self.aggregation_level = 'custom'
            else:
                self.query, self._custom_group_names = self._normalize_query(query)
            
            # Multi-dataset runs can compute bodyId-level matrices for each
            # dataset's intra-dataset comparison, using the same skip policy
            # as single-dataset runs.  Inter-dataset matrices remain type-level
            # because bodyIds are not comparable across datasets.
            self.skip_bodyId_level = skip_bodyId_level
            
        else:
            # Intra-dataset comparison mode
            self._cross_dataset_query = None
            # The UI supplies a one-element ``datasets`` list for the
            # single-dataset case. Normalize it before parsing mappings so
            # custom LabelMapper groups receive the actual dataset name too.
            if dataset is None and datasets:
                dataset = str(datasets[0])
            self.datasets = [dataset] if dataset else []
            self._log_pending = None
            
            # Parse group_map_csv if provided (overrides query)
            if group_map_csv is not None:
                self.query, self._custom_group_names = self._parse_group_map_csv(group_map_csv)
            elif custom_mapping_file is not None:
                # Custom groups from a LabelMapper preset (source side);
                # rows are groups, so custom aggregation is forced.
                self.query, self._custom_group_names = self._load_custom_groups_from_mapping(
                    custom_mapping_file, dataset
                )
                if self.aggregation_level != 'custom':
                    self._log(f"custom_mapping_file forces aggregation_level='custom' "
                              f"(was '{self.aggregation_level}')")
                    self.aggregation_level = 'custom'
            else:
                # Normalize query and detect nested list format
                self.query, self._custom_group_names = self._normalize_query(query)
            
            if dataset is None:
                raise ValueError(
                    "For intra-dataset comparison, 'dataset' parameter is required. "
                    "For cross-dataset comparison, use dict query format: "
                    "{'datasetA': [types], 'datasetB': [types]}"
                )
        
        self.dataset = self.datasets[0] if (self.is_cross_dataset or self.is_multi_dataset) else dataset
        self.top_k = top_k
        self.top_m = top_m
        self.min_synapse_threshold = min_synapse_threshold
        self.direction = direction
        self.output_dir = output_dir
        self.generate_heatmaps = generate_heatmaps
        self.skip_heatmap = not generate_heatmaps
        self.show_figures = show_figures
        self.verbose = verbose
        self.use_cache = use_cache
        self.ensure_cache_complete = ensure_cache_complete
        if not (self.is_cross_dataset or self.is_multi_dataset):
            self.skip_bodyId_level = skip_bodyId_level
        
        # Generate query name for output folder
        self.query_name = self._generate_query_name()
        
        # Initialize profiler with all datasets
        config = ProfilerConfig(
            top_k_bodyid=top_k,
            top_m_type=top_m,
            min_synapse_threshold=min_synapse_threshold,
            use_cache=use_cache
        )
        
        profiler_datasets = self.datasets if (self.is_cross_dataset or self.is_multi_dataset) else [dataset]
        self.profiler = ConnectivityProfiler(
            datasets=profiler_datasets,
            config=config,
            verbose=verbose
        )
        
        # Storage for profiles and results
        self.profiles: Dict[str, ConnectivityProfile] = {}
        
    def _log(self, msg: str):
        """Print message if verbose."""
        if self.verbose:
            print(f"[ConnectivityProfileComparer] {msg}")

    def _progress(self, step: int, total: int, label: str = ""):
        """Emit a structured step-progress event consumed by the web UI.

        The line ``[DROCAT][progress] <step>/<total> <label>`` drives the
        determinate progress bar + step label in the results panel; it is a
        control event, never shown in the execution log.
        """
        if self.verbose:
            msg = f"[DROCAT][progress] {int(step)}/{int(total)} {label}".rstrip()
            print(msg, flush=True)
    
    def _format_query_for_log(self, query: List, max_items: int = 5) -> str:
        """
        Format query list for logging, truncating if too long.
        
        Args:
            query: List of query items
            max_items: Maximum items to show before truncating (default: 5)
            
        Returns:
            Formatted string representation
        """
        if not isinstance(query, list):
            return str(query)
        
        n = len(query)
        if n <= max_items * 2:
            # Short enough to show fully
            return str(query)
        
        # Show first and last few items with count
        first_items = query[:max_items]
        last_items = query[-max_items:]
        middle_count = n - max_items * 2
        
        first_str = ', '.join(str(x) for x in first_items)
        last_str = ', '.join(str(x) for x in last_items)
        
        return f"[{first_str}, ... ({middle_count} more) ..., {last_str}] ({n} total items)"
    
    def _sort_types_string_first(self, types: List[str]) -> List[str]:
        """
        Sort types with string types first, numeric types (untyped neurons) last.
        
        Numeric-like strings represent bodyIds of untyped neurons.
        
        Args:
            types: List of type names (may include numeric strings)
            
        Returns:
            Sorted list with string types first, then numeric types
        """
        string_types = []
        numeric_types = []
        
        for t in types:
            if str(t).isdigit():
                numeric_types.append(t)
            else:
                string_types.append(t)
        
        # Sort each group alphabetically
        string_types.sort()
        numeric_types.sort()
        
        # Combine: strings first, then numeric
        return string_types + numeric_types
    
    def _ensure_connection_cache_complete(self) -> bool:
        """
        Ensure connection cache is complete for the dataset.
        
        This method calls FNC's build_connection_cache which:
        - Checks which neurons are already cached (O(1) lookup)
        - Only fetches missing neurons (incremental)
        - Uses progress bar for long operations
        
        Cache Hierarchy (must be built in order):
        -----------------------------------------
        Level 0: datasets/{dataset}/*_neuron_df - Authoritative neuron list  
        Level 1: neuron_indexes/{dataset}/neuron_index.parquet - Neuron metadata index
        Level 2: cache/{dataset}/connections.parquet - Connection data
        Level 3: Connectivity profiles (built after this check)
        
        Returns:
            True if cache is ready, False otherwise
        """
        try:
            # Import FindNeuronConnection - try relative first (src.coana), then absolute
            try:
                from ..coana import FindNeuronConnection
            except ImportError:
                try:
                    from coana import FindNeuronConnection
                except ImportError:
                    import sys
                    from pathlib import Path
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from coana import FindNeuronConnection
            
            # Initialize FindNeuronConnection
            # Use 'simple' mode to show loading progress
            fnc = FindNeuronConnection(
                dataset=self.dataset,
                use_cache=True,
                verbose_mode='simple',
                simple_fetch=True
            )
            
            # Build connection cache - incremental (skips already cached neurons)
            self._log("Building/verifying connection cache...")
            result = fnc.build_connection_cache(batch_size=1000, quiet=False)
            
            total_neurons = result.get('total_neurons', 0)
            already_cached = result.get('already_cached', 0)
            newly_cached = result.get('newly_cached', 0)
            total_connections = result.get('total_connections', 0)
            
            # Clear FNC's in-memory cache to free memory
            fnc._conn_df_cache = None
            fnc._conn_index = {}
            fnc._conn_index_post = {}
            
            # Also clear the module-level cache for this dataset
            safe_name = canonical_dataset_name(self.dataset).replace(':', '_').replace('.', '_')
            
            modules_to_check = ['coana', 'src.coana', 'src.comparison.coana']
            for mod_name in modules_to_check:
                try:
                    import sys
                    if mod_name in sys.modules:
                        mod = sys.modules[mod_name]
                        if hasattr(mod, '_FNC_CACHE'):
                            if safe_name in mod._FNC_CACHE:
                                mod._FNC_CACHE[safe_name] = {}
                except Exception:
                    pass
            
            del fnc
            import gc
            gc.collect()
            
            if total_neurons > 0:
                if newly_cached == 0:
                    self._log(f"Connection cache complete: {already_cached:,} neurons, {total_connections:,} connections")
                else:
                    self._log(f"Connection cache updated: +{newly_cached:,} neurons (total: {total_connections:,} connections)")
                return True
            else:
                self._log("WARNING: No neurons found in dataset")
                return False
            
        except Exception as e:
            self._log(f"WARNING: Could not ensure connection cache: {e}")
            return False
    
    def _normalize_query(
        self, 
        query: Union[str, int, List]
    ) -> Tuple[List, Optional[List[str]]]:
        """
        Normalize query input and detect nested list format.
        
        Supports multiple formats:
        - Single type/bodyId: 'Mi1' or 720575940610453042
        - List of types/bodyIds: ['Mi1', 'Tm3', 'aMe12']
        - Nested list with custom names: [['Group1', [id1, id2]], ['Group2', [id3]]]
        
        Args:
            query: Raw query input
            
        Returns:
            Tuple of (normalized_query, custom_group_names)
            - normalized_query: List of items to query
            - custom_group_names: List of custom names if nested format, else None
        """
        # Single item
        if isinstance(query, (str, int)):
            return [query], None
        
        # Empty list
        if not query:
            return [], None
        
        # Check for nested list format: [['name', [ids]], ...]
        # First element should be a list with 2 elements: [name, [ids]]
        if (isinstance(query, list) and 
            len(query) > 0 and 
            isinstance(query[0], list) and 
            len(query[0]) == 2 and 
            isinstance(query[0][0], str) and 
            isinstance(query[0][1], list)):
            
            # Nested format detected
            normalized = []
            group_names = []
            
            for item in query:
                if (isinstance(item, list) and 
                    len(item) == 2 and 
                    isinstance(item[0], str) and 
                    isinstance(item[1], list)):
                    group_name = item[0]
                    ids = item[1]
                    group_names.append(group_name)
                    normalized.append(ids)  # Keep as list for aggregation
                else:
                    # Invalid format - treat as single item
                    self._log(f"Warning: Invalid nested format item: {item}")
            
            return normalized, group_names
        
        # Simple list format
        return list(query), None
    
    def _parse_group_map_csv(self, csv_path: str) -> Tuple[List, List[str]]:
        """
        Parse group_map_csv file to construct query with custom group names.
        
        CSV format: columns 'group' and 'id_type_instance'
        Rows with the same 'group' value are grouped together.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Tuple of (normalized_query, custom_group_names)
        """
        import pandas as pd
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"group_map_csv not found: {csv_path}")
        
        self._log(f"Loading group map from: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Validate columns
        required_cols = ['group', 'id_type_instance']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"group_map_csv must have column '{col}'. Found: {list(df.columns)}")
        
        # Group by group name
        group_data = df.groupby('group', sort=False)['id_type_instance'].apply(list).to_dict()
        
        normalized = []
        group_names = []
        
        for group_name, identifiers in group_data.items():
            # Convert identifiers: if it looks like a bodyId (all digits), convert to int
            processed_ids = []
            for id_val in identifiers:
                id_str = str(id_val).strip()
                if id_str.isdigit():
                    processed_ids.append(int(id_str))
                else:
                    processed_ids.append(id_str)
            
            group_names.append(str(group_name))
            normalized.append(processed_ids)
        
        self._log(f"Loaded {len(group_names)} groups from CSV:")
        for name, ids in zip(group_names, normalized):
            n_count = len(ids) if isinstance(ids, list) else 1
            self._log(f"  {name}: {n_count} identifiers")
        
        return normalized, group_names

    def _load_custom_groups_from_mapping(
        self, mapping_path: str, dataset: str
    ) -> Tuple[List, List[str]]:
        """
        Load custom groups from a LabelMapper preset JSON (source side).
        
        The preset format is the LabelMapper overall JSON exported by the
        Settings tab's mapping presets:
            {"source_mapping": {
                "custom_label": ["grp1", "grp2"],
                "hemibrain:v1.2.1": [["aMe12", "aMe12_R"], ["aMe12_L"]]}}
        Each custom_label/std_label row becomes one group; the group members
        are the identifiers listed for the profiling dataset. Groups without
        any members in this dataset are dropped with a warning.
        
        Args:
            mapping_path: Path to the LabelMapper preset JSON.
            dataset: Profiling dataset; must appear as a key of the mapping.
        
        Returns:
            Tuple of (normalized_query, custom_group_names), each group
            being a list of bodyIds/types — same contract as
            _parse_group_map_csv.
        """
        import json as _json
        
        try:
            data = _json.loads(Path(mapping_path).read_text(encoding='utf-8'))
        except (OSError, ValueError) as e:
            raise ValueError(f"Could not read custom mapping file {mapping_path}: {e}")
        
        side = data.get('source_mapping') or {}
        labels = side.get('custom_label') or side.get('std_label') or []
        if not labels:
            self._log("Warning: mapping file has no source groups "
                      "(custom_label/std_label) — custom groups skipped")
            return [], []
        
        # The profiling dataset may appear under its exact name or the
        # normalized variant (colons/dots replaced by underscores).
        candidates = {dataset, canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')}
        ds_key = next(
            (k for k in side
             if k not in ('custom_label', 'std_label') and k in candidates),
            None
        )
        if ds_key is None:
            self._log(f"Warning: mapping file has no groups for dataset "
                      f"'{dataset}' — custom groups skipped")
            return [], []
        
        # Keep the per-dataset member lists (multi-dataset mode resolves each
        # group against its own dataset's members). Keys are normalized to the
        # dataset identifiers this comparer knows.
        self._mapping_ds_members = {}
        if self.is_multi_dataset:
            for side_key, rows in side.items():
                if side_key in ('custom_label', 'std_label'):
                    continue
                for known_ds in self.datasets:
                    if side_key in (known_ds, canonical_dataset_name(known_ds).replace(':', '_').replace('.', '_')):
                        processed_rows = []
                        for row in (rows or []):
                            if isinstance(row, str):
                                row = [row]
                            processed_rows.append([
                                int(str(v).strip()) if str(v).strip().isdigit() else str(v).strip()
                                for v in (row or [])
                            ])
                        self._mapping_ds_members[known_ds] = processed_rows
        
        normalized = []
        group_names = []
        group_rows = side.get(ds_key) or []
        for i, label in enumerate(labels):
            member_list = group_rows[i] if i < len(group_rows) else None
            if isinstance(member_list, str):
                member_list = [member_list]
            if not member_list:
                continue
            processed = []
            for id_val in member_list:
                id_str = str(id_val).strip()
                processed.append(int(id_str) if id_str.isdigit() else id_str)
            if not processed:
                continue
            group_names.append(str(label))
            normalized.append(processed)
        
        if not group_names:
            self._log(f"Warning: no groups with members in '{dataset}' "
                      f"found in mapping file")
            return [], []
        
        self._log(f"Loaded {len(group_names)} custom groups from mapping file:")
        for name, ids in zip(group_names, normalized):
            self._log(f"  {name}: {len(ids)} identifiers")
        
        return normalized, group_names

    @staticmethod
    def _looks_like_pattern(item: str) -> bool:
        """Whether a query item is likely a regex pattern rather than a
        literal type name (exact-type lookup is tried first, so a literal
        containing '.' still resolves as a type when one exists)."""
        return any(ch in item for ch in '*?[]()^$|+{}.')


    def _generate_query_name(self) -> str:
        """
        Generate a filesystem-safe name from the query.
        
        Returns:
            String like 'Mi1_Tm3_etc' or 'Mi1' for single query
        """
        if not self.query:
            return "unnamed"
        
        # Use custom group names if available
        if self._custom_group_names:
            first = str(self._custom_group_names[0])
        else:
            # Get first item, clean it
            first_item = self.query[0]
            if isinstance(first_item, list):
                first_item = first_item[0] if first_item else "group"
            first = str(first_item).replace('.*', '').replace('*', '')
        
        first = canonical_dataset_name(first).replace(':', '_').replace('.', '_').replace('/', '_')
        
        n_items = len(self._custom_group_names) if self._custom_group_names else len(self.query)
        
        if n_items == 1:
            return first
        else:
            return f"{first}_etc"
    
    def _get_output_path(self) -> Path:
        """Generate the full output path with timestamp.

        Run folders use the unique 4-char dataset labels (MCNS/HEMI/...);
        two versions of the same family gain a version suffix
        (MCNS_v1_0_MCNS_v0_9) — the same vocabulary as the Cross-Dataset
        Comparison tab's legends.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        datasets = list(getattr(self, 'datasets', None) or [])
        primary = getattr(self, 'dataset', None)
        if primary and not datasets:
            datasets = [primary]
        ds_part = ('_'.join(make_unique_dataset_labels(datasets))
                   if datasets else dataset_abbrev(primary))
        folder_name = (
            f"profiling_{ds_part}"
            f"_{self.query_name}_{timestamp}"
        )
        
        if self.output_dir:
            return Path(self.output_dir) / folder_name
        else:
            # Default to the project-local data folder so the layout stays
            # portable across machines.
            return (
                Path(__file__).resolve().parents[2]
                / 'local_data' / 'connectivity_profiling' / folder_name
            )
    
    def _map_query_item(self, item: Any, dataset: str) -> Any:
        """Map one query item (type / bodyId / pattern) into a dataset's naming.

        bodyIds and patterns pass through unchanged (patterns are expanded per
        dataset); type names resolve through the shared validity resolver
        (``comparison.type_resolver``) when the auto mapper is loaded.  Only
        an unambiguous one-target case renames the item here; valid splits
        are expanded (and conflicts dropped) by ``_mapped_query_for`` via
        ``_map_query_item_multi``.
        """
        if isinstance(item, int) or (isinstance(item, str) and item.isdigit()):
            return item
        item_str = str(item)
        if self._looks_like_pattern(item_str):
            return item_str
        targets, _status = self._map_query_item_multi(item_str, dataset)
        if len(targets) == 1 and targets[0] != item_str:
            return targets[0]
        return item_str

    def _map_query_item_multi(self, item_str: str, dataset: str
                              ) -> Tuple[List[str], str]:
        """Resolve one type item to its licensed target names and status.

        Unique renames/bridges yield one target; valid splits yield EVERY
        licensed branch; conflicts and an unavailable mapper yield ``[]``
        (fail closed — no raw same-name comparison); unmapped types keep
        the raw name as the counted long-tail fallback.  Every resolution
        is recorded on ``mapping_resolution_record``, and
        ``mapping_resolution_counts`` counts on a UNIQUE-resolution basis
        (one per distinct ``(dataset, item)`` — a repeated ``(dataset, item)``
        is not re-counted), so results can distinguish a real zero
        similarity from an unmapped or conflicted item without
        double-counting items resolved by more than one code path.
        """
        from .type_resolver import (
            STATUS_UNMAPPED, get_mapper_snapshot, expansion_targets,
            resolve_valid_targets,
        )
        if self._mapper_snapshot is None:
            self._mapper_snapshot = get_mapper_snapshot(self._type_mapper)
        if getattr(self, '_walk_caches', None) is None:
            # RES-7: one alias/bridge cache pair per comparison — the
            # auto-detected source namespace re-ran the get_type_bridges
            # BFS and get_alias_candidates walk on every (item x dataset).
            self._walk_caches = ({}, {})
        alias_cache, bridge_cache = self._walk_caches
        res = resolve_valid_targets(
            self._type_mapper, item_str, None, dataset,
            snapshot=self._mapper_snapshot,
            alias_cache=alias_cache, bridge_cache=bridge_cache)
        targets = list(expansion_targets(res))
        self.mapping_resolution_record.setdefault(dataset, {})[item_str] = {
            'status': res.status, 'targets': targets,
            'reason': res.reason,
        }
        resolution_key = (str(dataset), str(item_str))
        if resolution_key not in self._counted_query_resolutions:
            self._counted_query_resolutions.add(resolution_key)
            self.mapping_resolution_counts[res.status] = (
                self.mapping_resolution_counts.get(res.status, 0) + 1)
        if res.status == STATUS_UNMAPPED and targets:
            self.mapping_raw_fallback_used = True
        return targets, res.status

    def _mapped_query_for(self, dataset: str) -> List:
        """Query items (or custom-group members) mapped into a dataset's
        naming. Only meaningful in multi-dataset mode with a loaded mapper;
        otherwise the original query is returned unchanged.

        Valid splits EXPAND one-to-many into every licensed target;
        conflicted items are dropped for that dataset (fail closed, kept in
        ``mapping_resolution_record``); unmapped items keep their raw name.
        """
        if not self.is_multi_dataset or self._type_mapper is None:
            return self.query
        if self._custom_group_names:
            # Prefer the preset's per-dataset member lists (LabelMapper input);
            # fall back to mapping the primary dataset's members.
            members = getattr(self, '_mapping_ds_members', {}).get(dataset)
            if members is not None:
                return [list(m) for m in members]
            expanded_groups: List[list] = []
            for group in self.query:
                mapped_members: list = []
                for m in group:
                    for mapped in self._map_query_item_multi(str(m), dataset)[0]:
                        mapped_members.append(mapped)
                expanded_groups.append(mapped_members)
            return expanded_groups
        expanded_query: list = []
        for item in self.query:
            item_str = str(item)
            if (isinstance(item, int) or item_str.isdigit()
                    or self._looks_like_pattern(item_str)):
                expanded_query.append(item)
                continue
            expanded_query.extend(
                self._map_query_item_multi(item_str, dataset)[0])
        return expanded_query

    def _canonical_label(self, label: str, dataset: str) -> str:
        """Canonical merge key for a type label (shared resolver).

        Licensed renames key under the canonical target; a CONFLICTED type
        keys dataset-scoped (``dataset:raw``) so it cannot merge with
        another dataset's same-named anchor; unmapped/unavailable keep the
        raw name.  Falls back to the raw name when the mapper is off.
        """
        if self._type_mapper is not None:
            try:
                from .type_resolver import canonical_merge_key, get_mapper_snapshot
                if self._mapper_snapshot is None:
                    self._mapper_snapshot = get_mapper_snapshot(self._type_mapper)
                return canonical_merge_key(
                    self._type_mapper, label, dataset,
                    snapshot=self._mapper_snapshot,
                    cache=self._merge_key_cache).key
            except Exception:
                return label
        return label

    def _licensed_anchor_label(self, item_str: str, dataset: str) -> Optional[str]:
        """The single licensed label for a queried item in one dataset.

        Returns one label for a unique rename/bridge, an unmapped raw-name
        pass-through, or a same-namespace native name.  Returns ``None`` so
        the item CANNOT form an anchor when the resolver reports a CONFLICT
        (fail closed — never a raw same-name across datasets), an
        unavailable mapper, or a multi-target split (no arbitrary branch is
        chosen for an anchor).

        With auto type mapping DISABLED (no mapper), the raw name passes
        through unchanged — names are compared as-is, matching the
        documented ``use_auto_type_mapping=False`` behavior.
        """
        if self._type_mapper is None:
            return item_str
        targets, _status = self._map_query_item_multi(item_str, dataset)
        if len(targets) == 1:
            return targets[0]
        return None

    def _get_neurons_to_compare(
        self,
        dataset: Optional[str] = None,
        query: Optional[List] = None,
        custom_group_names: Optional[List[str]] = None,
        aggregation_level: Optional[str] = None,
    ) -> Dict[str, List]:
        """
        Get the neurons/groups to compare based on query and aggregation level.
        
        Handles:
        - Explicit groups (nested list / group_map_csv / LabelMapper preset):
          each group is one row, regardless of aggregation level
        - aggregation_level='type': bodyIds resolve to their type rows;
          exact types are one row per type; pattern items ('aMe.*') expand
          into their matched types, each an INDEPENDENT row; a coarse
          taxonomy label (cell class / cell type) that the dataset maps to
          several real types also expands into one row per real type
        - aggregation_level='bodyid': every individual bodyId is its own
          row (labels {bodyId}_{type}); patterns expand to types, then to
          their bodyIds
        - aggregation_level='custom': each flat query item is taken literally
          as one single-member group (no pattern expansion)
        
        Args:
            dataset: Dataset to resolve against (defaults to self.dataset).
            query: Query items to resolve (defaults to self.query).
            custom_group_names: Group names (defaults to self._custom_group_names).
        
        Returns:
            Dictionary mapping label -> list of identifiers (bodyIds or
            type names to resolve later).
        """
        dataset = dataset or self.dataset
        if query is None:
            query = self.query
        if custom_group_names is None:
            custom_group_names = self._custom_group_names
        effective_aggregation = aggregation_level or self.aggregation_level
        
        neurons = {}
        
        # If custom group names are set (nested list, CSV or LabelMapper)
        if custom_group_names:
            for idx, (group_name, ids) in enumerate(zip(custom_group_names, query)):
                # ids should be a list of bodyIds/types
                if isinstance(ids, list):
                    neurons[group_name] = ids
                else:
                    neurons[group_name] = [ids]
            return neurons
        
        if self.aggregation_level == 'custom':
            # No explicit group input: each flat item is its own group,
            # taken literally (patterns are NOT expanded — the user defines
            # the grouping explicitly).
            for item in query:
                try:
                    bid = int(item)
                    neurons[str(bid)] = [bid]
                except (ValueError, TypeError):
                    neurons[str(item)] = [str(item)]
            return neurons
        
        # Simple list format - original behavior
        # First, collect all numeric items (bodyIds) to do batch lookup
        bodyid_items = []
        type_items = []
        
        for item in query:
            try:
                bid = int(item)
                bodyid_items.append(bid)
            except (ValueError, TypeError):
                type_items.append(str(item))
        
        # Batch lookup types for all bodyIds at once (much more efficient)
        bodyid_to_type = {}
        if bodyid_items:
            self._log(f"Looking up types for {len(bodyid_items)} bodyIds...")
            bodyid_to_type = self.profiler.get_types_for_bodyids(bodyid_items, dataset)
        
        # Process bodyIds
        for bid in bodyid_items:
            ntype = bodyid_to_type.get(bid)
            if effective_aggregation == 'bodyid':
                # rows are individual neurons (tree-legend labels:
                # '{bodyId}_{instance}' / '{bodyId}_{type}_{L|R}')
                label = _body_id_display_label(dataset, bid, fallback_type=ntype)
                neurons[label] = [bid]
            elif ntype:
                if ntype not in neurons:
                    neurons[ntype] = []
                neurons[ntype].append(bid)
            else:
                # Unknown type, use bodyId as label
                neurons[str(bid)] = [bid]
        
        # Process type names/patterns
        for item_str in type_items:
            # 1) Coarse-taxonomy / fine-type expansion.  get_types_for_label
            #    groups matching rows by their real neuron type, so a cell
            #    class / cell type label (e.g. FAFB's 'circadian_clock') maps
            #    to several real types, each becoming an INDEPENDENT comparison
            #    row — mirroring the network tab's multi-column resolution.
            #    This must run before the union lookup in get_bodyids_for_type,
            #    which would otherwise collapse the whole label into a single
            #    row and trip the "Need at least 2 type profiles" guard.
            label_resolver = getattr(self.profiler, 'get_types_for_label', None)
            label_groups = (
                label_resolver(item_str, dataset) if label_resolver else {}
            )
            if label_groups:
                for real_type, ids in label_groups.items():
                    if effective_aggregation == 'bodyid':
                        for bid in ids:
                            neurons[_body_id_display_label(
                                dataset, bid, fallback_type=real_type)] = [bid]
                    else:
                        if real_type not in neurons:
                            neurons[real_type] = []
                        neurons[real_type].extend(
                            int(bid) if isinstance(bid, str) and bid.isdigit()
                            else bid for bid in ids)
                continue

            # 2) Regex pattern: expand into its matched types, each an
            #    INDEPENDENT row (get_types_for_label only matches an exact
            #    label, so patterns reach this branch untouched).
            if self._looks_like_pattern(item_str):
                matched_types = self.profiler.list_types(item_str, dataset)
                if matched_types:
                    for tname in matched_types:
                        tids = self.profiler.get_bodyids_for_type(tname, dataset)
                        if not tids:
                            continue
                        if effective_aggregation == 'bodyid':
                            for bid in tids:
                                neurons[_body_id_display_label(
                                    dataset, bid, fallback_type=tname)] = [bid]
                        else:
                            neurons[tname] = tids
                else:
                    # No exact type and no pattern match: keep the raw item
                    neurons[item_str] = [item_str]
                continue

            # 3) Union lookup: an exact type (or a NeuPrint dataset with no
            #    local taxonomy table).  All matching bodyIds form one row.
            body_ids = self.profiler.get_bodyids_for_type(item_str, dataset)
            if body_ids:
                if effective_aggregation == 'bodyid':
                    # every bodyId of the type is its own row
                    for bid in body_ids:
                        neurons[_body_id_display_label(
                            dataset, bid, fallback_type=item_str)] = [bid]
                else:
                    neurons[item_str] = body_ids
            else:
                # No exact type, no pattern, no taxonomy match: keep raw item
                neurons[item_str] = [item_str]
        
        return neurons
    
    def _extract_profiles_for_dataset(
        self,
        dataset: str,
        aggregation_level: Optional[str] = None,
    ) -> Tuple[Dict[str, ConnectivityProfile], Dict[Tuple[str, int], ConnectivityProfile]]:
        """
        Extract both type-aggregated profiles and individual bodyId profiles
        for ONE dataset (query names mapped into it in multi-dataset mode).
        
        Uses batch processing with the connection cache for efficiency.
        All profiles are built directly from the in-memory connection cache
        without individual disk I/O for each profile.
        
        Returns:
            Tuple of:
            - type_profiles: Dictionary mapping type_label -> aggregated ConnectivityProfile
            - bodyid_profiles: Dictionary mapping (type_label, bodyId) -> individual ConnectivityProfile
        """
        effective_aggregation = aggregation_level or self.aggregation_level
        query = self._mapped_query_for(dataset) if self.is_multi_dataset else self.query
        neurons = self._get_neurons_to_compare(
            dataset=dataset, query=query,
            custom_group_names=self._custom_group_names,
            aggregation_level=effective_aggregation,
        )  # {type_label: [bodyIds]}
        
        type_profiles = {}
        bodyid_profiles = {}  # Key: (type_label, bodyId)
        
        # Count total bodyIds for progress tracking
        total_bodyids = sum(len(ids) for ids in neurons.values())
        self._log(f"Extracting profiles for {len(neurons)} types/groups ({total_bodyids} total bodyIds)...")
        
        # Collect all unique bodyIds to extract
        all_bodyids = []
        bodyid_to_labels = {}  # Maps bodyId -> list of type labels
        
        for label, neuron_ids in neurons.items():
            for nid in neuron_ids:
                if isinstance(nid, int) or (isinstance(nid, str) and nid.isdigit()):
                    bid = int(nid) if isinstance(nid, str) else nid
                    all_bodyids.append(bid)
                    if bid not in bodyid_to_labels:
                        bodyid_to_labels[bid] = []
                    bodyid_to_labels[bid].append(label)
                else:
                    # Non-numeric (type name) - handle separately
                    all_bodyids.append(nid)
                    if nid not in bodyid_to_labels:
                        bodyid_to_labels[nid] = []
                    bodyid_to_labels[nid].append(label)
        
        # Use batch extraction with progress bar - builds profiles directly from connection cache
        # skip_profile_cache=True for large batches (>100) to avoid disk I/O overhead
        skip_cache = total_bodyids > 100
        all_profiles = self.profiler.get_profiles_batch(
            all_bodyids, 
            dataset, 
            force_refresh=False,
            skip_profile_cache=skip_cache,
            show_progress=self.verbose
        )
        
        # Organize profiles by type label
        for nid, profile in all_profiles.items():
            if profile is None:
                continue
            
            labels = bodyid_to_labels.get(nid, [])
            bid = profile.neuron_id if isinstance(profile.neuron_id, int) else nid
            
            for label in labels:
                bodyid_profiles[(label, bid)] = profile
        
        # Create aggregated type profiles
        if effective_aggregation == 'bodyid':
            # 'bodyid' aggregation: rows are individual neurons — labels from
            # _get_neurons_to_compare already embed the bodyId
            # ({bodyId}_{type}), so no mean-pooling is needed.
            self._log(f"Using bodyId aggregation: {len(bodyid_profiles)} individual rows")
            type_profiles = {}
            for (label, bid), profile in bodyid_profiles.items():
                type_profiles[label] = profile
        else:
            # Preferred: pool ALL adjacencies of each label's members straight
            # from the connection cache (no per-bodyId top-k truncation, so a
            # type profile represents the full connectivity of the type).
            conn_df = None
            try:
                conn_df = self.profiler._get_cached_conn_df(dataset)
            except Exception:
                conn_df = None
            pooled_done = False
            if (conn_df is not None and not conn_df.empty
                    and {'bodyId_pre', 'bodyId_post', 'type_pre', 'type_post',
                         'weight'} <= set(conn_df.columns)):
                bid_to_labels: Dict[Any, Any] = {}
                for label, neuron_ids in neurons.items():
                    for nid in neuron_ids:
                        key = int(nid) if isinstance(nid, str) and nid.isdigit() else nid
                        bid_to_labels.setdefault(key, []).append(label)
                try:
                    pooled = ProfileComparator.build_pooled_type_profiles(
                        conn_df, bid_to_labels, dataset=dataset,
                        min_weight=self.min_synapse_threshold,
                        normalize_types=lambda s: self.profiler._normalize_types_vectorized(
                            s, self.profiler.config.fuzzy_match))
                    type_profiles.update(pooled)
                    pooled_done = True
                    self._log(f"Aggregated type-level profiles by pooling ALL "
                              f"adjacencies ({len(pooled)} labels)")
                except Exception as e:
                    self._log(f"Pooled type profiling failed ({e}); falling back "
                              f"to bodyId profile aggregation")
            if not pooled_done:
                self._log("Aggregating type-level profiles...")
            for label, neuron_ids in tqdm(neurons.items(), desc="Aggregating types",
                                           disable=progress_bars_disabled(True, self.verbose), unit="type"):
                if label in type_profiles:
                    continue
                individual_profiles = []
                
                for nid in neuron_ids:
                    bid = int(nid) if isinstance(nid, str) and nid.isdigit() else nid
                    if (label, bid) in bodyid_profiles:
                        individual_profiles.append(bodyid_profiles[(label, bid)])
                
                # Create aggregated type profile
                if len(individual_profiles) == 1:
                    type_profiles[label] = individual_profiles[0]
                elif len(individual_profiles) > 1:
                    type_profiles[label] = self._aggregate_profiles_from_list(individual_profiles, label)
        
        self._log(f"Extracted {len(type_profiles)} type profiles, {len(bodyid_profiles)} bodyId profiles from {dataset}")
        return type_profiles, bodyid_profiles

    def _extract_all_profiles(self) -> Tuple[Dict[str, ConnectivityProfile], Dict[Tuple[str, int], ConnectivityProfile]]:
        """Extract profiles for the primary dataset (single-dataset mode)."""
        return self._extract_profiles_for_dataset(self.dataset)

    @staticmethod
    def _same_name_flag(
        per_ds: Dict[str, Tuple[str, ConnectivityProfile]],
        datasets: List[str],
    ) -> int:
        """Return 1 when every dataset has the same non-empty resolved name."""
        resolved_names = [
            '' if ds not in per_ds or per_ds[ds][0] is None
            else str(per_ds[ds][0]).strip()
            for ds in datasets
        ]
        return int(
            bool(resolved_names)
            and all(resolved_names)
            and len(set(resolved_names)) == 1
        )

    # ------------------------------------------------------------------
    # Inter-dataset comparisons (same queried neuron across datasets)
    # ------------------------------------------------------------------
    
    def _build_anchor_profiles(
        self, profiles_by_dataset: Dict[str, Dict[str, ConnectivityProfile]]
    ) -> Dict[str, Dict[str, Tuple[str, ConnectivityProfile]]]:
        """Map queried neurons/groups to their per-dataset profiles.

        Anchors are the QUERIED items (the "same neuron" concept across
        datasets):
        - custom groups: each group name is an anchor (members were mapped
          per dataset during extraction)
        - type names: anchor key is the canonical merge key (shared
          resolver); the per-dataset profile is the profile of the mapped
          name in that dataset
        - bodyIds: anchor is the bodyId; per-dataset profiles are the mapped
          type's profiles (the bodyId's type in the source dataset)
        - patterns: anchors are the matched types of the FIRST dataset

        Per-dataset labels come from ``_licensed_anchor_label``: a
        CONFLICTED type contributes no label in the dataset it conflicts
        toward (fail closed — it can never form an anchor by raw same-name),
        and a multi-target split contributes no label (no arbitrary branch),
        so an anchor only spans datasets with a genuine licensed match.

        Returns:
            Dict[anchor -> {dataset: (label_in_dataset, profile)}]; only
            anchors present in >= 2 datasets are kept.
        """
        anchors: Dict[str, Dict[str, Tuple[str, ConnectivityProfile]]] = {}
            
        if self._custom_group_names:
            for gname in self._custom_group_names:
                per_ds = {}
                for ds, profiles in profiles_by_dataset.items():
                    if gname in profiles:
                        per_ds[ds] = (gname, profiles[gname])
                if len(per_ds) >= 2:
                    anchors[gname] = per_ds
            return anchors
            
        for item in self.query:
            item_str = str(item)
            if item_str.isdigit():
                # bodyId: use its type (mapped per dataset) as the profile
                bid = int(item_str)
                per_ds = {}
                for ds, profiles in profiles_by_dataset.items():
                    ntype = (self.profiler.get_types_for_bodyids([bid], ds) or {}).get(bid)
                    if not ntype:
                        continue
                    licensed = self._licensed_anchor_label(ntype, ds)
                    candidate_labels = [licensed] if licensed is not None else []
                    if self.aggregation_level == 'bodyid':
                        # Must mirror the label the resolution step built for
                        # this bodyId (tree-legend display label).
                        candidate_labels.insert(0, _body_id_display_label(
                            ds, bid, fallback_type=ntype))
                    for label in candidate_labels:
                        if label in profiles:
                            per_ds[ds] = (label, profiles[label])
                            break
                if len(per_ds) >= 2:
                    anchors[item_str] = per_ds
            elif self._looks_like_pattern(item_str):
                # patterns: anchors = matched types in the first dataset
                matched = self.profiler.list_types(item_str, self.datasets[0])
                for tname in matched:
                    per_ds = {}
                    for ds, profiles in profiles_by_dataset.items():
                        label = self._licensed_anchor_label(tname, ds)
                        if label is not None and label in profiles:
                            per_ds[ds] = (label, profiles[label])
                    if len(per_ds) >= 2:
                        key = self._canonical_label(tname, self.datasets[0])
                        anchors[key] = per_ds
            else:
                # exact type name
                per_ds = {}
                for ds, profiles in profiles_by_dataset.items():
                    label = self._licensed_anchor_label(item_str, ds)
                    if label is not None and label in profiles:
                        per_ds[ds] = (label, profiles[label])
                if len(per_ds) >= 2:
                    key = self._canonical_label(item_str, self.datasets[0])
                    anchors[key] = per_ds
            
        return anchors
    
    def _compute_inter_dataset_matrices(
        self, anchor_profiles: Dict[str, Dict[str, Tuple[str, ConnectivityProfile]]]
    ) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
        """
        Compute per-anchor datasets × datasets similarity matrices using the
        homolog-finding backend algorithm (2-hop expanded partner types,
        resolved to canonical names through the shared validity-aware
        resolver — conflicts excluded, splits expanded — combined =
        0.5·jaccard + 0.5·rank).
    
        Returns:
            Dict[anchor -> {direction: {metric: DataFrame}}]
        """
        mapper = self._type_mapper
        directions = ['both', 'upstream', 'downstream'] if self.direction == 'both' else [self.direction]
        metrics = ['jaccard', 'weighted_jaccard', 'cosine', 'rank_corr', 'rank_corr_union']
        out: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}
            
        for anchor, per_ds in anchor_profiles.items():
            present = [ds for ds in self.datasets if ds in per_ds]
            n = len(present)
            matrices = {}
            for direction in directions:
                dir_name = 'overall' if direction == 'both' else direction
                metric_matrices = {m: np.full((n, n), np.nan) for m in metrics}
                for i, d1 in enumerate(present):
                    p1 = per_ds[d1][1]
                    for j, d2 in enumerate(present):
                        if i == j:
                            continue  # intra-dataset pair, not part of inter comparison
                        p2 = per_ds[d2][1]
                        if mapper is not None:
                            # Validity-aware canonicalization (shared
                            # resolver): conflicts excluded, splits expanded,
                            # statuses counted per run.
                            types_a = ProfileComparator._get_expanded_types_resolved(
                                p1, direction, mapper,
                                decision_cache=self._expansion_decision_cache,
                                status_counts_out=self._expansion_status_counts,
                            )
                            types_b = ProfileComparator._get_expanded_types_resolved(
                                p2, direction, mapper,
                                decision_cache=self._expansion_decision_cache,
                                status_counts_out=self._expansion_status_counts,
                            )
                            scores = self._compute_similarity_from_types(types_a, types_b)
                        else:
                            scores = ProfileComparator.combined_score(p1, p2, direction=direction)
                        metric_matrices['jaccard'][i, j] = scores.get('jaccard', 0.0)
                        metric_matrices['weighted_jaccard'][i, j] = scores.get('weighted_jaccard', 0.0)
                        metric_matrices['cosine'][i, j] = scores.get('cosine', 0.0)
                        
                        rank_val = scores.get('rank', np.nan)
                        metric_matrices['rank_corr'][i, j] = rank_val if not np.isnan(rank_val) else 0.0
                        # rank_corr_union = the RAW union-based rank — sign is
                        # meaningful, 0 = no monotonic relation (same semantics
                        # as the homolog results' rank_union column)
                        rank_union_val = scores.get('rank_union', np.nan)
                        metric_matrices['rank_corr_union'][i, j] = (
                            rank_union_val if not np.isnan(rank_union_val) else 0.0
                        )
                matrices[dir_name] = {
                    m: pd.DataFrame(metric_matrices[m], index=present, columns=present)
                    for m in metrics
                }
            out[anchor] = matrices
            
        return out

    def _aggregate_inter_dataset_matrices(
        self,
        inter_matrices: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
        anchor_profiles: Optional[Dict[str, Dict[str, Tuple[str, ConnectivityProfile]]]] = None,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Combine per-anchor inter-dataset scores into comparison heatmaps.

        The detailed inter-dataset calculation naturally produces one
        datasets × datasets matrix for each queried neuron.  That layout is
        useful for inspecting one anchor, but it does not scale when the
        profiling query contains many neurons.  Reports and overview
        visualizations use this companion layout instead:

        * rows: queried neuron/type anchors (bodyId anchors read
          ``'{bodyId}_{resolved type}'``)
        * columns: dataset pairs (for example ``dataset_a vs dataset_b``)
        * values: the selected similarity metric for that anchor/pair

        The original per-anchor matrices remain available to callers and are
        still exported as CSVs.  Missing anchors/pairs are kept as NaN so the
        CSV preserves the distinction between "not comparable" and a score
        of zero; renderers can decide how to display those cells.
        """
        if not inter_matrices:
            return {}

        pair_specs = list(combinations(self.datasets, 2))
        pair_labels = [f"{dataset_a} vs {dataset_b}" for dataset_a, dataset_b in pair_specs]
        anchors = list(inter_matrices.keys())
        anchor_row_labels = {
            anchor: self._anchor_row_label(anchor, (anchor_profiles or {}).get(anchor))
            for anchor in anchors
        }
        directions = (
            ['overall', 'upstream', 'downstream']
            if self.direction == 'both'
            else [self.direction]
        )
        metrics = [
            'jaccard', 'weighted_jaccard', 'cosine',
            'rank_corr', 'rank_corr_union',
        ]

        aggregate: Dict[str, Dict[str, pd.DataFrame]] = {}
        for direction in directions:
            if not any(direction in inter_matrices.get(anchor, {}) for anchor in anchors):
                continue
            available_metrics = [
                metric for metric in metrics
                if any(
                    metric in inter_matrices.get(anchor, {}).get(direction, {})
                    for anchor in anchors
                )
            ]
            metric_frames: Dict[str, pd.DataFrame] = {}
            for metric in available_metrics:
                rows = []
                for anchor in anchors:
                    anchor_matrix = (
                        inter_matrices.get(anchor, {})
                        .get(direction, {})
                        .get(metric)
                    )
                    values = []
                    for dataset_a, dataset_b in pair_specs:
                        value = np.nan
                        if anchor_matrix is not None:
                            try:
                                if dataset_a in anchor_matrix.index and dataset_b in anchor_matrix.columns:
                                    value = anchor_matrix.loc[dataset_a, dataset_b]
                                elif dataset_b in anchor_matrix.index and dataset_a in anchor_matrix.columns:
                                    value = anchor_matrix.loc[dataset_b, dataset_a]
                            except (KeyError, TypeError, ValueError):
                                value = np.nan
                        values.append(value)
                    rows.append(values)

                frame = pd.DataFrame(
                    rows,
                    index=[anchor_row_labels.get(a, a) for a in anchors],
                    columns=pair_labels, dtype=float)
                frame.index.name = 'neuron_type'
                metric_frames[metric] = frame
            aggregate[direction] = metric_frames

        return aggregate

    @staticmethod
    def _anchor_row_label(anchor, per_ds) -> str:
        """Aggregate-matrix row label for one anchor.

        bodyId anchors read ``'{bodyId}_{resolved type}'`` — the resolved
        per-dataset name from the first dataset that has one (dataset order
        first, then any). Named anchors (types/groups/patterns) pass through
        unchanged, and an unresolvable anchor keeps its raw text.
        """
        text = str(anchor)
        if not text.strip().isdigit():
            return text
        datasets_first = [entry for entry in (per_ds or {}).values()
                          if entry and str(entry[0] or "").strip()]
        if not datasets_first:
            return text
        return f"{text}_{datasets_first[0][0]}"
    
    def _extract_cross_dataset_profiles(
        self
    ) -> Tuple[Dict[str, Dict[str, ConnectivityProfile]], List[str], List[str]]:
        """
        Extract profiles for cross-dataset comparison.
        
        Returns:
            Tuple of:
            - profiles_by_dataset: Dict[dataset -> Dict[label -> ConnectivityProfile]]
            - row_labels: Labels for the first dataset (rows in N×M matrix)
            - col_labels: Labels for the second dataset (columns in N×M matrix)
        """
        if not self.is_cross_dataset or not self._cross_dataset_query:
            raise RuntimeError("_extract_cross_dataset_profiles called but not in cross-dataset mode")
        
        profiles_by_dataset: Dict[str, Dict[str, ConnectivityProfile]] = {}
        
        for dataset, neurons in self._cross_dataset_query.items():
            self._log(f"Extracting profiles from {dataset}...")
            
            # Optionally ensure connection cache is complete for this dataset
            # (full-dataset builds are opt-in; see ensure_cache_complete)
            if self.use_cache and self.ensure_cache_complete:
                self._ensure_connection_cache_complete_for_dataset(dataset)
            
            dataset_profiles = {}
            
            for neuron_query in neurons:
                # Resolve neuron query to bodyIds
                is_bodyid_query = isinstance(neuron_query, int) or str(
                    neuron_query).strip().isdigit()
                if isinstance(neuron_query, int):
                    # BodyId query
                    body_ids = [neuron_query]
                    label = str(neuron_query)
                elif is_bodyid_query:
                    body_ids = [int(neuron_query)]
                    label = neuron_query
                else:
                    # Type name or pattern
                    body_ids = self.profiler.get_bodyids_for_type(str(neuron_query), dataset)
                    label = str(neuron_query)

                if not body_ids:
                    self._log(f"  Warning: No bodyIds found for '{neuron_query}' in {dataset}")
                    continue

                # Get profiles for these bodyIds
                individual_profiles = []
                for bid in body_ids:
                    try:
                        profile = self.profiler.get_profile(bid, dataset)
                        if profile is not None:
                            individual_profiles.append(profile)
                    except Exception as e:
                        self._log(f"  Warning: Failed to get profile for {bid}: {e}")

                if not individual_profiles:
                    self._log(f"  Warning: No profiles extracted for '{neuron_query}'")
                    continue

                # BodyId queries carry their resolved type in the label so
                # the exported rows never read as a bare numeric id
                # (tree-legend rule, per dataset — each matrix axis below
                # belongs to exactly one dataset).
                if is_bodyid_query:
                    resolved_type = str(
                        getattr(individual_profiles[0], 'neuron_type', '') or ''
                    ).strip()
                    label = _body_id_display_label(
                        dataset, body_ids[0], fallback_type=resolved_type)
                
                # Aggregate profiles for this type
                if len(individual_profiles) == 1:
                    dataset_profiles[label] = individual_profiles[0]
                else:
                    dataset_profiles[label] = self._aggregate_profiles_from_list(
                        individual_profiles, label
                    )
            
            profiles_by_dataset[dataset] = dataset_profiles
            self._log(f"  Extracted {len(dataset_profiles)} profiles from {dataset}")
        
        # Get labels for row (first dataset) and column (second dataset)
        ds_list = list(self._cross_dataset_query.keys())
        row_labels = list(profiles_by_dataset.get(ds_list[0], {}).keys())
        col_labels = list(profiles_by_dataset.get(ds_list[1], {}).keys())
        
        return profiles_by_dataset, row_labels, col_labels
    
    def _ensure_connection_cache_complete_for_dataset(self, dataset: str) -> bool:
        """Ensure connection cache is complete for a specific dataset."""
        try:
            try:
                from ..coana import FindNeuronConnection
            except ImportError:
                try:
                    from coana import FindNeuronConnection
                except ImportError:
                    import sys
                    from pathlib import Path
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from coana import FindNeuronConnection
            
            fnc = FindNeuronConnection(
                dataset=dataset,
                use_cache=True,
                verbose_mode='simple',
                simple_fetch=True
            )
            
            result = fnc.build_connection_cache(batch_size=1000, quiet=not self.verbose)
            
            # Clear memory
            fnc._conn_df_cache = None
            del fnc
            import gc
            gc.collect()
            
            return True
        except Exception as e:
            self._log(f"WARNING: Could not ensure connection cache for {dataset}: {e}")
            return False
    
    def _compute_cross_dataset_similarity_matrices(
        self,
        profiles_by_dataset: Dict[str, Dict[str, ConnectivityProfile]],
        row_labels: List[str],
        col_labels: List[str],
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Compute N×M similarity matrices for cross-dataset comparison.
        
        Args:
            profiles_by_dataset: Dict[dataset -> Dict[label -> ConnectivityProfile]]
            row_labels: Labels for rows (first dataset)
            col_labels: Labels for columns (second dataset)
        
        Returns:
            Nested dictionary: {direction: {metric: DataFrame}}
        """
        ds_list = list(profiles_by_dataset.keys())
        profiles_a = profiles_by_dataset[ds_list[0]]
        profiles_b = profiles_by_dataset[ds_list[1]]
        
        n_rows = len(row_labels)
        n_cols = len(col_labels)
        
        directions = ['both', 'upstream', 'downstream'] if self.direction == 'both' else [self.direction]
        metrics = ['jaccard', 'weighted_jaccard', 'cosine', 'rank_corr', 'rank_corr_union']
        
        all_matrices = {}
        total_pairs = n_rows * n_cols
        
        # Get type mapper if enabled
        type_mapper = None
        if self.use_auto_type_mapping:
            type_mapper = get_type_mapper()
            if type_mapper._loaded:
                self._log(f"Using auto type mapping for cross-dataset comparison")
                self._log(f"  Partner types resolve via the shared validity-aware resolver (conflicts fail closed, splits expand, unmapped fall back to raw)")
            else:
                self._log(f"Warning: Auto type mapping requested but mapper not loaded")
                type_mapper = None
        
        for direction in directions:
            dir_name = 'overall' if direction == 'both' else direction
            
            # Initialize matrices for all metrics
            metric_matrices = {m: np.zeros((n_rows, n_cols)) for m in metrics}
            
            self._log(f"Computing cross-dataset {dir_name} similarity ({n_rows}×{n_cols}, {total_pairs} pairs)...")
            
            with tqdm(total=total_pairs, desc=f"  {dir_name}", disable=progress_bars_disabled(True, self.verbose)) as pbar:
                for i, row_label in enumerate(row_labels):
                    profile_a = profiles_a.get(row_label)
                    if profile_a is None:
                        pbar.update(n_cols)
                        continue
                    
                    for j, col_label in enumerate(col_labels):
                        profile_b = profiles_b.get(col_label)
                        if profile_b is None:
                            pbar.update(1)
                            continue
                        
                        # Use cross-dataset comparison with type standardization
                        if type_mapper is not None:
                            # Validity-aware canonicalization (shared
                            # resolver): conflicts excluded, splits expanded.
                            types_a = ProfileComparator._get_expanded_types_resolved(
                                profile_a, direction, type_mapper,
                                decision_cache=self._expansion_decision_cache,
                                status_counts_out=self._expansion_status_counts,
                            )
                            types_b = ProfileComparator._get_expanded_types_resolved(
                                profile_b, direction, type_mapper,
                                decision_cache=self._expansion_decision_cache,
                                status_counts_out=self._expansion_status_counts,
                            )
                            
                            # Compute metrics manually with standardized types
                            scores = self._compute_similarity_from_types(types_a, types_b)
                        else:
                            # Standard cross-dataset comparison
                            scores = ProfileComparator.combined_score(
                                profile_a, profile_b, direction=direction
                            )
                        
                        metric_matrices['jaccard'][i, j] = scores.get('jaccard', 0.0)
                        metric_matrices['weighted_jaccard'][i, j] = scores.get('weighted_jaccard', 0.0)
                        metric_matrices['cosine'][i, j] = scores.get('cosine', 0.0)
                        
                        
                        rank_val = scores.get('rank', np.nan)
                        metric_matrices['rank_corr'][i, j] = rank_val if not np.isnan(rank_val) else 0.0
                        
                        # rank_corr_union = the RAW union-based rank — sign is
                        # meaningful, 0 = no monotonic relation (same semantics
                        # as the homolog results' rank_union column)
                        rank_union_val = scores.get('rank_union', np.nan)
                        metric_matrices['rank_corr_union'][i, j] = (
                            rank_union_val if not np.isnan(rank_union_val) else 0.0
                        )
                        
                        pbar.update(1)
            
            # Convert to DataFrames
            all_matrices[dir_name] = {
                m: pd.DataFrame(metric_matrices[m], index=row_labels, columns=col_labels)
                for m in metrics
            }
        
        return all_matrices
    
    def _compute_similarity_from_types(
        self,
        types_a: Dict[str, float],
        types_b: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute similarity metrics from pre-computed type dictionaries.
        
        Used for cross-dataset comparison with type standardization.
        """
        from scipy.stats import spearmanr
        
        set_a = set(types_a.keys())
        set_b = set(types_b.keys())
        
        # Jaccard
        intersection = set_a & set_b
        union = set_a | set_b
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Cosine
        if not union:
            cosine = 0.0
        else:
            all_types = sorted(union)
            vec_a = np.array([types_a.get(t, 0.0) for t in all_types])
            vec_b = np.array([types_b.get(t, 0.0) for t in all_types])
            norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
            cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0
        
        # Weighted Jaccard: sum(min(w_a, w_b)) / sum(max(w_a, w_b)) over the
        # union — identical to the homolog scorer's weighted_jaccard
        if union:
            w_intersection = float(np.sum(np.minimum(vec_a, vec_b)))
            w_union = float(np.sum(np.maximum(vec_a, vec_b)))
            weighted_jaccard = w_intersection / w_union if w_union > 0 else 0.0
        else:
            weighted_jaccard = 0.0
        
        # Rank correlation on shared types
        shared_types = sorted(intersection)
        if len(shared_types) >= 3:
            weights_a = [types_a[t] for t in shared_types]
            weights_b = [types_b[t] for t in shared_types]
            if len(set(weights_a)) > 1 and len(set(weights_b)) > 1:
                try:
                    rank_corr, _ = spearmanr(weights_a, weights_b)
                    rank_corr = float(rank_corr) if not np.isnan(rank_corr) else np.nan
                except Exception:
                    rank_corr = np.nan
            else:
                rank_corr = np.nan
        else:
            rank_corr = np.nan
        
        # Rank correlation on the UNION of types (missing = 0.0) — the same
        # metric as the homolog backend's rank_union (batch_compare_cross_dataset),
        # so profiling and homolog report identical values for the same pair.
        union_types = sorted(union)
        if len(union_types) >= 3:
            weights_a_union = [types_a.get(t, 0.0) for t in union_types]
            weights_b_union = [types_b.get(t, 0.0) for t in union_types]
            if len(set(weights_a_union)) > 1 and len(set(weights_b_union)) > 1:
                try:
                    rank_union, _ = spearmanr(weights_a_union, weights_b_union)
                    rank_union = float(rank_union) if not np.isnan(rank_union) else np.nan
                except Exception:
                    rank_union = np.nan
            else:
                rank_union = np.nan
        else:
            rank_union = np.nan
        
        return {
            'jaccard': jaccard,
            'weighted_jaccard': weighted_jaccard,
            'cosine': cosine,
            'rank': rank_corr,
            'rank_union': rank_union,
        }

    def _aggregate_profiles_from_list(
        self,
        profiles: List[ConnectivityProfile],
        label: str
    ) -> Optional[ConnectivityProfile]:
        """
        Aggregate a list of profiles using mean pooling.
        
        Args:
            profiles: List of ConnectivityProfile objects
            label: Label for the aggregated profile
        
        Returns:
            Aggregated ConnectivityProfile or None
        """
        if not profiles:
            return None
        
        if len(profiles) == 1:
            return profiles[0]
        
        # Mean pooling of weights
        upstream_weights: Dict[str, List[float]] = {}
        downstream_weights: Dict[str, List[float]] = {}
        
        for profile in profiles:
            for partner, weight in profile.upstream_partners.items():
                if partner not in upstream_weights:
                    upstream_weights[partner] = []
                upstream_weights[partner].append(weight)
            
            for partner, weight in profile.downstream_partners.items():
                if partner not in downstream_weights:
                    downstream_weights[partner] = []
                downstream_weights[partner].append(weight)
        
        # Compute mean weights
        up_mean = {k: np.mean(v) for k, v in upstream_weights.items()}
        down_mean = {k: np.mean(v) for k, v in downstream_weights.items()}
        
        # Compute ranks from mean weights
        up_ranks = self._compute_ranks(up_mean)
        down_ranks = self._compute_ranks(down_mean)
        
        # Create aggregated profile
        return ConnectivityProfile(
            neuron_id=label,
            # Profiles can be aggregated while iterating over a dataset in
            # multi-dataset mode.  Use their source dataset rather than the
            # comparer’s first dataset, otherwise every aggregate is tagged
            # as the first dataset in the run.
            dataset=getattr(profiles[0], "dataset", None) or self.dataset,
            neuron_type=label,
            upstream_partners=up_mean,
            downstream_partners=down_mean,
            upstream_ranks=up_ranks,
            downstream_ranks=down_ranks,
            upstream_top_k=len(up_mean),
            downstream_top_k=len(down_mean),
            total_upstream_weight=sum(up_mean.values()),
            total_downstream_weight=sum(down_mean.values()),
            num_neurons_aggregated=len(profiles),
            actual_upstream_count=len(up_mean),
            actual_downstream_count=len(down_mean),
            unique_types_upstream=len(up_mean),
            unique_types_downstream=len(down_mean),
        )
    
    @staticmethod
    def _compute_ranks(weights: Dict[str, float]) -> Dict[str, int]:
        """Compute ranks from weights (higher weight = lower rank)."""
        if not weights:
            return {}
        sorted_partners = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return {partner: rank + 1 for rank, (partner, _) in enumerate(sorted_partners)}
    
    def _compute_similarity_matrices(
        self,
        profiles: Dict[str, ConnectivityProfile]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Compute pairwise similarity matrices for all profiles using ALL metrics.
        
        Outputs ALL metrics like FindHomologs.py:
        - jaccard: Set-based overlap (0-1)
        - cosine: Weight vector similarity (0-1)
        - rank_corr: Spearman correlation (-1 to 1)
        - rank_corr_union: Normalized rank correlation (0-1)
        
        Args:
            profiles: Dictionary mapping label -> ConnectivityProfile
        
        Returns:
            Nested dictionary: {direction: {metric: DataFrame}}
            - direction: 'overall', 'upstream', 'downstream'
            - metric: 'jaccard', 'cosine', 'rank_corr', 'rank_corr_union'
        """
        labels = sorted(profiles.keys())
        n = len(labels)
        
        directions = ['both', 'upstream', 'downstream'] if self.direction == 'both' else [self.direction]
        metrics = ['jaccard', 'weighted_jaccard', 'cosine', 'rank_corr', 'rank_corr_union']
        
        all_matrices = {}
        
        # Calculate total number of unique pairs to compute (for progress bar)
        total_pairs = n * (n - 1) // 2
        
        for direction in directions:
            dir_name = 'overall' if direction == 'both' else direction
            
            # Initialize matrices for all metrics
            metric_matrices = {m: np.zeros((n, n)) for m in metrics}
            
            self._log(f"Computing {dir_name} similarity matrices ({n}x{n}, {total_pairs} pairs, {len(metrics)} metrics)...")
            
            # Use progress bar for pairwise comparisons
            pair_count = 0
            with tqdm(total=total_pairs, desc=f"  {dir_name} similarities", disable=progress_bars_disabled(True, self.verbose)) as pbar:
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            # Self-similarity = 1.0 for all metrics
                            for m in metrics:
                                metric_matrices[m][i, j] = 1.0
                        elif i < j:
                            profile_a = profiles[labels[i]]
                            profile_b = profiles[labels[j]]
                            
                            scores = ProfileComparator.combined_score(
                                profile_a, profile_b, direction=direction
                            )
                            
                            # Extract all metrics
                            metric_matrices['jaccard'][i, j] = scores['jaccard']
                            metric_matrices['jaccard'][j, i] = scores['jaccard']
                            
                            metric_matrices['weighted_jaccard'][i, j] = scores['weighted_jaccard']
                            metric_matrices['weighted_jaccard'][j, i] = scores['weighted_jaccard']
                            
                            metric_matrices['cosine'][i, j] = scores['cosine']
                            metric_matrices['cosine'][j, i] = scores['cosine']
                            

                            
                            rank_val = scores['rank'] if not np.isnan(scores['rank']) else 0.0
                            metric_matrices['rank_corr'][i, j] = rank_val
                            metric_matrices['rank_corr'][j, i] = rank_val
                            
                            # rank_corr_union = RAW union-based rank (sign
                            # meaningful, 0 = neutral — same as the homolog
                            # rank_union)
                            rank_union_val = scores.get('rank_union', np.nan)
                            rank_union_val = rank_union_val if not np.isnan(rank_union_val) else 0.0
                            metric_matrices['rank_corr_union'][i, j] = rank_union_val
                            metric_matrices['rank_corr_union'][j, i] = rank_union_val
                            
                            pbar.update(1)
            
            # Convert to DataFrames
            all_matrices[dir_name] = {
                m: pd.DataFrame(metric_matrices[m], index=labels, columns=labels)
                for m in metrics
            }
        
        return all_matrices
        
        return all_matrices
    
    def _compute_bodyid_similarity_matrices(
        self,
        bodyid_profiles: Dict[Tuple[str, int], ConnectivityProfile]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Compute pairwise similarity matrices for bodyId-level profiles.

        Labels follow the tree-legend rule: '{bodyId}_{instance}' or
        '{bodyId}_{type}_{L|R}' (dataset-specific); the resolved comparison
        type label is the fallback so the axis never reads as a bare id.

        Args:
            bodyid_profiles: Dictionary mapping (type_label, bodyId) -> ConnectivityProfile

        Returns:
            Nested dictionary: {direction: {metric: DataFrame}}
        """
        # Sort by type, then bodyId
        keys = sorted(bodyid_profiles.keys(), key=lambda x: (x[0], x[1]))
        labels: List[str] = []
        seen_labels: Dict[str, Tuple[str, int]] = {}
        for key in keys:
            type_label, bid = key
            label = _body_id_display_label(
                bodyid_profiles[key].dataset, bid, fallback_type=type_label)
            if label in seen_labels and seen_labels[label] != key:
                # Same display label under two type groups: keep the axes
                # unique by re-appending the resolved type label.
                label = f"{label}_{type_label}"
            seen_labels[label] = key
            labels.append(label)
        n = len(labels)
        
        directions = ['both', 'upstream', 'downstream'] if self.direction == 'both' else [self.direction]
        metrics = ['jaccard', 'weighted_jaccard', 'cosine', 'rank_corr', 'rank_corr_union']
        
        all_matrices = {}
        
        # Calculate total number of unique pairs
        total_pairs = n * (n - 1) // 2
        
        for direction in directions:
            dir_name = 'overall' if direction == 'both' else direction
            
            metric_matrices = {m: np.zeros((n, n)) for m in metrics}
            
            self._log(f"Computing bodyId-level {dir_name} similarity matrices ({n}x{n}, {total_pairs} pairs)...")
            
            with tqdm(total=total_pairs, desc=f"  bodyId {dir_name}", disable=progress_bars_disabled(True, self.verbose)) as pbar:
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            for m in metrics:
                                metric_matrices[m][i, j] = 1.0
                        elif i < j:
                            profile_a = bodyid_profiles[keys[i]]
                            profile_b = bodyid_profiles[keys[j]]
                            
                            scores = ProfileComparator.combined_score(
                                profile_a, profile_b, direction=direction
                            )
                            
                            metric_matrices['jaccard'][i, j] = scores['jaccard']
                            metric_matrices['jaccard'][j, i] = scores['jaccard']

                            metric_matrices['weighted_jaccard'][i, j] = scores.get(
                                'weighted_jaccard', 0.0
                            )
                            metric_matrices['weighted_jaccard'][j, i] = scores.get(
                                'weighted_jaccard', 0.0
                            )


                            
                            metric_matrices['cosine'][i, j] = scores['cosine']
                            metric_matrices['cosine'][j, i] = scores['cosine']
                            
                            rank_val = scores['rank'] if not np.isnan(scores['rank']) else 0.0
                            metric_matrices['rank_corr'][i, j] = rank_val
                            metric_matrices['rank_corr'][j, i] = rank_val
                            
                            rank_union_val = scores.get('rank_union', np.nan)
                            rank_union_val = (
                                rank_union_val
                                if not np.isnan(rank_union_val) else 0.0
                            )
                            metric_matrices['rank_corr_union'][i, j] = rank_union_val
                            metric_matrices['rank_corr_union'][j, i] = rank_union_val
                            
                            pbar.update(1)
            
            all_matrices[dir_name] = {
                m: pd.DataFrame(metric_matrices[m], index=labels, columns=labels)
                for m in metrics
            }
        
        return all_matrices
    
    def _compute_type_avg_bodyid_matrices(
        self,
        bodyid_profiles: Dict[Tuple[str, int], ConnectivityProfile]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Compute type-level similarity matrices by averaging bodyId-level comparisons.
        
        For each pair of types (A, B):
        - If A == B (intra-type): average similarity across all bodyId pairs within type
        - If A != B (inter-type): average similarity across all bodyId pairs between types
        
        Args:
            bodyid_profiles: Dictionary mapping (type_label, bodyId) -> ConnectivityProfile
        
        Returns:
            Nested dictionary: {direction: {metric: DataFrame}}
        """
        # Group bodyIds by type
        type_bodyids: Dict[str, List[int]] = {}
        for (type_label, bid), profile in bodyid_profiles.items():
            if type_label not in type_bodyids:
                type_bodyids[type_label] = []
            type_bodyids[type_label].append(bid)
        
        type_labels = sorted(type_bodyids.keys())
        n = len(type_labels)
        
        directions = ['both', 'upstream', 'downstream'] if self.direction == 'both' else [self.direction]
        metrics = ['jaccard', 'weighted_jaccard', 'cosine', 'rank_corr', 'rank_corr_union']
        
        all_matrices = {}
        
        for direction in directions:
            dir_name = 'overall' if direction == 'both' else direction
            
            metric_matrices = {m: np.zeros((n, n)) for m in metrics}
            
            self._log(f"Computing type-avg-bodyId {dir_name} similarity matrices ({n}x{n})...")
            
            pbar = tqdm(total=n, desc=f"Type-avg-bodyId {dir_name}", disable=progress_bars_disabled(True, self.verbose))
            for i, type_a in enumerate(type_labels):
                for j, type_b in enumerate(type_labels):
                    if i == j:
                        # Intra-type: average similarity within type (excluding self-comparisons)
                        bids = type_bodyids[type_a]
                        if len(bids) < 2:
                            # Only one bodyId - no intra-type comparison possible
                            for m in metrics:
                                metric_matrices[m][i, j] = 1.0
                        else:
                            scores_list = {m: [] for m in metrics}
                            for idx_a, bid_a in enumerate(bids):
                                for bid_b in bids[idx_a+1:]:
                                    key_a = (type_a, bid_a)
                                    key_b = (type_a, bid_b)
                                    if key_a in bodyid_profiles and key_b in bodyid_profiles:
                                        scores = ProfileComparator.combined_score(
                                            bodyid_profiles[key_a],
                                            bodyid_profiles[key_b],
                                            direction=direction
                                        )
                                        scores_list['jaccard'].append(scores['jaccard'])
                                        scores_list['weighted_jaccard'].append(scores['weighted_jaccard'])
                                        scores_list['cosine'].append(scores['cosine'])
                                        scores_list['rank_corr'].append(
                                            scores['rank'] if not np.isnan(scores['rank']) else 0.0
                                        )
                                        scores_list['rank_corr_union'].append(
                                            scores.get('rank_union', np.nan)
                                            if not np.isnan(scores.get('rank_union', np.nan)) else 0.0
                                        )
                            
                            for m in metrics:
                                if scores_list[m]:
                                    metric_matrices[m][i, j] = np.mean(scores_list[m])
                                else:
                                    metric_matrices[m][i, j] = 1.0  # Default for single bodyId
                    elif i < j:
                        # Inter-type: average similarity between types
                        bids_a = type_bodyids[type_a]
                        bids_b = type_bodyids[type_b]
                        
                        scores_list = {m: [] for m in metrics}
                        for bid_a in bids_a:
                            for bid_b in bids_b:
                                key_a = (type_a, bid_a)
                                key_b = (type_b, bid_b)
                                if key_a in bodyid_profiles and key_b in bodyid_profiles:
                                    scores = ProfileComparator.combined_score(
                                        bodyid_profiles[key_a],
                                        bodyid_profiles[key_b],
                                        direction=direction
                                    )
                                    scores_list['jaccard'].append(scores['jaccard'])
                                    scores_list['weighted_jaccard'].append(scores['weighted_jaccard'])
                                    scores_list['cosine'].append(scores['cosine'])
                                    scores_list['rank_corr'].append(
                                        scores['rank'] if not np.isnan(scores['rank']) else 0.0
                                    )
                                    scores_list['rank_corr_union'].append(
                                        scores.get('rank_union', np.nan)
                                        if not np.isnan(scores.get('rank_union', np.nan)) else 0.0
                                    )
                        
                        for m in metrics:
                            if scores_list[m]:
                                avg_score = np.mean(scores_list[m])
                                metric_matrices[m][i, j] = avg_score
                                metric_matrices[m][j, i] = avg_score
                            else:
                                metric_matrices[m][i, j] = 0.0
                                metric_matrices[m][j, i] = 0.0
                
                pbar.update(1)
            
            pbar.close()
            
            all_matrices[dir_name] = {
                m: pd.DataFrame(metric_matrices[m], index=type_labels, columns=type_labels)
                for m in metrics
            }
        
        return all_matrices

    def _save_results(
        self,
        type_profiles: Dict[str, ConnectivityProfile],
        bodyid_profiles: Dict[Tuple[str, int], ConnectivityProfile],
        type_matrices: Dict[str, Dict[str, pd.DataFrame]],
        bodyid_matrices: Dict[str, Dict[str, pd.DataFrame]],
        type_avg_matrices: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, List[str]]:
        """
        Save results to output directory with separated type-level and bodyId-level outputs.
        
        Output Structure:
            {output_dir}/profiling_{query_name}_{timestamp}/
            ├── parameters.json
            ├── README.txt
            ├── profiles/
            │   ├── individual/          # Individual bodyId profiles
            │   └── aggregated/          # Type-aggregated profiles
            ├── type_level/
            │   ├── results/             # Type-aggregated similarity matrices
            │   └── visualization/       # Type-level heatmaps
            └── bodyid_level/
                ├── results/             # BodyId similarity matrices + type-avg-bodyId matrices
                └── visualization/       # BodyId heatmaps + type-avg heatmaps
        
        Args:
            type_profiles: Dictionary mapping type_label -> aggregated ConnectivityProfile
            bodyid_profiles: Dictionary mapping (type_label, bodyId) -> individual ConnectivityProfile
            type_matrices: Type-level similarity matrices (aggregated profiles)
            bodyid_matrices: BodyId-level similarity matrices
            type_avg_matrices: Type-level matrices from averaged bodyId comparisons
        
        Returns:
            Dictionary with lists of saved file paths
        """
        output_path = self._get_output_path()
        output_path.mkdir(parents=True, exist_ok=True)
        self._log(f"📁 Output folder: {output_path}")
        
        # Create main directory structure
        profiles_dir = output_path / 'profiles'
        individual_dir = profiles_dir / 'individual'
        aggregated_dir = profiles_dir / 'aggregated'
        
        type_level_dir = output_path / 'type_level'
        type_results_dir = type_level_dir / 'results'
        type_viz_dir = type_level_dir / 'visualization'
        
        bodyid_level_dir = output_path / 'bodyid_level'
        bodyid_results_dir = bodyid_level_dir / 'results'
        bodyid_viz_dir = bodyid_level_dir / 'visualization'
        
        # Create all directories
        for d in [individual_dir, aggregated_dir, type_results_dir, type_viz_dir, 
                  bodyid_results_dir, bodyid_viz_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        saved_files = {
            'matrices_saved': [], 
            'heatmaps_generated': [], 
            'profiles_saved': [],
            'output_path': str(output_path)
        }
        
        metrics_list = ['jaccard', 'weighted_jaccard', 'cosine', 'rank_corr', 'rank_corr_union']
        directions_available = list(type_matrices.keys())
        
        # Save parameters
        params = {
            'query': self.query if not self._custom_group_names else 
                      [{'group': n, 'ids': ids} for n, ids in zip(self._custom_group_names, self.query)],
            'custom_group_names': self._custom_group_names,
            'group_map_csv': self.group_map_csv,
            'custom_mapping_file': self.custom_mapping_file,
            'aggregation_level': self.aggregation_level,
            'query_name': self.query_name,
            'dataset': self.dataset,
            'top_k': self.top_k,
            'top_m': self.top_m,
            'min_synapse_threshold': self.min_synapse_threshold,
            'direction': self.direction,
            'metrics_computed': metrics_list,
            'directions_computed': directions_available,
            'n_type_profiles': len(type_profiles),
            'n_bodyid_profiles': len(bodyid_profiles),
            'type_labels': sorted(type_profiles.keys()),
            'timestamp': datetime.now().isoformat(),
        }
        params.update(auto_mapping_result_metadata(
            requested=self.use_auto_type_mapping,
            type_mapper=self._type_mapper,
        ))

        with open(output_path / 'parameters.json', 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, default=str)
        
        # Save README
        query_display = self._custom_group_names if self._custom_group_names else self.query
        readme_lines = [
            "Connectivity Profile Comparison Results",
            "=" * 40,
            f"Query: {query_display}",
            f"Dataset: {self.dataset}",
            f"Type Profiles: {len(type_profiles)}",
            f"BodyId Profiles: {len(bodyid_profiles)}",
            f"Metrics: {', '.join(metrics_list)}",
            f"Directions: {', '.join(directions_available)}",
            "",
            "Output Structure:",
            "├── parameters.json",
            "├── README.txt",
            "├── profiles/",
            "│   ├── individual/          # Individual bodyId profiles (connectivity)",
            "│   └── aggregated/          # Type-aggregated profiles",
            "├── type_level/",
            "│   ├── results/             # Type-aggregated similarity matrices",
            "│   └── visualization/       # Type-level heatmaps",
            "└── bodyid_level/",
            "    ├── results/",
            "    │   ├── bodyid_similarity_{metric}_{direction}.csv",
            "    │   └── type_avg_bodyid_similarity_{metric}_{direction}.csv",
            "    └── visualization/",
            "        ├── heatmap_bodyid_{direction}_{metric}.html",
            "        └── heatmap_type_avg_{direction}_{metric}.html",
            "",
            "Notes:",
            "- type_level: Compares aggregated (mean-pooled) type profiles",
            "- bodyid_level/bodyid_*: Direct bodyId-to-bodyId comparisons",
            "- bodyId axes read '{bodyId}_{instance}' (NeuPrint datasets) or",
            "  '{bodyId}_{type}_{L|R}' (FAFB/BANC); the resolved comparison type",
            "  is the fallback when the neuron table has no metadata",
            "- bodyid_level/type_avg_*: Type similarities averaged from bodyId pairs",
            "  (diagonal = intra-type avg, off-diagonal = inter-type avg)",
            "- report.html: Overall report linking every metric and heatmap",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        
        with open(output_path / 'README.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(readme_lines))
        
        # === Save Type-Level Results ===
        self._log("Saving type-level results...")
        for direction, metric_matrices in type_matrices.items():
            for metric, matrix in metric_matrices.items():
                csv_path = type_results_dir / f'type_similarity_{metric}_{direction}.csv'
                matrix.to_csv(csv_path)
                saved_files['matrices_saved'].append(str(csv_path))
        
        # === Save BodyId-Level Results ===
        self._log("Saving bodyId-level results...")
        for direction, metric_matrices in bodyid_matrices.items():
            for metric, matrix in metric_matrices.items():
                csv_path = bodyid_results_dir / f'bodyid_similarity_{metric}_{direction}.csv'
                matrix.to_csv(csv_path)
                saved_files['matrices_saved'].append(str(csv_path))
        
        # === Save Type-Avg-BodyId Results ===
        self._log("Saving type-avg-bodyId results...")
        for direction, metric_matrices in type_avg_matrices.items():
            for metric, matrix in metric_matrices.items():
                csv_path = bodyid_results_dir / f'type_avg_bodyid_similarity_{metric}_{direction}.csv'
                matrix.to_csv(csv_path)
                saved_files['matrices_saved'].append(str(csv_path))
        
        self._log(f"Saved {len(saved_files['matrices_saved'])} similarity matrices")
        
        # === Save Individual BodyId Profiles ===
        self._log("Saving individual connectivity profiles...")
        for (type_label, bid), profile in bodyid_profiles.items():
            safe_label = _bodyid_profile_stem(bid, type_label).replace(
                '/', '_').replace(':', '_').replace('.', '_')
            
            profile_data = {
                'bodyId': bid,
                'type': type_label,
                # Resolved type name, matching the to_dict()-based JSONs
                # written by the multi-/cross-dataset paths (now backfilled
                # by the profiler for bodyId-queried profiles).
                'neuron_type': profile.neuron_type,
                'dataset': profile.dataset,
                'instance': _body_id_instance_name(profile.dataset, bid),
                'upstream_partners': {str(k): float(v) for k, v in profile.upstream_partners.items()},
                'downstream_partners': {str(k): float(v) for k, v in profile.downstream_partners.items()},
                'upstream_ranks': {str(k): int(v) for k, v in (profile.upstream_ranks or {}).items()},
                'downstream_ranks': {str(k): int(v) for k, v in (profile.downstream_ranks or {}).items()},
                'total_upstream_weight': float(profile.total_upstream_weight),
                'total_downstream_weight': float(profile.total_downstream_weight),
            }
            
            profile_path = individual_dir / f'{safe_label}_profile.json'
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, indent=2)
            saved_files['profiles_saved'].append(str(profile_path))
        
        # === Save Aggregated Type Profiles ===
        for type_label, profile in type_profiles.items():
            safe_label = str(type_label).replace('/', '_').replace(':', '_').replace('.', '_')
            
            profile_data = {
                'type': type_label,
                'dataset': profile.dataset,
                'num_neurons_aggregated': profile.num_neurons_aggregated,
                'upstream_partners': {str(k): float(v) for k, v in profile.upstream_partners.items()},
                'downstream_partners': {str(k): float(v) for k, v in profile.downstream_partners.items()},
                'upstream_ranks': {str(k): int(v) for k, v in (profile.upstream_ranks or {}).items()},
                'downstream_ranks': {str(k): int(v) for k, v in (profile.downstream_ranks or {}).items()},
                'total_upstream_weight': float(profile.total_upstream_weight),
                'total_downstream_weight': float(profile.total_downstream_weight),
            }
            
            profile_path = aggregated_dir / f'{safe_label}_profile.json'
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, indent=2)
            saved_files['profiles_saved'].append(str(profile_path))
        
        self._log(f"Saved {len(saved_files['profiles_saved'])} connectivity profiles")
        
        # === Generate Heatmaps ===
        if self.generate_heatmaps:
            # Type-level heatmaps
            self._log("Generating type-level heatmaps...")
            self._generate_heatmaps_vispath(type_matrices, type_viz_dir, saved_files, prefix='type')
            
            # BodyId-level heatmaps  
            self._log("Generating bodyId-level heatmaps...")
            self._generate_heatmaps_vispath(bodyid_matrices, bodyid_viz_dir, saved_files, prefix='bodyid')
            
            # Type-avg-bodyId heatmaps
            self._log("Generating type-avg-bodyId heatmaps...")
            self._generate_heatmaps_vispath(type_avg_matrices, bodyid_viz_dir, saved_files, prefix='type_avg')

        # Keep one report entry point for the single-dataset workflow too.
        # It is still useful when heatmaps are disabled because all exported
        # metric matrices remain linked from the report.
        report_path = self._generate_single_dataset_report(
            output_path, type_matrices, bodyid_matrices, type_avg_matrices
        )
        saved_files['report_path'] = str(report_path)
        
        return saved_files

    @classmethod
    def _metric_display_name(cls, metric: str) -> str:
        """Return the reader-facing label used by profiling reports."""
        return {
            'jaccard': 'Jaccard Similarity',
            'weighted_jaccard': 'Weighted Jaccard Similarity',
            'cosine': 'Cosine Similarity',
            'rank_corr': 'Rank Correlation',
            'rank_corr_union': 'Rank Correlation (Union)',
            'combined': 'Combined Score',
        }.get(metric, str(metric).replace('_', ' ').title())

    @classmethod
    def _direction_display_name(cls, direction: str) -> str:
        """Use ``Overall`` for the both-directions matrix in reader-facing UI."""
        return {
            'overall': 'Overall',
            'upstream': 'Upstream',
            'downstream': 'Downstream',
        }.get(direction, str(direction).replace('_', ' ').title())

    @classmethod
    def _direction_note(cls, direction: str) -> str:
        return {
            'overall': 'Upstream + downstream connectivity',
            'upstream': 'Presynaptic input connectivity',
            'downstream': 'Postsynaptic output connectivity',
        }.get(direction, 'Connectivity profile similarity')

    def _report_directions(self) -> List[str]:
        """Return the direction groups that belong in the report."""
        return list(self._REPORT_DIRECTIONS) if self.direction == 'both' else [self.direction]

    @staticmethod
    def _report_css() -> str:
        """Return the self-contained report stylesheet."""
        return report_kit.report_css()

    @staticmethod
    def _report_script() -> str:
        """Return the small tab/resize controller used by report.html."""
        return report_kit.report_script()

    @staticmethod
    def _cluster_heatmap_matrix(matrix: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """Apply the same Ward/Euclidean ordering used by VisPath."""
        return report_kit.cluster_heatmap_matrix(matrix)

    @classmethod
    def _plotly_heatmap_fragment(
        cls,
        matrix: pd.DataFrame,
        title: str,
        metric: str,
        x_title: str,
        y_title: str,
        include_plotlyjs: bool = False,
        square_cells: bool = False,
    ) -> Tuple[Optional[str], bool]:
        """Render one clustered Plotly heatmap fragment without cell labels.

        ``square_cells`` is used for intra-dataset similarity matrices, where
        the row and column dimensions represent the same set of items.
        """
        style = report_kit.metric_style(metric, cls._metric_display_name(metric))
        return report_kit.plotly_heatmap_fragment(
            matrix=matrix,
            title=title,
            style=style,
            x_title=x_title,
            y_title=y_title,
            include_plotlyjs=include_plotlyjs,
            square_cells=square_cells,
        )

    def _append_report_tab_group(
        self,
        lines: List[str],
        group_id: str,
        tabs: List[Tuple[str, str]],
        render_panel: Any,
        panel_class: str = 'tab-panel',
    ) -> None:
        """Append a reusable button/panel tab group to the report."""
        report_kit.append_report_tab_group(
            lines, group_id, tabs, render_panel, panel_class=panel_class)

    def _append_report_heatmap(
        self,
        lines: List[str],
        output_path: Path,
        matrix: Optional[pd.DataFrame],
        heading: str,
        title: str,
        metric: str,
        x_title: str,
        y_title: str,
        csv_rel: Optional[str],
        vispath_rel: Optional[str],
        plotly_state: Dict[str, bool],
        square_cells: bool = False,
    ) -> None:
        """Append one report card with a Plotly heatmap and source links."""
        style = report_kit.metric_style(
            metric, self._metric_display_name(metric))
        report_kit.append_report_heatmap(
            lines=lines,
            output_path=output_path,
            matrix=matrix,
            heading=heading,
            title=title,
            style=style,
            x_title=x_title,
            y_title=y_title,
            csv_rel=csv_rel,
            vispath_rel=vispath_rel,
            plotly_state=plotly_state,
            square_cells=square_cells,
        )

    def _append_report_metric_grid(
        self,
        lines: List[str],
        output_path: Path,
        metric_matrices: Optional[Dict[str, pd.DataFrame]],
        title_prefix: str,
        csv_paths: Dict[str, str],
        vispath_paths: Dict[str, str],
        x_title: str,
        y_title: str,
        plotly_state: Dict[str, bool],
        square_cells: bool = False,
    ) -> None:
        """Append six metric cards in a responsive two-column grid."""
        report_kit.append_report_metric_grid(
            lines, output_path, metric_matrices, title_prefix,
            csv_paths, vispath_paths, x_title, y_title, plotly_state,
            styles=self._report_metric_styles(),
            square_cells=square_cells,
        )

    def _generate_single_dataset_report(
        self,
        output_path: Path,
        type_matrices: Dict[str, Dict[str, pd.DataFrame]],
        bodyid_matrices: Dict[str, Dict[str, pd.DataFrame]],
        type_avg_matrices: Dict[str, Dict[str, pd.DataFrame]],
    ) -> Path:
        """Create the tabbed single-dataset report with clustered heatmaps."""
        from html import escape

        sections = [
            ('Type-level', type_matrices, 'type', 'type_similarity'),
            ('BodyId-level', bodyid_matrices, 'bodyid', 'bodyid_similarity'),
            ('Type-average BodyId', type_avg_matrices, 'type_avg', 'type_avg_bodyid_similarity'),
        ]
        directions = self._report_directions()
        dataset_label = str(self.dataset)
        lines = [
            '<!DOCTYPE html>',
            "<html><head><meta charset='utf-8'>",
            f"<title>Connectivity Profiling Report — {escape(str(self.query_name))}</title>",
            self._report_css(),
            '</head><body><main class="report-shell">',
            '<header class="report-hero">',
            '<div class="report-kicker">DROCAT · Connectivity profiling</div>',
            '<h1 class="report-title">Connectivity profile report</h1>',
            '<p class="report-subtitle">Six similarity metrics, Ward-clustered to match '
            'the VisPath heatmap ordering. Use the VisPath editor links when you need '
            'to change clustering or inspect the source matrix.</p>',
            '<div class="report-meta">',
            f"<div class='meta-chip'><span>Dataset</span><strong>{escape(dataset_label)}</strong></div>",
            f"<div class='meta-chip'><span>Aggregation</span><strong>{escape(str(self.aggregation_level))}</strong></div>",
            f"<div class='meta-chip'><span>Query</span><strong>{escape(self._format_query_for_log(self.query))}</strong></div>",
            '</div>',
            f"<p class='report-note'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
            "Overall means upstream + downstream; metric values remain available on hover.</p>",
            '</header>',
        ]
        plotly_state = {'include_plotlyjs': True}

        def render_intra(_key: str, panel_id: str) -> None:
            lines.append(
                '<div class="section-card"><h2 class="section-heading">Intra-dataset</h2>'
                '<p class="section-summary">Similarity among the queried neurons, '
                'types, bodyIds, or custom groups in the selected dataset.</p>'
            )

            def render_dataset(_dataset_key: str, dataset_panel_id: str) -> None:
                def render_direction(direction: str, _direction_panel_id: str) -> None:
                    direction_label = self._direction_display_name(direction)
                    lines.append(
                        f"<div class='direction-intro'><h3 class='direction-title'>{escape(direction_label)}</h3>"
                        f"<div class='direction-note'>{escape(self._direction_note(direction))}</div></div>"
                    )
                    for level_title, matrices, prefix, csv_prefix in sections:
                        lines.append(
                            f"<section class='level-block'><div class='level-heading'>"
                            f"<h3>{escape(level_title)}</h3>"
                            "<span>2 cards per row · 5 metrics</span></div>"
                        )
                        if not matrices:
                            lines.append(
                                "<div class='heatmap-empty'>This profile level was "
                                "not computed for this run.</div></section>"
                            )
                            continue
                        metric_matrices = (matrices or {}).get(direction, {})
                        csv_paths = {}
                        vispath_paths = {}
                        for metric in self._REPORT_METRICS:
                            csv_name = f'{csv_prefix}_{metric}_{direction}.csv'
                            csv_paths[metric] = (
                                f'{prefix}_level/results/{csv_name}'
                                if prefix != 'type_avg'
                                else f'bodyid_level/results/{csv_name}'
                            )
                            vispath_paths[metric] = (
                                f'bodyid_level/visualization/heatmap_type_avg_{direction}_{metric}.html'
                                if prefix == 'type_avg'
                                else f'{prefix}_level/visualization/heatmap_{prefix}_{direction}_{metric}.html'
                            )
                        self._append_report_metric_grid(
                            lines, output_path, metric_matrices,
                            f'{level_title} · {direction_label}',
                            csv_paths, vispath_paths,
                            'Neuron / type', 'Neuron / type', plotly_state,
                            square_cells=True,
                        )
                        lines.append('</section>')

                self._append_report_tab_group(
                    lines,
                    f'{dataset_panel_id}-directions',
                    [(direction, self._direction_display_name(direction)) for direction in directions],
                    render_direction,
                    panel_class='tab-panel direction-panel',
                )

            self._append_report_tab_group(
                lines,
                f'{panel_id}-datasets',
                [(dataset_label, dataset_label)],
                render_dataset,
            )
            lines.append('</div>')

        self._append_report_tab_group(
            lines,
            'report-sections',
            [('intra', 'Intra-dataset')],
            render_intra,
        )
        lines.extend(['</main>', self._report_script(), '</body></html>'])
        report_path = output_path / 'report.html'
        report_path.write_text('\n'.join(lines), encoding='utf-8')
        self._log(f'Overall report (clustered metric tabs): {report_path}')
        return report_path
    
    def _generate_heatmaps_vispath(
        self,
        matrices: Dict[str, Dict[str, pd.DataFrame]],
        viz_dir: Path,
        saved_files: Dict[str, List[str]],
        prefix: str = ''
    ):
        """
        Generate heatmaps using VisualizePath with native clustering support.

        Creates one heatmap per (direction x metric) combination, matching
        the report.html color conventions via the shared report_kit.
        """
        styles = self._report_metric_styles()
        prefix_display = {
            'type': 'Type-Level',
            'bodyid': 'BodyId-Level',
            'type_avg': 'Type-Avg-BodyId',
        }.get(prefix, prefix.title() if prefix else '')
        replaced = report_kit.generate_standalone_heatmaps(
            matrices, viz_dir, styles,
            filename_builder=(
                lambda direction, metric_key:
                f'heatmap_{prefix}_{direction}_{metric_key}.html' if prefix
                else f'heatmap_{direction}_{metric_key}.html'),
            group_display=self._direction_display_name,
            vispath_title=(
                lambda direction, metric_key, gd:
                f"{prefix_display} {styles[metric_key].display_name} - {gd}"
                if prefix_display
                else f"{styles[metric_key].display_name} - {gd}"),
            fallback_title=(
                lambda direction, metric_key, gd:
                f"Connectivity Profile Similarity - "
                f"{prefix.replace('_', ' ').title()} {gd} - "
                f"{metric_key.replace('_', ' ').title()}"
                if prefix else
                f"Connectivity Profile Similarity - {gd} - "
                f"{metric_key.replace('_', ' ').title()}"),
            tqdm_desc=f"Generating {prefix_display} heatmaps",
            show_figures=self.show_figures,
            verbose=self.verbose,
            saved_files=saved_files,
            log=self._log,
        )
        if replaced:
            self._log(
                f"Heatmap visualization: replaced {replaced} non-finite "
                "similarity values with 0; CSV matrices retain the original values."
            )

    def _generate_heatmaps_fallback(
        self,
        matrices: Dict[str, Dict[str, pd.DataFrame]],
        viz_dir: Path,
        saved_files: Dict[str, List[str]],
        prefix: str = ''
    ):
        """Fallback heatmap generation using interactive_heatmap module."""
        report_kit.generate_heatmaps_fallback(
            matrices, viz_dir, self._report_metric_styles(),
            filename_builder=(
                lambda direction, metric_key:
                f'heatmap_{prefix}_{direction}_{metric_key}.html' if prefix
                else f'heatmap_{direction}_{metric_key}.html'),
            group_display=self._direction_display_name,
            fallback_title=(
                lambda direction, metric_key, gd:
                f"Connectivity Profile Similarity - "
                f"{prefix.replace('_', ' ').title()} {gd} - "
                f"{metric_key.replace('_', ' ').title()}"
                if prefix else
                f"Connectivity Profile Similarity - {gd} - "
                f"{metric_key.replace('_', ' ').title()}"),
            show_figures=self.show_figures,
            verbose=self.verbose,
            saved_files=saved_files,
            log=self._log,
        )

    def run(self) -> Dict[str, Any]:
        """
        Run the connectivity profile comparison analysis.
        
        Performs both type-level and bodyId-level comparisons:
        
        Output Structure:
            1. Type-Level: Compares aggregated (mean-pooled) type profiles
            2. BodyId-Level (if not skipped via skip_bodyId_level parameter):
               - Direct bodyId-to-bodyId similarity (labels: {bodyId}_{type})
               - Type-avg-bodyId: Type similarities averaged from bodyId pairs
                 (diagonal = intra-type avg, off-diagonal = inter-type avg)
        
        For cross-dataset mode (query is dict), generates N×M similarity matrices
        comparing types from the first dataset (rows) against the second dataset (columns).
        BodyId-level comparison is skipped in cross-dataset mode.
        
        Returns:
            Dictionary with results summary including:
            - n_type_profiles: Number of type profiles
            - n_bodyid_profiles: Number of bodyId profiles  
            - output_path: Path to output directory
            - matrices_saved: List of saved matrix file paths
            - heatmaps_generated: List of generated heatmap paths
            - type_matrices: Type-level similarity matrices
            - bodyid_matrices: BodyId-level similarity matrices (empty dict if skipped)
            - type_avg_matrices: Type-averaged-from-bodyId matrices (empty dict if skipped)
            - bodyid_level_skipped: Boolean indicating if bodyId computation was skipped
            - is_cross_dataset: Boolean indicating if this was cross-dataset comparison
        """
        # Per-run mapping bookkeeping reset (all branches): a reused
        # comparer must not accumulate resolutions or reuse stale caches
        # across run() calls.
        self.mapping_resolution_record = {}
        self.mapping_resolution_counts = {}
        self.mapping_raw_fallback_used = False
        self._counted_query_resolutions = set()
        self._expansion_decision_cache = {}
        self._expansion_status_counts = Counter()
        self._merge_key_cache = {}
        self._mapper_snapshot = None

        # Branch for cross-dataset comparison
        if self.is_cross_dataset:
            return self._run_cross_dataset()
        
        # Branch for multi-dataset profiling (intra + inter comparisons)
        if self.is_multi_dataset:
            return self._run_multi_dataset()
        
        self._log(f"Starting connectivity profile comparison for {self.dataset}")
        self._log(f"Query: {self._format_query_for_log(self.query)}")
        self._log(f"Aggregation level: {self.aggregation_level}")
        self._progress(1, 4, "Resolving queried neurons and datasets")
        
        # Step 0: Optionally ensure connection cache is complete BEFORE profile
        # extraction. Full-dataset completion is opt-in to avoid multi-hour
        # first-run cache builds for small profile queries.
        if self.use_cache and self.ensure_cache_complete:
            self._ensure_connection_cache_complete()
        
        # Step 1: Extract both type-aggregated and individual bodyId profiles
        self._progress(2, 4, "Extracting and aggregating connectivity profiles")
        self._log("Extracting type-aggregated and bodyId-level profiles...")
        type_profiles, bodyid_profiles = self._extract_all_profiles()
        
        if len(type_profiles) < 2:
            self._log("Error: Need at least 2 type profiles to compare")
            return {'n_type_profiles': len(type_profiles), 'error': 'Insufficient profiles'}
        
        # Determine whether to skip bodyId-level computation
        n_bodyid = len(bodyid_profiles)
        if self.aggregation_level == 'bodyid':
            # The main matrices already compare individual bodyIds, so the
            # separate bodyId-level and type-avg-bodyId steps would only
            # duplicate them.
            do_skip_bodyid = True
            self._log("⚠️  aggregation_level='bodyid': the main matrices already compare "
                      "individual neurons, so the separate bodyId-level and type-avg-bodyId "
                      "matrices are skipped.")
        elif self.skip_bodyId_level == 'auto':
            do_skip_bodyid = n_bodyid > 1000
            if do_skip_bodyid:
                n_pairs = n_bodyid * (n_bodyid - 1) // 2
                self._log(f"⚠️  Auto-skipping bodyId-level computation: {n_bodyid} profiles would require {n_pairs:,} pairwise comparisons")
                self._log(f"   Use skip_bodyId_level=False to force computation (may take hours)")
        else:
            do_skip_bodyid = bool(self.skip_bodyId_level)
            if do_skip_bodyid:
                self._log(f"Skipping bodyId-level computation (skip_bodyId_level={self.skip_bodyId_level})")
        
        # Step 2: Compute type-level similarity matrices (aggregated profiles)
        self._progress(3, 4, "Computing similarity matrices")
        self._log("Computing type-level similarity matrices (aggregated profiles)...")
        type_matrices = self._compute_similarity_matrices(type_profiles)
        
        # Steps 3-4: Compute bodyId-level matrices (if not skipped)
        if do_skip_bodyid:
            bodyid_matrices = {}
            type_avg_matrices = {}
        else:
            # Step 3: Compute bodyId-level similarity matrices
            self._log("Computing bodyId-level similarity matrices...")
            bodyid_matrices = self._compute_bodyid_similarity_matrices(bodyid_profiles)
            
            # Step 4: Compute type-avg-bodyId matrices (average bodyId similarities per type pair)
            self._log("Computing type-avg-bodyId similarity matrices...")
            type_avg_matrices = self._compute_type_avg_bodyid_matrices(bodyid_profiles)
        
        # Step 5: Save all results
        save_label = ("Saving matrices, profiles, and heatmaps"
                      if self.generate_heatmaps else "Saving matrices and profiles")
        self._progress(4, 4, save_label)
        saved_files = self._save_results(
            type_profiles, bodyid_profiles,
            type_matrices, bodyid_matrices, type_avg_matrices
        )
        
        # Print summary
        output_path = saved_files.get('output_path', '')
        self._log("")
        self._log("=" * 60)
        self._log("CONNECTIVITY PROFILING COMPLETE")
        self._log("=" * 60)
        self._log(f"Output: {output_path}")
        self._log(f"Type profiles compared: {len(type_profiles)}")
        self._log(f"BodyId profiles compared: {len(bodyid_profiles)}")
        if do_skip_bodyid:
            self._log(f"BodyId-level matrices: SKIPPED (n={n_bodyid} > 1000)")
        self._log(f"Types ({len(type_profiles)} total): {self._format_query_for_log(self._sort_types_string_first(list(type_profiles.keys())))}")
        self._log("")
        self._log("Output includes:")
        self._log("  - type_level/: Type-aggregated similarity matrices & heatmaps")
        if not do_skip_bodyid:
            self._log("  - bodyid_level/: BodyId similarity + type-avg-bodyId matrices & heatmaps")
        self._log("  - profiles/individual/: Individual bodyId connectivity profiles")
        self._log("  - profiles/aggregated/: Type-aggregated connectivity profiles")
        
        return {
            'n_type_profiles': len(type_profiles),
            'n_bodyid_profiles': len(bodyid_profiles),
            'output_path': output_path,
            'type_labels': sorted(type_profiles.keys()),
            'matrices_saved': saved_files['matrices_saved'],
            'heatmaps_generated': saved_files.get('heatmaps_generated', []),
            'report_path': saved_files.get('report_path', ''),
            'type_matrices': type_matrices,
            'bodyid_matrices': bodyid_matrices,
            'type_avg_matrices': type_avg_matrices,
            'bodyid_level_skipped': do_skip_bodyid,
            'is_cross_dataset': False,
        }
    
    @staticmethod
    def _safe_folder_name(name: Any) -> str:
        """Filesystem-safe folder name for datasets / anchors / labels."""
        return (str(name).replace('/', '_').replace(' ', '_')
                .replace(':', '_').replace('.', '_'))

    def _save_multi_dataset_results(
        self,
        profiles_by_dataset: Dict[str, Dict[str, ConnectivityProfile]],
        bodyid_profiles_by_dataset: Dict[
            str, Dict[Tuple[str, int], ConnectivityProfile]
        ],
        matrices_by_dataset: Dict[
            str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
        ],
        inter_matrices: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
        anchor_profiles: Dict[str, Dict[str, Tuple[str, ConnectivityProfile]]],
    ) -> Dict[str, Any]:
        """
        Save multi-dataset results in a reorganized output folder:

            {output}/profiling_{query}_{timestamp}/
            ├── report.html                  # overall summary (all heatmaps)
            ├── parameters.json
            ├── README.txt
            ├── intra_dataset/{dataset}/     # per-dataset N×N comparisons
            │   ├── results/similarity_{direction}_{metric}.csv
            │   ├── results/bodyid_similarity_{metric}_{direction}.csv
            │   ├── results/type_avg_bodyid_similarity_{metric}_{direction}.csv
            │   └── visualization/heatmap_{prefix}_{direction}_{metric}.html
            ├── cross_dataset/
            │   ├── mapping_summary.csv      # resolved names + same-name flag
            │   ├── all_types/                # overview: neurons × dataset pairs
            │   │   ├── results/similarity_{direction}_{metric}.csv
            │   │   └── visualization/heatmap_*.html
            └── profiles/{dataset}/
                ├── individual/*_profile.json
                └── aggregated/*_profile.json
        """
        output_path = self._get_output_path()
        output_path.mkdir(parents=True, exist_ok=True)
        intra_base = output_path / 'intra_dataset'
        cross_base = output_path / 'cross_dataset'
        profiles_base = output_path / 'profiles'
        cross_base.mkdir(parents=True, exist_ok=True)
        saved = {'matrices_saved': [], 'heatmaps_generated': [],
                 'profiles_saved': [], 'output_path': str(output_path)}

        # The per-anchor matrices are aggregated into one report-facing matrix
        # whose rows are anchors and whose columns are dataset pairs. Only the
        # consolidated form is exported.
        inter_type_matrices = self._aggregate_inter_dataset_matrices(
            inter_matrices, anchor_profiles)
        saved['inter_type_matrices'] = inter_type_matrices
        
        # --- intra-dataset matrices + heatmaps per dataset ---
        matrix_level_specs = {
            'type': ('similarity', 'intra'),
            'bodyid': ('bodyid_similarity', 'intra_bodyid'),
            'type_avg_bodyid': ('type_avg_bodyid_similarity', 'intra_type_avg'),
        }
        for ds, level_matrices in matrices_by_dataset.items():
            safe_ds = self._safe_folder_name(ds)
            results_dir = intra_base / safe_ds / 'results'
            viz_dir = intra_base / safe_ds / 'visualization'
            results_dir.mkdir(parents=True, exist_ok=True)
            viz_dir.mkdir(parents=True, exist_ok=True)
            for level, matrices in level_matrices.items():
                filename_prefix, heatmap_prefix = matrix_level_specs.get(
                    level, (None, None)
                )
                if filename_prefix is None:
                    continue
                for direction, metric_matrices in matrices.items():
                    for metric, mdf in metric_matrices.items():
                        csv_path = results_dir / (
                            f'{filename_prefix}_{direction}_{metric}.csv'
                            if level == 'type'
                            else f'{filename_prefix}_{metric}_{direction}.csv'
                        )
                        mdf.to_csv(csv_path)
                        saved['matrices_saved'].append(str(csv_path))
                if self.generate_heatmaps:
                    self._log(
                        f"Generating {level} intra-dataset heatmaps for {ds}..."
                    )
                    self._generate_heatmaps_vispath(
                        matrices, viz_dir, saved, prefix=heatmap_prefix
                    )
        
        # --- aggregated profiles per dataset ---
        for ds, profiles in profiles_by_dataset.items():
            ds_root = profiles_base / self._safe_folder_name(ds)
            ds_dir = ds_root / 'aggregated'
            ds_dir.mkdir(parents=True, exist_ok=True)
            for label, profile in profiles.items():
                safe_label = self._safe_folder_name(label)
                profile_path = ds_dir / f'{safe_label}_profile.json'
                with open(profile_path, 'w', encoding='utf-8') as f:
                    json.dump(profile.to_dict(), f, indent=2)
                saved['profiles_saved'].append(str(profile_path))

            bodyid_dir = ds_root / 'individual'
            bodyid_dir.mkdir(parents=True, exist_ok=True)
            for (type_label, bid), profile in bodyid_profiles_by_dataset.get(ds, {}).items():
                safe_label = self._safe_folder_name(
                    _bodyid_profile_stem(bid, type_label))
                profile_path = bodyid_dir / f'{safe_label}_profile.json'
                profile_data = profile.to_dict()
                # Instance name next to type/dataset (tree-legend export
                # convention); the to_dict value wins when already set.
                if not str(profile_data.get('instance') or '').strip():
                    profile_data['instance'] = _body_id_instance_name(
                        profile.dataset, bid)
                with open(profile_path, 'w', encoding='utf-8') as f:
                    json.dump(profile_data, f, indent=2)
                saved['profiles_saved'].append(str(profile_path))

        # --- aggregate inter-dataset matrices + heatmaps (all neurons) ---
        all_types_results = cross_base / 'all_types' / 'results'
        all_types_viz = cross_base / 'all_types' / 'visualization'
        all_types_results.mkdir(parents=True, exist_ok=True)
        all_types_viz.mkdir(parents=True, exist_ok=True)
        for direction, metric_matrices in inter_type_matrices.items():
            for metric, mdf in metric_matrices.items():
                csv_path = all_types_results / f'similarity_{direction}_{metric}.csv'
                mdf.to_csv(csv_path)
                saved['matrices_saved'].append(str(csv_path))
        if self.generate_heatmaps and inter_type_matrices:
            self._log("Generating aggregate inter-dataset heatmaps (all neurons)...")
            self._generate_heatmaps_vispath(
                inter_type_matrices,
                all_types_viz,
                saved,
                prefix='inter_all_types',
            )
        
        # --- name-mapping summary (anchor -> resolved name per dataset) ---
        summary_rows = []
        for anchor, per_ds in anchor_profiles.items():
            row = {'anchor': anchor}
            for ds in self.datasets:
                resolved_name = per_ds[ds][0] if ds in per_ds else ''
                resolved_name = '' if resolved_name is None else str(resolved_name).strip()
                row[ds] = resolved_name
            row['same name'] = self._same_name_flag(per_ds, self.datasets)
            summary_rows.append(row)
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = cross_base / 'mapping_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            saved['matrices_saved'].append(str(summary_path))
        
        # --- parameters.json ---
        params = {
            'query': self.query if not self._custom_group_names else
                      [{'group': n, 'ids': ids} for n, ids in zip(self._custom_group_names, self.query)],
            'custom_group_names': self._custom_group_names,
            'group_map_csv': self.group_map_csv,
            'custom_mapping_file': self.custom_mapping_file,
            'aggregation_level': self.aggregation_level,
            'query_name': self.query_name,
            'datasets': self.datasets,
            'top_k': self.top_k,
            'top_m': self.top_m,
            'min_synapse_threshold': self.min_synapse_threshold,
            'direction': self.direction,
            'use_auto_type_mapping': self._type_mapper is not None,
            'inter_anchors': list(anchor_profiles.keys()),
            'timestamp': datetime.now().isoformat(),
        }
        params.update(auto_mapping_result_metadata(
            requested=self.use_auto_type_mapping,
            type_mapper=self._type_mapper,
            snapshot=self._mapper_snapshot,
            resolution_counts=self.mapping_resolution_counts,
            partner_resolution_counts=dict(self._expansion_status_counts),
            raw_fallback_used=self.mapping_raw_fallback_used,
        ))
        with open(output_path / 'parameters.json', 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, default=str)
        
        # --- README.txt ---
        readme_lines = [
            "Multi-Dataset Connectivity Profile Comparison Results",
            "=" * 50,
            f"Datasets: {', '.join(self.datasets)}",
            f"Query: {self._format_query_for_log(self.query)}",
            f"Aggregation level: {self.aggregation_level}",
            f"Auto type mapping: {'ON' if self._type_mapper is not None else 'OFF'}",
            "",
            "intra_dataset/{dataset}/: per-dataset N×N similarity at three levels:",
            "  results/similarity_* (type), bodyid_similarity_* (bodyId), and",
            "  type_avg_bodyid_similarity_* (type similarity averaged from bodyIds).",
            "  BodyId axes read '{bodyId}_{instance}' (NeuPrint datasets) or",
            "  '{bodyId}_{type}_{L|R}' (FAFB/BANC); the resolved comparison type",
            "  is the fallback when the neuron table has no metadata.",
            "profiles/{dataset}/individual/: individual bodyId connectivity profiles.",
            "profiles/{dataset}/aggregated/: mean-pooled type/group profiles.",
            "cross_dataset/all_types/: overview matrices with neurons as rows and",
            "  dataset pairs as columns (one heatmap per direction/metric).",
            "cross_dataset/mapping_summary.csv: resolved name of each anchor per dataset",
            "  plus a numeric same-name flag (1 = same non-empty name everywhere).",
            "report.html: Plotly heatmaps with local VisPath editor links.",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        (output_path / 'README.txt').write_text('\n'.join(readme_lines), encoding='utf-8')
        
        # --- overall report with Plotly heatmaps and VisPath editor links ---
        # Generate the index even when heatmaps are disabled: the CSV matrix
        # links still make the run self-contained and explain which metrics
        # were computed.
        report_path = self._generate_overall_report(
            output_path,
            matrices_by_dataset,
            inter_matrices,
            anchor_profiles,
            inter_type_matrices=inter_type_matrices,
        )
        saved['report_path'] = str(report_path)
        
        return saved

    def _generate_overall_report(
        self,
        output_path: Path,
        matrices_by_dataset: Dict[
            str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
        ],
        inter_matrices: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
        anchor_profiles: Dict[str, Dict[str, Tuple[str, ConnectivityProfile]]],
        inter_type_matrices: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    ) -> Path:
        """Build the multi-dataset report with top-level and nested tabs."""
        from html import escape

        ds_list = list(self.datasets)
        directions = self._report_directions()
        if inter_type_matrices is None:
            inter_type_matrices = self._aggregate_inter_dataset_matrices(
                inter_matrices, anchor_profiles)

        lines = [
            '<!DOCTYPE html>',
            "<html><head><meta charset='utf-8'>",
            f"<title>Connectivity Profiling Report — {escape(str(self.query_name))}</title>",
            self._report_css(),
            '</head><body><main class="report-shell">',
            '<header class="report-hero">',
            '<div class="report-kicker">DROCAT · Connectivity profiling</div>',
            '<h1 class="report-title">Connectivity profile report</h1>',
            '<p class="report-subtitle">Browse each dataset independently, then switch '
            'to the inter-dataset overview. Every card uses the Ward-clustered order '
            'used by VisPath and keeps exact values in hover tooltips.</p>',
            '<div class="report-meta">',
            f"<div class='meta-chip'><span>Datasets</span><strong>{escape(' · '.join(ds_list))}</strong></div>",
            f"<div class='meta-chip'><span>Aggregation</span><strong>{escape(str(self.aggregation_level))}</strong></div>",
            f"<div class='meta-chip'><span>Query</span><strong>{escape(self._format_query_for_log(self.query))}</strong></div>",
            f"<div class='meta-chip'><span>Type mapping</span><strong>{'ON' if self._type_mapper is not None else 'OFF'}</strong></div>",
            '</div>',
            f"<p class='report-note'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
            "Overall means upstream + downstream. Use CSV for exact matrix values and "
            "the VisPath editor for interactive clustering controls.</p>",
            '</header>',
        ]
        plotly_state = {'include_plotlyjs': True}

        def render_intra(_key: str, panel_id: str) -> None:
            lines.append(
                '<div class="section-card"><h2 class="section-heading">Intra-dataset</h2>'
                '<p class="section-summary">Each dataset has its own tab. Direction '
                'sub-tabs keep the overall, upstream, and downstream views comparable.</p>'
            )

            def render_dataset(ds: str, dataset_panel_id: str) -> None:
                safe_ds = self._safe_folder_name(ds)
                raw_matrices = matrices_by_dataset.get(ds) or {}
                if any(direction in raw_matrices for direction in directions):
                    # Accept the old flat shape when rendering a report from
                    # an in-memory caller created before the level split.
                    level_matrices = {'type': raw_matrices}
                else:
                    level_matrices = raw_matrices

                level_specs = (
                    ('type', 'Type level', 'similarity', 'intra'),
                    ('bodyid', 'BodyId level', 'bodyid_similarity', 'intra_bodyid'),
                    (
                        'type_avg_bodyid',
                        'Type average of bodyIds',
                        'type_avg_bodyid_similarity',
                        'intra_type_avg',
                    ),
                )

                def render_level(level: str, level_panel_id: str) -> None:
                    matrices = level_matrices.get(level) or {}
                    level_label = next(
                        spec[1] for spec in level_specs if spec[0] == level
                    )
                    filename_prefix = next(
                        spec[2] for spec in level_specs if spec[0] == level
                    )
                    heatmap_prefix = next(
                        spec[3] for spec in level_specs if spec[0] == level
                    )

                    def render_direction(
                        direction: str, _direction_panel_id: str
                    ) -> None:
                        direction_label = self._direction_display_name(direction)
                        metric_matrices = matrices.get(direction, {})
                        if not metric_matrices:
                            lines.append(
                                "<div class='heatmap-empty'>This profile level "
                                "was not computed for this run.</div>"
                            )
                            return
                        if level == 'type':
                            csv_paths = {
                                metric: (
                                    f'intra_dataset/{safe_ds}/results/'
                                    f'similarity_{direction}_{metric}.csv'
                                )
                                for metric in self._REPORT_METRICS
                            }
                        else:
                            csv_paths = {
                                metric: (
                                    f'intra_dataset/{safe_ds}/results/'
                                    f'{filename_prefix}_{metric}_{direction}.csv'
                                )
                                for metric in self._REPORT_METRICS
                            }
                        vispath_paths = {
                            metric: (
                                f'intra_dataset/{safe_ds}/visualization/'
                                f'heatmap_{heatmap_prefix}_{direction}_{metric}.html'
                            )
                            for metric in self._REPORT_METRICS
                        }
                        lines.append(
                            f"<div class='direction-intro'><h3 class='direction-title'>{escape(direction_label)}</h3>"
                            f"<div class='direction-note'>{escape(self._direction_note(direction))}</div></div>"
                        )
                        self._append_report_metric_grid(
                            lines,
                            output_path,
                            metric_matrices,
                            f'Intra-dataset · {ds} · {level_label} · {direction_label}',
                            csv_paths,
                            vispath_paths,
                            'Neuron / type',
                            'Neuron / type',
                            plotly_state,
                            square_cells=True,
                        )

                    self._append_report_tab_group(
                        lines,
                        f'{level_panel_id}-directions',
                        [
                            (direction, self._direction_display_name(direction))
                            for direction in directions
                        ],
                        render_direction,
                        panel_class='tab-panel direction-panel',
                    )

                self._append_report_tab_group(
                    lines,
                    f'{dataset_panel_id}-levels',
                    [(key, label) for key, label, _, _ in level_specs],
                    render_level,
                )

            self._append_report_tab_group(
                lines,
                f'{panel_id}-datasets',
                [(ds, ds) for ds in ds_list],
                render_dataset,
            )
            lines.append('</div>')

        def render_inter(_key: str, panel_id: str) -> None:
            lines.append(
                '<div class="section-card"><h2 class="section-heading">Inter-dataset</h2>'
                '<p class="section-summary">Rows are queried neurons or types and '
                'columns are dataset pairs.</p>'
            )

            def render_direction(direction: str, _direction_panel_id: str) -> None:
                direction_label = self._direction_display_name(direction)
                metric_matrices = (inter_type_matrices or {}).get(direction, {})
                csv_paths = {
                    metric: (
                        f'cross_dataset/all_types/results/'
                        f'similarity_{direction}_{metric}.csv'
                    )
                    for metric in self._REPORT_METRICS
                }
                vispath_paths = {
                    metric: (
                        'cross_dataset/all_types/visualization/'
                        f'heatmap_inter_all_types_{direction}_{metric}.html'
                    )
                    for metric in self._REPORT_METRICS
                }
                lines.append(
                    f"<div class='direction-intro'><h3 class='direction-title'>{escape(direction_label)}</h3>"
                    f"<div class='direction-note'>{escape(self._direction_note(direction))} · "
                    "rows = neurons/types · columns = dataset pairs</div></div>"
                )
                self._append_report_metric_grid(
                    lines, output_path, metric_matrices,
                    f'Inter-dataset · {direction_label}',
                    csv_paths, vispath_paths,
                    'Dataset pair', 'Neuron / type', plotly_state,
                    square_cells=False,
                )

            self._append_report_tab_group(
                lines,
                f'{panel_id}-directions',
                [(direction, self._direction_display_name(direction)) for direction in directions],
                render_direction,
                panel_class='tab-panel direction-panel',
            )

            if anchor_profiles:
                lines.append(
                    '<details class="detail-block"><summary>Resolved names across datasets</summary>'
                    '<div style="overflow-x:auto"><table class="mapping-table"><thead><tr>'
                    '<th>Anchor</th>'
                    + ''.join(f'<th>{escape(str(ds))}</th>' for ds in ds_list)
                    + '<th>same name</th>'
                    + '</tr></thead><tbody>'
                )
                for anchor, per_ds in anchor_profiles.items():
                    same_name = self._same_name_flag(per_ds, ds_list)
                    lines.append(
                        f"<tr><td>{escape(str(anchor))}</td>"
                        + ''.join(
                            f"<td>{escape(str(per_ds[ds][0])) if ds in per_ds else '—'}</td>"
                            for ds in ds_list
                        )
                        + f'<td>{same_name}</td>'
                        + '</tr>'
                    )
                lines.append('</tbody></table></div></details>')

            lines.append('</div>')

        top_tabs = [('intra', 'Intra-dataset')]
        if inter_matrices or inter_type_matrices:
            top_tabs.append(('inter', 'Inter-dataset'))
        self._append_report_tab_group(
            lines,
            'report-sections',
            top_tabs,
            render_intra if len(top_tabs) == 1 else lambda key, panel_id: (
                render_intra(key, panel_id) if key == 'intra' else render_inter(key, panel_id)
            ),
        )
        lines.extend(['</main>', self._report_script(), '</body></html>'])

        report_path = output_path / 'report.html'
        report_path.write_text('\n'.join(lines), encoding='utf-8')
        self._log(f'Overall report (tabbed clustered heatmaps): {report_path}')
        return report_path

    def _run_multi_dataset(self) -> Dict[str, Any]:
        """
        Run multi-dataset profiling.
        
        The same query is resolved in every dataset (type names mapped per
        dataset via the type mapper):
        1. intra-dataset N×N similarity matrices per dataset
        2. inter-dataset comparisons of the SAME queried neuron across
           datasets (homolog-finding backend algorithm)
        3. a reorganized output folder plus a Plotly report with VisPath editor links
        """
        ds_list = self.datasets
        self._log("=" * 60)
        self._log("MULTI-DATASET CONNECTIVITY PROFILE COMPARISON")
        self._log("=" * 60)
        self._log(f"Datasets ({len(ds_list)}): {', '.join(ds_list)}")
        self._log(f"Query: {self._format_query_for_log(self.query)}")
        self._log(f"Aggregation level: {self.aggregation_level}")
        self._progress(1, 4, "Resolving queried neurons and datasets")
        
        # Auto type mapping: resolve each query name per dataset
        self._type_mapper = None
        self._mapper_snapshot = None
        # Fresh per-run mapping bookkeeping: a second run() on one comparer
        # must own exactly its own resolutions (no accumulation into the
        # saved metadata and no stale decision caches).
        self.mapping_resolution_record = {}
        self.mapping_resolution_counts = {}
        self.mapping_raw_fallback_used = False
        self._counted_query_resolutions = set()
        self._expansion_decision_cache = {}
        self._expansion_status_counts = Counter()
        self._merge_key_cache = {}
        if self.use_auto_type_mapping:
            try:
                candidate_mapper = get_type_mapper()
                if candidate_mapper is not None and candidate_mapper._loaded:
                    self._type_mapper = candidate_mapper
                elif candidate_mapper is not None:
                    self._log(
                        "Warning: type mapper failed to load: "
                        f"{getattr(candidate_mapper, 'last_load_error', None)!r} "
                        "(analysis will run on raw names)")
            except Exception as e:
                self._log(f"Warning: could not load the type mapper: {e}")
        if self._type_mapper is not None:
            from .type_resolver import get_mapper_snapshot
            self._mapper_snapshot = get_mapper_snapshot(self._type_mapper)
            self._log("Auto type mapping: ENABLED (names mapped per dataset via the "
                      f"male-cns v1.0 neuron info; source="
                      f"{self._mapper_snapshot.metadata()['source']})")
        else:
            self._log("Auto type mapping: DISABLED — names are used as-is per dataset")
        
        # Step 1: per-dataset extraction + intra-dataset matrices.  Always
        # extract type-aggregated profiles and retain the individual bodyId
        # profiles so each dataset can expose the same three levels as a
        # single-dataset run.
        profiles_by_dataset: Dict[str, Dict[str, ConnectivityProfile]] = {}
        bodyid_profiles_by_dataset: Dict[str, Dict[Tuple[str, int], ConnectivityProfile]] = {}
        matrices_by_dataset: Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]] = {}
        self._progress(2, 4, "Extracting profiles and computing intra-dataset matrices")
        for ds in ds_list:
            self._log("")
            self._log(f"--- Dataset: {ds} ---")
            if self.use_cache and self.ensure_cache_complete:
                self._ensure_connection_cache_complete_for_dataset(ds)
            # A bodyId aggregation selection still gets a type-level view in
            # the report; its bodyId view is provided by bodyid_matrices.
            if self.aggregation_level == 'bodyid':
                type_profiles, bodyid_profiles = self._extract_profiles_for_dataset(
                    ds, aggregation_level='type'
                )
            else:
                type_profiles, bodyid_profiles = self._extract_profiles_for_dataset(ds)
            profiles_by_dataset[ds] = type_profiles
            bodyid_profiles_by_dataset[ds] = bodyid_profiles

            level_matrices: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}
            if len(type_profiles) >= 2:
                self._log(f"Computing intra-dataset similarity matrices for {ds}...")
                level_matrices['type'] = self._compute_similarity_matrices(type_profiles)
            else:
                self._log(f"Warning: fewer than 2 profiles in {ds} — "
                          f"type-level intra-dataset matrices skipped")

            n_bodyid = len(bodyid_profiles)
            if self.skip_bodyId_level == 'auto':
                skip_bodyid = n_bodyid > 1000
            else:
                skip_bodyid = bool(self.skip_bodyId_level)
            if skip_bodyid:
                self._log(
                    f"Skipping bodyId-level intra-dataset matrices for {ds} "
                    f"(skip_bodyId_level={self.skip_bodyId_level}, n={n_bodyid})"
                )
            elif bodyid_profiles:
                self._log(f"Computing bodyId-level intra-dataset matrices for {ds}...")
                level_matrices['bodyid'] = self._compute_bodyid_similarity_matrices(
                    bodyid_profiles
                )
                level_matrices['type_avg_bodyid'] = (
                    self._compute_type_avg_bodyid_matrices(bodyid_profiles)
                )

            matrices_by_dataset[ds] = level_matrices
        
        # Step 2: inter-dataset comparisons (same queried neuron across datasets)
        self._progress(3, 4, "Computing inter-dataset matrices")
        self._log("")
        self._log("Computing inter-dataset comparisons "
                  "(same queried neuron across datasets)...")
        anchor_profiles = self._build_anchor_profiles(profiles_by_dataset)
        inter_matrices = self._compute_inter_dataset_matrices(anchor_profiles)
        self._log(f"Inter-dataset comparisons ready for "
                  f"{len(anchor_profiles)} queried neurons/groups")
        
        # Step 3: save everything + overall report
        self._progress(4, 4, "Saving results and report")
        saved_files = self._save_multi_dataset_results(
            profiles_by_dataset,
            bodyid_profiles_by_dataset,
            matrices_by_dataset,
            inter_matrices,
            anchor_profiles,
        )
        output_path = saved_files.get('output_path', '')
        
        # Summary
        self._log("")
        self._log("=" * 60)
        self._log("MULTI-DATASET PROFILING COMPLETE")
        self._log("=" * 60)
        self._log(f"Output: {output_path}")
        self._log(f"Datasets: {len(ds_list)}")
        self._log(f"Inter-dataset anchors (same neuron across datasets): "
                  f"{len(anchor_profiles)}")
        self._log("Output includes:")
        self._log("  - report.html: Plotly overview heatmaps with VisPath editor links")
        self._log("  - intra_dataset/{dataset}/: type, bodyId, and type-average-bodyId matrices & heatmaps")
        self._log("  - cross_dataset/all_types/: consolidated neurons × dataset-pair matrices and heatmaps")
        self._log("  - cross_dataset/mapping_summary.csv: resolved names + same-name flag")
        self._log("  - profiles/{dataset}/: individual and aggregated connectivity profiles")
        
        return {
            'is_multi_dataset': True,
            'n_type_profiles': sum(len(p) for p in profiles_by_dataset.values()),
            'n_bodyid_profiles': sum(len(p) for p in bodyid_profiles_by_dataset.values()),
            'output_path': output_path,
            'datasets': ds_list,
            'type_labels': sorted({lbl for pro in profiles_by_dataset.values() for lbl in pro}),
            'matrices_saved': saved_files.get('matrices_saved', []),
            'heatmaps_generated': saved_files.get('heatmaps_generated', []),
            'report_path': saved_files.get('report_path', ''),
            'bodyid_level_skipped': not any('bodyid' in levels for levels in matrices_by_dataset.values()),
            'is_cross_dataset': False,
            'use_auto_type_mapping': self._type_mapper is not None,
            'inter_anchors': list(anchor_profiles.keys()),
            'inter_type_matrices': saved_files.get('inter_type_matrices', {}),
            'intra_matrices_by_dataset': matrices_by_dataset,
            'type_matrices_by_dataset': {
                ds: levels.get('type', {})
                for ds, levels in matrices_by_dataset.items()
            },
            'bodyid_matrices_by_dataset': {
                ds: levels.get('bodyid', {})
                for ds, levels in matrices_by_dataset.items()
            },
            'type_avg_matrices_by_dataset': {
                ds: levels.get('type_avg_bodyid', {})
                for ds, levels in matrices_by_dataset.items()
            },
            'bodyid_profiles_by_dataset': bodyid_profiles_by_dataset,
        }

    def _run_cross_dataset(self) -> Dict[str, Any]:
        """
        Run cross-dataset connectivity profile comparison.
        
        Generates N×M similarity matrices comparing types from the first dataset (rows)
        against types from the second dataset (columns).
        
        Returns:
            Dictionary with results summary for cross-dataset comparison.
        """
        ds_list = list(self._cross_dataset_query.keys())
        self._log("=" * 60)
        self._log("CROSS-DATASET CONNECTIVITY PROFILE COMPARISON")
        self._log("=" * 60)
        self._log(f"Dataset A (rows): {ds_list[0]} with {len(self._cross_dataset_query[ds_list[0]])} types")
        self._log(f"Dataset B (cols): {ds_list[1]} with {len(self._cross_dataset_query[ds_list[1]])} types")
        if self.use_auto_type_mapping:
            self._log("Auto type mapping: ENABLED (partner types resolved via the shared validity-aware resolver)")
        else:
            self._log("Auto type mapping: DISABLED")
        self._log("BodyId-level comparison: SKIPPED (not applicable for cross-dataset)")
        self._log("")
        self._progress(1, 4, "Resolving queried neurons and datasets")
        
        # Step 1: Extract profiles from both datasets
        self._progress(2, 4, "Extracting connectivity profiles")
        self._log("Step 1: Extracting profiles from both datasets...")
        profiles_by_dataset, row_labels, col_labels = self._extract_cross_dataset_profiles()
        
        n_rows = len(row_labels)
        n_cols = len(col_labels)
        
        if n_rows == 0 or n_cols == 0:
            self._log("Error: No profiles extracted from one or both datasets")
            return {
                'is_cross_dataset': True,
                'error': 'No profiles extracted',
                'n_type_profiles': 0,
            }
        
        # Step 2: Compute cross-dataset similarity matrices
        self._progress(3, 4, "Computing similarity matrices")
        self._log(f"Step 2: Computing {n_rows}×{n_cols} cross-dataset similarity matrices...")
        cross_matrices = self._compute_cross_dataset_similarity_matrices(
            profiles_by_dataset, row_labels, col_labels
        )
        
        # Step 3: Save results
        self._progress(4, 4, "Saving results")
        self._log("Step 3: Saving results...")
        saved_files = self._save_cross_dataset_results(
            profiles_by_dataset, cross_matrices, row_labels, col_labels
        )
        
        output_path = saved_files.get('output_path', '')
        
        # Print summary
        self._log("")
        self._log("=" * 60)
        self._log("CROSS-DATASET COMPARISON COMPLETE")
        self._log("=" * 60)
        self._log(f"Output: {output_path}")
        self._log(f"Matrix dimensions: {n_rows} × {n_cols}")
        self._log(f"  Rows ({ds_list[0]}): {row_labels}")
        self._log(f"  Cols ({ds_list[1]}): {col_labels}")
        self._log("")
        self._log("Output includes:")
        self._log("  - cross_dataset/: N×M similarity matrices & heatmaps")
        self._log("  - profiles/: Extracted profiles from each dataset")
        
        return {
            'is_cross_dataset': True,
            'n_type_profiles': n_rows + n_cols,
            'n_bodyid_profiles': 0,
            'output_path': output_path,
            'row_labels': row_labels,
            'col_labels': col_labels,
            'datasets': ds_list,
            'matrices_saved': saved_files.get('matrices_saved', []),
            'heatmaps_generated': saved_files.get('heatmaps_generated', []),
            'cross_matrices': cross_matrices,
            'bodyid_level_skipped': True,
            'use_auto_type_mapping': self.use_auto_type_mapping,
        }
    
    def _save_cross_dataset_results(
        self,
        profiles_by_dataset: Dict[str, Dict[str, ConnectivityProfile]],
        cross_matrices: Dict[str, Dict[str, pd.DataFrame]],
        row_labels: List[str],
        col_labels: List[str],
    ) -> Dict[str, Any]:
        """
        Save cross-dataset comparison results.
        
        Args:
            profiles_by_dataset: Profiles organized by dataset
            cross_matrices: N×M similarity matrices by direction and metric
            row_labels: Labels for rows (first dataset)
            col_labels: Labels for columns (second dataset)
            
        Returns:
            Dictionary with paths to saved files
        """
        ds_list = list(self._cross_dataset_query.keys())

        # Create output directory. Run folders use the unique 4-char dataset
        # labels (MCNS/HEMI/...); two versions of the same family gain a
        # version suffix (MCNS_v1_0_vs_MCNS_v0_9) — the same vocabulary as
        # the Cross-Dataset Comparison tab's legends.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_labels = make_unique_dataset_labels(ds_list[:2])
        safe_ds_name = "_vs_".join(unique_labels).replace(' ', '_').replace('/', '_')
        base_dir = os.path.join(self.output_dir, f"profiling_{safe_ds_name}_{timestamp}")
        cross_dir = os.path.join(base_dir, "cross_dataset")
        profiles_dir = os.path.join(base_dir, "profiles")
        
        os.makedirs(cross_dir, exist_ok=True)
        os.makedirs(profiles_dir, exist_ok=True)
        
        matrices_saved = []
        heatmaps_generated = []
        
        # Save matrices and heatmaps
        for direction, metric_matrices in cross_matrices.items():
            for metric, matrix_df in metric_matrices.items():
                # Save CSV
                csv_path = os.path.join(cross_dir, f"{direction}_{metric}.csv")
                matrix_df.to_csv(csv_path)
                matrices_saved.append(csv_path)
                
                # Generate heatmap
                if not self.skip_heatmap:
                    try:
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        
                        figsize = (max(8, len(col_labels) * 0.5), max(6, len(row_labels) * 0.4))
                        fig, ax = plt.subplots(figsize=figsize)
                        
                        vmin = 0 if metric != 'rank_corr' else -1
                        vmax = 1
                        cmap = 'RdYlGn' if metric == 'rank_corr' else 'YlGnBu'
                        
                        sns.heatmap(
                            matrix_df,
                            annot=len(row_labels) <= 20 and len(col_labels) <= 20,
                            fmt='.2f' if len(row_labels) <= 20 else '',
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax,
                            ax=ax,
                        )
                        
                        ax.set_title(f"Cross-Dataset {direction.title()} {metric.replace('_', ' ').title()}\n{ds_list[0]} vs {ds_list[1]}")
                        ax.set_xlabel(ds_list[1])
                        ax.set_ylabel(ds_list[0])
                        
                        plt.tight_layout()
                        heatmap_path = os.path.join(cross_dir, f"{direction}_{metric}_heatmap.png")
                        fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        heatmaps_generated.append(heatmap_path)
                        
                    except Exception as e:
                        self._log(f"Warning: Failed to generate heatmap for {direction}_{metric}: {e}")
        
        # Save profiles
        for dataset, profiles in profiles_by_dataset.items():
            ds_dir = os.path.join(profiles_dir, dataset.replace(' ', '_'))
            os.makedirs(ds_dir, exist_ok=True)
            for label, profile in profiles.items():
                safe_label = label.replace('/', '_').replace(' ', '_')
                profile_path = os.path.join(ds_dir, f"{safe_label}_profile.json")
                profile_data = profile.to_dict()
                # Instance name next to type/dataset (tree-legend export
                # convention); '' when the neuron table has no metadata.
                if not str(profile_data.get('instance') or '').strip():
                    profile_data['instance'] = _body_id_instance_name(
                        profile.dataset, profile.neuron_id)
                with open(profile_path, 'w', encoding='utf-8') as f:
                    json.dump(profile_data, f, indent=2)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'datasets': ds_list,
            'query': {ds: list(types) for ds, types in self._cross_dataset_query.items()},
            'row_labels': row_labels,
            'col_labels': col_labels,
            'direction': self.direction,
            'use_auto_type_mapping': self.use_auto_type_mapping,
            'matrices_saved': [os.path.basename(p) for p in matrices_saved],
            'heatmaps_generated': [os.path.basename(p) for p in heatmaps_generated],
        }
        # Standard auto-type-mapping metadata block (provenance; this
        # cross-dataset dict-query path standardizes through the shared
        # resolver per matrix cell and does not track per-status counts).
        try:
            metadata.update(auto_mapping_result_metadata(
                requested=self.use_auto_type_mapping,
                type_mapper=(get_type_mapper()
                             if self.use_auto_type_mapping else None),
            ))
        except Exception as exc:
            self._log(f"Warning: could not build mapping metadata: {exc}")
        
        metadata_path = os.path.join(base_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'output_path': base_dir,
            'matrices_saved': matrices_saved,
            'heatmaps_generated': heatmaps_generated,
        }
    
    def compare_intra_inter_type(self) -> Dict[str, pd.DataFrame]:
        """
        Perform intra-type and inter-type comparisons for the query neuron types.
        
        This is only meaningful when query contains neuron type names.
        - Intra-type: Similarity between bodyIds within the same type
        - Inter-type: Similarity between bodyIds of different types
        
        Returns:
            Dictionary with 'intra_type' and 'inter_type' DataFrames
        """
        # Filter query to only include neuron types (not body IDs)
        neuron_types = [q for q in self.query if not str(q).isdigit()]
        
        if not neuron_types:
            self._log("No neuron types in query - cannot perform intra/inter-type comparison")
            return {'intra_type': pd.DataFrame(), 'inter_type': pd.DataFrame()}
        
        # Optionally ensure connection cache is complete before profile extraction
        if self.use_cache and self.ensure_cache_complete:
            self._ensure_connection_cache_complete()
        
        self._log(f"Computing intra-type and inter-type comparisons for: {neuron_types}")
        
        intra_results = []
        inter_results = []
        
        # Get bodyIds for each type
        type_bodyids = {}
        for ntype in neuron_types:
            body_ids = self.profiler.get_bodyids_for_type(ntype, self.dataset)
            if body_ids:
                type_bodyids[ntype] = body_ids
        
        # Extract all profiles
        all_profiles = {}
        for ntype, body_ids in type_bodyids.items():
            for bid in body_ids:
                try:
                    profile = self.profiler.get_profile(bid, self.dataset)
                    if profile is not None:
                        all_profiles[(ntype, bid)] = profile
                except Exception:
                    continue
        
        self._log(f"Extracted {len(all_profiles)} bodyId profiles")
        
        # Intra-type comparisons
        for ntype, body_ids in type_bodyids.items():
            for i, bid_a in enumerate(body_ids):
                for bid_b in body_ids[i+1:]:
                    key_a = (ntype, bid_a)
                    key_b = (ntype, bid_b)
                    
                    if key_a in all_profiles and key_b in all_profiles:
                        scores = ProfileComparator.combined_score(
                            all_profiles[key_a],
                            all_profiles[key_b],
                            direction=self.direction if self.direction != 'both' else 'both'
                        )
                        intra_results.append({
                            'type': ntype,
                            'bodyId_a': bid_a,
                            'bodyId_b': bid_b,
                            'jaccard': scores['jaccard'],
                            'cosine': scores['cosine'],
                            'rank_correlation': scores['rank'],
                        })
        
        # Inter-type comparisons (sample to avoid explosion)
        type_list = list(type_bodyids.keys())
        for i, type_a in enumerate(type_list):
            for type_b in type_list[i+1:]:
                # Sample up to 5 bodyIds per type for inter-type comparison
                sample_a = type_bodyids[type_a][:5]
                sample_b = type_bodyids[type_b][:5]
                
                for bid_a in sample_a:
                    for bid_b in sample_b:
                        key_a = (type_a, bid_a)
                        key_b = (type_b, bid_b)
                        
                        if key_a in all_profiles and key_b in all_profiles:
                            scores = ProfileComparator.combined_score(
                                all_profiles[key_a],
                                all_profiles[key_b],
                                direction=self.direction if self.direction != 'both' else 'both'
                            )
                            inter_results.append({
                                'type_a': type_a,
                                'type_b': type_b,
                                'bodyId_a': bid_a,
                                'bodyId_b': bid_b,
                                'jaccard': scores['jaccard'],
                                'cosine': scores['cosine'],
                                'rank_correlation': scores['rank'],
                            })
        
        intra_df = pd.DataFrame(intra_results)
        inter_df = pd.DataFrame(inter_results)
        
        # Save results
        results_dir = self._get_output_path() / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        if not intra_df.empty:
            intra_df.to_csv(results_dir / 'intra_type_comparisons.csv', index=False)
        if not inter_df.empty:
            inter_df.to_csv(results_dir / 'inter_type_comparisons.csv', index=False)
        
        self._log(f"Saved intra/inter type comparisons to {results_dir}")
        
        return {
            'intra_type': intra_df,
            'inter_type': inter_df,
        }
