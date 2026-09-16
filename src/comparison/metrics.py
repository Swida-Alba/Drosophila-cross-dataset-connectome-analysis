"""
ComparisonMetrics - Metrics and statistics for cross-dataset comparison.

This module provides functions for calculating comparison metrics including
similarity scores, connection classification, and graph structure analysis.

Optimized with polars for high-performance operations on large datasets.
"""

import numpy as np
from utils.naming_utils import split_hemi_suffix
import pandas as pd
import polars as pl
from typing import Dict, List, Tuple, Optional, Any, Set
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm


class ComparisonMetrics:
    """
    Comparison metrics and statistics calculator.
    
    Provides methods for:
    - Path count comparisons
    - Edge weight comparisons
    - Graph similarity metrics (Jaccard, correlation)
    - Connection classification (common, unique, differential)
    
    Example:
        >>> metrics = ComparisonMetrics()
        >>> similarity = metrics.calculate_jaccard_similarity(edges_a, edges_b)
        >>> common = metrics.find_common_connections(aligned_data, threshold=5)
    """
    
    def __init__(self):
        """Initialize ComparisonMetrics."""
        pass
    
    # =========================================================================
    # Path Count Metrics
    # =========================================================================
    
    def compare_path_counts(
        self,
        results: Dict[str, Dict[int, pd.DataFrame]],
        group_by: str = 'hop_count'
    ) -> pd.DataFrame:
        """
        Compare path counts across datasets and thresholds.
        
        Args:
            results: Nested dict {dataset: {threshold: DataFrame}}
            group_by: Column to group paths by ('hop_count', 'path_length', etc.)
            
        Returns:
            DataFrame with path counts per dataset/threshold/group
        """
        rows = []
        
        for dataset, thresh_data in results.items():
            for threshold, df in thresh_data.items():
                if df.empty:
                    rows.append({
                        'dataset': dataset,
                        'threshold': threshold,
                        'group': 'all',
                        'path_count': 0
                    })
                    continue
                
                # Total count
                rows.append({
                    'dataset': dataset,
                    'threshold': threshold,
                    'group': 'all',
                    'path_count': len(df)
                })
                
                # Grouped counts
                if group_by in df.columns:
                    for group_val, group_df in df.groupby(group_by):
                        rows.append({
                            'dataset': dataset,
                            'threshold': threshold,
                            'group': str(group_val),
                            'path_count': len(group_df)
                        })
        
        return pd.DataFrame(rows)
    
    # =========================================================================
    # Edge Weight Metrics
    # =========================================================================
    
    def compare_edge_weights(
        self,
        aligned_data: pd.DataFrame,
        datasets: List[str]
    ) -> pd.DataFrame:
        """
        Calculate edge weight statistics across datasets.
        
        Args:
            aligned_data: DataFrame with edge weights per dataset (columns)
            datasets: List of dataset column names
            
        Returns:
            DataFrame with weight statistics per edge
        """
        if aligned_data.empty:
            return pd.DataFrame()
        
        stats = aligned_data.copy()
        
        # Available datasets in data
        available = [d for d in datasets if d in aligned_data.columns]
        
        if len(available) < 2:
            return stats
        
        # Calculate statistics across datasets
        stats['mean_weight'] = aligned_data[available].mean(axis=1)
        stats['std_weight'] = aligned_data[available].std(axis=1)
        stats['min_weight'] = aligned_data[available].min(axis=1)
        stats['max_weight'] = aligned_data[available].max(axis=1)
        stats['weight_range'] = stats['max_weight'] - stats['min_weight']
        stats['cv'] = stats['std_weight'] / stats['mean_weight'].replace(0, np.nan)  # Coefficient of variation
        
        # Count datasets where edge is present
        stats['dataset_count'] = (aligned_data[available] > 0).sum(axis=1)
        
        return stats
    
    # =========================================================================
    # Similarity Metrics
    # =========================================================================
    
    def calculate_jaccard_similarity(
        self,
        edges_a: Set[Tuple],
        edges_b: Set[Tuple]
    ) -> float:
        """
        Calculate Jaccard similarity between two edge sets.
        
        Jaccard = |A ∩ B| / |A ∪ B|
        
        Args:
            edges_a: Set of edge tuples (source, target)
            edges_b: Set of edge tuples (source, target)
            
        Returns:
            Jaccard similarity coefficient (0-1)
        """
        if not edges_a and not edges_b:
            return 1.0  # Both empty = identical
        
        intersection = len(edges_a & edges_b)
        union = len(edges_a | edges_b)
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_ruzicka_similarity(
        self,
        weights_a: pd.Series,
        weights_b: pd.Series,
        use_normalized: bool = True
    ) -> float:
        """
        Calculate Ruzicka similarity (weighted Jaccard) between two weighted edge sets.
        
        Ruzicka = Σ min(W_A(e), W_B(e)) / Σ max(W_A(e), W_B(e))
        
        Also known as weighted Jaccard or generalized Jaccard for continuous values.
        This treats edge lists as sparse vectors where:
        - Keys (indices) = unique edge pairs
        - Values = edge weights (normalized by total if use_normalized=True)
        - Missing edges have weight 0
        
        This captures both structural overlap AND weight similarity.
        
        Note: Ruzicka is inherently scale-invariant (doubling all weights doesn't
        change the result), so normalization doesn't affect the outcome but is
        applied for consistency with other weight-sensitive metrics.
        
        Args:
            weights_a: Series of weights indexed by edge (format: "source -> target")
            weights_b: Series of weights indexed by edge (format: "source -> target")
            use_normalized: If True, normalize weights to proportions (default: True)
            
        Returns:
            Ruzicka similarity coefficient (0-1)
        """
        # Get union of all edges
        all_edges = weights_a.index.union(weights_b.index)
        
        if len(all_edges) == 0:
            return 1.0  # Both empty = identical
        
        # Optionally normalize weights to proportions
        if use_normalized:
            total_a = weights_a.sum()
            total_b = weights_b.sum()
            norm_a = weights_a / total_a if total_a > 0 else weights_a
            norm_b = weights_b / total_b if total_b > 0 else weights_b
        else:
            norm_a, norm_b = weights_a, weights_b
        
        # Vectorized: reindex both to union and fill missing with 0
        vec_a = norm_a.reindex(all_edges, fill_value=0.0)
        vec_b = norm_b.reindex(all_edges, fill_value=0.0)
        
        # Vectorized min/max calculations
        numerator = np.minimum(vec_a.values, vec_b.values).sum()
        denominator = np.maximum(vec_a.values, vec_b.values).sum()
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def calculate_weighted_correlation(
        self,
        weights_a: pd.Series,
        weights_b: pd.Series
    ) -> float:
        """
        Calculate Pearson correlation of edge weights for shared edges.
        
        Args:
            weights_a: Series of weights indexed by edge
            weights_b: Series of weights indexed by edge
            
        Returns:
            Pearson correlation coefficient (-1 to 1)
        """
        # Get shared edges
        shared = weights_a.index.intersection(weights_b.index)
        
        if len(shared) < 2:
            return np.nan
            
        wa = weights_a[shared]
        wb = weights_b[shared]
        
        # Check for zero variance to avoid numpy warnings
        if wa.std() == 0 or wb.std() == 0:
            return np.nan
        
        return wa.corr(wb)
    
    def calculate_all_pairwise_similarities(
        self,
        aligned_data: pd.DataFrame,
        datasets: List[str],
        threshold: int = 0,
        include_advanced_metrics: bool = True,
        path_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate pairwise similarity metrics for all dataset combinations.
        
        Includes Jaccard similarity plus advanced graph topology metrics 
        (SVD-based cosine, graph edit distance, graph kernel).
        
        Args:
            aligned_data: DataFrame with edge weights per dataset
            datasets: List of dataset column names
            threshold: Minimum weight to consider edge present
            include_advanced_metrics: Whether to compute SVD, GED, kernel metrics (slower)
            path_data: Optional DataFrame with path min_weights per dataset (for path rank correlation)
            
        Returns:
            DataFrame with similarity metrics per dataset pair
        """
        available = [d for d in datasets if d in aligned_data.columns]
        
        if len(available) < 2:
            return pd.DataFrame()
        
        rows = []
        
        for d1, d2 in combinations(available, 2):
            
            # Get edge sets
            edges_1 = set(aligned_data[aligned_data[d1] >= threshold].index)
            edges_2 = set(aligned_data[aligned_data[d2] >= threshold].index)
            
            # Get weight series for both datasets, dropping NaN (absent edges)
            weights_1 = aligned_data[d1].dropna()
            weights_2 = aligned_data[d2].dropna()
            
            # Calculate Jaccard (binary edge presence)
            jaccard = self.calculate_jaccard_similarity(edges_1, edges_2)
            
            # Calculate Ruzicka similarity (weighted Jaccard) - weight-aware edge overlap
            ruzicka = self.calculate_ruzicka_similarity(weights_1, weights_2)
            
            # Calculate weighted correlation (shared edges only)
            correlation = self.calculate_weighted_correlation(weights_1, weights_2)
            
            # Edge counts
            intersection = len(edges_1 & edges_2)
            union = len(edges_1 | edges_2)
            
            row = {
                'dataset_1': d1,
                'dataset_2': d2,
                'jaccard_similarity': jaccard,
                'ruzicka_similarity': ruzicka,
                'pearson_correlation': correlation,
                'edges_in_d1': len(edges_1),
                'edges_in_d2': len(edges_2),
                'common_edges': intersection,
                'union_edges': union,
                'unique_to_d1': len(edges_1 - edges_2),
                'unique_to_d2': len(edges_2 - edges_1),
            }
            
            # Advanced metrics organized by category:
            # ---------------------------------------------------------------
            # TOPOLOGY METRICS (binary edge presence, no weights):
            #   - jaccard_similarity (already computed above)
            #   - edge_rank_correlation (sorted edge list rank correlation)
            #   - kernel_similarity (WL graph kernel)
            #
            # WEIGHT-SENSITIVE METRICS (uses normalized weights):
            #   - ruzicka_similarity (already computed above, with normalized weights)
            #   - cosine_similarity (vector cosine, scale-invariant)
            #   - spearman_rank_correlation (rank of shared edge weights)
            #   - rv_coefficient (multivariate matrix similarity)
            # ---------------------------------------------------------------
            if include_advanced_metrics:
                # TOPOLOGY: Edge list rank correlation (union of edges)
                edge_rank_sim = self.calculate_edge_list_rank_correlation(weights_1, weights_2)
                row['edge_rank_correlation'] = edge_rank_sim
                
                # WEIGHT-SENSITIVE: Cosine similarity (union of edges)
                cosine_sim = self.calculate_cosine_similarity(weights_1, weights_2)
                row['cosine_similarity'] = cosine_sim
                
                # PATH-LEVEL: Path list rank correlation (union of paths)
                # Only compute if path_data is provided
                if path_data is not None and d1 in path_data.columns and d2 in path_data.columns:
                    paths_1 = path_data[d1].dropna()
                    paths_2 = path_data[d2].dropna()
                    path_rank_sim = self.calculate_path_list_rank_correlation(paths_1, paths_2)
                    row['path_rank_correlation'] = path_rank_sim
                else:
                    row['path_rank_correlation'] = np.nan
                
                # WEIGHT-SENSITIVE: Spearman rank correlation on SHARED edges only
                # Uses shared edges (not union) to avoid low coefficients from many 0s
                spearman_sim = self.calculate_spearman_rank_correlation(
                    weights_1, weights_2, use_shared_edges=True, use_normalized=True
                )
                row['spearman_rank_correlation'] = spearman_sim
                
                # WEIGHT-SENSITIVE: RV coefficient (multivariate matrix similarity)
                rv_coef = self.calculate_rv_coefficient(weights_1, weights_2, use_normalized=True)
                row['rv_coefficient'] = rv_coef
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def calculate_similarity_across_thresholds(
        self,
        results: Dict[str, Dict[int, pd.DataFrame]],
        datasets: List[str],
        thresholds: List[int],
        label_mapper: Optional[Any] = None,
        show_progress: bool = True,
        path_data_func: Optional[callable] = None,
        type_mapper: Optional[Any] = None,
        max_edges_for_metrics: Optional[int] = None,
        merge_policy: Optional[Any] = None,
        hemi_aware: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate similarity metrics across all thresholds.
        
        Args:
            results: Nested dict {dataset: {threshold: DataFrame}}
            datasets: List of dataset identifiers
            thresholds: List of thresholds to analyze
            label_mapper: Optional LabelMapper for standardizing labels
            show_progress: Whether to show progress bar (default True)
            path_data_func: Optional callable(threshold) -> DataFrame that returns
                           aligned path data for computing path rank correlation
            type_mapper: Optional CrossDatasetTypeMapper for type unification
            max_edges_for_metrics: Optional max edges threshold - if exceeded, skip
                                   expensive metrics computation at that threshold
            
        Returns:
            DataFrame with similarity metrics per threshold per dataset pair
        """
        all_rows = []
        
        # Show immediate feedback before the loop
        if show_progress:
            print(f"  Calculating similarities for {len(thresholds)} thresholds across {len(datasets)} datasets...")
        
        threshold_iter = thresholds
        if show_progress and len(thresholds) > 1:
            threshold_iter = tqdm(
                thresholds, 
                desc="Computing similarity metrics",
                unit="threshold",
                leave=True
            )
        
        for threshold in threshold_iter:
            if show_progress and len(thresholds) > 1:
                threshold_iter.set_postfix({"t": threshold})
            
            # Align data at this threshold
            aligned = self._align_results_at_threshold(
                results, datasets, threshold, label_mapper, type_mapper,
                merge_policy=merge_policy,
                hemi_aware=hemi_aware)
            
            if aligned.empty:
                continue
            
            # Get path data if available
            path_data = None
            if path_data_func is not None:
                try:
                    path_data = path_data_func(threshold)
                except Exception:
                    pass  # Path data not available
            
            # Check if we should skip expensive metrics due to large edge count
            if max_edges_for_metrics and len(aligned) > max_edges_for_metrics:
                continue
            
            # Calculate pairwise similarities (include advanced metrics for visualization)
            similarities = self.calculate_all_pairwise_similarities(
                aligned, datasets, threshold=1, include_advanced_metrics=True, path_data=path_data
            )
            similarities['threshold'] = threshold
            all_rows.append(similarities)
        
        if not all_rows:
            return pd.DataFrame()
        
        return pd.concat(all_rows, ignore_index=True)

    def calculate_similarity_across_queries(
        self,
        results: Dict[str, Dict[int, pd.DataFrame]],
        datasets: List[str],
        queries: List[Dict[str, Any]],
        label_mapper: Optional[Any] = None,
        show_progress: bool = True,
        path_data_func: Optional[callable] = None,
        type_mapper: Optional[Any] = None,
        max_edges_for_metrics: Optional[int] = None,
        merge_policy: Optional[Any] = None,
        hemi_aware: bool = False,
    ) -> pd.DataFrame:
        """Calculate pairwise similarities for explicit threshold queries.

        Each query supplies a threshold mapping keyed by dataset. The
        returned rows retain the query ID and the requested threshold per
        dataset so downstream exports never have to reconstruct a scalar
        threshold from a union.
        """
        rows = []
        query_iter = queries
        if show_progress and len(queries) > 1:
            query_iter = tqdm(
                queries,
                desc="Computing similarity metrics",
                unit="query",
                leave=True,
            )
        for query in query_iter:
            query_id = query.get('id') or query.get('query_id')
            threshold_map = query.get('thresholds') or query.get(
                'thresholds_by_dataset', {})
            aligned = self._align_results_for_threshold_map(
                results,
                datasets,
                threshold_map,
                label_mapper=label_mapper,
                type_mapper=type_mapper,
                merge_policy=merge_policy,
            hemi_aware=hemi_aware,
            )
            if aligned.empty or (
                max_edges_for_metrics and len(aligned) > max_edges_for_metrics
            ):
                continue
            path_data = None
            if path_data_func is not None:
                try:
                    path_data = path_data_func(query)
                except Exception:
                    path_data = None
            similarities = self.calculate_all_pairwise_similarities(
                aligned,
                datasets,
                threshold=1,
                include_advanced_metrics=True,
                path_data=path_data,
            )
            if similarities.empty:
                continue
            similarities['query_id'] = query_id
            similarities['query_label'] = query.get('label', query_id)
            for dataset in datasets:
                similarities[f'threshold_{dataset}'] = threshold_map.get(dataset)
            rows.append(similarities)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def generate_comparison_summary_for_queries(
        self,
        results: Dict[str, Dict[int, pd.DataFrame]],
        datasets: List[str],
        queries: List[Dict[str, Any]],
        label_mapper: Optional[Any] = None,
        type_mapper: Optional[Any] = None,
        show_progress: bool = True,
        max_edges_for_metrics: Optional[int] = None,
        merge_policy: Optional[Any] = None,
        hemi_aware: bool = False,
    ) -> Dict[str, Any]:
        """Generate a compact comparison summary for explicit query rows."""
        summary: Dict[str, Any] = {
            'datasets': list(datasets),
            'threshold_mode': 'combinations',
            'queries': [],
            'key_findings': [],
            'key_findings_per_query': {},
        }
        query_iter = queries
        if show_progress and len(queries) > 1:
            query_iter = tqdm(
                queries,
                desc="Generating query summaries",
                unit="query",
                leave=True,
            )
        for query in query_iter:
            query_id = query.get('id') or query.get('query_id')
            threshold_map = query.get('thresholds') or query.get(
                'thresholds_by_dataset', {})
            aligned = self._align_results_for_threshold_map(
                results,
                datasets,
                threshold_map,
                label_mapper=label_mapper,
                type_mapper=type_mapper,
                merge_policy=merge_policy,
            hemi_aware=hemi_aware,
            )
            query_summary = {
                'query_id': query_id,
                'query_label': query.get('label', query_id),
                'thresholds': dict(threshold_map),
                'edge_counts': {},
                'common_edges': 0,
                'unique_edges': {},
                'total_edges': int(len(aligned)),
            }
            available = [dataset for dataset in datasets if dataset in aligned.columns]
            if not aligned.empty and available:
                for dataset in available:
                    query_summary['edge_counts'][dataset] = int(
                        (aligned[dataset] > 0).sum()
                    )
                query_summary['common_edges'] = int(
                    (aligned[available] > 0).all(axis=1).sum()
                )
                for dataset in available:
                    others = [item for item in available if item != dataset]
                    if others:
                        query_summary['unique_edges'][dataset] = int(
                            ((aligned[dataset] > 0)
                             & ((aligned[others] > 0).sum(axis=1) == 0)).sum()
                        )
                    else:
                        query_summary['unique_edges'][dataset] = int(
                            (aligned[dataset] > 0).sum()
                        )
                if not (
                    max_edges_for_metrics
                    and len(aligned) > max_edges_for_metrics
                ):
                    sims = self.calculate_all_pairwise_similarities(
                        aligned,
                        datasets,
                        threshold=1,
                        include_advanced_metrics=True,
                    )
                    query_summary['pairwise_similarities'] = sims
                    if not sims.empty and 'jaccard_similarity' in sims:
                        query_summary['avg_jaccard'] = float(
                            sims['jaccard_similarity'].mean()
                        )
                query_summary['conservation_rate'] = (
                    query_summary['common_edges'] / len(aligned)
                    if len(aligned) else 0.0
                )
            else:
                query_summary['pairwise_similarities'] = pd.DataFrame()
                query_summary['conservation_rate'] = 0.0
            summary['queries'].append(query_summary)
            summary['key_findings_per_query'][query_id] = query_summary
            summary['key_findings'].append(
                f"{query_id}: {query_summary['common_edges']} common edges"
            )
        if summary['queries']:
            summary['selected_query'] = summary['queries'][len(summary['queries']) // 2]
            summary['summary_stats'] = summary['selected_query']
            summary['pairwise_similarities'] = summary['selected_query'].get(
                'pairwise_similarities', pd.DataFrame())
        return summary
    
    def _align_results_at_threshold(
        self,
        results: Dict[str, Dict[int, pd.DataFrame]],
        datasets: List[str],
        threshold: int,
        label_mapper: Optional[Any] = None,
        type_mapper: Optional[Any] = None,
        merge_policy: Optional[Any] = None,
        hemi_aware: bool = False,
    ) -> pd.DataFrame:
        """
        Align results from different datasets at a specific threshold.

        When type_mapper is provided, edges are merged using canonical (male-cns)
        type names so that equivalent types across datasets (e.g., MeVPaMe1 in
        male-cns and MTe46 in flywire) are correctly aligned.

        Args:
            results: Nested dict {dataset: {threshold: DataFrame}}
            datasets: List of dataset identifiers
            threshold: Threshold to align at
            label_mapper: Optional LabelMapper for custom labels
            type_mapper: Optional CrossDatasetTypeMapper for type unification
            merge_policy: Optional query-anchored MergePolicy — its group
                labels replace the canonical namespace as merge keys

        Returns:
            Aligned DataFrame with edge index and weight columns per dataset
        """
        return self._align_results_for_threshold_map(
            results,
            datasets,
            {dataset: threshold for dataset in datasets},
            label_mapper=label_mapper,
            type_mapper=type_mapper,
            merge_policy=merge_policy,
            hemi_aware=hemi_aware,
        )

    def _align_results_for_threshold_map(
        self,
        results: Dict[str, Dict[int, pd.DataFrame]],
        datasets: List[str],
        threshold_map: Dict[str, int],
        label_mapper: Optional[Any] = None,
        type_mapper: Optional[Any] = None,
        merge_policy: Optional[Any] = None,
        hemi_aware: bool = False,
    ) -> pd.DataFrame:
        """Align each dataset at its own requested threshold.

        This is the query-aware counterpart to the scalar alignment helper.
        The map is dataset-keyed so an advanced cross-dataset query such as
        {dataset_a: 3, dataset_b: 7} cannot accidentally fall back to a
        scalar union threshold.
        """
        dfs = []
        edge_display_names = {}  # canonical_edge -> display_name
        
        for dataset in datasets:
            threshold = threshold_map.get(dataset)
            if threshold is None:
                continue
            if dataset not in results or threshold not in results[dataset]:
                continue
            
            df = results[dataset][threshold]
            if df.empty:
                continue
            
            # Determine edge columns
            if 'std_label_pre' in df.columns and 'std_label_post' in df.columns:
                pre_col, post_col = 'std_label_pre', 'std_label_post'
            elif 'type_pre' in df.columns and 'type_post' in df.columns:
                pre_col, post_col = 'type_pre', 'type_post'
            else:
                pre_col, post_col = 'bodyId_pre', 'bodyId_post'
            
            # Apply label mapping if provided
            if label_mapper and 'type_pre' in df.columns:
                df = label_mapper.apply_to_dataframe(df.copy(), dataset)
                pre_col, post_col = 'std_label_pre', 'std_label_post'
            
            # Aggregate by edge
            weight_col = 'weight' if 'weight' in df.columns else 'path_weight'
            
            if weight_col in df.columns:
                agg_df = df.groupby([pre_col, post_col])[weight_col].sum().reset_index()
            else:
                agg_df = df.groupby([pre_col, post_col]).size().reset_index(name='count')
                weight_col = 'count'
            
            # Apply type mapping if provided - standardize edge names to canonical form
            # Using vectorized operations instead of slow iterrows
            if type_mapper is not None and pre_col in ['type_pre', 'std_label_pre']:
                # Vectorized: convert columns to strings
                original_pre_series = agg_df[pre_col].astype(str)
                original_post_series = agg_df[post_col].astype(str)
                
                # Vectorized: apply type mapping using map()
                # Build lookup dicts for this dataset's type mappings
                unique_pre = original_pre_series.unique()
                unique_post = original_post_series.unique()
                all_unique_types = set(unique_pre) | set(unique_post)

                # Create mapping dict for this dataset.  Routed through the
                # shared validity resolver (``canonical_merge_key``): licensed
                # renames map to their canonical target, while a CONFLICTED
                # type keeps a dataset-scoped key (``dataset:raw``) so two
                # datasets' same-named conflicted rows can never merge.
                # With a query-anchored MergePolicy (plan:
                # plan-query-anchored-cross-dataset-analysis.md) the policy's
                # group labels take precedence; types it does not govern
                # fall back to canonical_merge_key toward the policy's
                # anchor namespace (B3), which is byte-identical to the
                # historical male-cns default when no policy exists.
                from .type_resolver import canonical_merge_key
                merge_cache: dict = {}
                fallback_target = (
                    merge_policy.anchor_ds
                    if merge_policy is not None else None)
                canonical_map = {}
                # Hemisphere-suffix awareness (plan R1-b): with
                # separate_hemispheres, frame types can carry _L/_R/_U
                # suffixes while key_map/mapper relations key base names.
                # Strip the suffix for the lookup and re-apply it to the
                # canonical key so L and R stay DISTINCT rows that still
                # merge across datasets.
                for t in all_unique_types:
                    key = (
                        merge_policy.key_for(dataset, t)
                        if merge_policy is not None else None
                    )
                    suffix = ''
                    lookup_t = t
                    if hemi_aware and key is None:
                        lookup_t, suffix = split_hemi_suffix(t)
                        if suffix:
                            key = (
                                merge_policy.key_for(dataset, lookup_t)
                                if merge_policy is not None else None
                            )
                    if key is None:
                        resolved_key = canonical_merge_key(
                            type_mapper, lookup_t, dataset, fallback_target,
                            cache=merge_cache).key
                        key = f'{resolved_key}{suffix}' if suffix else resolved_key
                    canonical_map[t] = key
                
                # Apply mapping vectorized
                agg_df['canonical_pre'] = original_pre_series.map(canonical_map)
                agg_df['canonical_post'] = original_post_series.map(canonical_map)
                
                # Track display names for later (collect all dataset-specific names)
                for original, canonical in canonical_map.items():
                    if canonical not in edge_display_names:
                        edge_display_names[canonical] = {canonical}  # Include canonical itself
                    if original != canonical:
                        edge_display_names[canonical].add(original)
                
                # Re-aggregate by canonical edge (in case multiple original edges map to same canonical)
                agg_df = agg_df.groupby(['canonical_pre', 'canonical_post'])[weight_col].sum().reset_index()
                pre_col, post_col = 'canonical_pre', 'canonical_post'
            
            # Create edge index
            agg_df['edge'] = agg_df[pre_col].astype(str) + ' -> ' + agg_df[post_col].astype(str)
            agg_df = agg_df.set_index('edge')
            agg_df = agg_df[[weight_col]].rename(columns={weight_col: dataset})
            
            dfs.append(agg_df)
        
        if not dfs:
            return pd.DataFrame()
        
        # Merge all datasets
        aligned = pd.concat(dfs, axis=1, join='outer').fillna(0)
        
        # Skip expensive display name building for large datasets (>10k edges)
        # The canonical names are sufficient for computation; display names are cosmetic
        if type_mapper is not None and edge_display_names and len(aligned) <= 10000:
            # Pre-compute display name mappings for all unique types (vectorized)
            # Use type_mapper.get_display_name() for consistent format: "MeVPLo2(MTe07)"
            # Merge-policy group labels are ALREADY the final display identity
            # (B6): re-running get_display_name over them would reintroduce the
            # scoped cosmetic form the policy exists to remove.
            type_display_map = {}
            for canonical in edge_display_names.keys():
                if merge_policy is not None and merge_policy.is_group_label(
                        canonical):
                    type_display_map[canonical] = canonical
                else:
                    type_display_map[canonical] = type_mapper.get_display_name(
                        canonical, datasets)
            
            # Vectorized: split index, map, and rebuild
            def build_display_name(edge_key):
                if ' -> ' not in str(edge_key):
                    return str(edge_key)
                parts = str(edge_key).split(' -> ')
                pre_canonical = parts[0]
                post_canonical = parts[1] if len(parts) > 1 else ''
                pre_display = type_display_map.get(pre_canonical, pre_canonical)
                post_display = type_display_map.get(post_canonical, post_canonical)
                return f"{pre_display} -> {post_display}"
            
            # Use polars for fast string operations
            edge_list = aligned.index.tolist()
            display_index = [build_display_name(e) for e in edge_list]
            aligned.index = pd.Index(display_index)
            
            # Aggregate duplicate indices (sum weights) that may result from display name merging
            # This handles cases where different canonical names map to the same display name
            if aligned.index.duplicated().any():
                aligned = aligned.groupby(aligned.index).sum()
        
        return aligned
    
    # =========================================================================
    # Connection Classification
    # =========================================================================
    
    def find_common_connections(
        self,
        aligned_data: pd.DataFrame,
        datasets: List[str],
        threshold: int = 1
    ) -> pd.DataFrame:
        """
        Find connections present in ALL datasets above threshold.
        
        Args:
            aligned_data: DataFrame with edge weights per dataset
            datasets: List of dataset column names
            threshold: Minimum weight to consider edge present
            
        Returns:
            DataFrame with common connections
        """
        available = [d for d in datasets if d in aligned_data.columns]
        
        if not available:
            return pd.DataFrame()
        
        # Filter to edges present in all datasets
        mask = (aligned_data[available] >= threshold).all(axis=1)
        
        return aligned_data[mask].copy()
    
    def find_unique_connections(
        self,
        aligned_data: pd.DataFrame,
        datasets: List[str],
        threshold: int = 1
    ) -> Dict[str, pd.DataFrame]:
        """
        Find connections unique to each dataset.
        
        Args:
            aligned_data: DataFrame with edge weights per dataset
            datasets: List of dataset column names
            threshold: Minimum weight to consider edge present
            
        Returns:
            Dict mapping dataset name to DataFrame of unique connections
        """
        available = [d for d in datasets if d in aligned_data.columns]
        
        if not available:
            return {}
        
        unique = {}
        
        for dataset in available:
            # Present in this dataset but not in any other
            present_here = aligned_data[dataset] >= threshold
            present_elsewhere = (aligned_data[[d for d in available if d != dataset]] >= threshold).any(axis=1)
            
            mask = present_here & ~present_elsewhere
            unique[dataset] = aligned_data[mask].copy()
        
        return unique
    
    def find_differential_connections(
        self,
        aligned_data: pd.DataFrame,
        datasets: List[str],
        fold_threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        Find connections with large weight differences across datasets.
        
        Args:
            aligned_data: DataFrame with edge weights per dataset
            datasets: List of dataset column names
            fold_threshold: Minimum fold change to be considered differential
            
        Returns:
            DataFrame with differential connections and fold changes
        """
        available = [d for d in datasets if d in aligned_data.columns]
        
        if len(available) < 2:
            return pd.DataFrame()
        
        result = aligned_data.copy()
        
        # Calculate max fold change between any pair
        max_fc = pd.Series(1.0, index=aligned_data.index)
        
        for d1, d2 in combinations(available, 2):
            # Add small value to avoid division by zero
            fc = (aligned_data[d1] + 1) / (aligned_data[d2] + 1)
            fc_rev = (aligned_data[d2] + 1) / (aligned_data[d1] + 1)
            
            # Take max of fold change in either direction
            fc_max = pd.concat([fc, fc_rev], axis=1).max(axis=1)
            max_fc = pd.concat([max_fc, fc_max], axis=1).max(axis=1)
        
        result['max_fold_change'] = max_fc
        
        # Filter by threshold
        return result[result['max_fold_change'] >= fold_threshold].sort_values('max_fold_change', ascending=False)
    
    def find_conserved_strong_connections(
        self,
        aligned_data: pd.DataFrame,
        datasets: List[str],
        top_n: int = 50,
        threshold: int = 1
    ) -> pd.DataFrame:
        """
        Find high-weight connections shared across datasets.
        
        Args:
            aligned_data: DataFrame with edge weights per dataset
            datasets: List of dataset column names
            top_n: Number of top edges to consider per dataset
            threshold: Minimum weight to consider edge present
            
        Returns:
            DataFrame with conserved strong connections
        """
        available = [d for d in datasets if d in aligned_data.columns]
        
        if not available:
            return pd.DataFrame()
        
        # Get top edges for each dataset
        top_edges = set()
        for dataset in available:
            top = aligned_data.nlargest(top_n, dataset).index
            top_edges.update(top)
        
        # Find edges that are in top_n for ALL datasets
        conserved = []
        for edge in top_edges:
            is_top_in_all = True
            for dataset in available:
                dataset_top = set(aligned_data.nlargest(top_n, dataset).index)
                if edge not in dataset_top:
                    is_top_in_all = False
                    break
            
            if is_top_in_all:
                conserved.append(edge)
        
        if not conserved:
            return pd.DataFrame()
        
        result = aligned_data.loc[conserved].copy()
        result['mean_weight'] = result[available].mean(axis=1)
        
        return result.sort_values('mean_weight', ascending=False)
    
    # =========================================================================
    # Summary Statistics
    # =========================================================================
    
    def calculate_summary_statistics(
        self,
        aligned_data: pd.DataFrame,
        datasets: List[str],
        threshold: int = 1
    ) -> pd.DataFrame:
        """
        Calculate summary statistics for the comparison.
        
        Args:
            aligned_data: DataFrame with edge weights per dataset
            datasets: List of dataset column names
            threshold: Minimum weight to consider edge present
            
        Returns:
            DataFrame with summary statistics per dataset
        """
        available = [d for d in datasets if d in aligned_data.columns]
        
        rows = []
        for dataset in available:
            data = aligned_data[dataset]
            present = data >= threshold
            
            rows.append({
                'dataset': dataset,
                'total_edges': present.sum(),
                'total_weight': data[present].sum(),
                'mean_weight': data[present].mean(),
                'median_weight': data[present].median(),
                'max_weight': data[present].max(),
                'std_weight': data[present].std()
            })
        
        return pd.DataFrame(rows)
    
    def generate_comparison_summary(
        self,
        results: Dict[str, Dict[int, pd.DataFrame]],
        datasets: List[str],
        thresholds: List[int],
        label_mapper: Optional[Any] = None,
        type_mapper: Optional[Any] = None,
        show_progress: bool = True,
        max_edges_for_metrics: Optional[int] = None,
        merge_policy: Optional[Any] = None,
        hemi_aware: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive comparison summary.
        
        Args:
            results: Nested dict {dataset: {threshold: DataFrame}}
            datasets: List of dataset identifiers
            thresholds: List of thresholds
            label_mapper: Optional LabelMapper
            type_mapper: Optional CrossDatasetTypeMapper for type unification
            show_progress: Whether to show progress messages
            max_edges_for_metrics: Skip expensive metrics if edges exceed this
            
        Returns:
            Dictionary with all comparison metrics and findings
        """
        summary = {
            'datasets': datasets,
            'thresholds': thresholds,
            'key_findings': [],  # Legacy: simple list for backward compatibility
            'key_findings_per_threshold': {}  # New: per-threshold structured findings
        }
        
        # Calculate metrics at the middle threshold for backward compatibility
        mid_threshold = thresholds[len(thresholds) // 2]
        aligned = self._align_results_at_threshold(
            results, datasets, mid_threshold, label_mapper, type_mapper,
            merge_policy=merge_policy, hemi_aware=hemi_aware)
        
        if aligned.empty:
            summary['key_findings'].append("No data available for comparison")
            return summary
        
        # Summary statistics (at mid threshold)
        summary['summary_stats'] = self.calculate_summary_statistics(aligned, datasets, mid_threshold)
        
        # Pairwise similarities (at mid threshold) - skip if too many edges
        if max_edges_for_metrics and len(aligned) > max_edges_for_metrics:
            similarities = pd.DataFrame()
        else:
            similarities = self.calculate_all_pairwise_similarities(aligned, datasets, mid_threshold)
        summary['pairwise_similarities'] = similarities
        
        if not similarities.empty:
            avg_jaccard = similarities['jaccard_similarity'].mean()
            summary['key_findings'].append(f"Average Jaccard similarity: {avg_jaccard:.3f}")
            
            avg_corr = similarities['pearson_correlation'].mean()
            if not np.isnan(avg_corr):
                summary['key_findings'].append(f"Average weight correlation: {avg_corr:.3f}")
        
        # Common connections count (at mid threshold)
        common = self.find_common_connections(aligned, datasets, mid_threshold)
        summary['common_connections_count'] = len(common)
        summary['key_findings'].append(f"Common connections (threshold={mid_threshold}): {len(common)}")
        
        # Unique connections counts (at mid threshold)
        unique = self.find_unique_connections(aligned, datasets, mid_threshold)
        for dataset, unique_df in unique.items():
            summary['key_findings'].append(f"Unique to {dataset}: {len(unique_df)} connections")
        
        # =====================================================================
        # Generate per-threshold key findings
        # =====================================================================
        for threshold in thresholds:
            threshold_findings = {
                'threshold': threshold,
                'edge_counts': {},
                'common_edges': 0,
                'unique_edges': {},
                'avg_jaccard': 0.0,
                'avg_cosine': 0.0,
                'conservation_rate': 0.0
            }
            
            aligned_t = self._align_results_at_threshold(
                results, datasets, threshold, label_mapper, type_mapper,
                merge_policy=merge_policy,
                hemi_aware=hemi_aware)
            
            if aligned_t.empty:
                summary['key_findings_per_threshold'][threshold] = threshold_findings
                continue
            
            available_ds = [d for d in datasets if d in aligned_t.columns]
            
            # Edge counts per dataset
            for ds in available_ds:
                edge_count = (aligned_t[ds] > 0).sum()
                threshold_findings['edge_counts'][ds] = int(edge_count)
            
            # Common edges (present in all datasets)
            if available_ds:
                mask_all = (aligned_t[available_ds] > 0).all(axis=1)
                threshold_findings['common_edges'] = int(mask_all.sum())
            
            # Unique edges per dataset
            for ds in available_ds:
                other_ds = [d for d in available_ds if d != ds]
                if other_ds:
                    ds_present = aligned_t[ds] > 0
                    others_absent = (aligned_t[other_ds] > 0).sum(axis=1) == 0
                    unique_count = (ds_present & others_absent).sum()
                    threshold_findings['unique_edges'][ds] = int(unique_count)
                else:
                    threshold_findings['unique_edges'][ds] = int((aligned_t[ds] > 0).sum())
            
            # Pairwise similarities at this threshold - skip if too many edges
            if max_edges_for_metrics and len(aligned_t) > max_edges_for_metrics:
                sims = pd.DataFrame()
            else:
                sims = self.calculate_all_pairwise_similarities(aligned_t, datasets, threshold)
            if not sims.empty:
                threshold_findings['avg_jaccard'] = float(sims['jaccard_similarity'].mean())
                if 'rv_coefficient' in sims.columns:
                    threshold_findings['avg_rv'] = float(sims['rv_coefficient'].mean())
                if 'matrix_correlation' in sims.columns:
                    corr = sims['matrix_correlation'].mean()
                    threshold_findings['avg_correlation'] = float(corr) if not np.isnan(corr) else 0.0
            
            # Conservation rate - total_edges is the number of unique edges across all datasets (union)
            total_edges = len(aligned_t)
            threshold_findings['total_edges'] = int(total_edges)  # Add for consistency with pie chart
            if total_edges > 0:
                threshold_findings['conservation_rate'] = threshold_findings['common_edges'] / total_edges
            
            # Initialize path stats (will be populated later if path data is available)
            threshold_findings['total_paths'] = 0
            threshold_findings['common_paths'] = 0
            threshold_findings['path_conservation_rate'] = 0.0
            
            summary['key_findings_per_threshold'][threshold] = threshold_findings
        
        return summary
    
    # =========================================================================
    # Edge Density and Type Coverage Metrics
    # =========================================================================
    
    def calculate_edge_density(
        self,
        aligned_data: pd.DataFrame,
        datasets: List[str],
        source_count: int,
        target_count: int,
        threshold: int = 1
    ) -> pd.DataFrame:
        """
        Calculate edge density for each dataset.
        
        Edge density = actual edges / possible edges
        where possible edges = source_count * target_count
        
        Args:
            aligned_data: DataFrame with edge weights per dataset
            datasets: List of dataset column names
            source_count: Number of source neurons
            target_count: Number of target neurons
            threshold: Minimum weight to consider edge present
            
        Returns:
            DataFrame with edge density per dataset
        """
        possible_edges = source_count * target_count
        
        if possible_edges == 0:
            return pd.DataFrame()
        
        available = [d for d in datasets if d in aligned_data.columns]
        
        rows = []
        for dataset in available:
            actual_edges = (aligned_data[dataset] >= threshold).sum()
            density = actual_edges / possible_edges
            
            rows.append({
                'dataset': dataset,
                'actual_edges': actual_edges,
                'possible_edges': possible_edges,
                'edge_density': density,
                'edge_density_pct': round(density * 100, 2)
            })
        
        return pd.DataFrame(rows)
    
    def calculate_type_coverage(
        self,
        results: Dict[str, Dict[int, pd.DataFrame]],
        datasets: List[str],
        threshold: int,
        source_types: List[str],
        target_types: List[str]
    ) -> pd.DataFrame:
        """
        Calculate percentage of source/target types with connections.
        
        Args:
            results: Nested dict {dataset: {threshold: DataFrame}}
            datasets: List of dataset identifiers
            threshold: Threshold to analyze
            source_types: List of expected source types
            target_types: List of expected target types
            
        Returns:
            DataFrame with type coverage per dataset
        """
        rows = []
        
        for dataset in datasets:
            if dataset not in results or threshold not in results[dataset]:
                rows.append({
                    'dataset': dataset,
                    'source_types_connected': 0,
                    'source_types_total': len(source_types),
                    'source_coverage_pct': 0,
                    'target_types_connected': 0,
                    'target_types_total': len(target_types),
                    'target_coverage_pct': 0
                })
                continue
            
            df = results[dataset][threshold]
            if df.empty:
                rows.append({
                    'dataset': dataset,
                    'source_types_connected': 0,
                    'source_types_total': len(source_types),
                    'source_coverage_pct': 0,
                    'target_types_connected': 0,
                    'target_types_total': len(target_types),
                    'target_coverage_pct': 0
                })
                continue
            
            # Find type columns
            pre_col = 'type_pre' if 'type_pre' in df.columns else 'std_label_pre' if 'std_label_pre' in df.columns else None
            post_col = 'type_post' if 'type_post' in df.columns else 'std_label_post' if 'std_label_post' in df.columns else None
            
            source_connected = 0
            target_connected = 0
            
            if pre_col:
                connected_sources = df[pre_col].unique()
                source_connected = len(set(connected_sources) & set(source_types))
            
            if post_col:
                connected_targets = df[post_col].unique()
                target_connected = len(set(connected_targets) & set(target_types))
            
            rows.append({
                'dataset': dataset,
                'source_types_connected': source_connected,
                'source_types_total': len(source_types),
                'source_coverage_pct': round(source_connected / len(source_types) * 100, 2) if source_types else 0,
                'target_types_connected': target_connected,
                'target_types_total': len(target_types),
                'target_coverage_pct': round(target_connected / len(target_types) * 100, 2) if target_types else 0
            })
        
        return pd.DataFrame(rows)
    
    # =========================================================================
    # Degree Distribution Analysis
    # =========================================================================
    
    def calculate_degree_distribution(
        self,
        results: Dict[str, Dict[int, pd.DataFrame]],
        datasets: List[str],
        threshold: int
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate in-degree and out-degree distributions for each dataset.
        
        Args:
            results: Nested dict {dataset: {threshold: DataFrame}}
            datasets: List of dataset identifiers
            threshold: Threshold to analyze
            
        Returns:
            Dict with 'out_degree' and 'in_degree' DataFrames
        """
        out_degrees = {}
        in_degrees = {}
        
        for dataset in datasets:
            if dataset not in results or threshold not in results[dataset]:
                continue
            
            df = results[dataset][threshold]
            if df.empty:
                continue
            
            # Find node columns
            pre_col = 'type_pre' if 'type_pre' in df.columns else 'bodyId_pre'
            post_col = 'type_post' if 'type_post' in df.columns else 'bodyId_post'
            weight_col = 'weight' if 'weight' in df.columns else None
            
            # Out-degree (number of targets each source connects to)
            if weight_col:
                out_deg = df.groupby(pre_col).agg({
                    post_col: 'nunique',
                    weight_col: 'sum'
                }).rename(columns={post_col: 'out_degree', weight_col: 'total_weight'})
            else:
                out_deg = df.groupby(pre_col)[post_col].nunique().to_frame('out_degree')
            out_deg['dataset'] = dataset
            out_degrees[dataset] = out_deg.reset_index()
            
            # In-degree (number of sources each target receives from)
            if weight_col:
                in_deg = df.groupby(post_col).agg({
                    pre_col: 'nunique',
                    weight_col: 'sum'
                }).rename(columns={pre_col: 'in_degree', weight_col: 'total_weight'})
            else:
                in_deg = df.groupby(post_col)[pre_col].nunique().to_frame('in_degree')
            in_deg['dataset'] = dataset
            in_degrees[dataset] = in_deg.reset_index()
        
        # Combine into summary DataFrames
        out_summary = pd.concat(out_degrees.values(), ignore_index=True) if out_degrees else pd.DataFrame()
        in_summary = pd.concat(in_degrees.values(), ignore_index=True) if in_degrees else pd.DataFrame()
        
        return {
            'out_degree': out_summary,
            'in_degree': in_summary
        }
    
    def calculate_degree_statistics(
        self,
        degree_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate summary statistics for degree distributions.
        
        Args:
            degree_data: Dict from calculate_degree_distribution
            
        Returns:
            DataFrame with degree statistics per dataset
        """
        rows = []
        
        for degree_type in ['out_degree', 'in_degree']:
            df = degree_data.get(degree_type, pd.DataFrame())
            if df.empty:
                continue
            
            for dataset in df['dataset'].unique():
                subset = df[df['dataset'] == dataset]
                degree_col = degree_type
                
                if degree_col in subset.columns:
                    rows.append({
                        'dataset': dataset,
                        'degree_type': degree_type,
                        'mean': subset[degree_col].mean(),
                        'median': subset[degree_col].median(),
                        'std': subset[degree_col].std(),
                        'min': subset[degree_col].min(),
                        'max': subset[degree_col].max(),
                        'num_nodes': len(subset)
                    })
        
        return pd.DataFrame(rows)
    
    # =========================================================================
    # Top Edges Analysis
    # =========================================================================
    
    def get_top_edges_per_dataset(
        self,
        aligned_data: pd.DataFrame,
        datasets: List[str],
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        Get top N edges for each dataset ranked by weight.
        
        Args:
            aligned_data: DataFrame with edge weights per dataset
            datasets: List of dataset column names
            top_n: Number of top edges to return per dataset
            
        Returns:
            DataFrame with top edges per dataset
        """
        available = [d for d in datasets if d in aligned_data.columns]
        
        all_top_edges = []
        
        for dataset in available:
            # Get top N edges for this dataset
            if top_n > 0:
                top = aligned_data.nlargest(top_n, dataset).copy()
            else:
                # If top_n <= 0, include all edges sorted by weight
                top = aligned_data.sort_values(dataset, ascending=False).copy()
                # Filter out zero-weight edges if desired, or keep them? 
                # Usually "top edges" implies existing edges.
                top = top[top[dataset] > 0]
            
            top['dataset'] = dataset
            top['rank_in_dataset'] = range(1, len(top) + 1)
            top['weight'] = top[dataset]
            
            # Add presence info in other datasets
            for other in available:
                if other != dataset:
                    top[f'present_in_{other}'] = top[other] > 0
                    top[f'weight_in_{other}'] = top[other]
            
            # Keep edge index as column
            top['edge'] = top.index
            all_top_edges.append(top.reset_index(drop=True))

        if not all_top_edges:
            return pd.DataFrame()

        result = pd.concat(all_top_edges, ignore_index=True)

        # N4 cleanup: drop the redundant per-dataset weight columns (the
        # dataset + weight columns already carry the value), and fill the
        # self-referencing present/weight columns instead of leaving them
        # None for the reference dataset's own rows.
        for dataset in available:
            if f'present_in_{dataset}' in result.columns:
                self_mask = result['dataset'] == dataset
                result.loc[self_mask, f'present_in_{dataset}'] = True
                result.loc[self_mask, f'weight_in_{dataset}'] = \
                    result.loc[self_mask, 'weight']
            if dataset in result.columns:
                result = result.drop(columns=[dataset])
        return result
    
    def compare_top_edges_overlap(
        self,
        aligned_data: pd.DataFrame,
        datasets: List[str],
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        Compare overlap of top edges across datasets.
        
        Args:
            aligned_data: DataFrame with edge weights per dataset
            datasets: List of dataset column names
            top_n: Number of top edges to compare
            
        Returns:
            DataFrame with overlap statistics
        """
        from itertools import combinations
        
        available = [d for d in datasets if d in aligned_data.columns]
        
        if len(available) < 2:
            return pd.DataFrame()
        
        # Get top edges for each dataset
        top_edges = {}
        for dataset in available:
            if top_n > 0:
                top_edges[dataset] = set(aligned_data.nlargest(top_n, dataset).index)
            else:
                # All edges with weight > 0
                top_edges[dataset] = set(aligned_data[aligned_data[dataset] > 0].index)
        
        rows = []
        
        for d1, d2 in combinations(available, 2):
            intersection = len(top_edges[d1] & top_edges[d2])
            union = len(top_edges[d1] | top_edges[d2])
            
            # If top_n is not fixed, use size of d1 as reference for percentage
            ref_size = top_n if top_n > 0 else len(top_edges[d1])
            
            rows.append({
                'dataset_1': d1,
                'dataset_2': d2,
                'top_n': top_n,
                'overlap_count': intersection,
                'overlap_pct': round(intersection / ref_size * 100, 2) if ref_size > 0 else 0,
                'jaccard_top_edges': intersection / union if union > 0 else 0
            })
        
        # Also calculate all-dataset intersection
        if len(available) > 2:
            all_intersection = set.intersection(*top_edges.values())
            
            # For ALL comparison, if top_n is -1, use union size as reference
            if top_n > 0:
                ref_size = top_n
            else:
                all_union = set.union(*top_edges.values())
                ref_size = len(all_union)
                
            rows.append({
                'dataset_1': 'ALL',
                'dataset_2': 'ALL',
                'top_n': top_n,
                'overlap_count': len(all_intersection),
                'overlap_pct': round(len(all_intersection) / ref_size * 100, 2) if ref_size > 0 else 0,
                'jaccard_top_edges': None
            })
        
        return pd.DataFrame(rows)
    
    # =========================================================================
    # Advanced Graph Similarity Metrics (SVD, Graph Edit Distance, Kernels)
    # =========================================================================
    
    def calculate_frobenius_similarity(
        self,
        weights_a: pd.Series,
        weights_b: pd.Series
    ) -> float:
        """
        Calculate Frobenius norm similarity between two graphs.
        
        Measures absolute magnitude of edge weight differences.
        Sensitive to scale - different magnitudes reduce similarity.
        
        Formula: 1 - ||A - B||_F / (||A||_F + ||B||_F)
        
        Use when: Absolute connection strengths matter.
        
        Args:
            weights_a: Series of weights indexed by edge (format: "source -> target")
            weights_b: Series of weights indexed by edge (format: "source -> target")
            
        Returns:
            Frobenius similarity (0 to 1)
        """
        adj_a, adj_b = self._build_aligned_adjacency_matrices(weights_a, weights_b)
        
        if adj_a.size == 0 or adj_b.size == 0:
            return 0.0
        
        diff = adj_a - adj_b
        frob_diff = np.linalg.norm(diff, 'fro')
        norm_a = np.linalg.norm(adj_a, 'fro')
        norm_b = np.linalg.norm(adj_b, 'fro')
        sum_norm = norm_a + norm_b
        
        if sum_norm == 0:
            return 1.0  # Both empty
        
        frob_sim = 1.0 - (frob_diff / sum_norm)
        return max(0.0, frob_sim)
    
    def calculate_spearman_rank_correlation(
        self,
        weights_a: pd.Series,
        weights_b: pd.Series,
        use_shared_edges: bool = True,
        use_normalized: bool = True
    ) -> float:
        """
        Calculate Spearman rank correlation between edge weights.
        
        Uses SHARED edges (edges present in both graphs) to avoid the problem
        where many 0s in the union dilute the correlation coefficient.
        
        Returns raw Spearman correlation in [-1, 1] range:
        - 1.0 = perfect positive correlation (same ranking)
        - 0.0 = no correlation
        - -1.0 = perfect negative correlation (inverse ranking)
        - NaN = undefined (fewer than 3 shared edges)
        
        This metric answers: "For edges that exist in both graphs, are the
        strongest edges in graph A also the strongest in graph B?"
        
        Properties:
        - Scale-invariant: rank-based, doesn't matter if weights differ by 10x
        - Focuses on relative importance, not absolute values
        - Robust to outliers (uses ranks, not raw values)
        
        Args:
            weights_a: Series of weights indexed by edge
            weights_b: Series of weights indexed by edge
            use_shared_edges: If True, only compare edges present in both (default).
                              If False, use union with 0 for missing edges.
            use_normalized: If True, normalize weights to proportions (default: True)
            
        Returns:
            Rank correlation in [-1, 1], or NaN if undefined
        """
        from scipy.stats import spearmanr
        
        if use_shared_edges:
            # Only compare edges that exist in BOTH graphs
            shared_edges = set(weights_a.index) & set(weights_b.index)
            # Filter to edges with positive weight in both
            shared_edges = [e for e in shared_edges 
                           if weights_a.get(e, 0) > 0 and weights_b.get(e, 0) > 0]
            
            if len(shared_edges) < 3:  # Need at least 3 points for meaningful correlation
                return np.nan  # Undefined - return NaN
            
            a_vals = pd.Series([weights_a[e] for e in shared_edges])
            b_vals = pd.Series([weights_b[e] for e in shared_edges])
        else:
            # Use union (old behavior - causes low coefficients)
            all_edges = sorted(set(weights_a.index) | set(weights_b.index))
            a_vals = pd.Series([weights_a.get(e, 0) for e in all_edges])
            b_vals = pd.Series([weights_b.get(e, 0) for e in all_edges])
        
        # Normalize if requested
        if use_normalized:
            total_a = a_vals.sum()
            total_b = b_vals.sum()
            if total_a > 0:
                a_vals = a_vals / total_a
            if total_b > 0:
                b_vals = b_vals / total_b
        
        if len(a_vals) < 2 or a_vals.std() == 0 or b_vals.std() == 0:
            return np.nan  # Undefined
        
        corr, _ = spearmanr(a_vals, b_vals)
        
        if np.isnan(corr):
            return np.nan
        
        return corr  # Return raw correlation in [-1, 1]
    
    def calculate_rv_coefficient(
        self,
        weights_a: pd.Series,
        weights_b: pd.Series,
        use_normalized: bool = True
    ) -> float:
        """
        Calculate RV coefficient between two graph adjacency matrices.
        
        The RV coefficient (Robert & Escoufier, 1976) measures the similarity
        between two multivariate configurations. It's a multivariate 
        generalization of the squared Pearson correlation.
        
        Formula:
            RV = trace(A @ B^T @ B @ A^T) / sqrt(trace(A @ A^T)^2 * trace(B @ B^T)^2)
        
        Or equivalently using vectorized form:
            RV = <A, B>_F^2 / (||A||_F^2 * ||B||_F^2)
        
        where <A, B>_F = trace(A^T @ B) is the Frobenius inner product.
        
        Properties:
        - Range: [0, 1]
        - RV = 1 means matrices are proportional (same pattern)
        - RV = 0 means matrices are orthogonal (no similarity)
        - Scale-invariant: RV(A, B) = RV(kA, B) for any k > 0
        
        Note: RV is inherently scale-invariant, so normalization doesn't change
        the result but is applied for consistency with other weight-sensitive metrics.
        
        Args:
            weights_a: Series of weights indexed by edge (format: "source -> target")
            weights_b: Series of weights indexed by edge (format: "source -> target")
            use_normalized: If True, normalize weights to proportions (default: True)
            
        Returns:
            RV coefficient (0 to 1)
        """
        # Normalize weights if requested
        if use_normalized:
            total_a = weights_a.sum()
            total_b = weights_b.sum()
            norm_a = weights_a / total_a if total_a > 0 else weights_a
            norm_b = weights_b / total_b if total_b > 0 else weights_b
        else:
            norm_a, norm_b = weights_a, weights_b
        
        adj_a, adj_b = self._build_aligned_adjacency_matrices(norm_a, norm_b)
        
        if adj_a.size == 0 or adj_b.size == 0:
            return 0.0
        
        # Center the matrices (subtract mean)
        adj_a_centered = adj_a - np.mean(adj_a)
        adj_b_centered = adj_b - np.mean(adj_b)
        
        # Compute cross-product matrices
        # For adjacency matrices, we use the vectorized form
        a_flat = adj_a_centered.flatten()
        b_flat = adj_b_centered.flatten()
        
        # Frobenius inner product: <A, B>_F = sum(A * B)
        inner_ab = np.sum(a_flat * b_flat)
        norm_a_sq = np.sum(a_flat * a_flat)
        norm_b_sq = np.sum(b_flat * b_flat)
        
        if norm_a_sq == 0 or norm_b_sq == 0:
            return 1.0 if norm_a_sq == norm_b_sq else 0.0
        
        # RV = <A,B>^2 / (||A||^2 * ||B||^2)
        rv = (inner_ab ** 2) / (norm_a_sq * norm_b_sq)
        
        return float(max(0.0, min(1.0, rv)))
    
    def calculate_mantel_test(
        self,
        weights_a: pd.Series,
        weights_b: pd.Series,
        method: str = 'pearson',
        n_permutations: int = 999
    ) -> Tuple[float, float]:
        """
        Perform Mantel test for matrix correlation with permutation significance.
        
        The Mantel test (1967) assesses the correlation between two distance/
        adjacency matrices while accounting for the non-independence of matrix
        elements. It uses permutation to build a null distribution.
        
        Process:
        1. Compute observed correlation between matrices
        2. Randomly permute rows/columns of one matrix n_permutations times
        3. Compute correlation for each permutation
        4. P-value = proportion of permuted correlations >= observed
        
        Args:
            weights_a: Series of weights indexed by edge
            weights_b: Series of weights indexed by edge
            method: 'pearson' or 'spearman'
            n_permutations: Number of permutations (default: 999)
            
        Returns:
            Tuple of (correlation_similarity, p_value)
            - correlation_similarity: (corr + 1) / 2, mapped to [0, 1]
            - p_value: Significance level (< 0.05 = significant)
        """
        adj_a, adj_b = self._build_aligned_adjacency_matrices(weights_a, weights_b)
        
        if adj_a.size == 0 or adj_b.size == 0:
            return 0.0, 1.0
        
        n = adj_a.shape[0]
        if n < 3:
            # Too small for meaningful permutation test: fall back to the
            # plain matrix correlation (mapped to [0, 1]) without permutations.
            a_small = adj_a.flatten()
            b_small = adj_b.flatten()
            if method == 'spearman':
                from scipy.stats import spearmanr
                small_corr, _ = spearmanr(a_small, b_small)
            else:
                small_corr = np.corrcoef(a_small, b_small)[0, 1]
            if np.isnan(small_corr):
                return 0.5, 1.0
            return float((small_corr + 1.0) / 2.0), 1.0
        
        a_flat = adj_a.flatten()
        b_flat = adj_b.flatten()
        
        # Compute observed correlation
        if method == 'spearman':
            from scipy.stats import spearmanr
            observed_corr, _ = spearmanr(a_flat, b_flat)
        else:
            observed_corr = np.corrcoef(a_flat, b_flat)[0, 1]
        
        if np.isnan(observed_corr):
            return 0.5, 1.0
        
        # Permutation test
        count_greater = 0
        rng = np.random.default_rng(42)  # Reproducible results
        
        for _ in range(n_permutations):
            # Permute rows and columns of matrix B simultaneously
            perm = rng.permutation(n)
            adj_b_perm = adj_b[perm, :][:, perm]
            b_perm_flat = adj_b_perm.flatten()
            
            if method == 'spearman':
                perm_corr, _ = spearmanr(a_flat, b_perm_flat)
            else:
                perm_corr = np.corrcoef(a_flat, b_perm_flat)[0, 1]
            
            if not np.isnan(perm_corr) and perm_corr >= observed_corr:
                count_greater += 1
        
        # P-value: proportion of permutations with correlation >= observed
        p_value = (count_greater + 1) / (n_permutations + 1)
        
        # Convert correlation to similarity [0, 1]
        corr_similarity = (observed_corr + 1.0) / 2.0
        
        return float(corr_similarity), float(p_value)
    
    def _parse_edge(self, edge) -> Optional[Tuple[str, str]]:
        """
        Parse an edge into (source, target) tuple.
        
        Handles multiple edge formats:
        - Tuple: ('A', 'B') or ('A', 'B', ...)
        - String: 'A -> B'
        
        Returns:
            (source, target) tuple, or None if parsing fails
        """
        # Handle tuple/MultiIndex format
        if isinstance(edge, tuple):
            if len(edge) >= 2:
                return (str(edge[0]), str(edge[1]))
        # Handle string format
        elif isinstance(edge, str) and ' -> ' in edge:
            parts = edge.split(' -> ')
            if len(parts) == 2:
                return (parts[0], parts[1])
        return None
    
    def _build_aligned_adjacency_matrices(
        self,
        weights_a: pd.Series,
        weights_b: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build two adjacency matrices with the UNION of node vocabularies.
        
        This ensures both matrices have identical dimensions and node ordering,
        making matrix comparison meaningful. Edges missing in one graph will have
        weight 0 in its matrix.
        
        Args:
            weights_a: Series indexed by edge (tuple or "source -> target" string)
            weights_b: Series indexed by edge (tuple or "source -> target" string)
            
        Returns:
            Tuple of (adj_a, adj_b) with same dimensions and node ordering
        """
        # Collect all nodes from both graphs
        all_nodes = set()
        edges_a = []
        edges_b = []
        
        for edge, weight in weights_a.items():
            parsed = self._parse_edge(edge)
            if parsed is not None:
                src, tgt = parsed
                all_nodes.add(src)
                all_nodes.add(tgt)
                edges_a.append((src, tgt, weight))
        
        for edge, weight in weights_b.items():
            parsed = self._parse_edge(edge)
            if parsed is not None:
                src, tgt = parsed
                all_nodes.add(src)
                all_nodes.add(tgt)
                edges_b.append((src, tgt, weight))
        
        if not all_nodes:
            return np.array([]), np.array([])
        
        # Create shared node mapping (same order for both graphs)
        sorted_nodes = sorted(all_nodes)
        node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}
        n = len(sorted_nodes)
        
        # Build adjacency matrices
        adj_a = np.zeros((n, n))
        adj_b = np.zeros((n, n))
        
        for src, tgt, weight in edges_a:
            i = node_to_idx[src]
            j = node_to_idx[tgt]
            adj_a[i, j] = weight
        
        for src, tgt, weight in edges_b:
            i = node_to_idx[src]
            j = node_to_idx[tgt]
            adj_b[i, j] = weight
        
        return adj_a, adj_b
    
    def calculate_cosine_similarity(
        self,
        weights_a: pd.Series,
        weights_b: pd.Series
    ) -> float:
        """
        Calculate cosine similarity between two edge weight vectors (uses UNION).
        
        Cosine similarity measures the angle between two vectors, making it
        scale-invariant and useful for comparing edge weight distributions
        regardless of total connectivity strength.
        
        cos(A, B) = (A · B) / (||A|| * ||B||)
        
        Args:
            weights_a: Series of weights indexed by edge (e.g., "typeA -> typeB")
            weights_b: Series of weights indexed by edge
            
        Returns:
            Cosine similarity score in [0, 1], where 1 = identical direction.
            Returns NaN if both vectors are zero (undefined).
        """
        # Get union of all edges
        all_edges = sorted(set(weights_a.index) | set(weights_b.index))
        
        if len(all_edges) == 0:
            return np.nan  # Both empty = undefined
        
        # Build weight vectors (0 for missing edges)
        # Use _safe_series_get to handle duplicate indices gracefully
        vec_a = np.array([self._safe_series_get(weights_a, e, 0) for e in all_edges], dtype=float)
        vec_b = np.array([self._safe_series_get(weights_b, e, 0) for e in all_edges], dtype=float)
        
        # Compute norms
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        # Handle edge cases
        if norm_a == 0 and norm_b == 0:
            return np.nan  # Both zero vectors = undefined
        if norm_a == 0 or norm_b == 0:
            return 0.0  # One zero vector = orthogonal
        
        # Compute cosine similarity
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    
    def _safe_series_get(self, series: pd.Series, key, default=0) -> float:
        """
        Safely get a value from a pandas Series, handling duplicate indices.
        
        When a Series has duplicate indices, .get() returns a Series instead of
        a scalar. This helper aggregates duplicates by summing them.
        
        Args:
            series: Pandas Series to retrieve value from
            key: Index key to look up
            default: Default value if key not found
            
        Returns:
            Scalar float value
        """
        if key not in series.index:
            return float(default)
        val = series.get(key, default)
        # If duplicate indices exist, .get() returns a Series - sum them
        if isinstance(val, pd.Series):
            return float(val.sum())
        return float(val)
    
    def calculate_edge_list_rank_correlation(
        self,
        weights_a: pd.Series,
        weights_b: pd.Series,
        normalize: bool = False
    ) -> float:
        """
        Calculate rank correlation based on sorted edge lists (uses UNION).
        
        This metric compares the ranking of edges by weight between two graphs.
        Edges are sorted by weight, and Spearman rank correlation is computed
        on the union of edges (with 0 for missing edges).
        
        Note: For cross-dataset comparison, this returns the original Spearman
        correlation in [-1, 1]. For homolog finding and connectivity profile
        comparisons, use normalize=True to get values in [0, 1].
        
        Optimized with polars for high performance on large datasets.
        
        Args:
            weights_a: Series of weights indexed by edge (e.g., "typeA -> typeB")
            weights_b: Series of weights indexed by edge
            normalize: If True, normalize from [-1, 1] to [0, 1] (for homolog finding).
                       If False (default), return original correlation.
            
        Returns:
            Spearman rank correlation in [-1, 1] (or [0, 1] if normalize=True).
            Returns NaN if insufficient data or undefined.
        """
        from scipy.stats import spearmanr
        
        # Fast path: use polars for vectorized join
        # Convert pandas Series to polars DataFrames
        df_a = pl.DataFrame({
            'edge': weights_a.index.tolist(),
            'weight_a': weights_a.values.tolist()
        }).group_by('edge').agg(pl.col('weight_a').sum())  # Handle duplicates
        
        df_b = pl.DataFrame({
            'edge': weights_b.index.tolist(),
            'weight_b': weights_b.values.tolist()
        }).group_by('edge').agg(pl.col('weight_b').sum())  # Handle duplicates
        
        # Full outer join to get union
        merged = df_a.join(df_b, on='edge', how='full', coalesce=True).fill_null(0)
        
        if len(merged) < 3:
            return np.nan  # Insufficient data for correlation
        
        vec_a = merged['weight_a'].to_numpy()
        vec_b = merged['weight_b'].to_numpy()
        
        # Handle edge cases
        if vec_a.std() == 0 or vec_b.std() == 0:
            return np.nan  # Constant = undefined correlation
        
        # Compute Spearman rank correlation
        corr, _ = spearmanr(vec_a, vec_b)
        
        if np.isnan(corr):
            return np.nan
        
        # Return original or normalized
        if normalize:
            return (corr + 1.0) / 2.0
        return corr
    
    def calculate_path_list_rank_correlation(
        self,
        paths_a: pd.Series,
        paths_b: pd.Series,
        normalize: bool = False
    ) -> float:
        """
        Calculate rank correlation based on sorted path lists (uses UNION).
        
        This metric compares the ranking of paths by min_weight between two graphs.
        Paths are sorted by weight, and Spearman rank correlation is computed
        on the union of paths (with 0 for missing paths).
        
        Similar to edge_list_rank_correlation but operates on multi-hop paths
        like "aMe12->KCg-d->PPL103" instead of single edges.
        
        Optimized with polars for high performance on large datasets.
        
        Note: For cross-dataset comparison, this returns the original Spearman
        correlation in [-1, 1]. For homolog finding and connectivity profile
        comparisons, use normalize=True to get values in [0, 1].
        
        Args:
            paths_a: Series of min_weights indexed by path string (e.g., "A->B->C")
            paths_b: Series of min_weights indexed by path string
            normalize: If True, normalize from [-1, 1] to [0, 1] (for homolog finding).
                       If False (default), return original correlation.
            
        Returns:
            Spearman rank correlation in [-1, 1] (or [0, 1] if normalize=True).
            Returns NaN if insufficient data or undefined.
        """
        from scipy.stats import spearmanr
        
        # Fast path: use polars for vectorized join
        # Convert pandas Series to polars DataFrames
        df_a = pl.DataFrame({
            'path': paths_a.index.tolist(),
            'weight_a': paths_a.values.tolist()
        }).group_by('path').agg(pl.col('weight_a').sum())  # Handle duplicates
        
        df_b = pl.DataFrame({
            'path': paths_b.index.tolist(),
            'weight_b': paths_b.values.tolist()
        }).group_by('path').agg(pl.col('weight_b').sum())  # Handle duplicates
        
        # Full outer join to get union
        merged = df_a.join(df_b, on='path', how='full', coalesce=True).fill_null(0)
        
        if len(merged) < 3:
            return np.nan  # Insufficient data for correlation
        
        vec_a = merged['weight_a'].to_numpy()
        vec_b = merged['weight_b'].to_numpy()
        
        # Handle edge cases
        if vec_a.std() == 0 or vec_b.std() == 0:
            return np.nan  # Constant = undefined correlation
        
        # Compute Spearman rank correlation
        corr, _ = spearmanr(vec_a, vec_b)
        
        if np.isnan(corr):
            return np.nan
        
        # Return original or normalized
        if normalize:
            return (corr + 1.0) / 2.0
        return corr
    
    def calculate_graph_kernel_similarity(
        self,
        weights_a: pd.Series,
        weights_b: pd.Series,
        kernel_type: str = 'wl',
        n_iterations: int = 3
    ) -> float:
        """
        Calculate graph similarity using graph kernels.
        
        Implements Weisfeiler-Lehman (WL) subtree kernel for comparing graphs.
        Enhanced to incorporate edge weights by discretizing them into bins.
        
        Args:
            weights_a: Series of weights indexed by edge
            weights_b: Series of weights indexed by edge
            kernel_type: Kernel type ('wl' for Weisfeiler-Lehman, default)
            n_iterations: Number of WL iterations (default: 3)
            
        Returns:
            Normalized kernel similarity in [0, 1]
        """
        # Build graphs with shared vocabulary for fair comparison
        adj_a, labels_a, adj_b, labels_b = self._build_aligned_labeled_graphs(weights_a, weights_b)
        
        if len(labels_a) == 0 or len(labels_b) == 0:
            return 0.0 if (len(labels_a) == 0) != (len(labels_b) == 0) else 1.0
        
        if kernel_type == 'wl':
            # Compute WL feature vectors
            features_a = self._wl_features(adj_a, labels_a, n_iterations)
            features_b = self._wl_features(adj_b, labels_b, n_iterations)
            
            # Compute normalized kernel value (cosine similarity of feature vectors)
            return self._cosine_similarity_dicts(features_a, features_b)
        else:
            return np.nan
    
    def _build_aligned_labeled_graphs(
        self,
        weights_a: pd.Series,
        weights_b: pd.Series,
        weight_bins: int = 5
    ) -> Tuple[Dict[int, List[Tuple[int, str]]], Dict[int, str], Dict[int, List[Tuple[int, str]]], Dict[int, str]]:
        """
        Build two labeled graphs with shared node vocabulary.
        
        Edge weights are discretized into bins and included in edge labels,
        making the WL kernel weight-aware.
        
        Args:
            weights_a: Series indexed by "source -> target"
            weights_b: Series indexed by "source -> target"
            weight_bins: Number of weight discretization bins
            
        Returns:
            Tuple of (adj_a, labels_a, adj_b, labels_b)
        """
        # Collect all nodes and edges from both graphs
        all_nodes = set()
        edges_a = []
        edges_b = []
        all_weights = []
        
        for edge, weight in weights_a.items():
            if ' -> ' in str(edge):
                parts = str(edge).split(' -> ')
                if len(parts) == 2:
                    src, tgt = parts
                    all_nodes.add(src)
                    all_nodes.add(tgt)
                    edges_a.append((src, tgt, weight))
                    all_weights.append(weight)
        
        for edge, weight in weights_b.items():
            if ' -> ' in str(edge):
                parts = str(edge).split(' -> ')
                if len(parts) == 2:
                    src, tgt = parts
                    all_nodes.add(src)
                    all_nodes.add(tgt)
                    edges_b.append((src, tgt, weight))
                    all_weights.append(weight)
        
        if not all_nodes:
            return {}, {}, {}, {}
        
        # Create shared node mapping
        sorted_nodes = sorted(all_nodes)
        node_to_id = {node: i for i, node in enumerate(sorted_nodes)}
        
        # Compute weight bin boundaries (percentile-based)
        if all_weights:
            percentiles = np.percentile(all_weights, np.linspace(0, 100, weight_bins + 1))
        else:
            percentiles = [0, 100]
        
        def get_weight_bin(w):
            for i, p in enumerate(percentiles[1:]):
                if w <= p:
                    return f"w{i}"
            return f"w{len(percentiles)-2}"
        
        # Build adjacency lists with weighted edge labels
        adj_a = defaultdict(list)
        adj_b = defaultdict(list)
        
        for src, tgt, weight in edges_a:
            src_id = node_to_id[src]
            tgt_id = node_to_id[tgt]
            edge_label = get_weight_bin(weight)
            adj_a[src_id].append((tgt_id, edge_label))
            adj_a[tgt_id].append((src_id, edge_label))  # Undirected
        
        for src, tgt, weight in edges_b:
            src_id = node_to_id[src]
            tgt_id = node_to_id[tgt]
            edge_label = get_weight_bin(weight)
            adj_b[src_id].append((tgt_id, edge_label))
            adj_b[tgt_id].append((src_id, edge_label))  # Undirected
        
        # Node labels (same for both graphs - shared vocabulary)
        labels = {node_to_id[node]: node for node in sorted_nodes}
        
        return dict(adj_a), labels, dict(adj_b), labels
    
    def _wl_features(
        self,
        adj: Dict[int, List],
        labels: Dict[int, str],
        n_iterations: int
    ) -> Dict[str, int]:
        """
        Compute Weisfeiler-Lehman subtree features.
        
        Enhanced to handle edge labels (for weight-aware comparison).
        
        Args:
            adj: Adjacency list. Can be:
                 - Dict[int, List[int]] for unweighted
                 - Dict[int, List[Tuple[int, str]]] for weighted (neighbor_id, edge_label)
            labels: Node labels
            n_iterations: Number of WL iterations
            
        Returns:
            Feature vector as dict of {label: count}
        """
        import hashlib
        
        # Initialize features
        features: Dict[str, int] = defaultdict(int)
        current_labels = labels.copy()
        
        # Count initial labels
        for label in current_labels.values():
            features[f"0_{label}"] += 1
        
        # WL iterations
        for iteration in range(1, n_iterations + 1):
            new_labels = {}
            
            for node, label in current_labels.items():
                # Collect neighbor labels (handle both edge-labeled and simple adjacency)
                neighbors = adj.get(node, [])
                neighbor_strs = []
                for n in neighbors:
                    if isinstance(n, tuple):
                        # Edge-labeled: (neighbor_id, edge_label)
                        neighbor_id, edge_label = n
                        neighbor_strs.append(f"{current_labels.get(neighbor_id, '')}_{edge_label}")
                    else:
                        # Simple adjacency
                        neighbor_strs.append(current_labels.get(n, ''))
                
                neighbor_strs = sorted(neighbor_strs)
                
                # Create new label by hashing concatenation
                combined = f"{label}_{'_'.join(neighbor_strs)}"
                # Use hash for consistent compression
                new_label = hashlib.md5(combined.encode()).hexdigest()[:8]
                new_labels[node] = new_label
                
                # Count this label
                features[f"{iteration}_{new_label}"] += 1
            
            current_labels = new_labels
        
        return dict(features)
    
    def _cosine_similarity_dicts(
        self,
        dict_a: Dict[str, int],
        dict_b: Dict[str, int]
    ) -> float:
        """
        Compute cosine similarity between two feature count dicts.
        
        Args:
            dict_a: Feature counts for graph A
            dict_b: Feature counts for graph B
            
        Returns:
            Cosine similarity in [0, 1]
        """
        # Get all features
        all_features = set(dict_a.keys()) | set(dict_b.keys())
        
        if not all_features:
            return 1.0
        
        # Create vectors
        vec_a = np.array([dict_a.get(f, 0) for f in sorted(all_features)])
        vec_b = np.array([dict_b.get(f, 0) for f in sorted(all_features)])
        
        # Compute cosine similarity
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
