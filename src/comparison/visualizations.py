"""
ComparisonVisualizer - Visualization tools for cross-dataset comparison.

This module provides plotting functions for visualizing comparison results
including bar charts, heatmaps, Venn diagrams, and network comparisons.
"""

import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    from ..utils.naming_utils import canonical_dataset_name
except ImportError:  # pragma: no cover - direct package imports
    from utils.naming_utils import canonical_dataset_name

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class ComparisonVisualizer:
    """
    Visualization tools for cross-dataset comparison.
    
    Provides methods for:
    - Path count bar plots
    - Edge weight heatmaps
    - Similarity matrices
    - Venn diagram-style overlap charts
    - Connection comparison scatter plots
    
    Example:
        >>> viz = ComparisonVisualizer()
        >>> fig = viz.plot_path_counts(results, thresholds=[1, 5, 10])
        >>> plt.savefig("path_counts.png")
    """
    
    def __init__(self, style: str = 'whitegrid', verbose: bool = True):
        """
        Initialize ComparisonVisualizer.
        
        Args:
            style: Seaborn style to use (default: 'whitegrid')
            verbose: Whether to print progress messages (default: True)
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
        
        if HAS_SEABORN:
            sns.set_style(style)
        
        self.default_figsize = (10, 6)
        self.default_colors = plt.cm.Set2.colors
        self.verbose = verbose
    
    def _vprint(self, *args, **kwargs):
        """Print only if verbose is True. Uses tqdm.write for progress bar compatibility."""
        if self.verbose:
            from tqdm import tqdm
            tqdm.write(' '.join(str(a) for a in args))
    
    # =========================================================================
    # Path Count Visualizations
    # =========================================================================
    
    def plot_path_counts(
        self,
        results: Dict[str, Dict[int, pd.DataFrame]],
        thresholds: List[int],
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Path Counts by Dataset and Threshold",
        nickname_map: Dict[str, str] = None,
        x_label: str = "Weight Threshold"
    ) -> plt.Figure:
        """
        Plot bar chart of path counts across datasets and thresholds.
        
        Args:
            results: Nested dict {dataset: {threshold: DataFrame}}
            thresholds: List of thresholds to plot
            figsize: Figure size tuple
            title: Plot title
            nickname_map: Dict mapping dataset names to display names
            
        Returns:
            matplotlib Figure
        """
        figsize = figsize or self.default_figsize
        
        # Prepare data
        datasets = list(results.keys())
        if nickname_map is None:
            nickname_map = {d: d for d in datasets}
        n_datasets = len(datasets)
        n_thresholds = len(thresholds)
        
        # Create count matrix
        counts = np.zeros((n_datasets, n_thresholds))
        for i, dataset in enumerate(datasets):
            for j, threshold in enumerate(thresholds):
                if threshold in results[dataset]:
                    counts[i, j] = len(results[dataset][threshold])
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(n_thresholds)
        width = 0.8 / n_datasets
        
        for i, dataset in enumerate(datasets):
            offset = (i - n_datasets / 2 + 0.5) * width
            label = nickname_map.get(dataset, dataset)
            bars = ax.bar(x + offset, counts[i], width, label=label, 
                         color=self.default_colors[i % len(self.default_colors)])
            ax.bar_label(bars, padding=3, fontsize=8)

        ax.set_xlabel(x_label)
        ax.set_ylabel('Number of Paths')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(thresholds)
        ax.legend(title='Dataset')
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)  # Add space for labels
        ax.grid(False)  # Hide grid
        
        plt.tight_layout()
        return fig
    
    def plot_path_counts_stacked(
        self,
        results: Dict[str, Dict[int, pd.DataFrame]],
        thresholds: List[int],
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Path Counts by Threshold (Stacked)"
    ) -> plt.Figure:
        """
        Plot stacked bar chart of path counts.
        
        Args:
            results: Nested dict {dataset: {threshold: DataFrame}}
            thresholds: List of thresholds to plot
            figsize: Figure size tuple
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        figsize = figsize or self.default_figsize
        
        datasets = list(results.keys())
        
        # Create count matrix
        counts = {}
        for dataset in datasets:
            counts[dataset] = [len(results[dataset].get(t, pd.DataFrame())) for t in thresholds]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(thresholds))
        bottom = np.zeros(len(thresholds))
        
        for i, dataset in enumerate(datasets):
            ax.bar(x, counts[dataset], bottom=bottom, label=dataset,
                  color=self.default_colors[i % len(self.default_colors)])
            bottom += np.array(counts[dataset])
        
        ax.set_xlabel('Weight Threshold')
        ax.set_ylabel('Number of Paths')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(thresholds)
        ax.legend(title='Dataset')
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # Heatmap Visualizations
    # =========================================================================
    
    def plot_edge_weight_heatmap(
        self,
        aligned_data: pd.DataFrame,
        datasets: List[str],
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Edge Weights Across Datasets",
        max_edges: int = 50,
        nickname_map: Dict[str, str] = None
    ) -> plt.Figure:
        """
        Plot heatmap of edge weights across datasets.
        
        Args:
            aligned_data: DataFrame with edge index and dataset columns
            datasets: List of dataset column names
            figsize: Figure size tuple
            title: Plot title
            max_edges: Maximum number of edges to show
            nickname_map: Dict mapping dataset names to display names
            
        Returns:
            matplotlib Figure
        """
        if not HAS_SEABORN:
            raise ImportError("seaborn is required for heatmaps. Install with: pip install seaborn")
        
        available = [d for d in datasets if d in aligned_data.columns]
        
        if not available:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        # Select top edges by mean weight
        plot_data = aligned_data[available].copy()
        plot_data['mean'] = plot_data.mean(axis=1)
        plot_data = plot_data.nlargest(max_edges, 'mean').drop(columns=['mean'])
        
        # Rename columns to nicknames if provided
        if nickname_map:
            plot_data = plot_data.rename(columns={d: nickname_map.get(d, d) for d in available})
        
        # Create heatmap
        fig_height = max(6, len(plot_data) * 0.3)
        figsize = figsize or (10, fig_height)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create mask for zeros
        mask_zeros = plot_data == 0
        
        # Create annotation array that hides zeros
        annot_array = plot_data.values.astype(str)
        annot_array[mask_zeros.values] = ''
        
        sns.heatmap(
            plot_data,
            annot=annot_array,
            fmt='',
            cmap='YlOrRd',
            ax=ax,
            cbar_kws={'label': 'Edge Weight'},
            linewidths=0  # Hide grid lines
        )
        
        ax.set_title(title)
        ax.set_ylabel('Edge')
        ax.set_xlabel('Dataset')
        
        plt.tight_layout()
        return fig
    
    def plot_edge_weight_heatmap_all_thresholds(
        self,
        results: Dict[str, Dict[int, pd.DataFrame]],
        thresholds: List[int],
        align_func,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Edge Weights Across Datasets and Thresholds",
        max_edges: int = 30,
        nickname_map: Dict[str, str] = None
    ) -> plt.Figure:
        """
        Plot heatmap of edge weights across ALL datasets and thresholds in one figure.
        
        Each column represents (dataset, threshold) combination.
        This provides a comprehensive view of how edge weights change with different cutoffs.
        
        Args:
            results: Nested dict {dataset: {threshold: DataFrame}}
            thresholds: List of thresholds to include
            align_func: Function to align data at a threshold (ComparisonAnalyzer.get_aligned_data)
            figsize: Figure size tuple
            title: Plot title
            max_edges: Maximum number of edges to show
            nickname_map: Dict mapping dataset names to display names
            
        Returns:
            matplotlib Figure
        """
        if not HAS_SEABORN:
            raise ImportError("seaborn is required for heatmaps. Install with: pip install seaborn")
        
        datasets = list(results.keys())
        if nickname_map is None:
            nickname_map = {d: d for d in datasets}
        
        # First pass: identify top edges from first threshold to limit iteration
        top_edges = []
        sample_threshold = thresholds[0] if thresholds else None
        
        if sample_threshold is not None:
            try:
                aligned = align_func(sample_threshold)
                if not aligned.empty:
                    # Sum across all dataset columns for each edge
                    dataset_cols = [d for d in datasets if d in aligned.columns]
                    if dataset_cols:
                        aligned['_total'] = aligned[dataset_cols].sum(axis=1)
                        top_edges = aligned.nlargest(max_edges * 2, '_total').index.tolist()
                    else:
                        top_edges = list(aligned.index[:max_edges * 2])
            except Exception:
                pass
        
        top_edges_set = set(top_edges)
        
        # Collect data only for top edges across all thresholds
        all_data = {}
        
        for threshold in thresholds:
            try:
                aligned = align_func(threshold)
                if aligned.empty:
                    continue
                
                # Only process top edges
                edges_to_process = [e for e in aligned.index if e in top_edges_set] if top_edges_set else list(aligned.index[:max_edges * 2])
                
                for edge_key in edges_to_process:
                    for dataset in datasets:
                        if dataset in aligned.columns:
                            nick = nickname_map.get(dataset, dataset)[:12]
                            col_name = f"{nick}_t{threshold}"
                            if edge_key not in all_data:
                                all_data[edge_key] = {}
                            all_data[edge_key][col_name] = aligned.loc[edge_key, dataset]
            except Exception:
                continue
        
        if not all_data:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        # Build DataFrame
        heatmap_df = pd.DataFrame(all_data).T.fillna(0)
        
        # Select top edges by mean weight
        heatmap_df['_mean'] = heatmap_df.mean(axis=1)
        heatmap_df = heatmap_df.nlargest(max_edges, '_mean').drop(columns=['_mean'])
        
        # Sort columns by dataset then threshold
        col_order = []
        for dataset in datasets:
            nick = nickname_map.get(dataset, dataset)[:12]
            for threshold in sorted(thresholds):
                col = f"{nick}_t{threshold}"
                if col in heatmap_df.columns:
                    col_order.append(col)
        heatmap_df = heatmap_df[[c for c in col_order if c in heatmap_df.columns]]
        
        # Create figure
        fig_height = max(8, len(heatmap_df) * 0.35)
        fig_width = max(12, len(heatmap_df.columns) * 0.8)
        figsize = figsize or (fig_width, fig_height)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create annotation array that hides zeros
        annot_array = heatmap_df.values.astype(float)
        annot_array_str = np.where(annot_array == 0, '', np.char.mod('%.0f', annot_array))
        
        sns.heatmap(
            heatmap_df,
            annot=annot_array_str,
            fmt='',
            cmap='YlOrRd',
            ax=ax,
            cbar_kws={'label': 'Edge Weight'},
            linewidths=0  # Hide grid lines
        )
        
        ax.set_title(title)
        ax.set_ylabel('Edge')
        ax.set_xlabel('Dataset_Threshold')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def plot_similarity_matrix(
        self,
        similarities: pd.DataFrame,
        metric: str = 'jaccard_similarity',
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Pairwise Dataset Similarity"
    ) -> plt.Figure:
        """
        Plot similarity matrix heatmap.
        
        Args:
            similarities: DataFrame with pairwise similarity data
            metric: Which metric to plot
            figsize: Figure size tuple
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        if not HAS_SEABORN:
            raise ImportError("seaborn is required for heatmaps. Install with: pip install seaborn")
        
        figsize = figsize or (8, 6)
        
        # Get unique datasets
        datasets = sorted(set(similarities['dataset_1'].tolist() + similarities['dataset_2'].tolist()))
        n = len(datasets)
        
        # Create similarity matrix
        matrix = pd.DataFrame(1.0, index=datasets, columns=datasets)
        
        for _, row in similarities.iterrows():
            d1, d2 = row['dataset_1'], row['dataset_2']
            val = row[metric]
            matrix.loc[d1, d2] = val
            matrix.loc[d2, d1] = val
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={'label': metric.replace('_', ' ').title()}
        )
        
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def plot_dual_similarity_matrices(
        self,
        similarities: pd.DataFrame,
        figsize: Optional[Tuple[int, int]] = None,
        title_prefix: str = "Dataset Similarity"
    ) -> plt.Figure:
        """
        Plot both Jaccard and Cosine similarity matrices side by side.
        
        Jaccard measures binary edge presence overlap.
        Cosine measures weighted connection matrix similarity.
        
        Args:
            similarities: DataFrame with pairwise similarity data
            figsize: Figure size tuple
            title_prefix: Prefix for titles
            
        Returns:
            matplotlib Figure with two subplots
        """
        if not HAS_SEABORN:
            raise ImportError("seaborn is required for heatmaps. Install with: pip install seaborn")
        
        figsize = figsize or (14, 6)
        
        # Get unique datasets
        datasets = sorted(set(similarities['dataset_1'].tolist() + similarities['dataset_2'].tolist()))
        
        # Create both similarity matrices
        jaccard_matrix = pd.DataFrame(1.0, index=datasets, columns=datasets)
        cosine_matrix = pd.DataFrame(1.0, index=datasets, columns=datasets)
        
        for _, row in similarities.iterrows():
            d1, d2 = row['dataset_1'], row['dataset_2']
            
            # Jaccard (binary presence)
            jaccard_val = row.get('jaccard_similarity', 0)
            jaccard_matrix.loc[d1, d2] = jaccard_val
            jaccard_matrix.loc[d2, d1] = jaccard_val
            
            # SVD (graph topology similarity)
            svd_val = row.get('svd_similarity', 0)
            if pd.isna(svd_val):
                svd_val = 0
            cosine_matrix.loc[d1, d2] = svd_val
            cosine_matrix.loc[d2, d1] = svd_val
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Left: Jaccard similarity (binary)
        sns.heatmap(
            jaccard_matrix,
            annot=True,
            fmt='.3f',
            cmap='Greens',
            vmin=0,
            vmax=1,
            ax=axes[0],
            cbar_kws={'label': 'Jaccard Index'}
        )
        axes[0].set_title(f'{title_prefix}: Jaccard (Binary Edge Presence)')
        
        # Right: SVD similarity (graph topology)
        sns.heatmap(
            cosine_matrix,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            vmin=0,
            vmax=1,
            ax=axes[1],
            cbar_kws={'label': 'SVD Similarity'}
        )
        axes[1].set_title(f'{title_prefix}: SVD (Graph Topology)')
        
        plt.tight_layout()
        return fig
    
    def plot_similarity_per_threshold(
        self,
        threshold_similarities: Dict[int, pd.DataFrame],
        figsize: Optional[Tuple[int, int]] = None,
        metric: str = 'jaccard'
    ) -> plt.Figure:
        """
        Plot similarity matrices for each threshold level combined in one PNG using subplots.
        
        Args:
            threshold_similarities: Dict mapping threshold -> pairwise similarities DataFrame
            figsize: Figure size tuple (auto-calculated if None)
            metric: 'jaccard' or 'cosine' - which metric to plot
            
        Returns:
            matplotlib Figure with subplots for each threshold
        """
        if not HAS_SEABORN:
            raise ImportError("seaborn is required for heatmaps. Install with: pip install seaborn")
        
        thresholds = sorted(threshold_similarities.keys())
        n_thresholds = len(thresholds)
        
        if n_thresholds == 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, 'No threshold data available', ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig
        
        # Calculate grid layout
        n_cols = min(3, n_thresholds)
        n_rows = (n_thresholds + n_cols - 1) // n_cols
        
        # Auto-calculate figure size
        figsize = figsize or (5 * n_cols, 4 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes_flat = axes.flatten()
        
        # Support multiple metrics with consistent green colormap
        metric_col_map = {
            'jaccard': 'jaccard_similarity',
            'weighted_jaccard': 'weighted_jaccard',
            'svd': 'svd_similarity',
            'edge_rank': 'edge_rank_correlation',
            'kernel': 'kernel_similarity'
        }
        metric_label_map = {
            'jaccard': 'Jaccard Index',
            'weighted_jaccard': 'Weighted Jaccard',
            'svd': 'SVD Similarity',
            'edge_rank': 'Edge Rank Correlation',
            'kernel': 'WL Kernel Similarity'
        }
        
        metric_col = metric_col_map.get(metric, 'jaccard_similarity')
        metric_label = metric_label_map.get(metric, 'Jaccard Index')
        cmap = 'Greens'  # Consistent green colormap for all metrics
        
        for idx, threshold in enumerate(thresholds):
            ax = axes_flat[idx]
            similarities = threshold_similarities[threshold]
            
            if similarities is None or similarities.empty:
                ax.text(0.5, 0.5, f'No data\n(threshold={threshold})', 
                       ha='center', va='center', fontsize=12)
                ax.set_axis_off()
                ax.set_title(f'Threshold = {threshold}', fontsize=11)
                continue
            
            # Get unique datasets
            datasets = sorted(set(
                similarities['dataset_1'].tolist() + similarities['dataset_2'].tolist()
            ))
            
            # Create similarity matrix
            matrix = pd.DataFrame(1.0, index=datasets, columns=datasets)
            
            for _, row in similarities.iterrows():
                d1, d2 = row['dataset_1'], row['dataset_2']
                val = row.get(metric_col, 0)
                if pd.isna(val):
                    val = 0
                matrix.loc[d1, d2] = val
                matrix.loc[d2, d1] = val
            
            # Plot heatmap with square cells
            sns.heatmap(
                matrix,
                annot=True,
                fmt='.2f',
                cmap=cmap,
                vmin=0,
                vmax=1,
                ax=ax,
                cbar=True,
                cbar_kws={'label': metric_label, 'shrink': 0.7},
                annot_kws={'fontsize': 9},
                square=True,  # Make cells square
                linewidths=0  # Hide grid lines
            )
            ax.set_title(f'Threshold = {threshold}', fontsize=11, fontweight='bold')
            ax.tick_params(axis='both', labelsize=8)
        
        # Hide unused axes
        for idx in range(n_thresholds, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        fig.suptitle(f'{metric_label} Matrix by Threshold', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # Overlap/Venn Visualizations
    # =========================================================================
    
    def plot_edge_overlap(
        self,
        aligned_data: pd.DataFrame,
        datasets: List[str],
        threshold: int = 1,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Edge Overlap Between Datasets"
    ) -> plt.Figure:
        """
        Plot edge overlap as a bar chart showing common/unique edges.
        
        Args:
            aligned_data: DataFrame with edge weights per dataset
            datasets: List of dataset column names
            threshold: Minimum weight to consider edge present
            figsize: Figure size tuple
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        figsize = figsize or (10, 6)
        available = [d for d in datasets if d in aligned_data.columns]
        
        if len(available) < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Need at least 2 datasets", ha='center', va='center')
            return fig
        
        # Calculate overlap statistics
        edge_sets = {}
        for d in available:
            edge_sets[d] = set(aligned_data[aligned_data[d] >= threshold].index)
        
        # Get common and unique counts
        all_edges = set.union(*edge_sets.values())
        common_edges = set.intersection(*edge_sets.values())
        
        stats = []
        stats.append({'Category': 'Common (all datasets)', 'Count': len(common_edges)})
        
        for d in available:
            unique = edge_sets[d] - set.union(*[edge_sets[x] for x in available if x != d])
            stats.append({'Category': f'Unique to {d}', 'Count': len(unique)})
        
        stats.append({'Category': 'Total unique edges', 'Count': len(all_edges)})
        
        stats_df = pd.DataFrame(stats)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['green'] + [self.default_colors[i] for i in range(len(available))] + ['gray']
        ax.barh(stats_df['Category'], stats_df['Count'], color=colors)
        
        for i, (count, cat) in enumerate(zip(stats_df['Count'], stats_df['Category'])):
            ax.text(count + 1, i, str(count), va='center')
        
        ax.set_xlabel('Number of Edges')
        ax.set_title(title)
        ax.grid(False)  # Hide grid
        
        plt.tight_layout()
        return fig
    
    def plot_venn2_style(
        self,
        aligned_data: pd.DataFrame,
        datasets: List[str],
        threshold: int = 1,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Edge Overlap (2 Datasets)"
    ) -> plt.Figure:
        """
        Plot 2-set Venn diagram style overlap.
        
        Args:
            aligned_data: DataFrame with edge weights per dataset
            datasets: List of 2 dataset names
            threshold: Minimum weight to consider edge present
            figsize: Figure size tuple
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        if len(datasets) != 2:
            raise ValueError("Venn2 requires exactly 2 datasets")
        
        available = [d for d in datasets if d in aligned_data.columns]
        if len(available) != 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Both datasets must be in aligned_data", ha='center', va='center')
            return fig
        
        figsize = figsize or (8, 8)
        
        # Calculate sets
        set_a = set(aligned_data[aligned_data[available[0]] >= threshold].index)
        set_b = set(aligned_data[aligned_data[available[1]] >= threshold].index)
        
        only_a = len(set_a - set_b)
        only_b = len(set_b - set_a)
        both = len(set_a & set_b)
        
        # Create plot with circles
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw circles
        circle_a = plt.Circle((0.35, 0.5), 0.25, alpha=0.5, color='blue', label=available[0])
        circle_b = plt.Circle((0.65, 0.5), 0.25, alpha=0.5, color='orange', label=available[1])
        
        ax.add_patch(circle_a)
        ax.add_patch(circle_b)
        
        # Add text
        ax.text(0.25, 0.5, str(only_a), ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0.50, 0.5, str(both), ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0.75, 0.5, str(only_b), ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Labels
        ax.text(0.35, 0.85, available[0], ha='center', va='center', fontsize=12, color='blue')
        ax.text(0.65, 0.85, available[1], ha='center', va='center', fontsize=12, color='orange')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title)
        
        return fig
    
    # =========================================================================
    # Scatter/Comparison Plots
    # =========================================================================
    
    def plot_weight_scatter(
        self,
        aligned_data: pd.DataFrame,
        dataset_x: str,
        dataset_y: str,
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        log_scale: bool = False
    ) -> plt.Figure:
        """
        Plot scatter of edge weights between two datasets.
        
        Args:
            aligned_data: DataFrame with edge weights per dataset
            dataset_x: Dataset for x-axis
            dataset_y: Dataset for y-axis
            figsize: Figure size tuple
            title: Plot title
            log_scale: Use log scale
            
        Returns:
            matplotlib Figure
        """
        figsize = figsize or (8, 8)
        title = title or f"{dataset_x} vs {dataset_y} Edge Weights"
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = aligned_data[dataset_x]
        y = aligned_data[dataset_y]
        
        # Add small offset for log scale
        if log_scale:
            x = x + 1
            y = y + 1
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        ax.scatter(x, y, alpha=0.5, s=20)
        
        # Add diagonal line
        max_val = max(x.max(), y.max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
        
        # Add correlation
        corr = x.corr(y)
        ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}', transform=ax.transAxes,
               fontsize=10, verticalalignment='top')
        
        ax.set_xlabel(dataset_x + (' (log)' if log_scale else ''))
        ax.set_ylabel(dataset_y + (' (log)' if log_scale else ''))
        ax.set_title(title)
        ax.grid(False)  # Hide grid
        
        plt.tight_layout()
        return fig
    
    def plot_fold_change_distribution(
        self,
        differential: pd.DataFrame,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Distribution of Fold Changes"
    ) -> plt.Figure:
        """
        Plot histogram of fold changes.
        
        Args:
            differential: DataFrame with max_fold_change column
            figsize: Figure size tuple
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        figsize = figsize or (10, 6)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if 'max_fold_change' not in differential.columns:
            ax.text(0.5, 0.5, "No fold change data", ha='center', va='center')
            return fig
        
        fc = differential['max_fold_change']
        
        ax.hist(fc, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(2.0, color='red', linestyle='--', label='2x threshold')
        
        ax.set_xlabel('Maximum Fold Change')
        ax.set_ylabel('Number of Edges')
        ax.set_title(title)
        ax.legend()
        ax.grid(False)  # Hide grid
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # Multi-threshold Visualizations
    # =========================================================================
    
    def plot_similarity_vs_threshold(
        self,
        threshold_similarities: pd.DataFrame,
        metric: str = 'jaccard_similarity',
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Similarity vs Threshold"
    ) -> plt.Figure:
        """
        Plot similarity metrics across thresholds.
        
        Args:
            threshold_similarities: DataFrame with threshold and metric columns
            metric: Which metric to plot
            figsize: Figure size tuple
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        figsize = figsize or (10, 6)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if threshold_similarities.empty:
            ax.text(0.5, 0.5, "No similarity data", ha='center', va='center')
            return fig
        
        # Group by dataset pair
        for (d1, d2), group in threshold_similarities.groupby(['dataset_1', 'dataset_2']):
            label = f"{d1} vs {d2}"
            ax.plot(group['threshold'], group[metric], 'o-', label=label)
        
        ax.set_xlabel('Weight Threshold')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(False)  # Hide grid
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # Auto threshold-density alignment (plan §5 Phase E)
    # =========================================================================

    def plot_density_alignment_curves(
        self,
        density_curves: pd.DataFrame,
        windows: Optional[pd.DataFrame] = None,
        aligned: Optional[pd.DataFrame] = None,
        nickname_map: Optional[Dict[str, str]] = None,
        normalizer_label: str = 'edges per searched node',
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Query-scoped density vs Min Synapse Count",
    ) -> plt.Figure:
        """Two stacked panels vs threshold (plan §4.5b).

        Top: normalized edge density ``E(t)/N`` (primary, the alignment
        reference) with horizontal guides at the aligned density levels.
        Bottom: enumerated path count (diagnostic only — hub-inflated).
        Each dataset is drawn over its OWN domain ``[w_start, w_star]``
        (never truncated to the intersection) and its endpoints marked.
        """
        figsize = figsize or (11, 8)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        def _label(ds: str) -> str:
            return (nickname_map or {}).get(ds, ds)

        if density_curves is None or density_curves.empty:
            for ax in (ax1, ax2):
                ax.text(0.5, 0.5, "No density data", ha='center',
                        va='center')
            return fig

        for ds, group in density_curves.groupby('dataset'):
            group = group.sort_values('threshold')
            ax1.plot(group['threshold'], group['density'], 'o-',
                     label=_label(ds), markersize=3)
            ax2.plot(group['threshold'], group['path_count'], 'o--',
                     label=_label(ds), markersize=3, alpha=0.8)

        # Aligned density-level guides (horizontal) on the primary panel.
        if aligned is not None and not aligned.empty:
            levels = aligned[
                aligned.get('mode', pd.Series(dtype=str)).astype(str)
                .str.contains('horizontal', na=False)]
            for _, row in levels.iterrows():
                lvl = row.get('level_normalized')
                if lvl is not None and np.isfinite(float(lvl)):
                    ax1.axhline(float(lvl), color='grey', ls=':',
                                alpha=0.35, lw=1.0, zorder=0)

        # Window endpoints per dataset (own domain, §4.5b item 3).
        if windows is not None and not windows.empty:
            for _, row in windows.iterrows():
                for edge, style in (('w_start', '|'), ('w_star_measured', '|')):
                    val = row.get(edge)
                    if val is None or (isinstance(val, float) and
                                       not np.isfinite(val)):
                        continue
                    for ax in (ax1, ax2):
                        ax.axvline(float(val), color='grey', ls='-',
                                   alpha=0.20, lw=0.8, zorder=0)

        ax1.set_ylabel(f"Edge density ({normalizer_label})")
        ax1.set_title('Primary: normalized edge density (E(t)/N, fixed N)')
        ax2.set_ylabel('Enumerated paths (diagnostic)')
        ax2.set_xlabel('Min Synapse Count (per-connection weight)')
        ax2.set_title('Diagnostic: enumerated path count (hub-inflated)')
        for ax in (ax1, ax2):
            ax.grid(False)
        handles, labels = ax1.get_legend_handles_labels()
        if labels:
            fig.legend(handles, labels, loc='upper center',
                       ncol=min(len(labels), 6), frameon=False,
                       bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(title, y=1.06 if labels else 1.0)
        plt.tight_layout()
        return fig

    def plot_edge_density_curves(
        self,
        density_df: pd.DataFrame,
        nickname_map: Optional[Dict[str, str]] = None,
        best_matches: Optional[pd.DataFrame] = None,
        log_y: bool = False,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Edge density vs synapse threshold"
    ) -> plt.Figure:
        """Edge-density-vs-threshold curves (alignment spec, Feature D).

        One line per dataset over the prober's extended threshold grid:
        left panel = distinct type-pair count, right panel = pairs per
        neuron (the cross-dataset comparable normalization). Typed
        thresholds are marked; vertical guides show each dataset's
        best-matching thresholds for the anchors' typed points.

        Args:
            density_df: edge_density_per_threshold.csv content
                (dataset, threshold, pair_count, pairs_per_neuron,
                is_typed_threshold).
            nickname_map: dataset -> short display name.
            best_matches: threshold_alignment_best_matches.csv content —
                anchor rows add guides at each target dataset's best_t.
            log_y: logarithmic y axis (counts span orders of magnitude).
        """
        figsize = figsize or (14, 6)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        if density_df is None or density_df.empty:
            for ax in (ax1, ax2):
                ax.text(0.5, 0.5, "No edge density data", ha='center', va='center')
            return fig

        def _label(ds: str) -> str:
            return (nickname_map or {}).get(ds, ds)

        for ds, group in density_df.groupby('dataset'):
            group = group.sort_values('threshold')
            typed = group[group['is_typed_threshold'] == True]  # noqa: E712
            style = 'o-' if len(typed) else 'o:'
            ax1.plot(group['threshold'], group['pair_count'], style,
                     label=_label(ds), markersize=4)
            if group['pairs_per_neuron'].notna().any():
                ax2.plot(group['threshold'], group['pairs_per_neuron'], style,
                         label=_label(ds), markersize=4)
            for _, trow in typed.iterrows():
                for ax in (ax1, ax2):
                    ax.axvline(trow['threshold'], color='grey',
                               alpha=0.18, lw=0.8, zorder=0)

        # Best-match guides: for the anchor rows, mark each target
        # dataset's best_t with a dashed vertical in its own color.
        if best_matches is not None and not best_matches.empty:
            anchors = best_matches[best_matches.get('match_kind') == 'anchor']
            for _, row in anchors.iterrows():
                ax1.axvline(row['best_t'], color='grey', ls='--',
                            alpha=0.35, lw=1.0, zorder=0)

        for ax, ylabel in ((ax1, 'Distinct connection pairs (type level)'),
                           (ax2, 'Pairs per neuron')):
            ax.set_xlabel('Synapse weight threshold')
            ax.set_ylabel(ylabel)
            if log_y:
                ax.set_yscale('log')
            ax.grid(False)

        ax1.set_title('Edge count per threshold')
        ax2.set_title('Normalized per neuron (density-aligned)')
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center',
                   ncol=min(len(labels), 6), frameon=False,
                   bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(title, y=1.12 if labels else 1.0)
        plt.tight_layout()
        return fig

    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def save_figure(self, fig: plt.Figure, path: str, dpi: int = 600):
        """
        Save figure to file.
        
        Args:
            fig: matplotlib Figure
            path: Output path
            dpi: Resolution (default 450 for high quality)
            silent: If True, suppress "Saved" message
        """
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        # Print only the filename, not the full path
        if not hasattr(self, '_silent_mode') or not self._silent_mode:
            filename = os.path.basename(path)
            self._vprint(f"Saved: {filename}")
    
    def plot_threshold_comparison_subplots(
        self,
        results: Dict[str, Dict[int, pd.DataFrame]],
        thresholds: List[int],
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Comparison Across Threshold Levels",
        threshold_mode: str = 'standard',
    ) -> plt.Figure:
        """
        Create merged subplot visualization showing comparison at different thresholds.
        
        Creates a grid of subplots with:
        - Row 1: Edge counts per threshold per dataset
        - Row 2: Edge weight distributions (boxplots)
        - Row 3: Conservation metrics per threshold
        
        Args:
            results: Nested dict {dataset: {threshold: DataFrame}}
            thresholds: List of thresholds to visualize
            figsize: Figure size tuple
            title: Overall figure title
            threshold_mode: 'combinations' relabels the per-column titles as
                query points (the keys are query ids, not numeric thresholds).
            
        Returns:
            matplotlib Figure with subplots
        """
        datasets = list(results.keys())
        n_thresholds = len(thresholds)
        n_datasets = len(datasets)
        
        # Calculate figure size based on content
        if figsize is None:
            figsize = (max(12, n_thresholds * 3), 12)
        
        # Create figure with 3 rows
        fig, axes = plt.subplots(3, n_thresholds, figsize=figsize)
        if n_thresholds == 1:
            axes = axes.reshape(3, 1)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        # Use distinct colors for datasets
        colors = plt.cm.Set2(np.linspace(0, 1, n_datasets))
        color_map = {ds: colors[i] for i, ds in enumerate(datasets)}
        
        for i, threshold in enumerate(thresholds):
            # Row 1: Edge counts
            ax1 = axes[0, i]
            edge_counts = []
            for j, dataset in enumerate(datasets):
                df = results.get(dataset, {}).get(threshold, pd.DataFrame())
                count = len(df) if not df.empty else 0
                edge_counts.append(count)
            
            bars = ax1.bar(range(n_datasets), edge_counts, color=colors)
            if threshold_mode == 'combinations':
                # The keys are query ids (e.g. combo_001), not numeric
                # thresholds — never phrase them as "Threshold = ...".
                col_title = f'Query = {threshold}'
            else:
                col_title = f'Threshold = {threshold}'
            ax1.set_title(col_title, fontsize=10)
            ax1.set_xticks(range(n_datasets))
            ax1.set_xticklabels([d[:15] for d in datasets], rotation=45, ha='right', fontsize=7)
            ax1.set_ylabel('Edge Count')
            ax1.grid(False)  # Hide grid
            for bar, count in zip(bars, edge_counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(count),
                        ha='center', va='bottom', fontsize=8)
            
            # Add legend only to first plot in row 1
            if i == 0:
                legend_patches = [plt.Rectangle((0,0),1,1, facecolor=colors[j], label=datasets[j][:20]) 
                                 for j in range(n_datasets)]
                ax1.legend(handles=legend_patches, fontsize=7, loc='upper right', framealpha=0.9)
            
            # Row 2: Weight distributions (boxplots)
            ax2 = axes[1, i]
            
            # Collect weight data for boxplot
            weight_df_list = []
            for j, dataset in enumerate(datasets):
                df = results.get(dataset, {}).get(threshold, pd.DataFrame())
                if not df.empty and 'weight' in df.columns:
                    weights = df['weight'].dropna().values
                    for w in weights:
                        weight_df_list.append({'dataset': dataset, 'weight': w})
            
            if weight_df_list:
                weight_df = pd.DataFrame(weight_df_list)
                
                # Create boxplot (swarm removed for cleaner visualization)
                if HAS_SEABORN:
                    sns.boxplot(x='dataset', y='weight', data=weight_df, ax=ax2,
                               hue='dataset', palette=color_map, legend=False,
                               width=0.5, linewidth=1.5)
                else:
                    # Fallback to matplotlib boxplot
                    bp = ax2.boxplot([weight_df[weight_df['dataset']==d]['weight'].values 
                                     for d in datasets], labels=datasets, patch_artist=True)
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
            
            ax2.set_ylabel('Weight')
            ax2.set_xlabel('')
            ax2.tick_params(axis='x', rotation=45, labelsize=7)
            # Set ticks first, then labels to avoid matplotlib warning
            ax2.set_xticks(range(len(datasets)))
            ax2.set_xticklabels([d[:15] for d in datasets])
            ax2.grid(False)  # Hide grid
            
            # Row 3: Conservation / overlap
            ax3 = axes[2, i]
            # Calculate overlap between datasets
            edge_sets = {}
            for dataset in datasets:
                df = results.get(dataset, {}).get(threshold, pd.DataFrame())
                if not df.empty:
                    if 'type_pre' in df.columns and 'type_post' in df.columns:
                        edge_sets[dataset] = set(zip(df['type_pre'], df['type_post']))
                    else:
                        edge_sets[dataset] = set(range(len(df)))
                else:
                    edge_sets[dataset] = set()
            
            # Calculate pairwise overlaps
            if len(datasets) >= 2:
                all_edges = set()
                for s in edge_sets.values():
                    all_edges.update(s)
                
                # Common to all
                if edge_sets:
                    common_edges = edge_sets[datasets[0]].copy()
                    for s in edge_sets.values():
                        common_edges &= s
                else:
                    common_edges = set()
                
                unique_counts = []
                for dataset in datasets:
                    unique = edge_sets[dataset] - common_edges
                    unique_counts.append(len(unique))
                
                # Stacked bar: common + unique
                common_count = len(common_edges)
                x = range(n_datasets)
                common_bar = ax3.bar(x, [common_count] * n_datasets, color='#2ca02c', label='Common to all', alpha=0.8)
                unique_bars = ax3.bar(x, unique_counts, bottom=[common_count] * n_datasets, 
                                     color=colors, alpha=0.8, label='Unique')
                ax3.set_xticks(x)
                ax3.set_xticklabels([d[:15] for d in datasets], rotation=45, ha='right', fontsize=7)
                ax3.set_ylabel('Edges')
                ax3.grid(False)  # Hide grid
                
                # Add legend only to first plot with proper labels
                if i == 0:
                    # Create custom legend with common and unique for each dataset
                    from matplotlib.patches import Patch
                    legend_elements = [Patch(facecolor='#2ca02c', alpha=0.8, label='Common to all')]
                    for j, ds in enumerate(datasets):
                        legend_elements.append(Patch(facecolor=colors[j], alpha=0.8, 
                                                    label=f'Unique to {ds[:15]}'))
                    ax3.legend(handles=legend_elements, fontsize=6, loc='upper right', framealpha=0.9)
        
        plt.tight_layout()
        return fig
    
    def plot_similarity_matrix_per_threshold(
        self,
        results: Dict[str, Dict[int, pd.DataFrame]],
        thresholds: List[int],
        align_func,
        similarity_func=None,
        path_data_func=None,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Dataset Similarity at Each Threshold",
        show_progress: bool = True
    ) -> plt.Figure:
        """
        Plot similarity matrices for each threshold in a combined subplot figure.
        
        Creates a grid of similarity matrices showing the 4 v2.2
        representatives per threshold:
        - Edge: Jaccard, Cosine
        - Path: Path Jaccard
        - Graph: NetSimile-lite
        
        Args:
            results: Nested dict {dataset: {threshold: DataFrame}}
            thresholds: List of thresholds to include
            align_func: Function to get aligned data at a threshold
            similarity_func: Optional function to get cached similarities (avoids recalculation)
            path_data_func: Optional function to get path data at a threshold (for path rank)
            figsize: Figure size tuple
            title: Overall figure title
            show_progress: Whether to show progress bar (default True)
            
        Returns:
            matplotlib Figure with subplots
        """
        if not HAS_SEABORN:
            raise ImportError("seaborn is required for heatmaps")
        
        from .metrics import ComparisonMetrics
        metrics = ComparisonMetrics()
        
        n_thresholds = len(thresholds)
        # 4 v2.2 representatives per threshold: Jaccard, Cosine (edge);
        # Path Jaccard (path); NetSimile-lite (graph)
        n_metrics = 4
        n_cols = n_thresholds
        n_rows = n_metrics

        if figsize is None:
            figsize = (n_cols * 3.5, n_rows * 3)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

        datasets = list(results.keys())

        # Metric definitions with progress info — all [0, 1], single-hue
        # (plan-similarity-matrix-schema-v2 §3.1.1: level chips live in the
        # HTML report; the static grid stays uniform green)
        metric_configs = [
            ('jaccard_similarity', 'Jaccard', 'Greens', 0, 1),
            ('cosine_similarity', 'Cosine (union)', 'Greens', 0, 1),
            ('path_jaccard_similarity', 'Path Jaccard', 'Greens', 0, 1),
            ('netsimile_similarity', 'NetSimile-lite', 'Greens', 0, 1),
        ]
        
        # Use progress bar for threshold iteration
        threshold_iter = enumerate(thresholds)
        if show_progress and len(thresholds) > 1:
            threshold_iter = tqdm(
                list(enumerate(thresholds)),
                desc="Generating similarity heatmaps",
                unit="threshold",
                leave=False
            )
        
        for col_idx, threshold in threshold_iter:
            # Update progress with current threshold and metric
            if show_progress and len(thresholds) > 1:
                threshold_iter.set_postfix({"t": threshold})
            
            try:
                aligned = align_func(threshold)
                if aligned.empty:
                    continue
                
                available = [d for d in datasets if d in aligned.columns]
                
                # Use cached similarities if available, otherwise calculate
                # Get path data for this threshold
                path_data_t = None
                if path_data_func:
                    try:
                        path_data_t = path_data_func(threshold)
                    except Exception:
                        pass
                
                if similarity_func:
                    similarities = similarity_func(threshold)
                    # Add advanced metrics if not present (v2.2 representatives)
                    if similarities.empty or 'netsimile_similarity' not in similarities.columns or 'cosine_similarity' not in similarities.columns:
                        similarities = metrics.calculate_all_pairwise_similarities(
                            aligned, datasets, threshold=1, include_advanced_metrics=True, path_data=path_data_t
                        )
                else:
                    similarities = metrics.calculate_all_pairwise_similarities(
                        aligned, datasets, threshold=1, include_advanced_metrics=True, path_data=path_data_t
                    )
                
                if similarities.empty:
                    continue
                
                for row_idx, (metric_col, metric_label, cmap, vmin, vmax) in enumerate(metric_configs):
                    ax = axes[row_idx, col_idx]
                    
                    # Create similarity matrix - use NaN for undefined values
                    # Diagonal is always 1.0 for all metrics
                    matrix = pd.DataFrame(np.nan, index=available, columns=available)
                    for d in available:
                        matrix.loc[d, d] = 1.0
                    
                    for _, row in similarities.iterrows():
                        d1, d2 = row['dataset_1'], row['dataset_2']
                        if d1 in available and d2 in available:
                            val = row.get(metric_col, np.nan)
                            matrix.loc[d1, d2] = val
                            matrix.loc[d2, d1] = val
                    
                    # Plot heatmap with square cells
                    # Use custom annotation to show "N/A" for NaN values
                    annot_matrix = matrix.copy()
                    sns.heatmap(matrix, annot=True, fmt='.2f', cmap=cmap,
                               vmin=vmin, vmax=vmax, ax=ax, cbar=False,
                               xticklabels=[d[:12] for d in available],
                               yticklabels=[d[:12] for d in available],
                               square=True, linewidths=0)
                    
                    # Title: metric name for first row, threshold for first col
                    if col_idx == 0:
                        ax.set_ylabel(metric_label, fontsize=10, fontweight='bold')
                    if row_idx == 0:
                        ax.set_title(f't={threshold}', fontsize=10, fontweight='bold')
                    ax.tick_params(axis='both', labelsize=7)
                    
            except Exception as e:
                self._vprint(f"Warning: Could not create similarity matrix for threshold {threshold}: {e}")
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # Path-Level Heatmaps
    # =========================================================================
    
    def plot_path_heatmap(
        self,
        path_data: pd.DataFrame,
        datasets: List[str],
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Path Min-Weight Across Datasets",
        max_paths: int = 50,
        nickname_map: Dict[str, str] = None
    ) -> plt.Figure:
        """
        Plot heatmap of path min_weight values across datasets.
        
        Args:
            path_data: DataFrame with path index and dataset columns containing min_weight
            datasets: List of dataset column names
            figsize: Figure size tuple
            title: Plot title
            max_paths: Maximum number of paths to show
            nickname_map: Dict mapping dataset names to display names
            
        Returns:
            matplotlib Figure
        """
        if not HAS_SEABORN:
            raise ImportError("seaborn is required for heatmaps")
        
        available = [d for d in datasets if d in path_data.columns]
        
        if not available:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        # Select top paths by mean min_weight
        plot_data = path_data[available].copy()
        plot_data['mean'] = plot_data.mean(axis=1)
        plot_data = plot_data.nlargest(max_paths, 'mean').drop(columns=['mean'])
        
        # Rename columns to nicknames if provided
        if nickname_map:
            plot_data = plot_data.rename(columns={d: nickname_map.get(d, d) for d in available})
        
        # Create heatmap
        fig_height = max(6, len(plot_data) * 0.3)
        figsize = figsize or (10, fig_height)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create annotation array that hides zeros
        annot_array = plot_data.values.astype(float)
        annot_array_str = np.where(annot_array == 0, '', np.char.mod('%.0f', annot_array))
        
        sns.heatmap(
            plot_data,
            annot=annot_array_str,
            fmt='',
            cmap='Blues',
            ax=ax,
            cbar_kws={'label': 'Path Min-Weight'},
            linewidths=0  # Hide grid lines
        )
        
        ax.set_title(title)
        ax.set_ylabel('Path')
        ax.set_xlabel('Dataset')
        
        plt.tight_layout()
        return fig
    
    def plot_path_heatmap_all_thresholds(
        self,
        path_data_func,
        thresholds: List[int],
        datasets: List[str],
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Path Min-Weights Across Thresholds",
        max_paths: int = 30,
        nickname_map: Dict[str, str] = None
    ) -> plt.Figure:
        """
        Plot heatmap of path min_weight across all thresholds.
        
        Args:
            path_data_func: Function that returns path data for a given threshold
            thresholds: List of thresholds
            datasets: List of dataset names
            figsize: Figure size tuple
            title: Plot title
            max_paths: Maximum number of paths to show
            nickname_map: Dict mapping dataset names to display names
            
        Returns:
            matplotlib Figure
        """
        if not HAS_SEABORN:
            raise ImportError("seaborn is required for heatmaps")
        
        if nickname_map is None:
            nickname_map = {d: d for d in datasets}
        
        # Helper to safely extract scalar from potentially duplicate-indexed DataFrame
        def safe_get_value(df, row_key, col_key):
            """Extract scalar value, handling duplicate indices by summing."""
            try:
                val = df.loc[row_key, col_key]
                # If duplicate indices exist, .loc returns a Series - sum them
                if isinstance(val, pd.Series):
                    return float(val.sum())
                return float(val)
            except (KeyError, TypeError, ValueError):
                return 0.0
        
        # First pass: collect path means across all thresholds to find top paths
        # This avoids building full data for millions of paths
        path_totals = {}
        sample_threshold = thresholds[0] if thresholds else None
        
        if sample_threshold is not None:
            try:
                path_df = path_data_func(sample_threshold)
                if not path_df.empty and len(datasets) > 0:
                    # Sum across all dataset columns for each path
                    dataset_cols = [d for d in datasets if d in path_df.columns]
                    if dataset_cols:
                        path_df['_total'] = path_df[dataset_cols].sum(axis=1)
                        # Get top N paths
                        top_paths = path_df.nlargest(max_paths * 2, '_total').index.tolist()
                    else:
                        top_paths = list(path_df.index[:max_paths * 2])
                else:
                    top_paths = []
            except Exception:
                top_paths = []
        else:
            top_paths = []
        
        top_paths_set = set(top_paths)
        
        # Collect data only for top paths across all thresholds
        all_data = {}
        
        for threshold in thresholds:
            try:
                path_df = path_data_func(threshold)
                if path_df.empty:
                    continue
                
                # Only process top paths (or all if fewer than limit)
                paths_to_process = [p for p in path_df.index if p in top_paths_set] if top_paths_set else list(path_df.index[:max_paths * 2])
                
                for path_key in paths_to_process:
                    for dataset in datasets:
                        if dataset in path_df.columns:
                            nick = nickname_map.get(dataset, dataset)[:10]
                            col_name = f"{nick}_t{threshold}"
                            if path_key not in all_data:
                                all_data[path_key] = {}
                            all_data[path_key][col_name] = safe_get_value(path_df, path_key, dataset)
            except Exception:
                pass
        
        if not all_data:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No path data available", ha='center', va='center')
            return fig
        
        # Create DataFrame and ensure numeric dtype
        df = pd.DataFrame(all_data).T.fillna(0)
        # Convert all columns to float to avoid object dtype issues
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Select top paths by mean
        df['mean'] = df.mean(axis=1)
        df = df.nlargest(max_paths, 'mean').drop(columns=['mean'])
        
        # Sort columns by dataset then threshold (same as edge heatmap)
        col_order = []
        for dataset in datasets:
            nick = nickname_map.get(dataset, dataset)[:10]
            for threshold in sorted(thresholds):
                col = f"{nick}_t{threshold}"
                if col in df.columns:
                    col_order.append(col)
        df = df[[c for c in col_order if c in df.columns]]
        
        # Create heatmap
        fig_height = max(8, len(df) * 0.35)
        fig_width = max(12, len(df.columns) * 0.6)
        figsize = figsize or (fig_width, fig_height)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create annotation array that hides zeros
        if len(df) <= 30:
            annot_array = df.values.astype(float)
            annot_array_str = np.where(annot_array == 0, '', np.char.mod('%.0f', annot_array))
        else:
            annot_array_str = False
        
        sns.heatmap(
            df,
            annot=annot_array_str,
            fmt='',
            cmap='Blues',
            ax=ax,
            cbar_kws={'label': 'Path Min-Weight'},
            linewidths=0  # Hide grid lines
        )
        
        ax.set_title(title)
        ax.set_ylabel('Path')
        ax.set_xlabel('Dataset @ Threshold')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def plot_ratio_heatmap(
        self,
        ratio_data: pd.DataFrame,
        datasets: List[str],
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Path Ratio Across Datasets",
        max_edges: int = 50,
        nickname_map: Dict[str, str] = None
    ) -> plt.Figure:
        """
        Plot heatmap of min_ratio values (path-level statistics).
        
        Args:
            ratio_data: DataFrame with edge index and dataset columns containing min_ratio
            datasets: List of dataset column names
            figsize: Figure size tuple
            title: Plot title
            max_edges: Maximum number of edges to show
            nickname_map: Dict mapping dataset names to display names
            
        Returns:
            matplotlib Figure
        """
        if not HAS_SEABORN:
            raise ImportError("seaborn is required for heatmaps")
        
        available = [d for d in datasets if d in ratio_data.columns]
        
        if not available:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        # Select top by mean
        plot_data = ratio_data[available].copy()
        plot_data['mean'] = plot_data.mean(axis=1)
        plot_data = plot_data.nlargest(max_edges, 'mean').drop(columns=['mean'])
        
        # Rename columns to nicknames if provided
        if nickname_map:
            plot_data = plot_data.rename(columns={d: nickname_map.get(d, d) for d in available})
        
        # Create heatmap
        fig_height = max(6, len(plot_data) * 0.3)
        figsize = figsize or (10, fig_height)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create annotation array that hides zeros
        annot_array = plot_data.values.astype(float)
        annot_array_str = np.where(annot_array == 0, '', np.char.mod('%.4f', annot_array))
        
        sns.heatmap(
            plot_data,
            annot=annot_array_str,
            fmt='',
            cmap='Greens',
            ax=ax,
            cbar_kws={'label': 'Min Ratio'},
            linewidths=0  # Hide grid lines
        )
        
        ax.set_title(title)
        ax.set_ylabel('Edge')
        ax.set_xlabel('Dataset')
        
        plt.tight_layout()
        return fig
    
    def plot_traversal_probability_heatmap(
        self,
        prob_data: pd.DataFrame,
        datasets: List[str],
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Traversal Probability Across Datasets",
        max_paths: int = 50,
        nickname_map: Dict[str, str] = None
    ) -> plt.Figure:
        """
        Plot heatmap of traversal probability values (path-level statistics).
        
        Args:
            prob_data: DataFrame with path index and dataset columns containing path_prob
            datasets: List of dataset column names
            figsize: Figure size tuple
            title: Plot title
            max_paths: Maximum number of paths to show
            nickname_map: Dict mapping dataset names to display names
            
        Returns:
            matplotlib Figure
        """
        if not HAS_SEABORN:
            raise ImportError("seaborn is required for heatmaps")
        
        available = [d for d in datasets if d in prob_data.columns]
        
        if not available:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        # Select top by mean
        plot_data = prob_data[available].copy()
        plot_data['mean'] = plot_data.mean(axis=1)
        plot_data = plot_data.nlargest(max_paths, 'mean').drop(columns=['mean'])
        
        # Rename columns to nicknames if provided
        if nickname_map:
            plot_data = plot_data.rename(columns={d: nickname_map.get(d, d) for d in available})
        
        # Create heatmap with log scale annotation for small values
        fig_height = max(6, len(plot_data) * 0.3)
        figsize = figsize or (10, fig_height)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use log scale for better visualization of small probabilities
        import matplotlib.colors as mcolors
        log_norm = mcolors.LogNorm(vmin=plot_data[plot_data > 0].min().min(), 
                                    vmax=plot_data.max().max()) if (plot_data > 0).any().any() else None
        
        # Create annotation array that hides zeros
        annot_array = plot_data.values.astype(float)
        annot_array_str = np.where(annot_array == 0, '', np.char.mod('%.2e', annot_array))
        
        sns.heatmap(
            plot_data,
            annot=annot_array_str,
            fmt='',
            cmap='Oranges',
            ax=ax,
            norm=log_norm,
            cbar_kws={'label': 'Traversal Probability'},
            linewidths=0  # Hide grid lines
        )
        
        ax.set_title(title)
        ax.set_ylabel('Path')
        ax.set_xlabel('Dataset')
        
        plt.tight_layout()
        return fig
    
    def plot_ratio_heatmap_all_thresholds(
        self,
        ratio_data_func,
        thresholds: List[int],
        datasets: List[str],
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Min-Ratio Across All Thresholds",
        max_edges: int = 30,
        nickname_map: Dict[str, str] = None
    ) -> plt.Figure:
        """
        Plot heatmap of min_ratio across all thresholds.
        
        Args:
            ratio_data_func: Function that returns ratio data for a given threshold
            thresholds: List of thresholds
            datasets: List of dataset names
            figsize: Figure size tuple
            title: Plot title
            max_edges: Maximum number of edges to show
            nickname_map: Dict mapping dataset names to display names
            
        Returns:
            matplotlib Figure
        """
        if not HAS_SEABORN:
            raise ImportError("seaborn is required for heatmaps")
        
        # Create default nickname map if not provided
        if nickname_map is None:
            nickname_map = {d: d for d in datasets}
        
        # Helper to safely extract scalar from potentially duplicate-indexed DataFrame
        def safe_get_value(df, row_key, col_key):
            """Extract scalar value, handling duplicate indices by summing."""
            try:
                val = df.loc[row_key, col_key]
                # If duplicate indices exist, .loc returns a Series - sum them
                if isinstance(val, pd.Series):
                    return float(val.sum())
                return float(val)
            except (KeyError, TypeError, ValueError):
                return 0.0
        
        # First pass: identify top edges from first threshold to limit iteration
        top_edges = []
        sample_threshold = thresholds[0] if thresholds else None
        
        if sample_threshold is not None:
            try:
                ratio_df = ratio_data_func(sample_threshold)
                if ratio_df is not None and not ratio_df.empty and len(datasets) > 0:
                    # Sum across all dataset columns for each edge
                    dataset_cols = [d for d in datasets if d in ratio_df.columns]
                    if dataset_cols:
                        ratio_df['_total'] = ratio_df[dataset_cols].sum(axis=1)
                        top_edges = ratio_df.nlargest(max_edges * 2, '_total').index.tolist()
                    else:
                        top_edges = list(ratio_df.index[:max_edges * 2])
            except Exception:
                pass
        
        top_edges_set = set(top_edges)
        
        # Collect data only for top edges across all thresholds
        all_data = {}
        
        for threshold in thresholds:
            try:
                ratio_df = ratio_data_func(threshold)
                if ratio_df is None or ratio_df.empty:
                    continue
                
                # Only process top edges
                edges_to_process = [e for e in ratio_df.index if e in top_edges_set] if top_edges_set else list(ratio_df.index[:max_edges * 2])
                
                for edge_key in edges_to_process:
                    for dataset in datasets:
                        if dataset in ratio_df.columns:
                            # Use nickname if available, otherwise truncate dataset name
                            nick = nickname_map.get(dataset, dataset)[:12]
                            col_name = f"{nick}_t{threshold}"
                            if edge_key not in all_data:
                                all_data[edge_key] = {}
                            all_data[edge_key][col_name] = safe_get_value(ratio_df, edge_key, dataset)
            except Exception:
                pass
        
        if not all_data:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No ratio data available", ha='center', va='center')
            return fig
        
        # Create DataFrame and ensure numeric dtype
        df = pd.DataFrame(all_data).T.fillna(0)
        # Convert all columns to float to avoid object dtype issues
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Select top edges by mean
        df['mean'] = df.mean(axis=1)
        df = df.nlargest(max_edges, 'mean').drop(columns=['mean'])
        
        # Create heatmap
        fig_height = max(8, len(df) * 0.35)
        fig_width = max(12, len(df.columns) * 0.6)
        figsize = figsize or (fig_width, fig_height)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create annotation array that hides zeros
        if len(df) <= 30:
            annot_array = df.values.astype(float)
            annot_array_str = np.where(annot_array == 0, '', np.char.mod('%.4f', annot_array))
        else:
            annot_array_str = False
        
        sns.heatmap(
            df,
            annot=annot_array_str,
            fmt='',
            cmap='Greens',
            ax=ax,
            cbar_kws={'label': 'Min Ratio'},
            linewidths=0  # Hide grid lines
        )
        
        ax.set_title(title)
        ax.set_ylabel('Edge')
        ax.set_xlabel('Dataset @ Threshold')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def plot_traversal_probability_heatmap_all_thresholds(
        self,
        prob_data_func,
        thresholds: List[int],
        datasets: List[str],
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Traversal Probability Across All Thresholds",
        max_paths: int = 30,
        nickname_map: Dict[str, str] = None
    ) -> plt.Figure:
        """
        Plot heatmap of traversal probability across all thresholds.
        
        Args:
            prob_data_func: Function that returns probability data for a given threshold
            thresholds: List of thresholds
            datasets: List of dataset names
            figsize: Figure size tuple
            title: Plot title
            max_paths: Maximum number of paths to show
            nickname_map: Dict mapping dataset names to display names
            
        Returns:
            matplotlib Figure
        """
        if not HAS_SEABORN:
            raise ImportError("seaborn is required for heatmaps")
        
        # Create default nickname map if not provided
        if nickname_map is None:
            nickname_map = {d: d for d in datasets}
        
        # Helper to safely extract scalar from potentially duplicate-indexed DataFrame
        def safe_get_value(df, row_key, col_key):
            """Extract scalar value, handling duplicate indices by summing."""
            try:
                val = df.loc[row_key, col_key]
                # If duplicate indices exist, .loc returns a Series - sum them
                if isinstance(val, pd.Series):
                    return float(val.sum())
                return float(val)
            except (KeyError, TypeError, ValueError):
                return 0.0
        
        # First pass: identify top paths from first threshold to limit iteration
        top_paths = []
        sample_threshold = thresholds[0] if thresholds else None
        
        if sample_threshold is not None:
            try:
                prob_df = prob_data_func(sample_threshold)
                if prob_df is not None and not prob_df.empty and len(datasets) > 0:
                    # Sum across all dataset columns for each path
                    dataset_cols = [d for d in datasets if d in prob_df.columns]
                    if dataset_cols:
                        prob_df['_total'] = prob_df[dataset_cols].sum(axis=1)
                        top_paths = prob_df.nlargest(max_paths * 2, '_total').index.tolist()
                    else:
                        top_paths = list(prob_df.index[:max_paths * 2])
            except Exception:
                pass
        
        top_paths_set = set(top_paths)
        
        # Collect data only for top paths across all thresholds
        all_data = {}
        
        for threshold in thresholds:
            try:
                prob_df = prob_data_func(threshold)
                if prob_df is None or prob_df.empty:
                    continue
                
                # Only process top paths
                paths_to_process = [p for p in prob_df.index if p in top_paths_set] if top_paths_set else list(prob_df.index[:max_paths * 2])
                
                for path_key in paths_to_process:
                    for dataset in datasets:
                        if dataset in prob_df.columns:
                            # Use nickname if available, otherwise truncate dataset name
                            nick = nickname_map.get(dataset, dataset)[:12]
                            col_name = f"{nick}_t{threshold}"
                            if path_key not in all_data:
                                all_data[path_key] = {}
                            all_data[path_key][col_name] = safe_get_value(prob_df, path_key, dataset)
            except Exception:
                pass
        
        if not all_data:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No probability data available", ha='center', va='center')
            return fig
        
        # Create DataFrame and ensure numeric dtype
        df = pd.DataFrame(all_data).T.fillna(0)
        # Convert all columns to float to avoid object dtype issues
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Select top paths by mean
        df['mean'] = df.mean(axis=1)
        df = df.nlargest(max_paths, 'mean').drop(columns=['mean'])
        
        # Create heatmap
        fig_height = max(8, len(df) * 0.4)
        fig_width = max(14, len(df.columns) * 0.8)  # Wider cells
        figsize = figsize or (fig_width, fig_height)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use log scale for better visualization of small probabilities
        import matplotlib.colors as mcolors
        log_norm = mcolors.LogNorm(vmin=df[df > 0].min().min(), 
                                    vmax=df.max().max()) if (df > 0).any().any() else None
        
        # Create annotation array that hides zeros
        if len(df) <= 30:
            annot_array = df.values.astype(float)
            annot_array_str = np.where(annot_array == 0, '', np.char.mod('%.2e', annot_array))
        else:
            annot_array_str = False
        
        sns.heatmap(
            df,
            annot=annot_array_str,
            fmt='',
            cmap='Oranges',
            ax=ax,
            norm=log_norm,
            cbar_kws={'label': 'Traversal Probability'},
            linewidths=0,  # Hide grid lines
            annot_kws={'fontsize': 7}  # Smaller font for annotations
        )
        
        ax.set_title(title)
        ax.set_ylabel('Path')
        ax.set_xlabel('Dataset @ Threshold')
        plt.xticks(rotation=45, ha='right', fontsize=7)
        
        plt.tight_layout()
        return fig
    
    def save_all_plots(
        self,
        results: Dict[str, Dict[int, pd.DataFrame]],
        aligned_data: pd.DataFrame,
        similarities: pd.DataFrame,
        output_dir: str,
        thresholds: List[int],
        align_func=None,
        similarity_func=None,
        current_threshold: int = None,
        path_data_func=None,
        ratio_data_func=None,
        prob_data_func=None,
        output_base_path: str = None,
        nickname_map: Dict[str, str] = None,
        path_presence_matrix: pd.DataFrame = None,
        point_metadata_func=None,
        point_label_func=None,
        point_stem_func=None,
        threshold_mode: str = 'standard',
        silent: bool = False
    ):
        """
        Generate and save all standard plots.
        
        Args:
            results: Raw analysis results
            aligned_data: Aligned edge data
            similarities: Pairwise similarities
            output_dir: Directory to save plots
            thresholds: List of thresholds
            align_func: Function to get aligned data at any threshold (for multi-threshold plots)
            similarity_func: Function to get cached similarities at a threshold (avoids recalculation)
            current_threshold: Current threshold used for aligned_data (for labeling)
            path_data_func: Function to get path min_weight data at a given threshold
            ratio_data_func: Function to get ratio data at a given threshold
            prob_data_func: Function to get traversal probability data at a given threshold
            output_base_path: Base path for reading original dataset data
            nickname_map: Dict mapping dataset names to short nicknames for display
            path_presence_matrix: Optional DataFrame with path presence across datasets and thresholds
            point_metadata_func: Optional function returning requested/applied
                provenance for one comparison point
            point_label_func: Optional function returning a display label for
                one comparison point
            point_stem_func: Optional function returning a filesystem stem for
                one comparison point (for example ``query_combo_001``)
            threshold_mode: ``standard`` or ``combinations``; combination
                points are query rows, not scalar thresholds
            silent: If True, suppress per-file messages and show summary instead
        """
        # Track saved files for summary
        self._silent_mode = silent
        saved_count = 0
        
        os.makedirs(output_dir, exist_ok=True)
        datasets = list(results.keys())
        
        # Create nickname map if not provided
        if nickname_map is None:
            nickname_map = {d: d for d in datasets}
        
        # Create visualization_data subfolder for data files
        vis_data_dir = os.path.join(output_dir, "visualization_data")
        os.makedirs(vis_data_dir, exist_ok=True)
        
        # Determine current threshold for labels
        if current_threshold is None:
            current_threshold = thresholds[len(thresholds) // 2] if thresholds else 1

        def point_metadata(point):
            if point_metadata_func is None:
                return {}
            try:
                value = point_metadata_func(point) or {}
                return value if isinstance(value, dict) else {}
            except Exception:
                return {}

        def point_label(point):
            if point_label_func is None:
                return str(point)
            try:
                return str(point_label_func(point))
            except Exception:
                return str(point)

        def point_stem(point):
            if point_stem_func is None:
                return str(point)
            try:
                return str(point_stem_func(point))
            except Exception:
                return str(point)

        def point_threshold_value(point):
            """Keep scalar ``threshold`` columns numeric in Standard mode.

            A Custom combination row is a query identity, not a scalar
            threshold. Its identity is carried by ``query_id`` below, so
            query-keyed aggregate CSVs must not overload ``threshold`` with a
            query string.
            """
            return point if threshold_mode != 'combinations' else None

        def point_title(point):
            """Axis/title fragment for one comparison point.

            Standard points keep the historical scalar wording
            ("Threshold=10"). Custom combination points display their query
            label, never a numeric-threshold phrase.
            """
            if threshold_mode == 'combinations':
                return point_label(point)
            return f"Threshold={point}"

        def add_point_fields(row, point):
            # Keep the established Standard export schema unchanged.  Query
            # identity and threshold scope are required only when the caller
            # is exporting row-wise custom combinations; adding those columns
            # to scalar exports would make downstream Standard consumers
            # unnecessarily version-sensitive.
            if threshold_mode != 'combinations':
                return row
            meta = point_metadata(point)
            fields = {
                'threshold_scope': meta.get('threshold_scope', threshold_mode),
            }
            fields.update({
                'query_id': meta.get('query_id', point),
                'query_label': meta.get('query_label', point_label(point)),
            })
            for dataset, value in (meta.get('requested_thresholds') or {}).items():
                safe = str(dataset).replace(':', '_').replace('.', '_').replace('-', '_')
                fields[f'requested_{safe}'] = value
            for dataset, value in (meta.get('applied_thresholds') or {}).items():
                safe = str(dataset).replace(':', '_').replace('.', '_').replace('-', '_')
                fields[f'applied_{safe}'] = value
            if isinstance(row, pd.DataFrame):
                for key, value in fields.items():
                    row[key] = value
                return row
            for key, value in fields.items():
                row.setdefault(key, value)
            return row
        
        # Path counts
        fig = self.plot_path_counts(
            results, thresholds,
            nickname_map=nickname_map,
            title=("Path Counts by Dataset and Query"
                   if threshold_mode == 'combinations'
                   else "Path Counts by Dataset and Threshold"),
            x_label=("Query" if threshold_mode == 'combinations'
                     else "Weight Threshold"),
        )
        self.save_figure(fig, os.path.join(output_dir, "path_counts.png"))
        plt.close(fig)
        
        # Save path counts data
        path_counts_data = []
        for dataset in datasets:
            for threshold in thresholds:
                df = results.get(dataset, {}).get(threshold, pd.DataFrame())
                count = len(df) if not df.empty else 0
                path_counts_data.append(add_point_fields({
                    'dataset': dataset,
                    'threshold': point_threshold_value(threshold),
                    'count': count,
                }, threshold))
        pd.DataFrame(path_counts_data).to_csv(os.path.join(vis_data_dir, "path_counts.csv"), index=False)
        
        # Note: edge_heatmap.png removed (redundant with path/threshold comparison views)
        
        # Note: similarity_matrix.csv is saved with all thresholds in the per-threshold section below
        
        # Similarity matrices per threshold (combined subplot figure)
        # Use cached similarities if available to avoid recalculation
        if align_func and len(thresholds) > 1:
            try:
                fig = self.plot_similarity_matrix_per_threshold(
                    results, thresholds, align_func,
                    similarity_func=similarity_func,
                    path_data_func=path_data_func,
                    title=("Dataset Similarity for Each Query Row"
                           if threshold_mode == 'combinations'
                           else "Dataset Similarity at Each Threshold Level")
                )
                self.save_figure(fig, os.path.join(output_dir, "similarity_per_threshold.png"))
                plt.close(fig)
                
                # Save per-threshold similarity data (use cached if available)
                all_sim_data = []
                for threshold in thresholds:
                    try:
                        if similarity_func:
                            sim_t = similarity_func(threshold)
                        else:
                            from .metrics import ComparisonMetrics
                            metrics = ComparisonMetrics()
                            aligned_t = align_func(threshold)
                            if not aligned_t.empty:
                                # Get path data for this threshold
                                path_data_t = None
                                if path_data_func:
                                    try:
                                        path_data_t = path_data_func(threshold)
                                    except Exception:
                                        pass
                                sim_t = metrics.calculate_all_pairwise_similarities(
                                    aligned_t, datasets, threshold=1, path_data=path_data_t
                                )
                        if sim_t is not None and not sim_t.empty:
                            sim_t = sim_t.copy()
                            sim_t['threshold'] = point_threshold_value(threshold)
                            add_point_fields(sim_t, threshold)
                            all_sim_data.append(sim_t)
                            if threshold_mode == 'combinations':
                                sim_t.to_csv(
                                    os.path.join(
                                        vis_data_dir,
                                        f"similarity_{point_stem(threshold)}.csv"),
                                    index=False)
                    except Exception:
                        pass
                if all_sim_data:
                    # Save both individual similarity_matrix.csv (all thresholds combined) 
                    # and similarity_per_threshold.csv for backward compatibility
                    combined_sim = pd.concat(all_sim_data, ignore_index=True)
                    combined_sim.to_csv(os.path.join(vis_data_dir, "similarity_per_threshold.csv"), index=False)
                    combined_sim.to_csv(os.path.join(vis_data_dir, "similarity_matrix.csv"), index=False)
            except Exception as e:
                self._vprint(f"Warning: Could not create per-threshold similarity plot: {e}")
        
        # Edge overlap data export (visualization removed - redundant with
        # threshold_comparison).  Combination mode must retain one block per
        # query; the old selected-middle-point export was misleading.
        overlap_data = []
        overlap_points = thresholds if align_func else [current_threshold]
        for point in overlap_points:
            try:
                aligned_point = align_func(point) if align_func else aligned_data
            except Exception as exc:
                self._vprint(
                    f"Warning: Could not align overlap point {point}: {exc}"
                )
                continue
            if aligned_point is None or aligned_point.empty:
                continue
            available = [d for d in datasets if d in aligned_point.columns]
            for d in available:
                present = set(aligned_point[aligned_point[d] > 0].index)
                overlap_data.append(add_point_fields({
                    'threshold': point_threshold_value(point),
                    'dataset': d,
                    'edge_count': len(present),
                }, point))
            if available:
                common = set(aligned_point[
                    (aligned_point[available] > 0).all(axis=1)].index)
                overlap_data.append(add_point_fields({
                    'threshold': point_threshold_value(point),
                    'dataset': 'common_all',
                    'edge_count': len(common),
                }, point))
        if overlap_data:
            pd.DataFrame(overlap_data).to_csv(
                os.path.join(vis_data_dir, "edge_overlap.csv"), index=False)
        
        # Dataset overlap matrices per threshold (edge and path overlap)
        if align_func and path_data_func:
            try:
                overlap_matrices_data = []
                for threshold in thresholds:
                    aligned_t = align_func(threshold)
                    if aligned_t.empty:
                        continue
                    available_ds = [d for d in datasets if d in aligned_t.columns]
                    n = len(available_ds)
                    
                    # Compute edge overlap matrix (asymmetric)
                    edge_overlap_matrix = [[0 for _ in range(n)] for _ in range(n)]
                    for i1, d1 in enumerate(available_ds):
                        edges_in_d1 = set(aligned_t.index[aligned_t[d1] > 0])
                        edge_overlap_matrix[i1][i1] = len(edges_in_d1)
                        for i2, d2 in enumerate(available_ds):
                            if i1 != i2:
                                edges_in_d2 = set(aligned_t.index[aligned_t[d2] > 0])
                                edge_overlap_matrix[i1][i2] = len(edges_in_d1 & edges_in_d2)
                    
                    # Compute path overlap matrix (asymmetric)
                    path_overlap_matrix = [[0 for _ in range(n)] for _ in range(n)]
                    try:
                        path_data = path_data_func(threshold)
                        if path_data is not None and not path_data.empty:
                            for i1, d1 in enumerate(available_ds):
                                paths_in_d1 = set(path_data.index[path_data[d1] > 0]) if d1 in path_data.columns else set()
                                path_overlap_matrix[i1][i1] = len(paths_in_d1)
                                for i2, d2 in enumerate(available_ds):
                                    if i1 != i2:
                                        paths_in_d2 = set(path_data.index[path_data[d2] > 0]) if d2 in path_data.columns else set()
                                        path_overlap_matrix[i1][i2] = len(paths_in_d1 & paths_in_d2)
                    except Exception:
                        pass
                    
                    # Flatten to rows for CSV
                    for i1, d1 in enumerate(available_ds):
                        for i2, d2 in enumerate(available_ds):
                            diag_edge = edge_overlap_matrix[i1][i1]
                            diag_path = path_overlap_matrix[i1][i1]
                            overlap_matrices_data.append(add_point_fields({
                                'threshold': point_threshold_value(threshold),
                                'threshold_label': point_label(threshold),
                                'source_dataset': d1,
                                'target_dataset': d2,
                                'edge_count': edge_overlap_matrix[i1][i2],
                                'edge_proportion': edge_overlap_matrix[i1][i2] / diag_edge if diag_edge > 0 else 0,
                                'path_count': path_overlap_matrix[i1][i2],
                                'path_proportion': path_overlap_matrix[i1][i2] / diag_path if diag_path > 0 else 0
                            }, threshold))
                
                if overlap_matrices_data:
                    pd.DataFrame(overlap_matrices_data).to_csv(
                        os.path.join(vis_data_dir, "overlap_matrices_per_threshold.csv"), index=False
                    )
            except Exception as e:
                self._vprint(f"Warning: Could not save overlap matrices data: {e}")
        
        # Pairwise scatter plots
        if len(datasets) == 2:
            fig = self.plot_weight_scatter(aligned_data, datasets[0], datasets[1])
            self.save_figure(fig, os.path.join(output_dir, "weight_scatter.png"))
            plt.close(fig)
            
            # Save weight scatter data
            if not aligned_data.empty:
                scatter_data = aligned_data[[datasets[0], datasets[1]]].copy()
                scatter_data.to_csv(os.path.join(vis_data_dir, "weight_scatter.csv"))
        
        # Merged threshold comparison subplots
        if len(thresholds) > 1:
            fig = self.plot_threshold_comparison_subplots(
                results, thresholds, threshold_mode=threshold_mode)
            self.save_figure(fig, os.path.join(output_dir, "threshold_comparison.png"))
            plt.close(fig)
            
            # Save threshold comparison data
            comparison_data = []
            for dataset in datasets:
                for threshold in thresholds:
                    df = results.get(dataset, {}).get(threshold, pd.DataFrame())
                    if not df.empty:
                        weights = df['weight'].values if 'weight' in df.columns else [1] * len(df)
                        comparison_data.append({
                            'dataset': dataset,
                            'threshold': point_threshold_value(threshold),
                            'edge_count': len(df),
                            'mean_weight': np.mean(weights) if len(weights) > 0 else 0,
                            'max_weight': max(weights) if len(weights) > 0 else 0
                        })
                        add_point_fields(comparison_data[-1], threshold)
            pd.DataFrame(comparison_data).to_csv(os.path.join(vis_data_dir, "threshold_comparison.csv"), index=False)
        
        # Conservation across thresholds plot.  Standard only: the chart is a
        # threshold-trend figure, and Custom query rows are not a monotone
        # threshold schedule.  Per-query conservation rates are still exported
        # via key_findings_per_threshold.csv below.
        if len(thresholds) > 1 and threshold_mode != 'combinations':
            try:
                fig, conservation_data = self.plot_conservation_across_thresholds(
                    results, thresholds, align_func,
                    nickname_map=nickname_map,
                    path_presence_matrix=path_presence_matrix
                )
                if fig:
                    self.save_figure(fig, os.path.join(output_dir, "conservation_across_thresholds.png"))
                    plt.close(fig)
                
                # Save conservation data
                if conservation_data is not None and not conservation_data.empty:
                    conservation_data.to_csv(os.path.join(vis_data_dir, "conservation_across_thresholds.csv"), index=False)
                
                # Also generate Plotly version if possible
                try:
                    plotly_json = self.plot_conservation_across_thresholds_plotly(
                        results, thresholds, align_func,
                        nickname_map=nickname_map,
                        path_presence_matrix=path_presence_matrix
                    )
                    with open(
                            os.path.join(
                                vis_data_dir,
                                "conservation_across_thresholds.json"),
                            'w', encoding='utf-8') as f:
                        f.write(plotly_json)
                except Exception as e:
                    self._vprint(f"Warning: Could not create Plotly conservation plot: {e}")
                    
            except Exception as e:
                self._vprint(f"Warning: Could not create conservation plot: {e}")
        
        # Save key_findings_per_threshold data
        if align_func:
            from .metrics import ComparisonMetrics
            metrics = ComparisonMetrics()
            key_findings_data = []
            for threshold in thresholds:
                try:
                    aligned_t = align_func(threshold)
                    if aligned_t.empty:
                        continue
                    available_ds = [d for d in datasets if d in aligned_t.columns]
                    
                    # Calculate metrics for this threshold
                    row_data = {'threshold': point_threshold_value(threshold)}
                    add_point_fields(row_data, threshold)
                    
                    # Common edges
                    if available_ds:
                        mask_all = (aligned_t[available_ds] > 0).all(axis=1)
                        row_data['common_edges'] = int(mask_all.sum())
                    else:
                        row_data['common_edges'] = 0
                    
                    # Conservation rate
                    total_edges = len(aligned_t)
                    row_data['conservation_rate'] = row_data['common_edges'] / total_edges if total_edges > 0 else 0
                    
                    # Edge counts per dataset
                    for ds in available_ds:
                        safe_ds = canonical_dataset_name(ds).replace(':', '_').replace('.', '_').replace('-', '_')
                        row_data[f'edges_{safe_ds}'] = int((aligned_t[ds] > 0).sum())
                    
                    # Unique edges per dataset
                    for ds in available_ds:
                        safe_ds = canonical_dataset_name(ds).replace(':', '_').replace('.', '_').replace('-', '_')
                        other_ds = [d for d in available_ds if d != ds]
                        if other_ds:
                            ds_present = aligned_t[ds] > 0
                            others_absent = (aligned_t[other_ds] > 0).sum(axis=1) == 0
                            row_data[f'unique_{safe_ds}'] = int((ds_present & others_absent).sum())
                        else:
                            row_data[f'unique_{safe_ds}'] = int((aligned_t[ds] > 0).sum())
                    
                    # Similarity metrics.  The aligned frame is already
                    # thresholded; combination rows carry no scalar
                    # threshold, so apply the presence cut directly
                    # (matching the per-query similarity exports).
                    sims = metrics.calculate_all_pairwise_similarities(aligned_t, datasets, 1)
                    if not sims.empty:
                        row_data['avg_jaccard'] = float(sims['jaccard_similarity'].mean())
                        if 'svd_similarity' in sims.columns:
                            row_data['avg_svd'] = float(sims['svd_similarity'].mean())
                        elif 'pearson_correlation' in sims.columns:
                            corr = sims['pearson_correlation'].mean()
                            row_data['avg_svd'] = float(corr) if not np.isnan(corr) else 0.0
                    
                    key_findings_data.append(row_data)
                except Exception as e:
                    self._vprint(f"Warning: Could not compute key findings for threshold {threshold}: {e}")
            
            if key_findings_data:
                pd.DataFrame(key_findings_data).to_csv(
                    os.path.join(vis_data_dir, "key_findings_per_threshold.csv"), index=False
                )
        
        # =====================================================================
        # Path-level and Edge-level Heatmaps per Threshold
        # =====================================================================
        
        # Generate path heatmaps per threshold (using min_weight as path weight)
        if path_data_func and align_func:
            try:
                for threshold in thresholds:
                    path_df = path_data_func(threshold)
                    if path_df is not None and not path_df.empty:
                        fig = self.plot_path_heatmap(
                            path_df, datasets,
                            title=f"Path Min-Weight at {point_title(threshold)}",
                            nickname_map=nickname_map
                        )
                        self.save_figure(fig, os.path.join(
                            output_dir, f"path_heatmap_{point_stem(threshold)}.png"))
                        plt.close(fig)
                        
                        # Save data
                        path_export = path_df.copy()
                        if threshold_mode == 'combinations':
                            add_point_fields(path_export, threshold)
                        path_export.to_csv(os.path.join(
                            vis_data_dir,
                            f"path_heatmap_{point_stem(threshold)}.csv"))
                
                # All thresholds path heatmap.  Skipped for Custom
                # combinations: the per-query heatmaps above are the
                # comparison artifacts, and a combined panel would present
                # the raw-run schedule as a threshold axis.
                if len(thresholds) > 1 and threshold_mode != 'combinations':
                    fig = self.plot_path_heatmap_all_thresholds(
                        path_data_func, thresholds, datasets,
                        title="Path Min-Weights Across All Thresholds",
                        nickname_map=nickname_map
                    )
                    self.save_figure(fig, os.path.join(output_dir, "path_heatmap_all_thresholds.png"))
                    plt.close(fig)
            except Exception as e:
                self._vprint(f"Warning: Could not create path heatmaps: {e}")
        
        # Generate edge heatmaps per threshold
        if align_func:
            try:
                for threshold in thresholds:
                    aligned_t = align_func(threshold)
                    if not aligned_t.empty:
                        fig = self.plot_edge_weight_heatmap(
                            aligned_t, datasets,
                            title=f"Edge Weights at {point_title(threshold)}",
                            nickname_map=nickname_map
                        )
                        self.save_figure(fig, os.path.join(
                            output_dir, f"edge_heatmap_{point_stem(threshold)}.png"))
                        plt.close(fig)
                        
                        # Save edge heatmap data
                        edge_export = aligned_t.copy()
                        if threshold_mode == 'combinations':
                            add_point_fields(edge_export, threshold)
                        edge_export.to_csv(os.path.join(
                            vis_data_dir,
                            f"edge_heatmap_{point_stem(threshold)}.csv"))
                
                # All thresholds edge heatmap (Standard only — see the path
                # heatmap note above).
                if len(thresholds) > 1 and threshold_mode != 'combinations':
                    fig = self.plot_edge_weight_heatmap_all_thresholds(
                        results, thresholds, align_func,
                        title="Edge Weights Across All Thresholds",
                        nickname_map=nickname_map
                    )
                    self.save_figure(fig, os.path.join(output_dir, "edge_heatmap_all_thresholds.png"))
                    plt.close(fig)
            except Exception as e:
                self._vprint(f"Warning: Could not create edge heatmaps: {e}")
        
        # Generate ratio heatmaps from original dataset data (in by_ratio subfolder)
        if ratio_data_func:
            try:
                ratio_dir = os.path.join(output_dir, "by_ratio")
                os.makedirs(ratio_dir, exist_ok=True)
                ratio_data_dir = os.path.join(ratio_dir, "data")
                os.makedirs(ratio_data_dir, exist_ok=True)
                
                for threshold in thresholds:
                    ratio_df = ratio_data_func(threshold)
                    if ratio_df is not None and not ratio_df.empty:
                        fig = self.plot_ratio_heatmap(
                            ratio_df, datasets,
                            title=f"Path Min-Ratio at Threshold={threshold}",
                            nickname_map=nickname_map
                        )
                        self.save_figure(fig, os.path.join(ratio_dir, f"ratio_heatmap_{threshold}.png"))
                        plt.close(fig)
                        
                        # Save data
                        ratio_df.to_csv(os.path.join(ratio_data_dir, f"ratio_heatmap_{threshold}.csv"))
                
                # All thresholds ratio heatmap
                if len(thresholds) > 1:
                    fig = self.plot_ratio_heatmap_all_thresholds(
                        ratio_data_func, thresholds, datasets,
                        title="Path Min-Ratio Across All Thresholds",
                        nickname_map=nickname_map
                    )
                    self.save_figure(fig, os.path.join(ratio_dir, "ratio_heatmap_all_thresholds.png"))
                    plt.close(fig)
            except Exception as e:
                self._vprint(f"Warning: Could not create ratio heatmaps: {e}")
        
        # Generate traversal probability heatmaps from original dataset data (in by_probability subfolder)
        if prob_data_func:
            try:
                prob_dir = os.path.join(output_dir, "by_probability")
                os.makedirs(prob_dir, exist_ok=True)
                prob_data_dir = os.path.join(prob_dir, "data")
                os.makedirs(prob_data_dir, exist_ok=True)
                
                for threshold in thresholds:
                    prob_df = prob_data_func(threshold)
                    if prob_df is not None and not prob_df.empty:
                        fig = self.plot_traversal_probability_heatmap(
                            prob_df, datasets,
                            title=f"Traversal Probability at Threshold={threshold}",
                            nickname_map=nickname_map
                        )
                        self.save_figure(fig, os.path.join(prob_dir, f"traversal_prob_heatmap_{threshold}.png"))
                        plt.close(fig)
                        
                        # Save data
                        prob_df.to_csv(os.path.join(prob_data_dir, f"traversal_prob_heatmap_{threshold}.csv"))
                
                # All thresholds traversal probability heatmap
                if len(thresholds) > 1:
                    fig = self.plot_traversal_probability_heatmap_all_thresholds(
                        prob_data_func, thresholds, datasets,
                        title="Traversal Probability Across All Thresholds",
                        nickname_map=nickname_map
                    )
                    self.save_figure(fig, os.path.join(prob_dir, "traversal_prob_heatmap_all_thresholds.png"))
                    plt.close(fig)
            except Exception as e:
                self._vprint(f"Warning: Could not create traversal probability heatmaps: {e}")
        
        # Trend plots below are threshold-sensitivity figures.  Custom query
        # rows have a display order, not a numeric schedule, so emitting them
        # for combinations would dress query order up as a threshold axis
        # (plan non-goal).  Pair-level metrics per query are already exported
        # through the per-query similarity CSVs.
        # Generate Jaccard similarity trend plot
        if align_func and len(thresholds) > 1 and threshold_mode != 'combinations':
            try:
                fig, jaccard_df = self.plot_jaccard_similarity_trend(
                    align_func, thresholds, datasets,
                    title="Jaccard Similarity Across Thresholds",
                    nickname_map=nickname_map
                )
                self.save_figure(fig, os.path.join(output_dir, "jaccard_similarity_trend.png"))
                plt.close(fig)
                
                # Save data
                jaccard_df.to_csv(os.path.join(vis_data_dir, "jaccard_similarity_trend.csv"), index=False)
            except Exception as e:
                self._vprint(f"Warning: Could not create Jaccard similarity plot: {e}")
        
        # Generate Edge top-20 overlap trend plot (v2.2; replaces the
        # retired Edge Rank trend)
        if align_func and len(thresholds) > 1 and threshold_mode != 'combinations':
            try:
                fig, topk_df = self.plot_topk_overlap_trend(
                    align_func, thresholds, datasets,
                    title="Edge Top-20 Overlap Across Thresholds",
                    nickname_map=nickname_map
                )
                self.save_figure(fig, os.path.join(output_dir, "top20_overlap_trend.png"))
                plt.close(fig)

                # Save data
                topk_df.to_csv(os.path.join(vis_data_dir, "top20_overlap_trend.csv"), index=False)
            except Exception as e:
                self._vprint(f"Warning: Could not create Edge Top-20 overlap plot: {e}")

        # Generate NetSimile-lite trend plot (v2.2 graph-level; replaces
        # the retired Path Rank trend)
        if align_func and len(thresholds) > 1 and threshold_mode != 'combinations':
            try:
                fig, netsimile_df = self.plot_netsimile_trend(
                    align_func, thresholds, datasets,
                    title="NetSimile-lite Similarity Across Thresholds",
                    nickname_map=nickname_map
                )
                self.save_figure(fig, os.path.join(output_dir, "netsimile_trend.png"))
                plt.close(fig)

                # Save data
                netsimile_df.to_csv(os.path.join(vis_data_dir, "netsimile_trend.csv"), index=False)
            except Exception as e:
                self._vprint(f"Warning: Could not create NetSimile trend plot: {e}")
        
        # Generate Cosine Similarity trend plot
        if align_func and len(thresholds) > 1 and threshold_mode != 'combinations':
            try:
                fig, cosine_df = self.plot_cosine_similarity_trend(
                    align_func, thresholds, datasets,
                    title="Cosine Similarity Across Thresholds",
                    nickname_map=nickname_map
                )
                self.save_figure(fig, os.path.join(output_dir, "cosine_similarity_trend.png"))
                plt.close(fig)
                
                # Save data
                cosine_df.to_csv(os.path.join(vis_data_dir, "cosine_similarity_trend.csv"), index=False)
            except Exception as e:
                self._vprint(f"Warning: Could not create Cosine Similarity plot: {e}")
        
        # Note: vis_summary.pdf generation removed - use HTML report instead
        
        # Reset silent mode and show summary
        self._silent_mode = False
        
        # Count saved files for summary
        png_count = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
        csv_count = len([f for f in os.listdir(vis_data_dir) if f.endswith('.csv')])
        
        self._vprint(f"Saved {png_count} plots to: {output_dir}")
        self._vprint(f"Saved {csv_count} data files to: {vis_data_dir}")

    def plot_conservation_across_thresholds(
        self,
        results: Dict[str, Dict[int, pd.DataFrame]],
        thresholds: List[int],
        align_func=None,
        title: str = "Conservation of Edges and Paths Across Thresholds",
        nickname_map: Dict[str, str] = None,
        path_presence_matrix: pd.DataFrame = None
    ) -> Tuple[plt.Figure, pd.DataFrame]:
        """
        Plot conservation of edges and paths across thresholds.
        
        Shows:
        - Edge Counts (Linear)
        - Path Counts (Linear)
        - Edge Retention Rate (Count_i / Count_{i-1})
        - Path Retention Rate (Count_i / Count_{i-1})
        
        Args:
            results: Raw analysis results
            thresholds: List of thresholds
            align_func: Function to get aligned data (for common edges)
            title: Plot title
            nickname_map: Dict mapping dataset names to display names
            path_presence_matrix: Optional DataFrame with path presence info
            
        Returns:
            Tuple of (matplotlib Figure, DataFrame with plot data)
        """
        if not HAS_MATPLOTLIB:
            return None, None
            
        datasets = list(results.keys())
        if nickname_map is None:
            nickname_map = {d: d for d in datasets}
            
        # Sort thresholds to ensure correct line plotting and rate calculation
        sorted_thresholds = sorted(thresholds)
        
        # Prepare data for export
        plot_data = []
            
        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        ax_edge_count = axes[0, 0]
        ax_path_count = axes[0, 1]
        ax_edge_rate = axes[1, 0]
        ax_path_rate = axes[1, 1]
        
        # Colors and markers
        colors = plt.cm.tab10.colors
        markers = ['o', 's', '^', 'D', 'v', '<', '>']
        
        # Track totals for Overall Rate calculation
        total_edge_counts = [0] * len(sorted_thresholds)
        total_path_counts = [0] * len(sorted_thresholds)

        # Plot edges and paths for each dataset
        for i, dataset in enumerate(datasets):
            nick = nickname_map.get(dataset, dataset)
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            edge_counts = []
            path_counts = []
            
            for idx, t in enumerate(sorted_thresholds):
                # Get data for this threshold
                df = results.get(dataset, {}).get(t, pd.DataFrame())
                
                # Edge count (rows in dataframe)
                e_count = len(df) if not df.empty else 0
                edge_counts.append(e_count)
                total_edge_counts[idx] += e_count
                
                # Path count
                p_count = 0
                if path_presence_matrix is not None and not path_presence_matrix.empty:
                    # Use path presence matrix if available
                    # Try original name first, then sanitized name (replace : and . with _)
                    col_orig = f"{dataset}_t{t}"
                    dataset_safe = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
                    col_safe = f"{dataset_safe}_t{t}"
                    
                    col_name = None
                    if col_orig in path_presence_matrix.columns:
                        col_name = col_orig
                    elif col_safe in path_presence_matrix.columns:
                        col_name = col_safe
                    
                    if col_name is not None:
                        # Check for 'True' string or boolean True
                        vals = path_presence_matrix[col_name]
                        if vals.dtype == object:
                            # Handle both string 'True' and boolean True
                            p_count = ((vals == 'True') | (vals == True)).sum()
                        elif vals.dtype == bool:
                            p_count = vals.sum()
                        else:
                            p_count = (vals > 0).sum()
                    else:
                        # Fallback if column missing but matrix exists
                        if not df.empty and 'has_valid_path' in df.columns:
                            p_count = df['has_valid_path'].sum()
                        else:
                            p_count = len(df) if not df.empty else 0
                else:
                    # Fallback to edge-based estimation
                    if not df.empty and 'has_valid_path' in df.columns:
                        p_count = df['has_valid_path'].sum()
                    else:
                        p_count = len(df) if not df.empty else 0
                
                path_counts.append(p_count)
                total_path_counts[idx] += p_count
            
            # --- 1. Counts (Linear Scale) ---
            ax_edge_count.plot(sorted_thresholds, edge_counts, marker=marker, linestyle='-', 
                             color=color, label=nick, linewidth=2, alpha=0.8)
            
            ax_path_count.plot(sorted_thresholds, path_counts, marker=marker, linestyle='--',
                             color=color, label=nick, linewidth=2, alpha=0.8)
            
            # --- 2. Delta Rates (Drop Rate) ---
            # Delta[i] = (Count[i-1] - Count[i]) / Count[i-1]
            # Shows the fraction of edges/paths lost at each threshold step
            # For the first threshold, we can't calculate a rate from previous, so we skip or set to NaN
            edge_rates = [np.nan] 
            path_rates = [np.nan]
            
            for j in range(1, len(sorted_thresholds)):
                prev_e = edge_counts[j-1]
                curr_e = edge_counts[j]
                rate_e = ((prev_e - curr_e) / prev_e) if prev_e > 0 else 0
                edge_rates.append(rate_e)
                
                prev_p = path_counts[j-1]
                curr_p = path_counts[j]
                rate_p = ((prev_p - curr_p) / prev_p) if prev_p > 0 else 0
                path_rates.append(rate_p)
                
            ax_edge_rate.plot(sorted_thresholds, edge_rates, marker=marker, linestyle='-', 
                            color=color, label=nick, alpha=0.8)
            ax_path_rate.plot(sorted_thresholds, path_rates, marker=marker, linestyle='--', 
                            color=color, label=nick, alpha=0.8)
            
            # Collect data for export
            for j, t in enumerate(sorted_thresholds):
                plot_data.append({
                    'dataset': dataset,
                    'threshold': t,
                    'edge_count': edge_counts[j],
                    'path_count': path_counts[j],
                    'edge_retention_rate': edge_rates[j],
                    'path_retention_rate': path_rates[j]
                })

        # --- Overall Delta Rates ---
        overall_edge_rates = [np.nan]
        overall_path_rates = [np.nan]
        for j in range(1, len(sorted_thresholds)):
            prev_e = total_edge_counts[j-1]
            curr_e = total_edge_counts[j]
            overall_edge_rates.append(((prev_e - curr_e) / prev_e) if prev_e > 0 else 0)
            
            prev_p = total_path_counts[j-1]
            curr_p = total_path_counts[j]
            overall_path_rates.append(((prev_p - curr_p) / prev_p) if prev_p > 0 else 0)
            
        ax_edge_rate.plot(sorted_thresholds, overall_edge_rates, marker='*', linestyle='-', 
                        color='black', label="Overall", linewidth=2, markersize=10, alpha=0.5)
        ax_path_rate.plot(sorted_thresholds, overall_path_rates, marker='*', linestyle='--', 
                        color='black', label="Overall", linewidth=2, markersize=10, alpha=0.5)

        # --- Common Counts ---
        common_edge_counts = []
        common_path_counts = []
        
        for t in sorted_thresholds:
            # Common Edges
            c_edge = 0
            if align_func:
                try:
                    aligned = align_func(t)
                    if not aligned.empty:
                        available_ds = [d for d in datasets if d in aligned.columns]
                        if available_ds:
                            is_common = (aligned[available_ds] > 0).all(axis=1)
                            c_edge = int(is_common.sum())
                except Exception:
                    pass
            common_edge_counts.append(c_edge)
            
            # Common Paths
            common_paths = 0
            if path_presence_matrix is not None and not path_presence_matrix.empty:
                # Use path presence matrix
                # Try to match columns using sanitized names (replace : and . with _)
                available_cols = []
                for d in datasets:
                    col_orig = f"{d}_t{t}"
                    d_safe = canonical_dataset_name(d).replace(':', '_').replace('.', '_')
                    col_safe = f"{d_safe}_t{t}"
                    
                    if col_orig in path_presence_matrix.columns:
                        available_cols.append(col_orig)
                    elif col_safe in path_presence_matrix.columns:
                        available_cols.append(col_safe)
                
                if len(available_cols) == len(datasets):
                    # Check if all are True (handle both string 'True' and boolean True)
                    is_common = pd.Series(True, index=path_presence_matrix.index)
                    for col in available_cols:
                        vals = path_presence_matrix[col]
                        if vals.dtype == object:
                            is_common &= ((vals == 'True') | (vals == True))
                        elif vals.dtype == bool:
                            is_common &= vals
                        else:
                            is_common &= (vals > 0)
                    common_paths = int(is_common.sum())
            else:
                # Fallback to edge intersection
                path_sets = []
                for dataset in datasets:
                    df = results.get(dataset, {}).get(t, pd.DataFrame())
                    if not df.empty:
                        if 'has_valid_path' in df.columns:
                            valid_paths = df[df['has_valid_path'] == True]
                        else:
                            valid_paths = df
                            
                        if not valid_paths.empty:
                            # Use type_pre/type_post directly since results are already type-mapped
                            # (get_mapped_results() applies canonical type mapping)
                            # Don't use std_label_pre/std_label_post as those may contain display names
                            # like 'MeVPLo2(MTe07)' which won't match across datasets
                            pre_col = 'type_pre'
                            post_col = 'type_post'
                            if pre_col in valid_paths.columns and post_col in valid_paths.columns:
                                edges = set()
                                for _, row in valid_paths.iterrows():
                                    p = str(row[pre_col]).strip()
                                    q = str(row[post_col]).strip()
                                    if p and q:
                                        edges.add((p, q))
                                path_sets.append(edges)
                            else:
                                path_sets.append(set())
                        else:
                            path_sets.append(set())
                    else:
                        path_sets.append(set())
                
                if path_sets:
                    common_paths = len(set.intersection(*path_sets))
                else:
                    common_paths = 0
            
            common_path_counts.append(common_paths)
            
        ax_edge_count.plot(sorted_thresholds, common_edge_counts, marker='*', linestyle=':', 
                         color='black', label="Common", linewidth=2, markersize=10, alpha=0.5)
        
        ax_path_count.plot(sorted_thresholds, common_path_counts, marker='*', linestyle=':', 
                         color='black', label="Common", linewidth=2, markersize=10, alpha=0.5)
            
        # Add common counts to plot data
        for j, t in enumerate(sorted_thresholds):
            plot_data.append({
                'dataset': 'Common',
                'threshold': t,
                'edge_count': common_edge_counts[j],
                'path_count': common_path_counts[j],
                'edge_retention_rate': np.nan,
                'path_retention_rate': np.nan
            })
            plot_data.append({
                'dataset': 'Overall',
                'threshold': t,
                'edge_count': total_edge_counts[j],
                'path_count': total_path_counts[j],
                'edge_retention_rate': overall_edge_rates[j],
                'path_retention_rate': overall_path_rates[j]
            })

        # Formatting
        for ax in axes.flat:
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.set_xticks(sorted_thresholds)
        
        ax_edge_count.set_title("Edge Counts (Linear)")
        ax_edge_count.set_ylabel("Count")
        ax_edge_count.legend()
        
        ax_path_count.set_title("Path Counts (Linear)")
        ax_path_count.set_ylabel("Count")
        ax_path_count.legend()
        
        ax_edge_rate.set_title("Edge Delta Rate ($(N_{t-1} - N_t) / N_{t-1}$)")
        ax_edge_rate.set_ylabel("Delta Rate")
        ax_edge_rate.set_xlabel("Threshold")
        ax_edge_rate.legend()
        
        ax_path_rate.set_title("Path Delta Rate ($(N_{t-1} - N_t) / N_{t-1}$)")
        ax_path_rate.set_ylabel("Delta Rate")
        ax_path_rate.set_xlabel("Threshold")
        ax_path_rate.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        
        return fig, pd.DataFrame(plot_data)
    
    def plot_jaccard_similarity_trend(
        self,
        align_func,
        thresholds: List[int],
        datasets: List[str],
        title: str = "Jaccard Similarity Across Thresholds",
        nickname_map: Dict[str, str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[plt.Figure, pd.DataFrame]:
        """
        Plot Jaccard similarity trend across thresholds for all dataset pairs.
        
        Args:
            align_func: Function to get aligned data at a threshold
            thresholds: List of thresholds
            datasets: List of dataset names
            title: Plot title
            nickname_map: Dict mapping dataset names to display names
            figsize: Figure size tuple
            
        Returns:
            Tuple of (matplotlib Figure, DataFrame with Jaccard data)
        """
        if not HAS_MATPLOTLIB:
            return None, pd.DataFrame()
        
        if nickname_map is None:
            nickname_map = {d: d for d in datasets}
        
        sorted_thresholds = sorted(thresholds)
        
        # Collect Jaccard similarities for each pair at each threshold
        from itertools import combinations
        
        pair_data = {}  # {(d1, d2): {threshold: jaccard}}
        all_data = []  # For DataFrame export
        
        pairs = list(combinations(datasets, 2))
        for pair in pairs:
            pair_data[pair] = {}
        
        for threshold in sorted_thresholds:
            aligned = align_func(threshold)
            if aligned.empty:
                continue
            
            available = [d for d in datasets if d in aligned.columns]
            
            for d1, d2 in combinations(available, 2):
                pair_key = (d1, d2) if (d1, d2) in pair_data else (d2, d1)
                
                c1, c2 = aligned[d1], aligned[d2]
                s1 = set(aligned.index[c1 > 0])
                s2 = set(aligned.index[c2 > 0])
                inter, union = len(s1 & s2), len(s1 | s2)
                jac = inter / union if union > 0 else 0
                pair_data[pair_key][threshold] = jac
                
                # For export
                n1 = nickname_map.get(d1, d1)
                n2 = nickname_map.get(d2, d2)
                all_data.append({
                    'threshold': threshold,
                    'dataset1': n1,
                    'dataset2': n2,
                    'jaccard': jac,
                    'intersection': inter,
                    'union': union
                })
        
        # Create figure
        figsize = figsize or (10, 6)
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10.colors
        markers = ['o', 's', '^', 'D', 'v', '<', '>']
        
        for idx, pair_key in enumerate(pairs):
            d1, d2 = pair_key
            n1, n2 = nickname_map.get(d1, d1), nickname_map.get(d2, d2)
            
            x_vals = []
            y_vals = []
            for t in sorted_thresholds:
                if t in pair_data[pair_key]:
                    x_vals.append(t)
                    y_vals.append(pair_data[pair_key][t])
            
            if x_vals:
                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]
                ax.plot(x_vals, y_vals, marker=marker, color=color, linewidth=2,
                       markersize=8, label=f'{n1} vs {n2}')
        
        # Calculate and plot average
        avg_x = []
        avg_y = []
        for t in sorted_thresholds:
            vals = [pair_data[pk].get(t) for pk in pairs if t in pair_data.get(pk, {})]
            vals = [v for v in vals if v is not None]
            if vals:
                avg_x.append(t)
                avg_y.append(sum(vals) / len(vals))
                
                # Add average to export data
                all_data.append({
                    'threshold': t,
                    'dataset1': 'Average',
                    'dataset2': 'Average',
                    'jaccard': sum(vals) / len(vals),
                    'intersection': None,
                    'union': None
                })
        
        if avg_x:
            ax.plot(avg_x, avg_y, marker='*', color='black', linewidth=3,
                   markersize=12, linestyle='--', label='Average', alpha=0.7)
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Jaccard Index')
        ax.set_ylim(0, 1)
        ax.set_title(title)
        # Only show legend if there are labeled artists
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted_thresholds)
        
        plt.tight_layout()
        
        return fig, pd.DataFrame(all_data)

    def plot_topk_overlap_trend(
        self,
        align_func,
        thresholds: List[int],
        datasets: List[str],
        title: str = "Edge Top-20 Overlap Across Thresholds",
        nickname_map: Dict[str, str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[plt.Figure, pd.DataFrame]:
        """
        Plot the edge top-20 overlap trend across thresholds for all dataset
        pairs (v2.2 detail metric; replaces the retired Edge Rank trend).

        Args:
            align_func: Function to get aligned data at a threshold
            thresholds: List of thresholds
            datasets: List of dataset names
            title: Plot title
            nickname_map: Dict mapping dataset names to display names
            figsize: Figure size tuple

        Returns:
            Tuple of (matplotlib Figure, DataFrame with overlap data)
        """
        if not HAS_MATPLOTLIB:
            return None, pd.DataFrame()

        if nickname_map is None:
            nickname_map = {d: d for d in datasets}

        sorted_thresholds = sorted(thresholds)

        from itertools import combinations
        from .metrics import ComparisonMetrics
        metrics = ComparisonMetrics()

        pair_data = {}  # {(d1, d2): {threshold: overlap}}
        all_data = []  # For DataFrame export

        pairs = list(combinations(datasets, 2))
        for pair in pairs:
            pair_data[pair] = {}

        for threshold in sorted_thresholds:
            try:
                aligned = align_func(threshold)
            except Exception:
                continue
            if aligned.empty:
                continue

            available = [d for d in datasets if d in aligned.columns]

            for d1, d2 in combinations(available, 2):
                pair_key = (d1, d2) if (d1, d2) in pair_data else (d2, d1)

                # Get edge weights as Series
                weights_a = aligned[d1].dropna()
                weights_b = aligned[d2].dropna()

                # Top-20 overlap (head-of-ranking agreement)
                overlap = metrics.calculate_topk_overlap(weights_a, weights_b)
                pair_data[pair_key][threshold] = overlap

                # For export
                n1 = nickname_map.get(d1, d1)
                n2 = nickname_map.get(d2, d2)
                all_data.append({
                    'threshold': threshold,
                    'dataset1': n1,
                    'dataset2': n2,
                    'top20_overlap': overlap
                })
        
        # Create figure
        figsize = figsize or (10, 6)
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10.colors
        markers = ['o', 's', '^', 'D', 'v', '<', '>']
        
        for idx, pair_key in enumerate(pairs):
            d1, d2 = pair_key
            n1, n2 = nickname_map.get(d1, d1), nickname_map.get(d2, d2)
            
            x_vals = []
            y_vals = []
            for t in sorted_thresholds:
                if t in pair_data[pair_key]:
                    val = pair_data[pair_key][t]
                    # Skip NaN values in plotting
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        x_vals.append(t)
                        y_vals.append(val)
            
            if x_vals:
                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]
                ax.plot(x_vals, y_vals, marker=marker, color=color, linewidth=2,
                       markersize=8, label=f'{n1} vs {n2}')
        
        # Calculate and plot average (excluding NaN values)
        avg_x = []
        avg_y = []
        for t in sorted_thresholds:
            vals = [pair_data[pk].get(t) for pk in pairs if t in pair_data.get(pk, {})]
            # Filter out None and NaN values
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if vals:
                avg_x.append(t)
                avg_y.append(sum(vals) / len(vals))
                
                all_data.append({
                    'threshold': t,
                    'dataset1': 'Average',
                    'dataset2': 'Average',
                    'top20_overlap': sum(vals) / len(vals)
                })
        
        if avg_x:
            ax.plot(avg_x, avg_y, marker='*', color='black', linewidth=3,
                   markersize=12, linestyle='--', label='Average', alpha=0.7)
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Edge top-20 overlap')
        ax.set_ylim(0, 1)  # overlap ratio range
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(title)
        # Only show legend if there are labeled artists
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted_thresholds)
        
        plt.tight_layout()
        
        return fig, pd.DataFrame(all_data)

    def plot_netsimile_trend(
        self,
        align_func,
        thresholds: List[int],
        datasets: List[str],
        title: str = "NetSimile-lite Similarity Across Thresholds",
        nickname_map: Dict[str, str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[plt.Figure, pd.DataFrame]:
        """
        Plot the graph-level NetSimile-lite similarity trend across
        thresholds for all dataset pairs (v2.2 graph-level representative;
        replaces the retired Path Rank trend).

        Args:
            align_func: Function to get aligned data at a threshold
            thresholds: List of thresholds
            datasets: List of dataset names
            title: Plot title
            nickname_map: Dict mapping dataset names to display names
            figsize: Figure size tuple

        Returns:
            Tuple of (matplotlib Figure, DataFrame with similarity data)
        """
        if not HAS_MATPLOTLIB:
            return None, pd.DataFrame()

        if nickname_map is None:
            nickname_map = {d: d for d in datasets}

        sorted_thresholds = sorted(thresholds)

        from itertools import combinations
        from .metrics import ComparisonMetrics
        metrics = ComparisonMetrics()

        pair_data = {}  # {(d1, d2): {threshold: similarity}}
        all_data = []  # For DataFrame export

        pairs = list(combinations(datasets, 2))
        for pair in pairs:
            pair_data[pair] = {}

        for threshold in sorted_thresholds:
            try:
                aligned = align_func(threshold)
            except Exception:
                continue
            if aligned.empty:
                continue

            available = [d for d in datasets if d in aligned.columns]

            for d1, d2 in combinations(available, 2):
                pair_key = (d1, d2) if (d1, d2) in pair_data else (d2, d1)

                # Get edge weights as Series
                weights_a = aligned[d1].dropna()
                weights_b = aligned[d2].dropna()

                # NetSimile-lite similarity (alignment-free)
                sim = metrics.calculate_netsimile_similarity(weights_a, weights_b)
                pair_data[pair_key][threshold] = sim

                # For export
                n1 = nickname_map.get(d1, d1)
                n2 = nickname_map.get(d2, d2)
                all_data.append({
                    'threshold': threshold,
                    'dataset1': n1,
                    'dataset2': n2,
                    'netsimile_similarity': sim
                })
        
        # Create figure
        figsize = figsize or (10, 6)
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10.colors
        markers = ['o', 's', '^', 'D', 'v', '<', '>']
        
        for idx, pair_key in enumerate(pairs):
            d1, d2 = pair_key
            n1, n2 = nickname_map.get(d1, d1), nickname_map.get(d2, d2)
            
            x_vals = []
            y_vals = []
            for t in sorted_thresholds:
                if t in pair_data[pair_key]:
                    val = pair_data[pair_key][t]
                    # Skip NaN values in plotting
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        x_vals.append(t)
                        y_vals.append(val)
            
            if x_vals:
                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]
                ax.plot(x_vals, y_vals, marker=marker, color=color, linewidth=2,
                       markersize=8, label=f'{n1} vs {n2}')
        
        # Calculate and plot average (excluding NaN values)
        avg_x = []
        avg_y = []
        for t in sorted_thresholds:
            vals = [pair_data[pk].get(t) for pk in pairs if t in pair_data.get(pk, {})]
            # Filter out None and NaN values
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if vals:
                avg_x.append(t)
                avg_y.append(sum(vals) / len(vals))
                
                all_data.append({
                    'threshold': t,
                    'dataset1': 'Average',
                    'dataset2': 'Average',
                    'netsimile_similarity': sum(vals) / len(vals)
                })
        
        if avg_x:
            ax.plot(avg_x, avg_y, marker='*', color='black', linewidth=3,
                   markersize=12, linestyle='--', label='Average', alpha=0.7)
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('NetSimile-lite similarity')
        ax.set_ylim(0, 1)  # similarity range
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)  # Zero reference line
        ax.set_title(title)
        # Only show legend if there are labeled artists
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted_thresholds)
        
        plt.tight_layout()
        
        return fig, pd.DataFrame(all_data)

    def plot_cosine_similarity_trend(
        self,
        align_func,
        thresholds: List[int],
        datasets: List[str],
        title: str = "Cosine Similarity Across Thresholds",
        nickname_map: Dict[str, str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[plt.Figure, pd.DataFrame]:
        """
        Plot cosine similarity trend across thresholds for all dataset pairs.
        
        Uses the union of edges and compares weight distributions using cosine similarity.
        
        Args:
            align_func: Function to get aligned data at a threshold
            thresholds: List of thresholds
            datasets: List of dataset names
            title: Plot title
            nickname_map: Dict mapping dataset names to display names
            figsize: Figure size tuple
            
        Returns:
            Tuple of (matplotlib Figure, DataFrame with similarity data)
        """
        if not HAS_MATPLOTLIB:
            return None, pd.DataFrame()
        
        if nickname_map is None:
            nickname_map = {d: d for d in datasets}
        
        sorted_thresholds = sorted(thresholds)
        
        from itertools import combinations
        from .metrics import ComparisonMetrics
        metrics = ComparisonMetrics()
        
        pair_data = {}  # {(d1, d2): {threshold: cosine_sim}}
        all_data = []  # For DataFrame export
        
        pairs = list(combinations(datasets, 2))
        for pair in pairs:
            pair_data[pair] = {}
        
        for threshold in sorted_thresholds:
            try:
                aligned = align_func(threshold)
            except Exception:
                continue
            
            if aligned is None or aligned.empty:
                continue
            
            available = [d for d in datasets if d in aligned.columns]
            
            for d1, d2 in combinations(available, 2):
                pair_key = (d1, d2) if (d1, d2) in pair_data else (d2, d1)
                
                # Get edge weights as Series
                weights_a = aligned[d1].dropna()
                weights_b = aligned[d2].dropna()
                
                # Calculate cosine similarity
                cosine = metrics.calculate_cosine_similarity(weights_a, weights_b)
                pair_data[pair_key][threshold] = cosine
                
                # For export
                n1 = nickname_map.get(d1, d1)
                n2 = nickname_map.get(d2, d2)
                all_data.append({
                    'threshold': threshold,
                    'dataset1': n1,
                    'dataset2': n2,
                    'cosine_similarity': cosine
                })
        
        # Create figure
        figsize = figsize or (10, 6)
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10.colors
        markers = ['o', 's', '^', 'D', 'v', '<', '>']
        
        for idx, pair_key in enumerate(pairs):
            d1, d2 = pair_key
            n1, n2 = nickname_map.get(d1, d1), nickname_map.get(d2, d2)
            
            x_vals = []
            y_vals = []
            for t in sorted_thresholds:
                if t in pair_data[pair_key]:
                    val = pair_data[pair_key][t]
                    # Skip NaN values in plotting
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        x_vals.append(t)
                        y_vals.append(val)
            
            if x_vals:
                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]
                ax.plot(x_vals, y_vals, marker=marker, color=color, linewidth=2,
                       markersize=8, label=f'{n1} vs {n2}')
        
        # Calculate and plot average (excluding NaN values)
        avg_x = []
        avg_y = []
        for t in sorted_thresholds:
            vals = [pair_data[pk].get(t) for pk in pairs if t in pair_data.get(pk, {})]
            # Filter out None and NaN values
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if vals:
                avg_x.append(t)
                avg_y.append(sum(vals) / len(vals))
                
                all_data.append({
                    'threshold': t,
                    'dataset1': 'Average',
                    'dataset2': 'Average',
                    'cosine_similarity': sum(vals) / len(vals)
                })
        
        if avg_x:
            ax.plot(avg_x, avg_y, marker='*', color='black', linewidth=3,
                   markersize=12, linestyle='--', label='Average', alpha=0.7)
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Cosine Similarity')
        ax.set_ylim(0, 1)
        ax.set_title(title)
        # Only show legend if there are labeled artists
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted_thresholds)
        
        plt.tight_layout()
        
        return fig, pd.DataFrame(all_data)

    def plot_conservation_across_thresholds_plotly(
        self,
        results: Dict[str, Dict[int, pd.DataFrame]],
        thresholds: List[int],
        align_func=None,
        title: str = "Conservation of Edges and Paths Across Thresholds",
        nickname_map: Dict[str, str] = None,
        path_presence_matrix: pd.DataFrame = None
    ) -> str:
        """
        Generate an interactive Plotly JSON for conservation analysis.
        
        Args:
            results: Dictionary of results[dataset][threshold] -> DataFrame
            thresholds: List of thresholds
            align_func: Function to get aligned data for "Common" calculation
            title: Plot title
            nickname_map: Map of dataset names to display names
            path_presence_matrix: Optional DataFrame with path presence info
            
        Returns:
            JSON string of the Plotly figure
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import json
        
        datasets = list(results.keys())
        if nickname_map is None:
            nickname_map = {d: d for d in datasets}
            
        sorted_thresholds = sorted(thresholds)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("<b>Edge Counts</b> (Linear)", "<b>Path Counts</b> (Linear)", 
                           "<b>Edge Delta Rate</b><br>((N<sub>t-1</sub> - N<sub>t</sub>) / N<sub>t-1</sub>)", "<b>Path Delta Rate</b><br>((N<sub>t-1</sub> - N<sub>t</sub>) / N<sub>t-1</sub>)"),
            vertical_spacing=0.18,
            horizontal_spacing=0.1,
            shared_xaxes=True
        )
        
        # Colors and Markers
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        markers = ['circle', 'square', 'diamond', 'triangle-up', 'triangle-down', 'cross', 'x']
        
        # Track totals for Overall Rate calculation
        total_edge_counts = [0] * len(sorted_thresholds)
        total_path_counts = [0] * len(sorted_thresholds)

        for i, dataset in enumerate(datasets):
            nick = nickname_map.get(dataset, dataset)
            color = colors[i % len(colors)]
            marker_symbol = markers[i % len(markers)]
            
            edge_counts = []
            path_counts = []
            
            for idx, t in enumerate(sorted_thresholds):
                df = results.get(dataset, {}).get(t, pd.DataFrame())
                
                # Edge Count
                e_count = len(df) if not df.empty else 0
                edge_counts.append(e_count)
                total_edge_counts[idx] += e_count
                
                # Path Count
                p_count = 0
                if path_presence_matrix is not None and not path_presence_matrix.empty:
                    # Use path presence matrix if available
                    # Try original name first, then sanitized name (replace : and . with _)
                    col_orig = f"{dataset}_t{t}"
                    dataset_safe = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
                    col_safe = f"{dataset_safe}_t{t}"
                    
                    col_name = None
                    if col_orig in path_presence_matrix.columns:
                        col_name = col_orig
                    elif col_safe in path_presence_matrix.columns:
                        col_name = col_safe
                    
                    if col_name is not None:
                        vals = path_presence_matrix[col_name]
                        if vals.dtype == object:
                            # Handle both string 'True' and boolean True
                            p_count = int(((vals == 'True') | (vals == True)).sum())
                        elif vals.dtype == bool:
                            p_count = int(vals.sum())
                        else:
                            p_count = int((vals > 0).sum())
                    else:
                        # Fallback if column missing but matrix exists
                        if not df.empty and 'has_valid_path' in df.columns:
                            p_count = int(df['has_valid_path'].sum())
                        else:
                            p_count = len(df) if not df.empty else 0
                else:
                    # Fallback
                    if not df.empty and 'has_valid_path' in df.columns:
                        p_count = int(df['has_valid_path'].sum())
                    else:
                        p_count = len(df) if not df.empty else 0
                
                path_counts.append(p_count)
                total_path_counts[idx] += p_count
            
            # Calculate delta rates: (N_{t-1} - N_t) / N_{t-1}
            edge_rates = [None]
            path_rates = [None]
            for j in range(1, len(sorted_thresholds)):
                prev_e = edge_counts[j-1]
                curr_e = edge_counts[j]
                edge_rates.append(((prev_e - curr_e) / prev_e) if prev_e > 0 else 0)
                
                prev_p = path_counts[j-1]
                curr_p = path_counts[j]
                path_rates.append(((prev_p - curr_p) / prev_p) if prev_p > 0 else 0)
            
            # Common hover template
            hover_template = f"<b>{nick}</b>: %{{y}}<extra></extra>"
            rate_hover_template = f"<b>{nick}</b>: %{{y:.3f}}<extra></extra>"

            # Add traces
            # 1. Edge Counts
            fig.add_trace(
                go.Scatter(x=sorted_thresholds, y=edge_counts, mode='lines+markers',
                          name=f"{nick}", line=dict(color=color, width=2),
                          marker=dict(symbol=marker_symbol, size=8),
                          opacity=0.6,
                          legendgroup=nick, showlegend=True,
                          hovertemplate=hover_template),
                row=1, col=1
            )
            
            # 2. Path Counts
            fig.add_trace(
                go.Scatter(x=sorted_thresholds, y=path_counts, mode='lines+markers',
                          name=f"{nick} (Paths)", line=dict(color=color, width=2, dash='dash'),
                          marker=dict(symbol=marker_symbol, size=8),
                          opacity=0.6,
                          legendgroup=nick, showlegend=False,
                          hovertemplate=hover_template),
                row=1, col=2
            )
            
            # 3. Edge Rates
            fig.add_trace(
                go.Scatter(x=sorted_thresholds, y=edge_rates, mode='lines+markers',
                          name=f"{nick} (Edge Rate)", line=dict(color=color, width=2),
                          marker=dict(symbol=marker_symbol, size=8),
                          opacity=0.6,
                          legendgroup=nick, showlegend=False,
                          hovertemplate=rate_hover_template),
                row=2, col=1
            )
            
            # 4. Path Rates
            fig.add_trace(
                go.Scatter(x=sorted_thresholds, y=path_rates, mode='lines+markers',
                          name=f"{nick} (Path Rate)", line=dict(color=color, width=2, dash='dash'),
                          marker=dict(symbol=marker_symbol, size=8),
                          opacity=0.6,
                          legendgroup=nick, showlegend=False,
                          hovertemplate=rate_hover_template),
                row=2, col=2
            )

        # --- Overall Delta Rates ---
        overall_edge_rates = [None]
        overall_path_rates = [None]
        for j in range(1, len(sorted_thresholds)):
            prev_e = total_edge_counts[j-1]
            curr_e = total_edge_counts[j]
            overall_edge_rates.append(((prev_e - curr_e) / prev_e) if prev_e > 0 else 0)
            
            prev_p = total_path_counts[j-1]
            curr_p = total_path_counts[j]
            overall_path_rates.append(((prev_p - curr_p) / prev_p) if prev_p > 0 else 0)
            
        # Add Overall Edge Rate Trace
        fig.add_trace(
            go.Scatter(x=sorted_thresholds, y=overall_edge_rates, mode='lines+markers',
                      name="Overall", line=dict(color='black', width=3, dash='solid'),
                      marker=dict(symbol='star', size=10, color='black'),
                      opacity=0.3,
                      legendgroup="Overall", showlegend=True,
                      hovertemplate="<b>Overall</b>: %{y:.3f}<extra></extra>"),
            row=2, col=1
        )
        
        # Add Overall Path Rate Trace
        fig.add_trace(
            go.Scatter(x=sorted_thresholds, y=overall_path_rates, mode='lines+markers',
                      name="Overall", line=dict(color='black', width=3, dash='dash'),
                      marker=dict(symbol='star', size=10, color='black'),
                      opacity=0.3,
                      legendgroup="Overall", showlegend=False,
                      hovertemplate="<b>Overall</b>: %{y:.3f}<extra></extra>"),
            row=2, col=2
        )

        # --- Common Counts ---
        common_edge_counts = []
        common_path_counts = []
        
        for t in sorted_thresholds:
            # Common Edges
            c_edge = 0
            if align_func:
                try:
                    aligned = align_func(t)
                    if not aligned.empty:
                        available_ds = [d for d in datasets if d in aligned.columns]
                        if available_ds:
                            is_common = (aligned[available_ds] > 0).all(axis=1)
                            c_edge = int(is_common.sum())
                except Exception:
                    pass
            common_edge_counts.append(c_edge)
            
            # Common Paths
            if path_presence_matrix is not None and not path_presence_matrix.empty:
                # Use path presence matrix
                # Build column list with sanitized names
                available_cols = []
                for d in datasets:
                    col_orig = f"{d}_t{t}"
                    d_safe = canonical_dataset_name(d).replace(':', '_').replace('.', '_')
                    col_safe = f"{d_safe}_t{t}"
                    
                    if col_orig in path_presence_matrix.columns:
                        available_cols.append(col_orig)
                    elif col_safe in path_presence_matrix.columns:
                        available_cols.append(col_safe)
                
                if len(available_cols) == len(datasets):
                    is_common = pd.Series(True, index=path_presence_matrix.index)
                    for col in available_cols:
                        vals = path_presence_matrix[col]
                        if vals.dtype == object:
                            # Handle both string 'True' and boolean True
                            is_common &= ((vals == 'True') | (vals == True))
                        elif vals.dtype == bool:
                            is_common &= vals
                        else:
                            is_common &= (vals > 0)
                    common_path_counts.append(int(is_common.sum()))
                else:
                    common_path_counts.append(0)
            else:
                # Intersection of edges that are valid paths in all datasets
                path_sets = []
                for dataset in datasets:
                    df = results.get(dataset, {}).get(t, pd.DataFrame())
                    if not df.empty:
                        # Filter for valid paths
                        if 'has_valid_path' in df.columns:
                            valid_paths = df[df['has_valid_path'] == True]
                        else:
                            # If column missing, assume all are valid (e.g. path mode or base threshold)
                            valid_paths = df
                        
                        if not valid_paths.empty:
                            # Use type_pre/type_post directly since results are already type-mapped
                            # (get_mapped_results() applies canonical type mapping)
                            # Don't use std_label_pre/std_label_post as those may contain display names
                            # like 'MeVPLo2(MTe07)' which won't match across datasets
                            pre_col = 'type_pre'
                            post_col = 'type_post'
                            
                            if pre_col in valid_paths.columns and post_col in valid_paths.columns:
                                # Create set of edges (ensure string and strip whitespace)
                                edges = set()
                                for _, row in valid_paths.iterrows():
                                    p = str(row[pre_col]).strip()
                                    q = str(row[post_col]).strip()
                                    if p and q:  # Skip empty
                                        edges.add((p, q))
                                path_sets.append(edges)
                            else:
                                path_sets.append(set())
                        else:
                            path_sets.append(set())
                    else:
                        path_sets.append(set())
                
                if path_sets:
                    common_paths = set.intersection(*path_sets)
                    common_path_counts.append(len(common_paths))
                else:
                    common_path_counts.append(0)
            
        # Add Common Edge Counts Trace
        fig.add_trace(
            go.Scatter(x=sorted_thresholds, y=common_edge_counts, mode='lines+markers',
                      name="Common", line=dict(color='black', width=3, dash='dot'),
                      marker=dict(symbol='star', size=12, color='black'),
                      opacity=0.3,
                      legendgroup="Common", showlegend=True,
                      hovertemplate="<b>Common (Edges)</b>: %{y}<extra></extra>"),
            row=1, col=1
        )
        
        # Add Common Path Counts Trace
        fig.add_trace(
            go.Scatter(x=sorted_thresholds, y=common_path_counts, mode='lines+markers',
                      name="Common", line=dict(color='black', width=3, dash='dot'),
                      marker=dict(symbol='star', size=12, color='black'),
                      opacity=0.3,
                      legendgroup="Common", showlegend=False,
                      hovertemplate="<b>Common (Paths)</b>: %{y}<extra></extra>"),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=20)),
            height=900,
            width=None,  # Responsive
            template="plotly_white",
            legend=dict(
                orientation="v", 
                yanchor="top", y=1, 
                xanchor="left", x=1.02,
                title=dict(text="Datasets"),
                bordercolor="Black",
                borderwidth=1
            ),
            margin=dict(l=60, r=150, t=80, b=60),
            hovermode="x unified"
        )
        
        # Update axes
        fig.update_xaxes(title_text="Threshold", tickvals=sorted_thresholds, row=2, col=1)
        fig.update_xaxes(title_text="Threshold", tickvals=sorted_thresholds, row=2, col=2)
        # Hide x-labels for top row since shared_xaxes=True handles it, but we need to ensure ticks align
        fig.update_xaxes(tickvals=sorted_thresholds, row=1, col=1)
        fig.update_xaxes(tickvals=sorted_thresholds, row=1, col=2)
        
        fig.update_yaxes(title_text="Count", row=1, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Count", row=1, col=2, gridcolor='lightgray')
        fig.update_yaxes(title_text="Delta Rate", row=2, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Delta Rate", row=2, col=2, gridcolor='lightgray')
        
        return json.dumps(fig.to_dict(), cls=None)

    def _generate_vis_summary_pdf(self, output_dir: str):
        """
        Generate a consolidated PDF from all PNG files in output_dir and subfolders.
        Each image is padded to fit on an A4 page with margins.
        
        Files are sorted by category:
        1. General overview visualizations (root folder)
        2. Per-threshold visualizations (sorted by threshold number)
        3. Other subfolders alphabetically
        
        Args:
            output_dir: Directory containing PNG files
        """
        from PIL import Image
        import re
        
        # A4 dimensions at 300 DPI for PDF (in pixels)
        PDF_DPI = 300
        A4_WIDTH = int(8.27 * PDF_DPI)   # 210mm = 8.27 inches
        A4_HEIGHT = int(11.69 * PDF_DPI)  # 297mm = 11.69 inches
        MARGIN = 100  # pixels margin
        
        # Usable area
        usable_width = A4_WIDTH - 2 * MARGIN
        usable_height = A4_HEIGHT - 2 * MARGIN
        
        # Collect all PNG files with categorization
        root_files = []
        threshold_files = {}  # threshold -> list of files
        other_folders = {}  # folder_name -> list of files
        
        for root, dirs, files in os.walk(output_dir):
            # Skip visualization_data folder
            if 'visualization_data' in root:
                continue
            
            rel_root = os.path.relpath(root, output_dir)
            
            for f in sorted(files):
                if not f.endswith('.png'):
                    continue
                    
                full_path = os.path.join(root, f)
                
                if rel_root == '.':
                    # Root folder - general visualizations
                    root_files.append(full_path)
                else:
                    # Check if it's a threshold folder (e.g., "threshold_1", "minsyn_5")
                    threshold_match = re.search(r'(?:threshold|minsyn)[_\s]*(\d+)', rel_root, re.IGNORECASE)
                    if threshold_match:
                        thresh_num = int(threshold_match.group(1))
                        if thresh_num not in threshold_files:
                            threshold_files[thresh_num] = []
                        threshold_files[thresh_num].append(full_path)
                    else:
                        # Other subfolder
                        folder_name = rel_root.split(os.sep)[0]
                        if folder_name not in other_folders:
                            other_folders[folder_name] = []
                        other_folders[folder_name].append(full_path)
        
        # Build ordered list: root files first, then thresholds (sorted), then others
        png_files = []
        
        # 1. Root files (general overview) - sorted alphabetically
        png_files.extend(sorted(root_files))
        
        # 2. Threshold files - sorted by threshold number, then by filename
        for thresh in sorted(threshold_files.keys()):
            png_files.extend(sorted(threshold_files[thresh]))
        
        # 3. Other folders - sorted alphabetically by folder name, then by filename
        for folder in sorted(other_folders.keys()):
            png_files.extend(sorted(other_folders[folder]))
        
        if not png_files:
            self._vprint("No PNG files found for PDF generation")
            return
        
        # Convert PNGs to A4 pages with padding
        pages = []
        for png_path in png_files:
            try:
                img = Image.open(png_path)
                
                # Convert to RGB if necessary (removes alpha channel)
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate scaling to fit in usable area while maintaining aspect ratio
                img_width, img_height = img.size
                scale_w = usable_width / img_width
                scale_h = usable_height / img_height
                scale = min(scale_w, scale_h, 1.0)  # Don't upscale
                
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                # Resize if needed
                if scale < 1.0:
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Create A4 page with white background
                page = Image.new('RGB', (A4_WIDTH, A4_HEIGHT), (255, 255, 255))
                
                # Center the image on the page
                x = (A4_WIDTH - new_width) // 2
                y = (A4_HEIGHT - new_height) // 2
                page.paste(img, (x, y))
                
                pages.append(page)
            except Exception as e:
                self._vprint(f"Warning: Could not process {png_path}: {e}")
        
        if not pages:
            self._vprint("No images could be processed for PDF generation")
            return
        
        # Save as multi-page PDF without compression - to parent directory (outside visualizations folder)
        parent_dir = os.path.dirname(output_dir)
        pdf_path = os.path.join(parent_dir, "vis_summary.pdf")
        pages[0].save(
            pdf_path,
            "PDF",
            save_all=True,
            append_images=pages[1:] if len(pages) > 1 else [],
            resolution=PDF_DPI,
            quality=100  # No compression
        )
        self._vprint(f"Generated PDF summary: {pdf_path}")
