"""
Profile Visualization Module

This module provides visualization tools for connectivity profiles,
including grouped bar charts, similarity heatmaps, and HTML reports.

Main components:
- ProfileVisualizer: Class with methods for generating visualizations
- HTML report generation for verification results

Visualization Types:
- Grouped bar charts: Compare profiles side-by-side across datasets
- Similarity heatmaps: Show pairwise similarity scores
- Partner overlap networks: Interactive vis.js networks
- HTML dashboard: Comprehensive verification report

Example:
    >>> from src.comparison import ProfileVisualizer
    >>> 
    >>> fig = ProfileVisualizer.plot_profile_comparison(
    ...     profiles, 'aMe12', direction='both'
    ... )
    >>> plt.show()
"""

import warnings

# Suppress matplotlib warnings about too many figures
warnings.filterwarnings("ignore", message="More than 20 figures have been opened")


def _inline_plotly() -> str:
    """plotly.js as an inline ``<script>`` block (plan-offline-html-exports
    Phase C): read from the installed plotly.py's package_data so the
    library is version-matched to the figure producer and the HTML works
    offline; CDN tag when plotly is unavailable. Also removes this
    module's historical unpinned ``plotly-latest`` reference."""
    try:
        from vendored_assets import inline_plotly as _inline
    except ImportError:  # pragma: no cover - direct package imports
        try:
            from src.vendored_assets import inline_plotly as _inline
        except ImportError:
            return ('<script src="https://cdn.plot.ly/'
                    'plotly-2.35.2.min.js"></script>')
    return _inline()

# Set non-interactive backend before importing pyplot to prevent display issues
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.max_open_warning'] = 100

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd

# Import connectivity profile types
from .connectivity_profiler import ConnectivityProfile
from .profile_comparator import ProfileComparator


# ============================================================================
# Profile Visualizer Class
# ============================================================================

class ProfileVisualizer:
    """
    Visualization utilities for connectivity profiles.
    
    Provides methods for generating various visualizations to understand
    and compare connectivity profiles across datasets.
    
    All methods are static and can be called without instantiation.
    """
    
    @staticmethod
    def plot_profile_comparison(
        profiles: Dict[str, ConnectivityProfile],
        neuron_type: str,
        direction: str = 'both',
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ) -> Any:
        """
        Generate grouped bar chart comparing profiles across datasets.
        
        Creates a side-by-side bar chart showing the top partners for each
        dataset, with shared partners highlighted.
        
        When direction='both', creates separate subplots for upstream and
        downstream to avoid confusing summed proportions.
        
        Args:
            profiles: Dict mapping dataset name to ConnectivityProfile
            neuron_type: Type name for title
            direction: 'upstream', 'downstream', or 'both'
            output_path: Optional path to save figure
            figsize: Figure size tuple (width, height)
        
        Returns:
            matplotlib Figure object
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("matplotlib required for visualization")
        
        datasets = list(profiles.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))
        
        # If direction is 'both', create two subplots (upstream and downstream)
        if direction == 'both':
            fig, axes = plt.subplots(2, 1, figsize=figsize)
            
            for ax_idx, dir_name in enumerate(['upstream', 'downstream']):
                ax = axes[ax_idx]
                
                # Collect partners for this direction
                all_partners = set()
                for profile in profiles.values():
                    if dir_name == 'upstream':
                        all_partners.update(profile.upstream_partners.keys())
                    else:
                        all_partners.update(profile.downstream_partners.keys())
                
                if not all_partners:
                    ax.text(0.5, 0.5, f'No {dir_name} partner data available', 
                           ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Sort by average weight
                partner_avg_weight = {}
                for partner in all_partners:
                    weights = []
                    for profile in profiles.values():
                        if dir_name == 'upstream':
                            w = profile.upstream_partners.get(partner, 0)
                        else:
                            w = profile.downstream_partners.get(partner, 0)
                        if w > 0:
                            weights.append(w)
                    partner_avg_weight[partner] = np.mean(weights) if weights else 0
                
                sorted_partners = sorted(all_partners, key=lambda p: partner_avg_weight[p], reverse=True)
                
                # Limit to top 12 partners for readability
                if len(sorted_partners) > 12:
                    sorted_partners = sorted_partners[:12]
                
                x = np.arange(len(sorted_partners))
                width = 0.8 / len(datasets)
                
                for i, (dataset, profile) in enumerate(profiles.items()):
                    if dir_name == 'upstream':
                        partners = profile.upstream_partners
                    else:
                        partners = profile.downstream_partners
                    
                    values = [partners.get(p, 0) for p in sorted_partners]
                    offset = (i - len(datasets)/2 + 0.5) * width
                    ax.bar(x + offset, values, width, label=dataset, color=colors[i])
                
                # Identify shared partners
                shared_partners = set(sorted_partners)
                for profile in profiles.values():
                    if dir_name == 'upstream':
                        partners = set(profile.upstream_partners.keys())
                    else:
                        partners = set(profile.downstream_partners.keys())
                    shared_partners &= partners
                
                # Add green background for shared partners
                for i, partner in enumerate(sorted_partners):
                    if partner in shared_partners:
                        ax.axvspan(i - 0.45, i + 0.45, alpha=0.15, color='green', zorder=0)
                
                # Labels and styling
                ax.set_xlabel('Partner Type', fontsize=10)
                # Upstream = pre-synaptic inputs, Downstream = post-synaptic outputs
                ylabel = 'Pre-synaptic Proportion' if dir_name == 'upstream' else 'Post-synaptic Proportion'
                ax.set_ylabel(ylabel, fontsize=10)
                ax.set_title(f'{dir_name.capitalize()} Partners ({ylabel})', fontsize=11, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(sorted_partners, rotation=45, ha='right', fontsize=8)
                ax.grid(axis='y', alpha=0.3)
                ax.set_ylim(0, None)  # Ensure y-axis starts at 0
                
                # Add legend only to first subplot
                if ax_idx == 0:
                    shared_patch = mpatches.Patch(color='green', alpha=0.15, 
                                                  label=f'Shared partners')
                    handles, labels = ax.get_legend_handles_labels()
                    handles.append(shared_patch)
                    labels.append('Shared')
                    ax.legend(handles, labels, loc='upper right', fontsize=8)
            
            fig.suptitle(f'{neuron_type} Connectivity Profile Comparison', fontsize=13, fontweight='bold')
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=600, bbox_inches='tight')
            
            return fig
        
        # Single direction (upstream or downstream only)
        all_partners = set()
        for profile in profiles.values():
            if direction == 'upstream':
                all_partners.update(profile.upstream_partners.keys())
            else:
                all_partners.update(profile.downstream_partners.keys())
        
        if not all_partners:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No partner data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Sort by average weight across datasets
        partner_avg_weight = {}
        for partner in all_partners:
            weights = []
            for profile in profiles.values():
                if direction == 'upstream':
                    w = profile.upstream_partners.get(partner, 0)
                else:
                    w = profile.downstream_partners.get(partner, 0)
                if w > 0:
                    weights.append(w)
            partner_avg_weight[partner] = np.mean(weights) if weights else 0
        
        sorted_partners = sorted(all_partners, key=lambda p: partner_avg_weight[p], reverse=True)
        
        # Limit to top 15 partners for readability
        if len(sorted_partners) > 15:
            sorted_partners = sorted_partners[:15]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]//2))
        
        x = np.arange(len(sorted_partners))
        width = 0.8 / len(datasets)
        
        for i, (dataset, profile) in enumerate(profiles.items()):
            if direction == 'upstream':
                partners = profile.upstream_partners
            else:
                partners = profile.downstream_partners
            
            values = [partners.get(p, 0) for p in sorted_partners]
            offset = (i - len(datasets)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=dataset, color=colors[i])
        
        # Identify shared partners
        shared_partners = set(sorted_partners)
        for profile in profiles.values():
            if direction == 'upstream':
                partners = set(profile.upstream_partners.keys())
            else:
                partners = set(profile.downstream_partners.keys())
            shared_partners &= partners
        
        # Add green background for shared partners
        for i, partner in enumerate(sorted_partners):
            if partner in shared_partners:
                ax.axvspan(i - 0.45, i + 0.45, alpha=0.15, color='green', zorder=0)
        
        # Labels and styling
        ax.set_xlabel('Partner Type', fontsize=11)
        # Upstream = pre-synaptic inputs, Downstream = post-synaptic outputs
        ylabel = 'Pre-synaptic Proportion' if direction == 'upstream' else 'Post-synaptic Proportion'
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{neuron_type} Connectivity Profile Comparison ({direction} - {ylabel})', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_partners, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, None)
        
        # Add legend
        shared_patch = mpatches.Patch(color='green', alpha=0.15, 
                                      label=f'Shared partners ({len(shared_partners)})')
        handles, labels = ax.get_legend_handles_labels()
        handles.append(shared_patch)
        labels.append(f'Shared partners ({len(shared_partners)})')
        ax.legend(handles, labels, loc='upper right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=600, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_similarity_heatmap(
        similarity_matrix: pd.DataFrame,
        title: str = 'Cross-Dataset Profile Similarity',
        output_path: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        cmap: str = 'RdYlGn',
        vmin: float = 0.0,
        vmax: float = 1.0,
        annot: bool = True
    ) -> Any:
        """
        Generate heatmap of similarity scores.
        
        Args:
            similarity_matrix: DataFrame with similarity values
            title: Plot title
            output_path: Optional path to save figure
            figsize: Figure size (auto-calculated if None to prevent text overlap)
            cmap: Colormap name
            vmin: Minimum value for color scale
            vmax: Maximum value for color scale
            annot: Show values in cells
        
        Returns:
            matplotlib Figure object
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("matplotlib and seaborn required for visualization")
        
        # Handle empty matrix
        if similarity_matrix.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'No similarity data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Auto-calculate figure size to prevent text overlap
        # Width: base + per-column width, Height: base + per-row height
        if figsize is None:
            n_cols = len(similarity_matrix.columns)
            n_rows = len(similarity_matrix.index)
            # Minimum 2 inches per column, 0.4 inches per row
            width = max(10, 3 + n_cols * 2.5)
            height = max(8, 2 + n_rows * 0.45)
            figsize = (width, height)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            similarity_matrix,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=annot,
            fmt='.2f',
            square=False,  # Allow rectangular cells for better space usage
            linewidths=0.5,
            cbar_kws={'label': 'Similarity Score'}
        )
        
        ax.set_title(title, fontsize=13, pad=15)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=600, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_multi_metric_heatmaps(
        metric_matrices: Dict[str, pd.DataFrame],
        output_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'RdYlGn'
    ) -> Dict[str, Any]:
        """
        Generate heatmaps for multiple metrics in a single call.
        
        Round 4 Feature: Visualize similarity matrix of each single score.
        
        Args:
            metric_matrices: Dict with keys like 'rank', 'jaccard', 'cosine'
            output_dir: Optional directory to save all figures
            figsize: Figure size for each heatmap
            cmap: Colormap name
        
        Returns:
            Dict mapping metric name to matplotlib Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for visualization")
        
        figures = {}
        
        for metric_name, matrix in metric_matrices.items():
            title = f'Connectivity Profile Similarity ({metric_name.capitalize()})'
            
            # Handle rank correlation which can be negative
            if metric_name == 'rank':
                vmin, vmax = -1.0, 1.0
            else:
                vmin, vmax = 0.0, 1.0
            
            fig = ProfileVisualizer.plot_similarity_heatmap(
                similarity_matrix=matrix,
                title=title,
                output_path=str(Path(output_dir) / f'similarity_heatmap_{metric_name}.png') if output_dir else None,
                figsize=figsize,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )
            figures[metric_name] = fig
            plt.close(fig)  # Close to avoid memory issues
        
        return figures
    
    @staticmethod
    def plot_directional_heatmaps(
        directional_matrices: Dict[str, pd.DataFrame],
        metric_name: str = 'rank',
        output_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'RdYlGn'
    ) -> Dict[str, Any]:
        """
        Generate heatmaps for upstream/downstream directions.
        
        Round 4 Feature: Keep upstream and downstream scores separate.
        
        Args:
            directional_matrices: Dict with keys 'upstream', 'downstream', 'both'
            metric_name: Name of metric for title
            output_dir: Optional directory to save figures
            figsize: Figure size for each heatmap
            cmap: Colormap name
        
        Returns:
            Dict mapping direction to matplotlib Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for visualization")
        
        figures = {}
        
        direction_titles = {
            'upstream': 'Upstream Partners',
            'downstream': 'Downstream Partners',
            'both': 'Both Directions'
        }
        
        for direction, matrix in directional_matrices.items():
            title = f'Connectivity Profile Similarity - {direction_titles.get(direction, direction)} ({metric_name})'
            
            fig = ProfileVisualizer.plot_similarity_heatmap(
                similarity_matrix=matrix,
                title=title,
                output_path=str(Path(output_dir) / f'similarity_heatmap_{direction}.png') if output_dir else None,
                figsize=figsize,
                cmap=cmap
            )
            figures[direction] = fig
            plt.close(fig)
        
        return figures

    @staticmethod
    def plot_verification_summary(
        verification_df: pd.DataFrame,
        title: str = 'Type Verification Summary',
        output_path: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Any:
        """
        Generate bar chart summarizing verification results.
        
        Deduplicates types that appear multiple times (e.g., source and target)
        and shows aggregated scores with error bars when applicable.
        
        Args:
            verification_df: DataFrame from batch_verify_types
            title: Plot title
            output_path: Optional path to save figure
            figsize: Figure size (auto-calculated if None to prevent ylabel overlap)
        
        Returns:
            matplotlib Figure object
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("matplotlib required for visualization")
        
        if verification_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No verification data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Deduplicate types - aggregate scores for types appearing multiple times
        score_col = 'avg_rank_corr'
        
        # Group by neuron_type and aggregate
        type_groups = verification_df.groupby('neuron_type')
        
        agg_data = []
        for type_name, group in type_groups:
            scores = group[score_col].dropna()
            # Merge roles
            roles = group['role'].unique()
            merged_role = '/'.join(sorted(set(r for r in roles if pd.notna(r))))
            
            # For error bars: use min_score and max_score if available (variance across dataset pairs)
            min_score = group.get('min_score', pd.Series(dtype=float)).dropna()
            max_score = group.get('max_score', pd.Series(dtype=float)).dropna()
            
            mean_val = scores.mean() if len(scores) > 0 else np.nan
            
            # Calculate error bar bounds: use min/max across all entries if available
            if len(min_score) > 0 and len(max_score) > 0:
                err_low = mean_val - min_score.min() if not pd.isna(mean_val) else 0
                err_high = max_score.max() - mean_val if not pd.isna(mean_val) else 0
            else:
                # Fallback: use std across group rows (less meaningful but shows variance)
                std_val = scores.std() if len(scores) > 1 else 0
                err_low = err_high = std_val if not pd.isna(std_val) else 0
            
            agg_data.append({
                'neuron_type': type_name,
                'mean_score': mean_val if not pd.isna(mean_val) else 0,
                'err_low': max(0, err_low) if not pd.isna(err_low) else 0,
                'err_high': max(0, err_high) if not pd.isna(err_high) else 0,
                'role': merged_role,
                'count': len(group)
            })
        
        df = pd.DataFrame(agg_data)
        df = df.sort_values('mean_score', ascending=True)
        
        # Auto-calculate figure size based on number of unique types
        if figsize is None:
            n_types = len(df)
            height = max(6, n_types * 0.35)  # At least 0.35 inches per type
            figsize = (12, height)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar chart with asymmetric error bars (min to max range)
        y_pos = np.arange(len(df))
        xerr = np.array([[df['err_low'].values], [df['err_high'].values]]).reshape(2, -1)
        bars = ax.barh(y_pos, df['mean_score'], xerr=xerr, 
                      color='#4C70B0', capsize=3, error_kw={'linewidth': 1, 'alpha': 0.7})
        
        # Labels with role indicators
        labels = []
        for _, row in df.iterrows():
            label = row['neuron_type']
            if row['count'] > 1:
                label += f" ({row['role']})"  # Show role if duplicated
            labels.append(label)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Rank Correlation')
        ax.set_title(title)
        ax.set_xlim(-1.0, 1.0)
        
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=600, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_role_comparison(
        verification_df: pd.DataFrame,
        title: str = 'Verification by Role',
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> Any:
        """
        Generate grouped bar chart with swarm plot comparing verification scores by role.
        
        Args:
            verification_df: DataFrame with 'role' column (source/target/intermediate)
            title: Plot title
            output_path: Optional path to save figure
            figsize: Figure size
        
        Returns:
            matplotlib Figure object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for visualization")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if verification_df.empty or 'role' not in verification_df.columns:
            ax.text(0.5, 0.5, 'No role data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Deduplicate types - keep only first occurrence of each neuron_type
        df_dedup = verification_df.drop_duplicates(subset='neuron_type', keep='first').copy()
        
        # Group by role
        role_colors = {'source': '#2196F3', 'target': '#4CAF50', 'intermediate': '#9C27B0', 
                       'source/target': '#FF9800'}
        
        # Only show source, target, intermediate (not combined source/target)
        # Types with 'source/target' will be counted in both source and target bars
        roles = ['source', 'target', 'intermediate']
        
        # Filter to available roles (check if any data exists for each role)
        available_roles = []
        for r in roles:
            # Check if any type has this role (including source/target types)
            has_data = any(r in str(role).split('/') for role in df_dedup['role'].values)
            if has_data:
                available_roles.append(r)
        roles = available_roles
        
        x = np.arange(len(roles))
        width = 0.6
        
        # Calculate stats for each role
        means = []
        stds = []
        counts = []
        all_points = []  # For swarm plot
        
        score_col = 'avg_rank_corr'
        
        for r in roles:
            # Match role in the role string (e.g., 'source' matches both 'source' and 'source/target')
            mask = df_dedup['role'].apply(lambda x: r in str(x).split('/'))
            role_scores = df_dedup.loc[mask, score_col].dropna()
            means.append(role_scores.mean() if len(role_scores) > 0 else 0)
            stds.append(role_scores.std() if len(role_scores) > 1 else 0)
            counts.append(len(role_scores))
            all_points.append(role_scores.values)
        
        colors = [role_colors.get(r, '#9E9E9E') for r in roles]
        
        # Draw boxplot first (without outliers, as swarm will show them)
        bp = ax.boxplot(all_points, positions=x, widths=width*0.8, patch_artist=True,
                       showfliers=False, zorder=1)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
        
        # Add swarm plot (individual data points)
        for i, (points, color) in enumerate(zip(all_points, colors)):
            if len(points) > 0:
                # Add jitter to x positions for swarm effect
                jitter = np.random.uniform(-width*0.3, width*0.3, len(points))
                ax.scatter(np.full(len(points), x[i]) + jitter, points, 
                          c=color, alpha=0.7, s=50, edgecolors='white', linewidths=0.5, zorder=2)
        
        # Add count labels
        for i, count in enumerate(counts):
            ax.text(x[i], ax.get_ylim()[1] * 0.95, f'n={count}', 
                   ha='center', va='top', fontsize=10, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels([r.replace('/', '/\n') for r in roles])
        ax.set_ylabel('Rank Correlation')
        ax.set_title(title)
        ax.set_ylim(-1.05, 1.05)
        
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=600, bbox_inches='tight')
        
        return fig
    
    # ========================================================================
    # Inter-Type Similarity Heatmaps
    # ========================================================================
    
    @staticmethod
    def generate_inter_type_similarity_heatmap(
        similarity_matrix: pd.DataFrame,
        output_path: str,
        title: str = 'Inter-Type Connectivity Similarity',
        cluster: bool = True,
        use_vispath: bool = False
    ) -> str:
        """
        Generate interactive inter-type similarity heatmap.
        
        Args:
            similarity_matrix: DataFrame with types as index/columns, similarity as values
            output_path: Path to save the HTML file
            title: Title for the heatmap
            cluster: Whether to apply hierarchical clustering
            use_vispath: If True, use VisualizePath for visualization (requires vispath)
            
        Returns:
            Path to generated HTML file
        """
        import json
        
        if similarity_matrix.empty:
            return None
        
        # Apply hierarchical clustering if requested
        if cluster and len(similarity_matrix) > 2:
            try:
                from scipy.cluster.hierarchy import linkage, leaves_list
                from scipy.spatial.distance import squareform
                
                # Convert similarity to distance
                # Handle NaN values by replacing with 0
                sim_vals = similarity_matrix.fillna(0).values
                
                # Ensure symmetry
                sim_vals = (sim_vals + sim_vals.T) / 2
                np.fill_diagonal(sim_vals, 1.0)
                
                # Convert to distance (1 - similarity)
                dist_matrix = 1 - sim_vals
                dist_matrix = np.clip(dist_matrix, 0, 2)  # Ensure valid range
                np.fill_diagonal(dist_matrix, 0)
                
                # Compute linkage
                condensed = squareform(dist_matrix)
                Z = linkage(condensed, method='average')
                order = leaves_list(Z)
                
                # Reorder matrix
                labels = list(similarity_matrix.index)
                new_labels = [labels[i] for i in order]
                similarity_matrix = similarity_matrix.loc[new_labels, new_labels]
            except Exception as e:
                # Clustering failed, use original order
                pass
        
        labels = list(similarity_matrix.index)
        values = similarity_matrix.fillna(0).values.tolist()
        
        # Generate interactive HTML heatmap
        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
{_inline_plotly()}
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a237e;
            margin-bottom: 20px;
        }}
        #heatmap {{
            width: 100%;
            height: 800px;
        }}
        .info {{
            color: #666;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="info">
            {'Hierarchically clustered to show similar types together. ' if cluster else ''}
            Hover over cells for details. Click and drag to zoom.
        </p>
        <div id="heatmap"></div>
    </div>
    <script>
        const labels = {json.dumps(labels)};
        const values = {json.dumps(values)};
        
        const annotations = values.flatMap((row, i) => 
            row.map((val, j) => ({{
                x: labels[j],
                y: labels[i],
                text: (val !== null && !isNaN(val)) ? val.toFixed(2) : 'N/A',
                showarrow: false,
                font: {{
                    color: val > 0.5 ? 'white' : 'black',
                    size: Math.min(12, 300 / labels.length)
                }}
            }}))
        );
        
        const trace = {{
            z: values,
            x: labels,
            y: labels,
            type: 'heatmap',
            colorscale: [
                [0, '#f8f9fa'],
                [0.25, '#a8dadc'],
                [0.5, '#457b9d'],
                [0.75, '#2a6f97'],
                [1, '#1d3557']
            ],
            zmin: -1,
            zmax: 1,
            colorbar: {{
                title: 'Similarity',
                titleside: 'right'
            }},
            hovertemplate: '<b>%{{y}} vs %{{x}}</b><br>Similarity: %{{z:.3f}}<extra></extra>'
        }};
        
        const layout = {{
            margin: {{ l: 150, r: 50, t: 50, b: 150 }},
            xaxis: {{
                tickangle: -45,
                side: 'bottom',
                scaleanchor: 'y',
                constrain: 'domain'
            }},
            yaxis: {{
                autorange: 'reversed',
                constrain: 'domain'
            }},
            annotations: annotations
        }};
        
        Plotly.newPlot('heatmap', [trace], layout, {{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        }});
    </script>
</body>
</html>'''
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_file)
    
    @staticmethod
    def generate_all_inter_type_heatmaps(
        profiles_by_dataset: Dict[str, Dict[str, 'ConnectivityProfile']],
        output_dir: str,
        metric: str = 'rank',
        direction: str = 'both',
        cluster: bool = True
    ) -> Dict[str, str]:
        """
        Generate all inter-type similarity heatmaps (intra and cross-dataset).
        
        Args:
            profiles_by_dataset: Dict of {dataset: {type_name: ConnectivityProfile}}
            output_dir: Directory to save output files
            metric: Similarity metric ('rank', 'jaccard', 'cosine')
            direction: 'upstream', 'downstream', or 'both'
            cluster: Whether to apply hierarchical clustering
            
        Returns:
            Dict mapping name to file path
        """
        from .profile_comparator import ProfileComparator
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Compute inter-type matrices
        results = ProfileComparator.compute_inter_type_rank_correlation_matrix(
            profiles_by_dataset, metric=metric, direction=direction
        )
        
        # 1. Intra-dataset heatmaps (one per dataset)
        for dataset, matrix in results.get('intra_dataset', {}).items():
            if matrix is not None and not matrix.empty:
                safe_ds_name = dataset.replace('/', '_').replace('\\', '_')
                output_file = output_path / f'inter_type_similarity_{safe_ds_name}.html'
                path = ProfileVisualizer.generate_inter_type_similarity_heatmap(
                    matrix,
                    str(output_file),
                    title=f'Inter-Type Similarity - {dataset}',
                    cluster=cluster
                )
                if path:
                    saved_files[f'intra_{safe_ds_name}'] = path
        
        # 2. All types combined matrix (cross-dataset)
        all_types_matrix = results.get('all_types_matrix')
        if all_types_matrix is not None and not all_types_matrix.empty:
            output_file = output_path / 'inter_type_similarity_all.html'
            path = ProfileVisualizer.generate_inter_type_similarity_heatmap(
                all_types_matrix,
                str(output_file),
                title='Inter-Type Similarity Across All Datasets',
                cluster=cluster
            )
            if path:
                saved_files['all_types'] = path
        
        # 3. Cross-dataset summary CSV
        cross_dataset_df = results.get('cross_dataset')
        if cross_dataset_df is not None and not cross_dataset_df.empty:
            csv_path = output_path / 'inter_type_cross_dataset.csv'
            cross_dataset_df.to_csv(csv_path, index=False)
            saved_files['cross_dataset_csv'] = str(csv_path)
        
        return saved_files

    @staticmethod
    def _generate_plotly_heatmap_div(
        df: pd.DataFrame, 
        title: str = '', 
        colorscale: str = 'Viridis',
        height: int = 800
    ) -> str:
        """Generate Plotly heatmap HTML div with hierarchical clustering."""
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot
            
            # Clustering
            row_order = df.index.tolist()
            col_order = df.columns.tolist()
            df_ordered = df
            
            try:
                from scipy.cluster.hierarchy import linkage, leaves_list
                from scipy.spatial.distance import pdist
                
                # Only cluster if matrix is large enough and has no NaNs
                if df.shape[0] > 2 and df.shape[1] > 2 and not df.isna().any().any():
                    # Cluster rows
                    row_dist = pdist(df.values, metric='euclidean')
                    row_linkage = linkage(row_dist, method='ward')
                    row_idx = leaves_list(row_linkage)
                    row_order = [df.index[i] for i in row_idx]
                    
                    # Cluster cols
                    col_dist = pdist(df.values.T, metric='euclidean')
                    col_linkage = linkage(col_dist, method='ward')
                    col_idx = leaves_list(col_linkage)
                    col_order = [df.columns[i] for i in col_idx]
                    
                    # Reorder
                    df_ordered = df.loc[row_order, col_order]
            except ImportError:
                pass
            except Exception as e:
                print(f"Clustering failed: {e}")
                df_ordered = df

            # Create figure
            fig = go.Figure(data=go.Heatmap(
                z=df_ordered.values,
                x=df_ordered.columns,
                y=df_ordered.index,
                colorscale=colorscale,
                colorbar=dict(title='Similarity'),
                hovertemplate='Row: %{y}<br>Col: %{x}<br>Value: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=title,
                height=height,
                xaxis=dict(tickangle=-45, side='bottom'),
                yaxis=dict(autorange='reversed', side='left'),
                margin=dict(b=150, l=150, r=50, t=80),
                font=dict(family="Segoe UI, sans-serif"),
                plot_bgcolor='white'
            )
            
            # Return div
            return plot(fig, output_type='div', include_plotlyjs='cdn')
            
        except ImportError:
            return f"<div class='status-error'>Plotly not installed. Cannot generate interactive heatmap.</div>"
        except Exception as e:
            return f"<div class='status-error'>Error generating heatmap: {str(e)}</div>"

    @staticmethod
    def generate_html_report(
        verification_results: Dict[str, pd.DataFrame],
        profiles: Optional[Dict[str, Dict[str, ConnectivityProfile]]] = None,
        similarity_matrix: Optional[pd.DataFrame] = None,
        metric_matrices: Optional[Dict[str, pd.DataFrame]] = None,
        output_path: str = 'verification_report.html',
        title: str = 'Connectivity Profile Verification Report',
        profile_comparison_url: Optional[str] = None,
        main_report_url: Optional[str] = None,
        inter_type_heatmap_files: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate comprehensive interactive HTML report.
        
        Args:
            verification_results: Dict with 'summary', 'source', 'target', 'intermediate'
            profiles: Optional nested dict of profiles for detailed views
            similarity_matrix: Optional similarity matrix DataFrame (rank correlation)
            metric_matrices: Optional dict of metric-specific matrices (jaccard, cosine, rank)
            output_path: Path to save HTML file
            title: Report title
            profile_comparison_url: Optional URL/path to connectivity profile comparison HTML
            main_report_url: Optional URL/path to main comparison report HTML
            inter_type_heatmap_files: Optional dict of inter-type heatmap file paths
        
        Returns:
            Path to generated HTML file
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Start building HTML
        html_parts = [f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
{_inline_plotly()}
    <style>
        :root {{
            --primary-color: #1a237e;
            --secondary-color: #3949ab;
            --success-color: #2e7d32;
            --warning-color: #ff9800;
            --danger-color: #f44336;
            --light-bg: #f8f9fa;
            --border-color: #e9ecef;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        
        header {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border-left: 5px solid var(--primary-color);
        }}
        
        h1 {{
            margin: 0 0 5px 0;
            font-size: 1.8em;
            color: var(--primary-color);
        }}
        
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            transition: transform 0.2s;
        }}
        
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        .card h3 {{
            margin: 0 0 10px 0;
            font-size: 0.85em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .card .value {{
            font-size: 2.2em;
            font-weight: 600;
            color: var(--primary-color);
        }}
        
        .section {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .section h2 {{
            margin-top: 0;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--border-color);
            color: #444;
            font-size: 1.4em;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 0.95em;
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background: var(--light-bg);
            font-weight: 600;
            color: #555;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .heatmap-container {{
            width: 100%;
            min-height: 600px;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            margin-top: 20px;
        }}
        
        .nav-link {{
            display: inline-block;
            margin-top: 10px;
            color: var(--secondary-color);
            text-decoration: none;
            font-weight: 500;
        }}
        
        .nav-link:hover {{
            text-decoration: underline;
        }}
        
        footer {{
            text-align: center;
            padding: 20px;
            color: #888;
            font-size: 0.9em;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <div class="timestamp">Generated: {timestamp}</div>
            {'<a href="' + main_report_url + '" class="nav-link">← Back to Main Comparison Report</a>' if main_report_url else ''}
        </header>
''']
        
        # Summary cards
        summary_df = verification_results.get('summary', pd.DataFrame())
        if not summary_df.empty:
            total_types = len(summary_df)
            # Use rank_corr as the primary score
            avg_rank_corr = summary_df['avg_rank_corr'].mean() if 'avg_rank_corr' in summary_df.columns else np.nan
            verified = (summary_df['verification_status'] == 'verified').sum() if 'verification_status' in summary_df.columns else np.nan
            
            html_parts.append(f'''
        <div class="summary-cards">
            <div class="card">
                <h3>Total Types Verified</h3>
                <div class="value">{total_types}</div>
            </div>
            <div class="card">
                <h3>Verified</h3>
                <div class="value" style="color: var(--success-color);">{verified if not pd.isna(verified) else 'N/A'}</div>
            </div>
            <div class="card">
                <h3>Avg Rank Correlation</h3>
                <div class="value">{avg_rank_corr:.3f}</div>
            </div>
        </div>
            ''')

        # Similarity Heatmap (Interactive)
        if similarity_matrix is not None and not similarity_matrix.empty:
            heatmap_div = ProfileVisualizer._generate_plotly_heatmap_div(
                similarity_matrix, 
                title='Pairwise Similarity (Rank Correlation)',
                colorscale='Viridis'
            )
            
            html_parts.append(f'''
        <div class="section">
            <h2>Similarity Matrix</h2>
            <p>Pairwise rank correlation between connectivity profiles. Rows/columns are clustered by similarity.</p>
            <div class="heatmap-container">
                {heatmap_div}
            </div>
        </div>
            ''')

        
        # Add link to connectivity profile comparison if available
        if profile_comparison_url:
            html_parts.append(f'''
        <div class="section" style="background: linear-gradient(135deg, #e8f5e9, #f1f8e9); padding: 15px 25px;">
            <h3 style="margin: 0 0 8px 0; color: #2e7d32;">🔗 Related Reports</h3>
            <p style="margin: 0;">
                <a href="{profile_comparison_url}" style="color: #1565c0; font-weight: 500; text-decoration: none; font-size: 1.1em;">
                    📊 View Detailed Connectivity Profile Comparison →
                </a>
                <br><span style="color: #666; font-size: 0.9em;">Interactive bar charts comparing upstream/downstream partners for each neuron type</span>
            </p>
        </div>
''')
        
        # Summary table
        if not summary_df.empty:
            # Filter to keep only types appearing in at least 2 datasets (not all NaN)
            # and sort by summed rank_corr within role groups
            summary_df = summary_df.copy()
            
            # Keep only rows where avg_rank_corr is not NaN (type found in at least 2 datasets)
            if 'avg_rank_corr' in summary_df.columns:
                summary_df = summary_df[summary_df['avg_rank_corr'].notna()].copy()
            
            # Sort: group by role (source, target, intermediate), then by datasets_found, then by rank_corr
            role_order = {'source': 0, 'source/target': 0.5, 'target': 1, 'intermediate': 2}
            if 'role' in summary_df.columns:
                summary_df['role_order'] = summary_df['role'].apply(
                    lambda x: min([role_order.get(r.strip(), 3) for r in str(x).split('/')]) if pd.notna(x) else 3
                )
                sort_cols = ['role_order']
                ascending = [True]
                if 'datasets_found' in summary_df.columns:
                    sort_cols.append('datasets_found')
                    ascending.append(False)
                if 'avg_rank_corr' in summary_df.columns:
                    sort_cols.append('avg_rank_corr')
                    ascending.append(False)
                summary_df = summary_df.sort_values(sort_cols, ascending=ascending).drop(columns=['role_order'])
            
            # Check for directional columns (include in main table)
            has_directional = any(col in summary_df.columns for col in ['avg_rank_corr_upstream', 'avg_rank_corr_downstream'])
            has_directional_jaccard = any(col in summary_df.columns for col in ['avg_jaccard_upstream', 'avg_jaccard_downstream'])
            
            html_parts.append(f'''
        <div class="section">
            <h2>📊 Verification Summary</h2>
            <p style="margin-bottom: 15px; color: #666;">
                Only showing neuron types found in at least 2 datasets. 
                <strong>Rank Correlation</strong> is used as the primary similarity metric (average of upstream and downstream).
                <br>Sorted by: Role → Datasets Found → Rank Corr.
                <br><strong>Jaccard</strong> = partner set overlap (|A ∩ B| / |A ∪ B|).
            </p>
            <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Neuron Type</th>
                        <th>Role</th>
                        <th>Found In</th>
                        <th>Rank Corr (Both)</th>
                        {"<th>Up↑</th><th>Down↓</th>" if has_directional else ""}
                        {"<th>Jaccard (Both)</th><th>Jaccard (Up)</th><th>Jaccard (Down)</th>" if has_directional_jaccard else "<th>Jaccard</th>"}
                    </tr>
                </thead>
                <tbody>
''')
            
            # Helper function for colored score cells (raw rank correlation, -1..1)
            def score_cell_colored(val):
                """Generate colored cell HTML for a raw rank corr score value [-1,1]."""
                if pd.isna(val) or val == '-':
                    return '<td style="background: #f5f5f5; text-align: center;">-</td>'
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    return f'<td style="text-align: center;">{val}</td>'
                
                # Diverging color centered on 0: positive (green) / negative (red)
                if val >= 0:
                    bg = f'rgba(76, 175, 80, {0.25 + 0.5 * min(val, 1.0):.3f})'
                else:
                    bg = f'rgba(244, 67, 54, {0.25 + 0.5 * min(-val, 1.0):.3f})'
                return f'<td style="background: {bg}; text-align: center;">{val:.3f}</td>'
            
            # Helper function for Jaccard colored cells (different thresholds)
            def jaccard_cell_colored(val):
                """Generate colored cell HTML for Jaccard similarity [0,1] with specific thresholds."""
                if pd.isna(val) or val == '-':
                    return '<td style="background: #f5f5f5; text-align: center;">-</td>'
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    return f'<td style="text-align: center;">{val}</td>'
                
                # Jaccard thresholds: >0.5 very high, >0.3 high, >0.2 medium, >0.1 low, <0.1 very low
                if val > 0.5:
                    bg = 'rgba(46, 125, 50, 0.7)'  # Dark Green - Very High
                elif val > 0.3:
                    bg = 'rgba(76, 175, 80, 0.6)'  # Green - High
                elif val > 0.2:
                    bg = 'rgba(255, 193, 7, 0.6)'  # Yellow - Medium
                elif val > 0.1:
                    bg = 'rgba(255, 152, 0, 0.6)'  # Orange - Low
                else:
                    bg = 'rgba(244, 67, 54, 0.5)'  # Red - Very Low
                return f'<td style="background: {bg}; text-align: center;">{val:.3f}</td>'
            
            for _, row in summary_df.iterrows():
                role = row.get('role', '')
                status = row.get('verification_status', '')
                
                # Get individual metric scores (raw rank correlation)
                rank_corr = row.get('rank_correlation', row.get('avg_rank_corr', np.nan))
                jaccard = row.get('jaccard', row.get('avg_jaccard', np.nan))
                cosine = row.get('cosine', row.get('avg_cosine', np.nan))
                
                # Directional scores
                upstream = row.get('avg_rank_corr_upstream', np.nan)
                downstream = row.get('avg_rank_corr_downstream', np.nan)
                
                # Datasets found info
                datasets_found = row.get('datasets_found', '?')
                total_datasets = row.get('total_datasets', '?')
                found_str = f"{datasets_found}/{total_datasets}"
                
                role_class = f"role-{role.replace('/', '-')}" if role else ""
                
                # Score bar color - use raw rank_corr as primary score (range -1..1)
                score = rank_corr if isinstance(rank_corr, (int, float)) and not pd.isna(rank_corr) else np.nan
                if pd.isna(score):
                    bar_color = '#ccc'
                    score_display = 'N/A'
                    bar_width = 0
                elif score >= 0:
                    bar_color = 'var(--high-color, #2e7d32)'
                    score_display = f'{score:.3f}'
                    bar_width = min(abs(score), 1.0) * 100
                else:
                    bar_color = 'var(--low-color, #c62828)'
                    score_display = f'{score:.3f}'
                    bar_width = min(abs(score), 1.0) * 100
                
                # Use normalized values for colored cells
                directional_cols = f"{score_cell_colored(upstream)}{score_cell_colored(downstream)}" if has_directional else ""
                
                # Jaccard columns - directional and combined
                # Use avg_jaccard_both for proper set-based Jaccard on combined up+down partners
                jaccard_both = row.get('avg_jaccard_both', row.get('avg_jaccard', np.nan))
                jaccard_up = row.get('avg_jaccard_upstream', np.nan)
                jaccard_down = row.get('avg_jaccard_downstream', np.nan)
                if has_directional_jaccard:
                    jaccard_cols = f"{jaccard_cell_colored(jaccard_both)}{jaccard_cell_colored(jaccard_up)}{jaccard_cell_colored(jaccard_down)}"
                else:
                    jaccard_cols = jaccard_cell_colored(jaccard_both)
                
                html_parts.append(f'''
                    <tr>
                        <td><strong>{row['neuron_type']}</strong></td>
                        <td><span class="role-tag {role_class}">{role}</span></td>
                        <td style="text-align: center;">{found_str}</td>
                        <td>
                            <div class="score-bar">
                                <div class="score-bar-fill" style="width: {bar_width}%; background: {bar_color};"></div>
                            </div>
                            <span>{score_display}</span>
                        </td>
                        {directional_cols}
                        {jaccard_cols}
                    </tr>
''')
            
            html_parts.append('''
                </tbody>
            </table>
            </div>
        </div>
''')
        

        
        # Jaccard similarity matrix (if metric_matrices available)
        jaccard_matrix = None
        if metric_matrices and 'jaccard' in metric_matrices:
            jaccard_matrix = metric_matrices['jaccard']
        
        if jaccard_matrix is not None and not jaccard_matrix.empty:
            heatmap_div = ProfileVisualizer._generate_plotly_heatmap_div(
                jaccard_matrix, 
                title='Jaccard Similarity (Partner Overlap)',
                colorscale='Blues'
            )
            
            html_parts.append(f'''
        <div class="section">
            <h2>Jaccard Similarity Matrix</h2>
            <p>Jaccard similarity scores for each neuron type across dataset pairs. Measures partner set overlap (intersection/union).</p>
            <div class="heatmap-container">
                {heatmap_div}
            </div>
        </div>
            ''')
            
        # Cosine similarity matrix (if metric_matrices available)
        cosine_matrix = None
        if metric_matrices and 'cosine' in metric_matrices:
            cosine_matrix = metric_matrices['cosine']
        
        if cosine_matrix is not None and not cosine_matrix.empty:
            heatmap_div = ProfileVisualizer._generate_plotly_heatmap_div(
                cosine_matrix, 
                title='Cosine Similarity (Vector Alignment)',
                colorscale='Greens'
            )
            
            html_parts.append(f'''
        <div class="section">
            <h2>Cosine Similarity Matrix</h2>
            <p>Cosine similarity scores for each neuron type across dataset pairs. Measures orientation of connectivity vectors.</p>
            <div class="heatmap-container">
                {heatmap_div}
            </div>
        </div>
            ''')

            # Dataset pair similarity summary - average scores across all types for each dataset pair
            # Include both rank correlation and Jaccard
            
            # Re-derive matrix_filtered for the summary table since we removed the static table generation
            matrix_filtered = pd.DataFrame()
            if similarity_matrix is not None and not similarity_matrix.empty:
                 matrix_filtered = similarity_matrix.dropna(how='all').copy()
            
            if not matrix_filtered.empty:
                # Get Jaccard matrix for avg Jaccard calculation
                jaccard_matrix_for_summary = None
                if metric_matrices and 'jaccard' in metric_matrices:
                    jaccard_matrix_for_summary = metric_matrices['jaccard']
                
                html_parts.append('''
        <div class="section">
            <h2>📈 Dataset Pair Similarity Summary</h2>
            <p>Average scores across all neuron types for each dataset pair. 
               Higher scores indicate better overall similarity between datasets.</p>
            <table>
                <thead>
                    <tr>
                        <th>Dataset Pair</th>
                        <th>Avg Rank Corr</th>
                        <th>Avg Jaccard</th>
                        <th>N Types</th>
                        <th>Pos (>&gt;0)</th>
                        <th>Zero (=0)</th>
                        <th>Neg (&lt;0)</th>
                    </tr>
                </thead>
                <tbody>
''')
                for col in matrix_filtered.columns:
                    col_values = matrix_filtered[col].dropna()
                    if len(col_values) > 0:
                        avg_score = col_values.mean()
                        n_types = len(col_values)
                        n_positive = (col_values > 0).sum()
                        n_zero = (col_values == 0).sum()
                        n_negative = (col_values < 0).sum()
                        
                        # Get avg Jaccard for this dataset pair
                        avg_jaccard = np.nan
                        if jaccard_matrix_for_summary is not None and col in jaccard_matrix_for_summary.columns:
                            jaccard_col_values = jaccard_matrix_for_summary[col].dropna()
                            if len(jaccard_col_values) > 0:
                                avg_jaccard = jaccard_col_values.mean()
                        
                        # Color for avg score - diverging on raw rank correlation
                        if avg_score >= 0:
                            score_color = f'rgba(76, 175, 80, {0.25 + 0.5 * min(avg_score, 1.0):.3f})'
                        else:
                            score_color = f'rgba(244, 67, 54, {0.25 + 0.5 * min(-avg_score, 1.0):.3f})'
                        
                        # Color for Jaccard (different thresholds)
                        if np.isnan(avg_jaccard):
                            jaccard_color = '#f5f5f5'
                            jaccard_display = '-'
                        elif avg_jaccard > 0.5:
                            jaccard_color = 'rgba(46, 125, 50, 0.7)'
                            jaccard_display = f'{avg_jaccard:.3f}'
                        elif avg_jaccard > 0.3:
                            jaccard_color = 'rgba(76, 175, 80, 0.7)'
                            jaccard_display = f'{avg_jaccard:.3f}'
                        elif avg_jaccard > 0.2:
                            jaccard_color = 'rgba(255, 193, 7, 0.7)'
                            jaccard_display = f'{avg_jaccard:.3f}'
                        elif avg_jaccard > 0.1:
                            jaccard_color = 'rgba(255, 152, 0, 0.7)'
                            jaccard_display = f'{avg_jaccard:.3f}'
                        else:
                            jaccard_color = 'rgba(244, 67, 54, 0.5)'
                            jaccard_display = f'{avg_jaccard:.3f}'
                        
                        html_parts.append(f'''                    <tr>
                        <td><strong>{col}</strong></td>
                        <td style="background: {score_color}; text-align: center;">{avg_score:.3f}</td>
                        <td style="background: {jaccard_color}; text-align: center;">{jaccard_display}</td>
                        <td style="text-align: center;">{n_types}</td>
                        <td style="text-align: center; color: #2e7d32;">{n_positive}</td>
                        <td style="text-align: center; color: #888;">{n_zero}</td>
                        <td style="text-align: center; color: #c62828;">{n_negative}</td>
                    </tr>
''')
                
                html_parts.append('''                </tbody>
            </table>
        </div>
''')
        
        # Interpretation guide
        html_parts.append('''
        <div class="section">
            <h2>📖 Interpretation Guide</h2>
            <h3>Score Metrics</h3>
            <ul>
                <li><strong>Rank Correlation (Primary):</strong> Average of upstream and downstream Spearman correlations (raw, -1 to 1). Measures whether the same partners are ranked similarly in both datasets; 1 = identical ranking, 0 = no monotonic relation, -1 = inverse ranking.</li>
                <li><strong>Jaccard:</strong> Set-based overlap of partner types: |A ∩ B| / |A ∪ B|. Range: 0 (no shared partners) to 1 (identical partner sets). Average of upstream and downstream directions. Ignores connection weights.</li>
            </ul>
            
            <h3>Notes</h3>
            <ul>
                <li>Only neuron types appearing in at least 2 datasets are shown (NaN types are filtered out).</li>
                <li>Types are sorted by: Role → Datasets Found → Rank Correlation (all descending within group).</li>
                <li><strong>Gray/Error</strong> indicates type was not found in enough datasets for meaningful comparison.</li>
            </ul>
        </div>
''')
        
        # Inter-type comparison section (if heatmaps available)
        if inter_type_heatmap_files:
            html_parts.append('''
        <div class="section">
            <h2>🔗 Inter-Type Comparison Heatmaps</h2>
            <p style="color: #666; margin-bottom: 15px;">
                Compare connectivity profiles between different neuron types. 
                Heatmaps show similarity scores between types, with hierarchical clustering 
                applied to group similar types together.
            </p>
            <div style="display: flex; flex-wrap: wrap; gap: 15px;">
''')
            for name, path in inter_type_heatmap_files.items():
                # Get relative path for linking
                relative_path = Path(path).name if '/' in str(path) else path
                display_name = name.replace('_', ' ').replace('intra ', '').title()
                if 'all_types' in name:
                    display_name = '🌐 All Types Combined'
                elif 'csv' in name:
                    display_name = '📊 Cross-Dataset CSV'
                else:
                    display_name = f'📈 {display_name}'
                
                html_parts.append(f'''
                <a href="connectivity_profile_verification/visualizations/inter_type_heatmaps/{relative_path}" 
                   style="display: inline-block; padding: 12px 20px; background: linear-gradient(135deg, #667eea, #764ba2); 
                          color: white; text-decoration: none; border-radius: 8px; font-weight: 500;
                          box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: transform 0.2s, box-shadow 0.2s;">
                    {display_name}
                </a>
''')
            html_parts.append('''
            </div>
        </div>
''')
        
        html_parts.append('''
        <footer>
            <p>Generated by Connectivity Profile Verification Module</p>
            <p>drocat</p>
        </footer>
    </div>
</body>
</html>
''')
        
        # Write HTML file
        html_content = ''.join(html_parts)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_file)
    
    @staticmethod
    def save_all_visualizations(
        verification_results: Dict[str, pd.DataFrame],
        similarity_matrix: Optional[pd.DataFrame] = None,
        profiles: Optional[Dict[str, Dict[str, ConnectivityProfile]]] = None,
        output_dir: str = 'profile_visualizations',
        metric_matrices: Optional[Dict[str, pd.DataFrame]] = None,
        directional_matrices: Optional[Dict[str, pd.DataFrame]] = None,
        profiles_by_dataset: Optional[Dict[str, Dict[str, ConnectivityProfile]]] = None,
        generate_inter_type_heatmaps: bool = True
    ) -> Dict[str, str]:
        """
        Save all visualizations to a directory.
        
        Round 4 Enhanced: Supports individual metric and directional heatmaps.
        Round 7: Added inter-type similarity heatmaps.
        
        Args:
            verification_results: Dict with verification DataFrames
            similarity_matrix: Optional combined similarity matrix
            profiles: Optional profile dict for bar charts
            output_dir: Output directory
            metric_matrices: Optional dict of metric-specific matrices (jaccard, cosine, rank)
            directional_matrices: Optional dict of directional matrices (upstream, downstream, both)
            profiles_by_dataset: Optional dict of {dataset: {type: profile}} for inter-type analysis
            generate_inter_type_heatmaps: Whether to generate inter-type similarity heatmaps
        
        Returns:
            Dict mapping visualization name to file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        try:
            import matplotlib.pyplot as plt
            
            # 1. Verification summary bar chart
            summary = verification_results.get('summary', pd.DataFrame())
            if not summary.empty:
                fig = ProfileVisualizer.plot_verification_summary(
                    summary, 
                    output_path=str(output_path / 'verification_summary.png')
                )
                plt.close(fig)
                saved_files['summary'] = str(output_path / 'verification_summary.png')
            
            # 2. Role comparison chart
            if not summary.empty and 'role' in summary.columns:
                fig = ProfileVisualizer.plot_role_comparison(
                    summary,
                    output_path=str(output_path / 'role_comparison.png')
                )
                plt.close(fig)
                saved_files['role_comparison'] = str(output_path / 'role_comparison.png')
            
            # 3. Overall (both directions) similarity heatmap
            if similarity_matrix is not None and not similarity_matrix.empty:
                fig = ProfileVisualizer.plot_similarity_heatmap(
                    similarity_matrix,
                    title='Connectivity Profile Similarity (Overall)',
                    output_path=str(output_path / 'similarity_heatmap_overall.png')
                )
                plt.close(fig)
                saved_files['heatmap_overall'] = str(output_path / 'similarity_heatmap_overall.png')
            
            # 4. Round 4: Individual metric heatmaps
            if metric_matrices:
                metric_dir = output_path / 'metric_heatmaps'
                metric_dir.mkdir(exist_ok=True)
                
                figs = ProfileVisualizer.plot_multi_metric_heatmaps(
                    metric_matrices,
                    output_dir=str(metric_dir)
                )
                for metric_name in metric_matrices.keys():
                    saved_files[f'heatmap_{metric_name}'] = str(
                        metric_dir / f'similarity_heatmap_{metric_name}.png'
                    )
            
            # 5. Round 4: Directional heatmaps (upstream/downstream separate)
            if directional_matrices:
                dir_heatmap_dir = output_path / 'directional_heatmaps'
                dir_heatmap_dir.mkdir(exist_ok=True)
                
                figs = ProfileVisualizer.plot_directional_heatmaps(
                    directional_matrices,
                    output_dir=str(dir_heatmap_dir)
                )
                for dir_name in directional_matrices.keys():
                    saved_files[f'heatmap_{dir_name}'] = str(
                        dir_heatmap_dir / f'similarity_heatmap_{dir_name}.png'
                    )
            
            # 6. Profile comparison charts (for top types)
            if profiles:
                profile_dir = output_path / 'profile_charts'
                profile_dir.mkdir(exist_ok=True)
                
                for neuron_type, dataset_profiles in list(profiles.items())[:10]:
                    try:
                        fig = ProfileVisualizer.plot_profile_comparison(
                            dataset_profiles,
                            neuron_type,
                            output_path=str(profile_dir / f'{neuron_type}_profile.png')
                        )
                        plt.close(fig)
                        saved_files[f'profile_{neuron_type}'] = str(
                            profile_dir / f'{neuron_type}_profile.png'
                        )
                    except Exception:
                        pass
            
            # 7. Inter-type similarity heatmaps (Round 7: new feature)
            if generate_inter_type_heatmaps and profiles_by_dataset:
                inter_type_dir = output_path / 'inter_type_heatmaps'
                inter_type_dir.mkdir(exist_ok=True)
                
                inter_type_files = ProfileVisualizer.generate_all_inter_type_heatmaps(
                    profiles_by_dataset,
                    str(inter_type_dir),
                    metric='rank',
                    direction='both',
                    cluster=True
                )
                saved_files.update(inter_type_files)
            else:
                inter_type_files = {}
            
        except ImportError:
            print("Warning: matplotlib not available for visualizations")
            inter_type_files = {}
        
        # 8. HTML report - save to parent directory as connectivity_profile_comparison.html
        # Output path is like .../connectivity_profile_verification/visualizations
        # We want to save to .../connectivity_profile_comparison.html (parent of parent)
        parent_dir = output_path.parent.parent if output_path.parent.name == 'visualizations' else output_path.parent
        html_filename = 'connectivity_profile_comparison.html'
        html_full_path = parent_dir / html_filename
        
        # Link back to main comparison report
        main_report_url = 'comparison_report.html'
        
        html_path = ProfileVisualizer.generate_html_report(
            verification_results,
            profiles=profiles,
            similarity_matrix=similarity_matrix,
            output_path=str(html_full_path),
            main_report_url=main_report_url,
            inter_type_heatmap_files=inter_type_files if inter_type_files else None
        )
        saved_files['html_report'] = html_path
        
        return saved_files
