"""
DataLoader - I/O utilities for comparison data.

This module provides utilities for loading and saving comparison analysis data,
including FindNeuronConnection outputs and comparison results.
"""

import os
import json
import glob
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from ..utils.naming_utils import canonical_dataset_name
except ImportError:  # pragma: no cover - direct package imports
    from utils.naming_utils import canonical_dataset_name


class DataLoader:
    """
    I/O utilities for loading and saving comparison data.
    
    Handles:
    - Loading FindNeuronConnection outputs from structured folders
    - Saving comparison results to CSV/JSON
    - Managing parameters.json
    - Loading/saving intermediate results
    
    Example:
        >>> loader = DataLoader('/path/to/comparison_results')
        >>> loader.save_parameters(params.to_dict(), configs, label_mapper)
        >>> results = loader.load_dataset_results('hemibrain_v1_2_1', threshold=5)
    """
    
    def __init__(self, base_path: str):
        """
        Initialize DataLoader.
        
        Args:
            base_path: Base path for comparison results
        """
        self.base_path = os.path.abspath(base_path)
        self.dataset_data_path = os.path.join(self.base_path, 'dataset_data')
        self.comparison_results_path = os.path.join(self.base_path, 'comparison_results')
        # Note: visualizations moved to comparison_visualizations at base level
        self.visualizations_path = os.path.join(self.base_path, 'comparison_visualizations')
    
    def ensure_directories(self) -> None:
        """Create all necessary directories."""
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.dataset_data_path, exist_ok=True)
        os.makedirs(self.comparison_results_path, exist_ok=True)
        # comparison_visualizations is created by _generate_visualizations, not here
    
    # =========================================================================
    # Parameters I/O
    # =========================================================================
    
    def save_parameters(
        self,
        params_dict: Dict,
        configs: List[Any],
        label_mapper: Optional[Any] = None
    ) -> str:
        """
        Save complete parameters to parameters.json.
        
        Args:
            params_dict: ComparisonParameters as dictionary
            configs: List of DatasetConfig objects
            label_mapper: Optional LabelMapper instance
            
        Returns:
            Path to saved file
        """
        # Add source/target groups from configs
        params_dict['source_groups'] = {}
        params_dict['target_groups'] = {}
        
        for config in configs:
            params_dict['source_groups'][config.dataset] = config.source_neurons
            params_dict['target_groups'][config.dataset] = config.target_neurons
        
        # Add label mapping if provided
        if label_mapper and label_mapper.has_mapping():
            params_dict['label_mapping'] = label_mapper.export_to_parameters()
        
        # Save
        filepath = os.path.join(self.base_path, 'parameters.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(params_dict, f, indent=2, default=str)
        
        return filepath
    
    def load_parameters(self) -> Dict:
        """
        Load parameters from parameters.json.
        
        Returns:
            Dictionary with all parameters
        """
        filepath = os.path.join(self.base_path, 'parameters.json')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Parameters file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    # =========================================================================
    # Dataset Results I/O
    # =========================================================================
    
    def get_dataset_path(self, dataset: str, threshold: int) -> str:
        """
        Get path to dataset output folder.
        
        Args:
            dataset: Dataset identifier
            threshold: Synapse threshold
            
        Returns:
            Path to dataset/threshold folder
        """
        safe_name = self._sanitize_name(dataset)
        return os.path.join(self.dataset_data_path, safe_name, f'minsyn_{threshold}')
    
    def dataset_results_exist(self, dataset: str, threshold: int) -> bool:
        """
        Check if results exist for a dataset/threshold combination.
        
        Args:
            dataset: Dataset identifier
            threshold: Synapse threshold
            
        Returns:
            True if results exist
        """
        path = self.get_dataset_path(dataset, threshold)
        if not os.path.exists(path):
            return False
        
        # Check for at least one output file
        csv_files = glob.glob(os.path.join(path, '*.csv'))
        return len(csv_files) > 0
    
    def load_dataset_results(
        self,
        dataset: str,
        threshold: int,
        file_type: str = 'paths'
    ) -> pd.DataFrame:
        """
        Load results for a specific dataset and threshold.
        
        Args:
            dataset: Dataset identifier
            threshold: Synapse threshold
            file_type: Type of file to load ('paths', 'edges', 'summary')
            
        Returns:
            DataFrame with results
        """
        path = self.get_dataset_path(dataset, threshold)
        
        if not os.path.exists(path):
            return pd.DataFrame()
        
        # Find matching file
        pattern = f'*{file_type}*.csv'
        matches = glob.glob(os.path.join(path, pattern))
        
        if not matches:
            # Try without pattern (load any CSV)
            matches = glob.glob(os.path.join(path, '*.csv'))
            if matches:
                # Take the first one that matches file_type
                for m in matches:
                    if file_type in os.path.basename(m).lower():
                        matches = [m]
                        break
        
        if not matches:
            return pd.DataFrame()
        
        # Load the first matching file
        return pd.read_csv(matches[0])
    
    def load_all_dataset_results(
        self,
        datasets: List[str],
        thresholds: List[int],
        file_type: str = 'paths'
    ) -> Dict[str, Dict[int, pd.DataFrame]]:
        """
        Load results for all datasets and thresholds.
        
        Args:
            datasets: List of dataset identifiers
            thresholds: List of synapse thresholds
            file_type: Type of file to load
            
        Returns:
            Nested dict: {dataset: {threshold: DataFrame}}
        """
        results = {}
        
        for dataset in datasets:
            results[dataset] = {}
            for threshold in thresholds:
                df = self.load_dataset_results(dataset, threshold, file_type)
                if not df.empty:
                    results[dataset][threshold] = df
        
        return results
    
    def save_dataset_result(
        self,
        df: pd.DataFrame,
        dataset: str,
        threshold: int,
        filename: str
    ) -> str:
        """
        Save a dataset result to the appropriate folder.
        
        Args:
            df: DataFrame to save
            dataset: Dataset identifier
            threshold: Synapse threshold
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        path = self.get_dataset_path(dataset, threshold)
        os.makedirs(path, exist_ok=True)
        
        filepath = os.path.join(path, filename)
        df.to_csv(filepath, index=False)
        
        return filepath
    
    # =========================================================================
    # Comparison Results I/O
    # =========================================================================
    
    def save_comparison_result(
        self,
        df: pd.DataFrame,
        filename: str,
        include_timestamp: bool = False
    ) -> str:
        """
        Save a comparison result to comparison_results folder.
        
        Args:
            df: DataFrame to save
            filename: Output filename (without path)
            include_timestamp: Whether to add timestamp to filename
            
        Returns:
            Path to saved file
        """
        if include_timestamp:
            base, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{base}_{timestamp}{ext}"
        
        filepath = os.path.join(self.comparison_results_path, filename)
        df.to_csv(filepath, index=True)
        
        return filepath
    
    def load_comparison_result(self, filename: str) -> pd.DataFrame:
        """
        Load a comparison result from comparison_results folder.
        
        Args:
            filename: Filename to load
            
        Returns:
            DataFrame with results
        """
        filepath = os.path.join(self.comparison_results_path, filename)
        if not os.path.exists(filepath):
            return pd.DataFrame()
        
        return pd.read_csv(filepath, index_col=0)
    
    def save_metadata_comparison(self, df: pd.DataFrame) -> str:
        """
        Save dataset metadata comparison table.
        
        Args:
            df: Metadata comparison DataFrame
            
        Returns:
            Path to saved file
        """
        filepath = os.path.join(self.base_path, 'dataset_metadata_comparison.csv')
        df.to_csv(filepath, index=True)
        return filepath
    
    def load_metadata_comparison(self) -> pd.DataFrame:
        """
        Load dataset metadata comparison table.
        
        Returns:
            Metadata comparison DataFrame
        """
        filepath = os.path.join(self.base_path, 'dataset_metadata_comparison.csv')
        if not os.path.exists(filepath):
            return pd.DataFrame()
        
        return pd.read_csv(filepath, index_col=0)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _sanitize_name(self, name: str) -> str:
        """Convert name to filesystem-safe format."""
        return canonical_dataset_name(name).replace(':', '_').replace('.', '_').replace('-', '_')
    
    def list_available_datasets(self) -> List[str]:
        """
        List datasets with available results.
        
        Returns:
            List of dataset folder names
        """
        if not os.path.exists(self.dataset_data_path):
            return []
        
        return [d for d in os.listdir(self.dataset_data_path)
                if os.path.isdir(os.path.join(self.dataset_data_path, d))]
    
    def list_available_thresholds(self, dataset: str) -> List[int]:
        """
        List available thresholds for a dataset.
        
        Args:
            dataset: Dataset identifier
            
        Returns:
            List of available thresholds
        """
        safe_name = self._sanitize_name(dataset)
        dataset_path = os.path.join(self.dataset_data_path, safe_name)
        
        if not os.path.exists(dataset_path):
            return []
        
        thresholds = []
        for d in os.listdir(dataset_path):
            # Ratio-basis runs materialize `minratio_{decimal}` folders
            # ('.'→'_', '-'→'neg'); their tiers are floats.
            if d.startswith('minsyn_'):
                try:
                    thresholds.append(int(d.replace('minsyn_', '')))
                except ValueError:
                    pass
            elif d.startswith('minratio_'):
                try:
                    value = float(
                        d.replace('minratio_', '').replace('_', '.')
                        .replace('neg', '-'))
                    thresholds.append(
                        int(value) if value.is_integer() else value)
                except ValueError:
                    pass

        return sorted(thresholds)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of available data.
        
        Returns:
            Dictionary with summary information
        """
        datasets = self.list_available_datasets()
        
        summary = {
            'base_path': self.base_path,
            'datasets_count': len(datasets),
            'datasets': {}
        }
        
        for dataset in datasets:
            thresholds = self.list_available_thresholds(dataset)
            summary['datasets'][dataset] = {
                'thresholds': thresholds,
                'threshold_count': len(thresholds)
            }
        
        return summary
    
    def export_summary_report(self, analysis_results: Dict) -> str:
        """
        Export a summary report in Markdown format.
        
        Args:
            analysis_results: Dictionary with analysis results
            
        Returns:
            Path to saved report
        """
        filepath = os.path.join(self.comparison_results_path, 'summary_report.md')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Cross-Dataset Comparison Summary Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Add datasets section
            f.write("## Datasets Compared\n\n")
            for dataset in analysis_results.get('datasets', []):
                f.write(f"- {dataset}\n")
            f.write("\n")
            
            # Add thresholds section
            f.write("## Thresholds Analyzed\n\n")
            f.write(f"Synapse thresholds: {analysis_results.get('thresholds', [])}\n\n")
            
            # Add key findings if available
            if 'key_findings' in analysis_results:
                f.write("## Key Findings\n\n")
                for finding in analysis_results['key_findings']:
                    f.write(f"- {finding}\n")
                f.write("\n")
            
            # Add output files section
            f.write("## Output Files\n\n")
            f.write("| File | Description |\n")
            f.write("|------|-------------|\n")
            
            # List files in comparison_results folder
            if os.path.exists(self.comparison_results_path):
                for filename in sorted(os.listdir(self.comparison_results_path)):
                    if filename.endswith('.csv'):
                        f.write(f"| `{filename}` | Comparison data |\n")
                    elif filename.endswith('.png'):
                        f.write(f"| `{filename}` | Visualization |\n")
        
        return filepath
