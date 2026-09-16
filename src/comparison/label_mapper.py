"""
LabelMapper - Cross-dataset label standardization for comparison analysis.

This module provides the LabelMapper class for standardizing neuron identifiers
across different datasets with varying naming conventions.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from collections import defaultdict

try:
    from ..utils.naming_utils import canonical_dataset_name
except ImportError:  # pragma: no cover - direct package imports
    from utils.naming_utils import canonical_dataset_name


class LabelMapper:
    """
    Cross-dataset label standardization for comparison analysis.
    
    Handles mapping logic across ALL datasets (not embedded in DatasetConfig).
    Supports CSV and JSON input formats, with asymmetric neuron counts per dataset.
    
    Key Features:
    - Centralized mapping logic for cross-dataset consistency
    - Supports CSV format (one file for sources, one for targets, ALL datasets)
    - Supports JSON format with nested group structure
    - Unknown IDs resolve to themselves (identity fallback), so unmapped
      types keep their raw names through apply_to_dataframe
    - Exports mapping to parameters.json for reproducibility
    - Provides mapping summary for user verification
    
    CSV Format Example:
        std_label,hemibrain_v1_2_1,male-cns_v0_9,flywire_FAFB_v783
        aMe12_grp1,aMe12,aMe12,720575940610453042
        aMe12_grp1,aMe12_R,aMe12-like,720575940610453043
        aMe12_grp1,,aMe12_variant,
        
    JSON Format Example:
        {
            "source_mapping": {
                "custom_label": ["aMe12_grp1", "aMe12_grp2"],
                "hemibrain:v1.2.1": [["aMe12", "aMe12_R"], ["aMe12_L"]]
            },
            "target_mapping": { ... },
            "intermediate_mapping": { ... }
        }
    
    Example Usage:
        >>> # From unified JSON file (contains source, target, and/or intermediate mappings)
        >>> mapper = LabelMapper(overall_mapping_json='mappings/all_mappings.json')

        >>> # From separate files
        >>> mapper = LabelMapper(
        ...     source_mapping_file='mappings/source_mapping.csv',
        ...     target_mapping_file='mappings/target_mapping.csv'
        ... )
        
        >>> # From dictionaries
        >>> mapper = LabelMapper(
        ...     source_mapping_dict={
        ...         'hemibrain:v1.2.1': [['aMe12', 'aMe12_R']],
        ...         'male-cns:v0.9': [['aMe12', 'aMe12-like']]
        ...     },
        ...     source_labels=['aMe12_grp']
        ... )
        
        >>> # Get standardized label
        >>> std = mapper.get_std_label('hemibrain:v1.2.1', 'aMe12', 'source')
    """
    
    def __init__(
        self,
        source_mapping_file: Optional[str] = None,
        target_mapping_file: Optional[str] = None,
        overall_mapping_json: Optional[str] = None,
        source_mapping_dict: Optional[Dict] = None,
        target_mapping_dict: Optional[Dict] = None,
        intermediate_mapping_dict: Optional[Dict] = None,
        source_labels: Optional[List[str]] = None,
        target_labels: Optional[List[str]] = None,
        intermediate_labels: Optional[List[str]] = None,
        intermediate_mapping_file: Optional[str] = None,
    ):
        """
        Initialize LabelMapper.
        
        Args:
            source_mapping_file: Path to CSV or JSON file for source mappings
            target_mapping_file: Path to CSV or JSON file for target mappings
            intermediate_mapping_file: Path to CSV or JSON file for intermediate mappings
            overall_mapping_json: Path to a single JSON file containing all mapping information.
                                 This file should contain 'source_mapping', 'target_mapping', and optionally
                                 'intermediate_mapping' keys. ONLY JSON format is supported for this parameter.
            source_mapping_dict: Dictionary mapping datasets to source neurons
            target_mapping_dict: Dictionary mapping datasets to target neurons
            intermediate_mapping_dict: Dictionary mapping datasets to intermediate neurons
            source_labels: Standard labels for source groups (used with dict input)
            target_labels: Standard labels for target groups (used with dict input)
            intermediate_labels: Standard labels for intermediate groups (used with dict input)
        """
        # Internal storage: {std_label: {dataset: [neuron_ids]}}
        self._source_mapping: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
        self._target_mapping: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
        self._intermediate_mapping: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
        
        # Reverse lookup: {dataset: {neuron_id: std_label}}
        self._source_reverse: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._target_reverse: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._intermediate_reverse: Dict[str, Dict[str, str]] = defaultdict(dict)

        # Handle unified mapping file
        if overall_mapping_json:
            if source_mapping_file is None:
                source_mapping_file = overall_mapping_json
            if target_mapping_file is None:
                target_mapping_file = overall_mapping_json
            if intermediate_mapping_file is None:
                intermediate_mapping_file = overall_mapping_json
        
        # Store file paths for export
        self._source_file = source_mapping_file
        self._target_file = target_mapping_file
        self._intermediate_file = intermediate_mapping_file
        
        # Load from files if provided
        if source_mapping_file:
            self._load_source_from_file(source_mapping_file)
        elif source_mapping_dict:
            self._load_source_from_dict(source_mapping_dict, source_labels or [])
        
        if target_mapping_file:
            self._load_target_from_file(target_mapping_file)
        elif target_mapping_dict:
            self._load_target_from_dict(target_mapping_dict, target_labels or [])
            
        if intermediate_mapping_file:
            self._load_intermediate_from_file(intermediate_mapping_file)
        elif intermediate_mapping_dict:
            self._load_intermediate_from_dict(intermediate_mapping_dict, intermediate_labels or [])
        
        # Build reverse lookups
        self._build_reverse_lookups()

    @property
    def is_empty(self) -> bool:
        """Check if the mapper has any mappings."""
        return (not self._source_mapping and 
                not self._target_mapping and 
                not self._intermediate_mapping)
    
    def merge(self, other: 'LabelMapper') -> None:
        """
        Merge another LabelMapper into this one.
        
        Args:
            other: Another LabelMapper instance to merge
        """
        # Merge source mappings
        for label, ds_map in other._source_mapping.items():
            for ds, neurons in ds_map.items():
                # Avoid duplicates
                current = set(self._source_mapping[label][ds])
                new_neurons = [n for n in neurons if n not in current]
                self._source_mapping[label][ds].extend(new_neurons)
        
        # Merge target mappings
        for label, ds_map in other._target_mapping.items():
            for ds, neurons in ds_map.items():
                current = set(self._target_mapping[label][ds])
                new_neurons = [n for n in neurons if n not in current]
                self._target_mapping[label][ds].extend(new_neurons)
                
        # Merge intermediate mappings
        for label, ds_map in other._intermediate_mapping.items():
            for ds, neurons in ds_map.items():
                current = set(self._intermediate_mapping[label][ds])
                new_neurons = [n for n in neurons if n not in current]
                self._intermediate_mapping[label][ds].extend(new_neurons)
        
        # Rebuild reverse lookups
        self._build_reverse_lookups()

    def validate_datasets(self, datasets: List[str], role: str = 'both') -> None:
        """
        Validate that all required datasets are present in the mapping.
        
        Args:
            datasets: List of dataset names to check
            role: 'source', 'target', or 'both'
            
        Raises:
            ValueError: If any dataset is missing from the mapping
        """
        missing = []
        
        # Convert datasets to set for faster lookup
        datasets_set = set(datasets)
        
        if role in ['source', 'both']:
            # Check source mapping
            # Get all datasets present in source mapping
            present_datasets = set()
            for label_map in self._source_mapping.values():
                present_datasets.update(label_map.keys())
            
            # Check if all required datasets are present
            missing_source = datasets_set - present_datasets
            if missing_source:
                for ds in missing_source:
                    missing.append(f"{ds} (source)")
                    
        if role in ['target', 'both']:
            # Check target mapping
            present_datasets = set()
            for label_map in self._target_mapping.values():
                present_datasets.update(label_map.keys())
                
            # Check if all required datasets are present
            missing_target = datasets_set - present_datasets
            if missing_target:
                for ds in missing_target:
                    missing.append(f"{ds} (target)")
        
        if missing:
            raise ValueError(f"Datasets missing from LabelMapper: {', '.join(missing)}")

    def get_mapped_label(self, label: str, dataset: str) -> Optional[Union[str, int, List[Union[str, int]]]]:
        """
        Get the mapped label(s) or ID(s) for a given standard label in a specific dataset.
        
        Args:
            label: The standard label (or custom_label) to look up.
            dataset: The target dataset name.
            
        Returns:
            The mapped value. If the mapping contains a single value, returns that value.
            If it contains multiple values, returns a list.
            Returns None if no mapping is found.
        """
        # Check source mapping
        if label in self._source_mapping and dataset in self._source_mapping[label]:
            values = self._source_mapping[label][dataset]
            if values:
                return values[0] if len(values) == 1 else values
        
        # Check target mapping
        if label in self._target_mapping and dataset in self._target_mapping[label]:
            values = self._target_mapping[label][dataset]
            if values:
                return values[0] if len(values) == 1 else values
                
        # Check intermediate mapping
        if label in self._intermediate_mapping and dataset in self._intermediate_mapping[label]:
            values = self._intermediate_mapping[label][dataset]
            if values:
                return values[0] if len(values) == 1 else values
                
        return None

    def to_dict(self) -> Dict:
        """
        Convert current mappings to a dictionary compatible with _load_from_json.
        
        Returns:
            Dictionary with 'source_mapping', 'target_mapping', and/or 'intermediate_mapping'
        """
        output = {}
        
        # Export source mapping
        if self._source_mapping:
            source_labels = sorted(self._source_mapping.keys())
            source_data = {'custom_label': source_labels}
            
            # Get all datasets
            datasets = set()
            for label_map in self._source_mapping.values():
                datasets.update(label_map.keys())
            
            for ds in sorted(datasets):
                ds_groups = []
                for label in source_labels:
                    ds_groups.append(self._source_mapping[label].get(ds, []))
                source_data[ds] = ds_groups
                
            output['source_mapping'] = source_data
            
        # Export target mapping
        if self._target_mapping:
            target_labels = sorted(self._target_mapping.keys())
            target_data = {'custom_label': target_labels}
            
            # Get all datasets
            datasets = set()
            for label_map in self._target_mapping.values():
                datasets.update(label_map.keys())
            
            for ds in sorted(datasets):
                ds_groups = []
                for label in target_labels:
                    ds_groups.append(self._target_mapping[label].get(ds, []))
                target_data[ds] = ds_groups
                
            output['target_mapping'] = target_data

        # Export intermediate mapping
        if self._intermediate_mapping:
            intermediate_labels = sorted(self._intermediate_mapping.keys())
            intermediate_data = {'custom_label': intermediate_labels}
            
            # Get all datasets
            datasets = set()
            for label_map in self._intermediate_mapping.values():
                datasets.update(label_map.keys())
            
            for ds in sorted(datasets):
                ds_groups = []
                for label in intermediate_labels:
                    ds_groups.append(self._intermediate_mapping[label].get(ds, []))
                intermediate_data[ds] = ds_groups
                
            output['intermediate_mapping'] = intermediate_data
            
        return output

    def export_to_json(self, filepath: str) -> None:
        """
        Export current mappings to a JSON file compatible with _load_from_json.
        
        Args:
            filepath: Path to save the JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def _load_source_from_file(self, filepath: str) -> None:
        """Load source mapping from CSV or JSON file."""
        if filepath.endswith('.json'):
            self._load_from_json(filepath, 'source')
        else:
            self._load_from_csv(filepath, 'source')
    
    def _load_target_from_file(self, filepath: str) -> None:
        """Load target mapping from CSV or JSON file."""
        if filepath.endswith('.json'):
            self._load_from_json(filepath, 'target')
        else:
            self._load_from_csv(filepath, 'target')

    def _load_intermediate_from_file(self, filepath: str) -> None:
        """Load intermediate mapping from CSV or JSON file."""
        if filepath.endswith('.json'):
            self._load_from_json(filepath, 'intermediate')
        else:
            self._load_from_csv(filepath, 'intermediate')
    
    def _load_from_csv(self, filepath: str, role: str) -> None:
        """
        Load mapping from CSV file.
        
        Supports two CSV formats:
        
        Format 1 (New - Expanded): Each row is one pattern, grouped by custom_label
            custom_label,std_pattern,dataset1,dataset2,notes
            MBON14,MBON14.*_R,MBON14.*_R,MBON14.*_R,Right hemisphere
            MBON14,MBON14.*_L,MBON14.*_L,MBON14.*_L,Left hemisphere
            
        Format 2 (Legacy): std_label with semicolon-separated patterns
            std_label,dataset1,dataset2,dataset3
            label1,id1;id2,id3,id4
        
        Args:
            filepath: Path to CSV file
            role: 'source', 'target', or 'intermediate'
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Mapping file not found: {filepath}")
        
        df = pd.read_csv(filepath, dtype=str)
        
        # Detect format by checking for 'custom_label' or 'std_label' column
        # Both are supported for backward compatibility
        label_col = None
        if 'custom_label' in df.columns:
            label_col = 'custom_label'
        elif 'std_label' in df.columns:
            label_col = 'std_label'
            
        if label_col == 'custom_label' or 'std_pattern' in df.columns:
            # Use expanded format if custom_label is present OR std_pattern is present
            self._load_csv_expanded_format(df, role)
        elif label_col == 'std_label':
            # Use legacy format if only std_label is present
            self._load_csv_legacy_format(df, role)
        else:
            raise ValueError(f"CSV must have 'custom_label' or 'std_label' column. Found: {df.columns.tolist()}")
        # Direct loader calls (post-construction reloads) must keep the
        # reverse lookups in sync with the forward mappings.
        self._build_reverse_lookups()
    
    def _load_csv_expanded_format(self, df: pd.DataFrame, role: str) -> None:
        """
        Load from new expanded CSV format where each row is one pattern.
        
        Format:
            custom_label,std_pattern,dataset1,dataset2,notes
            MBON14,MBON14.*_R,MBON14.*_R,MBON14.*_R,Right hemisphere
            MBON14,MBON14.*_L,MBON14.*_L,MBON14.*_L,Left hemisphere
        
        Rows with the same custom_label are grouped together.
        Supports both 'custom_label' and 'std_label' for backward compatibility.
        
        Args:
            df: DataFrame with custom_label or std_label column
            role: 'source', 'target', or 'intermediate'
        """
        # Identify label column (support both for backward compatibility)
        label_col = 'custom_label' if 'custom_label' in df.columns else 'std_label'
        
        # Identify dataset columns (exclude labels, patterns, notes, etc.)
        exclude_cols = {'custom_label', 'std_label', 'std_pattern', 'notes', 'description', 'comment'}
        dataset_cols = [c for c in df.columns if c.lower() not in exclude_cols]
        
        if role == 'source':
            mapping = self._source_mapping
        elif role == 'target':
            mapping = self._target_mapping
        else:
            mapping = self._intermediate_mapping
        
        for _, row in df.iterrows():
            custom_label = row.get(label_col, '')
            
            # Auto-generate label if missing
            if pd.isna(custom_label) or str(custom_label).strip() == '':
                # Use std_pattern or first non-empty value
                if 'std_pattern' in row and pd.notna(row['std_pattern']):
                    custom_label = self.auto_generate_label(str(row['std_pattern']))
                else:
                    for col in dataset_cols:
                        if pd.notna(row[col]) and str(row[col]).strip():
                            custom_label = self.auto_generate_label(str(row[col]))
                            break
                if pd.isna(custom_label) or str(custom_label).strip() == '':
                    continue  # Skip rows with no valid data
            
            custom_label = str(custom_label).strip()
            
            for col in dataset_cols:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    # Convert column name back to dataset format
                    dataset = self._unsanitize_dataset_name(col)
                    neuron_id = str(value).strip()
                    
                    # Try to convert to int if it looks like a bodyId
                    try:
                        if neuron_id.isdigit():
                            neuron_id = int(neuron_id)
                    except:
                        pass
                    
                    # Add to mapping (rows with same custom_label are aggregated)
                    if neuron_id not in mapping[custom_label][dataset]:
                        mapping[custom_label][dataset].append(neuron_id)
    
    def _load_csv_legacy_format(self, df: pd.DataFrame, role: str) -> None:
        """
        Load from legacy CSV format with semicolon-separated patterns.
        
        Format:
            std_label,dataset1,dataset2,dataset3
            label1,id1;id2,id3,id4
        
        Args:
            df: DataFrame with std_label or custom_label column
            role: 'source', 'target', or 'intermediate'
        """
        # Identify label column
        label_col = 'std_label' if 'std_label' in df.columns else 'custom_label'
        
        # Get dataset columns (all except label column)
        dataset_cols = [c for c in df.columns if c != label_col]
        
        if role == 'source':
            mapping = self._source_mapping
        elif role == 'target':
            mapping = self._target_mapping
        else:
            mapping = self._intermediate_mapping
        
        for _, row in df.iterrows():
            std_label = row[label_col]
            
            # Auto-generate label if missing
            if pd.isna(std_label) or str(std_label).strip() == '':
                # Find first non-empty value in row for auto-generation
                for col in dataset_cols:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        std_label = self.auto_generate_label(str(row[col]).split(';')[0])
                        break
                if pd.isna(std_label) or str(std_label).strip() == '':
                    continue  # Skip rows with no valid data
            
            std_label = str(std_label).strip()
            
            for col in dataset_cols:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    # Convert column name back to dataset format
                    dataset = self._unsanitize_dataset_name(col)
                    
                    # Handle semicolon-separated values (legacy format)
                    for neuron_id in str(value).split(';'):
                        neuron_id = neuron_id.strip()
                        if not neuron_id:
                            continue
                        
                        # Try to convert to int if it looks like a bodyId
                        try:
                            if neuron_id.isdigit():
                                neuron_id = int(neuron_id)
                        except:
                            pass
                        
                        if neuron_id not in mapping[std_label][dataset]:
                            mapping[std_label][dataset].append(neuron_id)
    
    def _load_from_json(self, filepath: str, role: str) -> None:
        """
        Load mapping from JSON file.
        
        JSON Format:
            {
                "source_mapping": {  # or "target_mapping" or "intermediate_mapping"
                    "custom_label": ["label1", "label2"],  # or "std_label" for legacy
                    "dataset1": [["id1", "id2"], ["id3"]],
                    "dataset2": [["id4"], ["id5", "id6"]]
                }
            }
        
        Args:
            filepath: Path to JSON file
            role: 'source', 'target', or 'intermediate'
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Mapping file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        key = f'{role}_mapping'
        if key not in data:
            # A unified file (overall_mapping_json) may legitimately omit a
            # role (e.g. source-only mappings); skip it. Files without ANY
            # mapping key are still rejected as malformed.
            if not any(k in data for k in ('source_mapping', 'target_mapping', 'intermediate_mapping')):
                raise ValueError(f"JSON must have '{key}' key. Found: {list(data.keys())}")
            return
        
        mapping_data = data[key]
        
        # Support both 'custom_label' (preferred) and 'std_label' (legacy)
        label_key = None
        if 'custom_label' in mapping_data:
            label_key = 'custom_label'
        elif 'std_label' in mapping_data:
            label_key = 'std_label'
        else:
            raise ValueError(f"JSON {key} must have 'custom_label' or 'std_label' array")
        
        std_labels = mapping_data[label_key]
        
        if role == 'source':
            mapping = self._source_mapping
        elif role == 'target':
            mapping = self._target_mapping
        else:
            mapping = self._intermediate_mapping
        
        # Get dataset keys (all except label key)
        dataset_keys = [k for k in mapping_data.keys() if k != label_key]
        
        for dataset in dataset_keys:
            groups = mapping_data[dataset]
            
            if len(groups) != len(std_labels):
                raise ValueError(f"Dataset '{dataset}' has {len(groups)} groups but {len(std_labels)} std_labels")
            
            for i, (label, group) in enumerate(zip(std_labels, groups)):
                # Touch the slot so EMPTY groups still register the dataset
                # for this label (validate_datasets must see every compared
                # dataset even when a group has no members there).
                mapping[label][dataset]
                if isinstance(group, list):
                    for neuron_id in group:
                        mapping[label][dataset].append(neuron_id)
                else:
                    mapping[label][dataset].append(group)
        # Direct loader calls (post-construction reloads) must keep the
        # reverse lookups in sync with the forward mappings.
        self._build_reverse_lookups()
    
    def _load_source_from_dict(self, mapping_dict: Dict, labels: List[str]) -> None:
        """
        Load source mapping from dictionary.
        
        Args:
            mapping_dict: {dataset: [[group1_ids], [group2_ids], ...]}
            labels: ['label1', 'label2', ...] matching group count
        """
        self._load_from_dict(mapping_dict, labels, 'source')
    
    def _load_target_from_dict(self, mapping_dict: Dict, labels: List[str]) -> None:
        """
        Load target mapping from dictionary.
        
        Args:
            mapping_dict: {dataset: [[group1_ids], [group2_ids], ...]}
            labels: ['label1', 'label2', ...] matching group count
        """
        self._load_from_dict(mapping_dict, labels, 'target')

    def _load_intermediate_from_dict(self, mapping_dict: Dict, labels: List[str]) -> None:
        """
        Load intermediate mapping from dictionary.
        
        Args:
            mapping_dict: {dataset: [[group1_ids], [group2_ids], ...]}
            labels: ['label1', 'label2', ...] matching group count
        """
        self._load_from_dict(mapping_dict, labels, 'intermediate')
    
    def _load_from_dict(self, mapping_dict: Dict, labels: List[str], role: str) -> None:
        """
        Load mapping from dictionary input.
        
        Args:
            mapping_dict: {dataset: [[group1_ids], [group2_ids], ...]} or {dataset: [flat_ids]}
            labels: Standard labels for groups
            role: 'source', 'target', or 'intermediate'
        """
        if role == 'source':
            mapping = self._source_mapping
        elif role == 'target':
            mapping = self._target_mapping
        else:
            mapping = self._intermediate_mapping
        
        for dataset, neurons in mapping_dict.items():
            # Ensure neurons is a list of groups
            if not neurons:
                continue
            
            # Check if already grouped (list of lists)
            if isinstance(neurons[0], list):
                groups = neurons
            else:
                # Handle flat list: check if it matches labels length
                # If labels are provided and length matches neurons length (and > 1),
                # treat each neuron as a separate group
                if labels and len(labels) == len(neurons) and len(labels) > 1:
                    groups = [[n] for n in neurons]
                else:
                    # Wrap in single group
                    groups = [neurons]
            
            # Auto-generate labels if not provided
            if not labels:
                labels = [f'{role}_grp{i+1}' for i in range(len(groups))]
            
            if len(groups) > len(labels):
                # Extend labels if needed
                labels = labels + [f'{role}_grp{i+1}' for i in range(len(labels), len(groups))]
            
            for label, group in zip(labels, groups):
                # Touch the slot so EMPTY groups still register the dataset
                # for this label (parity with the JSON loading path).
                mapping[label][dataset]
                for neuron_id in group:
                    mapping[label][dataset].append(neuron_id)
        # Direct loader calls (post-construction merges/reloads) must keep
        # the reverse lookups in sync with the forward mappings.
        self._build_reverse_lookups()
    
    def _build_reverse_lookups(self) -> None:
        """Build reverse lookup dictionaries for fast std_label retrieval."""
        # Source reverse lookup
        for std_label, dataset_dict in self._source_mapping.items():
            for dataset, neuron_ids in dataset_dict.items():
                for neuron_id in neuron_ids:
                    self._source_reverse[dataset][str(neuron_id)] = std_label
        
        # Target reverse lookup
        for std_label, dataset_dict in self._target_mapping.items():
            for dataset, neuron_ids in dataset_dict.items():
                for neuron_id in neuron_ids:
                    self._target_reverse[dataset][str(neuron_id)] = std_label

        # Intermediate reverse lookup
        for std_label, dataset_dict in self._intermediate_mapping.items():
            for dataset, neuron_ids in dataset_dict.items():
                for neuron_id in neuron_ids:
                    self._intermediate_reverse[dataset][str(neuron_id)] = std_label
    
    def _sanitize_dataset_name(self, dataset: str) -> str:
        """Convert dataset name to column-safe format."""
        return canonical_dataset_name(dataset).replace(':', '_').replace('.', '_').replace('-', '_')
    
    def _unsanitize_dataset_name(self, column: str) -> str:
        """
        Attempt to convert sanitized column name back to dataset format.
        Note: This is a best-effort conversion as some info may be lost.
        """
        # Common patterns
        if column.startswith('hemibrain_v'):
            return column.replace('hemibrain_v', 'hemibrain:v').replace('_', '.', 1)
        elif column.startswith('male_cns_v') or column.startswith('male-cns_v'):
            return column.replace('male_cns_v', 'male-cns:v').replace('_', '.')
        elif 'flywire' in column.lower() or 'fafb' in column.lower():
            return column  # FlyWire names typically don't have colons
        else:
            return column

    @staticmethod
    def _split_hemi_suffix(label: str) -> Tuple[str, str]:
        """Split hemisphere suffix (_L/_R/_U) from a label.

        Returns (base, suffix) where suffix includes leading underscore.
        Delegates to the shared ``utils.naming_utils`` implementation.
        """
        try:
            from ..utils.naming_utils import split_hemi_suffix
        except ImportError:  # pragma: no cover - direct package imports
            from utils.naming_utils import split_hemi_suffix
        return split_hemi_suffix(label)
    
    def get_std_label(self, dataset: str, original_id: Union[str, int], role: str) -> str:
        """
        Get standardized label for a neuron ID.
        
        Args:
            dataset: Dataset identifier
            original_id: Original neuron ID (type name or bodyId)
            role: 'source', 'target', or 'intermediate'
            
        Returns:
            Standardized label, or auto-generated if not found
        """
        if role == 'source':
            reverse = self._source_reverse
        elif role == 'target':
            reverse = self._target_reverse
        else:
            reverse = self._intermediate_reverse

        str_id = str(original_id)

        # Try exact match across the dataset-key candidates
        label, _matched = self._reverse_lookup(reverse, dataset, str_id)
        if label is not None:
            return label

        # Try base label without hemisphere suffix and re-apply suffix
        base_id, hemi_suffix = self._split_hemi_suffix(str_id)
        if hemi_suffix:
            label, _matched = self._reverse_lookup(reverse, dataset, base_id)
            if label is not None:
                return f"{label}{hemi_suffix}"

        # Auto-generate label
        return self.auto_generate_label(original_id)
    
    def get_neurons_for_label(self, std_label: str, dataset: str, role: str) -> List:
        """
        Get all neuron IDs for a standard label in a specific dataset.
        
        Args:
            std_label: Standard label
            dataset: Dataset identifier
            role: 'source', 'target', or 'intermediate'
            
        Returns:
            List of neuron IDs (may be empty if dataset doesn't have this label)
        """
        if role == 'source':
            mapping = self._source_mapping
        elif role == 'target':
            mapping = self._target_mapping
        else:
            mapping = self._intermediate_mapping
        
        if std_label in mapping:
            # Dataset-key candidates (raw / sanitized / unsanitized — same
            # treatment as the reverse lookups).
            for key in self._dataset_keys(dataset):
                ds_map = mapping[std_label].get(key)
                if ds_map:
                    return ds_map

        # Fallback: strip hemisphere suffix and try base label
        base_label, _ = self._split_hemi_suffix(std_label)
        if base_label in mapping:
            for key in self._dataset_keys(dataset):
                ds_map = mapping[base_label].get(key)
                if ds_map:
                    return ds_map

        return []
    
    def get_all_std_labels(self, role: str) -> List[str]:
        """
        Get all standard labels for a role.
        
        Args:
            role: 'source', 'target', or 'intermediate'
            
        Returns:
            List of all standard labels
        """
        if role == 'source':
            mapping = self._source_mapping
        elif role == 'target':
            mapping = self._target_mapping
        else:
            mapping = self._intermediate_mapping
        return list(mapping.keys())
    
    def get_datasets(self, role: str) -> List[str]:
        """
        Get all datasets that have mappings for a role.
        
        Args:
            role: 'source', 'target', or 'intermediate'
            
        Returns:
            List of dataset identifiers
        """
        if role == 'source':
            mapping = self._source_mapping
        elif role == 'target':
            mapping = self._target_mapping
        else:
            mapping = self._intermediate_mapping
        datasets = set()
        for label_dict in mapping.values():
            datasets.update(label_dict.keys())
        return list(datasets)
    
    @staticmethod
    def auto_generate_label(original_id: Union[str, int]) -> str:
        """
        Auto-generate a standard label for an unmapped ID.
        
        Args:
            original_id: Original neuron ID
            
        Returns:
            Generated label (returns original_id as string to allow merging)
        """
        return str(original_id)
    
    def get_mapping_summary(self) -> pd.DataFrame:
        """
        Generate a summary DataFrame of all mappings for user verification.
        
        Returns:
            DataFrame with columns: role, std_label, dataset, neuron_count, neurons
        """
        rows = []
        
        for std_label, dataset_dict in self._source_mapping.items():
            for dataset, neurons in dataset_dict.items():
                rows.append({
                    'role': 'source',
                    'std_label': std_label,
                    'dataset': dataset,
                    'neuron_count': len(neurons),
                    'neurons': str(neurons[:5]) + ('...' if len(neurons) > 5 else '')
                })
        
        for std_label, dataset_dict in self._target_mapping.items():
            for dataset, neurons in dataset_dict.items():
                rows.append({
                    'role': 'target',
                    'std_label': std_label,
                    'dataset': dataset,
                    'neuron_count': len(neurons),
                    'neurons': str(neurons[:5]) + ('...' if len(neurons) > 5 else '')
                })

        for std_label, dataset_dict in self._intermediate_mapping.items():
            for dataset, neurons in dataset_dict.items():
                rows.append({
                    'role': 'intermediate',
                    'std_label': std_label,
                    'dataset': dataset,
                    'neuron_count': len(neurons),
                    'neurons': str(neurons[:5]) + ('...' if len(neurons) > 5 else '')
                })
        
        return pd.DataFrame(rows)
    
    def export_to_parameters(self) -> Dict:
        """
        Export mapping configuration for parameters.json inclusion.
        
        Returns:
            Dictionary with mapping info for reproducibility
        """
        # Convert defaultdicts to regular dicts for JSON serialization
        resolved_source = {}
        for std_label, dataset_dict in self._source_mapping.items():
            resolved_source[std_label] = dict(dataset_dict)
        
        resolved_target = {}
        for std_label, dataset_dict in self._target_mapping.items():
            resolved_target[std_label] = dict(dataset_dict)

        resolved_intermediate = {}
        for std_label, dataset_dict in self._intermediate_mapping.items():
            resolved_intermediate[std_label] = dict(dataset_dict)
        
        return {
            'source_mapping_file': self._source_file,
            'target_mapping_file': self._target_file,
            'intermediate_mapping_file': self._intermediate_file,
            'resolved_source_mapping': resolved_source,
            'resolved_target_mapping': resolved_target,
            'resolved_intermediate_mapping': resolved_intermediate
        }
    
    def has_mapping(self) -> bool:
        """Check if any mappings are defined."""
        return bool(self._source_mapping) or bool(self._target_mapping) or bool(self._intermediate_mapping)
    
    def get_all_neurons_for_dataset(self, dataset: str, role: str) -> List:
        """
        Get all neuron IDs for a dataset in a specific role.
        
        This is used when LabelMapper is passed to ComparisonParameters
        to provide dataset-specific neuron lists.
        
        Args:
            dataset: Dataset identifier
            role: 'source', 'target', or 'intermediate'
            
        Returns:
            List of neuron IDs for this dataset
        """
        if role == 'source':
            mapping = self._source_mapping
        elif role == 'target':
            mapping = self._target_mapping
        else:
            mapping = self._intermediate_mapping
        all_neurons = []
        for std_label, dataset_dict in mapping.items():
            for key in self._dataset_keys(dataset):
                if key in dataset_dict:
                    all_neurons.extend(dataset_dict[key])
                    break

        return all_neurons
    
    def _dataset_keys(self, dataset: str) -> List[str]:
        """Candidate lookup keys for one dataset (2026-09-15 review fix).

        Mappers loaded from CSV store SANITIZED column names while the
        dict/JSON lanes store RAW dataset names, so a query may arrive in
        either form. Returns ``[dataset, sanitized, unsanitized]``
        (deduplicated, order preserved).
        """
        candidates = [dataset]
        sanitized = self._sanitize_dataset_name(dataset)
        if sanitized not in candidates:
            candidates.append(sanitized)
        unsanitized = self._unsanitize_dataset_name(sanitized)
        if unsanitized not in candidates:
            candidates.append(unsanitized)
        return candidates

    def _reverse_lookup(self, reverse: Dict, dataset: str,
                        str_id: str) -> Tuple[Optional[str], Optional[str]]:
        """First reverse-lookup hit for ``str_id`` across the dataset-key
        candidates. Returns ``(std_label, matched_key)`` or ``(None, None)``."""
        for key in self._dataset_keys(dataset):
            table = reverse.get(key)
            if table and str_id in table:
                return table[str_id], key
        return None, None

    def get_label(self, dataset: str, original_id: Union[str, int]) -> str:
        """
        Get standardized label for a neuron ID, checking all mappings.
        Priority: Source -> Target -> Intermediate

        Tries the raw dataset name, its sanitized column form, and the
        best-effort unsanitized form; for IDs carrying a hemisphere suffix
        (``_L``/``_R``/``_U``) the base ID is looked up and the suffix
        re-applied, matching ``get_std_label``.

        Args:
            dataset: Dataset identifier
            original_id: Original neuron ID

        Returns:
            Standardized label, or the original ID when unmapped
        """
        str_id = str(original_id)

        # Check source, then target, then intermediate across every
        # dataset-key candidate.
        for reverse in (self._source_reverse, self._target_reverse,
                        self._intermediate_reverse):
            label, _matched = self._reverse_lookup(reverse, dataset, str_id)
            if label is not None:
                return label

        # Try base label without hemisphere suffix and re-apply suffix
        base_id, hemi_suffix = self._split_hemi_suffix(str_id)
        if hemi_suffix:
            for reverse in (self._source_reverse, self._target_reverse,
                            self._intermediate_reverse):
                label, _matched = self._reverse_lookup(reverse, dataset,
                                                       base_id)
                if label is not None:
                    return f"{label}{hemi_suffix}"

        return self.auto_generate_label(original_id)

    def apply_to_dataframe(self, df: pd.DataFrame, dataset: str) -> pd.DataFrame:
        """
        Apply standard labels to a connection DataFrame.
        
        Adds 'std_label_pre' and 'std_label_post' columns based on type columns.
        Uses get_label() to check all mappings (source, target, intermediate).
        
        Args:
            df: DataFrame with type_pre and type_post columns
            dataset: Dataset identifier for this data
            
        Returns:
            DataFrame with added std_label columns
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Apply to pre (source) neurons
        if 'type_pre' in df.columns:
            df['std_label_pre'] = df['type_pre'].apply(
                lambda x: self.get_label(dataset, x) if pd.notna(x) else ''
            )
        elif 'bodyId_pre' in df.columns:
            df['std_label_pre'] = df['bodyId_pre'].apply(
                lambda x: self.get_label(dataset, x)
            )
        
        # Apply to post (target) neurons
        if 'type_post' in df.columns:
            df['std_label_post'] = df['type_post'].apply(
                lambda x: self.get_label(dataset, x) if pd.notna(x) else ''
            )
        elif 'bodyId_post' in df.columns:
            df['std_label_post'] = df['bodyId_post'].apply(
                lambda x: self.get_label(dataset, x)
            )
        
        return df
    
    def __repr__(self) -> str:
        source_labels = len(self._source_mapping)
        target_labels = len(self._target_mapping)
        return f"LabelMapper(source_labels={source_labels}, target_labels={target_labels})"
    
    @staticmethod
    def verify_csv_format(filepath: str) -> Dict[str, Any]:
        """
        Verify CSV mapping file format and return validation results.
        
        Expected CSV Format:
            custom_label,dataset1,dataset2,...
            label1,neuron_id1,neuron_id2,...
            label2,neuron_id3,neuron_id4,...
        
        Args:
            filepath: Path to CSV file to verify
            
        Returns:
            Dict with 'valid', 'errors', 'warnings', and 'summary' keys
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        if not os.path.exists(filepath):
            result['valid'] = False
            result['errors'].append(f"File not found: {filepath}")
            return result
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Failed to read CSV: {str(e)}")
            return result
        
        # Check for required column
        if 'custom_label' not in df.columns and 'std_label' not in df.columns:
            result['valid'] = False
            result['errors'].append(
                "Missing required column: 'custom_label' (or 'std_label' for legacy format). "
                f"Found columns: {df.columns.tolist()}"
            )
            return result
        
        label_col = 'custom_label' if 'custom_label' in df.columns else 'std_label'
        
        # Check for at least one dataset column
        exclude_cols = {'custom_label', 'std_label', 'std_pattern', 'notes', 'description', 'comment'}
        dataset_cols = [c for c in df.columns if c.lower() not in exclude_cols]
        
        if not dataset_cols:
            result['valid'] = False
            result['errors'].append("No dataset columns found. Add columns like 'hemibrain_v1_2_1', 'male_cns_v0_9', etc.")
            return result
        
        # Check for empty labels
        empty_labels = df[label_col].isna().sum()
        if empty_labels > 0:
            result['warnings'].append(f"{empty_labels} rows have empty {label_col} values (will be auto-generated)")
        
        # Check for duplicate labels with different values
        if label_col in df.columns:
            label_counts = df[label_col].value_counts()
            duplicates = label_counts[label_counts > 1]
            if len(duplicates) > 0:
                result['warnings'].append(
                    f"Labels appear multiple times (will be grouped): {duplicates.index.tolist()[:5]}"
                )
        
        # Summary
        result['summary'] = {
            'rows': len(df),
            'label_column': label_col,
            'dataset_columns': dataset_cols,
            'unique_labels': df[label_col].nunique()
        }
        
        return result
    
    @staticmethod
    def verify_json_format(filepath: str) -> Dict[str, Any]:
        """
        Verify JSON mapping file format and return validation results.
        
        Expected JSON Format:
            {
                "source_mapping": {
                    "custom_label": ["label1", "label2"],
                    "dataset1": [["id1", "id2"], ["id3"]],
                    "dataset2": [["id4"], ["id5", "id6"]]
                },
                "target_mapping": {
                    "custom_label": ["label1"],
                    "dataset1": [["id7"]]
                }
            }
        
        Args:
            filepath: Path to JSON file to verify
            
        Returns:
            Dict with 'valid', 'errors', 'warnings', and 'summary' keys
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        if not os.path.exists(filepath):
            result['valid'] = False
            result['errors'].append(f"File not found: {filepath}")
            return result
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            result['valid'] = False
            result['errors'].append(f"Invalid JSON: {str(e)}")
            return result
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Failed to read file: {str(e)}")
            return result
        
        # Check for at least one mapping section
        has_source = 'source_mapping' in data
        has_target = 'target_mapping' in data
        has_intermediate = 'intermediate_mapping' in data
        
        if not has_source and not has_target and not has_intermediate:
            result['valid'] = False
            result['errors'].append(
                "JSON must have 'source_mapping', 'target_mapping', or 'intermediate_mapping' keys. "
                f"Found: {list(data.keys())}"
            )
            return result
        
        summary_sections = {}
        
        # Validate each section
        for section_name in ['source_mapping', 'target_mapping', 'intermediate_mapping']:
            if section_name not in data:
                continue
            
            section = data[section_name]
            
            # Check for label key
            label_key = None
            if 'custom_label' in section:
                label_key = 'custom_label'
            elif 'std_label' in section:
                label_key = 'std_label'
            
            if not label_key:
                result['valid'] = False
                result['errors'].append(
                    f"'{section_name}' must have 'custom_label' (or 'std_label') array. "
                    f"Found keys: {list(section.keys())}"
                )
                continue
            
            labels = section[label_key]
            if not isinstance(labels, list):
                result['valid'] = False
                result['errors'].append(f"'{section_name}.{label_key}' must be an array")
                continue
            
            # Check dataset arrays
            dataset_keys = [k for k in section.keys() if k not in ['custom_label', 'std_label']]
            
            for dataset_key in dataset_keys:
                dataset_values = section[dataset_key]
                if not isinstance(dataset_values, list):
                    result['valid'] = False
                    result['errors'].append(f"'{section_name}.{dataset_key}' must be an array")
                    continue
                
                if len(dataset_values) != len(labels):
                    result['valid'] = False
                    result['errors'].append(
                        f"'{section_name}.{dataset_key}' has {len(dataset_values)} groups "
                        f"but {len(labels)} labels defined"
                    )
            
            summary_sections[section_name] = {
                'label_key': label_key,
                'label_count': len(labels),
                'datasets': dataset_keys
            }
        
        result['summary'] = summary_sections
        
        return result
    
    @classmethod
    def verify_mapping_file(cls, filepath: str) -> Dict[str, Any]:
        """
        Auto-detect file type and verify format.
        
        Args:
            filepath: Path to CSV or JSON mapping file
            
        Returns:
            Dict with 'valid', 'errors', 'warnings', and 'summary' keys
        """
        if filepath.lower().endswith('.json'):
            return cls.verify_json_format(filepath)
        elif filepath.lower().endswith('.csv'):
            return cls.verify_csv_format(filepath)
        else:
            return {
                'valid': False,
                'errors': [f"Unsupported file type. Use .csv or .json: {filepath}"],
                'warnings': [],
                'summary': {}
            }
