"""
DatasetConfig - Per-dataset configuration for cross-dataset comparison.

This module defines the DatasetConfig dataclass that encapsulates dataset-specific
settings. In the optimized workflow, DatasetConfig is typically auto-generated
from dataset strings - users rarely need to create it manually.

Dataset Detection:
- FAFB identifiers (including historical ``flywire_FAFB_*`` names) use the
  FAFB local release path.
- BANC identifiers use the standalone public BANC release path.
- Everything else is a NeuPrint dataset and requires a client for authentication.

Note: Source/target neurons and max_interlayer are now defined in ComparisonParameters
(shared across all datasets), not in DatasetConfig.
"""

from dataclasses import dataclass
from typing import Optional, Any

try:
    from ..utils.naming_utils import canonical_dataset_name
except ImportError:  # pragma: no cover - direct package imports
    from utils.naming_utils import canonical_dataset_name

try:
    from ..flywire_ids import (
        is_banc_dataset,
        is_fafb_dataset,
        is_local_connectome_dataset,
    )
except ImportError:  # pragma: no cover - direct package imports
    from flywire_ids import (
        is_banc_dataset,
        is_fafb_dataset,
        is_local_connectome_dataset,
    )


@dataclass
class DatasetConfig:
    """
    Encapsulates per-dataset configuration for cross-dataset comparison.
    
    This class holds dataset-specific parameters like connection info.
    Source/target neurons and max_interlayer are defined in ComparisonParameters
    (shared across all datasets).
    
    Dataset Detection:
    - FAFB (including historical ``flywire_FAFB_*`` names) -> FAFB local data
    - BANC -> standalone public BANC release data
    - Otherwise -> NeuPrint dataset (all use neuprint.janelia.org)
    
    Attributes:
        dataset: Dataset identifier (e.g., 'hemibrain:v1.2.1', 'flywire_FAFB_v783')
        client: NeuPrint client instance (shared across NeuPrint datasets to avoid repeated login)
        name: Human-readable name for the dataset (auto-generated from dataset)
        
    Example:
        >>> # Auto-create from string (recommended)
        >>> config = DatasetConfig.from_string('hemibrain:v1.2.1', client=neuprint_client)
        >>>
        >>> # Manual creation
        >>> config = DatasetConfig(dataset='hemibrain:v1.2.1', client=my_client)
    """
    
    dataset: str
    """Dataset identifier (e.g., 'hemibrain:v1.2.1', 'flywire_FAFB_v783')"""
    
    client: Optional[Any] = None
    """NeuPrint client instance (shared to avoid repeated login)"""
    
    name: str = ''
    """Human-readable name for the dataset"""
    
    def __post_init__(self):
        """Initialize derived attributes after dataclass initialization."""
        # Auto-generate name from dataset if not provided
        if not self.name:
            self.name = self._sanitize_dataset_name(self.dataset)
    
    def _sanitize_dataset_name(self, dataset: str) -> str:
        """
        Convert dataset identifier to a filesystem-safe name.
        
        Args:
            dataset: Dataset identifier (e.g., 'hemibrain:v1.2.1')
            
        Returns:
            Sanitized name (e.g., 'hemibrain_v1_2_1')
        """
        return canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
    
    @property
    def safe_name(self) -> str:
        """Get filesystem-safe version of dataset name."""
        return self._sanitize_dataset_name(self.dataset)
    
    @property
    def is_flywire(self) -> bool:
        """Compatibility alias for the FAFB-only classification."""
        return self.is_fafb

    @property
    def is_fafb(self) -> bool:
        """Check if this is the FAFB local release."""
        return is_fafb_dataset(self.dataset)

    @property
    def is_banc(self) -> bool:
        """Check if this is the standalone BANC public release."""
        return is_banc_dataset(self.dataset)

    @property
    def is_local(self) -> bool:
        """Check if this is either supported local release."""
        return is_local_connectome_dataset(self.dataset)
    
    @property
    def is_neuprint(self) -> bool:
        """Check if this is a NeuPrint-based dataset."""
        return not self.is_local
    
    @classmethod
    def from_string(cls, dataset_str: str, client: Optional[Any] = None) -> 'DatasetConfig':
        """
        Create DatasetConfig from a dataset identifier string.
        
        Args:
            dataset_str: Dataset identifier (e.g., 'hemibrain:v1.2.1', 'flywire_FAFB_v783')
            client: NeuPrint client instance (required for NeuPrint datasets, 
                    shared to avoid repeated login)
            
        Returns:
            DatasetConfig instance
            
        Example:
            >>> config = DatasetConfig.from_string('hemibrain:v1.2.1', client=neuprint_client)
            >>> config = DatasetConfig.from_string('flywire_FAFB_v783')  # No client needed
        """
        return cls(
            dataset=dataset_str,
            client=client,
        )
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation (excluding client object)
        """
        return {
            'dataset': self.dataset,
            'name': self.name,
            'is_flywire': self.is_flywire,
            'is_fafb': self.is_fafb,
            'is_banc': self.is_banc,
            'is_local': self.is_local,
            'is_neuprint': self.is_neuprint,
            # Note: client is intentionally excluded
        }
    
    @classmethod
    def from_dict(cls, data: dict, client: Optional[Any] = None) -> 'DatasetConfig':
        """
        Create DatasetConfig from dictionary.
        
        Args:
            data: Dictionary with configuration data
            client: Optional NeuPrint client
            
        Returns:
            DatasetConfig instance
        """
        return cls(
            dataset=data['dataset'],
            name=data.get('name', ''),
            client=client,
        )
    
    def __repr__(self) -> str:
        if self.is_banc:
            return f"DatasetConfig(dataset='{self.dataset}', type='banc')"
        if self.is_fafb:
            return f"DatasetConfig(dataset='{self.dataset}', type='fafb')"
        return f"DatasetConfig(dataset='{self.dataset}', type='neuprint')"
