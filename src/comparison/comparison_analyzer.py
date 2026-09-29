"""
ComparisonAnalyzer - Main orchestrator for cross-dataset comparison.

This module provides the primary interface for running path analyses across
multiple datasets and comparing results.

Optimized Workflow:
    1. Create ComparisonParameters with all settings (datasets, neurons, thresholds)
    2. Create ComparisonAnalyzer with parameters
    3. Run comparison and generate reports

Example:
    >>> params = ComparisonParameters(
    ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    ...     source_neurons=['MBON14.*_R'],
    ...     target_neurons=['KCg-d.*_R', 'PPL101.*_R'],
    ...     max_interlayer=2,
    ...     thresholds=[1, 3, 5, 10, 20],
    ... )
    >>> analyzer = ComparisonAnalyzer(params)
    >>> results = analyzer.run_comparison()
"""

import os
import json
from pathlib import Path
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Optional, Any, Union, Tuple, Set
import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    from ..flywire_ids import is_fafb_dataset, is_local_connectome_dataset
except ImportError:  # pragma: no cover - direct package imports
    from flywire_ids import is_fafb_dataset, is_local_connectome_dataset

from .dataset_config import DatasetConfig
from .comparison_parameters import ComparisonParameters
from .label_mapper import LabelMapper
from .data_loader import DataLoader
from .metrics import ComparisonMetrics
from .interactive_heatmap import generate_interactive_heatmap
from .point_context import (
    ComparisonPoint,
    point_from_value,
    points_from_parameters,
    safe_point_id,
)

try:
    from utils.label_utils import is_untyped_type_label, untyped_side
except ImportError:  # pragma: no cover - direct package imports
    try:
        from src.utils.label_utils import is_untyped_type_label, untyped_side
    except ImportError:
        is_untyped_type_label = None
        untyped_side = None

try:
    from utils.threshold_state import applied_threshold_provenance
except ImportError:  # pragma: no cover - direct package imports
    from src.utils.threshold_state import applied_threshold_provenance


def _escape_cypher_string_fallback(value):
    """Inline escape fallback (only used when src.utils.api_utils is unavailable)."""
    if not isinstance(value, str):
        return str(value)
    return value.replace('\\', '\\\\').replace("'", "\\'")


def _wildcard_pattern_to_regex(pattern: str) -> str:
    """Convert a user wildcard pattern to a regex pattern.

    Every ``*`` means "any string".  Pre-existing ``.*`` sequences are already
    that wildcard and must not be mangled into ``..`` (the naive chained
    ``replace('*', '.*')`` turned ``X.*`` into ``X..*``, which no longer
    matched the bare ``X``).
    """
    placeholder = '\x00'
    return pattern.replace('.*', placeholder).replace('*', '.*').replace(placeholder, '.*')


class ComparisonAnalyzer:
    """
    Main orchestrator for cross-dataset comparison analysis.
    
    Handles:
    - Running path analysis on multiple datasets
    - Coordinating label mapping
    - Computing comparison metrics
    - Generating reports
    
    Example:
        >>> # Simple workflow - ComparisonParameters first
        >>> params = ComparisonParameters(
        ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
        ...     source_neurons=['MBON14.*_R'],
        ...     target_neurons=['KCg-d.*_R'],
        ...     max_interlayer=2,
        ...     thresholds=[1, 3, 5, 10, 20],
        ...     output_folder='./comparison_output'
        ... )
        >>> 
        >>> analyzer = ComparisonAnalyzer(params)
        >>> results = analyzer.run_comparison()
        >>> report = analyzer.generate_report()
    """
    
    # Normalization map for neurotransmitter names
    # Key: uppercase variation, Value: canonical name (lowercase)
    _NT_NORMALIZATION_MAP = {
        'ACH': 'acetylcholine',
        'ACETYLCHOLINE': 'acetylcholine',
        'GABA': 'gaba',
        'GLUT': 'glutamate',
        'GLUTAMATE': 'glutamate',
        'DA': 'dopamine',
        'DOPAMINE': 'dopamine', 
        'SER': 'serotonin',
        'SEROTONIN': 'serotonin',
        'OCT': 'octopamine',
        'OCTOPAMINE': 'octopamine',
        'HIS': 'histamine',
        'HISTAMINE': 'histamine',
        'UNKNOWN': 'unknown',
        'NONE': 'unknown',
        'NO_CONS': 'unknown'
    }

    def __init__(
        self,
        parameters: ComparisonParameters,
        label_mapper: Optional[LabelMapper] = None,
        verbose: bool = True
    ):
        """
        Initialize ComparisonAnalyzer.
        
        Args:
            parameters: ComparisonParameters with all analysis settings
            label_mapper: Optional LabelMapper for cross-dataset standardization
            verbose: Print progress messages
        """
        self.parameters = parameters
        self.label_mapper = label_mapper
        self.verbose = verbose
        self.separate_hemispheres = parameters.separate_hemispheres
        
        # Track if a mapper was explicitly provided
        has_user_mapper = self.label_mapper is not None or self.parameters.overall_label_mapper is not None
        
        # Extract and merge LabelMappers from parameters
        if self.label_mapper is None:
            # Initialize unified mapper
            self.label_mapper = LabelMapper()
            
            # Merge overall_label_mapper from parameters
            # Note: ComparisonParameters.__post_init__ ensures that source/target mappers
            # are already merged into overall_label_mapper if they existed.
            if self.parameters.overall_label_mapper:
                 self.label_mapper.merge(self.parameters.overall_label_mapper)

        # Initialize components
        self.metrics = ComparisonMetrics()
        
        # Storage for results
        self.raw_results: Dict[str, Dict[int, pd.DataFrame]] = {}
        self.aligned_results: Dict[Any, pd.DataFrame] = {}
        # Query-aware alignment cache.  Standard-mode scalar keys continue to
        # live in aligned_results for compatibility; combination rows are
        # keyed by their stable query ID here.
        self.query_aligned_results: Dict[str, pd.DataFrame] = {}
        self.comparison_report: Optional[Dict] = None
        
        # Cache for expensive calculations (reused across export/visualizations)
        self._similarity_cache: Dict[int, pd.DataFrame] = {}  # threshold -> similarities
        self._hemisphere_symmetry_cache: Dict[int, Dict[str, Dict]] = {}
        self._network_aligned_cache: Dict[Any, pd.DataFrame] = {}
        self._output_base_printed: bool = False  # Track if base dir was printed
        # Shared-resolver state for canonical merge keys (auto type
        # mapping): one snapshot per run, a merge-key cache, and per-status
        # resolution counts exported with the run's mapping metadata.
        self._mapper_snapshot = None
        self._merge_key_cache: Dict = {}
        self._mapping_status_counts: Dict[str, int] = {}
        self._conflicted_merge_types: Dict[str, str] = {}
        # Union type-resolution coverage (type_coverage.py): query id ->
        # {(type, dataset): TypeCoverageEntry}. None = not built yet; the
        # pass is lazy because it reads the local neuron tables and
        # connection caches once per dataset.
        self._type_coverage_cache: Optional[Dict] = None
        # Query-anchored MergePolicy (plan:
        # plan-query-anchored-cross-dataset-analysis.md) — built lazily
        # once per run; False = build attempted and not applicable/failed
        # (every path then stays on the canonical merge fallback).
        self._merge_policy = None
        # Fix A: StrongestFirst effective cutoff (τ) per (dataset, threshold).
        # None = complete enumeration; a number = τ-bounded (budget bit).
        self._path_taus: Dict[tuple, Optional[float]] = {}
        # Feature G/F: full per-run state for the skip logic and the
        # sensitivity exports: {tau, budget_bitten, paths_complete,
        # skipped, duplicate_of, pathfinding} per (dataset, threshold).
        self._path_run_meta: Dict[tuple, Dict] = {}
        # Feature F: per-threshold state from FindAllPathMultiThreshold
        # runs (tau/budget_bitten/paths_complete/replayed), keyed like
        # _path_run_meta entries.
        self._replay_results: Dict[tuple, Dict] = {}
        # Untyped-neuron drop (default on): edges touching untyped neurons
        # (bodyId-fallback / Unknown / empty type labels) are removed from
        # the cross-dataset results; the dropped rows are exported and the
        # counts appended to the run's user_warning_notes.
        self._untyped_dropped_records: list = []
        self._untyped_drop_stats: Dict[tuple, Dict] = {}
        
        # Resolve dataset configurations from strings
        self._dataset_configs: Dict[str, DatasetConfig] = {}
        self._resolve_dataset_configs()
        
        # Validate datasets if LabelMapper is provided
        if has_user_mapper and self.label_mapper:
            # Determine role to validate based on usage in parameters
            role_to_validate = 'both'
            
            # Check if mapper is used for source/target
            # Note: ComparisonParameters moves mapper to _source_mapper/_target_mapper in __post_init__
            is_source_mapper = getattr(self.parameters, '_source_mapper', None) is not None
            is_target_mapper = getattr(self.parameters, '_target_mapper', None) is not None
            
            # Fallback: check if source_neurons/target_neurons ARE mappers
            if not is_source_mapper:
                is_source_mapper = isinstance(self.parameters.source_neurons, LabelMapper)
            if not is_target_mapper:
                is_target_mapper = isinstance(self.parameters.target_neurons, LabelMapper)
            
            # If explicitly passed in init but not in params, assume 'both' (or check params types)
            # If in params, restrict validation to relevant role
            if is_source_mapper and not is_target_mapper:
                role_to_validate = 'source'
            elif is_target_mapper and not is_source_mapper:
                role_to_validate = 'target'
            
            # Get dataset names from parameters
            dataset_names = [
                ds.dataset if isinstance(ds, DatasetConfig) else ds 
                for ds in self.parameters.datasets
            ]
            self.label_mapper.validate_datasets(dataset_names, role=role_to_validate)
        
        # Setup output directory and data loader
        if parameters.output_folder:
            self.data_loader = DataLoader(parameters.full_output_path)
            self.data_loader.ensure_directories()
        else:
            self.data_loader = None
    
    def _resolve_dataset_configs(self):
        """
        Resolve dataset strings to DatasetConfig objects.
        
        Handles both string identifiers and existing DatasetConfig objects.
        For NeuPrint datasets, shares the client to avoid repeated login.
        """
        for ds in self.parameters.datasets:
            if isinstance(ds, str):
                # Create DatasetConfig from string
                # For NeuPrint datasets, we'll set the client when needed
                config = DatasetConfig.from_string(ds)
                self._dataset_configs[ds] = config
            elif isinstance(ds, DatasetConfig):
                # Use existing config
                self._dataset_configs[ds.dataset] = ds
            else:
                raise ValueError(f"Dataset must be string or DatasetConfig, got {type(ds)}")
    
    def _get_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """Get DatasetConfig for a dataset name."""
        if dataset_name in self._dataset_configs:
            return self._dataset_configs[dataset_name]
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _log(self, message: str, level: str = 'info'):
        """Print message if verbose mode enabled.
        
        Args:
            message: Message to print
            level: Log level ('info', 'warn', 'debug'). Debug messages only shown with extra verbosity.
        """
        if not self.verbose:
            return
        # Skip repetitive debug messages
        if level == 'debug':
            return
        prefix = "⚠️ " if level == 'warn' else ""
        # Use tqdm.write to avoid interfering with progress bars
        tqdm.write(f"[Comparison] {prefix}{message}")

    def _progress(self, step: int, total: int, label: str = ""):
        """Emit a structured step-progress event consumed by the web UI.

        The line ``[DROCAT][progress] <step>/<total> <label>`` drives the
        determinate progress bar + step label in the results panel; it is a
        control event, never shown in the execution log.  Uses ``tqdm.write``
        like :meth:`_log` so it never interleaves with an active bar.
        """
        if self.verbose:
            tqdm.write(
                f"[DROCAT][progress] {int(step)}/{int(total)} {label}".rstrip()
            )
    
    def _log_file(self, filepath: str, description: str = "Saved"):
        """Log file save with relative path (prints base dir only once).
        
        Args:
            filepath: Full file path
            description: Action description (default: "Saved")
        """
        if not self.verbose:
            return
        
        base_dir = self.parameters.full_output_path if self.parameters else None
        
        # Print base directory once at the start
        if base_dir and not self._output_base_printed:
            tqdm.write(f"[Comparison] Output directory: {base_dir}")
            self._output_base_printed = True
        
        # Show only relative path from base dir
        if base_dir and filepath.startswith(base_dir):
            rel_path = os.path.relpath(filepath, base_dir)
            tqdm.write(f"[Comparison] {description}: {rel_path}")
        else:
            tqdm.write(f"[Comparison] {description}: {filepath}")

    # ------------------------------------------------------------------
    # Shared pathfinding provenance helpers
    # ------------------------------------------------------------------

    _PATH_PROVENANCE_KEYS = (
        'threshold_scope', 'requested_threshold', 'applied_threshold',
        'applied_threshold_source', 'strongest_first_budget',
        'strongest_first_budget_bitten', 'strongest_first_tau',
        'tau_canonical', 'strongest_dropped_bottleneck', 'edge_budget',
        'edge_budget_applied', 'edge_budget_landing', 'edge_weight_floor',
        'strongest_retained_bottleneck', 'paths_complete',
    )

    def _path_provenance_from_state(self, threshold: int,
                                    state: Optional[Dict] = None) -> Dict:
        """Return the canonical provenance block for a comparison run.

        ``FindNeuronConnection`` writes the same block to its per-dataset
        folder.  Comparison runs also need it in memory because aggregate
        exports may be built from a replay slice, a cached folder, or the
        edge-mode aggregation path rather than directly from the FNC object.
        Missing legacy fields are reconstructed through the shared formula so
        old cache metadata cannot make ``tau`` look like ``applied_threshold``.
        """
        state = dict(state or {})
        parameters = getattr(self, 'parameters', None)
        requested = state.get('requested_threshold', threshold)
        if requested is None:
            requested = threshold
        strongest_tau = state.get('strongest_first_tau')
        if strongest_tau is None:
            strongest_tau = state.get('tau')
        budget_bitten = bool(
            state.get('strongest_first_budget_bitten', False)
            or state.get('budget_bitten', False)
        )
        budget = state.get('strongest_first_budget')
        if budget in (None, '', 0, '0'):
            budget = getattr(parameters, 'max_paths_bodyid', None)
        budget = int(budget) if budget not in (None, '', 0, '0') else 1000000

        pruning_record = state.get('graph_pruning_record') or {}
        edge_budget = state.get('edge_budget')
        path_mode = state.get('path_mode', getattr(parameters, 'path_mode', 'all'))
        if path_mode != 'shortest' and edge_budget in (None, '', 0, '0'):
            configured_edge_budget = getattr(
                parameters, 'graph_edge_limit_bodyid', None)
            if configured_edge_budget not in (None, '', 0, '0'):
                edge_budget = configured_edge_budget
        edge_landing = state.get('edge_budget_landing')
        if edge_landing is None:
            edge_landing = pruning_record.get('landing')
        strongest_retained = state.get('strongest_retained_bottleneck')
        if strongest_retained is None:
            strongest_retained = pruning_record.get('strongest_retained')

        prov = applied_threshold_provenance(
            requested_threshold=requested,
            strongest_first_tau=strongest_tau,
            strongest_first_budget_bitten=budget_bitten,
            strongest_dropped_bottleneck=state.get(
                'strongest_dropped_bottleneck',
                state.get('strongest_dropped')),
            tau_canonical=state.get('tau_canonical'),
            edge_weight_floor=state.get('edge_weight_floor'),
            edge_budget_landing=edge_landing,
            edge_budget=edge_budget,
            strongest_retained_bottleneck=strongest_retained,
        )
        # A pre-provenance cache may contain only the already-computed
        # applied value/source. Preserve that explicit legacy result when
        # there is not enough mechanism state to recompute it; new and full
        # cache rows always take the shared formula above.
        mechanism_keys = (
            'strongest_first_budget_bitten', 'budget_bitten',
            'strongest_dropped_bottleneck', 'strongest_dropped',
            'tau_canonical', 'edge_weight_floor',
        )
        if (state.get('applied_threshold') is not None
                and state.get('applied_threshold_source')
                and not any(state.get(key) is not None
                            for key in mechanism_keys)):
            prov['applied_threshold'] = int(state['applied_threshold'])
            prov['applied_threshold_source'] = str(
                state['applied_threshold_source'])
            prov['paths_complete'] = bool(state.get(
                'paths_complete',
                prov['applied_threshold_source'] == 'requested'))
        # A skipped/collapsed row's authoritative applied point is the
        # folder it aliases. The orchestrator only aliases a requested
        # threshold to a canonical folder >= it (weaker aliases are
        # materialized instead), so this is >= requested structurally.
        if state.get('skipped') and state.get('applied_folder') is not None:
            folder = int(state['applied_folder'])
            prov['applied_threshold'] = folder
            if not prov.get('tau_canonical'):
                prov['tau_canonical'] = folder
            if prov.get('applied_threshold_source') in (None, 'requested'):
                prov['applied_threshold_source'] = 'strongest_first_budget'
            if state.get('paths_complete') is None:
                prov['paths_complete'] = False
        prov['strongest_first_budget'] = budget
        return prov

    def _path_run_meta_from_state(self, dataset_name: str, threshold: int,
                                  state: Optional[Dict] = None) -> Dict:
        """Normalize partial/legacy per-threshold state to the full contract."""
        state = dict(state or {})
        parameters = getattr(self, 'parameters', None)
        prov = self._path_provenance_from_state(threshold, state)
        meta = dict(state)
        meta.update(prov)
        meta.update({
            # Historical comparison exports use these shorter names.
            'tau': prov.get('strongest_first_tau'),
            'budget_bitten': prov.get('strongest_first_budget_bitten', False),
            'skipped': bool(state.get('skipped', False)),
            'duplicate_of': state.get('duplicate_of'),
            'applied_folder': state.get('applied_folder', threshold),
            'pathfinding': state.get(
                'pathfinding', getattr(parameters, 'pathfinding',
                                       'StrongestFirst')),
            'path_mode': state.get(
                'path_mode', getattr(parameters, 'path_mode', 'all')),
            'comparison_mode': state.get(
                'comparison_mode', getattr(parameters, 'comparison_mode',
                                           'path')),
        })
        if 'paths_complete' not in state:
            meta['paths_complete'] = prov['paths_complete']
        # Preserve replay/fallback markers while normalizing all contract
        # values above.
        return meta

    def _path_run_meta_from_fnc(self, dataset_name: str, threshold: int,
                                fnc, path_mode: str) -> Dict:
        """Capture the post-enumeration FNC provenance for one comparison row."""
        state = dict(getattr(fnc, '_last_provenance', {}) or {})
        if not state:
            # Keep the comparison contract usable with lightweight FNC test
            # doubles and older integrations that expose public attributes
            # but predate ``_last_provenance``.
            state = {
                'strongest_first_budget': getattr(
                    fnc, 'strongest_first_budget', None),
                'strongest_first_budget_bitten': getattr(
                    fnc, 'strongest_first_budget_bitten', False),
                'strongest_first_tau': getattr(
                    fnc, 'strongest_first_cutoff', None),
                'tau_canonical': getattr(fnc, 'tau_canonical', None),
                'strongest_dropped_bottleneck': getattr(
                    fnc, 'strongest_dropped_bottleneck', None),
                'edge_budget': getattr(fnc, 'edge_budget', None),
                'edge_budget_landing': getattr(
                    fnc, 'edge_budget_landing', None),
                'edge_weight_floor': getattr(
                    fnc, 'edge_weight_floor', None),
                'strongest_retained_bottleneck': getattr(
                    fnc, 'strongest_retained_bottleneck', None),
            }
        state.setdefault('requested_threshold', threshold)
        state.setdefault('path_mode', path_mode)
        state.setdefault('applied_folder', threshold)
        return self._path_run_meta_from_state(dataset_name, threshold, state)

    def _read_cached_path_meta(self, dataset_name: str,
                               threshold: int) -> Dict:
        """Read a per-dataset folder's provenance when resuming from disk."""
        folder = self.parameters.get_dataset_output_path(dataset_name,
                                                          threshold)
        attrs_path = os.path.join(folder, 'all_attributes.json')
        if os.path.exists(attrs_path):
            try:
                with open(attrs_path, encoding='utf-8') as handle:
                    attrs = json.load(handle)
                if isinstance(attrs, dict):
                    return self._path_run_meta_from_state(
                        dataset_name, threshold, attrs)
            except (OSError, ValueError, TypeError):
                pass
        return self._path_run_meta_from_state(dataset_name, threshold, {})

    def _neutralize_edge_mode_path_meta(self) -> None:
        """F-XD-004: make edge-mode path-run provenance tell edge truth.

        Must run AFTER ``_complete_path_run_meta`` — that pass re-derives
        tau/bitten from the per-run FNC state and clobbers an earlier
        neutralization (round-6 Windows finding F-P3: the exported root
        showed tau repopulated and no side_path_run because the original
        placement ran first).
        """
        for (ds_name, t), meta in list(self._path_run_meta.items()):
            meta = dict(meta or {})
            side = {k: meta.get(k) for k in (
                'strongest_first_tau', 'tau', 'tau_canonical',
                'strongest_first_budget_bitten', 'budget_bitten',
                'graph_pruning_record', 'edge_budget',
                'edge_budget_landing', 'path_mode') if meta.get(k) is not None}
            meta.update({
                'edge_mode': True,
                'side_path_run': side,
                'comparison_mode': 'edge',
                'strongest_first_tau': None,
                'tau': None,
                'strongest_first_budget_bitten': False,
                'budget_bitten': False,
                'graph_pruning_record': {},
                'edge_budget': None,
                'edge_budget_landing': None,
            })
            self._path_run_meta[(ds_name, t)] = meta

    def _complete_path_run_meta(self) -> None:
        """Normalize every configured comparison row before exporting it."""
        for dataset_name in self.parameters.get_dataset_names():
            for threshold in self.parameters.get_thresholds_for_dataset(
                    dataset_name):
                key = (dataset_name, threshold)
                meta = self._path_run_meta.get(key)
                if meta is None:
                    # A cache-only legacy folder may have result files but no
                    # metadata.  Keep the row observable and explicitly mark
                    # it as a requested-threshold complete fallback.
                    meta = self._read_cached_path_meta(dataset_name,
                                                       threshold)
                self._path_run_meta[key] = self._path_run_meta_from_state(
                    dataset_name, threshold, meta)

    def _path_provenance_row(self, dataset_name: str, threshold: int) -> Dict:
        """Build the stable CSV/JSON projection used by comparison outputs."""
        meta = self._path_run_meta.get((dataset_name, threshold), {}) or {}
        parameters = getattr(self, 'parameters', None)
        prov = self._path_provenance_from_state(threshold, meta)
        stats = getattr(self, '_untyped_drop_stats', {}).get(
            (dataset_name, threshold)) or {}
        if not stats:
            # Aliased rows: the drop happened while materializing the
            # applied threshold, so fall back to that key.
            try:
                applied, _, _, _ = self._applied_state_for(
                    dataset_name, threshold)
            except Exception:
                applied = None
            if applied is not None and int(applied) != int(threshold):
                stats = getattr(self, '_untyped_drop_stats', {}).get(
                    (dataset_name, int(applied))) or {}
        edge_meta = {k: meta.get(k) for k in (
            'edge_mode', 'side_path_run', 'comparison_mode')
            if meta.get(k) is not None}
        return {
            **edge_meta,
            'dataset': dataset_name,
            'threshold': threshold,
            'threshold_scope': (
                'query' if getattr(parameters, 'threshold_mode', 'standard')
                == 'combinations' else 'scalar'),
            'requested_threshold': prov['requested_threshold'],
            'applied_threshold': prov['applied_threshold'],
            'applied_threshold_source': prov['applied_threshold_source'],
            'strongest_first_budget': prov['strongest_first_budget'],
            'strongest_first_budget_bitten': prov[
                'strongest_first_budget_bitten'],
            'strongest_first_tau': prov['strongest_first_tau'],
            # Backward-compatible short name used by existing summaries.
            'tau': prov['strongest_first_tau'],
            'tau_canonical': prov['tau_canonical'],
            'strongest_dropped_bottleneck': prov[
                'strongest_dropped_bottleneck'],
            'strongest_dropped': prov['strongest_dropped_bottleneck'],
            'strongest_retained_bottleneck': prov[
                'strongest_retained_bottleneck'],
            'edge_budget': prov['edge_budget'],
            'edge_budget_applied': prov['edge_budget_applied'],
            'edge_budget_landing': prov['edge_budget_landing'],
            'edge_weight_floor': prov['edge_weight_floor'],
            'paths_complete': prov['paths_complete'],
            'pruned': prov['edge_budget_applied'],
            'skipped': bool(meta.get('skipped', False)),
            'duplicate_of': meta.get('duplicate_of'),
            'applied_folder': meta.get('applied_folder', threshold),
            'path_mode': meta.get('path_mode',
                                  getattr(parameters, 'path_mode', 'all')),
            'comparison_mode': meta.get(
                'comparison_mode',
                getattr(parameters, 'comparison_mode', 'path')),
            'drop_untyped': bool(getattr(parameters, 'drop_untyped',
                                        True)),
            'untyped_dropped_rows': stats.get('rows'),
            'untyped_dropped_neurons': stats.get('neurons'),
            'untyped_dropped_fraction': stats.get('fraction'),
        }

    @staticmethod
    def _safe_query_filename_id(query_id: Any) -> str:
        """Return a portable filename slug without changing query identity.

        Query IDs remain verbatim in manifests and CSV columns.  Only the
        filesystem-facing part is normalized so API callers cannot create
        path separators or platform-specific special names.
        """
        return safe_point_id(query_id)

    # ------------------------------------------------------------------
    # Union type-resolution coverage (type_coverage.py)
    # ------------------------------------------------------------------

    def _type_coverage(self) -> Dict[str, Dict[Tuple[str, str], Any]]:
        """Union-of-appeared-types resolution for every query (lazy, cached).

        Alignment only sees types a dataset recruited; this pass resolves
        the union of appeared types into EVERY dataset and classifies the
        absences (below threshold / not in dataset / unmapped / ...).  A
        failure of the pass must never block the export.
        """
        if self._type_coverage_cache is None:
            try:
                from .type_coverage import build_type_coverage
                self._type_coverage_cache = build_type_coverage(self)
            except Exception as exc:  # noqa: BLE001
                self._log(f"Warning: type coverage pass failed: {exc}")
                self._type_coverage_cache = {}
        return self._type_coverage_cache

    def _type_coverage_for_query(self, point_key: Any) -> Dict[Tuple[str, str], Any]:
        """Coverage entries for one query id / scalar threshold point."""
        coverage = self._type_coverage()
        if not coverage:
            return {}
        try:
            record = self._query_record(point_key)
        except Exception:  # noqa: BLE001
            return {}
        return coverage.get(str(record.get('id') or ''), {}) or {}

    def _type_coverage_txt_lines(self) -> List[str]:
        """Human-readable TYPE COVERAGE section lines for the txt report."""
        coverage = self._type_coverage()
        if not coverage:
            return []
        try:
            from .type_coverage import STATUS_LABELS
            nickname_map = self.parameters.get_nickname_map()
        except Exception:  # noqa: BLE001
            nickname_map = {}
        lines = [
            'TYPE COVERAGE (UNION RESOLUTION):',
            '----------------------------------------------------------------------',
            ('  Union of types that appeared in ANY dataset, resolved into '
             'EVERY dataset via the type mapper.  Only absent (dataset, '
             'type) pairs are listed; present types are omitted.  Full '
             'table: comparison_results/type_resolution_union.csv'),
            '',
        ]
        wrote_any = False
        for query in self.get_threshold_queries():
            query_id = str(query.get('id') or query.get('query_id') or '')
            query_label = query.get('label', query_id)
            entries = coverage.get(query_id) or {}
            if not entries:
                continue
            per_type: Dict[str, List[str]] = {}
            for (type_name, dataset), entry in sorted(entries.items()):
                if entry.present:
                    continue
                nick = nickname_map.get(dataset, dataset)
                text = f'{nick} {entry.status_label}'
                if entry.detail:
                    text += f' ({entry.detail})'
                per_type.setdefault(type_name, []).append(text)
            if not per_type:
                continue
            lines.append(f'  query = {query_id} ({query_label}):')
            for type_name, notes in per_type.items():
                lines.append(f'    {type_name}: ' + '; '.join(notes))
            wrote_any = True
            lines.append('')
        if not wrote_any:
            return []
        return lines

    
    def _save_csv(self, df: pd.DataFrame, filepath: str, index: bool = False):
        """Save DataFrame to CSV with UTF-8 encoding for cross-platform compatibility.
        
        Uses polars for faster writes when available, falls back to pandas.
        Ensures Windows/macOS/Linux compatibility with explicit UTF-8 encoding.
        
        Args:
            df: DataFrame to save
            filepath: Output file path
            index: Whether to include row index (default: False)
        """
        if df is None or (hasattr(df, 'empty') and df.empty):
            return
        
        try:
            import polars as pl
            # Reset index if needed to avoid conversion issues
            if index:
                df_to_save = df.reset_index()
            else:
                df_to_save = df.reset_index(drop=True) if df.index.name or not df.index.equals(pd.RangeIndex(len(df))) else df
            
            # Convert pandas to polars and write - faster for large files
            pl_df = pl.from_pandas(df_to_save)
            pl_df.write_csv(filepath)
        except ImportError:
            # Fallback to pandas with explicit UTF-8 encoding
            df.to_csv(filepath, index=index, encoding='utf-8')
        except Exception:
            # Fallback for any polars conversion issues
            df.to_csv(filepath, index=index, encoding='utf-8')
    
    def _read_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Read CSV with polars (faster) and convert to pandas.
        
        Uses polars for faster reads when available, falls back to pandas.
        Ensures cross-platform compatibility.
        
        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments passed to pandas read_csv
            
        Returns:
            pandas DataFrame
        """
        try:
            import polars as pl
            # Use polars for faster reading ONLY when no pandas-specific kwargs
            # were passed. Silently dropping dtype=/index_col=/header= here
            # used to turn str bodyId columns into int64 and break joins.
            if kwargs:
                return pd.read_csv(filepath, encoding='utf-8', **kwargs)
            return pl.read_csv(filepath, infer_schema_length=10000).to_pandas()
        except ImportError:
            return pd.read_csv(filepath, encoding='utf-8', **kwargs)
        except Exception:
            # Fallback for polars issues (schema inference, etc.)
            return pd.read_csv(filepath, encoding='utf-8', **kwargs)
    
    def _compared_thresholds(self) -> Optional[Set[int]]:
        """Per-dataset thresholds the compared queries actually read.

        ``None`` means "no schedule information" (callers then scan every
        key, the legacy behaviour).  A fresh auto-mode run ALSO holds the
        bootstrap probe (``min(thresholds)``, density capture only — no
        compared matrix uses it) in ``raw_results``, while a later re-export
        never loads it; restricting the universe to the compared schedule
        keeps exported artifacts identical between the two (found 2026-09-18:
        the suspects CSV differed 1068 vs 779 rows for the SAME run).
        """
        try:
            combos = getattr(self.parameters, 'threshold_combinations', None)
            if combos:
                return {int(v) for q in combos
                        for v in (q.get('thresholds') or {}).values()}
            thresholds = getattr(self.parameters, 'thresholds', None)
            return {int(t) for t in (thresholds or [])} or None
        except Exception:
            return None

    def _compared_result_types(self) -> Set[str]:
        """Result types restricted to the compared schedule (deterministic
        across a fresh run and any later re-export)."""
        type_map = self._collect_result_types_by_dataset(
            only_thresholds=self._compared_thresholds())
        return {type_name for values in type_map.values()
                for type_name in values}

    def _collect_result_types_by_dataset(
            self, only_thresholds: Optional[Set[int]] = None
    ) -> Dict[str, Set[str]]:
        """Collect result type names while retaining their source dataset.

        ``raw_results`` is keyed by dataset and then threshold.  Keeping the
        first key through this scan lets report consumers resolve a raw name
        in the namespace where it was actually observed instead of guessing
        from the name alone.

        ``only_thresholds`` (optional set of ints) restricts the scan to the
        given per-dataset threshold keys — pass the compared schedule so the
        type universe cannot depend on whether the process also ran an
        auto-bootstrap probe.
        """
        types_by_dataset: Dict[str, Set[str]] = {}
        for dataset, thresh_results in self.raw_results.items():
            dataset_types = types_by_dataset.setdefault(dataset, set())
            for threshold, result in thresh_results.items():
                if only_thresholds is not None and threshold not in \
                        only_thresholds:
                    continue
                # Handle DataFrame directly (path/edge analysis results)
                if isinstance(result, pd.DataFrame) and not result.empty:
                    # Check common type columns
                    for col in ['type_pre', 'type_post', 'from_type', 'to_type', 
                                'std_label_pre', 'std_label_post']:
                        if col in result.columns:
                            dataset_types.update(
                                result[col].dropna().unique())
                
                # Handle dict structure if present (legacy format)
                elif isinstance(result, dict):
                    for key in ['type_level', 'edge_level']:
                        df = result.get(key)
                        if df is not None and hasattr(df, 'empty') and not df.empty:
                            for col in ['from', 'to', 'from_type', 'to_type',
                                        'type_pre', 'type_post']:
                                if col in df.columns:
                                    dataset_types.update(
                                        df[col].dropna().unique())
        
        return {
            dataset: {value for value in values if isinstance(value, str)}
            for dataset, values in types_by_dataset.items()
            if any(isinstance(value, str) for value in values)
        }

    def _collect_result_types(self) -> Set[str]:
        """Collect all unique neuron types present in comparison results."""
        return {
            type_name
            for values in self._collect_result_types_by_dataset().values()
            for type_name in values
        }

    def get_hemisphere_symmetry_summaries(self) -> Dict[int, Dict[str, Dict]]:
        """Load hemisphere symmetry summaries for all datasets and thresholds.

        Returns:
            Dict mapping threshold -> {dataset_name: summary_dict}
        """
        if not self.parameters.symmetry_analysis:
            return {}

        dataset_names = self.parameters.get_dataset_names()
        thresholds = self.parameters.thresholds
        summaries: Dict[int, Dict[str, Dict]] = {}

        for threshold in thresholds:
            if threshold in self._hemisphere_symmetry_cache:
                summaries[threshold] = self._hemisphere_symmetry_cache[threshold]
                continue

            threshold_summaries: Dict[str, Dict] = {}
            for dataset in dataset_names:
                summary_path = os.path.join(
                    self._resolve_dataset_output_path(dataset, threshold),
                    'hemisphere_symmetry',
                    'symmetry_summary.json'
                )
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, 'r', encoding='utf-8') as f:
                            threshold_summaries[dataset] = json.load(f)
                    except Exception as e:
                        self._log(f"Warning: Failed to load symmetry summary for {dataset} t={threshold}: {e}", level='warn')

            self._hemisphere_symmetry_cache[threshold] = threshold_summaries
            summaries[threshold] = threshold_summaries

        return summaries

    def get_hemisphere_symmetry_summaries_for_query(
            self, query: Any) -> Dict[str, Dict]:
        """Load hemisphere symmetry summaries for one query row.

        The raw-run symmetry JSON is keyed by (dataset, threshold); a query
        simply selects each dataset's own requested threshold cell. No
        summary is inferred from the raw threshold union.
        """
        if not self.parameters.symmetry_analysis:
            return {}
        if not isinstance(query, dict):
            query = self._query_record(query)
        thresholds = query.get('thresholds') or {}
        summaries: Dict[str, Dict] = {}
        for dataset, threshold in thresholds.items():
            summary_path = os.path.join(
                self._resolve_dataset_output_path(dataset, int(threshold)),
                'hemisphere_symmetry',
                'symmetry_summary.json'
            )
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summaries[dataset] = json.load(f)
                except Exception as e:
                    self._log(
                        f"Warning: Failed to load symmetry summary for "
                        f"{dataset} at requested threshold {threshold}: {e}",
                        level='warn')
        return summaries
    
    def get_mapped_results(self) -> Dict[str, Dict[int, pd.DataFrame]]:
        """Get raw_results with type mapping applied.
        
        Creates a copy of raw_results where type_pre and type_post columns
        are replaced with canonical (mapped) type names. This is used for
        visualizations and conservation analysis to properly compare types
        across datasets that may use different naming conventions.
        
        Returns:
            Dict mapping dataset -> threshold -> DataFrame with mapped types
        """
        if not self.parameters.auto_type_mapping or not self.parameters._auto_type_mapper:
            return self.raw_results
        
        mapped_results = {}
        
        for dataset, thresh_results in self.raw_results.items():
            mapped_results[dataset] = {}
            
            for threshold, df in thresh_results.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Create a copy to avoid modifying original
                    mapped_df = df.copy()
                    
                    # Map type columns to canonical names
                    if 'type_pre' in mapped_df.columns:
                        mapped_df['type_pre'] = mapped_df['type_pre'].apply(
                            lambda t: self._get_canonical_type(t, dataset) if pd.notna(t) else t
                        )
                    if 'type_post' in mapped_df.columns:
                        mapped_df['type_post'] = mapped_df['type_post'].apply(
                            lambda t: self._get_canonical_type(t, dataset) if pd.notna(t) else t
                        )
                    
                    mapped_results[dataset][threshold] = mapped_df
                else:
                    mapped_results[dataset][threshold] = df
        
        return mapped_results
    
    def _get_canonical_type(self, type_name: str, dataset: str) -> str:
        """Get canonical (male-cns) type name for a given type.

        Routes through the shared validity resolver
        (``comparison.type_resolver.canonical_merge_key``) so cross-dataset
        path/edge merging honors the panel's policy: licensed renames map
        to their canonical target, while a CONFLICTED type is kept
        dataset-scoped (``dataset:raw``) and can never merge with another
        dataset's same-named rows.

        Per-status counts are recorded on ``self`` on a UNIQUE-resolution
        basis (one per distinct ``(dataset, type)`` — the resolver invokes
        the callback only when it computes an answer, not on a merge-key
        cache hit), so a type resolved once per path row is not
        over-counted.  The conflicted-type record is idempotent.

        Args:
            type_name: Original type name
            dataset: Dataset the type comes from

        Returns:
            Canonical merge key (canonical name, dataset-scoped name for
            conflicts, or the raw name when unmapped/unavailable).
        """
        if not self.parameters.auto_type_mapping or not self.parameters._auto_type_mapper:
            return type_name
        from .type_resolver import canonical_merge_key, get_mapper_snapshot
        if self._mapper_snapshot is None:
            self._mapper_snapshot = get_mapper_snapshot(
                self.parameters._auto_type_mapper)
        merge_key = canonical_merge_key(
            self.parameters._auto_type_mapper, type_name, dataset,
            snapshot=self._mapper_snapshot, cache=self._merge_key_cache,
            on_status=self._note_mapping_status)
        if merge_key.status == 'conflict':
            self._conflicted_merge_types[f'{dataset}:{type_name}'] = merge_key.key
        return merge_key.key

    def _note_mapping_status(self, status: str) -> None:
        """Count one mapping resolution by status (unique-resolution basis)."""
        counts = self._mapping_status_counts
        counts[status] = counts.get(status, 0) + 1
    
    def _get_display_type(self, canonical_name: str) -> str:
        """Get display name for a canonical type showing all dataset variants.
        
        If auto_type_mapping is enabled, returns format like 'MeVPaMe1(MTe46)'.
        Otherwise returns the canonical name unchanged.
        
        Args:
            canonical_name: Canonical (male-cns) type name
            
        Returns:
            Display name with variants in parentheses
        """
        if not self.parameters.auto_type_mapping or not self.parameters._auto_type_mapper:
            return canonical_name
        # Pass all datasets being compared to get full mapping info
        datasets = self.parameters.get_dataset_names()
        return self.parameters._auto_type_mapper.get_display_name(canonical_name, datasets)
    
    def _build_path_key_with_mapping(self, path_nodes: list, dataset: str) -> Tuple[str, str]:
        """Build canonical and display path keys from path nodes.
        
        Args:
            path_nodes: List of type names in the path
            dataset: Dataset the path comes from
            
        Returns:
            Tuple of (canonical_key, display_key) where:
            - canonical_key: Path with canonical names for merging
            - display_key: Path with display names (variants in parentheses)
        """
        # Get canonical names for each node
        canonical_nodes = [self._get_canonical_type(node, dataset) for node in path_nodes]
        canonical_key = ' → '.join(canonical_nodes)
        
        # Get display names for each canonical node
        display_nodes = [self._get_display_type(node) for node in canonical_nodes]
        display_key = ' → '.join(display_nodes)
        
        return canonical_key, display_key

    def _print_intermediate_mapping_summary(self):
        """Print summary of intermediate neuron type mappings.
        
        Shows:
        - Number of types with cross-dataset name differences
        - Count of N-to-1 and 1-to-N mappings
        - Reference to export file for details
        """
        if not self.parameters.auto_type_mapping or not self.parameters._auto_type_mapper:
            return
        
        # Use the helper method to collect all types from results
        # (schedule-bound: the auto-bootstrap probe must not change exports)
        str_types = self._compared_result_types()
        
        if not str_types:
            return
        
        # Get mapping summary
        mapper = self.parameters._auto_type_mapper
        dataset_names = self.parameters.get_dataset_names()
        summary = mapper.get_intermediate_mapping_summary(str_types, dataset_names)
        
        # Print summary
        output_path = self.parameters.full_output_path
        self._log("Auto type mapping summary for intermediate neurons:")
        self._log(f"  • {summary['total_types']} neuron types in comparison results")
        if summary['mapped_count'] > 0:
            self._log(f"  • {summary['mapped_count']} types have cross-dataset name differences")
        if summary['n_to_1_count'] > 0:
            self._log(f"  ⚠️ {summary['n_to_1_count']} N-to-1 type mappings (check aggregation)")
        if summary['one_to_n_count'] > 0:
            self._log(f"  ⚠️ {summary['one_to_n_count']} 1-to-N type mappings (check aggregation)")
        
        if summary['mapped_count'] > 0 or summary['n_to_1_count'] > 0 or summary['one_to_n_count'] > 0:
            self._log(f"  → Check auto_type_mapping.csv and auto_type_mapping_conflicts.csv in output folder")
            self._write_user_warning_notes_for_mapping(mapper, str_types, dataset_names)

        # Canonical merge-key policy (shared resolver): conflicted types are
        # kept dataset-scoped in path/edge merging, never merged under their
        # raw same-name.
        if self._conflicted_merge_types:
            self._log(
                f"  ⚠️ {len(self._conflicted_merge_types)} conflicted type(s) kept "
                f"dataset-scoped in path/edge merges (no raw same-name merging): "
                f"{', '.join(sorted(self._conflicted_merge_types)[:5])}"
                + (" …" if len(self._conflicted_merge_types) > 5 else ""))

    # Notes that describe current run STATE (not a historical log line): a
    # re-export replaces the previous copy instead of appending, so
    # contradictory counts can never accumulate (found 2026-09-18: four
    # copies of the merge-policy / BANC-auto notes after three re-exports,
    # two of them disagreeing — 1060 vs 593 mappings — because the first
    # predated the schedule-bound type universe).
    #
    # A "block" is one top-level header line (starting with ``[`` at column
    # 0) plus its following lines: bullets of the SAME family (tagged
    # ``- [merge fan-in] …``, ``- [type granularity] …`` or untagged) move
    # with the header, while a bullet of ANOTHER tagged family ends the
    # block (``- [untyped dropped] …`` is its own note family and must
    # survive — that family has no header line of its own).
    _STATE_NOTE_PREFIXES = ('[same-name-first]',
                            '[multi-value type cells]',
                            '[merge policy]',
                            '[BANC auto labels]',
                            '[BANC alignment fallback]')
    # Tagged bullet families that are NOT state-owned: their bullets always
    # survive, and one ends an enclosing state block.
    _FOREIGN_BULLET_TAGS = ('[untyped dropped]',)

    def _replace_state_notes(self, blocks: List[str]) -> List[str]:
        """Drop prior copies of the STATE notes from ``user_warning_notes``
        so a re-export does not accumulate contradictory counts.

        Block-aware: a state block is its header line plus every following
        line that is not itself a top-level header (so multi-line bullet
        lists are removed whole, never left orphaned).
        """
        note_path = os.path.join(
            self.parameters.full_output_path, 'user_warning_notes.txt')
        if not os.path.exists(note_path):
            return blocks
        try:
            with open(note_path, encoding='utf-8') as f:
                lines = f.read().splitlines()
        except OSError:
            return blocks
        import re as _re
        bullet_tag = _re.compile(r'^-\s*\[([^\]]+)\]')
        kept: List[str] = []
        skipping = False
        for ln in lines:
            if ln.startswith('['):
                skipping = ln.startswith(self._STATE_NOTE_PREFIXES)
            elif skipping:
                # A bullet of a foreign family ends the state block; its own
                # untagged/tagged continuations move with it.
                m = bullet_tag.match(ln)
                if m and f'[{m.group(1)}]' in self._FOREIGN_BULLET_TAGS:
                    skipping = False
            if skipping:
                continue
            kept.append(ln)
        # Drop the old header lines and trailing blanks; re-emit exactly one
        # header below.
        kept = [ln for ln in kept
                if ln.strip() not in ('User warning notes',
                                      '==================')]
        while kept and not kept[-1].strip():
            kept.pop()
        try:
            with open(note_path, 'w', encoding='utf-8') as f:
                f.write('User warning notes\n==================\n')
                if kept:
                    f.write('\n' + '\n'.join(kept) + '\n')
        except OSError:
            return blocks
        return blocks

    def _append_user_warning_notes(self, out_dir: str, blocks: List[str]) -> None:
        """Append pre-formatted blocks to ``user_warning_notes.txt``.

        Several writers append to this file (threshold comparability,
        untyped drops, query resolution, auto-mapping caveats) and the
        order depends on the run, so the title header is written by
        whichever call creates the file (plan rev-3 item 4).  A file that
        predates the header discipline is healed in place: the title is
        prepended instead of leaving the notes untitled forever.
        """
        if not blocks:
            return
        note_path = os.path.join(out_dir, 'user_warning_notes.txt')
        header = 'User warning notes\n==================\n\n'
        try:
            existing = ''
            if os.path.exists(note_path):
                try:
                    with open(note_path, encoding='utf-8') as f:
                        existing = f.read()
                except OSError:
                    existing = ''
            if not existing.strip():
                with open(note_path, 'w', encoding='utf-8') as f:
                    f.write(header)
                    f.write('\n'.join(blocks) + '\n')
            elif existing.startswith('User warning notes'):
                with open(note_path, 'a', encoding='utf-8') as f:
                    f.write('\n' + '\n'.join(blocks) + '\n')
            else:
                # Heal a headerless legacy file.
                with open(note_path, 'w', encoding='utf-8') as f:
                    f.write(header)
                    f.write(existing.rstrip('\n') + '\n')
                    f.write('\n' + '\n'.join(blocks) + '\n')
        except Exception as e:
            self._log(f"Warning: could not append user warning notes: {e}")

    def _write_user_warning_notes_for_mapping(self, mapper, type_names, dataset_names):
        """Write auto-type-mapping caveats to user_warning_notes.txt.

        The run guide renders this file, so expanded (renamed) type
        mappings and N-to-1 / 1-to-N conflicts surface next to the results
        with a recommendation to double check them.
        """
        try:
            notes = mapper.build_user_warning_notes(type_names, dataset_names)
        except Exception as e:
            self._log(f"  Warning: could not build auto type mapping notes: {e}")
            return None
        if not notes:
            return None
        try:
            os.makedirs(self.parameters.full_output_path, exist_ok=True)
            body = ('Auto type mapping changed how some neuron type names '
                    'were matched across datasets:\n\n'
                    + ''.join(f'- {note}\n' for note in notes))
            self._append_user_warning_notes(
                self.parameters.full_output_path, [body])
            notes_path = os.path.join(
                self.parameters.full_output_path, 'user_warning_notes.txt')
            self._log(f"  ⚠️ Wrote {notes_path} - please double check the automatic mappings")
            return notes_path
        except Exception as e:
            self._log(f"  Warning: could not write user_warning_notes.txt: {e}")
            return None

    def _generate_mode_specific_note(self) -> str:
        """Generate HTML note specific to the comparison mode used."""
        mode = getattr(self.parameters, 'comparison_mode', 'path')
        
        if mode == 'edge':
            return '''
            <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px 16px; margin-top: 12px; border-radius: 0 6px 6px 0;">
                <strong>⚠️ Edge-Based Comparison Mode:</strong> This analysis uses edge-based filtering where each synapse connection 
                is evaluated independently by its weight. Edges are aggregated at the neuron type level. This mode avoids the 
                path-filtering artifacts where strong edges might appear absent due to weak intermediate edges on paths. 
                <br><br>
                <strong>Caveat 1 - Dead-ends:</strong> Edge-based comparison may include <strong>dead-end connections</strong> in the network—edges 
                that are strongly connected but don't contribute to any complete source→target path. These appear in individual 
                datasets but may not be functionally relevant to the circuit. Check the path presence matrix to identify 
                edges that form complete paths versus isolated strong connections.
                <br><br>
                <strong>Caveat 2 - Weight Mismatch:</strong> Edge weights in the <em>Edge Presence Matrix</em> represent the <strong>total 
                synapse count between all neuron pairs</strong> of the source and target types that meet the threshold. However, edge weights 
                shown in the <em>Path Presence Matrix</em> (hop weights) represent only synapses from <strong>neurons actually participating 
                in paths</strong>. The same A→B edge may show different weights: e.g., edge matrix shows 500 synapses (all type-A to type-B 
                connections), while path matrix shows 120 (only neurons on paths from source to target). This is expected behavior—path 
                weights are subsets of edge weights.
            </div>'''
        else:
            return '''
            <div style="background: #fee2e2; border-left: 4px solid #ef4444; padding: 12px 16px; margin-top: 12px; border-radius: 0 6px 6px 0;">
                <strong>⚠️ Path-Based Filtering Caveat:</strong> This analysis uses path-based filtering where edges are discovered 
                through paths from source to target neurons. <strong>Strong edges may appear absent</strong> (marked ❌) if they only 
                exist on paths with weaker intermediate edges that fall below the threshold. An edge marked as "non-existent" in one 
                dataset may actually exist but was filtered due to path context, not edge absence. Compare results across threshold 
                levels to identify such cases—if an edge appears at lower thresholds but disappears at higher ones, it may indicate 
                path-filtering artifacts rather than true biological differences.
            </div>'''
    
    # =========================================================================
    # Path Analysis
    # =========================================================================
    
    def run_path_analysis(
        self,
        dataset_name: str,
        threshold: int,
        verbose_mode: str = 'simple'
    ) -> pd.DataFrame:
        """
        Run path analysis for a single dataset at a specific threshold.
        
        Args:
            dataset_name: Dataset identifier string
            threshold: Weight threshold for path finding
            verbose_mode: Verbosity level for the path run ('simple', 'full', 'silent')
            
        Returns:
            DataFrame with path analysis results
        """
        # Import here to avoid circular imports
        # Use absolute import since src is on sys.path
        from coana import FindNeuronConnection
        
        # Only log when verbose (not 'silent')
        if verbose_mode != 'silent':
            self._log(f"Running analysis: \033[94m{dataset_name} @ threshold={threshold}\033[0m")
        
        # Get dataset config
        config = self._get_dataset_config(dataset_name)
        
        # Get source/target neurons from ComparisonParameters
        source_neurons = self.parameters.get_source_neurons_for_dataset(dataset_name)
        target_neurons = self.parameters.get_target_neurons_for_dataset(dataset_name)
        
        # Get max_interlayer from ComparisonParameters (shared across all datasets)
        max_interlayer = self.parameters.max_interlayer
        
        # Set up output path to redirect FNC output into comparison folder structure
        # Output goes to: {comparison_output}/dataset_data/{dataset}/minsyn_{threshold}/
        safe_dataset_name = self.parameters._sanitize_name(dataset_name)
        fnc_output_path = self.parameters.get_dataset_output_path(dataset_name, threshold)
        
        # Determine custom names based on labels
        # If single label provided, use it as custom name.
        # If multiple labels provided, leave empty to allow auto-naming (or group naming).
        custom_source_name = ''
        if self.parameters.source_labels and len(self.parameters.source_labels) == 1:
            custom_source_name = self.parameters.source_labels[0]
            
        custom_target_name = ''
        if self.parameters.target_labels and len(self.parameters.target_labels) == 1:
            custom_target_name = self.parameters.target_labels[0]

        # Let FindNeuronConnection handle client creation
        # It will auto-detect client_type from dataset name and create/reuse clients as needed
        # - If dataset is FAFB or BANC -> uses its local-release data
        # - Otherwise -> uses NeuPrint (creates client using dataset name and token from env var)
        # Determine if force_API_fetching should be applied (only for FAFB datasets)
        is_fafb = is_fafb_dataset(dataset_name)
        use_force_api = self.parameters.force_API_fetching if is_fafb else False
        
        fnc = FindNeuronConnection(
            sourceNeurons=source_neurons,
            targetNeurons=target_neurons,
            custom_source_name=custom_source_name,
            custom_target_name=custom_target_name,
            max_interlayer=max_interlayer,
            min_synapse_num=threshold,
            min_traversal_probability=0,  # Use 0 to match FindPath.py - default 0.001 can miss weak but important edges
            min_ratio=0,
            dataset=dataset_name,
            # Redirect output to comparison folder structure
            saveas=fnc_output_path,  # Absolute path - overrides data_folder
            verbose_mode=verbose_mode,  # Verbosity level for FindAllPath
            skip_bodyId=self.parameters.skip_bodyId,  # Skip bodyId-level processing if requested
            label_mapper=self.label_mapper,  # Pass label mapper for standardization
            pathfinding=self.parameters.pathfinding,  # Pass pathfinding algorithm
            graph_edge_limit_bodyid=self.parameters.graph_edge_limit_bodyid,  # bodyId edge limit (deep searches)
            max_paths_bodyid=self.parameters.max_paths_bodyid,  # StrongestFirst path budget (None = per-mode default)
            edgeN_limit=self.parameters.edgeN_limit,  # Visualization Edge Limit
            search_columns=self.parameters.search_columns,  # Column scope for neuron name resolution
            force_API_fetching=use_force_api,  # Use CAVE API for FAFB if enabled
            cache_only=self.parameters.cache_only,  # Use cache-only mode if enabled
            separate_hemispheres=self.parameters.separate_hemispheres,
            symmetry_analysis=self.parameters.symmetry_analysis,
            keep_only_hemisphere_conserved_connections=self.parameters.keep_only_hemisphere_conserved_connections,
            # Same as the replay constructor above: the comparison-level
            # drop_untyped fires post label-mapping in the analyzer, so the
            # delegated per-dataset run keeps every row.
            drop_untyped=False,
            # Density curves are a first-class output of every mode: capture
            # the query-scoped classified cone for all delegated runs.
            capture_density=True,
        )

        # Initialize and run analysis
        # Use FindAllPath()/FindShortestPath() as specified in TODO_comparison.md:
        # "DO NOT use the FindDirectConnection() function, because the FindAllPath() 
        # function can already include direct connections as 1-hop paths"
        # (FindShortestPath likewise includes direct connections as 1-hop
        # shortest paths).
        fnc.InitializeNeuronInfo()
        source_df = getattr(fnc, "source_df", None)
        target_df = getattr(fnc, "target_df", None)
        print(
            f"[DROCAT][neuron-match] source={len(source_df) if source_df is not None else 0} "
            f"target={len(target_df) if target_df is not None else 0}",
            flush=True,
        )
        if self.parameters.path_mode == 'shortest':
            fnc.FindShortestPath(find_reciprocal=self.parameters.find_reciprocal)
        else:
            fnc.FindAllPath(find_reciprocal=self.parameters.find_reciprocal)

        # Capture the same post-enumeration provenance for BOTH modes.  The
        # old code populated this block only in the all-path branch, which
        # made shortest-path aggregate exports fall back to the requested
        # threshold even when its StrongestFirst budget had bitten.
        meta = self._path_run_meta_from_fnc(
            dataset_name, threshold, fnc, self.parameters.path_mode)
        self._path_taus[(dataset_name, threshold)] = meta.get('tau')
        self._path_run_meta[(dataset_name, threshold)] = meta
        
        # Get results - FindAllPath saves both path data and connection data
        # For comparison metrics, we need the connection data (edge-level) format:
        # - data_details/connection_info_bodyId.csv has bodyId_pre, bodyId_post, weight, etc.
        # - This is the correct format for comparison metrics
        return self._load_fnc_results(fnc, dataset_name, threshold)

    def _apply_label_mapping_to_dataframe(
        self,
        dataset_name: str,
        threshold: int,
        conn_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply standardized labels before any comparison-level filtering."""
        if conn_df is None or conn_df.empty:
            return conn_df
        conn_df = conn_df.copy()
        if 'dataset' not in conn_df.columns:
            conn_df['dataset'] = dataset_name
        if 'threshold' not in conn_df.columns:
            conn_df['threshold'] = threshold
        if self.label_mapper and not self.label_mapper.is_empty:
            self._log(f"Applying label mapping to {dataset_name} results")
            conn_df = self.label_mapper.apply_to_dataframe(
                conn_df, dataset_name)
            # Standardized labels are the comparison identity.  Do not let
            # native type names survive in the columns consumed downstream.
            if 'std_label_pre' in conn_df.columns:
                mask = conn_df['std_label_pre'] != ''
                conn_df.loc[mask, 'type_pre'] = conn_df.loc[
                    mask, 'std_label_pre']
            if 'std_label_post' in conn_df.columns:
                mask = conn_df['std_label_post'] != ''
                conn_df.loc[mask, 'type_post'] = conn_df.loc[
                    mask, 'std_label_post']
        return conn_df

    def _finalize_loaded_result(self, dataset_name: str, threshold: int,
                                conn_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a cached/replayed frame exactly like a fresh result."""
        conn_df = self._apply_label_mapping_to_dataframe(
            dataset_name, threshold, conn_df)
        return self._drop_untyped_neurons(dataset_name, threshold, conn_df)

    def _load_fnc_results(
        self,
        fnc,
        dataset_name: str,
        threshold: int,
    ) -> pd.DataFrame:
        """Load the per-threshold connection table from a finished FNC run.

        Shared by ``run_path_analysis`` and the Feature F replay path (the
        multi-threshold orchestrator materializes each threshold's folder;
        the analyzer reads the same files it always did).
        """
        conn_df = pd.DataFrame()

        if hasattr(fnc, 'allpath_folder') and fnc.allpath_folder:
            # Try to load connection data (edge-level format for metrics)
            conn_file = os.path.join(
                fnc.allpath_folder, 'data_details', 'connection_info_bodyId.csv'
            )

            if os.path.exists(conn_file):
                try:
                    # Use Polars for faster CSV reading
                    import polars as pl
                    conn_df = pl.read_csv(conn_file, infer_schema_length=10000).to_pandas()
                    self._log(f"Loaded {len(conn_df)} connections from connection_info_bodyId.csv")
                except Exception as e:
                    self._log(f"Warning: Could not read connection file: {e}")
            else:
                # Fallback: try connection_type.csv (type-level)
                conn_type_file = os.path.join(
                    fnc.allpath_folder, 'data_details', 'connection_type.csv'
                )
                if os.path.exists(conn_type_file):
                    try:
                        # Use Polars for faster CSV reading
                        import polars as pl
                        conn_df = pl.read_csv(conn_type_file, infer_schema_length=10000).to_pandas()
                        self._log(f"Loaded {len(conn_df)} connections from connection_type.csv")
                    except Exception as e:
                        self._log(f"Warning: Could not read connection type file: {e}")

        # Add dataset identifier
        if not conn_df.empty:
            conn_df = conn_df.copy()
            conn_df['dataset'] = dataset_name
            conn_df['threshold'] = threshold

        # Fresh, cached, and replayed frames all use the same order:
        # standardized labels first, then the shared untyped predicate.
        return self._finalize_loaded_result(dataset_name, threshold, conn_df)
    
    def run_edge_analysis(
        self,
        dataset_name: str,
        threshold: int,
        base_edges: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Run edge-based analysis for a single dataset at a specific threshold.
        
        Unlike path-based analysis, this queries edges directly between neuron types
        without path context. Edges are filtered by weight threshold independently.
        
        Args:
            dataset_name: Dataset identifier string
            threshold: Weight threshold for edge filtering
            base_edges: Optional pre-fetched edges at lowest threshold (for efficiency)
            
        Returns:
            DataFrame with edge analysis results
        """
        self._log(f"Running edge analysis: \033[94m{dataset_name} @ threshold={threshold}\033[0m")
        
        # Get source/target neurons
        source_neurons = self.parameters.get_source_neurons_for_dataset(dataset_name)
        target_neurons = self.parameters.get_target_neurons_for_dataset(dataset_name)
        
        # If base_edges provided, filter from it
        if base_edges is not None and not base_edges.empty:
            filtered = base_edges[base_edges['weight'] >= threshold].copy()
            filtered['threshold'] = threshold
            self._log(f"Filtered {len(filtered)} edges from base at threshold={threshold}", 'debug')
            return self._finalize_loaded_result(
                dataset_name, threshold, filtered)
        
        # Otherwise, query edges directly
        conn_df = self._query_edges_for_dataset(
            dataset_name, source_neurons, target_neurons, threshold
        )
        
        if not conn_df.empty:
            conn_df = conn_df.copy()
            conn_df['dataset'] = dataset_name
            conn_df['threshold'] = threshold
            
        conn_df = self._finalize_loaded_result(
            dataset_name, threshold, conn_df)
        return conn_df
    
    def _query_edges_for_dataset(
        self,
        dataset_name: str,
        source_neurons: List,
        target_neurons: List,
        min_weight: int = 1
    ) -> pd.DataFrame:
        """
        Query all edges between source and target neuron types.
        
        This queries edges directly without path context, capturing all connections
        between relevant neuron types regardless of path existence.
        
        Args:
            dataset_name: Dataset identifier
            source_neurons: List of source neuron types/patterns
            target_neurons: List of target neuron types/patterns
            min_weight: Minimum edge weight
            
        Returns:
            DataFrame with columns: bodyId_pre, bodyId_post, type_pre, type_post, weight
        """
        # Check whether the dataset is one of the exact local releases or a
        # NeuPrint dataset.  BANC and FAFB share file mechanics, not identity.
        is_local = is_local_connectome_dataset(dataset_name)
        
        if is_local:
            return self._query_edges_local(dataset_name, source_neurons, target_neurons, min_weight)
        else:
            return self._query_edges_neuprint(dataset_name, source_neurons, target_neurons, min_weight)
    
    def _query_edges_neuprint(
        self,
        dataset_name: str,
        source_neurons: List,
        target_neurons: List,
        min_weight: int = 1
    ) -> pd.DataFrame:
        """Query edges from NeuPrint database."""
        try:
            from neuprint import Client
            
            token = self.parameters.resolve_token()
            client = Client('neuprint.janelia.org', dataset=dataset_name, token=token)
            
            # Import API utilities for Cypher escaping
            try:
                from src.utils.api_utils import escape_cypher_string
            except ImportError:
                escape_cypher_string = _escape_cypher_string_fallback
            
            # Build type patterns for Cypher query
            # Handle regex patterns (convert .* to Cypher regex)
            def format_types_for_cypher(types_list):
                formatted = []
                for t in types_list:
                    if isinstance(t, str):
                        escaped_t = escape_cypher_string(t)
                        if '.*' in t or '*' in t:
                            # Convert to Cypher regex pattern
                            pattern = _wildcard_pattern_to_regex(escaped_t)
                            formatted.append(f"a.type =~ '{pattern}'")
                        else:
                            formatted.append(f"a.type = '{escaped_t}'")
                return formatted
            
            # Build source and target type conditions
            source_conditions = format_types_for_cypher(source_neurons)
            target_conditions = format_types_for_cypher(target_neurons)
            
            # Also include intermediate types (neurons that connect source to target)
            # We want edges where:
            # 1. pre is source type and post is any type (outgoing from sources)
            # 2. pre is any type and post is target type (incoming to targets)
            # 3. edges between any neurons that could be intermediates
            
            # For simplicity, query edges involving source or target types
            source_cond_str = ' OR '.join([c.replace('a.type', 'pre.type') for c in source_conditions]) if source_conditions else 'false'
            target_cond_str = ' OR '.join([c.replace('a.type', 'post.type') for c in target_conditions]) if target_conditions else 'false'
            
            # Query edges where pre is source-like OR post is target-like
            # This captures the relevant subgraph
            query = f"""
            MATCH (pre:Neuron)-[c:ConnectsTo]->(post:Neuron)
            WHERE c.weight >= {min_weight}
            AND (({source_cond_str}) OR ({target_cond_str}))
            AND pre.type IS NOT NULL AND post.type IS NOT NULL
            RETURN pre.bodyId AS bodyId_pre, pre.type AS type_pre,
                   post.bodyId AS bodyId_post, post.type AS type_post,
                   c.weight AS weight
            """
            
            with tqdm(total=1, desc=f"  ⏳ Querying NeuPrint for {dataset_name}",
                      bar_format='{desc}...', leave=False) as pbar:
                result = client.fetch_custom(query)
                pbar.update(1)
            
            if not result.empty:
                self._log(f"Queried {len(result)} edges from NeuPrint for {dataset_name}")
            
            return result
            
        except Exception as e:
            self._log(f"Warning: Failed to query NeuPrint edges for {dataset_name}: {e}")
            return pd.DataFrame()
    
    def _query_edges_local(
        self,
        dataset_name: str,
        source_neurons: List,
        target_neurons: List,
        min_weight: int = 1
    ) -> pd.DataFrame:
        """Query edges from exact FAFB or standalone BANC local files."""
        import re
        
        # Load local connection data
        safe_name = self.parameters._sanitize_name(dataset_name)
        datasets_folder = self._get_datasets_folder()
        
        # Try different file patterns for connections
        conn_files = [
            os.path.join(datasets_folder, safe_name, f'{safe_name}_merged_connections.parquet'),
            os.path.join(datasets_folder, safe_name, f'{safe_name}_merged_connections.csv'),
            os.path.join(datasets_folder, safe_name, f'{safe_name}_connections.parquet'),
            os.path.join(datasets_folder, safe_name, f'{safe_name}_connections.csv'),
            os.path.join(datasets_folder, safe_name, 'connections.parquet'),
            os.path.join(datasets_folder, safe_name, 'connections.csv'),
        ]
        
        conn_df = None
        for conn_file in conn_files:
            if os.path.exists(conn_file):
                try:
                    file_size_mb = os.path.getsize(conn_file) / (1024 * 1024)
                    with tqdm(total=1, desc=f"  ⏳ Loading connections ({file_size_mb:.1f} MB)", 
                              bar_format='{desc}', leave=False) as pbar:
                        if conn_file.endswith('.parquet'):
                            conn_df = pd.read_parquet(conn_file)
                        else:
                            conn_df = self._read_csv(conn_file)
                        pbar.update(1)
                    self._log(f"Loaded connections from {conn_file}")
                    break
                except Exception as e:
                    self._log(f"Warning: Could not load {conn_file}: {e}")
        
        if conn_df is None or conn_df.empty:
            self._log(f"Warning: No connection data found for {dataset_name}")
            return pd.DataFrame()
        
        # Standardize column names
        col_mapping = {
            'pre_pt_root_id': 'bodyId_pre',
            'post_pt_root_id': 'bodyId_post',
            'pre_type': 'type_pre',
            'post_type': 'type_post',
            'syn_count': 'weight',
            'neuropil': 'roi'
        }
        conn_df = conn_df.rename(columns={k: v for k, v in col_mapping.items() if k in conn_df.columns})
        
        # Filter by weight
        if 'weight' in conn_df.columns:
            conn_df = conn_df[conn_df['weight'] >= min_weight]
        
        # If type columns don't exist, join with neuron info to get them
        if 'type_pre' not in conn_df.columns or 'type_post' not in conn_df.columns:
            # Load neuron info file
            neuron_files = [
                os.path.join(datasets_folder, safe_name, f'{safe_name}_allneurons_neuron_df.parquet'),
                os.path.join(datasets_folder, safe_name, f'{safe_name}_allneurons_neuron_df.csv'),
                os.path.join(datasets_folder, safe_name, f'{safe_name}_neurons.parquet'),
                os.path.join(datasets_folder, safe_name, f'{safe_name}_neurons.csv'),
            ]
            
            neuron_df = None
            for neuron_file in neuron_files:
                if os.path.exists(neuron_file):
                    try:
                        if neuron_file.endswith('.parquet'):
                            neuron_df = pd.read_parquet(neuron_file)
                        else:
                            neuron_df = self._read_csv(neuron_file)
                        self._log(f"Loaded neuron info from {neuron_file}")
                        break
                    except Exception as e:
                        self._log(f"Warning: Could not load {neuron_file}: {e}")
            
            if neuron_df is not None and not neuron_df.empty:
                # Identify bodyId and type columns in neuron_df
                bodyid_col = None
                type_col = None
                for col in ['bodyId', 'root_id', 'pt_root_id', 'segment_id']:
                    if col in neuron_df.columns:
                        bodyid_col = col
                        break
                for col in ['type', 'cell_type', 'hemibrain_type']:
                    if col in neuron_df.columns:
                        type_col = col
                        break
                
                if bodyid_col and type_col:
                    # Create mapping dict for efficiency
                    type_map = dict(zip(neuron_df[bodyid_col], neuron_df[type_col]))
                    
                    # Map types to connections with progress bar
                    with tqdm(total=2, desc="  ⏳ Mapping bodyId → type", 
                              bar_format='{desc}: {n}/{total} columns', leave=False) as pbar:
                        if 'type_pre' not in conn_df.columns:
                            conn_df['type_pre'] = conn_df['bodyId_pre'].map(type_map)
                        pbar.update(1)
                        if 'type_post' not in conn_df.columns:
                            conn_df['type_post'] = conn_df['bodyId_post'].map(type_map)
                        pbar.update(1)
                    
                    self._log(f"Mapped types for {len(conn_df)} connections")
        
        # Filter by source/target types if type columns exist
        if 'type_pre' in conn_df.columns and 'type_post' in conn_df.columns:
            def matches_patterns(type_val, patterns):
                if pd.isna(type_val):
                    return False
                for p in patterns:
                    if isinstance(p, str):
                        if '.*' in p or '*' in p:
                            pattern = _wildcard_pattern_to_regex(p)
                            if re.match(pattern, str(type_val)):
                                return True
                        elif str(type_val) == p:
                            return True
                return False
            
            # Keep edges where pre matches source OR post matches target
            with tqdm(total=2, desc="  ⏳ Filtering by source/target types",
                      bar_format='{desc}: {n}/{total} masks', leave=False) as pbar:
                mask_pre = conn_df['type_pre'].apply(lambda x: matches_patterns(x, source_neurons))
                pbar.update(1)
                mask_post = conn_df['type_post'].apply(lambda x: matches_patterns(x, target_neurons))
                pbar.update(1)
            mask = mask_pre | mask_post
            conn_df = conn_df[mask]
        
        self._log(f"Filtered to {len(conn_df)} edges for {dataset_name}")
        return conn_df
    
    def _reset_mapping_state(self) -> None:
        """Reset per-run canonicalization bookkeeping.

        Called at the start of each analysis run so a reused analyzer owns
        exactly its own merge-key resolutions/caches (no accumulation into
        the exported metadata and no stale decisions across runs).
        """
        self._mapper_snapshot = None
        self._merge_key_cache = {}
        self._mapping_status_counts = {}
        self._conflicted_merge_types = {}
        self._type_coverage_cache = None

    def run_all_analyses(self, skip_existing: bool = True) -> Dict[str, Dict[int, pd.DataFrame]]:
        """
        Run analysis for all datasets and thresholds.
        
        Uses comparison_mode from parameters:
        - 'path': Path-based analysis using FindAllPath()
        - 'edge': Edge-based analysis querying edges directly
        
        Args:
            skip_existing: Skip if results already cached
            
        Returns:
            Nested dict {dataset_name: {threshold: DataFrame}}
        """
        self._reset_mapping_state()
        mode = self.parameters.comparison_mode
        self._log(f"Starting analysis across all datasets and thresholds (mode={mode})")
        
        dataset_names = self.parameters.get_dataset_names()

        # Auto threshold-density alignment (plan §5 Phase D): measure the
        # per-dataset windows with ONE bootstrap enumeration per dataset at
        # the requested floor, then install the density-aligned combination
        # rows so the rest of the pipeline runs through the combination
        # engine unchanged.
        if (mode != 'edge'
                and getattr(self.parameters, 'threshold_mode', '') == 'auto'
                and self.parameters.path_mode == 'all'):
            if not self._bootstrap_auto_mode():
                self._log("Auto threshold mode: bootstrap unavailable — "
                          "running the requested thresholds as-is.")
        
        if mode == 'edge':
            return self._run_all_edge_analyses(skip_existing)
        else:
            return self._run_all_path_analyses(skip_existing)
    
    # ------------------------------------------------------------------
    # Feature G helpers: per-threshold run-meta persistence (resume)
    # ------------------------------------------------------------------

    def _threshold_meta_path(self, dataset_name: str) -> str:
        safe_name = self.parameters._sanitize_name(dataset_name)
        return os.path.join(
            self.parameters.full_output_path, 'dataset_data',
            safe_name, 'threshold_meta.json')

    def _load_threshold_meta(self, dataset_name: str) -> Dict[int, Dict]:
        """Load persisted per-threshold run state ({tau, budget_bitten,
        skipped, duplicate_of, ...}) so a resumed run reconstructs the
        skip state instead of re-running collapsed thresholds (G §13.5
        item 6)."""
        meta_path = self._threshold_meta_path(dataset_name)
        if not os.path.exists(meta_path):
            return {}
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            return {int(k): v for k, v in raw.items()}
        except Exception as e:
            self._log(f"Warning: could not load threshold meta for "
                      f"{dataset_name}: {e}")
        return {}

    def _store_threshold_meta(self, dataset_name: str, meta_map: Dict[int, Dict]) -> None:
        meta_path = self._threshold_meta_path(dataset_name)
        try:
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in meta_map.items()}, f,
                          indent=2, default=str)
        except Exception as e:
            self._log(f"Warning: could not persist threshold meta for "
                      f"{dataset_name}: {e}")

    def _effective_tau(self, meta: Optional[Dict]) -> Optional[float]:
        """Effective threshold of a run: the budget τ when the run was
        budget-bitten, else its natural τ (min emitted bottleneck). None
        = unknown/no paths (never skip)."""
        if not meta:
            return None
        return meta.get('tau')

    def _g_skip_allowed(self, dataset_name: str, threshold: int,
                        prev_meta: Optional[Dict]) -> bool:
        """Feature G duplicate-threshold skip rule (§13.1/§13.5).

        'all' mode only, path comparison mode only. Skip iff the previous
        actual run of this dataset collapsed everything up to its
        effective τ >= this threshold (identical path sets by the
        τ-equivalence), the algorithm is unchanged, and the threshold was
        not already materialized independently. Shortest mode never skips
        (min-hop sets are NOT nested across thresholds — §13.5 item 1).
        """
        if self.parameters.path_mode != 'all':
            return False
        if not prev_meta:
            return False
        if prev_meta.get('skipped'):
            return False  # prev was itself aliased; its τ is carried by an earlier run
        if prev_meta.get('tau') is None:
            return False
        if threshold > prev_meta['tau']:
            return False  # new material exists above τ
        if prev_meta.get('pathfinding') != self.parameters.pathfinding:
            return False  # paranoia clause: only skip identical algorithm (§13.5 item 8)
        return True

    def _is_duplicate_threshold(self, dataset_name: str, threshold: int) -> bool:
        """Feature G: True when this input threshold was skipped as a
        τ-collapse duplicate for this dataset (its results alias an
        earlier run)."""
        meta = self._path_run_meta.get((dataset_name, threshold))
        return bool(meta and meta.get('skipped'))

    def get_threshold_queries(self) -> List[Dict[str, Any]]:
        """Return the normalized comparison query rows for this run."""
        getter = getattr(self.parameters, 'get_threshold_queries', None)
        if callable(getter):
            return getter()
        datasets = self.parameters.get_dataset_names()
        if getattr(self.parameters, 'threshold_mode', 'standard') == 'combinations':
            rows = getattr(self.parameters, 'threshold_combinations', None) or []
            return [
                {
                    'id': str(row.get('id') or row.get('query_id')
                             or f'combo_{index:03d}'),
                    'label': row.get('label') or row.get('query_label')
                             or f'Combination {index}',
                    'thresholds': dict(row.get('thresholds') or row.get(
                        'thresholds_by_dataset', {})),
                }
                for index, row in enumerate(rows, start=1)
            ]
        return [
            {
                'id': f'threshold_{int(threshold)}',
                'label': f'N={int(threshold)}',
                'thresholds': {dataset: int(threshold) for dataset in datasets},
            }
            for threshold in (getattr(self.parameters, 'thresholds', []) or [])
        ]

    def get_comparison_points(self) -> List[ComparisonPoint]:
        """Return the canonical report/export points in display order."""
        return points_from_parameters(self.parameters)

    def _comparison_point(self, value: Any) -> ComparisonPoint:
        """Resolve a scalar threshold, query ID, or point object."""
        return point_from_value(value, self.parameters, self.get_comparison_points())

    def _comparison_point_thresholds(self, value: Any) -> Dict[str, int]:
        return dict(self._comparison_point(value).thresholds_by_dataset)

    def _comparison_point_label(self, value: Any) -> str:
        return self._comparison_point(value).display_label

    def _comparison_point_metadata(self, value: Any) -> Dict[str, Any]:
        """Return query-aware labels and provenance for visualization exports."""
        point = self._comparison_point(value)
        requested = dict(point.thresholds_by_dataset)
        applied = {}
        provenance = {}
        for dataset, threshold in requested.items():
            row = self._path_provenance_row(dataset, threshold)
            provenance[dataset] = row
            applied[dataset] = row.get('applied_threshold')
        return {
            'point_id': point.point_id,
            'query_id': point.query_id,
            'query_label': point.display_label,
            'threshold_scope': 'query' if point.mode == 'combinations' else 'scalar',
            'requested_thresholds': requested,
            'applied_thresholds': applied,
            'provenance': provenance,
        }

    def _query_record(self, query_or_id: Any) -> Dict[str, Any]:
        """Resolve a query object, ID, or standard scalar threshold."""
        if isinstance(query_or_id, dict):
            query_id = query_or_id.get('id') or query_or_id.get('query_id')
            if not query_id:
                raise ValueError("A threshold query must have an id")
            return query_or_id
        if isinstance(query_or_id, str):
            for query in self.get_threshold_queries():
                if query.get('id') == query_or_id:
                    return query
            raise KeyError(f"Unknown threshold query: {query_or_id}")
        try:
            scalar = int(query_or_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid threshold query: {query_or_id!r}") from exc
        matches = [
            query for query in self.get_threshold_queries()
            if len(set(query.get('thresholds', {}).values())) == 1
            and next(iter(query.get('thresholds', {}).values()), None) == scalar
        ]
        if matches:
            return matches[0]
        if self.parameters.threshold_mode == 'combinations':
            raise ValueError(
                "A scalar threshold is not a valid key in advanced "
                "combination mode; pass a query id or query record"
            )
        return {
            'id': f'threshold_{scalar}',
            'label': f'N={scalar}',
            'thresholds': {
                dataset: scalar
                for dataset in self.parameters.get_dataset_names()
            },
            'dataset_order': self.parameters.get_dataset_names(),
        }

    def _threshold_for_dataset(self, threshold_or_query: Any,
                               dataset: str) -> Any:
        """Resolve the threshold used by one dataset in a query-aware call."""
        if isinstance(threshold_or_query, dict):
            return (threshold_or_query.get('thresholds', {}) or {}).get(dataset)
        if (isinstance(threshold_or_query, str)
                and self.parameters.threshold_mode == 'combinations'):
            query = self._query_record(threshold_or_query)
            return (query.get('thresholds', {}) or {}).get(dataset)
        return threshold_or_query

    def get_aligned_data_for_query(self, query_or_id: Any) -> pd.DataFrame:
        """Align raw results using the threshold map of one query row."""
        query = self._query_record(query_or_id)
        query_id = query.get('id') or query.get('query_id')
        if query_id in self.query_aligned_results:
            return self.query_aligned_results[query_id]

        dataset_names = self.parameters.get_dataset_names()
        threshold_map = query.get('thresholds') or query.get(
            'thresholds_by_dataset', {})
        type_mapper = (
            self.parameters._auto_type_mapper
            if self.parameters.auto_type_mapping else None
        )
        aligned = self.metrics._align_results_for_threshold_map(
            self.raw_results,
            dataset_names,
            threshold_map,
            label_mapper=self._policy_label_mapper(),
            type_mapper=type_mapper,
            merge_policy=self._merge_policy_or_none(),
        )
        if self.parameters.keep_only_hemisphere_conserved_connections:
            aligned = self._filter_hemisphere_unconserved(
                aligned,
                dataset_names,
                f"query_{query_id}",
            )
        self.query_aligned_results[query_id] = aligned
        return aligned

    def _analysis_thresholds(self) -> List[int]:
        """Input thresholds minus Feature-G duplicates — the list the
        heavy per-threshold analysis, rendering, and exports iterate.

        A threshold is dropped only when EVERY dataset skipped it (its
        results alias an earlier run); if any dataset ran fresh, the
        threshold still carries new material for the combined exports."""
        dataset_names = self.parameters.get_dataset_names()
        out = []
        for t in self.parameters.thresholds:
            if any(not self._is_duplicate_threshold(ds, t)
                   for ds in dataset_names):
                out.append(t)
        return out

    def _applied_state_for(self, dataset: str, threshold: int):
        """F5/F6 + canonical tau: (applied_threshold, pruned,
        edge_weight_floor, applied_threshold_source) for one (dataset,
        threshold) from the run meta.

        Delegates to the shared canonical formula
        (``utils.threshold_state.applied_threshold_provenance`` — the same
        one the per-dataset folders' parameters.txt reports), so summary
        rows and folders can never diverge:

        - applied_threshold is the CANONICAL (minimal) threshold that
          reproduces this run's output: w2 + 1 for a budget-bitten run
          (w2 = strongest dropped path bottleneck; the gap (w2, tau]
          contains no paths, so every threshold in [w2+1, tau] yields the
          identical set), the natural tau for a floored-but-unbitten run
          (a lossy floor RAISES the effective cutoff — at least w0), and
          the asked threshold for complete runs.
        - ``applied_threshold_source`` names the contributing
          mechanism(s): 'requested' | 'strongest_first_budget' |
          'edge_budget' | 'strongest_first_budget+edge_budget'.
        """
        meta = self._path_run_meta.get((dataset, threshold)) or {}
        prov = self._path_provenance_from_state(threshold, meta)
        floor = prov.get('edge_weight_floor')
        tau = prov.get('strongest_first_tau')
        canonical = prov.get('tau_canonical')
        complete = bool(prov.get('paths_complete', True))
        applied_folder = meta.get('applied_folder')

        if tau is not None and not complete \
                and meta.get('tau_canonical') is None \
                and meta.get('skipped') and applied_folder is not None:
            # Collapsed row with missing canonical bookkeeping: the
            # applied folder IS the materialized equivalent threshold —
            # never fall back to the bare landing tau here.
            return (int(applied_folder), floor is not None, floor,
                    'strongest_first_budget')

        applied = prov['applied_threshold']
        if applied is None:
            # Degenerate meta (no tau / no floor): fall back to the asked
            # threshold rather than reporting None.
            applied = threshold
        return applied, prov['edge_budget_applied'], floor, \
            prov['applied_threshold_source']

    # ------------------------------------------------------------------
    # Threshold view-model: one resolved requested→applied view shared by
    # the exporter, report and used-data writers (plan §2).
    # ------------------------------------------------------------------

    def _dataset_folder_index(self, dataset: str) -> Dict[int, str]:
        """Map ``requested_threshold -> folder name`` for a dataset on disk.

        Built by reading each data folder's own ``all_attributes.json``
        ``requested_threshold``. This lets a LATER analysis (fresh process,
        no in-memory meta) resolve the new grammar folder names
        (``..._equal_applied_floor`` / ``..._applied_floor``) as well as
        historical ``minsyn_{N}`` folders. Cached per dataset.
        """
        cache = getattr(self, '_dataset_folder_index_cache', None)
        if cache is None:
            cache = {}
            self._dataset_folder_index_cache = cache
        if dataset in cache:
            return cache[dataset]
        index: Dict[int, str] = {}
        safe = self.parameters._sanitize_name(dataset)
        base = os.path.join(self.parameters.dataset_data_path, safe)
        if os.path.isdir(base):
            for name in os.listdir(base):
                folder = os.path.join(base, name)
                if not os.path.isdir(folder) or not name.startswith('minsyn_'):
                    continue
                if name.endswith('_skipped'):
                    continue
                req = self._folder_requested_threshold(folder)
                if req is not None and req not in index:
                    index[int(req)] = name
        cache[dataset] = index
        return index

    def _resolve_dataset_output_path(self, dataset: str,
                                     threshold: int) -> str:
        """On-disk path for a (dataset, threshold) cell, honoring the
        applied-folder naming grammar — disk-aware, so later analysis can
        find the folder without an in-memory lookup.

        Resolution order: registered lookup -> the run-meta applied folder's
        grammar name -> the dataset's on-disk requested->folder index ->
        plain ``minsyn_{threshold}``. Always returns an existing directory
        when one exists.
        """
        safe = self.parameters._sanitize_name(dataset)
        base = os.path.join(self.parameters.dataset_data_path, safe)
        requested = self.parameters.get_thresholds_for_dataset(dataset)

        def _existing(name):
            if not name:
                return None
            path = os.path.join(base, name)
            return path if os.path.isdir(path) else None

        # 1. Registered lookup (this process's reconcile/run).
        lookup = getattr(self.parameters, '_applied_folder_lookup', None) or {}
        name = lookup.get((dataset, int(threshold)))
        found = _existing(name)
        if found:
            return found
        # 2. In-memory meta -> applied folder grammar name.
        meta = self._path_run_meta.get((dataset, int(threshold)))
        if meta:
            applied = self.get_applied_folder(dataset, int(threshold))
            if applied is not None:
                name = self._applied_folder_name(dataset, applied)
                found = _existing(name)
                if found:
                    return found
        # 3. Disk index by the folder's own requested_threshold.
        name = self._dataset_folder_index(dataset).get(int(threshold))
        found = _existing(name)
        if found:
            return found
        # 4. Try grammar candidates for the threshold value itself.
        for candidate in self.parameters.applied_folder_name_candidates(
                int(threshold)):
            found = _existing(candidate)
            if found:
                return found
        # 5. Nothing on disk: return the grammar path a writer would use.
        applied = None
        if meta:
            applied = self.get_applied_folder(dataset, int(threshold))
        if applied is not None:
            return os.path.join(
                base, self._applied_folder_name(dataset, applied))
        return self.parameters.get_dataset_output_path(dataset, threshold)

    def _reconcile_applied_folders(self) -> None:
        """Rename materialized folders to the applied grammar, create
        ``_skipped`` markers for pruned thresholds, and write the
        per-dataset APPLIED_THRESHOLDS.md note.

        Non-destructive: data is only ever renamed (never deleted); the
        source of truth is each folder's own all_attributes.json
        ``applied_threshold``, not its name. Skipped requested thresholds
        get a marker folder containing a README that explains the applied
        target and reason.
        """
        out_root = getattr(self.parameters, 'full_output_path', None)
        if not out_root:
            return
        manifest = {}
        for dataset in self.parameters.get_dataset_names():
            requested = list(
                self.parameters.get_thresholds_for_dataset(dataset))
            if not requested:
                continue
            safe = self.parameters._sanitize_name(dataset)
            base = os.path.join(
                getattr(self.parameters, 'dataset_data_path',
                        os.path.join(out_root, 'dataset_data')), safe)
            if not os.path.isdir(base):
                continue
            actions = []
            # requested threshold -> canonical (applied) threshold, from the
            # provenance row (never the raw folder id).
            applied_by_requested: Dict[int, Optional[int]] = {}
            for t in requested:
                row = self._path_provenance_row(dataset, int(t))
                value = row.get('applied_threshold')
                applied_by_requested[int(t)] = (
                    int(value) if value is not None else None)

            # A level is a "collapse floor" when at least one requested
            # threshold OTHER than the level itself aliases to it. Such a
            # level is stored as `minsyn_{L}_applied_floor`; a level that
            # only its own requested run occupies keeps the bare
            # `minsyn_{L}`. (A requested level that coincides with a floor it
            # aliases to is served by that floor folder — no separate marker.)
            floor_values = {
                applied_by_requested[int(t)]
                for t in requested
                if applied_by_requested.get(int(t)) is not None
                and applied_by_requested[int(t)] != int(t)
            }

            def _grammar(applied):
                return self.parameters.applied_folder_name(
                    applied, requested, is_floor=int(applied) in floor_values)

            # On-disk data folders (skip markers) grouped by their own
            # recorded applied value; each folder also carries its run's
            # requested threshold (min_synapse_num).
            existing = {}
            for name in os.listdir(base):
                folder = os.path.join(base, name)
                if not os.path.isdir(folder) or name.endswith('_skipped'):
                    continue
                if not name.startswith('minsyn_'):
                    continue
                applied = self._folder_applied_threshold(folder)
                if applied is None:
                    continue
                run_requested = self._folder_requested_threshold(folder)
                existing.setdefault(int(applied), []).append(
                    (name, folder, run_requested))

            # Distinct applied values = every requested threshold's applied
            # plus any folder's recorded applied (legacy/resume).
            applied_values = sorted(
                {a for a in applied_by_requested.values() if a is not None}
                | set(existing))

            for applied in applied_values:
                target_name = _grammar(applied)
                target_path = os.path.join(base, target_name)
                group = existing.get(applied, [])
                # Keep the folder whose OWN run requested this applied value
                # (the exact run); else the first. Never prefer a folder
                # merely because its name already matches the grammar.
                keep = None
                for entry in group:
                    if entry[2] == applied:
                        keep = entry
                        break
                if keep is None and group:
                    keep = group[0]

                if keep is not None and os.path.abspath(
                        keep[1]) != os.path.abspath(target_path):
                    # Clear the target slot if a redundant folder occupies
                    # it (its requested threshold gets a marker below), then
                    # move the correct run into the grammar name.
                    if os.path.isdir(target_path):
                        for entry in group:
                            if os.path.abspath(entry[1]) == \
                                    os.path.abspath(target_path) and \
                                    entry is not keep:
                                rq = entry[2] if entry[2] is not None \
                                    else applied
                                self._replace_with_marker(
                                    dataset, int(rq), applied, entry[1])
                                actions.append(
                                    (entry[0],
                                     f'minsyn_{rq}_skipped',
                                     'marker+quarantined'))
                                break
                    try:
                        os.rename(keep[1], target_path)
                        actions.append((keep[0], target_name, 'renamed'))
                        keep = (target_name, target_path, keep[2])
                    except OSError as e:
                        self._log(
                            f"  Warning: could not rename {keep[0]} → "
                            f"{target_name}: {e}")

                # Any remaining data folder for this applied value holds the
                # same set — replace it with a marker for its requested
                # threshold.
                for entry in group:
                    if keep is not None and os.path.abspath(entry[1]) == \
                            os.path.abspath(keep[1]):
                        continue
                    if not os.path.isdir(entry[1]):
                        continue  # already consumed above
                    run_requested = entry[2]
                    if run_requested is None:
                        run_requested = applied
                    self._replace_with_marker(
                        dataset, int(run_requested), applied, entry[1])
                    actions.append(
                        (entry[0], f'minsyn_{run_requested}_skipped',
                         'marker+quarantined'))
                # Register the lookup for the applied value and every
                # requested threshold that resolves to it.
                self.parameters.set_applied_folder_lookup(
                    dataset, applied, target_name)
                for t in requested:
                    if applied_by_requested.get(int(t)) == applied:
                        self.parameters.set_applied_folder_lookup(
                            dataset, int(t), target_name)

            # Normalize each requested row's meta to its TRUE on-disk state:
            # the folder that holds its data is the applied value's folder.
            # A requested threshold whose applied value differs aliases to it
            # and gets a marker. A requested threshold that COINCIDES with
            # its applied floor (served by `minsyn_{L}_applied_floor`) is
            # skipped SILENTLY — no marker, since the floor folder already
            # represents it explicitly.
            for t in requested:
                applied = applied_by_requested.get(int(t))
                meta = self._path_run_meta.get((dataset, int(t)))
                if applied is None:
                    continue
                coincident = int(applied) == int(t)
                aliased = not coincident
                if meta is not None:
                    meta['applied_folder'] = applied
                    meta['skipped'] = bool(aliased)
                    meta['duplicate_of'] = applied if aliased else None
                if aliased:
                    self._write_skipped_marker(
                        dataset, t, applied_by_requested)
            # APPLIED_THRESHOLDS.md note.
            self._write_applied_thresholds_note(dataset, requested)
            manifest[dataset] = actions
        self._log(
            "Applied-folder reconcile complete "
            f"({sum(len(v) for v in manifest.values())} action(s); "
            "one data folder per applied value; markers refreshed)")

    def _folder_requested_threshold(self, folder: str) -> Optional[int]:
        """The requested threshold a run's folder was enumerated at."""
        attrs_path = os.path.join(folder, 'all_attributes.json')
        if os.path.exists(attrs_path):
            try:
                with open(attrs_path, encoding='utf-8') as f:
                    attrs = json.load(f)
                value = attrs.get('requested_threshold')
                if value is None:
                    value = attrs.get('min_synapse_num')
                if value is not None:
                    return int(value)
            except Exception:
                pass
        return None

    def _replace_with_marker(self, dataset: str, requested: int,
                             applied: Optional[int], folder: str) -> None:
        """Retire a redundant data folder (same applied set as the kept one)
        under its ``_skipped`` marker — RENAME ONLY, never delete.

        Plan §10 (cross-dataset-threshold-display-coordination, user-CONFIRMED):
        "data is never deleted, only renamed … the only irreversible action
        is a rename".  The folder's data moves to ``_replaced/`` beside the
        marker (the reconcile/data-loader scans key on the ``minsyn_``
        prefix, so a quarantined name is invisible to them) — a mis-grouped
        "duplicate" (the §4 mislabeled-folder bugs) must cost a rename-back,
        not a re-capture."""
        self._write_skipped_marker(
            dataset, requested, {requested: applied})
        if not os.path.isdir(folder):
            return
        import time as _time
        folder = folder.rstrip('/') or folder
        base = os.path.dirname(folder) or '.'
        quarantine = os.path.join(base, '_replaced')
        target = os.path.join(
            quarantine,
            f"{os.path.basename(folder)}_{_time.strftime('%Y%m%d_%H%M%S')}")
        try:
            os.makedirs(quarantine, exist_ok=True)
            os.rename(folder, target)
            self._log(
                f"  Reconcile: {os.path.basename(folder)} duplicates the "
                f"applied set — data kept at "
                f"{os.path.relpath(target, base)} beside the skipped marker "
                f"(nothing is deleted)")
        except OSError as e:
            # Bookkeeping must never fail a run: the marker is already
            # written, so the folder simply stays where it is.
            self._log(
                f"  Warning: could not quarantine redundant "
                f"{os.path.basename(folder)}: {e}")


    def _folder_applied_threshold(self, folder: str) -> Optional[int]:
        attrs_path = os.path.join(folder, 'all_attributes.json')
        if os.path.exists(attrs_path):
            try:
                with open(attrs_path, encoding='utf-8') as f:
                    attrs = json.load(f)
                value = attrs.get('applied_threshold')
                if value is None:
                    value = attrs.get('min_synapse_num')
                if value is not None:
                    return int(value)
            except Exception:
                pass
        name = os.path.basename(folder)
        suffix = name[len('minsyn_'):].split('_')[0]
        return int(suffix) if suffix.isdigit() else None

    def _write_skipped_marker(self, dataset: str, requested: int,
                              applied_by_requested: Dict) -> None:
        marker = self.parameters.get_skipped_output_path(dataset, requested)
        try:
            os.makedirs(marker, exist_ok=True)
        except OSError:
            return
        applied = applied_by_requested.get(int(requested))
        meta = self._path_run_meta.get((dataset, int(requested)), {}) or {}
        row = self._path_provenance_row(dataset, int(requested))
        reason = self._skip_reason(meta, row)
        target = (self._applied_folder_name(dataset, applied)
                  if applied is not None else '—')
        text = (
            f"Skipped threshold: requested Min Synapse Count = {requested}\n"
            f"Applied threshold for this dataset: "
            f"{row.get('applied_threshold')}   "
            f"(data folder: {target})\n"
            f"Reason: {reason}\n"
            f"Applied-threshold source: "
            f"{row.get('applied_threshold_source')}\n"
            f"Paths complete: {row.get('paths_complete')}\n"
            f"See also: ../APPLIED_THRESHOLDS.md\n")
        try:
            with open(os.path.join(marker, 'README.txt'), 'w',
                      encoding='utf-8') as f:
                f.write(text)
        except OSError:
            pass

    @staticmethod
    def _skip_reason(meta: Dict, row: Dict) -> str:
        tau = row.get('strongest_first_tau')
        if meta.get('duplicate_of') is not None and tau is not None:
            return (f"StrongestFirst path budget bit at tau={tau}; all intact "
                    f"paths with bottleneck >= {tau} were retained. The "
                    f"requested threshold is <= tau, so it reproduces the "
                    f"identical materialized path set and needs no separate "
                    f"run or data.")
        if row.get('edge_budget_applied'):
            return (f"The Edge Budget floor ({row.get('edge_weight_floor')}) "
                    "removed edges this threshold requires; the materialized "
                    "set is the floored run and this threshold has no "
                    "distinct output.")
        return ("This requested threshold was pruned/aliased to the applied "
                "threshold; its data is the applied folder.")

    def _write_applied_thresholds_note(self, dataset: str,
                                       requested: List[int]) -> None:
        base = os.path.join(
            self.parameters.dataset_data_path,
            self.parameters._sanitize_name(dataset))
        lines = [f"# Applied thresholds — {dataset}", ""]
        data_folders = []
        markers = []
        for t in requested:
            row = self._path_provenance_row(dataset, int(t))
            applied = row.get('applied_threshold')
            applied_int = int(applied) if applied is not None else int(t)
            folder = self._applied_folder_name(dataset, applied_int)
            data_folders.append(folder)
            meta = self._path_run_meta.get((dataset, int(t)), {}) or {}
            skipped = bool(meta.get('skipped'))
            if skipped:
                marker = self.parameters.skipped_folder_name(t)
                markers.append(marker)
                lines.append(
                    f"requested {t:<3} -> applied {applied_int:<3} "
                    f"({row.get('applied_threshold_source')}; aliased to "
                    f"{folder}; marker: {marker})")
            else:
                lines.append(
                    f"requested {t:<3} -> applied {applied_int:<3} "
                    f"({row.get('applied_threshold_source')}; data: "
                    f"{folder})")
        # List the actual on-disk data folders so the note can never point
        # at a folder that does not exist.
        disk = sorted(
            n for n in os.listdir(base)
            if n.startswith('minsyn_') and not n.endswith('_skipped')
            and os.path.isdir(os.path.join(base, n)))
        lines.append("")
        lines.append("data folders on disk:  " + (', '.join(disk) or '—'))
        lines.append("referenced data folders:  " + (
            ', '.join(sorted(set(data_folders))) or '—'))
        lines.append("skipped markers: " + (
            ', '.join(sorted(set(markers))) or '—'))
        lines.append("")
        try:
            path = os.path.join(base, 'APPLIED_THRESHOLDS.md')
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            self._log(f"Saved: dataset_data/"
                      f"{self.parameters._sanitize_name(dataset)}/"
                      f"APPLIED_THRESHOLDS.md")
        except OSError as e:
            self._log(f"Warning: could not write APPLIED_THRESHOLDS.md: {e}")

    def get_applied_threshold_map(self) -> Dict[str, Any]:
        """Requested threshold -> {dataset: applied threshold} for labels.

        The report/UI render this alongside the requested value so a section
        titled ``t=5`` can state that it actually compared applied
        19/6/17, and a fully-aliased request can be marked as such.
        """
        datasets = self.parameters.get_dataset_names()
        out: Dict[str, Any] = {}
        for t in getattr(self.parameters, 'thresholds', None) or []:
            row = {}
            for ds in datasets:
                if t not in self.parameters.get_thresholds_for_dataset(ds):
                    continue
                view = self.get_threshold_view(ds, int(t))
                row[ds] = {
                    'applied': view['applied_threshold'],
                    'status': view['status'],
                }
            out[str(t)] = row
        return out

    def _applied_folder_is_floor(self, dataset: str, applied: int) -> bool:
        """True when ``applied`` is a collapse floor: some requested
        threshold resolves to it at a DIFFERENT value (aliases to it), so the
        folder name carries the ``_applied_floor`` suffix."""
        for t in self.parameters.get_thresholds_for_dataset(dataset):
            row = self._path_provenance_row(dataset, int(t))
            value = row.get('applied_threshold')
            if value is not None and int(value) == int(applied) \
                    and int(t) != int(applied):
                return True
        return False

    def _applied_folder_name(self, dataset: str, applied: int) -> str:
        """Resolved on-disk folder name for an applied value."""
        requested = self.parameters.get_thresholds_for_dataset(dataset)
        return self.parameters.applied_folder_name(
            applied, requested,
            is_floor=self._applied_folder_is_floor(dataset, applied))

    def resolve_query_inputs(self) -> List[Dict[str, Any]]:
        """Structured per-token/per-dataset resolution of the query inputs.

        Uses the standalone ``query_resolver`` (same backend as the auto
        type mapper) so the report can show the method/status/confidence of
        every source/target token — including an explicit low-confidence
        same-name fallback — instead of an invisible pass-through.  With
        auto mapping active, taxonomy-column values (e.g. FAFB
        ``cell_type=circadian_clock``) expand per dataset and bridge into
        the other datasets through member mapping (Route A).
        """
        try:
            from .query_resolver import (
                DatasetTaxonomyResolver, resolve_query_tokens,
            )
        except ImportError:  # pragma: no cover
            from query_resolver import (
                DatasetTaxonomyResolver, resolve_query_tokens,
            )

        mapper = getattr(self.parameters, '_auto_type_mapper', None)
        if not getattr(self.parameters, 'auto_type_mapping', False):
            mapper = None
        datasets = self.parameters.get_dataset_names()
        taxonomy_resolver = None
        if mapper is not None:
            taxonomy_resolver = DatasetTaxonomyResolver(mapper).resolve
        records: List[Dict[str, Any]] = []
        records.extend(resolve_query_tokens(
            list(self.parameters.source_neurons or []), datasets, mapper,
            role='source', source_dataset=self.parameters.source_dataset,
            taxonomy_resolver=taxonomy_resolver))
        records.extend(resolve_query_tokens(
            list(self.parameters.target_neurons or []), datasets, mapper,
            role='target', source_dataset=self.parameters.source_dataset,
            taxonomy_resolver=taxonomy_resolver))
        self._query_resolution_records = records
        return records

    # ------------------------------------------------------------------
    # Query-anchored merge policy (plan:
    # plan-query-anchored-cross-dataset-analysis.md)
    # ------------------------------------------------------------------
    def _merge_policy_or_none(self):
        """Lazily built, cached per-run MergePolicy — ``None`` when auto
        mapping is off, the run has nothing policy-governed, or the build
        failed; every alignment path then stays byte-identical to the
        canonical merge fallback."""
        if not (
                getattr(self.parameters, 'auto_type_mapping', False)
                and getattr(self.parameters, '_auto_type_mapper', None)
                is not None):
            return None
        cached = getattr(self, '_merge_policy', None)
        if cached is not None:
            return cached if cached is not False else None
        try:
            from .merge_policy import build_merge_policy
            policy = build_merge_policy(
                self.parameters._auto_type_mapper,
                source_tokens=self.parameters.source_neurons,
                target_tokens=self.parameters.target_neurons,
                datasets=self.parameters.get_dataset_names(),
                source_dataset=self.parameters.source_dataset,
                log=self._log,
            )
        except Exception as exc:  # noqa: BLE001 — fallback, never fatal
            self._log(f"Merge policy build failed ({exc}); using the "
                      "canonical merge fallback")
            policy = False
        if policy is None:
            policy = False  # nothing to govern — don't rebuild
        self._merge_policy = policy
        return policy if policy is not False else None

    def _policy_label_mapper(self):
        """Synthesized raw→group-label mapper for the alignment lane
        (Decision 8) — deliberately SEPARATE from the user's
        ``self.label_mapper`` (user mappings win by construction)."""
        policy = self._merge_policy_or_none()
        if policy is None:
            return None
        try:
            return policy.synthesized_label_mapper()
        except Exception as exc:  # noqa: BLE001
            self._log(f"Merge policy label synthesis failed ({exc})")
            return None

    def _annotate_auto_type_mapping_csv(self, map_path: str, policy) -> None:
        """Additive trailing ``anchor_group``/``auto_only`` columns on the
        run's auto_type_mapping.csv (historical prefix stays stable)."""
        if not os.path.exists(map_path):
            return
        from .merge_policy import row_anchor_group as _row_anchor_group
        frame = self._read_csv(map_path)
        if frame.empty:
            return
        meta_columns = {'mapping_origin', 'mapping_support', 'anchor_group',
                        'auto_only'}
        auto_only_pairs = set()
        try:
            from .merge_policy import auto_only_edges
            names = set()
            for column in frame.columns:
                if column in meta_columns:
                    continue
                names.update(
                    str(value).strip()
                    for value in frame[column].dropna()
                    if str(value).strip())
            for row in auto_only_edges(
                    self.parameters._auto_type_mapper, names,
                    self.parameters.get_dataset_names()):
                auto_only_pairs.add((
                    row['source_dataset'], row['source_type'],
                    row['target_dataset'], row['target_type']))
        except Exception as exc:  # noqa: BLE001
            self._log(f"auto_only column scan skipped: {exc}")
        dataset_columns = [c for c in frame.columns if c not in meta_columns]
        anchor_groups: List[str] = []
        auto_only_flags: List[str] = []
        for _, row in frame.iterrows():
            # NaN-safe: an empty CSV cell must read as '', never as the
            # string 'nan' (str(float('nan')) is truthy).
            values = {
                ds: ('' if pd.isna(row.get(ds)) else str(row.get(ds)).strip())
                for ds in dataset_columns
            }
            # UNAMBIGUOUS RULE: tag only rows where every non-empty
            # endpoint resolves to the SAME group (a row that merely
            # touches a group through one endpoint stays blank).
            anchor_groups.append(_row_anchor_group(policy, values))
            flagged = ''
            if auto_only_pairs:
                named = [(ds, name) for ds, name in values.items() if name]
                if any(
                    (da, ta, db, tb) in auto_only_pairs
                    for da, ta in named for db, tb in named if da != db
                ):
                    flagged = 'auto_only'
            auto_only_flags.append(flagged)
        frame['anchor_group'] = anchor_groups
        frame['auto_only'] = auto_only_flags
        frame.to_csv(map_path, index=False)

    def _write_merge_policy_warnings(self, policy, result_types,
                                     dataset_names) -> None:
        """Route the policy warnings (B3/B1 fan-in) and the B4 auto-only
        BANC block into user_warning_notes.txt (mirrored by the run
        guide); mappings stay VALID — the custom label mapper is the
        removal path."""
        blocks: List[str] = []
        if policy.warnings:
            blocks.append(
                '[merge policy] ' + policy.summary_line() + '\n'
                + '\n'.join(f'- {warning}' for warning in policy.warnings))
        rows: List[Dict[str, Any]] = []
        try:
            from .merge_policy import auto_only_edges
            rows = auto_only_edges(
                self.parameters._auto_type_mapper,
                result_types or [], dataset_names or [])
        except Exception as exc:  # noqa: BLE001
            self._log(f"BANC auto-label scan skipped: {exc}")
        if rows:
            by_direction: Dict[Tuple[str, str], int] = {}
            for row in rows:
                direction = (row['source_dataset'], row['target_dataset'])
                by_direction[direction] = by_direction.get(direction, 0) + 1
            direction_txt = '; '.join(
                f'{src} -> {tgt}: {count}'
                for (src, tgt), count in sorted(by_direction.items(),
                                                key=lambda item: -item[1]))
            listed = '\n'.join(
                f"- {row['source_dataset']} {row['source_type']} -> "
                f"{row['target_dataset']} {row['target_type']} "
                f"({row['total_votes']} auto vote(s))"
                for row in rows[:20])
            more = (f"\n... and {len(rows) - 20} more"
                    if len(rows) > 20 else '')
            blocks.append(
                '[BANC auto labels] '
                f"{len(rows)} mapping(s) rest on auto-transferred labels "
                f"only (no curated vote) — by direction: {direction_txt}.\n"
                f"{listed}{more}\n"
                "These remain valid; to exclude or override them use the "
                "custom label mapper (LabelMapper / overall_mapping_json).")
        # Alignment-fallback lane disclosure: mappings the weaker
        # fafb_alignment_cell_type lane filled where the curated
        # fafb_cell_type pass had no winner (mapper keep-rule; curated
        # winners and curated conflicts are never touched).
        try:
            mapper = self.parameters._auto_type_mapper
            al_rows = mapper.alignment_fallback_rows(
                filter_types=set(result_types or []) or None,
                datasets=dataset_names or [])
        except Exception as exc:  # noqa: BLE001
            al_rows = []
            self._log(f"BANC alignment-fallback scan skipped: {exc}")
        if al_rows:
            listed_al = '\n'.join(
                f"- {row['source_dataset']} {row['source_type']} -> "
                f"{row['target_dataset']} {row['target_type']} "
                f"({row['winner_votes']}/{row['total_votes']} alignment "
                f"vote(s), {row['verified_votes']} fafb_match-verified)"
                for row in al_rows[:20])
            more_al = (f"\n... and {len(al_rows) - 20} more"
                       if len(al_rows) > 20 else '')
            blocks.append(
                '[BANC alignment fallback] '
                f"{len(al_rows)} mapping(s) in this run were filled by the "
                "fafb_alignment_cell_type fallback lane (no curated "
                "fafb_cell_type winner) — a lower-evidence tier; the "
                "fafb_match column agrees with it less often than with "
                "curated labels (96.5% vs 99.2% whole-release), so please "
                f"double check these before interpretation.\n{listed_al}"
                f"{more_al}\n"
                "These remain valid; to exclude or override them use the "
                "custom label mapper (LabelMapper / overall_mapping_json).")
        # Same-name-first disclosure + multi-value `type` cells
        # (plan-samename-first-fanout-resolution §3;
        # plan-type-column-multivalue-normalization Stage 1).  Counts are
        # scoped EXACTLY like the exports (result types + run datasets) so
        # the note and the CSVs agree.
        try:
            mapper = self.parameters._auto_type_mapper
            counts = mapper.same_name_first_summary(
                filter_types=set(result_types or []) or None,
                datasets=dataset_names or None)
            if counts.get('selected') or counts.get('gated_held'):
                blocks.append(
                    '[same-name-first] '
                    f"{counts.get('selected', 0)} fan-out(s) in this run had "
                    f"the source's own name among their candidates and were "
                    f"SELECTED; {counts.get('gated_held', 0)} were kept "
                    "unmapped (not every rival candidate has its own 1-to-1 "
                    "pairing, so the same-name candidate was not selected; "
                    "they keep their ordinary status); "
                    f"{counts.get('excluded_evidence_only', 0)} evidence-only "
                    "(N-to-1) fan-out(s) are excluded by policy. "
                    f"{counts.get('rivals_exported', 0)} rival relation(s) in "
                    "total are listed for verification in "
                    "auto_type_mapping_suspects.csv (one row per rival, with "
                    "the per-rival pairing evidence and the ready-made "
                    "custom-mapper entry).")
            mv = mapper.multivalue_summary(datasets=dataset_names or None)
            if mv:
                ds_txt = ', '.join(f'{k} ({v})'
                                   for k, v in sorted(mv.items()))
                blocks.append(
                    '[multi-value type cells] '
                    f"{sum(mv.values())} release type name(s) are "
                    "comma-joined multi-value annotations (a release lists "
                    "several candidate names in one `type` cell); they are "
                    "kept atomic, noted here, and marked in the type-mapping "
                    "grid row when it is rendered (top rows by appearance); "
                    "the conflicts CSV carries multivalue_source/source_parts "
                    f"for the full set. Datasets: {ds_txt}.")
        except Exception as exc:  # noqa: BLE001 — disclosure only
            self._log(f"Same-name-first notes skipped: {exc}")
        # These two notes are per-run STATE (they describe the current
        # mapper scoping), so a re-export must REPLACE prior copies rather
        # than accumulate contradictory stale counts.
        if blocks:
            blocks = self._replace_state_notes(blocks)
        if blocks:
            os.makedirs(self.parameters.full_output_path, exist_ok=True)
            self._append_user_warning_notes(
                self.parameters.full_output_path, blocks)
            self._log("  ⚠️ Merge-policy notes appended to "
                      "user_warning_notes.txt")

    def get_applied_folder(self, dataset: str, requested: int) -> int:
        """Physical applied-threshold folder for one (dataset, requested) cell.

        Prefers the orchestrator's ``applied_folder``; falls back to the
        canonical applied threshold so callers never build a path from a
        requested threshold that was collapsed away.
        """
        meta = self._path_run_meta.get((dataset, requested), {}) or {}
        folder = meta.get('applied_folder')
        if folder is None:
            folder = self._applied_state_for(dataset, requested)[0]
        if folder is None:
            folder = requested
        return int(folder)

    def get_applied_thresholds(self, dataset: str) -> List[int]:
        """Distinct materialized (applied) thresholds for one dataset."""
        applied = []
        for t in self.parameters.get_thresholds_for_dataset(dataset):
            value = self.get_applied_folder(dataset, t)
            if value is not None:
                applied.append(int(value))
        return sorted(set(applied))

    def get_threshold_view(self, dataset: str, requested: int) -> Dict[str, Any]:
        """Resolved view for one (dataset, requested) cell."""
        meta = self._path_run_meta.get((dataset, requested), {}) or {}
        applied, pruned, floor, source = self._applied_state_for(
            dataset, requested)
        applied_folder = self.get_applied_folder(dataset, requested)
        skipped = bool(meta.get('skipped', False))
        # An aliased row's data IS the applied folder; report that folder's
        # threshold (the canonical equivalent it aliases). The orchestrator
        # never aliases to a weaker folder, so no clamp is applied.
        if skipped and applied_folder is not None:
            applied = int(applied_folder)
        return {
            'dataset': dataset,
            'requested_threshold': int(requested),
            'applied_threshold': int(applied) if applied is not None else None,
            'applied_threshold_source': source,
            'applied_folder': applied_folder,
            'status': 'aliased' if skipped else 'applied',
            'is_exact': (not skipped and applied is not None
                         and int(applied) == int(requested)),
            'duplicate_of': meta.get('duplicate_of'),
            'skipped': skipped,
            'edge_weight_floor': floor,
            'paths_complete': bool(meta.get('paths_complete',
                                            not pruned)),
        }

    def get_threshold_queries_view(self) -> List[Dict[str, Any]]:
        """Per-query requested→applied view (standard and combinations)."""
        datasets = self.parameters.get_dataset_names()
        views = []
        for query in self.get_threshold_queries():
            query_id = query.get('id') or query.get('query_id')
            thresholds = dict(query.get('thresholds') or {})
            applied_map = {}
            status_map = {}
            for ds in datasets:
                if ds not in thresholds:
                    continue
                cell = self.get_threshold_view(ds, int(thresholds[ds]))
                applied_map[ds] = cell['applied_threshold']
                status_map[ds] = cell['status']
            fully_aliased = bool(status_map) and all(
                s == 'aliased' for s in status_map.values())
            applied_txt = '/'.join(
                str(applied_map[d]) for d in datasets if d in applied_map)
            views.append({
                'id': query_id,
                'label': query.get('label', query_id),
                'requested_thresholds': thresholds,
                'applied_thresholds': applied_map,
                'status': status_map,
                'is_fully_aliased': fully_aliased,
                'display_label': (f"{query.get('label', query_id)} "
                                  f"(applied {applied_txt})"
                                  if applied_txt else
                                  str(query.get('label', query_id))),
            })
        return views

    def comparability_report(self) -> Dict[str, Any]:
        """How comparable the materialized thresholds are across datasets.

        ``common`` is the set of thresholds every dataset materialized at
        the SAME applied value (the only like-for-like comparison points).
        Emits advisory warnings when that set is empty or a single value,
        or when most requested thresholds were pruned/aliased.
        """
        datasets = self.parameters.get_dataset_names()
        materialized = {
            ds: self.get_applied_thresholds(ds) for ds in datasets}
        # A dataset that materialized NOTHING has no comparable threshold:
        # including it (as an empty set) is what makes `common` honest —
        # excluding it would let 'comparable across all datasets' print
        # while one dataset is absent (F-XD-007).
        sets = [set(v) for v in materialized.values()]
        common = sorted(set.intersection(*sets)) if sets else []
        pairwise_common = {}
        for i, a in enumerate(datasets):
            for b in datasets[i + 1:]:
                shared = set(materialized.get(a, [])) & set(
                    materialized.get(b, []))
                pairwise_common[f'{a}|{b}'] = sorted(shared)
        requested = list(getattr(self.parameters, 'thresholds', None) or [])
        skipped_all = [
            t for t in requested
            if datasets and all(
                self._is_duplicate_threshold(ds, t) for ds in datasets)
        ]
        skipped_fraction = (
            len(skipped_all) / len(requested) if requested else 0.0)

        warnings = []
        level = 'none'
        if requested and not common:
            level = 'critical'
            warnings.append(
                'No threshold is comparable across all datasets after '
                'budget pruning. Cross-dataset tables at every requested '
                'threshold mix different applied thresholds; adjust the '
                'threshold list or query and rerun, or use the '
                'density-aligned comparison.')
        elif requested and len(common) < 2:
            level = 'warning'
            warnings.append(
                f"Only threshold {common[0]} is comparable across all "
                'datasets after budget pruning; lower requested thresholds '
                'compare different applied cutoffs and should not be read '
                'as like-for-like.')
        if requested and skipped_fraction >= 0.5:
            level = level if level != 'none' else 'warning'
            warnings.append(
                f'{len(skipped_all)} of {len(requested)} requested '
                'thresholds were pruned/aliased; consider a shorter, '
                'better-spaced threshold list.')
        return {
            'datasets': datasets,
            'requested_thresholds': requested,
            'materialized_thresholds': materialized,
            'common_materialized': common,
            'pairwise_common': pairwise_common,
            'skipped_thresholds': skipped_all,
            'skipped_fraction': round(skipped_fraction, 4),
            'warning_level': level,
            'warnings': warnings,
        }

    def _threshold_query_manifest_rows(self) -> List[Dict[str, Any]]:
        """Build the query/dataset provenance join for comparison exports."""
        rows = []
        dataset_order = self.parameters.get_dataset_names()
        mode = getattr(self.parameters, 'threshold_mode', 'standard')
        for query_index, query in enumerate(self.get_threshold_queries(), start=1):
            query_id = query.get('id') or query.get('query_id')
            threshold_map = query.get('thresholds') or {}
            for dataset_order_index, dataset in enumerate(dataset_order, start=1):
                if dataset not in threshold_map:
                    continue
                threshold = int(threshold_map[dataset])
                row = self._path_provenance_row(dataset, threshold)
                row.update({
                    'query_id': query_id,
                    'query_index': query_index,
                    'query_label': query.get('label', query_id),
                    'threshold_mode': mode,
                    'dataset_order': dataset_order_index,
                    'requested_threshold': threshold,
                    'raw_run_key': (
                        f"{self.parameters._sanitize_name(dataset)}"
                        f"/minsyn_{threshold}"
                    ),
                })
                rows.append(row)
        return rows

    def _export_threshold_query_manifest(self, comparison_results_dir: str):
        """Export one query/dataset row with full applied-threshold provenance."""
        rows = self._threshold_query_manifest_rows()
        if not rows:
            return
        manifest = pd.DataFrame(rows)
        preferred = [
            'query_id', 'query_index', 'query_label', 'threshold_mode',
            'threshold_scope',
            'dataset', 'dataset_order', 'requested_threshold',
            'applied_threshold', 'applied_threshold_source',
            'strongest_first_budget', 'strongest_first_budget_bitten',
            'strongest_first_tau', 'tau', 'tau_canonical',
            'strongest_dropped_bottleneck', 'strongest_retained_bottleneck',
            'edge_budget', 'edge_budget_applied', 'edge_budget_landing',
            'edge_weight_floor', 'paths_complete', 'skipped',
            'duplicate_of', 'applied_folder', 'raw_run_key', 'path_mode',
            'comparison_mode', 'drop_untyped',
        ]
        manifest = manifest[
            [column for column in preferred if column in manifest.columns]
            + [column for column in manifest.columns if column not in preferred]
        ]
        path = os.path.join(
            comparison_results_dir, 'threshold_combinations.csv'
        )
        self._save_csv(manifest, path)
        self._log_file(path, 'Threshold query manifest')

    def _export_effective_threshold_banner(self):
        """Persist per-dataset applied-threshold provenance for the UI/guide.

        The historical file was written only for all-path tau collapses.  It
        is now the comparison run's compact, always-present notice payload:
        shortest runs, complete runs, edge-budget-only runs, and collapsed
        runs all carry the same complete provenance rows.  The old
        ``input``/``effective``/``applied_folder`` keys remain for readers
        that only understand the collapse banner.
        """
        dataset_names = self.parameters.get_dataset_names()
        datasets_out = {}
        banner_parts = []
        any_skip = False
        for ds in dataset_names:
            thresholds = self.parameters.get_thresholds_for_dataset(ds)
            mapping = {}
            skipped = []
            effective = []
            aliased_folders = []
            tau_seen = None
            applied_thresholds = {}
            runs = []
            for t in thresholds:
                meta = self._path_run_meta.get((ds, t), {})
                row = self._path_provenance_row(ds, t)
                runs.append(row)
                applied_thresholds[str(t)] = row['applied_threshold']
                applied = meta.get('applied_folder')
                if meta.get('skipped') and applied is not None:
                    mapping[str(t)] = applied
                    skipped.append(t)
                    any_skip = True
                    # The aliased canonical folder is a REAL applied
                    # point on disk (e.g. minsyn_9) — keep it visible in
                    # the applied list so the reader sees what the asked
                    # threshold became.
                    if applied not in thresholds:
                        aliased_folders.append(applied)
                tau = row.get('tau_canonical') or row.get('tau')
                # Edge mode applies no tau (F-XD-004/R7-1 F-P3 nuance): the
                # block-level tau must not resurrect the side path runs'
                # value the run rows just nulled.
                if getattr(meta, 'get', lambda k: None)('edge_mode'):
                    tau = None
                if tau is not None and not meta.get('skipped'):
                    tau_seen = tau if tau_seen is None else max(tau_seen, tau)
                if not meta.get('skipped'):
                    effective.append(row['applied_threshold'])
            if thresholds:
                # When every asked threshold collapsed, the applied
                # folders ARE the effective points — show them instead of
                # an empty list. Aliased canonical folders (materialized
                # at w2+1) join the applied list too.
                effective = effective or sorted(set(mapping.values()))
                effective = sorted(set(effective) | set(aliased_folders))
                datasets_out[ds] = {
                    'input': thresholds,
                    'effective': effective,
                    'skipped': skipped,
                    'applied_folder': mapping,
                    'tau': tau_seen,
                    'applied_thresholds': applied_thresholds,
                    'runs': runs,
                }
                eff_txt = ', '.join(str(t) for t in effective) or '—'
                alias_txt = ''
                if mapping:
                    alias_txt = ' (aliased: ' + ', '.join(
                        f'{t}→{applied}'
                        for t, applied in mapping.items()) + ')'
                head = f"{ds}: input [{', '.join(map(str, thresholds))}]"
                source_txt = ', '.join(sorted({
                    str(r['applied_threshold_source']) for r in runs
                }))
                if tau_seen is not None:
                    banner_parts.append(
                        f"{head} → applied [{eff_txt}] (sources: "
                        f"{source_txt}; τ={tau_seen:g}){alias_txt}")
                else:
                    banner_parts.append(
                        f"{head} → applied [{eff_txt}] (sources: "
                        f"{source_txt}){alias_txt}")
        manifest_rows = self._threshold_query_manifest_rows()
        query_out = []
        for query in self.get_threshold_queries():
            query_id = query.get('id') or query.get('query_id')
            query_rows = [
                row for row in manifest_rows if row.get('query_id') == query_id
            ]
            query_out.append({
                'id': query_id,
                'label': query.get('label', query_id),
                'requested_thresholds': dict(query.get('thresholds') or {}),
                'applied_thresholds': {
                    row['dataset']: row.get('applied_threshold')
                    for row in query_rows
                },
                'runs': query_rows,
            })
        if any_skip:
            prefix = 'Threshold collapse: '
            suffix = (' — thresholds below τ skipped as duplicates '
                      '(aliased to the applied folder).')
        else:
            prefix = 'Pathfinding provenance: '
            suffix = ' — see comparison_results/pathfinding_provenance.csv.'
        comparability = self.comparability_report()
        threshold_views = self.get_threshold_queries_view()
        payload = {
            'banner': prefix + '; '.join(banner_parts) + suffix,
            'datasets': datasets_out,
            'threshold_mode': getattr(
                self.parameters, 'threshold_mode', 'standard'),
            'threshold_dataset_order': dataset_names,
            'combinations': query_out,
            'queries': query_out,
            'threshold_views': threshold_views,
            'comparability': comparability,
            'path_mode': getattr(self.parameters, 'path_mode', 'all'),
            'comparison_mode': getattr(self.parameters, 'comparison_mode',
                                       'path'),
            'provenance_file': 'comparison_results/pathfinding_provenance.csv',
        }
        try:
            out_path = os.path.join(self.parameters.full_output_path,
                                    'effective_thresholds.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, default=str)
            self._log("Saved: effective_thresholds.json")
        except Exception as e:
            self._log(f"Warning: could not write effective_thresholds.json: {e}")
        self._write_threshold_view_and_notes(payload, comparability)

    def _write_threshold_view_and_notes(self, payload: Dict[str, Any],
                                        comparability: Dict[str, Any]) -> None:
        """Persist the shared threshold view and surface comparability
        warnings in the log and the run's user_warning_notes.txt."""
        out_dir = self.parameters.full_output_path
        if out_dir:
            view_dir = os.path.join(out_dir, 'comparison_report_used_data')
            try:
                os.makedirs(view_dir, exist_ok=True)
                view_path = os.path.join(view_dir, 'threshold_view.json')
                with open(view_path, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=2, default=str)
                self._log("Saved: comparison_report_used_data/threshold_view.json")
            except Exception as e:
                self._log(f"Warning: could not write threshold_view.json: {e}")

        warnings = list(comparability.get('warnings') or [])
        level = comparability.get('warning_level', 'none')
        if level == 'critical':
            self._log('THRESHOLD COMPARABILITY: ' + warnings[0])
        elif level == 'warning':
            self._log('Threshold comparability warning: ' + warnings[0])
        if warnings and out_dir:
            self._append_user_warning_notes(
                out_dir,
                [f"- [threshold comparability] {warning}"
                 for warning in warnings])

    def _is_untyped_type_value(self, value) -> bool:
        """True when a type label means 'untyped': empty, an explicit
        Unknown/none sentinel, or the bodyId-fallback label (the neuron's
        own id used as its type when no name resolved).

        Delegates to the shared predicate in ``utils.label_utils`` so
        pathfinding (``FindNeuronConnection.drop_untyped``) and comparison
        always agree on the untyped definition."""
        if is_untyped_type_label is not None:
            return is_untyped_type_label(value)
        s = str(value).strip()
        if not s or s.lower() in {"unknown", "nan", "none",
                                  "null", "<na>", "<null>"}:
            return True
        return s.isdigit()

    def _drop_untyped_neurons(self, dataset_name: str, threshold: int,
                              df: pd.DataFrame) -> pd.DataFrame:
        """Drop edges touching untyped neurons (drop_untyped=True).

        A neuron is untyped when its resolved type label is empty / an
        Unknown sentinel / its own bodyId (the fallback label). Such
        edges can never match across datasets (each dataset's bodyIds
        are disjoint) and only dilute the aligned comparisons, so they
        are removed from the cross-dataset results by default. The
        dropped rows are kept for the explicit records export and the
        dropped-neuron counts are appended to user_warning_notes.
        """
        if df is None or df.empty:
            return df
        if not getattr(self.parameters, "drop_untyped", True):
            return df
        if "type_pre" not in df.columns or "type_post" not in df.columns:
            return df
        pre = df["type_pre"].astype(str).str.strip()
        post = df["type_post"].astype(str).str.strip()
        untyped_pre = pre.map(self._is_untyped_type_value)
        untyped_post = post.map(self._is_untyped_type_value)
        drop_mask = untyped_pre | untyped_post
        if not drop_mask.any():
            return df
        dropped = df[drop_mask].copy()
        # Side flag shared with the pathfinding records
        # (data_details/untyped_dropped_records.csv) so both file formats
        # carry the same schema.
        if untyped_side is not None:
            dropped["untyped_side"] = [
                untyped_side(bool(p), bool(q))
                for p, q in zip(untyped_pre[drop_mask],
                                untyped_post[drop_mask])
            ]
        # The loader paths may have added these provenance columns already
        # (connection_type/connection_info branches pre-fill them) —
        # overwrite instead of insert so the drop never raises on a
        # duplicate column.
        if "threshold" in dropped.columns:
            dropped["threshold"] = threshold
        else:
            dropped.insert(0, "threshold", threshold)
        if "dataset" in dropped.columns:
            dropped["dataset"] = dataset_name
        else:
            dropped.insert(0, "dataset", dataset_name)
        self._untyped_dropped_records.append(dropped)
        neurons = set()
        has_body_ids = False
        for col in ("bodyId_pre", "bodyId_post"):
            if col in dropped.columns:
                has_body_ids = True
                neurons |= set(dropped[col].astype(str))
        self._untyped_drop_stats[(dataset_name, threshold)] = {
            "rows": len(dropped),
            # None (not 0) when the frame carries no bodyId columns: the
            # distinct-neuron count is unavailable there and must not be
            # reported as a literal zero.
            "neurons": len(neurons) if has_body_ids else None,
            "untyped_pre": int(untyped_pre[drop_mask].sum()),
            "untyped_post": int(untyped_post[drop_mask].sum()),
            # Denominator for the dropped fraction (issue #10). Without it
            # the counts cannot be normalized across datasets with very
            # different annotation completeness.
            "total_rows": int(len(df)),
            "fraction": round(len(dropped) / len(df), 6) if len(df) else 0.0,
        }
        return df[~drop_mask].copy()

    def _export_untyped_drop_records(self):
        """Export the dropped untyped rows and append the per-run counts
        to the run root's user_warning_notes.txt."""
        if not self._untyped_drop_stats:
            return
        out_dir = self.parameters.full_output_path
        query_refs = {}
        if self.parameters.threshold_mode == 'combinations':
            for query in self.get_threshold_queries():
                query_id = query.get('id') or query.get('query_id')
                for dataset, threshold in (query.get('thresholds') or {}).items():
                    query_refs.setdefault((dataset, int(threshold)), []).append(
                        str(query_id))
        if self._untyped_dropped_records:
            recs = pd.concat(self._untyped_dropped_records, ignore_index=True)
            if query_refs and {'dataset', 'threshold'} <= set(recs.columns):
                recs['query_id'] = [
                    ';'.join(query_refs.get((dataset, int(threshold)), []))
                    for dataset, threshold in zip(
                        recs['dataset'], recs['threshold'])
                ]
                recs['query_label'] = recs['query_id']
            rec_dir = os.path.join(out_dir, "comparison_results")
            os.makedirs(rec_dir, exist_ok=True)
            rec_path = os.path.join(rec_dir, "untyped_dropped_records.csv")
            recs.to_csv(rec_path, index=False)
        lines = []
        total_rows = 0
        for (ds, t), s in sorted(self._untyped_drop_stats.items()):
            total_rows += s["rows"]
            provenance = self._path_provenance_row(ds, t)
            refs = query_refs.get((ds, int(t)), [])
            # The same join the CSV's `query_id` column uses, so a note names
            # a cell that exists there rather than a third id scheme. The
            # embedded ';' carries no following space, so it cannot be read
            # as this line's '; ' field separator.
            query_text = (
                f"; query_id={';'.join(refs)}" if refs else "")
            # 'n/a' when the frame carries no bodyId columns — a literal 0
            # would read as "no untyped neurons were involved", which is
            # unknowable there.
            neuron_text = ('n/a' if s.get("neurons") is None
                           else str(s["neurons"]))
            lines.append(
                f"- [untyped dropped] {ds} @ t={t}: {s['rows']} edges / "
                f"{neuron_text} distinct untyped neurons dropped "
                f"(drop_untyped=True; pre-side {s['untyped_pre']}, "
                f"post-side {s['untyped_post']}; requested_threshold={t}; "
                f"applied_threshold={provenance.get('applied_threshold')}"
                f"{query_text})")
        self._append_user_warning_notes(out_dir, lines)
        self._log(f"Untyped neurons dropped by default: {total_rows} edges "
                  f"across {len(self._untyped_drop_stats)} dataset-threshold "
                  f"runs (records: comparison_results/"
                  f"untyped_dropped_records.csv; counts appended to "
                  f"user_warning_notes.txt)")

    def _run_all_path_analyses(self, skip_existing: bool = True) -> Dict[str, Dict[int, pd.DataFrame]]:
        """Run path-based analyses for all datasets and their thresholds.

        Standard mode gives every dataset the same scalar query schedule.
        Combination mode derives a per-dataset raw-run schedule from the
        complete query rows, so repeated ``(dataset, threshold)`` cells are
        executed once and later aligned by their owning query row.  The
        derived schedule is an execution/cache detail, not an independent
        comparison axis; the deprecated ``dataset_thresholds`` adapter is the
        only legacy exception.

        Feature G: iterating ascending, an input threshold t whose
        previous (same-dataset) run has effective τ >= t is SKIPPED — its
        path set is identical to that run's by the τ-equivalence; the
        frame is aliased and the exports carry
        ``skipped/duplicate_of/tau`` markers.

        Feature F: when ``replay_paths`` is enabled and path_mode='all',
        the dataset's pending thresholds are executed in ONE
        ``FindAllPathMultiThreshold`` call — enumeration happens at the
        lowest pending threshold only and higher thresholds are
        materialized from the bottleneck-annotated path set.
        """
        dataset_names = self.parameters.get_dataset_names()
        replay_enabled = (
            self.parameters.replay_paths
            and self.parameters.path_mode == 'all'
        )

        for dataset_name in dataset_names:
            if dataset_name not in self.raw_results:
                self.raw_results[dataset_name] = {}

            thresholds_ds = self.parameters.get_thresholds_for_dataset(dataset_name)
            self._log(f"Processing \033[94m{dataset_name}\033[0m "
                      f"({len(thresholds_ds)} thresholds: {thresholds_ds})")

            meta_map = self._load_threshold_meta(dataset_name)
            lowest_threshold = thresholds_ds[0] if thresholds_ds else None

            # Thresholds iterate ASCENDING. prev_meta tracks the most
            # recent ACTUAL run (fresh or resumed) so the Feature G skip
            # rule sees τ from runs executed earlier in this same loop.
            prev_meta: Optional[Dict] = None
            prev_t: Optional[int] = None
            pending: list = []

            def _note_actual(threshold, meta):
                nonlocal prev_meta, prev_t
                meta = self._path_run_meta_from_state(
                    dataset_name, threshold, meta)
                meta.setdefault('_threshold', threshold)
                self._path_run_meta[(dataset_name, threshold)] = meta
                meta_map[threshold] = dict(meta)
                if not meta.get('skipped'):
                    prev_meta = meta
                    prev_t = threshold

            for threshold in thresholds_ds:
                meta = meta_map.get(threshold)

                # Already in memory (fresh run earlier in this session)?
                if skip_existing and threshold in self.raw_results[dataset_name]:
                    if meta and not meta.get('skipped'):
                        _note_actual(threshold, meta)
                    continue

                if skip_existing and self.parameters.output_folder:
                    if meta and meta.get('skipped'):
                        dup_df = self.raw_results[dataset_name].get(
                            meta.get('duplicate_of'))
                        if dup_df is None and meta.get('applied_folder'):
                            # F5: the applied (tau) folder holds the real
                            # output for collapsed thresholds — alias its
                            # cached frame.
                            dup_df = self._try_load_cached(
                                dataset_name, meta.get('applied_folder'))
                        if dup_df is not None:
                            # Resume key is the aliased frame (G §13.5 item 9)
                            self.raw_results[dataset_name][threshold] = \
                                self._finalize_loaded_result(
                                    dataset_name, threshold, dup_df)
                            self._path_run_meta[(dataset_name, threshold)] = \
                                self._path_run_meta_from_state(
                                    dataset_name, threshold, meta)
                            continue
                        # duplicate frame unavailable: fall through and
                        # re-run this threshold normally
                    else:
                        cached = self._try_load_cached(dataset_name, threshold)
                        if cached is not None:
                            self.raw_results[dataset_name][threshold] = \
                                self._finalize_loaded_result(
                                    dataset_name, threshold, cached)
                            _note_actual(
                                threshold,
                                meta or self._read_cached_path_meta(
                                    dataset_name, threshold),
                            )
                            continue

                # Feature G duplicate-threshold skip (legacy mode decides
                # here; the replay batch marks collapses post-execution).
                if not replay_enabled and self._g_skip_allowed(
                        dataset_name, threshold, prev_meta):
                    tau = prev_meta.get('tau')
                    bitten = bool(prev_meta.get('budget_bitten'))
                    skip_meta = {
                        'tau': tau,
                        'strongest_first_tau': prev_meta.get(
                            'strongest_first_tau', tau),
                        'tau_canonical': prev_meta.get('tau_canonical'),
                        'strongest_dropped_bottleneck': prev_meta.get(
                            'strongest_dropped_bottleneck'),
                        'budget_bitten': bitten,
                        'strongest_first_budget_bitten': prev_meta.get(
                            'strongest_first_budget_bitten', bitten),
                        'strongest_first_budget': prev_meta.get(
                            'strongest_first_budget'),
                        'paths_complete': bool(prev_meta.get(
                            'paths_complete', not bitten)),
                        'skipped': True,
                        'duplicate_of': prev_t,
                        'applied_folder': prev_meta.get(
                            'applied_folder', prev_t),
                        'edge_weight_floor': prev_meta.get(
                            'edge_weight_floor'),
                        'edge_budget_landing': prev_meta.get(
                            'edge_budget_landing'),
                        'edge_budget': prev_meta.get('edge_budget'),
                        'pathfinding': prev_meta.get('pathfinding'),
                        'strongest_retained_bottleneck': prev_meta.get(
                            'strongest_retained_bottleneck'),
                        'graph_pruning_record': dict(
                            prev_meta.get('graph_pruning_record') or {}),
                    }
                    # G §13.5 item 5: alias the frame as an independent
                    # COPY — never mutate it, and never share it by
                    # reference (a future in-place edit of one key would
                    # otherwise corrupt the other).
                    self.raw_results[dataset_name][threshold] = \
                        self.raw_results[dataset_name][prev_t].copy()
                    _note_actual(threshold, skip_meta)
                    self._store_threshold_meta(dataset_name, meta_map)
                    self._log(
                        f"  threshold={threshold}: SKIPPED (duplicate of "
                        f"t={prev_t}: identical set up to τ={tau:g})"
                        if tau is not None else
                        f"  threshold={threshold}: SKIPPED (duplicate of t={prev_t})")
                    continue

                if replay_enabled:
                    pending.append(threshold)
                    continue

                # Legacy per-threshold run (interleaved so later skip
                # decisions see this run's τ).
                verbose = 'simple' if threshold == lowest_threshold else 'silent'
                result_df = self.run_path_analysis(dataset_name, threshold,
                                                   verbose_mode=verbose)
                self.raw_results[dataset_name][threshold] = result_df
                run_meta = self._path_run_meta.get((dataset_name, threshold), {
                    'tau': None, 'budget_bitten': False,
                    'paths_complete': True, 'skipped': False,
                    'duplicate_of': None,
                    'pathfinding': self.parameters.pathfinding,
                })
                _note_actual(threshold, run_meta)
                if self.parameters.output_folder:
                    self._save_result(dataset_name, threshold, result_df)

            if pending:
                self._execute_replay_batch(
                    dataset_name, pending, lowest_threshold, meta_map,
                    _note_actual,
                    prev_meta_init=prev_meta, prev_t_init=prev_t)
                self._store_threshold_meta(dataset_name, meta_map)
            elif any(m.get('skipped') for m in meta_map.values()):
                self._store_threshold_meta(dataset_name, meta_map)

        # F7: auto-extend collapsed thresholds (opt-in). Global schedule:
        # points = k * tau_ref while <= 2x the max asked threshold, so the
        # expanded points stay shared across datasets and every dataset
        # gains fresh material at each one. Each point is a REAL threshold
        # (own key + folder) run through the same replay batch; Feature G
        # skipping applies between the points as usual.
        if (self.parameters.auto_extend_thresholds
                and self.parameters.threshold_mode != 'combinations'
                and self.parameters.path_mode == 'all'
                and self.parameters.replay_paths):
            asked_max = 0
            taus = []
            for ds in dataset_names:
                for t in self.parameters.get_thresholds_for_dataset(ds):
                    asked_max = max(asked_max, t)
                    m = self._path_run_meta.get((ds, t), {})
                    tau = m.get('tau')
                    if tau is not None and not m.get('skipped'):
                        taus.append(tau)
            if taus:
                tau_ref = int(round(max(taus)))
                points = []
                k = 2
                while k * tau_ref <= 2 * asked_max:
                    points.append(k * tau_ref)
                    k += 1
                points = sorted({p for p in points if p > tau_ref})
            else:
                points = []
            if not points:
                self._log("  F7: tau exceeds the 2x asked-max cap — no "
                          "auto-extension (raise thresholds >= tau to probe "
                          "further)")
            else:
                # Standard mode has one global schedule.  Snapshot each
                # dataset's pre-extension schedule before updating the
                # shared list; otherwise the first dataset would make the
                # new points appear present for every later dataset and only
                # the first dataset would be replayed.  The deprecated
                # dataset_thresholds adapter remains per-dataset when an
                # older caller supplied it explicitly.
                base_schedules = {
                    dataset_name: self.parameters.get_thresholds_for_dataset(
                        dataset_name)
                    for dataset_name in dataset_names
                }
                uses_legacy_schedule = (
                    self.parameters.dataset_thresholds is not None)
                if not uses_legacy_schedule:
                    self.parameters.thresholds = sorted(
                        set(self.parameters.thresholds) | set(points))
                for dataset_name in dataset_names:
                    current = base_schedules[dataset_name]
                    new_points = [p for p in points if p not in current]
                    if not new_points:
                        continue
                    merged = sorted(set(current) | set(new_points))
                    # Persist the extended lists — exports iterate them.
                    if uses_legacy_schedule:
                        self.parameters.dataset_thresholds[dataset_name] = merged
                    meta_map = self._load_threshold_meta(dataset_name)
                    self._log(f"  F7 auto-extension for {dataset_name}: "
                              f"+{new_points}")

                    def _note_expansion(t, m, _ds=dataset_name):
                        m.setdefault('_threshold', t)
                        self._path_run_meta[(_ds, t)] = m
                        meta_map[t] = dict(m)

                    self._execute_replay_batch(
                        dataset_name, new_points, min(merged), meta_map,
                        _note_expansion)
                    self._store_threshold_meta(dataset_name, meta_map)
                self._export_effective_threshold_banner()
        elif (
            self.parameters.auto_extend_thresholds
            and self.parameters.threshold_mode == 'combinations'
        ):
            self._log(
                "Advanced threshold combinations keep their explicit rows; "
                "auto-extension is not applied.",
                level='warn',
            )

        # Replay, legacy, and cache paths all pass through one final
        # normalization before aggregate exports are written.
        self._complete_path_run_meta()
        self._reconcile_applied_folders()
        self._export_untyped_drop_records()
        self._export_effective_threshold_banner()
        self._log(f"Completed path analysis for {len(dataset_names)} datasets")
        return self.raw_results

    def _execute_replay_batch(
        self,
        dataset_name: str,
        pending: list,
        lowest_threshold: Optional[int],
        meta_map: Dict[int, Dict],
        note_actual,
        prev_meta_init: Optional[Dict] = None,
        prev_t_init: Optional[int] = None,
    ) -> None:
        """Run the pending thresholds of one dataset via ONE
        FindAllPathMultiThreshold call (Feature F).

        The orchestrator enumerates at the first pending threshold and
        materializes every higher threshold from the bottleneck slice —
        thresholds collapsed by τ (Feature G) keep their materialized
        folder (built from the same slice, §9.7) and are additionally
        marked/aliased for the exports. On replay declination (budget
        exceeded / capture failure) this falls back to the interleaved
        legacy per-threshold loop, skip logic included.
        """
        executed: Dict[int, Dict] = {}
        verbose = 'simple' if pending[0] == lowest_threshold else 'silent'
        thresholds_str = ', '.join(str(t) for t in pending)
        self._log(f"  Replay batch for {dataset_name}: [{thresholds_str}] "
                  f"(enumerate at {pending[0]}, materialize the rest)")
        results = self._run_multi_threshold_replay(
            dataset_name, list(pending), verbose)

        if results is None:
            self._log("  Replay unavailable — falling back to "
                      "per-threshold enumeration", level='warn')
            # Seed the fallback chain with the last resumed run so skip
            # decisions span the batch boundary.
            prev_meta = prev_meta_init
            prev_t = prev_t_init
            for threshold in pending:
                if self._g_skip_allowed(dataset_name, threshold, prev_meta):
                    tau = prev_meta.get('tau')
                    bitten = bool(prev_meta.get('budget_bitten'))
                    skip_meta = {
                        'tau': tau,
                        'strongest_first_tau': prev_meta.get(
                            'strongest_first_tau', tau),
                        'tau_canonical': prev_meta.get('tau_canonical'),
                        'strongest_dropped_bottleneck': prev_meta.get(
                            'strongest_dropped_bottleneck'),
                        'budget_bitten': bitten,
                        'strongest_first_budget_bitten': prev_meta.get(
                            'strongest_first_budget_bitten', bitten),
                        'strongest_first_budget': prev_meta.get(
                            'strongest_first_budget'),
                        'paths_complete': bool(prev_meta.get(
                            'paths_complete', not bitten)),
                        'skipped': True,
                        'duplicate_of': prev_t,
                        'applied_folder': prev_meta.get(
                            'applied_folder', prev_t),
                        'edge_weight_floor': prev_meta.get(
                            'edge_weight_floor'),
                        'edge_budget_landing': prev_meta.get(
                            'edge_budget_landing'),
                        'edge_budget': prev_meta.get('edge_budget'),
                        'pathfinding': prev_meta.get('pathfinding'),
                        'strongest_retained_bottleneck': prev_meta.get(
                            'strongest_retained_bottleneck'),
                        'graph_pruning_record': dict(
                            prev_meta.get('graph_pruning_record') or {}),
                        '_threshold': threshold,
                    }
                    # Independent copy (directive 2): shared references
                    # would let a future in-place edit corrupt both keys.
                    self.raw_results[dataset_name][threshold] = \
                        self.raw_results[dataset_name][prev_t].copy()
                    self._path_run_meta[(dataset_name, threshold)] = skip_meta
                    meta_map[threshold] = dict(skip_meta)
                    self._log(f"  threshold={threshold}: SKIPPED (duplicate "
                              f"of t={prev_t})")
                    continue
                verbose = 'simple' if threshold == lowest_threshold else 'silent'
                df = self.run_path_analysis(dataset_name, threshold,
                                            verbose_mode=verbose)
                self.raw_results[dataset_name][threshold] = df
                run_meta = self._path_run_meta.get((dataset_name, threshold), {
                    'tau': None, 'budget_bitten': False,
                    'paths_complete': True, 'skipped': False,
                    'duplicate_of': None,
                    'pathfinding': self.parameters.pathfinding,
                })
                run_meta.setdefault('_threshold', threshold)
                self._path_run_meta[(dataset_name, threshold)] = run_meta
                meta_map[threshold] = dict(run_meta)
                if self.parameters.output_folder:
                    self._save_result(dataset_name, threshold, df)
                if not run_meta.get('skipped'):
                    prev_meta = run_meta
                    prev_t = threshold
            return

        prev_t = None
        prev_meta = None
        for t in pending:
            raw = results.get(t)
            if raw is None:
                continue
            if raw.get('_reenumerate'):
                # W4: the threshold sits below the floored slice's floor
                # (t0 < t < w0) — the slice cannot serve it. Enumerate
                # individually; its own floor/budget semantics apply.
                verbose = 'simple' if t == lowest_threshold else 'silent'
                df = self.run_path_analysis(dataset_name, t,
                                            verbose_mode=verbose)
                self.raw_results[dataset_name][t] = df
                run_meta = self._path_run_meta.get((dataset_name, t), {
                    'tau': None, 'budget_bitten': False,
                    'paths_complete': True, 'skipped': False,
                    'duplicate_of': None,
                    'pathfinding': self.parameters.pathfinding,
                })
                run_meta.setdefault('_threshold', t)
                self._path_run_meta[(dataset_name, t)] = run_meta
                meta_map[t] = dict(run_meta)
                if self.parameters.output_folder:
                    self._save_result(dataset_name, t, df)
                if not run_meta.get('skipped'):
                    prev_t = t
                    prev_meta = run_meta
                continue
            # F5 tau-folder discipline: the orchestrator decides the
            # folders — collapsed thresholds carry skipped=True plus the
            # applied folder (the tau folder) and have NO minsyn_{t} on
            # disk; their frames alias the applied folder's output.
            if raw.get('skipped'):
                applied = raw.get('applied_folder')
                meta = {
                    'tau': raw.get('tau'),
                    'strongest_first_tau': raw.get(
                        'strongest_first_tau', raw.get('tau')),
                    # The canonical minimal equivalent. Never fabricate it
                    # from a weaker applied folder: if the orchestrator did
                    # not supply one, leave None so the shared formula
                    # recomputes it from the mechanism state.
                    'tau_canonical': raw.get('tau_canonical'),
                    'strongest_dropped_bottleneck': raw.get(
                        'strongest_dropped_bottleneck'),
                    'budget_bitten': bool(raw.get('budget_bitten')),
                    'strongest_first_budget_bitten': bool(raw.get(
                        'strongest_first_budget_bitten',
                        raw.get('budget_bitten'))),
                    'strongest_first_budget': raw.get(
                        'strongest_first_budget'),
                    'paths_complete': bool(raw.get('paths_complete', False)),
                    'skipped': True,
                    'duplicate_of': raw.get('duplicate_of', applied),
                    'applied_folder': applied,
                    'pathfinding': self.parameters.pathfinding,
                    'edge_weight_floor': raw.get('edge_weight_floor'),
                    'edge_budget': raw.get('edge_budget'),
                    'edge_budget_applied': bool(raw.get(
                        'edge_budget_applied', False)),
                    'edge_budget_landing': raw.get('edge_budget_landing'),
                    'strongest_retained_bottleneck': raw.get(
                        'strongest_retained_bottleneck'),
                    'graph_pruning_record': dict(
                        raw.get('graph_pruning_record') or {}),
                }
                df = self._load_multi_threshold_result(dataset_name, applied)
                self.raw_results[dataset_name][t] = df
                note_actual(t, meta)
                executed[t] = meta
                self._path_taus[(dataset_name, t)] = meta.get('tau')
                if meta.get('tau') is not None:
                    self._log(
                        f"  threshold={t}: SKIPPED (duplicate of "
                        f"minsyn_{applied}: identical set at "
                        f"τ={meta['tau']:g})")
                else:
                    self._log(
                        f"  threshold={t}: SKIPPED (duplicate of "
                        f"minsyn_{applied})")
                continue
            meta = {
                'tau': raw.get('tau'),
                'strongest_first_tau': raw.get(
                    'strongest_first_tau', raw.get('tau')),
                'tau_canonical': raw.get('tau_canonical'),
                'strongest_dropped_bottleneck': raw.get(
                    'strongest_dropped_bottleneck'),
                'budget_bitten': bool(raw.get('budget_bitten')),
                'strongest_first_budget_bitten': bool(raw.get(
                    'strongest_first_budget_bitten', raw.get('budget_bitten'))),
                'strongest_first_budget': raw.get('strongest_first_budget'),
                'paths_complete': bool(raw.get('paths_complete', True)),
                'skipped': False,
                'duplicate_of': None,
                'applied_folder': int(t),
                'pathfinding': self.parameters.pathfinding,
                'edge_weight_floor': raw.get('edge_weight_floor'),
                'edge_budget': raw.get('edge_budget'),
                'edge_budget_applied': bool(raw.get(
                    'edge_budget_applied', False)),
                'edge_budget_landing': raw.get('edge_budget_landing'),
                'strongest_retained_bottleneck': raw.get(
                    'strongest_retained_bottleneck'),
                'graph_pruning_record': dict(
                    raw.get('graph_pruning_record') or {}),
            }
            df = self._load_multi_threshold_result(dataset_name, t)
            self.raw_results[dataset_name][t] = df
            # Feature G collapse within the replay batch: the folder was
            # materialized from the same slice (no extra enumeration),
            # and the exports mark the threshold as skipped/aliased.
            if prev_meta is not None and self._g_skip_allowed(
                    dataset_name, t, prev_meta):
                meta['skipped'] = True
                meta['duplicate_of'] = prev_t
                tau_txt = (f"{meta.get('tau'):g}" if meta.get('tau')
                           is not None else "?")
                self._log(f"  threshold={t}: collapsed (≡ t={prev_t} at "
                          f"τ={tau_txt}) — materialized from the same "
                          f"replay slice")
            note_actual(t, meta)
            executed[t] = meta
            self._path_taus[(dataset_name, t)] = meta.get('tau')
            if self.parameters.output_folder and not meta.get('skipped'):
                self._save_result(dataset_name, t, df)
            if not meta.get('skipped'):
                prev_t = t
                prev_meta = meta

    def _run_multi_threshold_replay(self, dataset_name: str, thresholds: list,
                                    verbose_mode: str = 'simple') -> Optional[Dict[int, Dict]]:
        """One FindNeuronConnection run materializing all thresholds (F).

        Returns {threshold: meta} or None when replay was declined and the
        caller must fall back to per-threshold enumeration.
        """
        from coana import FindNeuronConnection

        threshold = thresholds[0]
        if verbose_mode != 'silent':
            self._log(f"Running replay analysis: \033[94m{dataset_name} "
                      f"@ thresholds={thresholds}\033[0m")

        config = self._get_dataset_config(dataset_name)
        source_neurons = self.parameters.get_source_neurons_for_dataset(dataset_name)
        target_neurons = self.parameters.get_target_neurons_for_dataset(dataset_name)
        max_interlayer = self.parameters.max_interlayer

        safe_dataset_name = self.parameters._sanitize_name(dataset_name)
        fnc_output_path = self.parameters.get_dataset_output_path(dataset_name, threshold)

        custom_source_name = ''
        if self.parameters.source_labels and len(self.parameters.source_labels) == 1:
            custom_source_name = self.parameters.source_labels[0]
        custom_target_name = ''
        if self.parameters.target_labels and len(self.parameters.target_labels) == 1:
            custom_target_name = self.parameters.target_labels[0]

        is_fafb = is_fafb_dataset(dataset_name)
        use_force_api = self.parameters.force_API_fetching if is_fafb else False

        fnc = FindNeuronConnection(
            sourceNeurons=source_neurons,
            targetNeurons=target_neurons,
            custom_source_name=custom_source_name,
            custom_target_name=custom_target_name,
            max_interlayer=max_interlayer,
            min_synapse_num=threshold,
            min_traversal_probability=0,
            min_ratio=0,
            dataset=dataset_name,
            saveas=fnc_output_path,
            verbose_mode=verbose_mode,
            skip_bodyId=self.parameters.skip_bodyId,
            label_mapper=self.label_mapper,
            pathfinding=self.parameters.pathfinding,
            graph_edge_limit_bodyid=self.parameters.graph_edge_limit_bodyid,
            max_paths_bodyid=self.parameters.max_paths_bodyid,
            edgeN_limit=self.parameters.edgeN_limit,
            search_columns=self.parameters.search_columns,
            force_API_fetching=use_force_api,
            cache_only=self.parameters.cache_only,
            separate_hemispheres=self.parameters.separate_hemispheres,
            symmetry_analysis=self.parameters.symmetry_analysis,
            keep_only_hemisphere_conserved_connections=self.parameters.keep_only_hemisphere_conserved_connections,
            # The delegated run must NOT pre-filter with dataset-native
            # labels: the comparison-level drop_untyped runs AFTER
            # standardized cross-dataset labels are resolved (post
            # label-mapping), which is the later, authoritative timing.
            drop_untyped=False,
            # Density curves for every mode (see run_path_analysis).
            capture_density=True,
        )

        fnc.InitializeNeuronInfo()
        # Same provenance marker as run_path_analysis: the UI runner parses
        # this to record query history; without it the replay flow (the
        # default path) never reports the resolved set sizes and the tab
        # silently skips recording.
        source_df = getattr(fnc, "source_df", None)
        target_df = getattr(fnc, "target_df", None)
        print(
            f"[DROCAT][neuron-match] source={len(source_df) if source_df is not None else 0} "
            f"target={len(target_df) if target_df is not None else 0}",
            flush=True,
        )
        if self.parameters.path_mode == 'shortest':
            # Not reachable: replay is only enabled for path_mode='all'
            fnc.FindShortestPath(find_reciprocal=self.parameters.find_reciprocal)
            return None
        try:
            results = fnc.FindAllPathMultiThreshold(
                thresholds,
                find_reciprocal=self.parameters.find_reciprocal,
            )
        except Exception as e:
            # Any orchestrator failure degrades to the legacy
            # per-threshold enumeration rather than losing the run.
            self._log(f"  Replay run failed ({e}) — falling back to "
                      f"per-threshold enumeration", level='warn')
            return None

        if results.get('_fallback'):
            return None

        for t in thresholds:
            meta = results.get(t)
            if not meta:
                continue
            self._replay_results[(dataset_name, t)] = dict(meta)

        return results

    def _load_multi_threshold_result(self, dataset_name: str, threshold: int) -> pd.DataFrame:
        """Read one replayed threshold's connection table from its folder.

        The multi-threshold orchestrator materialized the folder already;
        reuse the standard cached-result loader (the same file the next
        ``_try_load_cached`` resume would find). Applies the same post-load
        label-mapper fold-in as ``_load_fnc_results`` so mapper runs match
        the legacy per-threshold flow.
        """
        df = self._try_load_cached(dataset_name, threshold,
                                   remove_stale=True)
        if df is None:
            df = pd.DataFrame()
        return self._finalize_loaded_result(dataset_name, threshold, df)
    
    def _run_all_edge_analyses(self, skip_existing: bool = True) -> Dict[str, Dict[int, pd.DataFrame]]:
        """
        Run edge-based analyses for all datasets and thresholds.
        
        Optimized Edge Mode Workflow:
        1. Run the path tool (FindAllPath, or FindShortestPath when
           path_mode='shortest') at LOWEST threshold to get bodyId connections
        2. Filter bodyId data by ALL thresholds and aggregate ALL to type-level immediately
        3. Run the path tool for remaining thresholds (only for output consistency, no aggregation)
        
        This approach:
        - Fetches connections only once (at lowest threshold)
        - Computes all edge aggregations upfront using FastGraph
        - Runs the path tool for other thresholds only to generate path output files
        """
        from core.fast_graph import FastGraph

        dataset_names = self.parameters.get_dataset_names()
        path_tool = 'FindShortestPath' if self.parameters.path_mode == 'shortest' else 'FindAllPath'

        for dataset_name in dataset_names:
            if dataset_name not in self.raw_results:
                self.raw_results[dataset_name] = {}

            # Feature E: this dataset's OWN threshold list (vertical
            # comparison overrides); the lowest drives the single fetch.
            dataset_thresholds = sorted(
                self.parameters.get_thresholds_for_dataset(dataset_name))
            lowest_threshold = dataset_thresholds[0] if dataset_thresholds else None

            self._log(f"Edge mode analysis for \033[94m{dataset_name}\033[0m ({len(dataset_thresholds)} thresholds: {dataset_thresholds})")

            # ===== Step 1: Run the path tool for LOWEST threshold =====
            self._log(f"Running {path_tool} for {dataset_name} @ threshold={lowest_threshold}")
            self.run_path_analysis(dataset_name, lowest_threshold, verbose_mode='simple')

            # ===== Step 2: Get bodyId-level connections =====
            bodyid_df, label_map = self._get_bodyid_connections_for_dataset(
                dataset_name, lowest_threshold, skip_existing=True
            )

            if bodyid_df is None or bodyid_df.empty:
                self._log(f"Warning: No bodyId connections found for {dataset_name}")
                for threshold in dataset_thresholds:
                    self.raw_results[dataset_name][threshold] = pd.DataFrame()
                # Still run the path tool for other thresholds for output consistency
                remaining_thresholds = [t for t in dataset_thresholds if t != lowest_threshold]
                for threshold in tqdm(remaining_thresholds, desc=f"  {dataset_name} thresholds", leave=False, unit="thr"):
                    self.run_path_analysis(dataset_name, threshold, verbose_mode='silent')
                continue

            # Edge mode aggregates the bodyId frame directly instead of using
            # the already-mapped frame returned by _load_fnc_results.  Apply
            # the same label-mapping -> untyped-filter order here before any
            # type aggregation; otherwise the checkbox silently did nothing
            # for edge-mode comparison results.
            bodyid_df = self._apply_label_mapping_to_dataframe(
                dataset_name, lowest_threshold, bodyid_df)
            label_map = self._build_label_map_from_df(bodyid_df)
            self._log(f"Loaded {len(bodyid_df)} bodyId-level connections from threshold={lowest_threshold}")

            # Get source/target types for path finding
            source_types = set(self.parameters.get_source_neurons_for_dataset(dataset_name))
            target_types = set(self.parameters.get_target_neurons_for_dataset(dataset_name))
            max_layers = self.parameters.max_interlayer + 1
            if self.parameters.path_mode == 'shortest' and self.parameters.max_interlayer <= 0:
                max_layers = None  # unlimited depth in shortest mode

            # ===== Step 3: Filter and aggregate for ALL thresholds at once =====
            self._log(f"Aggregating edges for all {len(dataset_thresholds)} thresholds...")
            for threshold in tqdm(dataset_thresholds, desc=f"  Aggregating", leave=False, unit="thr"):
                self._process_threshold_aggregation(
                    dataset_name, threshold, bodyid_df, label_map,
                    source_types, target_types, max_layers, skip_existing,
                    path_mode=self.parameters.path_mode
                )

            # ===== Step 4: Run the path tool for remaining thresholds (output consistency only) =====
            remaining_thresholds = [t for t in dataset_thresholds if t != lowest_threshold]
            if remaining_thresholds:
                for threshold in tqdm(remaining_thresholds, desc=f"  {dataset_name} thresholds", leave=False, unit="thr"):
                    self.run_path_analysis(dataset_name, threshold, verbose_mode='silent')

        self._complete_path_run_meta()

        # Edge-mode truth (F-XD-004, ordering fixed after round-6 F-P3):
        # the compared data is exactly `weight >= requested` — the path
        # runs above exist for output consistency, and their tau/budget
        # provenance must not masquerade as the edge filter's applied
        # threshold. This must run AFTER _complete_path_run_meta(): that
        # pass re-derives tau/bitten from the per-run FNC state and would
        # clobber an earlier neutralization (exactly what the Windows
        # round-6 run observed — tau repopulated, side_path_run absent).
        if self.parameters.comparison_mode == 'edge':
            self._neutralize_edge_mode_path_meta()

        self._export_untyped_drop_records()
        self._export_effective_threshold_banner()
        self._log(f"Completed edge analysis for {len(dataset_names)} datasets")
        return self.raw_results
    
    def _process_threshold_aggregation(
        self,
        dataset_name: str,
        threshold: int,
        bodyid_df: pd.DataFrame,
        label_map: Dict,
        source_types: set,
        target_types: set,
        max_layers: Optional[int],
        skip_existing: bool,
        path_mode: str = 'all'
    ):
        """Process edge aggregation for a single threshold."""
        # Check if already computed in memory
        if skip_existing and threshold in self.raw_results[dataset_name]:
            existing = self.raw_results[dataset_name][threshold]
            if not existing.empty:
                self._log(f"  Skipping threshold={threshold} (already computed)")
                return
        
        # Filter bodyId edges by current threshold, then apply the shared
        # comparison-level untyped predicate after standardized labels.
        filtered_df = bodyid_df[bodyid_df['weight'] >= threshold].copy()
        filtered_df = self._drop_untyped_neurons(
            dataset_name, threshold, filtered_df)
        
        if filtered_df.empty:
            self._log(f"  threshold={threshold}: No edges meet threshold")
            result_df = pd.DataFrame()
        else:
            # Aggregate to type-level using FastGraph
            result_df = self._aggregate_and_find_paths(
                filtered_df, label_map, source_types, target_types,
                max_layers, dataset_name, threshold, path_mode=path_mode
            )
            valid_count = result_df['has_valid_path'].sum() if not result_df.empty and 'has_valid_path' in result_df.columns else 0
            self._log(f"  threshold={threshold}: {len(result_df)} type edges ({valid_count} with valid paths)")
        
        self.raw_results[dataset_name][threshold] = result_df
        
        # Save aggregated edge data to edge_mode_data folder
        if self.parameters.output_folder and not result_df.empty:
            self._save_edge_mode_result(dataset_name, threshold, result_df)
    
    def _get_bodyid_connections_for_dataset(
        self, 
        dataset_name: str, 
        threshold: int,
        skip_existing: bool = True,
        run_findallpath: bool = False
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Get bodyId-level connections for a dataset at the lowest threshold.
        
        Args:
            dataset_name: Dataset identifier
            threshold: Threshold level
            skip_existing: Skip if cached data exists
            run_findallpath: If True, run FindAllPath to generate data (default False)
        
        Returns:
            Tuple of (bodyId DataFrame, label_map dict mapping bodyId -> type)
        """
        # Check for cached bodyId data
        if self.parameters.output_folder:
            cache_dir = self._resolve_dataset_output_path(
                dataset_name, threshold)
            bodyid_file = os.path.join(cache_dir, 'data_details', 'connection_info_bodyId.csv')
            
            if skip_existing and os.path.exists(bodyid_file):
                try:
                    df = self._read_csv(bodyid_file)
                    if not df.empty:
                        # Build label map from the data
                        label_map = self._build_label_map_from_df(df)
                        self._log(f"Loaded cached bodyId data from {bodyid_file}")
                        return df, label_map
                except Exception as e:
                    self._log(f"Warning: Could not load cached bodyId data: {e}")
        
        # Only run the path tool if explicitly requested (avoids duplicate runs)
        if run_findallpath:
            path_tool = 'FindShortestPath' if self.parameters.path_mode == 'shortest' else 'FindAllPath'
            self._log(f"Running {path_tool} for {dataset_name} @ threshold={threshold} to get bodyId connections")
            self.run_path_analysis(dataset_name, threshold)
            
            # Try to load the bodyId file that was generated
            if self.parameters.output_folder:
                cache_dir = self._resolve_dataset_output_path(
                    dataset_name, threshold)
                bodyid_file = os.path.join(cache_dir, 'data_details', 'connection_info_bodyId.csv')
                
                if os.path.exists(bodyid_file):
                    try:
                        df = self._read_csv(bodyid_file)
                        if not df.empty:
                            label_map = self._build_label_map_from_df(df)
                            return df, label_map
                    except Exception:
                        pass
        
        # Fallback: query edges directly (this is the fast path when skip_bodyId=True)
        self._log(f"Querying bodyId edges directly for {dataset_name}")
        source_neurons = self.parameters.get_source_neurons_for_dataset(dataset_name)
        target_neurons = self.parameters.get_target_neurons_for_dataset(dataset_name)
        
        df = self._query_edges_for_dataset(dataset_name, source_neurons, target_neurons, threshold)
        if df is not None and not df.empty:
            label_map = self._build_label_map_from_df(df)
            return df, label_map
        
        return None, {}
    
    def _build_label_map_from_df(self, df: pd.DataFrame) -> Dict:
        """Build bodyId -> type label map from connection DataFrame."""
        label_map = {}
        
        # Check column names
        pre_id_col = 'bodyId_pre' if 'bodyId_pre' in df.columns else None
        post_id_col = 'bodyId_post' if 'bodyId_post' in df.columns else None
        pre_type_col = 'type_pre' if 'type_pre' in df.columns else None
        post_type_col = 'type_post' if 'type_post' in df.columns else None
        
        if pre_id_col and pre_type_col:
            for _, row in df[[pre_id_col, pre_type_col]].drop_duplicates().iterrows():
                if pd.notna(row[pre_id_col]) and pd.notna(row[pre_type_col]):
                    label_map[row[pre_id_col]] = row[pre_type_col]
        
        if post_id_col and post_type_col:
            for _, row in df[[post_id_col, post_type_col]].drop_duplicates().iterrows():
                if pd.notna(row[post_id_col]) and pd.notna(row[post_type_col]):
                    label_map[row[post_id_col]] = row[post_type_col]
        
        return label_map
    
    def _aggregate_and_find_paths(
        self,
        bodyid_df: pd.DataFrame,
        label_map: Dict,
        source_types: set,
        target_types: set,
        max_layers: Optional[int],
        dataset_name: str,
        threshold: int,
        path_mode: str = 'all'
    ) -> pd.DataFrame:
        """
        Aggregate bodyId edges to type-level and find valid paths.
        
        Args:
            bodyid_df: DataFrame with bodyId-level connections (already filtered by threshold)
            label_map: Dict mapping bodyId -> type
            source_types: Set of source neuron types
            target_types: Set of target neuron types
            max_layers: Maximum path length (None = unlimited, shortest mode only)
            dataset_name: Dataset name for metadata
            threshold: Current threshold for metadata
            path_mode: 'all' (every path within max_layers) or 'shortest'
                (edges valid only when they lie on a per-pair minimum-hop path)
            
        Returns:
            DataFrame with type-level edges and has_valid_path flag
        """
        from core.fast_graph import FastGraph
        
        # Determine column names
        pre_id_col = 'bodyId_pre' if 'bodyId_pre' in bodyid_df.columns else 'pre_pt_root_id'
        post_id_col = 'bodyId_post' if 'bodyId_post' in bodyid_df.columns else 'post_pt_root_id'
        weight_col = 'weight' if 'weight' in bodyid_df.columns else 'syn_count'
        
        # Build bodyId-level graph
        G_bodyid = FastGraph()
        G_bodyid.build_from_dataframe(bodyid_df, pre_id_col, post_id_col, weight_col)
        
        # Aggregate to type-level graph
        G_type, edge_df = G_bodyid.aggregate_by_label(label_map, return_edge_df=True)
        
        if edge_df.empty:
            return pd.DataFrame()
        
        # Find valid paths at type level
        # Get all type nodes that are sources or targets
        graph_source_types = [t for t in source_types if G_type.has_node(t)]
        graph_target_types = [t for t in target_types if G_type.has_node(t)]
        
        # Find paths (all paths, or only per-pair shortest paths)
        valid_edges = set()
        if graph_source_types and graph_target_types:
            try:
                if path_mode == 'shortest':
                    # Edges are valid only when they lie on a per-pair
                    # minimum-hop path; cutoff None = unlimited depth.
                    paths = list(G_type.find_paths_shortest(
                        graph_source_types, graph_target_types, cutoff=max_layers
                    ))
                else:
                    # Use memoized DFS for efficiency
                    paths = list(G_type.find_paths_memoized_dfs(
                        graph_source_types, graph_target_types, max_layers, 
                        direction='backward', verbose=False
                    ))
                
                # Extract edges from paths
                for path in paths:
                    for i in range(len(path) - 1):
                        valid_edges.add((path[i], path[i+1]))
                        
                self._log(f"Found {len(paths)} paths, {len(valid_edges)} unique edges in paths", 'debug')
            except Exception as e:
                self._log(f"Warning: Path finding failed: {e}")
        
        # Add metadata to edge DataFrame
        edge_df['has_valid_path'] = edge_df.apply(
            lambda r: (r['type_pre'], r['type_post']) in valid_edges, axis=1
        )
        edge_df['dataset'] = dataset_name
        edge_df['threshold'] = threshold
        
        # Compute additional metrics
        # Get total_post from bodyid_df if available
        if 'total_post' in bodyid_df.columns:
            # Aggregate total_post by type_post (take first value since it should be same for all bodyIds of same type)
            total_post_map = {}
            if 'type_post' in bodyid_df.columns:
                for post_type in edge_df['type_post'].unique():
                    mask = bodyid_df['type_post'] == post_type
                    if mask.any():
                        total_post_map[post_type] = bodyid_df.loc[mask, 'total_post'].iloc[0]
            
            edge_df['total_post'] = edge_df['type_post'].map(total_post_map)
            edge_df['connection_ratio'] = edge_df['weight'] / edge_df['total_post'].fillna(1)
        
        # Compute traversal probability if possible
        # traversal_prob = weight / sum(all outgoing weights from pre)
        outgoing_weights = edge_df.groupby('type_pre')['weight'].sum().to_dict()
        edge_df['traversal_probability'] = edge_df.apply(
            lambda r: r['weight'] / outgoing_weights.get(r['type_pre'], 1), axis=1
        )
        
        # Add conn_layer info based on path structure
        # This requires knowing which types are in which layer
        # For now, infer from source/target membership
        def get_conn_layer(row):
            pre, post = row['type_pre'], row['type_post']
            if pre in source_types:
                return '0->1'
            elif post in target_types:
                return '1->2'  # Assuming 2-hop max
            else:
                return '1->2'  # Default intermediate
        
        edge_df['conn_layer'] = edge_df.apply(get_conn_layer, axis=1)
        
        return edge_df
    
    # =========================================================================
    # Dataset Metadata Collection
    # =========================================================================
    
    def _get_datasets_folder(self) -> str:
        """Get the path to the datasets folder."""
        # Assume datasets folder is at project root level
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(src_dir)
        return os.path.join(project_root, 'datasets')
    
    def _get_metadata_path(self, dataset_name: str) -> str:
        """Get path to the metadata file for a dataset."""
        safe_name = self.parameters._sanitize_name(dataset_name)
        datasets_folder = self._get_datasets_folder()
        return os.path.join(datasets_folder, safe_name, f'{safe_name}_metadata.json')
    
    def _load_cached_metadata(self, dataset_name: str) -> Optional[Dict]:
        """Try to load cached metadata from local file."""
        metadata_path = self._get_metadata_path(dataset_name)
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self._log(f"Warning: Failed to load cached metadata for {dataset_name}: {e}")
        return None
    
    def _save_metadata(self, dataset_name: str, metadata: Dict) -> None:
        """Save metadata to local cache file.

        A failed collection (``source == 'error'``, i.e. the placeholder built
        by :meth:`_create_empty_metadata`) is never written to disk: the file
        lives inside ``datasets/<name>/``, so persisting it created a
        metadata-only folder that the dataset listing reported as a
        half-installed release, and a transient failure became sticky
        (2026-09-18 retest, F9).  The error payload stays in-memory for this
        call only, so the next run retries the fetch.
        """
        if (metadata or {}).get('source') == 'error':
            self._log(f"Not caching error metadata for {dataset_name}; the "
                      "next collection will retry the fetch")
            return
        metadata_path = self._get_metadata_path(dataset_name)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            self._log(f"Saved metadata to: {metadata_path}")
        except Exception as e:
            self._log(f"Warning: Failed to save metadata for {dataset_name}: {e}")
    
    def _fetch_neuprint_metadata(self, dataset_name: str) -> Dict:
        """Fetch metadata from NeuPrint server."""
        try:
            from neuprint import Client
            
            # Parse dataset name for server connection
            token = self.parameters.resolve_token()
            client = Client('neuprint.janelia.org', dataset=dataset_name, token=token)
            
            # Get neuron counts
            q = "MATCH (n:Neuron) RETURN count(n) AS total_neurons"
            result = client.fetch_custom(q)
            total_neurons = int(result.iloc[0]['total_neurons']) if not result.empty else 0
            
            # Get typed neuron count
            q = "MATCH (n:Neuron) WHERE n.type IS NOT NULL AND n.type <> '' RETURN count(n) AS typed_neurons"
            result = client.fetch_custom(q)
            typed_neurons = int(result.iloc[0]['typed_neurons']) if not result.empty else 0
            
            # Get synapse counts
            q = "MATCH (n:Neuron) RETURN sum(n.pre) AS total_pre, sum(n.post) AS total_post"
            result = client.fetch_custom(q)
            total_pre = int(result.iloc[0]['total_pre']) if not result.empty and result.iloc[0]['total_pre'] else 0
            total_post = int(result.iloc[0]['total_post']) if not result.empty and result.iloc[0]['total_post'] else 0
            
            # Get ROI coverage
            q = "MATCH (m:Meta) RETURN m.primaryRois AS rois"
            result = client.fetch_custom(q)
            rois = result.iloc[0]['rois'] if not result.empty else []
            
            # Get neuron counts per ROI
            roi_counts = {}
            if rois:
                for roi in rois[:20]:  # Limit to first 20 ROIs for performance
                    try:
                        q = f"MATCH (n:Neuron) WHERE n.`{roi}` = true RETURN count(n) AS count"
                        result = client.fetch_custom(q)
                        roi_counts[roi] = int(result.iloc[0]['count']) if not result.empty else 0
                    except Exception:
                        pass
            
            metadata = {
                'dataset': dataset_name,
                'source': 'neuprint',
                'fetched_at': datetime.now().isoformat(),
                'neuron_counts': {
                    'total': total_neurons,
                    'typed': typed_neurons,
                    'untyped': total_neurons - typed_neurons,
                    'type_coverage': typed_neurons / total_neurons if total_neurons > 0 else 0
                },
                'synapse_counts': {
                    'total_presynaptic': total_pre,
                    'total_postsynaptic': total_post,
                    'total': total_pre + total_post
                },
                'roi_coverage': {
                    'roi_list': rois if rois else [],
                    'roi_count': len(rois) if rois else 0,
                    'neuron_counts_per_roi': roi_counts
                },
                'coverage_notes': self._get_coverage_notes(dataset_name)
            }

            # Feature A: per-neuron synapse density. NeuPrint neuron tables
            # carry pre + post, so one fetch_neurons call provides everything;
            # on failure the block is simply absent (density stays optional).
            try:
                from .metadata_density import compute_synapse_density
                neuron_df = client.fetch_neurons(
                    'MATCH (n:Neuron) RETURN n.bodyId AS bodyId, '
                    'n.pre AS pre, n.post AS post',
                    format='pandas')
                density = compute_synapse_density(
                    neuron_df, is_flywire_source=False)
                if density:
                    metadata['synapse_density'] = density
                    metadata['density_generated_at'] = datetime.now().isoformat()
            except Exception as density_exc:
                self._log(f"Note: synapse density not computed for {dataset_name}: "
                          f"{density_exc}", level='debug')

            return metadata

        except Exception as e:
            self._log(f"Warning: Failed to fetch NeuPrint metadata for {dataset_name}: {e}")
            return self._create_empty_metadata(dataset_name, str(e))

    def _fetch_local_metadata(self, dataset_name: str) -> Dict:
        """Fetch metadata from local dataset files."""
        safe_name = self.parameters._sanitize_name(dataset_name)
        datasets_folder = self._get_datasets_folder()
        dataset_path = os.path.join(datasets_folder, safe_name)
        
        neuron_file = os.path.join(dataset_path, f'{safe_name}_allneurons_neuron_df.csv')
        neuron_parquet = os.path.join(dataset_path, f'{safe_name}_allneurons_neuron_df.parquet')
        
        # Try to load neuron data
        neuron_df = None
        if os.path.exists(neuron_parquet):
            try:
                neuron_df = pd.read_parquet(neuron_parquet)
            except Exception:
                pass
        if neuron_df is None and os.path.exists(neuron_file):
            try:
                neuron_df = self._read_csv(neuron_file)
            except Exception:
                pass
        
        if neuron_df is None:
            return self._create_empty_metadata(dataset_name, "No local data found")
        
        total_neurons = len(neuron_df)
        
        # Count typed neurons
        type_col = None
        for col in ['type', 'cell_type', 'hemibrain_type', 'class']:
            if col in neuron_df.columns:
                type_col = col
                break
        
        if type_col:
            # 'Unknown' is the untyped placeholder — exclude it from the
            # typed count (otherwise coverage reads a misleading 100%).
            _tv = neuron_df[type_col].astype('string').str.strip().fillna('')
            typed_neurons = int(((_tv != '') & (_tv != 'Unknown')).sum())
        else:
            typed_neurons = 0
        
        # Synapse counts
        pre_col = 'pre' if 'pre' in neuron_df.columns else None
        post_col = 'post' if 'post' in neuron_df.columns else None
        
        total_pre = int(neuron_df[pre_col].sum()) if pre_col else 0
        total_post = int(neuron_df[post_col].sum()) if post_col else 0
        
        # Check for ROI data (parquet from current pulls, CSV from older ones)
        roi_file = os.path.join(dataset_path, f'{safe_name}_allneurons_roi_count_df.csv')
        roi_parquet = os.path.join(dataset_path, f'{safe_name}_allneurons_roi_count_df.parquet')
        roi_counts = {}
        rois = []

        roi_df = None
        if os.path.exists(roi_parquet):
            try:
                roi_df = pd.read_parquet(roi_parquet)
            except Exception:
                pass
        if roi_df is None and os.path.exists(roi_file):
            try:
                roi_df = self._read_csv(roi_file)
            except Exception:
                pass

        if roi_df is not None:
            try:
                # Get ROI columns (usually all except bodyId)
                roi_cols = [c for c in roi_df.columns if c not in ['bodyId', 'Unnamed: 0']]
                rois = roi_cols
                for col in roi_cols[:20]:  # Limit
                    roi_counts[col] = int((roi_df[col] > 0).sum())
            except Exception:
                pass
        
        metadata = {
            'dataset': dataset_name,
            'source': 'local',
            'fetched_at': datetime.now().isoformat(),
            'neuron_counts': {
                'total': total_neurons,
                'typed': typed_neurons,
                'untyped': total_neurons - typed_neurons,
                'type_coverage': typed_neurons / total_neurons if total_neurons > 0 else 0
            },
            'synapse_counts': {
                'total_presynaptic': total_pre,
                'total_postsynaptic': total_post,
                'total': total_pre + total_post
            },
            'roi_coverage': {
                'roi_list': rois,
                'roi_count': len(rois),
                'neuron_counts_per_roi': roi_counts
            },
            'coverage_notes': self._get_coverage_notes(dataset_name)
        }

        # Feature A: per-neuron synapse density. Local FAFB/BANC
        # tables carry post only — pre is derived from
        # merged_connections.parquet and flagged in pre_source.
        try:
            from .metadata_density import compute_synapse_density
            density = compute_synapse_density(
                neuron_df, dataset_path=dataset_path)
            if density:
                metadata['synapse_density'] = density
                metadata['density_generated_at'] = datetime.now().isoformat()
        except Exception as density_exc:
            self._log(f"Note: synapse density not computed for {dataset_name}: "
                      f"{density_exc}", level='debug')

        return metadata
    
    def _create_empty_metadata(self, dataset_name: str, error_msg: str) -> Dict:
        """Create empty metadata structure with error message.

        In-memory only: :meth:`_save_metadata` refuses to persist an error
        payload, so a failed collection leaves no sidecar behind and the next
        run retries (F9).
        """
        return {
            'dataset': dataset_name,
            'source': 'error',
            'fetched_at': datetime.now().isoformat(),
            'error': error_msg,
            'neuron_counts': {'total': 0, 'typed': 0, 'untyped': 0, 'type_coverage': 0},
            'synapse_counts': {'total_presynaptic': 0, 'total_postsynaptic': 0, 'total': 0},
            'roi_coverage': {'roi_list': [], 'roi_count': 0, 'neuron_counts_per_roi': {}},
            'coverage_notes': self._get_coverage_notes(dataset_name)
        }
    
    def _get_coverage_notes(self, dataset_name: str) -> str:
        """Get known coverage notes for a dataset."""
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
    
    def collect_dataset_metadata(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """
        Collect metadata for all datasets.
        
        Metadata is cached locally in datasets/{dataset}/{dataset}_metadata.json.
        If cached file exists and force_refresh=False, uses cached data --
        except a cached error placeholder (``source == 'error'``), which is
        never terminal: the fetch is retried (F9).
        
        Args:
            force_refresh: If True, fetch fresh metadata even if cached exists
            
        Returns:
            Dict mapping dataset name to metadata dict
        """
        self._log("Collecting dataset metadata...")
        
        all_metadata = {}
        
        for dataset_name in self.parameters.get_dataset_names():
            # Try cached first
            if not force_refresh:
                cached = self._load_cached_metadata(dataset_name)
                if cached and cached.get('source') != 'error':
                    self._log(f"Loaded cached metadata for {dataset_name}")
                    all_metadata[dataset_name] = cached
                    continue
                if cached:
                    self._log(f"Ignoring cached error metadata for "
                              f"{dataset_name} (retrying the fetch)")
            
            # Fetch fresh metadata
            self._log(f"Fetching metadata for {dataset_name}...")
            
            # Determine if local or NeuPrint dataset
            if is_local_connectome_dataset(dataset_name):
                metadata = self._fetch_local_metadata(dataset_name)
            else:
                metadata = self._fetch_neuprint_metadata(dataset_name)
            
            # Save to cache
            self._save_metadata(dataset_name, metadata)
            all_metadata[dataset_name] = metadata
        
        # Store for later use
        self._dataset_metadata = all_metadata
        
        return all_metadata
    
    def generate_metadata_comparison_table(self) -> pd.DataFrame:
        """
        Generate a comparison table from collected metadata.
        
        Returns:
            DataFrame comparing key metrics across datasets
        """
        if not hasattr(self, '_dataset_metadata') or not self._dataset_metadata:
            self.collect_dataset_metadata()
        
        rows = []
        for dataset_name, metadata in self._dataset_metadata.items():
            nc = metadata.get('neuron_counts', {})
            sc = metadata.get('synapse_counts', {})
            rc = metadata.get('roi_coverage', {})
            sd = metadata.get('synapse_density', {})

            rows.append({
                'dataset': dataset_name,
                'total_neurons': nc.get('total', 0),
                'typed_neurons': nc.get('typed', 0),
                'untyped_neurons': nc.get('untyped', 0),
                'type_coverage_pct': round(nc.get('type_coverage', 0) * 100, 2),
                'total_presynaptic': sc.get('total_presynaptic', 0),
                'total_postsynaptic': sc.get('total_postsynaptic', 0),
                'total_synapses': sc.get('total', 0),
                # Feature A: median synapses (pre+post) per neuron — the
                # whole-dataset density used by the threshold-equivalence
                # note (NaN when the metadata predates the density block).
                'median_synapse_density_per_neuron': round(
                    sd['per_neuron_median'], 2)
                    if sd.get('per_neuron_median') is not None else None,
                'synapse_density_pre_source': sd.get('pre_source', ''),
                'roi_count': rc.get('roi_count', 0),
                # N7: fall back to the built-in coverage note when the cached
                # metadata JSON predates note generation.
                'coverage_notes': metadata.get('coverage_notes')
                    or self._get_coverage_notes(dataset_name)
            })
        
        return pd.DataFrame(rows)
    
    def _query_fingerprint(self) -> dict:
        """Identity of the current comparison query for cache validation.

        A cached ``connections_edge.csv`` is only a valid answer for the
        query that produced it; reusing a folder across queries (or modes)
        must not silently load the previous answer (F-XD-003).
        """
        p = self.parameters
        def _digest(value) -> str:
            import hashlib
            joined = "|".join(sorted(str(v) for v in value)) \
                if isinstance(value, (list, tuple, set)) else str(value)
            return hashlib.md5(joined.encode("utf-8", "surrogatepass")) \
                .hexdigest()[:12]
        return {
            'source_neurons': _digest(getattr(p, 'source_neurons', None) or []),
            'target_neurons': _digest(getattr(p, 'target_neurons', None) or []),
            'comparison_mode': str(getattr(p, 'comparison_mode', '')),
            'max_interlayer': getattr(p, 'max_interlayer', None),
            'pathfinding': str(getattr(p, 'pathfinding', '')),
        }

    def _cached_result_matches_query(self, dirpath: str) -> bool:
        """Validate a dataset-threshold folder against the current query.

        Sidecar absent (legacy folder) -> accepted with a debug note; the
        fingerprint only starts protecting folders it was written to.
        """
        import json as _json
        sidecar = os.path.join(dirpath, "connections_edge.fingerprint.json")
        if not os.path.exists(sidecar):
            self._log("Cached result has no query fingerprint "
                      "(legacy folder) — accepting", 'debug')
            return True
        try:
            with open(sidecar, "r", encoding="utf-8") as fh:
                recorded = _json.load(fh)
        except (OSError, ValueError):
            return False
        return recorded == self._query_fingerprint()

    def _try_load_cached(self, dataset_name: str, threshold: int,
                         remove_stale: bool = False) -> Optional[pd.DataFrame]:
        """Try to load cached result from disk.

        ``remove_stale=True`` (the replay loader's shape): a fingerprint
        mismatch DELETES the stale ``connections_edge.csv`` + sidecar pair
        instead of leaving it to be re-rejected by every later reader of
        this folder — round-7 J6 observed the stale pair surviving a
        mismatched re-derivation, so the folder kept claiming the old
        query (Windows report, 0 of 2 sidecars rewritten).
        """
        if not self.parameters.output_folder:
            return None
        
        output_dir = self.parameters.get_dataset_output_path(dataset_name, threshold)
        
        # First try our own cached connections_edge.csv (edge mode output)
        filepath = os.path.join(output_dir, "connections_edge.csv")
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            if not self._cached_result_matches_query(output_dir):
                self._log(
                    f"Cached {dataset_name} @ {threshold} was written for a "
                    "different query — ignoring it", 'always')
                if remove_stale:
                    try:
                        os.unlink(filepath)
                        os.unlink(os.path.join(
                            output_dir,
                            "connections_edge.fingerprint.json"))
                        self._log(
                            f"Removed the stale pair under {output_dir} "
                            "(named a different query)", 'always')
                    except OSError as exc:
                        self._log(f"Could not remove the stale pair: {exc}",
                                  'always')
            else:
                try:
                    df = self._read_csv(filepath)
                    if not df.empty:
                        self._log(f"Loading cached: {dataset_name} @ {threshold}", 'debug')
                        return df
                except (pd.errors.EmptyDataError, Exception):
                    pass  # File is empty or corrupted, try other sources
        
        # Also try legacy paths.csv (for backward compatibility)
        filepath = os.path.join(output_dir, "paths.csv")
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            try:
                df = self._read_csv(filepath)
                if not df.empty:
                    self._log(f"Loading cached: {dataset_name} @ {threshold}", 'debug')
                    return df
            except (pd.errors.EmptyDataError, Exception):
                pass
        
        # Also try to find the FindNeuronConnection output file (connection_info_bodyId.csv)
        conn_file = os.path.join(output_dir, 'data_details', 'connection_info_bodyId.csv')
        if os.path.exists(conn_file) and os.path.getsize(conn_file) > 0:
            try:
                df = self._read_csv(conn_file)
                if not df.empty:
                    self._log(f"Loading cached: {dataset_name} @ {threshold}", 'debug')
                    # Add dataset info if missing
                    if 'dataset' not in df.columns:
                        df['dataset'] = dataset_name
                        df['threshold'] = threshold
                    return df
            except (pd.errors.EmptyDataError, Exception):
                pass
        
        # Also try connection_type.csv (type-level path mode output)
        conn_type_file = os.path.join(output_dir, 'data_details', 'connection_type.csv')
        if os.path.exists(conn_type_file) and os.path.getsize(conn_type_file) > 0:
            try:
                df = self._read_csv(conn_type_file)
                if not df.empty:
                    self._log(f"Loading cached: {dataset_name} @ {threshold}", 'debug')
                    # Add dataset info if missing
                    if 'dataset' not in df.columns:
                        df['dataset'] = dataset_name
                        df['threshold'] = threshold
                    return df
            except (pd.errors.EmptyDataError, Exception):
                pass
        
        return None
    
    def _save_result(self, dataset_name: str, threshold: int, df: pd.DataFrame):
        """Save result to disk (edge mode output)."""
        if not self.parameters.output_folder:
            return
        
        # Don't save empty DataFrames - they cause read errors later
        if df is None or df.empty:
            self._log(f"Skipping save for {dataset_name} @ {threshold} (empty result)", 'debug')
            return
        
        dirpath = self.parameters.get_dataset_output_path(dataset_name, threshold)
        os.makedirs(dirpath, exist_ok=True)
        
        # Save to connections_edge.csv (edge mode cached version)
        filepath = os.path.join(dirpath, "connections_edge.csv")
        self._save_csv(df, filepath)
        try:
            with open(os.path.join(dirpath,
                                   "connections_edge.fingerprint.json"),
                      "w", encoding="utf-8") as fh:
                json.dump(self._query_fingerprint(), fh, indent=2)
        except (OSError, TypeError, ValueError):
            pass
        self._log_file(filepath)
    
    def _save_edge_mode_result(self, dataset_name: str, threshold: int, df: pd.DataFrame):
        """Save aggregated edge mode result to edge_mode_data folder."""
        if not self.parameters.output_folder:
            return
        
        # Don't save empty DataFrames
        if df is None or df.empty:
            self._log(f"Skipping edge_mode save for {dataset_name} @ {threshold} (empty result)", 'debug')
            return
        
        # Save to: comparison_results_{}/edge_mode_data/{dataset}/connections_edge_{threshold}.csv
        safe_name = self.parameters._sanitize_name(dataset_name)
        edge_mode_dir = os.path.join(
            self.parameters.full_output_path,
            'edge_mode_data',
            safe_name
        )
        os.makedirs(edge_mode_dir, exist_ok=True)
        
        # Save aggregated connections_edge_{threshold}.csv
        filepath = os.path.join(edge_mode_dir, f"connections_edge_{threshold}.csv")
        self._save_csv(df, filepath)
        self._log_file(filepath)
    
    # =========================================================================
    # Comparison Analysis
    # =========================================================================
    
    def run_comparison(self, skip_existing: bool = True) -> Dict[str, Any]:
        """
        Run full comparison analysis: path analysis + comparison metrics.
        
        This is the main entry point for running a complete comparison.
        
        Args:
            skip_existing: Skip if results already cached
            
        Returns:
            Dictionary with all comparison metrics and findings
        """
        if self.parameters and self.parameters.full_output_path:
            self._log(f"📁 Output folder: {self.parameters.full_output_path}")
        self._progress(1, 5, "Resolving datasets and thresholds")
        # Run path analyses
        mode_label = ("Running edge analyses"
                      if getattr(self.parameters, "comparison_mode", "path") == 'edge'
                      else "Running path analyses")
        self._progress(2, 5, mode_label)
        self.run_all_analyses(skip_existing=skip_existing)
        
        # Compute comparison metrics
        self._progress(3, 5, "Computing cross-dataset metrics")
        return self.run_comparison_analysis()

    def run_comparison_analysis(self) -> Dict[str, Any]:
        """
        Run full comparison analysis on results.

        Returns:
            Dictionary with all comparison metrics and findings
        """
        # Ensure analyses have been run
        if not self.raw_results:
            self.run_all_analyses()
        
        self._log("Computing comparison metrics")
        self._log("  Step 1/2: Generating comparison summary...")

        dataset_names = self.parameters.get_dataset_names()
        coverage = self.dataset_coverage()
        no_data = [ds for ds, info in coverage.items()
                   if info.get('status') != 'ok']
        if no_data:
            self._log(
                "⚠️ Dataset coverage warning: " + ", ".join(no_data)
                + " produced no data — analyses run on the remaining "
                "datasets only. Check dataset_data/<dataset>/run_log.txt.")

        # Get type mapper for auto type mapping (if enabled)
        type_mapper = self.parameters._auto_type_mapper if self.parameters.auto_type_mapping else None
        # One shared-resolver snapshot per run: canonical merge keys for
        # path/edge merging reuse this run's decision cache and load state.
        if type_mapper is not None and self._mapper_snapshot is None:
            from .type_resolver import get_mapper_snapshot
            self._mapper_snapshot = get_mapper_snapshot(type_mapper)

        if self.parameters.threshold_mode == 'combinations':
            queries = self.get_threshold_queries()
            merge_policy = self._merge_policy_or_none()
            policy_label_mapper = self._policy_label_mapper()
            summary = self.metrics.generate_comparison_summary_for_queries(
                results=self.raw_results,
                datasets=dataset_names,
                queries=queries,
                label_mapper=policy_label_mapper,
                type_mapper=type_mapper,
                max_edges_for_metrics=self.parameters.max_edges_for_metrics,
                merge_policy=merge_policy,
                hemi_aware=self.parameters.separate_hemispheres,
            )
            self._log("  Step 2/2: Calculating query similarities...")
            similarities = self.metrics.calculate_similarity_across_queries(
                results=self.raw_results,
                datasets=dataset_names,
                queries=queries,
                label_mapper=policy_label_mapper,
                path_data_func=self._get_path_data_for_query,
                type_mapper=type_mapper,
                max_edges_for_metrics=self.parameters.max_edges_for_metrics,
                merge_policy=merge_policy,
                hemi_aware=self.parameters.separate_hemispheres,
            )
        else:
            # Generate comprehensive summary. Pass label_mapper=None because
            # raw_results are already mapped.
            merge_policy = self._merge_policy_or_none()
            policy_label_mapper = self._policy_label_mapper()
            summary = self.metrics.generate_comparison_summary(
                results=self.raw_results,
                datasets=dataset_names,
                thresholds=self.parameters.thresholds,
                label_mapper=policy_label_mapper,
                type_mapper=type_mapper,
                max_edges_for_metrics=self.parameters.max_edges_for_metrics,
                merge_policy=merge_policy,
                hemi_aware=self.parameters.separate_hemispheres
            )

            self._log("  Step 2/2: Calculating cross-threshold similarities...")

            # Feature G: τ-collapse duplicates share their predecessor's path
            # set — skip their similarity computation (identical rows) and let
            # the report show the effective thresholds only.
            similarity_thresholds = self._analysis_thresholds()

            similarities = self.metrics.calculate_similarity_across_thresholds(
                results=self.raw_results,
                datasets=dataset_names,
                thresholds=similarity_thresholds,
                label_mapper=policy_label_mapper,
                path_data_func=self._get_path_data_for_threshold,
                type_mapper=type_mapper,
                max_edges_for_metrics=self.parameters.max_edges_for_metrics,
                merge_policy=merge_policy,
                hemi_aware=self.parameters.separate_hemispheres
            )
        summary['threshold_similarities'] = similarities

        # Cache per-threshold similarities for reuse in visualizations
        if not similarities.empty:
            if (
                self.parameters.threshold_mode == 'combinations'
                and 'query_id' in similarities.columns
            ):
                for query_id, query_sims in similarities.groupby('query_id'):
                    self._similarity_cache[query_id] = query_sims.copy()
            elif 'threshold' in similarities.columns:
                for threshold in self.parameters.thresholds:
                    thresh_sims = similarities[similarities['threshold'] == threshold]
                    if not thresh_sims.empty:
                        self._similarity_cache[threshold] = thresh_sims.copy()

        # Store for later use
        self.comparison_report = summary
        
        # Print intermediate type mapping summary if auto_type_mapping is enabled
        if self.parameters.auto_type_mapping and self.parameters._auto_type_mapper:
            self._print_intermediate_mapping_summary()
        
        return summary
    
    def get_cached_similarities(self, threshold: int) -> pd.DataFrame:
        """
        Get cached pairwise similarities at a threshold.
        
        Uses cached values if available, otherwise computes and caches.
        
        Args:
            threshold: Weight threshold
            
        Returns:
            DataFrame with pairwise similarities
        """
        if threshold in self._similarity_cache:
            return self._similarity_cache[threshold].copy()
        
        # Compute if not cached
        aligned = self.get_aligned_data(threshold)
        if aligned.empty:
            return pd.DataFrame()
        
        dataset_names = self.parameters.get_dataset_names()
        similarities = self.metrics.calculate_all_pairwise_similarities(
            aligned, dataset_names, threshold=1, include_advanced_metrics=True
        )
        
        # Cache for future use
        if not similarities.empty:
            self._similarity_cache[threshold] = similarities.copy()
        
        return similarities
    
    def get_aligned_data(self, threshold: int) -> pd.DataFrame:
        """
        Get aligned edge data at a specific threshold.
        
        In combination mode the argument is a query id/label; passing a bare
        scalar integer raises unless it uniquely identifies a uniform query
        row, because the union threshold is not a comparison point.
        
        Args:
            threshold: Weight threshold (standard) or query id (combinations)
            
        Returns:
            DataFrame with edges aligned across datasets
        """
        if self.parameters.threshold_mode == 'combinations':
            return self.get_aligned_data_for_query(threshold)
        if threshold in self.aligned_results:
            return self.aligned_results[threshold]
        
        dataset_names = self.parameters.get_dataset_names()

        # Get type mapper for auto type mapping (if enabled)
        type_mapper = self.parameters._auto_type_mapper if self.parameters.auto_type_mapping else None
        # One shared-resolver snapshot per run: canonical merge keys for
        # path/edge merging reuse this run's decision cache and load state.
        if type_mapper is not None and self._mapper_snapshot is None:
            from .type_resolver import get_mapper_snapshot
            self._mapper_snapshot = get_mapper_snapshot(type_mapper)
        
        # Pass label_mapper=None because raw_results are already mapped in run_path_analysis/run_edge_analysis
        aligned = self.metrics._align_results_at_threshold(
            self.raw_results,
            dataset_names,
            threshold,
            label_mapper=self._policy_label_mapper(),
            type_mapper=type_mapper,
            merge_policy=self._merge_policy_or_none()
        )

        # Optionally filter edges to only hemisphere-conserved pairs
        # Works for edges with _L/_R/_U suffixes; edges without hemisphere info are kept as-is
        if self.parameters.keep_only_hemisphere_conserved_connections:
            aligned = self._filter_hemisphere_unconserved(aligned, dataset_names, threshold)
        
        self.aligned_results[threshold] = aligned
        return aligned

    def get_aligned_data_for_network(self, threshold: int) -> pd.DataFrame:
        """Get aligned edge data for network visualizations.

        When find_reciprocal=True, this uses reciprocal_connection_type.csv
        outputs (if available) to build the network graph.
        """
        query = (
            self._query_record(threshold)
            if self.parameters.threshold_mode == 'combinations'
            else None
        )
        cache_key = query.get('id') if query else threshold
        if not self.parameters.find_reciprocal:
            return self.get_aligned_data(threshold)

        if cache_key in self._network_aligned_cache:
            return self._network_aligned_cache[cache_key]

        dataset_names = self.parameters.get_dataset_names()
        type_mapper = self.parameters._auto_type_mapper if self.parameters.auto_type_mapping else None
        threshold_map = query.get('thresholds', {}) if query else {}

        # Build temporary raw_results from reciprocal files when available
        reciprocal_results: Dict[str, Dict[int, pd.DataFrame]] = {}
        for dataset in dataset_names:
            reciprocal_results[dataset] = {}
            safe_name = self.parameters._sanitize_name(dataset)
            dataset_threshold = (
                threshold_map.get(dataset)
                if query else threshold
            )
            reciprocal_path = os.path.join(
                self._resolve_dataset_output_path(dataset, dataset_threshold),
                'find_reciprocal',
                'reciprocal_connection_type.csv'
            )
            if os.path.exists(reciprocal_path):
                try:
                    df = self._read_csv(reciprocal_path)
                    reciprocal_results[dataset][dataset_threshold] = df
                except Exception as e:
                    self._log(f"Warning: Failed to read reciprocal network data for {dataset} t={threshold}: {e}", level='warn')

        # If no reciprocal data found, fall back to standard aligned data
        has_any = any(
            (threshold_map.get(ds) if query else threshold)
            in reciprocal_results.get(ds, {})
            for ds in dataset_names
        )
        if not has_any:
            return self.get_aligned_data(threshold)

        aligned = (
            self.metrics._align_results_for_threshold_map(
                reciprocal_results,
                dataset_names,
                threshold_map,
                label_mapper=self._policy_label_mapper(),
                type_mapper=type_mapper,
                merge_policy=self._merge_policy_or_none(),
            )
            if query else
            self.metrics._align_results_at_threshold(
                reciprocal_results,
                dataset_names,
                threshold,
                label_mapper=self._policy_label_mapper(),
                type_mapper=type_mapper,
                merge_policy=self._merge_policy_or_none()
            )
        )

        if self.parameters.keep_only_hemisphere_conserved_connections:
            aligned = self._filter_hemisphere_unconserved(aligned, dataset_names, threshold)

        self._network_aligned_cache[cache_key] = aligned
        return aligned

    def _filter_hemisphere_unconserved(self, aligned: pd.DataFrame, dataset_names: List[str], 
                                        threshold: int = None) -> pd.DataFrame:
        """
        Filter out hemisphere-unconserved edges and save them to a separate file.
        
        An edge is considered "conserved" if both it and its mirror counterpart
        (L->L paired with R->R, or L->R paired with R->L) are present.
        
        Edges without hemisphere suffixes (_L/_R/_U) in their labels are kept as-is
        since they cannot be evaluated for hemisphere conservation.
        
        Args:
            aligned: Aligned edge data
            dataset_names: List of dataset names
            threshold: Optional threshold value for file naming
            
        Returns:
            Filtered DataFrame with only conserved edges
        """
        if aligned is None or aligned.empty:
            return aligned

        def extract_hemi(label: str):
            base = label.split('(')[0].strip() if '(' in label else label
            hemi = None
            if base.endswith(('_L', '_R', '_U')):
                hemi = base[-1]
                base = base[:-2]
            return base, hemi

        def opposite(hemi: str) -> str:
            return 'R' if hemi == 'L' else 'L'

        aligned = aligned.copy()
        aligned.index = aligned.index.astype(str)
        index_set = set(aligned.index)
        
        # Track unconserved edges for saving
        unconserved_edges = []
        unconserved_reasons = []

        for edge_key in aligned.index:
            if ' -> ' not in edge_key:
                continue
            pre, post = edge_key.split(' -> ', 1)
            base_pre, hemi_pre = extract_hemi(pre)
            base_post, hemi_post = extract_hemi(post)

            if hemi_pre not in ('L', 'R') or hemi_post not in ('L', 'R'):
                # Edge doesn't have proper L/R hemisphere info - keep it as-is
                # (Cannot evaluate hemisphere conservation without hemisphere suffixes)
                continue

            # Mirror counterpart: flip the hemisphere of both endpoints.
            # (opposite() handles same-side and cross-hemisphere edges alike.)
            counterpart = f"{base_pre}_{opposite(hemi_pre)} -> {base_post}_{opposite(hemi_post)}"

            if counterpart not in index_set:
                original_weights = {ds: aligned.at[edge_key, ds] for ds in dataset_names if ds in aligned.columns and aligned.at[edge_key, ds] > 0}
                if original_weights:
                    unconserved_edges.append(edge_key)
                    unconserved_reasons.append(f"Missing counterpart: {counterpart}")
                aligned.loc[edge_key, dataset_names] = 0
                continue

            for ds in dataset_names:
                if ds not in aligned.columns:
                    continue
                w = aligned.at[edge_key, ds]
                w2 = aligned.at[counterpart, ds]
                if not (w > 0 and w2 > 0):
                    if w > 0:
                        unconserved_edges.append(edge_key)
                        unconserved_reasons.append(f"Counterpart {counterpart} has weight=0 in {ds}")
                    aligned.at[edge_key, ds] = 0
        
        # Save unconserved edges to file
        if unconserved_edges and hasattr(self, 'parameters') and self.parameters.full_output_path:
            try:
                results_dir = os.path.join(self.parameters.full_output_path, 'comparison_results')
                os.makedirs(results_dir, exist_ok=True)
                
                threshold_suffix = f"_t{threshold}" if threshold else ""
                unconserved_file = os.path.join(results_dir, f"hemisphere_unconserved_edges{threshold_suffix}.csv")
                
                unconserved_df = pd.DataFrame({
                    'edge': unconserved_edges,
                    'reason': unconserved_reasons
                })
                unconserved_df.to_csv(unconserved_file, index=False)
                self._log(f"Saved {len(unconserved_edges)} unconserved edges to hemisphere_unconserved_edges{threshold_suffix}.csv")
            except Exception as e:
                self._log(f"Warning: Could not save unconserved edges: {e}")

        return aligned
    
    def get_common_connections(self, threshold: int) -> pd.DataFrame:
        """
        Get connections present in all datasets at threshold.
        
        Args:
            threshold: Weight threshold
            
        Returns:
            DataFrame with common connections
        """
        aligned = self.get_aligned_data(threshold)
        dataset_names = self.parameters.get_dataset_names()
        return self.metrics.find_common_connections(aligned, dataset_names, threshold)
    
    def get_unique_connections(self, threshold: int) -> Dict[str, pd.DataFrame]:
        """
        Get connections unique to each dataset.
        
        Args:
            threshold: Weight threshold
            
        Returns:
            Dict mapping dataset name to unique connections DataFrame
        """
        aligned = self.get_aligned_data(threshold)
        dataset_names = self.parameters.get_dataset_names()
        return self.metrics.find_unique_connections(aligned, dataset_names, threshold)
    
    def get_differential_connections(self, threshold: int, fold_threshold: float = 2.0) -> pd.DataFrame:
        """
        Get connections with large weight differences.
        
        Args:
            threshold: Weight threshold for presence
            fold_threshold: Minimum fold change
            
        Returns:
            DataFrame with differential connections
        """
        aligned = self.get_aligned_data(threshold)
        dataset_names = self.parameters.get_dataset_names()
        return self.metrics.find_differential_connections(aligned, dataset_names, fold_threshold)
    
    # =========================================================================
    # Report Generation
    # =========================================================================

    def _generate_combination_report(self, output_path: Optional[str] = None,
                                     _skip_log: bool = False) -> str:
        """Generate the text report for explicit row-wise threshold queries.

        Combination mode deliberately does not reuse the scalar-threshold
        report below: ``self.parameters.thresholds`` is only the raw-run
        union in this mode, so iterating it would silently report a comparison
        that was never requested.  Every metric row here is therefore keyed
        by the stable query id and uses that query's per-dataset threshold
        map.
        """
        if not _skip_log:
            self._log("Generating combination-query comparison report text...")

        dataset_names = self.parameters.get_dataset_names()
        queries = self.get_threshold_queries()
        lines = [
            "=" * 70,
            "CROSS-DATASET COMPARISON REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
        ]
        # Hemisphere-aware notice (plan round-4 F-3) — combination text
        # report variant.
        if getattr(self.parameters, 'separate_hemispheres', False):
            lines.append("🧠 HEMISPHERE-AWARE RUN: neuron type names carry "
                         "_L/_R/_U hemisphere suffixes.")
            lines.append("")
        lines.extend([
            "DATASETS:",
            "-" * 40,
        ])
        lines.extend(f"  • {dataset}" for dataset in dataset_names)
        lines.extend([
            "",
            "ANALYSIS PARAMETERS:",
            "-" * 40,
            f"  Source neurons: {self.parameters.source_neurons}",
            f"  Target neurons: {self.parameters.target_neurons}",
            f"  Max interlayer: {self.parameters.max_interlayer}",
            f"  Comparison mode: {self.parameters.comparison_mode}",
            "  Threshold mode: combinations (one complete row per query)",
            "",
            "THRESHOLD QUERIES:",
            "-" * 70,
            "  query_id | label | requested thresholds",
        ])
        for query in queries:
            threshold_text = ", ".join(
                f"{dataset}={query['thresholds'].get(dataset)}"
                for dataset in dataset_names
            )
            lines.append(
                f"  {query['id']} | {query.get('label', query['id'])} | "
                f"{threshold_text}")
        lines.extend([
            "",
            "PATHFINDING THRESHOLD/BOTTLENECK PROVENANCE:",
            "-" * 70,
            "  query_id | dataset | requested | applied | source | SF budget | "
            "SF bite | tau | Edge budget | floor applied | w0 | w1 | w2 | "
            "W* | paths_complete",
        ])
        for query in queries:
            query_id = query['id']
            for dataset in dataset_names:
                requested = int(query['thresholds'][dataset])
                row = self._path_provenance_row(dataset, requested)
                lines.append(
                    f"  {query_id} | {dataset} | {row['requested_threshold']} | "
                    f"{row['applied_threshold']} | "
                    f"{row['applied_threshold_source']} | "
                    f"{row['strongest_first_budget']} | "
                    f"{row['strongest_first_budget_bitten']} | "
                    f"{row['tau']} | {row['edge_budget']} | "
                    f"{row['edge_budget_applied']} | "
                    f"{row['edge_weight_floor']} | "
                    f"{row['edge_budget_landing']} | "
                    f"{row['strongest_dropped_bottleneck']} | "
                    f"{row['strongest_retained_bottleneck']} | "
                    f"{row['paths_complete']}")
        lines.extend([
            "  Definitions: SF budget = effective Max Paths budget; tau = "
            "StrongestFirst landing; Edge budget = graph edge cap; floor "
            "applied = whether the graph cap fired; w0 = Edge Budget floor; "
            "w1 = landing tier; w2 = strongest dropped "
            "bottleneck; W* = strongest retained bottleneck. The complete "
            "machine-readable block is comparison_results/"
            "threshold_combinations.csv.",
            "",
            "KEY METRICS BY THRESHOLD QUERY:",
            "-" * 70,
            f"{'Query':>12} | {'Label':<20} | {'Total Edges':>12} | "
            f"{'Conserved':>10} | {'Edge Rate':>10} | "
            f"{'Total Paths':>12} | {'Path Rate':>10}",
            "-" * 70,
        ])
        for query in queries:
            aligned = self.get_aligned_data_for_query(query)
            if aligned.empty:
                continue
            available = [dataset for dataset in dataset_names
                          if dataset in aligned.columns]
            total_edges = len(aligned)
            common_edges = int((aligned[available] > 0).all(axis=1).sum()) \
                if available else 0
            edge_rate = (common_edges / total_edges * 100) \
                if total_edges else 0
            path_count = common_paths = path_rate = 0
            try:
                path_data = self._get_path_data_for_query(query)
                if not path_data.empty:
                    path_count = len(path_data)
                    path_available = [dataset for dataset in dataset_names
                                      if dataset in path_data.columns]
                    common_paths = int(
                        (path_data[path_available] > 0).all(axis=1).sum()
                    ) if path_available else 0
                    path_rate = (common_paths / path_count * 100) \
                        if path_count else 0
            except Exception:
                pass
            lines.append(
                f"{query['id']:>12} | {str(query.get('label', ''))[:20]:<20} | "
                f"{total_edges:>12} | {common_edges:>10} | "
                f"{edge_rate:>9.1f}% | {path_count:>12} | "
                f"{path_rate:>9.1f}%")
        lines.append("")

        lines.extend([
            "PAIRWISE SIMILARITIES BY THRESHOLD QUERY:",
            "-" * 70,
        ])
        similarities = self.comparison_report.get('threshold_similarities',
                                                    pd.DataFrame()) \
            if self.comparison_report else pd.DataFrame()
        if isinstance(similarities, pd.DataFrame) and not similarities.empty:
            for query_id, query_sims in similarities.groupby('query_id') \
                    if 'query_id' in similarities.columns else []:
                label = next((q.get('label', query_id) for q in queries
                              if q['id'] == query_id), query_id)
                lines.append(f"\n  {query_id} ({label}):")
                for _, row in query_sims.iterrows():
                    def _num(key):
                        value = row.get(key, 0)
                        if pd.isna(value):
                            value = 0
                        return value
                    lines.append(
                        f"    {row.get('dataset_1')} vs "
                        f"{row.get('dataset_2')}: Jaccard "
                        f"{_num('jaccard_similarity'):.3f} | "
                        f"Cosine {_num('cosine_similarity'):.3f} | "
                        f"Top-20 {_num('top20_overlap'):.3f} | "
                        f"NetSimile {_num('netsimile_similarity'):.3f} | "
                        f"Common {_num('common_edges')}")

        else:
            lines.append("  (No pairwise similarity rows found)")

        lines.extend([
            "",
            "DATASET OVERLAP SUMMARY:",
            "-" * 70,
        ])
        for query in queries:
            aligned = self.get_aligned_data_for_query(query)
            if aligned.empty:
                continue
            available = [dataset for dataset in dataset_names
                          if dataset in aligned.columns]
            lines.append(f"\n  Query = {query['id']} ({query.get('label')})")
            for left_index, left in enumerate(available):
                left_edges = set(aligned.index[aligned[left] > 0])
                for right in available[left_index + 1:]:
                    right_edges = set(aligned.index[aligned[right] > 0])
                    overlap = len(left_edges & right_edges)
                    left_pct = overlap / len(left_edges) * 100 \
                        if left_edges else 0
                    right_pct = overlap / len(right_edges) * 100 \
                        if right_edges else 0
                    lines.append(
                        f"    {left} ∩ {right}: {overlap} edges "
                        f"({left_pct:.0f}% of left, {right_pct:.0f}% of right)")

        type_coverage_lines = self._type_coverage_txt_lines()
        if type_coverage_lines:
            lines.extend([""] + type_coverage_lines)

        lines.extend([
            "",
            "QUERY-KEYED EXPORTS:",
            "-" * 70,
            "  Each query has its own comparison matrices and visualization "
            "data. The filenames use a filesystem-safe query ID; the CSV "
            "query_id/query_label fields retain the original identity.",
        ])
        for query in queries:
            query_id = query['id']
            safe_query_id = self._safe_query_filename_id(query_id)
            lines.extend([
                f"  {query_id}:",
                f"    comparison_results/edge_presence_matrix_query_"
                f"{safe_query_id}.csv",
                f"    comparison_results/path_presence_matrix_query_"
                f"{safe_query_id}.csv",
                f"    similarity_matrices/similarity_query_{safe_query_id}.csv",
                f"    comparison_visualizations/edge_heatmap_query_"
                f"{safe_query_id}.png",
                f"    comparison_visualizations/path_heatmap_query_"
                f"{safe_query_id}.png",
            ])
        lines.extend([
            "  Ratio and traversal-probability filtering is disabled for "
            "pathfinding comparisons; no by_ratio/by_probability query "
            "folders are produced.",
        ])

        lines.extend([
            "",
            "=" * 70,
            "For interactive visualizations, see: comparison_report.html",
            "For the exact query-to-run join, see: "
            "comparison_results/threshold_combinations.csv",
            "=" * 70,
        ])
        report_text = "\n".join(lines)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as handle:
                handle.write(report_text)
            self._log(f"Report saved to: {output_path}")
        return report_text

    def generate_report(self, output_path: Optional[str] = None, _skip_log: bool = False) -> str:
        """
        Generate human-readable comparison report.
        
        Args:
            output_path: Optional path to save report
            _skip_log: Internal flag to skip logging (used when called from export_results)
            
        Returns:
            Report text
        """
        if not _skip_log:
            self._log("Generating comparison report text...")
        
        # Ensure comparison has been run
        if self.comparison_report is None:
            self.run_comparison_analysis()

        if self.parameters.threshold_mode == 'combinations':
            return self._generate_combination_report(
                output_path=output_path, _skip_log=_skip_log)
        
        thresholds_str = ', '.join(str(t) for t in self.parameters.thresholds)
        mid_threshold = self.parameters.thresholds[len(self.parameters.thresholds) // 2]
        dataset_names = self.parameters.get_dataset_names()
        
        lines = []
        lines.append("=" * 70)
        lines.append("CROSS-DATASET COMPARISON REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")
        
        # Hemisphere-aware notice (plan round-4 F-3): make the _L/_R/_U
        # type split visible in the text report too.
        if getattr(self.parameters, 'separate_hemispheres', False):
            lines.append("🧠 HEMISPHERE-AWARE RUN: neuron type names carry "
                         "_L/_R/_U hemisphere suffixes.")
            lines.append("")
        
        # Datasets
        lines.append("DATASETS:")
        lines.append("-" * 40)
        for dataset_name in dataset_names:
            lines.append(f"  • {dataset_name}")
        lines.append("")
        
        # Analysis parameters with explicit threshold info
        lines.append("ANALYSIS PARAMETERS:")
        lines.append("-" * 40)
        lines.append(f"  Source neurons: {self.parameters.source_neurons}")
        lines.append(f"  Target neurons: {self.parameters.target_neurons}")
        lines.append(f"  Max interlayer: {self.parameters.max_interlayer}")
        lines.append(f"  Comparison mode: {self.parameters.comparison_mode}")
        lines.append("")
        
        # Explicit threshold section
        lines.append("SYNAPSE CUTOFF THRESHOLDS (min_synapse_num):")
        lines.append("-" * 40)
        lines.append(f"  Thresholds analyzed: {thresholds_str}")
        lines.append("  NOTE: Results are highly sensitive to threshold choice.")
        lines.append("")

        # The report is also a durable, human-readable explanation of the
        # actual threshold state.  ``tau`` is the StrongestFirst landing
        # value; ``applied_threshold`` is the canonical equivalent threshold
        # and must not be substituted with tau when a bottleneck gap exists.
        lines.append("PATHFINDING THRESHOLD/BOTTLENECK PROVENANCE:")
        lines.append("-" * 70)
        lines.append(
            "  dataset | requested | applied | source | SF budget | SF bite | "
            "tau | Edge budget | floor applied | w0 | w1 | w2 | W* | "
            "paths_complete")
        for dataset_name in dataset_names:
            for threshold in self.parameters.get_thresholds_for_dataset(
                    dataset_name):
                row = self._path_provenance_row(dataset_name, threshold)
                lines.append(
                    f"  {dataset_name} | {row['requested_threshold']} | "
                    f"{row['applied_threshold']} | "
                    f"{row['applied_threshold_source']} | "
                    f"{row['strongest_first_budget']} | "
                    f"{row['strongest_first_budget_bitten']} | "
                    f"{row['tau']} | {row['edge_budget']} | "
                    f"{row['edge_budget_applied']} | "
                    f"{row['edge_weight_floor']} | "
                    f"{row['edge_budget_landing']} | "
                    f"{row['strongest_dropped_bottleneck']} | "
                    f"{row['strongest_retained_bottleneck']} | "
                    f"{row['paths_complete']}")
        lines.append(
            "  Definitions: SF budget = effective Max Paths budget; tau = "
            "StrongestFirst landing; Edge budget = graph edge cap; w0 = Edge Budget "
            "floor; w1 = Edge Budget landing tier; w2 = strongest dropped "
            "bottleneck; W* = strongest retained bottleneck. The complete "
            "machine-readable block is comparison_results/"
            "pathfinding_provenance.csv.")
        lines.append("")
        
        # Key metrics by threshold (matches HTML summary section)
        lines.append("KEY METRICS BY THRESHOLD:")
        lines.append("-" * 70)
        lines.append(f"{'Threshold':>10} | {'Total Edges':>12} | {'Conserved':>10} | {'Edge Rate':>10} | {'Total Paths':>12} | {'Path Rate':>10}")
        lines.append("-" * 70)
        
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            
            available_ds = [d for d in dataset_names if d in aligned.columns]
            total_edges = len(aligned)
            
            # Conserved edges (present in ALL datasets)
            if available_ds:
                mask_all = (aligned[available_ds] > 0).all(axis=1)
                common_edges = int(mask_all.sum())
            else:
                common_edges = 0
            
            edge_rate = (common_edges / total_edges * 100) if total_edges > 0 else 0
            
            # Path data
            try:
                path_data = self._get_path_data_for_threshold(threshold)
                if not path_data.empty:
                    total_paths = len(path_data)
                    path_mask = (path_data[available_ds] > 0).all(axis=1) if available_ds else pd.Series([False])
                    common_paths = int(path_mask.sum())
                    path_rate = (common_paths / total_paths * 100) if total_paths > 0 else 0
                else:
                    total_paths, common_paths, path_rate = 0, 0, 0
            except Exception:
                total_paths, common_paths, path_rate = 0, 0, 0
            
            lines.append(f"{threshold:>10} | {total_edges:>12} | {common_edges:>10} | {edge_rate:>9.1f}% | {total_paths:>12} | {path_rate:>9.1f}%")
        
        lines.append("")

        # Hemisphere symmetry summary (per dataset & threshold)
        symmetry_summaries = self.get_hemisphere_symmetry_summaries()
        if symmetry_summaries:
            lines.append("HEMISPHERE SYMMETRY SUMMARY:")
            lines.append("-" * 70)
            for threshold in self.parameters.thresholds:
                summaries = symmetry_summaries.get(threshold, {})
                if not summaries:
                    continue
                lines.append(f"Threshold t={threshold}:")
                lines.append("  Dataset | Ipsi Jaccard | Contra Jaccard | Ipsi Conserved/Union | Contra Conserved/Union | Types Conserved/Union | Counts L/R")
                lines.append("  " + "-" * 64)
                for dataset in dataset_names:
                    summary = summaries.get(dataset)
                    if not summary:
                        continue
                    ipsi = summary.get('ipsi', {})
                    contra = summary.get('contra', {})
                    types = summary.get('neuron_types', {})
                    counts = summary.get('hemisphere_counts', {}).get('total', {})
                    ipsi_j = ipsi.get('jaccard', 0)
                    contra_j = contra.get('jaccard', 0)
                    ipsi_cons = f"{ipsi.get('conserved', 0)}/{ipsi.get('union', 0)}"
                    contra_cons = f"{contra.get('conserved', 0)}/{contra.get('union', 0)}"
                    types_cons = f"{types.get('types_conserved', 0)}/{types.get('types_union', 0)}"
                    lr_counts = f"{counts.get('L', 0)}/{counts.get('R', 0)}"
                    lines.append(
                        f"  {dataset:<7} | {ipsi_j:>11.3f} | {contra_j:>13.3f} | {ipsi_cons:>19} | {contra_cons:>20} | {types_cons:>20} | {lr_counts:>9}"
                    )
                lines.append("")
        else:
            lines.append("HEMISPHERE SYMMETRY SUMMARY:")
            lines.append("-" * 70)
            lines.append("  (No hemisphere symmetry summaries found)")
            lines.append("")
        
        # Edge counts per dataset per threshold
        lines.append("EDGE COUNTS PER DATASET:")
        lines.append("-" * 70)
        header = f"{'Threshold':>10}"
        for d in dataset_names:
            short_name = d.split(':')[0][:10] if ':' in d else d[:10]
            header += f" | {short_name:>12}"
        lines.append(header)
        lines.append("-" * 70)
        
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            row = f"{threshold:>10}"
            for d in dataset_names:
                count = int((aligned[d] > 0).sum()) if d in aligned.columns else 0
                row += f" | {count:>12}"
            lines.append(row)
        
        lines.append("")
        
        # Pairwise similarities at ALL thresholds
        lines.append("PAIRWISE SIMILARITIES AT ALL THRESHOLD LEVELS:")
        lines.append("-" * 70)
        
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            
            sim_t = self.metrics.calculate_all_pairwise_similarities(aligned, dataset_names, threshold=1)
            if sim_t.empty:
                continue
            
            lines.append(f"\n  Threshold = {threshold}:")
            for _, row in sim_t.iterrows():
                jaccard = row.get('jaccard_similarity', 0)
                svd = row.get('svd_similarity', 0)
                if not self.raw_results: 
                    svd = 0
                pearson = row.get('pearson_correlation', 0)
                if pd.isna(pearson):
                    pearson = 0
                common = row.get('common_edges', 0)
                unique_d1 = row.get('unique_to_d1', 0)
                unique_d2 = row.get('unique_to_d2', 0)
                lines.append(f"    {row['dataset_1']} vs {row['dataset_2']}:")
                lines.append(f"      Jaccard: {jaccard:.3f} | SVD: {svd:.3f} | Weight Corr: {pearson:.3f}")
                lines.append(f"      Common: {common} | Unique to D1: {unique_d1} | Unique to D2: {unique_d2}")
        
        lines.append("")
        
        # Dataset overlap summary
        lines.append("DATASET OVERLAP SUMMARY:")
        lines.append("-" * 70)
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            available_ds = [d for d in dataset_names if d in aligned.columns]
            n = len(available_ds)
            
            lines.append(f"\n  Threshold = {threshold}:")
            # Edge overlap
            for i, d1 in enumerate(available_ds):
                edges_in_d1 = set(aligned.index[aligned[d1] > 0])
                for j, d2 in enumerate(available_ds):
                    if i < j:
                        edges_in_d2 = set(aligned.index[aligned[d2] > 0])
                        overlap = len(edges_in_d1 & edges_in_d2)
                        pct1 = (overlap / len(edges_in_d1) * 100) if edges_in_d1 else 0
                        pct2 = (overlap / len(edges_in_d2) * 100) if edges_in_d2 else 0
                        short1 = d1.split(':')[0][:8] if ':' in d1 else d1[:8]
                        short2 = d2.split(':')[0][:8] if ':' in d2 else d2[:8]
                        lines.append(f"    {short1} ∩ {short2}: {overlap} edges ({pct1:.0f}% of D1, {pct2:.0f}% of D2)")

        type_coverage_lines = self._type_coverage_txt_lines()
        if type_coverage_lines:
            lines.append("")
            lines.extend(type_coverage_lines)

        lines.append("")
        lines.append("=" * 70)
        lines.append("For interactive visualizations, see: comparison_report.html")
        lines.append("For connectivity profile verification, see: connectivity_profile_comparison.html")
        lines.append("=" * 70)
        
        report_text = "\n".join(lines)
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            self._log(f"Report saved to: {output_path}")
        
        return report_text
    
    def export_results(self, output_dir: Optional[str] = None):
        """
        Export all results to files following TODO_comparison.md structure.
        
        Output structure:
            {output_folder}/{saveas}/
            ├── parameters.json
            ├── comparison_report.txt
            ├── comparison_report.html          # Interactive HTML report
            ├── comparison_visualizations/       # Visualizations (renamed from visualizations/)
            │   ├── path_counts.png
            │   ├── edge_heatmap.png
            │   ├── similarity_matrix.png
            │   ├── similarity_per_threshold.png
            │   ├── edge_overlap.png
            │   ├── threshold_comparison.png
            │   ├── heatmaps/                   # Interactive HTML heatmaps
            │   │   ├── edge_heatmap.html
            │   │   └── edge_heatmap_all_thresholds.html
            │   └── visualization_data/         # Data files for visualizations
            │       ├── path_counts.csv
            │       ├── edge_heatmap.csv
            │       ├── similarity_matrix.csv
            │       └── ...
            ├── dataset_data/               # Raw FindNeuronConnection outputs (created by FNC)
            │   ├── {dataset_1}/minsyn_{threshold}/
            │   └── {dataset_2}/minsyn_{threshold}/
            └── comparison_results/         # Cross-dataset comparison outputs
                ├── path_count_comparison.csv
                ├── edge_weight_comparison.csv
                ├── edge_presence_matrix_minsyn_{threshold}.csv
                ├── path_presence_matrix_minsyn_{threshold}.csv
                └── ...
        
        Args:
            output_dir: Directory to save results (defaults to parameters.full_output_path)
        """
        out_dir = output_dir or self.parameters.full_output_path
        if not out_dir:
            raise ValueError("No output directory specified")
        
        self._log("Exporting results...")
        self._progress(4, 5, "Exporting reports and result tables")
        self._log("  Step 1: Saving parameters and report...")
        
        # Ensure data loader is initialized
        if self.data_loader is None:
            self.data_loader = DataLoader(out_dir)
        self.data_loader.ensure_directories()
        
        # Create comparison_results subfolder
        comparison_results_dir = os.path.join(out_dir, "comparison_results")
        os.makedirs(comparison_results_dir, exist_ok=True)
        
        # Save parameters
        params_path = os.path.join(out_dir, "parameters.json")
        parameters_payload = self.parameters.to_dict()
        parameters_payload['pathfinding_provenance'] = {
            'file': 'comparison_results/pathfinding_provenance.csv',
            'notice': 'effective_thresholds.json',
            'fields': list(self._PATH_PROVENANCE_KEYS),
            'tau_definition': 'StrongestFirst landing / natural bottleneck',
            'applied_threshold_definition': (
                'canonical equivalent Min Synapse Count for the materialized '
                'output; requested threshold when no lossy budget applies'),
        }
        parameters_payload['threshold_query_manifest'] = {
            'file': 'comparison_results/threshold_combinations.csv',
            'threshold_mode': getattr(
                self.parameters, 'threshold_mode', 'standard'),
            'dataset_order': self.parameters.get_dataset_names(),
            'query_count': len(self.get_threshold_queries()),
            'description': (
                'One row per threshold query and dataset. The requested '
                'threshold is joined to the run-level applied threshold and '
                'pathfinding bottleneck provenance.'
            ),
        }
        # Keep the complete query definition and its per-dataset applied
        # provenance in the portable parameters file as well as in the CSV
        # join.  ``threshold_combinations`` remains the user input; this
        # second block is the executed/query-level projection used by report
        # consumers that do not load comparison_results/ first.
        if getattr(self.parameters, 'threshold_mode', 'standard') == 'combinations':
            manifest_rows = self._threshold_query_manifest_rows()
            query_payload = []
            for query in self.get_threshold_queries():
                query_id = query.get('id') or query.get('query_id')
                query_rows = [
                    row for row in manifest_rows
                    if row.get('query_id') == query_id
                ]
                query_payload.append({
                    'id': query_id,
                    'label': query.get('label', query_id),
                    'requested_thresholds': dict(
                        query.get('thresholds') or {}),
                    'applied_thresholds': {
                        row['dataset']: row.get('applied_threshold')
                        for row in query_rows
                    },
                    'provenance': query_rows,
                })
            parameters_payload['threshold_queries'] = query_payload
        else:
            # Standard runs use scalar points, but still expose the same
            # normalized key so downstream readers need no mode-specific
            # schema discovery.
            parameters_payload['threshold_queries'] = [
                {
                    'id': f'threshold_{threshold}',
                    'label': f'N={threshold}',
                    'requested_thresholds': {
                        dataset: threshold
                        for dataset in self.parameters.get_dataset_names()
                    },
                    'applied_thresholds': {
                        dataset: self._path_provenance_row(
                            dataset, threshold).get('applied_threshold')
                        for dataset in self.parameters.get_dataset_names()
                    },
                }
                for threshold in self.parameters.thresholds
            ]
        with open(params_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(parameters_payload, f, indent=2, default=str)
            
        # Save label mapping (always generate a compatible JSON)
        label_map_path = os.path.join(out_dir, "label_map.json")
        self._export_label_map(label_map_path)
        
        # Export auto type mapping if enabled (filtered to result types and used datasets only)
        if self.parameters.auto_type_mapping and self.parameters._auto_type_mapper:
            try:
                # Collect types from results for filtered export
                # (schedule-bound: deterministic fresh-run vs re-export)
                result_types = self._compared_result_types()
                dataset_names = self.parameters.get_dataset_names()
                
                auto_map_path = os.path.join(out_dir, "auto_type_mapping.csv")
                self.parameters._auto_type_mapper.export_mapping(
                    auto_map_path, 
                    filter_types=result_types if result_types else None,
                    datasets=dataset_names,
                    only_different=True  # Only export mappings where types differ across datasets
                )
                self._log_file(auto_map_path, "Auto type mapping")

                # Per-bridge type-level mapping table (plan
                # plan-type-mapper-fine-granularity-export.md §4):
                # additive companion to the compact matrix above — one row
                # per (type pair, bridge chain) with per-bridge
                # annotations and bridge-refined {…} bodyId pools.
                try:
                    per_bridge_path = os.path.join(
                        out_dir, "auto_type_mapping_per_bridge.csv")
                    self.parameters._auto_type_mapper.export_mapping_per_bridge(
                        per_bridge_path,
                        source_types=result_types if result_types else None,
                        target_datasets=dataset_names or None,
                    )
                    self._log_file(per_bridge_path,
                                   "Auto type mapping (per-bridge)")
                except Exception as exc:
                    self._log(f"Warning: could not write per-bridge type "
                              f"mapping export: {exc}")

                # Standard auto-type-mapping metadata block (plan
                # unify-mapper-backends Workstream C): requested vs active
                # mapper, source table, version, load error, and this run's
                # per-status canonical-merge resolution counts.
                try:
                    from .profile_comparator import auto_mapping_result_metadata
                    mapping_meta = auto_mapping_result_metadata(
                        requested=self.parameters.auto_type_mapping,
                        type_mapper=self.parameters._auto_type_mapper,
                        snapshot=self._mapper_snapshot,
                        resolution_counts=self._mapping_status_counts,
                    )
                    if self._conflicted_merge_types:
                        mapping_meta['conflicted_merge_types_dataset_scoped'] = (
                            dict(sorted(self._conflicted_merge_types.items())))
                    mapping_path = os.path.join(out_dir, "auto_type_mapping.json")
                    with open(mapping_path, 'w', encoding='utf-8') as mf:
                        import json as _json
                        _json.dump(mapping_meta, mf, indent=2, default=str)
                    self._log_file(mapping_path, "Auto type mapping metadata")
                except Exception as exc:
                    self._log(f"Warning: could not write auto type mapping metadata: {exc}")

                # Query-anchored merge policy (plan:
                # plan-query-anchored-cross-dataset-analysis.md): export
                # the resolution topology, annotate the mapping CSV with
                # additive anchor_group/auto_only columns, and route the
                # policy + BANC auto-label warnings into the run notes.
                policy = self._merge_policy_or_none()
                if policy is not None:
                    try:
                        topology_path = os.path.join(
                            out_dir, "type_resolution_topology.json")
                        with open(topology_path, 'w', encoding='utf-8') as tf:
                            import json as _json
                            _json.dump(policy.topology_dict(), tf,
                                       indent=2, default=str)
                        self._log_file(topology_path,
                                       "Type resolution topology")
                    except Exception as exc:
                        self._log(f"Warning: could not write type "
                                  f"resolution topology: {exc}")
                    try:
                        self._annotate_auto_type_mapping_csv(
                            os.path.join(out_dir, "auto_type_mapping.csv"),
                            policy)
                    except Exception as exc:
                        self._log(f"Warning: could not annotate "
                                  f"auto_type_mapping.csv: {exc}")
                    try:
                        self._write_merge_policy_warnings(
                            policy, result_types or [], dataset_names)
                    except Exception as exc:
                        self._log(f"Warning: could not write merge-policy "
                                  f"warnings: {exc}")

                # Also export conflicts if any (filtered to result types)
                if self.parameters._auto_type_mapper.has_conflicts():
                    conflicts_path = os.path.join(out_dir, "auto_type_mapping_conflicts.csv")
                    self.parameters._auto_type_mapper.export_conflicts(
                        conflicts_path,
                        filter_types=result_types if result_types else None,
                        datasets=dataset_names,
                    )
                    self._log_file(conflicts_path, "Type mapping conflicts")

                # Same-name-first SUSPECT relations (plan-samename-first-
                # fanout-resolution): the rival candidates that were NOT
                # selected, with per-rival adjudication evidence.
                try:
                    suspects_path = os.path.join(
                        out_dir, "auto_type_mapping_suspects.csv")
                    n_suspects = self.parameters._auto_type_mapper.export_suspects(
                        suspects_path,
                        filter_types=result_types if result_types else None,
                        datasets=dataset_names,
                    )
                    if n_suspects:
                        self._log_file(suspects_path,
                                       "Type mapping suspects")
                except Exception as exc:
                    self._log(f"Warning: could not write suspects export: {exc}")

                # Bridge visualizations for the run's mapped pairs
                try:
                    mapper = self.parameters._auto_type_mapper
                    flows = []
                    seen_pairs = set()
                    for result_type in (result_types or [])[:300]:
                        source_ds = (
                            mapper.detect_type_source(result_type)
                            or (dataset_names[0] if dataset_names else "")
                        )
                        if not source_ds:
                            continue
                        for dataset_name in dataset_names:
                            pair = (result_type, dataset_name)
                            if pair in seen_pairs:
                                continue
                            bridges = mapper.get_type_bridges(
                                result_type, source_ds, dataset_name
                            )
                            if not bridges:
                                continue
                            seen_pairs.add(pair)
                            final = bridges[0][-1]["value"]
                            count = mapper.get_type_neuron_count(
                                final, dataset_name) or 1
                            flows.append({
                                "source_dataset": source_ds,
                                "target_dataset": dataset_name,
                                "source_type": result_type,
                                "source_count": count,
                                "foreign_type": final,
                                "foreign_count": count,
                                "matched_origin": "auto type mapping",
                                "bridges": bridges,
                            })
                    if flows:
                        from comparison.mapping_visualization import (
                            render_mapping_sankey_html,
                            write_mapping_network_html,
                        )
                        sankey_path = os.path.join(out_dir, "mapping_sankey.html")
                        sankey_html = render_mapping_sankey_html(flows)
                        if sankey_html:
                            with open(sankey_path, "w",
                                      encoding="utf-8") as handle:
                                handle.write(sankey_html)
                            self._log_file(sankey_path,
                                           "Mapping bridge Sankey")
                        network_path = os.path.join(out_dir, "mapping_network.html")
                        write_mapping_network_html(flows, network_path, open_browser=False)
                        self._log_file(network_path, "Mapping bridge network")
                except Exception as map_exc:
                    self._log(f"Mapping visualization skipped: {map_exc}")
            except Exception as e:
                self._log(f"Warning: Failed to export auto type mapping: {e}", level='warn')
        
        # Save report (skip logging since we already logged "Saving parameters and report")
        report_path = os.path.join(out_dir, "comparison_report.txt")
        self.generate_report(report_path, _skip_log=True)
        
        self._log("  Step 2: Exporting cross-dataset comparisons...")

        # Query identity/provenance is exported before aggregate tables so it
        # is available as the canonical join for every comparison output.
        self._export_threshold_query_manifest(comparison_results_dir)
        
        # === Cross-dataset comparison results ===
        self._export_cross_dataset_comparisons(comparison_results_dir)

        # === Union type-resolution coverage (one row per query × type ×
        # dataset: resolved name + absence diagnosis) ===
        try:
            self._export_type_resolution_union(comparison_results_dir)
        except Exception as e:
            self._log(f"Warning: type resolution union export failed: {e}")

        # === Intra-dataset threshold sensitivity ===
        self._export_intra_dataset_comparisons(comparison_results_dir)

        # === Threshold alignment (Features C/D data: prober best matches,
        # typed matrix, extended-grid density) ===
        try:
            self._export_threshold_alignment(comparison_results_dir)
        except Exception as e:
            self._log(f"Warning: threshold alignment export failed: {e}")

        # === Density curves for every queried dataset (all pathfinding
        # modes; cone-scoped, debris-free). Aligned rows are auto-only and
        # gated inside. ===
        try:
            self._export_density_alignment(comparison_results_dir)
        except Exception as e:
            self._log(f"Warning: density alignment export failed: {e}")

        # === Reciprocal connectivity as an independent comparison artifact ===
        try:
            self._export_reciprocal_comparisons(comparison_results_dir)
        except Exception as e:
            self._log(f"Warning: reciprocal comparison export failed: {e}")

        self._log("  Step 3: Generating visualizations...")
        self._progress(5, 5, "Generating comparison visualizations and HTML report")
        
        # Generate matplotlib visualizations to comparison_visualizations/ at base level
        try:
            self._generate_visualizations(out_dir)
        except Exception as e:
            self._log(f"Warning: Failed to generate visualizations: {e}")
        
        # NOTE: Connectivity profile verification is NOT run here automatically.
        # Call run_connectivity_profile_verification() separately after export_results()
        # if needed. Parameters can be set in ComparisonParameters:
        #   - verification_direction, verification_mode, verification_top_k, etc.
        # Example:
        #   analyzer.run_connectivity_profile_verification()  # Uses params from ComparisonParameters
        
        self._log("  Step 4: Generating HTML report...")
        
        # Generate interactive HTML report at base level
        try:
            html_report_path = os.path.join(out_dir, "comparison_report.html")
            self.generate_html_report(html_report_path)
        except Exception as e:
            self._log(f"Warning: Failed to generate HTML report: {e}")

        # The report computes the type appearance order; persist it into the
        # mapping exports so CSV/report/UI agree (plan §7B.4).
        try:
            self._augment_type_mapping_with_appearance(out_dir)
        except Exception as e:
            self._log(f"Warning: could not persist type appearance order: {e}")
        # Structured input-query resolution + low-confidence warnings (§7E).
        try:
            self._export_query_resolution_records(out_dir)
        except Exception as e:
            self._log(f"Warning: could not persist query resolution: {e}")
        # Self-describing run manifest (plan §12 item 13).
        try:
            self._write_run_manifest(out_dir)
        except Exception as e:
            self._log(f"Warning: could not write run_manifest.json: {e}")

        self._log(f"All results exported to: {out_dir}")
        self._log(f"Note: Run connectivity_profile_comparison() separately for profile verification.")

    def dataset_coverage(self) -> Dict[str, Dict[str, Any]]:
        """Per configured dataset: did it produce any data? (plan B, 2026-09-15)

        A configured dataset whose runs all came back empty (e.g. a
        silent fetch failure) must be visible instead of quietly dropping
        out of every analysis: ``status`` is ``ok`` when at least one
        threshold returned rows, ``no_data`` otherwise. ``failed`` is
        reserved for runners that record an explicit exception.
        """
        coverage: Dict[str, Dict[str, Any]] = {}
        for ds in self.parameters.get_dataset_names():
            thresholds = getattr(self, 'raw_results', {}).get(ds, {}) or {}
            thresholds_with_data = 0
            total_rows = 0
            for _t, df in thresholds.items():
                try:
                    if df is not None and not df.empty:
                        thresholds_with_data += 1
                        total_rows += int(len(df))
                except Exception:
                    continue
            coverage[ds] = {
                'status': 'ok' if thresholds_with_data else 'no_data',
                'thresholds_with_data': thresholds_with_data,
                'total_rows': total_rows,
            }
        return coverage

    def _write_run_manifest(self, out_dir: str) -> None:
        """Emit one self-describing manifest the report/UI/scripts share."""
        try:
            params = self.parameters.to_dict()
        except Exception:
            params = {}
        manifest = {
            'created_at': __import__('datetime').datetime.now().isoformat(),
            'datasets': self.parameters.get_dataset_names(),
            'nicknames': self.parameters.get_dataset_nicknames(),
            'parameters': params,
            'applied_thresholds': {
                ds: self.get_applied_thresholds(ds)
                for ds in self.parameters.get_dataset_names()},
            'threshold_views': self.get_threshold_queries_view(),
            'comparability': self.comparability_report(),
            'dataset_coverage': self.dataset_coverage(),
            'untyped_drop': {
                f'{ds}@{t}': stats for (ds, t), stats in
                (getattr(self, '_untyped_drop_stats', {}) or {}).items()},
            'alignment': {
                'reference': (self.parameters.get_dataset_names() or [None])[0],
                'suggested_combinations': getattr(
                    self, '_suggested_combinations', None) or [],
            },
            # Present only on auto-mode runs: how the density-aligned schedule
            # resolved ('installed' | 'verticals_only' | 'degraded'), which
            # datasets yielded no capture, and why.
            **({'auto_mode_status': self._auto_bootstrap_status}
               if getattr(self, '_auto_bootstrap_status', None) else {}),
            'provenance_file': 'comparison_results/pathfinding_provenance.csv',
            'code_version': getattr(self, '_code_version', None),
        }
        with open(os.path.join(out_dir, 'run_manifest.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, default=str)
        self._log("Saved: run_manifest.json")

    def _augment_type_mapping_with_appearance(self, out_dir: str) -> None:
        """Add ``appearance_rank`` (+ counts) to auto_type_mapping.csv.

        Ranks come from the report's presence-matrix first-appearance order
        (``analyzer._type_appearance_ranks``); also emits a standalone
        ``comparison_report_used_data/type_appearance_order.csv``.
        """
        ranks = getattr(self, '_type_appearance_ranks', None)
        if not ranks:
            return
        map_path = os.path.join(out_dir, 'auto_type_mapping.csv')
        if os.path.exists(map_path):
            df = pd.read_csv(map_path)
            if not df.empty and len(df.columns):
                first_col = df.columns[0]
                df['appearance_rank'] = df[first_col].map(
                    lambda name: ranks.get(str(name)))
                df = df.sort_values(
                    'appearance_rank', na_position='last', kind='stable')
                df.to_csv(map_path, index=False)
        used_dir = os.path.join(out_dir, 'comparison_report_used_data')
        os.makedirs(used_dir, exist_ok=True)
        order_df = pd.DataFrame(
            [{'canonical_type': k, 'appearance_rank': v}
             for k, v in sorted(ranks.items(), key=lambda kv: kv[1])])
        order_df.to_csv(
            os.path.join(used_dir, 'type_appearance_order.csv'), index=False)

    def _export_query_resolution_records(self, out_dir: str) -> None:
        """Persist the structured query resolution and warn on the
        low-confidence classes (plan §7E, rev 2)."""
        try:
            records = self.resolve_query_inputs()
        except Exception as e:
            self._log(f"WARNING: query resolution records not exported: "
                      f"{e!r}")
            return
        if not records:
            return
        used_dir = os.path.join(out_dir, 'comparison_report_used_data')
        os.makedirs(used_dir, exist_ok=True)
        pd.DataFrame(records).to_csv(
            os.path.join(used_dir, 'query_resolution.csv'), index=False)
        same_name = sorted({r['token'] for r in records
                            if r.get('status') == 'same_name_fallback'})
        conflicts = sorted({r['token'] for r in records
                            if r.get('status') == 'conflict'})
        contradicted = sorted({
            (r['token'], r.get('dataset'), str(r.get('evidence_chain') or ''))
            for r in records
            if r.get('status') == 'same_name_identity'
            and r.get('evidence') == 'contradicted'})
        tax_mapped = sorted({r['token'] for r in records
                             if r.get('status') == 'taxonomy_mapped'
                             and 'no counterpart' in (r.get('note') or '')})
        lines = []
        if same_name:
            lines.append(
                "- [query resolution] Low-confidence same-name matches: "
                + ', '.join(same_name)
                + ". These tokens are not native type names in that dataset "
                  "and have no mapped evidence; verify they are genuine "
                  "cross-dataset equivalents (see comparison_report_used_data/"
                  "query_resolution.csv and the report's Query Resolution "
                  "section).")
        if contradicted:
            parts = [f"{tok} ({ds}: {str(chain).replace('curated counterpart: ', '')})"
                     for tok, ds, chain in contradicted]
            lines.append(
                "- [query resolution] Same-name identities with "
                "counter-evidence: " + '; '.join(parts)
                + " — the query keeps the native name, but the cross-dataset "
                  "relation names another counterpart; check whether it "
                  "matters for your interpretation.")
        if tax_mapped:
            lines.append(
                "- [query resolution] Taxonomy member mapping had gaps for: "
                + ', '.join(tax_mapped)
                + " — some member types have no counterpart in a dataset "
                  "(details in query_resolution.csv notes).")
        if conflicts:
            lines.append(
                "- [query resolution] Unresolved mapping conflicts: "
                + ', '.join(conflicts)
                + " — no automatic target; these tokens may match nothing.")
        if lines:
            self._append_user_warning_notes(out_dir, lines)
        self._log(
            "Saved: comparison_report_used_data/query_resolution.csv"
            + (f" ({len(same_name)} same-name fallback token(s))"
               if same_name else ""))
    
    def _export_label_map(self, filepath: str):
        """
        Export label mapping to JSON, creating one from parameters if needed.
        
        Ensures that a valid LabelMapper JSON is always saved, merging data from
        an existing LabelMapper and any list-based parameters.
        
        When auto_type_mapping=True:
        - Saves the auto-mapped types per dataset (e.g., MeVPLo2 → MTe07 for FAFB)
        - Removes types that don't exist in specific datasets from those dataset entries
        """
        output = {}
        dataset_names = self.parameters.get_dataset_names()
        auto_mapper = self.parameters._auto_type_mapper if self.parameters.auto_type_mapping else None
        
        # Helper for smart labels
        def generate_smart_labels(groups, user_labels, suffix=""):
            if user_labels:
                if isinstance(user_labels, list) and len(user_labels) == len(groups):
                    return user_labels
                if isinstance(user_labels, str) and len(groups) == 1:
                    return [user_labels]
            
            # Generate defaults
            labels = []
            for i, g in enumerate(groups):
                if len(g) == 1:
                    labels.append(str(g[0]))
                else:
                    labels.append(f"Group_{i+1}{suffix}")
            return labels
        
        def resolve_types_for_dataset(neurons: list, dataset: str) -> list:
            """Resolve neuron types for a specific dataset using auto mapping."""
            if not auto_mapper:
                return neurons
            
            resolved = []
            for neuron in neurons:
                if isinstance(neuron, list):
                    # Handle grouped neurons
                    resolved_group = resolve_types_for_dataset(neuron, dataset)
                    if resolved_group:  # Only add if not empty
                        resolved.append(resolved_group)
                elif isinstance(neuron, str):
                    # Skip regex patterns - pass through as-is
                    if '*' in neuron or ('.' in neuron and '.*' in neuron):
                        resolved.append(neuron)
                        continue
                    
                    # Try to resolve type using auto mapper (shared validity
                    # resolver: only an unambiguous one-target case renames;
                    # splits/conflicts stay unmapped here)
                    source_ds = auto_mapper.detect_type_source(neuron)
                    if source_ds:
                        from .type_resolver import resolve_one_target
                        mapped = resolve_one_target(
                            auto_mapper, neuron, source_ds, dataset)
                        if mapped:
                            resolved.append(mapped)
                        # If no mapping found, the type doesn't exist in this dataset - skip it
                    else:
                        # Type not found in any dataset, pass through (might be regex or new type)
                        resolved.append(neuron)
                else:
                    # Non-string (bodyId), pass through
                    resolved.append(neuron)
            return resolved

        # --- Source Mapping ---
        # Priority 1: Analyzer's label_mapper (contains merged overall/source/target mappers)
        if self.label_mapper:
            d = self.label_mapper.to_dict()
            if 'source_mapping' in d and d['source_mapping']:
                output['source_mapping'] = d['source_mapping']
        
        # Priority 2: List in parameters (fallback if no explicit mapping)
        if 'source_mapping' not in output and self.parameters.source_neurons:
            groups = self.parameters.get_source_groups()
            labels = generate_smart_labels(groups, self.parameters.source_labels, suffix="_source")
            
            source_data = {'custom_label': labels}
            
            # If auto_type_mapping is enabled, resolve types per dataset
            if auto_mapper:
                for ds in dataset_names:
                    resolved_groups = []
                    for group in groups:
                        if isinstance(group, list):
                            resolved_group = resolve_types_for_dataset(group, ds)
                        else:
                            resolved_group = resolve_types_for_dataset([group], ds)
                        if resolved_group:  # Only add non-empty groups
                            resolved_groups.append(resolved_group)
                    source_data[ds] = resolved_groups if resolved_groups else groups
            else:
                for ds in dataset_names:
                    source_data[ds] = groups
                    
            output['source_mapping'] = source_data

        # --- Target Mapping ---
        # Priority 1: Analyzer's label_mapper
        if self.label_mapper:
            d = self.label_mapper.to_dict()
            if 'target_mapping' in d and d['target_mapping']:
                output['target_mapping'] = d['target_mapping']
                
        # Priority 2: List in parameters (fallback)
        if 'target_mapping' not in output and self.parameters.target_neurons:
            groups = self.parameters.get_target_groups()
            labels = generate_smart_labels(groups, self.parameters.target_labels, suffix="_target")
            
            target_data = {'custom_label': labels}
            
            # If auto_type_mapping is enabled, resolve types per dataset
            if auto_mapper:
                for ds in dataset_names:
                    resolved_groups = []
                    for group in groups:
                        if isinstance(group, list):
                            resolved_group = resolve_types_for_dataset(group, ds)
                        else:
                            resolved_group = resolve_types_for_dataset([group], ds)
                        if resolved_group:  # Only add non-empty groups
                            resolved_groups.append(resolved_group)
                    target_data[ds] = resolved_groups if resolved_groups else groups
            else:
                for ds in dataset_names:
                    target_data[ds] = groups
                    
            output['target_mapping'] = target_data

        # --- Intermediate Mapping ---
        if self.label_mapper:
            d = self.label_mapper.to_dict()
            if 'intermediate_mapping' in d:
                output['intermediate_mapping'] = d['intermediate_mapping']
        
        # Add auto_type_mapping metadata
        output['metadata'] = {
            'auto_type_mapping': self.parameters.auto_type_mapping,
            'description': 'Type mappings per dataset. When auto_type_mapping=True, types are resolved to their dataset-specific equivalents.'
        }

        if output:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2)

    def _export_cross_dataset_combinations(self, comparison_results_dir: str):
        """Export comparison tables for explicit row-wise threshold queries.

        The legacy exporter is scalar-threshold oriented.  Combination mode
        uses this path so every aggregate row carries its query ID and the
        requested threshold map instead of iterating the derived union.
        """
        dataset_names = self.parameters.get_dataset_names()
        path_counts = []
        edge_rows = []
        summary_rows = []
        motif_rows = []

        for query in self.get_threshold_queries():
            query_id = query.get('id') or query.get('query_id')
            query_label = query.get('label', query_id)
            threshold_map = query.get('thresholds', {})
            aligned = self.get_aligned_data_for_query(query)

            self._export_presence_matrix(
                comparison_results_dir, query, silent=True)
            self._export_path_presence_matrix(
                comparison_results_dir, query, silent=True)
            motif_rows.extend(
                self._export_motif_analysis(comparison_results_dir, query)
                or []
            )

            for dataset in dataset_names:
                threshold = int(threshold_map[dataset])
                df = self.raw_results.get(dataset, {}).get(
                    threshold, pd.DataFrame())
                if not df.empty:
                    if {'type_pre', 'type_post'} <= set(df.columns):
                        count = int(
                            df[['type_pre', 'type_post']].drop_duplicates().shape[0]
                        )
                    elif {'source', 'target'} <= set(df.columns):
                        count = int(
                            df[['source', 'target']].drop_duplicates().shape[0]
                        )
                    else:
                        count = int(len(df))
                    total_weight = (
                        float(df['weight'].sum())
                        if 'weight' in df.columns else 0.0
                    )
                else:
                    count = 0
                    total_weight = 0.0
                provenance = self._path_provenance_row(dataset, threshold)
                row = {
                    'query_id': query_id,
                    'query_label': query_label,
                    'threshold_mode': 'combinations',
                    'dataset': dataset,
                    'threshold': threshold,
                    'connection_count': count,
                    'total_weight': total_weight,
                    'avg_weight': total_weight / count if count else 0,
                    **provenance,
                }
                path_counts.append(row)
                summary_rows.append({
                    'query_id': query_id,
                    'query_label': query_label,
                    'threshold_mode': 'combinations',
                    'dataset': self.parameters._sanitize_name(dataset),
                    'threshold': threshold,
                    'total_edges': count,
                    'total_layer_rows': len(df),
                    'total_weight': total_weight,
                    'mean_weight': (
                        float(df['weight'].mean())
                        if not df.empty and 'weight' in df.columns else 0
                    ),
                    **provenance,
                })

            if aligned.empty:
                continue
            available = [
                dataset for dataset in dataset_names
                if dataset in aligned.columns
            ]
            coverage = self._type_coverage_for_query(query_id)
            for edge_key, values in aligned.iterrows():
                edge_text = str(edge_key)
                if ' -> ' in edge_text:
                    source, target = edge_text.split(' -> ', 1)
                else:
                    source, target = edge_text, ''
                row = {
                    'query_id': query_id,
                    'query_label': query_label,
                    'threshold_mode': 'combinations',
                    'edge_key': edge_text,
                    'source': source,
                    'target': target,
                    'presence_count': int(
                        (values[available] > 0).sum()) if available else 0,
                    'total_datasets': len(available),
                }
                for dataset in available:
                    safe_name = self.parameters._sanitize_name(dataset)
                    row[f'weight_{safe_name}'] = values.get(dataset, 0)
                    row[f'threshold_{safe_name}'] = threshold_map.get(dataset)
                    # Endpoint coverage statuses (type_coverage.py): why a
                    # dataset shows no weight for this edge.
                    src_entry = coverage.get((source, dataset))
                    tgt_entry = coverage.get((target, dataset))
                    row[f'source_status_{safe_name}'] = (
                        src_entry.status if src_entry is not None else '')
                    row[f'target_status_{safe_name}'] = (
                        tgt_entry.status if tgt_entry is not None else '')
                    provenance = self._path_provenance_row(
                        dataset, threshold_map[dataset])
                    row[f'applied_threshold_{safe_name}'] = provenance.get(
                        'applied_threshold')
                    row[f'applied_source_{safe_name}'] = provenance.get(
                        'applied_threshold_source')
                    # Keep the edge-level aggregate self-describing.  The
                    # normalized threshold_combinations.csv remains the
                    # canonical long-form join, while these columns make a
                    # single exported edge row sufficient for audit tools.
                    for field in (
                            'strongest_first_budget',
                            'strongest_first_budget_bitten',
                            'strongest_first_tau', 'tau_canonical',
                            'strongest_dropped_bottleneck',
                            'strongest_retained_bottleneck', 'edge_budget',
                            'edge_budget_applied', 'edge_budget_landing',
                            'edge_weight_floor', 'paths_complete'):
                        row[f'{field}_{safe_name}'] = provenance.get(field)
                weights = [
                    values.get(dataset, 0) for dataset in available
                    if values.get(dataset, 0) > 0
                ]
                row['max_weight'] = max(weights) if weights else 0
                row['avg_weight'] = (
                    sum(weights) / len(weights) if weights else 0
                )
                edge_rows.append(row)

        if path_counts:
            self._save_csv(
                pd.DataFrame(path_counts),
                os.path.join(comparison_results_dir, 'path_count_comparison.csv'),
            )
        if summary_rows:
            self._save_csv(
                pd.DataFrame(summary_rows),
                os.path.join(comparison_results_dir, 'unified_summary.csv'),
            )
        if edge_rows:
            edge_df = pd.DataFrame(edge_rows)
            self._save_csv(
                edge_df,
                os.path.join(comparison_results_dir, 'edge_weight_comparison.csv'),
            )
            self._save_csv(
                edge_df,
                os.path.join(
                    comparison_results_dir, 'unified_edge_comparison.csv'),
            )
        if motif_rows:
            self._save_csv(
                pd.DataFrame(motif_rows),
                os.path.join(comparison_results_dir, 'motif_analysis.csv'),
            )

        # The scalar HTML generator historically owns these files.  Advanced
        # threshold rows bypass that generator, so persist the same similarity
        # data here with the stable query id as the join key instead of
        # inventing a scalar threshold from the raw-run union.
        similarities = (self.comparison_report or {}).get(
            'threshold_similarities', pd.DataFrame())
        if isinstance(similarities, pd.DataFrame) and not similarities.empty:
            similarity_dir = os.path.join(
                self.parameters.full_output_path, 'similarity_matrices')
            os.makedirs(similarity_dir, exist_ok=True)
            all_similarity_rows = []
            for query_id, query_df in similarities.groupby('query_id'):
                query_df = query_df.copy()
                query_df['threshold_mode'] = 'combinations'
                all_similarity_rows.append(query_df)
                safe_query_id = self._safe_query_filename_id(query_id)
                query_path = os.path.join(
                    similarity_dir,
                    f'similarity_query_{safe_query_id}.csv',
                )
                self._save_csv(query_df, query_path)
            if all_similarity_rows:
                self._save_csv(
                    pd.concat(all_similarity_rows, ignore_index=True),
                    os.path.join(similarity_dir, 'similarity_by_query.csv'),
                )

        # Metadata and neuron-count exports are query-independent and remain
        # useful in combination mode.  Scalar top-edge/degree helpers are not
        # called here because their old APIs would silently choose a union
        # threshold.
        self._export_metadata_comparison(comparison_results_dir)
        self._export_neuron_counts_comparison(comparison_results_dir)

    def _export_cross_dataset_comparisons(self, comparison_results_dir: str):
        """
        Export cross-dataset comparison results at each threshold level.
        
        Compares connections across datasets at the same synapse count cutoff.
        """
        if self.parameters.threshold_mode == 'combinations':
            self._export_cross_dataset_combinations(comparison_results_dir)
            return

        dataset_names = self.parameters.get_dataset_names()
        
        # 1. Path count comparison across datasets
        # N6: count UNIQUE (source, target) pairs, not rows — conn rows
        # keep per-layer occurrences of the same pair.
        path_counts = []
        for dataset in dataset_names:
            for threshold in self.parameters.thresholds:
                df = self.raw_results.get(dataset, {}).get(threshold, pd.DataFrame())
                if not df.empty:
                    if 'type_pre' in df.columns and 'type_post' in df.columns:
                        count = df[['type_pre', 'type_post']].drop_duplicates().shape[0]
                    elif 'source' in df.columns and 'target' in df.columns:
                        count = df[['source', 'target']].drop_duplicates().shape[0]
                    else:
                        count = len(df)
                    total_weight = df['weight'].sum() if 'weight' in df.columns else 0
                else:
                    count = 0
                    total_weight = 0
                path_counts.append({
                    'dataset': dataset,
                    'threshold': threshold,
                    'connection_count': count,
                    'total_weight': total_weight,
                    'avg_weight': total_weight / count if count > 0 else 0
                })
        
        if path_counts:
            path_count_df = pd.DataFrame(path_counts)
            self._save_csv(path_count_df, os.path.join(comparison_results_dir, "path_count_comparison.csv"))
            self._log("Saved: path_count_comparison.csv")
        
        # 2. Common and unique connections at each threshold
        # To-Do List 5 Item 2: Remove redundant files
        # - unique_to_*.csv per threshold → merged into unique_to_{dataset}.csv 
        # - common_connections_*.csv → redundant with edge_presence_matrix
        # - conserved_strong_connections_*.csv → redundant with edge_presence_matrix
        # - aligned_data/ folder → removed
        
        # Collect all motif data for unified export (To-Do List 5 Item 6)
        all_motif_data = []
        
        # Use progress bar for threshold exports
        # Feature G: thresholds skipped as τ-collapse duplicates are
        # excluded here — their matrices would be identical to the
        # duplicated run's (directive: skip downstream analysis/export).
        analysis_thresholds = self._analysis_thresholds()
        threshold_iter = tqdm(
            analysis_thresholds,
            desc="  Exporting matrices",
            unit="thr",
            leave=False
        ) if self.verbose else analysis_thresholds

        for threshold in analysis_thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            
            # Export presence matrices directly to comparison_results/ (no output_cutoffs subfolder)
            self._export_presence_matrix(comparison_results_dir, threshold, silent=True)
            
            # Export path presence matrix (multi-hop paths)
            self._export_path_presence_matrix(comparison_results_dir, threshold, silent=True)
            
            # Collect motif analysis data (for unified export)
            motif_data = self._export_motif_analysis(comparison_results_dir, threshold)
            if motif_data:
                all_motif_data.extend(motif_data)
        
        # Log summary after loop
        self._log(f"Exported edge/path presence matrices for {len(self.parameters.thresholds)} thresholds")
        
        # To-Do List 5 Item 6: Save unified motif_analysis.csv with all thresholds
        if all_motif_data:
            motif_df = pd.DataFrame(all_motif_data)
            # Reorder columns: dataset, threshold first
            cols = ['dataset', 'threshold'] + [c for c in motif_df.columns if c not in ['dataset', 'threshold']]
            motif_df = motif_df[cols]
            self._save_csv(motif_df, os.path.join(comparison_results_dir, "motif_analysis.csv"))
            self._log(f"Saved: motif_analysis.csv (unified, {len(motif_df)} rows)")
        
        # 3. Edge weight comparison matrix - includes all datasets, thresholds, with presence/difference cols
        # Use vectorized operations for performance
        all_threshold_dfs = []
        
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            
            available = [d for d in dataset_names if d in aligned.columns]
            safe_names = {d: self.parameters._sanitize_name(d) for d in available}
            
            # Build threshold dataframe using vectorized operations
            # Ensure index is a flat string index (not MultiIndex)
            # N2: wrap in pd.Series — Index.astype(str) returns an Index,
            # and Index.str.split(expand=True) yields a MultiIndex (not a
            # DataFrame), which silently sent every row to the whole-key
            # fallback (source = full key, target = '').
            if isinstance(aligned.index, pd.MultiIndex):
                edge_keys = pd.Series([f"{idx[0]} -> {idx[1]}" for idx in aligned.index], index=aligned.index)
            else:
                edge_keys = pd.Series(aligned.index.astype(str), index=aligned.index)
            
            # Parse edge keys vectorized with defensive handling.
            # N2 fix: rsplit (not fillna-with-Series, which raises on the
            # duplicate aligned-index labels that per-layer rows produce)
            # and dedupe rows to one per edge key.
            try:
                split_keys = edge_keys.str.rsplit(' -> ', n=1, expand=True)
                if isinstance(split_keys, pd.DataFrame) and split_keys.shape[1] == 2:
                    source_col = split_keys[0].fillna('')
                    target_col = split_keys[1].fillna('')
                else:
                    source_col = edge_keys
                    target_col = pd.Series('', index=aligned.index)
            except Exception:
                src_list, tgt_list = [], []
                for _k in edge_keys:
                    _s = str(_k)
                    if ' -> ' in _s:
                        _a, _b = _s.rsplit(' -> ', 1)
                        src_list.append(_a)
                        tgt_list.append(_b)
                    else:
                        src_list.append(_s)
                        tgt_list.append('')
                source_col = pd.Series(src_list, index=aligned.index)
                target_col = pd.Series(tgt_list, index=aligned.index)

            threshold_df = pd.DataFrame({
                'edge_key': edge_keys,
                'source': source_col.values,
                'target': target_col.values,
                'threshold': threshold,
            }, index=aligned.index)
            
            # Add weight columns for each dataset
            for dataset in available:
                safe_name = safe_names[dataset]
                threshold_df[f'weight_{safe_name}'] = aligned[dataset]
            
            # Calculate presence count vectorized
            threshold_df['presence_count'] = (aligned[available] > 0).sum(axis=1)
            threshold_df['total_datasets'] = len(available)
            
            # Calculate statistics vectorized
            weight_values = aligned[available].copy()
            weight_values = weight_values.replace(0, np.nan)  # Exclude zeros from stats
            
            threshold_df['max_weight'] = weight_values.max(axis=1).fillna(0)
            threshold_df['avg_weight'] = weight_values.mean(axis=1).round(2).fillna(0)
            
            # Weight diff and ratio (only meaningful when >1 dataset has the edge)
            has_multiple = weight_values.notna().sum(axis=1) > 1
            max_vals = weight_values.max(axis=1)
            min_vals = weight_values.min(axis=1)
            
            threshold_df['weight_diff'] = (max_vals - min_vals).where(has_multiple, 0).fillna(0)
            
            # Weight ratio - avoid division by zero
            ratio = (max_vals / min_vals).round(2)
            threshold_df['weight_ratio'] = ratio.where(has_multiple & (min_vals > 0), '')
            # Handle cases where min is 0 but has_multiple is True
            threshold_df.loc[has_multiple & (min_vals == 0), 'weight_ratio'] = ''
            threshold_df.loc[~has_multiple, 'weight_ratio'] = 1.0

            # Aligned rows keep per-layer occurrences of the same pair —
            # collapse to one row per edge key (per-dataset weights are
            # identical across those rows).
            threshold_df = threshold_df.reset_index(drop=True).drop_duplicates(
                subset=['edge_key'])

            # N5: label untyped-neuron bodyId fallbacks so bodyId-vs-type
            # rows are visible in cross-dataset comparisons.
            for _col in ['source', 'target']:
                _is_bodyid = threshold_df[_col].astype(str).str.fullmatch(r'\d+')
                threshold_df.loc[_is_bodyid, _col] = (
                    'bodyId:' + threshold_df.loc[_is_bodyid, _col].astype(str)
                    + ' (untyped)')

            all_threshold_dfs.append(threshold_df)
        
        if all_threshold_dfs:
            edge_weight_df = pd.concat(all_threshold_dfs, ignore_index=True)
            # Order columns logically
            col_order = ['edge_key', 'source', 'target', 'threshold', 'presence_count', 'total_datasets']
            # Add weight columns
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                weight_col = f'weight_{safe_name}'
                if weight_col in edge_weight_df.columns:
                    col_order.append(weight_col)
            col_order.extend(['max_weight', 'avg_weight', 'weight_diff', 'weight_ratio'])
            col_order = [c for c in col_order if c in edge_weight_df.columns]
            edge_weight_df = edge_weight_df[col_order]
            
            self._save_csv(edge_weight_df, os.path.join(comparison_results_dir, "edge_weight_comparison.csv"))
            self._log(f"Saved: edge_weight_comparison.csv ({len(edge_weight_df)} edges)")
        
        # 4. Top edges comparison
        self._export_top_edges_comparison(comparison_results_dir)
        
        # 5. Degree distribution analysis
        self._export_degree_distribution(comparison_results_dir)
        
        # 6. Dataset metadata comparison
        self._export_metadata_comparison(comparison_results_dir)
        
        # 7. Unified summary CSVs (merged across thresholds)
        self._export_unified_summary(comparison_results_dir)
        
        # 8. Source/target neuron counts comparison
        self._export_neuron_counts_comparison(comparison_results_dir)
    
    def _export_intra_dataset_comparisons(self, comparison_results_dir: str):
        """
        Export intra-dataset threshold sensitivity analysis.
        
        Compares how connections change across different threshold levels within each dataset.
        """
        if self.parameters.threshold_mode == 'combinations':
            self._export_combination_threshold_sensitivity(comparison_results_dir)
            return

        dataset_names = self.parameters.get_dataset_names()

        sensitivity_data = []
        provenance_data = []

        def _cap_fields(dataset, threshold):
            """Fix A + Feature G + F6: per-row run state.

            Returns (tau, paths_complete, skipped, duplicate_of, floor).
            tau is the budget cutoff for budget-bitten runs; for COMPLETE
            runs it is the natural τ (min emitted bottleneck) with
            paths_complete=True — meaning "every threshold up to τ yields
            this identical set". Skipped thresholds (Feature G τ collapse)
            carry the duplicated run's state plus the markers. floor is
            the Edge-Budget floor (Fix D) when the lossy floor fired.
            """
            meta = self._path_run_meta.get((dataset, threshold))
            if meta is not None:
                tau = meta.get('tau')
                complete = bool(meta.get('paths_complete', tau is None))
                return (tau, complete, bool(meta.get('skipped')),
                        meta.get('duplicate_of'),
                        meta.get('edge_weight_floor'))
            tau = self._path_taus.get((dataset, threshold))
            if tau is not None:
                return tau, False, False, None, None   # budgeted: complete only at >= tau
            return None, True, False, None, None       # complete enumeration

        for dataset in dataset_names:
            prev_edges = None
            prev_threshold = None

            for threshold in self.parameters.get_thresholds_for_dataset(dataset):
                df = self.raw_results.get(dataset, {}).get(threshold, pd.DataFrame())
                provenance_row = self._path_provenance_row(dataset, threshold)
                provenance_data.append(dict(provenance_row))
                tau = provenance_row['tau']
                paths_complete = provenance_row['paths_complete']
                skipped = provenance_row['skipped']
                duplicate_of = provenance_row['duplicate_of']
                floor = provenance_row['edge_weight_floor']

                # Concern 1: asked vs APPLIED threshold side by side.
                # applied = the CANONICAL minimal threshold reproducing
                # this run's output (w2 + 1 for a budget-bitten run, the
                # asked threshold for complete runs); tau is the budget
                # landing (the collapse bound — every threshold in
                # [applied, tau] yields this identical set).
                dropped = provenance_row['strongest_dropped_bottleneck']
                drop_stats = self._untyped_drop_stats.get((dataset, threshold), {})
                row = dict(provenance_row)
                # Retain the old column name used by downstream notebooks.
                row['strongest_dropped'] = dropped
                # Untyped-neuron drop (default on).
                row['untyped_dropped_rows'] = drop_stats.get('rows', 0)
                row['untyped_dropped_neurons'] = drop_stats.get('neurons', 0)

                if df.empty:
                    row.update({
                        'edge_count': 0,
                        'edges_retained_from_prev': None,
                        'retention_rate': None,
                        'edges_lost': None,
                    })
                    sensitivity_data.append(row)
                    prev_edges = set()
                    prev_threshold = threshold
                    continue

                # Create edge identifiers
                if 'type_pre' in df.columns and 'type_post' in df.columns:
                    current_edges = set(zip(df['type_pre'], df['type_post']))
                elif 'bodyId_pre' in df.columns and 'bodyId_post' in df.columns:
                    current_edges = set(zip(df['bodyId_pre'], df['bodyId_post']))
                else:
                    current_edges = set(range(len(df)))

                edge_count = len(current_edges)

                if prev_edges is not None:
                    retained = len(current_edges & prev_edges)
                    lost = len(prev_edges - current_edges)
                    retention_rate = retained / len(prev_edges) if prev_edges else None
                else:
                    retained = None
                    lost = None
                    retention_rate = None

                row.update({
                    'edge_count': edge_count,
                    'edges_retained_from_prev': retained,
                    'retention_rate': retention_rate,
                    'edges_lost': lost,
                })
                sensitivity_data.append(row)

                prev_edges = current_edges
                prev_threshold = threshold
        
        if sensitivity_data:
            sensitivity_df = pd.DataFrame(sensitivity_data)
            self._save_csv(sensitivity_df, os.path.join(comparison_results_dir, "threshold_sensitivity.csv"))
            self._log("Saved: threshold_sensitivity.csv")
        if provenance_data:
            provenance_df = pd.DataFrame(provenance_data)
            self._save_csv(
                provenance_df,
                os.path.join(comparison_results_dir,
                             "pathfinding_provenance.csv"),
            )
            self._log("Saved: pathfinding_provenance.csv")

    def _export_combination_threshold_sensitivity(self, comparison_results_dir: str):
        """Export query-keyed raw-run sensitivity/provenance for combinations.

        The union of cell values is an execution schedule, not a comparison
        axis.  Keep this diagnostic useful by repeating a raw run under each
        query identity, rather than emitting rows that look like independent
        scalar comparisons.
        """
        sensitivity_rows = []
        provenance_rows = []
        datasets = self.parameters.get_dataset_names()
        for query_index, query in enumerate(self.get_threshold_queries(), start=1):
            query_id = query.get('id') or query.get('query_id')
            query_label = query.get('label', query_id)
            threshold_map = query.get('thresholds') or {}
            for dataset in datasets:
                threshold = int(threshold_map[dataset])
                provenance = self._path_provenance_row(dataset, threshold)
                provenance.update({
                    'query_id': query_id,
                    'query_index': query_index,
                    'query_label': query_label,
                    'threshold_mode': 'combinations',
                    'requested_threshold': threshold,
                    'threshold_scope': 'query_cell',
                })
                provenance_rows.append(dict(provenance))

                df = self.raw_results.get(dataset, {}).get(
                    threshold, pd.DataFrame())
                if {'type_pre', 'type_post'} <= set(df.columns):
                    edge_count = int(
                        df[['type_pre', 'type_post']].drop_duplicates().shape[0]
                    )
                elif {'source', 'target'} <= set(df.columns):
                    edge_count = int(
                        df[['source', 'target']].drop_duplicates().shape[0]
                    )
                else:
                    edge_count = int(len(df))
                row = dict(provenance)
                row.update({
                    'edge_count': edge_count,
                    # Query rows are not a monotone within-dataset schedule;
                    # adjacent-threshold retention is therefore undefined.
                    'edges_retained_from_prev': None,
                    'retention_rate': None,
                    'edges_lost': None,
                })
                sensitivity_rows.append(row)

        if sensitivity_rows:
            self._save_csv(
                pd.DataFrame(sensitivity_rows),
                os.path.join(comparison_results_dir, 'threshold_sensitivity.csv'),
            )
            self._log('Saved: threshold_sensitivity.csv (query-keyed)')
        if provenance_rows:
            self._save_csv(
                pd.DataFrame(provenance_rows),
                os.path.join(comparison_results_dir,
                             'pathfinding_provenance.csv'),
            )
            self._log('Saved: pathfinding_provenance.csv (query-keyed)')

    # ------------------------------------------------------------------
    # Feature C: query-specific threshold alignment (alignment spec §6)
    # ------------------------------------------------------------------

    def _alignment_extract_for_dataset(self, dataset_name: str):
        """Lowest-threshold extract feeding the prober (spec §6.1).

        Prefers the run's mapped lowest-threshold frame (in memory, both
        modes, type-mapped); falls back to the persisted bodyId CSV /
        direct query used by edge mode.
        """
        mapped = self.get_mapped_results()
        thresholds_ds = self.parameters.get_thresholds_for_dataset(dataset_name)
        lowest = thresholds_ds[0] if thresholds_ds else None
        df = mapped.get(dataset_name, {}).get(lowest) if lowest is not None else None
        if isinstance(df, pd.DataFrame) and not df.empty \
                and {'type_pre', 'type_post', 'weight'} <= set(df.columns):
            return df
        # Fallback: bodyId-level extract (path mode at the lowest threshold
        # or the edge-mode direct query). Type names there are unmapped —
        # consistent per dataset, still valid for edge-count matching.
        bodyid_df, _ = self._get_bodyid_connections_for_dataset(
            dataset_name, lowest, skip_existing=True)
        if isinstance(bodyid_df, pd.DataFrame) and not bodyid_df.empty \
                and {'type_pre', 'type_post', 'weight'} <= set(bodyid_df.columns):
            return bodyid_df
        return None

    def _read_applied_folder_provenance(self, dataset: str,
                                        applied: int) -> Dict:
        """Read the authoritative provenance of the folder that holds an
        applied threshold (its own all_attributes.json, if present)."""
        try:
            folder = self._resolve_dataset_output_path(dataset, applied)
            attrs_path = os.path.join(folder, 'all_attributes.json')
            if os.path.exists(attrs_path):
                with open(attrs_path, encoding='utf-8') as f:
                    attrs = json.load(f)
                if isinstance(attrs, dict):
                    return attrs
        except Exception:
            pass
        return {}

    def _export_threshold_alignment(self, comparison_results_dir: str):
        """Export the alignment outputs (spec §6.4) and log the summary.

        - threshold_alignment_best_matches.csv — prober best matches per
          (dataset pair, anchor threshold) incl. named anchor rows and a
          global-best row; Jaccard/rank computed ONLY at best_t; carries
          the anchor's per-run tau/paths_complete state.
        - threshold_alignment_matrix.csv (+ heatmap) — typed grid points
          only (§6.3).
        - edge_density_per_threshold.csv — typed counts + the prober's
          extended-grid counts (feeds the Feature D curve plot).

        In combination mode these files are explicitly raw-run schedule
        diagnostics.  They compare the deduplicated cell-value grid for
        density/alignment exploration; they are not comparison queries.  The
        query-specific comparison exports are keyed by ``query_id`` in the
        combination manifest and aggregate tables.
        """
        try:
            from .threshold_alignment import (
                EdgeDensityProber, build_alignment_matrix,
                edge_count_distance, jaccard, rank_similarity,
                ALIGNMENT_TOLERANCE,
            )
        except ImportError:  # pragma: no cover - direct package imports
            from threshold_alignment import (
                EdgeDensityProber, build_alignment_matrix,
                edge_count_distance, jaccard, rank_similarity,
                ALIGNMENT_TOLERANCE,
            )

        dataset_names = self.parameters.get_dataset_names()
        if len(dataset_names) < 1:
            return

        # --- Build one prober per dataset on the lowest-threshold extract
        probers: Dict[str, EdgeDensityProber] = {}
        extract_sizes: Dict[str, int] = {}
        for ds in dataset_names:
            extract = self._alignment_extract_for_dataset(ds)
            probers[ds] = EdgeDensityProber(extract)
            extract_sizes[ds] = 0 if extract is None else len(extract)

        # Anchors are the MATERIALIZED (applied) thresholds, not the
        # requested list: under tau-collapse several requested thresholds
        # share one applied frame, which previously emitted duplicate anchor
        # rows. Each anchor carries the requested thresholds that alias to it.
        anchor_map = {ds: self.get_applied_thresholds(ds)
                      for ds in dataset_names}
        aliased_map: Dict[str, Dict[int, List[int]]] = {}
        for ds in dataset_names:
            per_applied: Dict[int, List[int]] = {}
            for req in self.parameters.get_thresholds_for_dataset(ds):
                applied = self.get_applied_folder(ds, int(req))
                if applied is not None and int(applied) != int(req) \
                        and self._is_duplicate_threshold(ds, int(req)):
                    per_applied.setdefault(int(applied), []).append(int(req))
            aliased_map[ds] = per_applied
        max_anchor = max(
            (max(ts) for ts in anchor_map.values() if ts), default=10)
        cap = max(2 * int(max_anchor), 30)
        # §3.5: no reason to share ONE grid — each dataset gets its own
        # cap from its own anchors, so a dataset is never truncated to
        # another dataset's request parameter.
        dataset_caps = {
            ds: max(2 * int(max(anchor_map[ds])), 30)
            if anchor_map[ds] else max(2, min(cap, 30))
            for ds in dataset_names
        }

        # --- Edge density per threshold (typed + extended grid)
        density_rows = []
        neuron_totals = {}
        # §3.4/§4.3: the pairing numerator is query-scoped, so the
        # denominator must be too. Use the searched-graph node count from
        # the density capture when present; otherwise fall back to the
        # whole-dataset neuron total and label the scope honestly.
        searched_nodes: Dict[str, Optional[int]] = {}
        try:
            for _ds in dataset_names:
                _meta, _bns, _ew, _cls = self._load_density_artifacts(_ds)
                if _meta:
                    searched_nodes[_ds] = int(
                        _meta.get('n_nodes_typed')
                        or _meta.get('n_annotated_nodes')
                        or _meta.get('n_nodes') or 0) or None
                else:
                    searched_nodes[_ds] = None
        except Exception:
            searched_nodes = {ds: None for ds in dataset_names}
        try:
            if not getattr(self, '_dataset_metadata', None):
                self.collect_dataset_metadata(force_refresh=False)
            neuron_totals = {
                ds: int(self._dataset_metadata.get(ds, {}).get(
                    'neuron_counts', {}).get('total', 0) or 0)
                for ds in dataset_names
            }
        except Exception:
            pass

        typed_set = {t: set(ts) for t, ts in anchor_map.items()}
        for ds in dataset_names:
            denom_scope = ('searched_graph'
                           if searched_nodes.get(ds) else 'whole_dataset')
            denom = (searched_nodes.get(ds) if denom_scope == 'searched_graph'
                     else neuron_totals.get(ds))
            for t in range(1, dataset_caps[ds] + 1):
                count = probers[ds].count(t)
                density_rows.append({
                    'dataset': ds,
                    'threshold': t,
                    'pair_count': count,
                    'pairs_per_neuron': round(count / denom, 4)
                        if denom else None,
                    'basis': 'type_pairs',
                    'denominator_scope': denom_scope,
                    'is_typed_threshold': t in typed_set[ds],
                })
        density_df = pd.DataFrame(density_rows)
        alignment_scope = (
            'raw_run_schedule_diagnostic'
            if self.parameters.threshold_mode == 'combinations'
            else 'standard_threshold_grid'
        )
        density_df['threshold_mode'] = getattr(
            self.parameters, 'threshold_mode', 'standard')
        density_df['threshold_scope'] = alignment_scope
        density_path = os.path.join(comparison_results_dir, "edge_density_per_threshold.csv")
        self._save_csv(density_df, density_path)
        self._log("Saved: edge_density_per_threshold.csv")
        self._alignment_density_df = density_df
        self._alignment_grid_points = [
            (ds, t) for ds in dataset_names for t in anchor_map[ds]]
        self._alignment_anchor_map = {}

        # --- Typed-threshold-only alignment matrix (§6.3)
        grid_points = self._alignment_grid_points
        mapped_names = bool(
            self.parameters.auto_type_mapping
            and self.parameters._auto_type_mapper)
        matrix_df = build_alignment_matrix(grid_points, probers, mapped_names)
        if not matrix_df.empty:
            matrix_df['threshold_mode'] = getattr(
                self.parameters, 'threshold_mode', 'standard')
            matrix_df['threshold_scope'] = alignment_scope
            matrix_path = os.path.join(comparison_results_dir, "threshold_alignment_matrix.csv")
            self._save_csv(matrix_df, matrix_path)
            self._log("Saved: threshold_alignment_matrix.csv")
        self._alignment_matrix_df = matrix_df

        # --- Best matches via the prober (bisection, extended range)
        best_rows = []
        refs = [ds for ds in dataset_names if anchor_map[ds]]
        for ref_ds in refs:
            ref_taus = anchor_map[ref_ds]
            for anchor_t in ref_taus:
                n_a = probers[ref_ds].count(anchor_t)
                # Provenance of the applied anchor. Prefer the folder that
                # actually holds this applied value (authoritative), then a
                # non-skipped requester, then any requester.
                applied_meta = self._read_applied_folder_provenance(
                    ref_ds, anchor_t)
                if not applied_meta:
                    fallback = {}
                    for req in self.parameters.get_thresholds_for_dataset(
                            ref_ds):
                        if self.get_applied_folder(ref_ds, int(req)) != anchor_t:
                            continue
                        candidate = self._path_run_meta.get(
                            (ref_ds, int(req)), {}) or {}
                        if not candidate.get('skipped'):
                            fallback = candidate
                            break
                        fallback = fallback or candidate
                    applied_meta = fallback
                for other_ds in dataset_names:
                    if other_ds == ref_ds:
                        continue
                    if probers[other_ds].total_pairs == 0:
                        continue
                    match = probers[other_ds].best_match(
                        n_a, cap=dataset_caps.get(other_ds, cap))
                    best_t = match.get('best_t')
                    inter = match.get('best_t_range')
                    if best_t is None:
                        # Out-of-range: still report the shortfall honestly.
                        if match.get('match_status') != \
                                'target_density_below_range':
                            continue
                    jac = rank = None
                    if mapped_names and best_t is not None:
                        set_a = probers[ref_ds].edge_set(anchor_t)
                        set_b = probers[other_ds].edge_set(best_t)
                        jac = jaccard(set_a, set_b)
                        rank = rank_similarity(
                            probers[ref_ds].edge_weights(anchor_t),
                            probers[other_ds].edge_weights(best_t))
                    aliases = aliased_map.get(ref_ds, {}).get(int(anchor_t), [])
                    best_rows.append({
                        'reference_dataset': ref_ds,
                        'anchor_threshold': anchor_t,
                        'anchor_aliased_from': ','.join(
                            str(a) for a in aliases) if aliases else '',
                        'target_dataset': other_ds,
                        'anchor_pair_count': n_a,
                        'best_t': best_t,
                        'best_t_range': (f'{inter[0]}-{inter[1]}'
                                         if inter else None),
                        'count_at_best_t': match['count_at_best_t'],
                        'count_distance': round(match['count_distance'], 4),
                        'match_status': match.get(
                            'match_status', 'outside_tolerance'),
                        'within_tolerance': match['count_distance'] <= ALIGNMENT_TOLERANCE,
                        'jaccard_at_best': round(jac, 4) if jac is not None else None,
                        'rank_similarity_at_best': round(rank, 4) if rank is not None else None,
                        'anchor_tau': applied_meta.get('tau'),
                        'anchor_paths_complete': applied_meta.get(
                            'paths_complete', applied_meta.get('tau') is None),
                        'anchor_skipped': bool(applied_meta.get('skipped')),
                        'match_kind': 'anchor',
                    })
                    if best_t is not None:
                        key = (ref_ds, anchor_t)
                        self._alignment_anchor_map.setdefault(
                            key, {})[other_ds] = best_t

        # Global best row: the (reference anchor, target) pair with the
        # minimal edge-count distance overall.
        if best_rows:
            global_best = min(best_rows, key=lambda r: (
                r['count_distance'],
                -(r['jaccard_at_best'] or 0),
                -(r['rank_similarity_at_best'] or -1),
            ))
            global_row = dict(global_best)
            global_row.update({
                'reference_dataset': global_best['reference_dataset'],
                'match_kind': 'global_best',
                'anchor_threshold': global_best['anchor_threshold'],
                'target_dataset': global_best['target_dataset'],
            })
            best_rows.append(global_row)

        if best_rows:
            best_df = pd.DataFrame(best_rows)
            best_df['threshold_mode'] = getattr(
                self.parameters, 'threshold_mode', 'standard')
            best_df['threshold_scope'] = alignment_scope
            best_path = os.path.join(comparison_results_dir, "threshold_alignment_best_matches.csv")
            self._save_csv(best_df, best_path)
            self._log("Saved: threshold_alignment_best_matches.csv")
            self._alignment_best_df = best_df

            # One log line per run, anchor rows of the first dataset (§6.4)
            first_ds = refs[0]
            anchor_rows = [r for r in best_rows if r['match_kind'] == 'anchor']
            parts = []
            for anchor_t in anchor_map[first_ds][:3]:
                segs = []
                for r in anchor_rows:
                    if r['reference_dataset'] == first_ds \
                            and r['anchor_threshold'] == anchor_t:
                        segs.append(f"{r['target_dataset']}@{r['best_t']} "
                                    f"(d={r['count_distance']:.2f})")
                if segs:
                    parts.append(f"{first_ds}@{anchor_t} aligns best with "
                                 + ", ".join(segs))
            if parts:
                self._log("Alignment: " + "; ".join(parts))
            # Alignment-derived threshold combinations (plan §7C): suggest
            # density-equivalent query rows consumable by the advanced
            # combination mode.
            try:
                self._suggest_and_export_combinations(
                    best_df, dataset_order=dataset_names,
                    comparison_results_dir=comparison_results_dir)
            except Exception as e:
                self._log(f"Warning: could not suggest combinations: {e}")
        if self.parameters.threshold_mode == 'combinations':
            self._log(
                "Threshold alignment files are raw-run schedule diagnostics "
                "in combination mode; query comparisons are keyed by "
                "threshold_combinations.csv."
            )

    # ------------------------------------------------------------------
    # Auto threshold-density alignment (plan §5 Phases B/C/D/E/F/G)
    # ------------------------------------------------------------------

    def _dataset_density_dir(self, dataset_name: str):
        """Dataset-level ``_density/`` folder holding the shared arrays."""
        base = getattr(self.parameters, 'dataset_data_path', None)
        if not base:
            return None
        return os.path.join(base, self.parameters._sanitize_name(dataset_name),
                            '_density')

    def _load_density_artifacts(self, dataset_name: str):
        """Load ``(meta, path_bottlenecks, edge_weights, edge_classes)``.

        Prefers the dataset-level ``_density/`` folder; falls back to any
        ``minsyn_*`` folder carrying a ``density_meta.json`` whose
        ``density_source`` resolves to its array directory. Edge classes
        (typed/untyped/debris) come from ``density_edges.npz``; legacy
        captures without it return ``classes=None`` (all-edges basis).
        Returns ``(None, None, None, None)`` when no capture exists.
        """
        import numpy as np
        ddir = self._dataset_density_dir(dataset_name)
        candidates = []
        if ddir and os.path.exists(os.path.join(ddir, 'density_meta.json')):
            candidates.append(ddir)
        safe = self.parameters._sanitize_name(dataset_name)
        ds_root = os.path.join(
            getattr(self.parameters, 'dataset_data_path', ''), safe)
        if os.path.isdir(ds_root):
            try:
                for entry in sorted(os.listdir(ds_root)):
                    p = os.path.join(ds_root, entry, 'density_meta.json')
                    if entry != '_density' and os.path.exists(p):
                        candidates.append(os.path.join(ds_root, entry))
            except OSError:
                pass
        for folder in candidates:
            try:
                with open(os.path.join(folder, 'density_meta.json'),
                          encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:
                continue
            src = meta.get('density_source') or '.'
            arr_dir = os.path.normpath(os.path.join(folder, src))
            bns_path = os.path.join(arr_dir, 'density_path_bottlenecks.npy')
            ew_path = os.path.join(arr_dir, 'density_edge_weights.npy')
            npz_path = os.path.join(arr_dir, 'density_edges.npz')
            try:
                bns = (np.load(bns_path) if os.path.exists(bns_path)
                       else np.array([], dtype=np.float64))
                cls = None
                if os.path.exists(npz_path):
                    # The classified npz is the authoritative capture.
                    with np.load(npz_path) as z:
                        ew = np.asarray(z['weight'], dtype=np.int32)
                        cls = z['cls']
                else:
                    ew = (np.load(ew_path) if os.path.exists(ew_path)
                          else np.array([], dtype=np.int32))
            except Exception:
                continue
            return meta, bns, ew, cls
        return None, None, None, None

    def _build_density_curves(self):
        """Assemble per-dataset curves, windows and metas from artifacts.

        Returns ``(curves, windows, metas)`` where ``curves[ds]`` is
        ``{'thresholds', 'path_count', 'edge_count', 'density', 'basis'}``
        over the dataset's own integer domain, ``windows[ds] =
        (w_start, w_star)`` and ``metas[ds]`` is the persisted meta.
        Datasets without a capture are omitted (caller warns).

        The edge basis follows the run's ``drop_untyped`` policy over the
        classified cone (typed / untyped / debris): with the drop on, only
        edges between typed neurons count and ``N`` is the typed-node count;
        with the drop off, untyped edges are included but debris edges are
        still excluded (debris is never counted in any universe). Legacy
        captures without classes fall back to the all-edges basis.
        """
        import numpy as np
        from .threshold_density import (
            density_curve, dataset_window, normalized_edge_density,
        )
        # Mirror of coana.DENSITY_CLS_TYPED / _DEBRIS (kept local to avoid a
        # coana import from the comparison layer).
        _CLS_TYPED, _CLS_DEBRIS = 0, 2
        normalizer = getattr(
            self.parameters, 'density_normalizer', 'per_node') or 'per_node'
        drop_untyped = bool(getattr(self.parameters, 'drop_untyped', True))
        curves, windows, metas = {}, {}, {}
        for ds in self.parameters.get_dataset_names():
            meta, bns, ew, cls = self._load_density_artifacts(ds)
            if meta is None:
                continue
            lo, hi = dataset_window(meta)
            windows[ds] = (lo, hi)
            if cls is not None and len(cls) == len(ew):
                if drop_untyped:
                    ew_used = ew[cls == _CLS_TYPED]
                    basis = 'bodyId_edges_typed'
                    n_denom = meta.get('n_nodes_typed')
                else:
                    ew_used = ew[cls != _CLS_DEBRIS]
                    basis = 'bodyId_edges_all_but_debris'
                    n_denom = (meta.get('n_nodes_typed') or 0) + \
                        (meta.get('n_nodes_untyped') or 0) or None
            else:
                ew_used = ew
                basis = 'bodyId_edges'
                n_denom = meta.get('n_annotated_nodes') or meta.get('n_nodes')
            meta = dict(meta)
            meta['curve_n'] = n_denom
            meta['basis'] = basis
            if hi is None or hi < lo:
                t_grid = [int(lo)]
            else:
                t_grid = list(range(int(lo), int(hi) + 1))
            curve = density_curve(bns, ew_used, t_grid)
            dens = normalized_edge_density(
                curve['edge_count'], meta, normalizer)
            # Active-basis edge count at w_start — the number the curves
            # CSV's first row reports, directly comparable against the
            # capture-level (all-class) meta n_edges.
            meta['n_edges_active_basis'] = int(curve['edge_count'][0])
            metas[ds] = meta
            curves[ds] = {
                'thresholds': [int(t) for t in curve['t_grid']],
                'path_count': [int(c) for c in curve['path_count']],
                'edge_count': [int(c) for c in curve['edge_count']],
                'density': [float(d) for d in dens],
                'basis': basis,
            }
        return curves, windows, metas

    def _density_aligned_rows(self, curves, windows, metas):
        """Build the vertical + horizontal aligned rows (plan §4.4/§4.5).

        Datasets without a density capture are EXCLUDED from the horizontal
        (density-matched) rows instead of vetoing them entirely (plan C,
        2026-09-15: one missing capture used to drop every horizontal row).
        Rows covering only the captured datasets are tagged
        ``partial_datasets`` so the export and the report can label them;
        vertical rows keep the full shared integer spine.
        """
        from .threshold_density import (
            align_horizontal, align_vertical, vertical_rows,
        )
        order = list(self.parameters.get_dataset_names())
        captured = [ds for ds in order if curves.get(ds)]
        partial = [ds for ds in order if ds not in captured]
        rows: List[Dict[str, Any]] = []
        seen = set()

        def _add(row):
            thresholds = row.get('thresholds') or {}
            # Horizontal rows cover the CAPTURED datasets; vertical rows
            # keep the full shared spine. Anything else is malformed.
            expected = order if row.get('mode') == 'vertical' else captured
            if sorted(str(ds) for ds in thresholds) != \
                    sorted(str(ds) for ds in expected):
                return
            key = tuple(sorted(
                (ds, int(v)) for ds, v in thresholds.items()))
            if key in seen:
                # Identical to an earlier row (e.g. a horizontal level that
                # rounds onto a vertical point): emit once, tag both ( §4.6).
                for existing in rows:
                    if tuple(sorted(
                            (ds, int(v)) for ds, v in
                            existing['thresholds'].items())) == key:
                        origins = set(
                            (existing.get('mode') or '').split('+'))
                        origins.add(row['mode'])
                        existing['mode'] = '+'.join(sorted(
                            o for o in origins if o))
                        if partial and existing.get('mode') != 'vertical':
                            existing['partial_datasets'] = list(partial)
                        break
                return
            seen.add(key)
            if partial and row.get('mode') != 'vertical':
                row['partial_datasets'] = list(partial)
            rows.append(row)

        for r in vertical_rows(align_vertical(windows, K=5), order):
            _add(r)
        for r in align_horizontal(
                {ds: curves[ds] for ds in captured}, levels=4):
            _add(r)
        return rows

    def _export_density_alignment(self, comparison_results_dir: str) -> None:
        """Export density curves + windows for EVERY pathfinding mode.

        The per-dataset curves span ``[w_start, w_star_measured]`` over the
        dataset's searched cone (its lowest executed threshold), with the
        edge basis following the run's ``drop_untyped`` policy and debris
        always excluded. The aligned same-threshold/same-density rows and
        the ``[auto threshold]`` warnings are auto-mode products and are
        written only when this run measured them. Datasets without a
        capture are excluded from the horizontal rows (rows carry
        ``partial_datasets``; a ``[density partial]`` note is appended)
        instead of vetoing them. No-op with an explicit note when no
        dataset carries a capture.
        """
        curves, windows, metas = self._build_density_curves()
        if not curves:
            self._log("Density curves: no density capture found — skipping "
                      "density exports.")
            return
        is_auto = bool(getattr(self.parameters, 'threshold_auto', False))
        order = list(self.parameters.get_dataset_names())
        normalizer = getattr(
            self.parameters, 'density_normalizer', 'per_node') or 'per_node'
        caption = ("Density curves are computed on the query's bodyId "
                   "searched graph; the compared matrices are type-level "
                   "projections of the same search. A threshold here is a "
                   "per-connection synapse count (Min Synapse Count).")

        # --- density_curves.csv (Phase B) ---
        rows = []
        for ds in order:
            curve = curves.get(ds)
            if not curve:
                continue
            meta = metas.get(ds, {})
            lo, hi = windows.get(ds, (None, None))
            materialized = set(int(t) for t in self.get_applied_thresholds(ds))
            for i, t in enumerate(curve['thresholds']):
                rows.append({
                    'dataset': ds,
                    'threshold': int(t),
                    'path_count': curve['path_count'][i],
                    'edge_count': curve['edge_count'][i],
                    'density': (round(curve['density'][i], 6)
                                if curve['density'][i] is not None else None),
                    'normalized_edge_density': (
                        round(curve['density'][i], 6)
                        if curve['density'][i] is not None else None),
                    'normalizer': normalizer,
                    'basis': curve['basis'],
                    'w_start': lo,
                    'w_star_measured': hi,
                    'path_complete_from': meta.get('path_complete_from'),
                    'is_materialized': bool(int(t) in materialized),
                })
        if rows:
            density_curves_df = pd.DataFrame(rows)
            self._save_csv(density_curves_df,
                           os.path.join(comparison_results_dir,
                                        'density_curves.csv'))
            self._log("Saved: density_curves.csv")
            self._density_curves_df = density_curves_df
            self._log("  [density] " + caption)

        # --- density_windows.csv (Phase B) ---
        win_rows = []
        for ds in order:
            meta = metas.get(ds)
            if meta is None:
                continue
            lo, hi = windows.get(ds, (None, None))
            win_rows.append({
                'dataset': ds,
                'applied': meta.get('applied'),
                'w_start': lo,
                'w_star_stored': meta.get('strongest_retained_bottleneck'),
                'w_star_measured': hi,
                'w_star_mismatch': bool(meta.get('w_star_mismatch')),
                'budget_bitten': bool(meta.get('budget_bitten')),
                'paths_complete': bool(meta.get('paths_complete')),
                'path_complete_from': meta.get('path_complete_from'),
                'n_paths': meta.get('n_paths'),
                'n_edges': meta.get('n_edges'),
                'n_edges_active_basis': meta.get('n_edges_active_basis'),
                'n_nodes': meta.get('n_nodes'),
                'n_nodes_typed': meta.get('n_nodes_typed'),
                'n_nodes_untyped': meta.get('n_nodes_untyped'),
                'n_nodes_debris': meta.get('n_nodes_debris'),
                'n_annotated_nodes': meta.get('n_annotated_nodes'),
                'denominator': meta.get('denominator'),
            })
        if win_rows:
            windows_df = pd.DataFrame(win_rows)
            self._save_csv(windows_df, os.path.join(
                comparison_results_dir, 'density_windows.csv'))
            self._log("Saved: density_windows.csv")
            self._density_windows_df = windows_df

        status = getattr(self, '_auto_bootstrap_status', None)
        if status:
            self._append_auto_mode_status_notes(status)

        if not is_auto:
            # Aligned rows are the auto mode's measured product; other modes
            # get the curves/windows diagnostics only.
            return

        # --- aligned rows (Phase F) ---
        aligned = self._density_aligned_rows(curves, windows, metas)
        if aligned:
            # Id/label consistency: the bootstrap already installed the
            # aligned rows and the whole run (folders, provenance,
            # similarity cache) is keyed by THOSE ids. Re-installing here
            # with freshly generated ids would break every per-query join,
            # so reuse the installed id/label whenever the threshold cell
            # matches, and never re-install during export.
            installed = {}
            if getattr(self.parameters, 'threshold_auto', False):
                for q in (self.parameters.threshold_combinations or []):
                    installed[tuple(sorted(
                        (ds, int(v))
                        for ds, v in (q.get('thresholds') or {}).items()))] = (
                        str(q.get('id')), str(q.get('label')))
            # B7 fallback ids for not-installed rows (same scheme as the
            # installer, with the same `.4g` de-duplication).
            fallback_ids: Dict[str, int] = {}

            def _fallback_id(base: str) -> str:
                count = fallback_ids.get(base, 0)
                fallback_ids[base] = count + 1
                return base if count == 0 else f"{base}_{count + 1}"

            best_rows = []
            for i, r in enumerate(aligned, start=1):
                lvl_c = r['level_continuous']
                lvl_n = r['level_normalized']
                cell_key = tuple(sorted(
                    (ds, int(v)) for ds, v in r['thresholds'].items()))
                if cell_key in installed:
                    row_id, row_label = installed[cell_key]
                else:
                    if r.get('mode') == 'vertical':
                        base = (f"threshold={int(lvl_c)}"
                                if lvl_c is not None
                                else f"vertical_{i:03d}")
                    else:
                        base = f"aligned_density={lvl_n:.4g}"
                    row_id, row_label = _fallback_id(base), None
                if row_label is None:
                    _ds_txt = ', '.join(
                        f"{ds} {int(r['thresholds'][ds])}"
                        for ds in order if ds in r['thresholds'])
                    row_label = (
                        (f"threshold={int(lvl_c)}"
                         if lvl_c is not None else r['mode'])
                        if r['mode'] == 'vertical'
                        else f"aligned_density={lvl_n:.4g} ({_ds_txt})")
                row = {
                    'id': row_id,
                    'label': row_label,
                    'mode': r['mode'],
                    'level_continuous': (
                        round(float(lvl_c), 6) if lvl_c is not None else None),
                    'level_normalized': (
                        round(float(lvl_n), 6) if lvl_n is not None else None),
                    'degenerate': bool(r.get('degenerate')),
                    'clamped': bool(r.get('clamped')),
                    'w_star_mismatch': bool(any(
                        (metas.get(ds) or {}).get('w_star_mismatch')
                        for ds in order)),
                }
                if r.get('partial_datasets'):
                    row['partial_datasets'] = ', '.join(
                        r['partial_datasets'])
                for ds in order:
                    row[ds] = (int(r['thresholds'][ds])
                               if ds in r['thresholds'] else None)
                # Achieved per-dataset density at the chosen integer
                # threshold: a level maps to a plateau, and coarse curves
                # (e.g. BANC) can overshoot the level on quantization —
                # surface the realized value instead of hiding it (§4.5b
                # item 4).
                if r['mode'] != 'vertical' and lvl_n is not None:
                    achieved = {}
                    for ds, ds_t in r['thresholds'].items():
                        curve = curves.get(ds) or {}
                        dens = dict(zip(curve.get('thresholds') or [],
                                        curve.get('density') or []))
                        achieved[ds] = dens.get(int(ds_t))
                    if achieved and all(
                            v is not None for v in achieved.values()):
                        for ds, dens_v in achieved.items():
                            row[f'density_at_{ds}'] = round(float(dens_v), 6)
                        row['max_abs_deviation'] = round(max(
                            abs(v - float(lvl_n))
                            for v in achieved.values()), 6)
                best_rows.append(row)
            aligned_df = pd.DataFrame(best_rows)
            self._save_csv(aligned_df, os.path.join(
                comparison_results_dir,
                'density_alignment_best_matches.csv'))
            self._log("Saved: density_alignment_best_matches.csv")
            self._density_alignment_df = aligned_df
            self._density_aligned_rows_cache = aligned
            # NOTE: installation of these rows as the run's combination
            # schedule is owned by the bootstrap (_bootstrap_auto_mode),
            # which runs BEFORE any (dataset, threshold) execution. Do NOT
            # re-install here: the run's provenance, folders and similarity
            # cache are keyed by the installed ids, and a re-install with
            # regenerated ids would break every per-query join.

        self._append_density_warnings(
            curves, windows, metas, aligned, normalizer, comparison_results_dir)

    def _append_auto_mode_status_notes(self, status: Dict[str, Any]) -> None:
        """Record how auto mode resolved in the run folder, not just stdout.

        The bootstrap's diagnostics go through ``_log`` (a ``tqdm.write``), so a
        degraded schedule used to leave no trace in the outputs and a Standard
        report read as if it were the intended one (plan
        §auto-mode-density-regression P4, 2026-09-21).
        """
        outcome = str(status.get('outcome') or '')
        if outcome not in ('degraded', 'verticals_only'):
            return
        uncaptured = [str(ds) for ds in (status.get('uncaptured') or [])]
        failures = status.get('failures') or {}
        if outcome == 'degraded':
            head = (
                '- [auto threshold] auto mode did NOT resolve: no '
                'density-aligned schedule could be measured, so the run '
                'compared the requested thresholds as-is. The aligned-rows '
                'table, the per-density analysis sections and the alignment '
                'guides are absent for this reason, not by design.')
        else:
            head = (
                f'- [auto threshold] auto mode resolved PARTIALLY: the '
                f'{status.get("vertical_rows", 0)} installed '
                'vertical (same-threshold) row(s) run as queries, but the '
                'horizontal (density-matched) rows need every dataset\'s '
                'density curve and were not installed; any horizontal row in '
                'the aligned-rows table is advisory only.')
        lines = [head]
        if uncaptured:
            lines.append('  • no density capture: ' + ', '.join(uncaptured))
        for ds in sorted(failures):
            lines.append(f'  • {ds} bootstrap measurement failed: '
                         f'{failures[ds]}')
        if status.get('reason'):
            lines.append(f'  • reason: {status["reason"]}')
        lines.append('  • machine-readable copy: run_manifest.json → '
                     'auto_mode_status')
        if failures:
            lines.append(
                '  • the traceback is in '
                'dataset_data/<dataset>/run_log.txt')
        try:
            self._append_user_warning_notes(
                self.parameters.full_output_path, ['\n'.join(lines)])
        except Exception as e:
            self._log(f"Warning: could not append auto-mode notes: {e}")

    def _append_density_warnings(self, curves, windows, metas, aligned,
                                 normalizer, comparison_results_dir) -> None:
        """Append the auto-mode ``[auto threshold]`` / ``[density]`` notes."""
        order = list(self.parameters.get_dataset_names())
        blocks = []
        # Per-run summary + window table.
        lines = ['- [auto threshold] density-aligned mode: measured per-dataset '
                 'windows (Min Synapse Count) and the aligned rows:']
        for ds in order:
            meta = metas.get(ds)
            if meta is None:
                lines.append(f'  • {ds}: no density capture (excluded).')
                continue
            lo, hi = windows.get(ds, (None, None))
            n_typed = (meta.get('n_nodes_typed')
                       or meta.get('n_annotated_nodes')
                       or meta.get('n_nodes'))
            lines.append(
                f'  • {ds}: applied={meta.get("applied")}, '
                f'w_start={lo}, w_star_measured={hi}, '
                f'n_paths={meta.get("n_paths")}, n_edges={meta.get("n_edges")}, '
                f'N={n_typed} ({meta.get("denominator")}); untyped nodes='
                f'{meta.get("n_nodes_untyped")}, debris nodes='
                f'{meta.get("n_nodes_debris")}.')
        vert = [r for r in aligned if 'vertical' in (r.get('mode') or '')]
        horiz = [r for r in aligned if 'horizontal' in (r.get('mode') or '')]
        lines.append(f'  • vertical (same-threshold) rows: '
                     f'{len(vert)}; horizontal (same-density) rows: {len(horiz)}.')
        missing = [ds for ds in order if metas.get(ds) is None]
        if missing and horiz:
            lines.append(
                '  • [density partial] horizontal rows exclude '
                + ', '.join(missing)
                + ' (no density capture); they are computed over the '
                'captured datasets only.')
        if not vert:
            lines.append(
                '  • NOTE: no vertical row exists — the datasets\' measured '
                'windows do not all intersect (max w_start > min '
                'w_star_measured), so no threshold is complete for every '
                'dataset. Horizontal (same-density) rows carry the aligned '
                'comparison.')
        blocks.append('\n'.join(lines))

        for ds in order:
            meta = metas.get(ds)
            if meta is None:
                continue
            lo, hi = windows.get(ds, (None, None))
            if hi is not None and lo is not None and hi <= lo:
                blocks.append(
                    f'- [auto threshold] dataset {ds} has a single admissible '
                    f'threshold (w_star_measured == w_start == {lo}); its '
                    'aligned comparison degenerates to one point.')
            if meta.get('w_star_mismatch'):
                blocks.append(
                    f'- [auto threshold] stored strongest_retained_bottleneck='
                    f'{meta.get("strongest_retained_bottleneck")} differs from '
                    f'the measured retained ceiling {hi} for dataset {ds} — '
                    'provenance corrected/diagnostic (auto mode uses the '
                    'measured value).')
            if not meta.get('n_paths'):
                blocks.append(
                    f'- [auto threshold] dataset {ds} has 0 paths in the '
                    'aligned window (empty path curve; edge curve may still '
                    'be non-empty); consider the core subset of datasets.')
        blocks.append(
            f'- [density] y-axis is E(t)/N with a t-independent N '
            f'({normalizer} normalizer; N = typed nodes in the searched '
            'graph with Drop Untyped on, typed+untyped with it off; debris '
            'ids absent from the curated table are always excluded). x is '
            'the per-connection Min Synapse Count over the query\'s '
            'searched graph.')
        blocks.append('- [density] ' + (
            "Density alignment is computed on the query's bodyId searched "
            "graph; the compared matrices are type-level projections of the "
            "same search. A threshold here is a per-connection synapse count "
            "(Min Synapse Count)."))
        try:
            self._append_user_warning_notes(
                self.parameters.full_output_path, blocks)
        except Exception as e:
            self._log(f"Warning: could not append density notes: {e}")

    def _bootstrap_measure_floor(self, dataset: str, boot_t: int) -> None:
        """Enumerate one dataset's floor cone for the auto-mode bootstrap.

        Raises whatever the delegated run raised: the caller scopes the damage
        to this dataset rather than abandoning the whole measurement (plan
        §auto-mode-density-regression P3, 2026-09-21 — one dataset's failed
        cache bookkeeping used to discard every dataset's density capture).
        """
        if boot_t in self.raw_results.get(dataset, {}):
            return
        if self.parameters.output_folder:
            cached = self._try_load_cached(dataset, boot_t)
            if cached is not None:
                self.raw_results.setdefault(dataset, {})[boot_t] = \
                    self._finalize_loaded_result(dataset, boot_t, cached)
                return
        df = self.run_path_analysis(dataset, boot_t, verbose_mode='simple')
        self.raw_results.setdefault(dataset, {})[boot_t] = df
        if self.parameters.output_folder:
            self._save_result(dataset, boot_t, df)

    def _bootstrap_auto_mode(self) -> bool:
        """Measure per-dataset density, then install aligned combinations.

        Plan §5 Phase D. Runs ONE enumeration per dataset at the requested
        floor (``min(thresholds)``, defaulting to 3 when no threshold chips
        were given — auto mode never requires them) with density capture on,
        builds the curves/windows from the persisted artifacts, and installs the
        aligned rows as the run's combination schedule. A dataset that cannot
        be measured costs only itself: its absence narrows the vertical spine's
        window and drops the horizontal rows, and is reported through
        ``_auto_bootstrap_status``. Returns True when rows were installed;
        False leaves the caller to fall back to the requested thresholds.
        """
        from .threshold_density import align_horizontal, align_vertical
        datasets = self.parameters.get_dataset_names()
        base = [int(t) for t in (self.parameters.thresholds or [])] or [3]
        boot_t = max(1, min(base))
        self._log(f"Auto threshold mode: bootstrap enumeration at t={boot_t} "
                  f"for {len(datasets)} dataset(s)")
        saved_thresholds = list(self.parameters.thresholds)
        self.parameters.thresholds = [boot_t]

        def _status(outcome: str, *, vertical_rows: int = 0,
                    horizontal_rows: int = 0, uncaptured=(),
                    failures=None, reason: Optional[str] = None) -> Dict[str, Any]:
            """One key set for every outcome, so the manifest stays stable.

            ``reason`` explains a schedule that failed for a reason the
            per-dataset failures do not cover (no capture at all, no aligned
            row in range, a rejected install) and is None otherwise — present
            either way, because readers are told to find this block by key
            rather than by outcome.
            """
            return {
                'outcome': outcome,
                'floor_threshold': boot_t,
                'requested_thresholds': list(saved_thresholds),
                'vertical_rows': vertical_rows,
                'horizontal_rows': horizontal_rows,
                'uncaptured': [str(ds) for ds in (uncaptured or ())],
                'failures': dict(failures or {}),
                'reason': reason,
            }
        failures: Dict[str, str] = {}
        try:
            for ds in datasets:
                error = None
                for attempt in (1, 2):
                    try:
                        self._bootstrap_measure_floor(ds, boot_t)
                        error = None
                        break
                    except Exception as e:
                        import traceback
                        error = e
                        self._log(
                            f"Auto threshold bootstrap: {ds} measurement "
                            f"attempt {attempt} failed "
                            f"({type(e).__name__}: {e})", level='warn')
                        self._log(f"  {traceback.format_exc()}")
                if error is not None:
                    failures[str(ds)] = f'{type(error).__name__}: {error}'
        finally:
            self.parameters.thresholds = saved_thresholds

        curves, windows, metas = self._build_density_curves()
        if not curves:
            self._log("Auto threshold mode: no density capture produced — "
                      "falling back to the requested thresholds.")
            # Applied after the restore above: without the measured rows the
            # run still needs a schedule — default to the physical-minimum
            # floor when no chips were set (the finally would otherwise
            # overwrite this fallback with the empty list).
            self.parameters.thresholds = saved_thresholds or [3]
            self._auto_bootstrap_status = _status(
                'degraded', uncaptured=datasets, failures=failures,
                reason='no density capture was measured for any dataset')
            return False
        order = list(datasets)
        uncaptured = [str(ds) for ds in order if not curves.get(ds)]
        if uncaptured:
            self._log(
                "Auto threshold mode: no density capture for "
                + ', '.join(uncaptured) + " — installing the vertical "
                "(same-threshold) spine only; the density-matched horizontal "
                "rows need every dataset's curve.", level='warn')
        nick_by_ds = {}
        try:
            _nicks = self.parameters.get_dataset_nicknames()
            nick_by_ds = {ds: _nicks[i] for i, ds in enumerate(order)}
        except Exception:
            nick_by_ds = {ds: ds for ds in order}

        def _ds_txt(thresholds):
            return ', '.join(
                f"{nick_by_ds.get(ds, ds)} {int(thresholds[ds])}"
                for ds in order)

        rows = []
        # B7 query id/label scheme: verticals read exactly like the
        # per-threshold analysis they mirror (id+label `threshold={N}`);
        # horizontals self-identify as density-matched
        # (`aligned_density={level:.4g} (...)` with the explicit
        # per-dataset thresholds).  `.4g` collision guard: two distinct
        # levels could round to the same `.4g` string — ids are
        # de-duplicated deterministically (append `_2`, `_3`, ...).
        used_ids: Dict[str, int] = {}

        def _unique_id(base: str) -> str:
            count = used_ids.get(base, 0)
            used_ids[base] = count + 1
            return base if count == 0 else f"{base}_{count + 1}"

        for r in align_vertical(windows, K=5):
            v_id = _unique_id(f'threshold={r}')
            rows.append({'id': v_id, 'label': v_id,
                         'row_mode': 'vertical',
                         'thresholds': {ds: int(r) for ds in order}})
        # Density-matched rows need every dataset's curve: a row built without
        # one would silently omit a column while its label still promises the
        # full dataset set (and the schedule validator rejects a row missing a
        # dataset), so a partial measurement installs the vertical spine only.
        horizontal_rows = [] if uncaptured else list(
            align_horizontal(curves, levels=4))
        for i, r in enumerate(horizontal_rows, start=1):
            if len(r.get('thresholds', {})) != len(order):
                continue
            # Horizontal rows are per-dataset by construction: carry the
            # explicit per-dataset thresholds in the label itself.
            h_id = _unique_id(f"aligned_density={r['level_normalized']:.4g}")
            rows.append({
                'id': h_id,
                'label': (f'{h_id} '
                          f"({_ds_txt(r['thresholds'])})"),
                'row_mode': 'horizontal',
                'thresholds': r['thresholds']})
        # De-duplicate identical threshold rows, keeping the vertical label.
        seen, unique = set(), []
        for r in rows:
            key = tuple((ds, int(r['thresholds'][ds])) for ds in order)
            if key in seen:
                continue
            seen.add(key)
            unique.append(r)
        if not unique:
            self._log("Auto threshold mode: no aligned rows in range — "
                      "falling back to the requested thresholds.")
            self.parameters.thresholds = saved_thresholds or [3]
            self._auto_bootstrap_status = _status(
                'degraded', uncaptured=uncaptured, failures=failures,
                reason='the measured windows admit no aligned row')
            return False
        try:
            self.parameters.install_auto_combinations(unique)
        except Exception as e:
            import traceback
            self._log(f"Auto threshold mode: could not install rows: "
                      f"{type(e).__name__}: {e}")
            self._log(f"  {traceback.format_exc()}")
            self.parameters.thresholds = saved_thresholds or [3]
            self._auto_bootstrap_status = _status(
                'degraded', uncaptured=uncaptured, failures=failures,
                reason=f'could not install the aligned rows: '
                       f'{type(e).__name__}: {e}')
            return False
        self._auto_alignment_rows = unique
        self._auto_bootstrap_status = _status(
            'installed' if not uncaptured else 'verticals_only',
            vertical_rows=sum(
                1 for r in unique if 'vertical' in (r.get('row_mode') or '')),
            horizontal_rows=sum(
                1 for r in unique if 'horizontal' in (r.get('row_mode') or '')),
            uncaptured=uncaptured, failures=failures)
        self._log(f"Auto threshold mode: installed {len(unique)} aligned "
                  f"query row(s) across {len(order)} dataset(s)")
        return True

    def suggest_threshold_combinations(self, reference_dataset=None,
                                       tolerance=None) -> List[Dict[str, Any]]:
        """Density-equivalent combination rows from the last alignment run.

        Consumes ``_alignment_best_df`` (or the exported CSV when the run
        has been reloaded) and returns ``threshold_combinations``-shaped
        rows. ``reference_dataset`` defaults to the first dataset.
        """
        try:
            from .threshold_alignment import (
                ALIGNMENT_TOLERANCE, suggest_combination_rows,
            )
        except ImportError:  # pragma: no cover
            from threshold_alignment import (
                ALIGNMENT_TOLERANCE, suggest_combination_rows,
            )
        df = getattr(self, '_alignment_best_df', None)
        if df is None:
            path = os.path.join(self.parameters.full_output_path,
                                'comparison_results',
                                'threshold_alignment_best_matches.csv')
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                except Exception:
                    df = None
        if df is None:
            return []
        datasets = self.parameters.get_dataset_names()
        reference = reference_dataset or (datasets[0] if datasets else None)
        if reference is None:
            return []
        return suggest_combination_rows(
            df, reference, datasets,
            tolerance=(ALIGNMENT_TOLERANCE if tolerance is None
                       else float(tolerance)))

    def _suggest_and_export_combinations(self, best_df, dataset_order,
                                         comparison_results_dir) -> None:
        try:
            from .threshold_alignment import (
                ALIGNMENT_TOLERANCE, suggest_combination_rows,
            )
        except ImportError:  # pragma: no cover
            from threshold_alignment import (
                ALIGNMENT_TOLERANCE, suggest_combination_rows,
            )
        datasets = [d for d in self.parameters.get_dataset_names()]
        reference = datasets[0] if datasets else None
        if reference is None:
            return
        rows = suggest_combination_rows(
            best_df, reference, datasets, tolerance=ALIGNMENT_TOLERANCE)
        if not rows:
            return
        self._suggested_combinations = rows
        try:
            import json as _json
            out_path = os.path.join(comparison_results_dir,
                                    'suggested_threshold_combinations.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                _json.dump({
                    'reference_dataset': reference,
                    'tolerance': ALIGNMENT_TOLERANCE,
                    'combinations': rows,
                }, f, indent=2, default=str)
            self._log(
                f"Saved: suggested_threshold_combinations.json "
                f"({len(rows)} aligned query row(s))")
        except Exception as e:
            self._log(f"Warning: could not write suggested combinations: {e}")

    def _export_reciprocal_comparisons(self, comparison_results_dir: str) -> None:
        """Export reciprocal connectivity as its own comparison artifact.

        Mirrors the single-dataset pathfinding treatment (plan §9 / §12 item
        11): the reciprocal aligned matrix, its cross-dataset conservation,
        and (when available) the directed-vs-reciprocal motif counts are
        written under ``comparison_results/reciprocal/`` rather than being a
        transient network-only frame.
        """
        if not getattr(self.parameters, 'find_reciprocal', False):
            return
        dataset_names = self.parameters.get_dataset_names()
        out_dir = os.path.join(comparison_results_dir, 'reciprocal')
        os.makedirs(out_dir, exist_ok=True)

        thresholds = self._analysis_thresholds()
        wrote_any = False
        for t in thresholds:
            try:
                aligned = self.get_aligned_data_for_network(t)
            except Exception as e:
                self._log(f"  Warning: reciprocal alignment t={t} failed: {e}")
                continue
            if aligned is None or aligned.empty:
                continue
            safe_t = self._safe_query_filename_id(t)
            self._save_csv(
                aligned, os.path.join(out_dir, f'reciprocal_aligned_t{safe_t}.csv'))
            # Conservation = present at positive weight in every dataset.
            available = [d for d in dataset_names if d in aligned.columns]
            if available:
                mask_all = (aligned[available] > 0).all(axis=1)
                conserved = aligned[mask_all].copy()
                if not conserved.empty:
                    conserved.to_csv(
                        os.path.join(
                            out_dir, f'reciprocal_conserved_t{safe_t}.csv'))
            wrote_any = True
        if wrote_any:
            self._log("Saved: comparison_results/reciprocal/ "
                      "(reciprocal aligned matrices + conserved edges)")

    def _export_threshold_alignment_heatmap(self, vis_dir: str):
        """Render the typed-grid alignment distance heatmap (§6.4)."""
        matrix_df = getattr(self, '_alignment_matrix_df', None)
        grid_points = getattr(self, '_alignment_grid_points', None)
        if matrix_df is None or matrix_df.empty or not grid_points:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np

            labels = [f"{ds}@{t}" for ds, t in grid_points]
            index = {gp: i for i, gp in enumerate(grid_points)}
            n = len(grid_points)
            dist = np.full((n, n), np.nan)
            for _, row in matrix_df.iterrows():
                i = index[(row['dataset_a'], row['threshold_a'])]
                j = index[(row['dataset_b'], row['threshold_b'])]
                dist[i, j] = dist[j, i] = row['edge_count_distance']
            np.fill_diagonal(dist, 0.0)

            fig, ax = plt.subplots(figsize=(1.1 * n + 2.5, 0.95 * n + 2.0))
            masked = np.ma.masked_invalid(dist)
            im = ax.imshow(masked, cmap='RdYlGn_r', vmin=0, vmax=1)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
            for i in range(n):
                for j in range(n):
                    if not np.isnan(dist[i, j]):
                        ax.text(j, i, f"{dist[i, j]:.2f}",
                                ha='center', va='center', fontsize=7)
            ax.set_title('Threshold alignment: edge-count distance\n'
                         '(typed thresholds; 0 = same connection-pair density)')
            fig.colorbar(im, ax=ax, shrink=0.8, label='edge-count distance')
            fig.tight_layout()
            out = os.path.join(vis_dir, "threshold_alignment_matrix.png")
            fig.savefig(out, dpi=200, bbox_inches='tight')
            plt.close(fig)
            self._log_file(out, "Threshold alignment matrix heatmap")
        except Exception as e:
            self._log(f"Warning: alignment heatmap failed: {e}")
    
    def _export_top_edges_comparison(self, comparison_results_dir: str):
        """Export report tables capped by the top_edges parameter.

        This report-row cap is independent of graph discovery and the
        per-visualization drawn-edge limit.
        """
        dataset_names = self.parameters.get_dataset_names()
        top_n = self.parameters.top_edges
        
        # Use the middle threshold for top edges analysis
        mid_threshold = self.parameters.thresholds[len(self.parameters.thresholds) // 2]
        aligned = self.get_aligned_data(mid_threshold)
        
        if aligned.empty:
            return
        
        # Get top edges per dataset
        top_edges = self.metrics.get_top_edges_per_dataset(aligned, dataset_names, top_n)
        if not top_edges.empty:
            self._save_csv(top_edges, os.path.join(comparison_results_dir, "top_edges_comparison.csv"))
            self._log("Saved: top_edges_comparison.csv")
        
        # Get overlap statistics
        overlap = self.metrics.compare_top_edges_overlap(aligned, dataset_names, top_n)
        if not overlap.empty:
            self._save_csv(overlap, os.path.join(comparison_results_dir, "top_edges_overlap.csv"))
            self._log("Saved: top_edges_overlap.csv")
    
    def _export_degree_distribution(self, comparison_results_dir: str):
        """
        Export degree distribution analysis to CSV.
        
        To-Do List 5 Item 5: Renamed files to degree_out and degree_in, 
        added threshold indication and merged all thresholds into unified files.
        """
        dataset_names = self.parameters.get_dataset_names()
        
        # Collect degree data for ALL thresholds (To-Do List 5 Item 5)
        all_out_degree = []
        all_in_degree = []
        
        for threshold in self.parameters.thresholds:
            degree_data = self.metrics.calculate_degree_distribution(
                self.raw_results, dataset_names, threshold
            )
            
            # Add threshold column to each row
            out_degree = degree_data.get('out_degree', pd.DataFrame())
            if not out_degree.empty:
                out_degree = out_degree.copy()
                out_degree['threshold'] = threshold
                all_out_degree.append(out_degree)
            
            in_degree = degree_data.get('in_degree', pd.DataFrame())
            if not in_degree.empty:
                in_degree = in_degree.copy()
                in_degree['threshold'] = threshold
                all_in_degree.append(in_degree)
        
        # Save unified out-degree data (renamed from out_degree_distribution.csv)
        if all_out_degree:
            unified_out = pd.concat(all_out_degree, ignore_index=True)
            # Reorder columns: threshold first
            cols = ['threshold'] + [c for c in unified_out.columns if c != 'threshold']
            unified_out = unified_out[cols]
            self._save_csv(unified_out, os.path.join(comparison_results_dir, "degree_out.csv"))
            self._log("Saved: degree_out.csv (unified across thresholds)")
        
        # Save unified in-degree data (renamed from in_degree_distribution.csv)
        if all_in_degree:
            unified_in = pd.concat(all_in_degree, ignore_index=True)
            # Reorder columns: threshold first
            cols = ['threshold'] + [c for c in unified_in.columns if c != 'threshold']
            unified_in = unified_in[cols]
            self._save_csv(unified_in, os.path.join(comparison_results_dir, "degree_in.csv"))
            self._log("Saved: degree_in.csv (unified across thresholds)")
        
        # Save degree statistics summary for ALL thresholds
        all_degree_stats = []
        
        for threshold in self.parameters.thresholds:
            degree_data = self.metrics.calculate_degree_distribution(
                self.raw_results, dataset_names, threshold
            )
            
            if degree_data.get('out_degree', pd.DataFrame()).empty and degree_data.get('in_degree', pd.DataFrame()).empty:
                continue
            
            degree_stats = self.metrics.calculate_degree_statistics(degree_data)
            if not degree_stats.empty:
                degree_stats['threshold'] = threshold
                all_degree_stats.append(degree_stats)
        
        if all_degree_stats:
            unified_stats = pd.concat(all_degree_stats, ignore_index=True)
            # Reorder columns: threshold first
            cols = ['threshold'] + [c for c in unified_stats.columns if c != 'threshold']
            unified_stats = unified_stats[cols]
            self._save_csv(unified_stats, os.path.join(comparison_results_dir, "degree_statistics.csv"))
            self._log("Saved: degree_statistics.csv (all thresholds)")
    
    def _export_metadata_comparison(self, comparison_results_dir: str):
        """Export dataset metadata comparison to CSV."""
        try:
            # Collect metadata (uses cache if available)
            self.collect_dataset_metadata(force_refresh=False)
            
            # Generate comparison table
            metadata_df = self.generate_metadata_comparison_table()
            if not metadata_df.empty:
                # Save to comparison_results folder
                self._save_csv(metadata_df, os.path.join(comparison_results_dir, "dataset_metadata_comparison.csv"))
                self._log("Saved: dataset_metadata_comparison.csv")
                
                # Also save to the main output folder
                out_dir = os.path.dirname(comparison_results_dir)
                self._save_csv(metadata_df, os.path.join(out_dir, "dataset_metadata_comparison.csv"))
        except Exception as e:
            self._log(f"Warning: Failed to export metadata comparison: {e}")
    
    def _export_unified_summary(self, comparison_results_dir: str):
        """
        Export unified summary CSVs that merge data across all thresholds.
        
        Creates:
        1. unified_edge_comparison.csv - All edges across all thresholds with presence/weights
        2. unified_summary.csv - High-level summary per dataset per threshold
        
        This reduces the number of output files and provides a comprehensive view.
        """
        dataset_names = self.parameters.get_dataset_names()
        
        # 1. Unified edge comparison across all thresholds
        unified_edges = []
        
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            
            available = [d for d in dataset_names if d in aligned.columns]
            
            for edge_key, row in aligned.iterrows():
                if ' -> ' in str(edge_key):
                    parts = str(edge_key).split(' -> ')
                    source = parts[0]
                    target = parts[1] if len(parts) > 1 else ''
                else:
                    source = str(edge_key)
                    target = ''
                # N5: label untyped bodyId fallbacks
                if isinstance(source, str) and source.isdigit():
                    source = f'bodyId:{source} (untyped)'
                if isinstance(target, str) and target.isdigit():
                    target = f'bodyId:{target} (untyped)'
                
                edge_data = {
                    'edge_key': edge_key,
                    'source': source,
                    'target': target,
                    'threshold': threshold,
                }
                
                # Add weights per dataset
                conservation_count = 0
                for dataset in available:
                    safe_name = self.parameters._sanitize_name(dataset)
                    weight = row[dataset] if dataset in row else 0
                    edge_data[f'{safe_name}_weight'] = weight
                    # N3: consistent 1/0 presence (mixed True/0 broke CSV typing)
                    edge_data[f'{safe_name}_present'] = int(weight > 0)
                    if weight > 0:
                        conservation_count += 1
                
                edge_data['conservation'] = f"{conservation_count}/{len(available)}"
                unified_edges.append(edge_data)
        
        if unified_edges:
            unified_df = pd.DataFrame(unified_edges)
            self._save_csv(unified_df, os.path.join(comparison_results_dir, "unified_edge_comparison.csv"))
            self._log(f"Saved: unified_edge_comparison.csv ({len(unified_df)} entries)")
        
        # 2. Unified summary per dataset per threshold
        summary_data = []
        
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            
            for threshold in self.parameters.thresholds:
                df = self.raw_results.get(dataset, {}).get(threshold, pd.DataFrame())
                # Full requested/applied/budget state on every row.  Keep
                # the threshold alias for compatibility with older readers.
                applied_state = self._path_provenance_row(dataset, threshold)
                
                if df.empty:
                    summary_data.append({
                        **applied_state,
                        'dataset': safe_name,
                        'threshold': threshold,
                        'total_edges': 0,
                        'total_weight': 0,
                        'mean_weight': 0,
                        'unique_sources': 0,
                        'unique_targets': 0,
                    })
                    continue
                
                # Extract source/target columns
                if 'type_pre' in df.columns and 'type_post' in df.columns:
                    sources = df['type_pre'].nunique()
                    targets = df['type_post'].nunique()
                elif 'source' in df.columns and 'target' in df.columns:
                    sources = df['source'].nunique()
                    targets = df['target'].nunique()
                else:
                    sources = 0
                    targets = 0
                
                weight_col = 'weight' if 'weight' in df.columns else None
                total_weight = df[weight_col].sum() if weight_col else len(df)
                mean_weight = df[weight_col].mean() if weight_col else 1
                
                summary_data.append({
                    **applied_state,
                    'dataset': safe_name,
                    'threshold': threshold,
                    # N6: unique (source, target) pairs — conn rows keep
                    # per-layer occurrences of the same pair.
                    'total_edges': int(df[['type_pre', 'type_post']].drop_duplicates().shape[0])
                        if 'type_pre' in df.columns and 'type_post' in df.columns
                        else len(df),
                    'total_layer_rows': len(df),
                    'total_weight': round(total_weight, 2),
                    'mean_weight': round(mean_weight, 2),
                    'unique_sources': sources,
                    'unique_targets': targets,
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            self._save_csv(summary_df, os.path.join(comparison_results_dir, "unified_summary.csv"))
            self._log(f"Saved: unified_summary.csv ({len(summary_df)} entries)")
        
        # 3. Export unified presence matrix with all thresholds as columns
        self._export_unified_presence_matrix(comparison_results_dir)
        
        # 4. Export merged unique connections per dataset
        self._export_merged_unique_connections(comparison_results_dir)
        
        # 5. To-Do List 5 Item 1: Export unified path presence matrix with all thresholds
        self._export_unified_path_presence_matrix(comparison_results_dir)

    def _export_unified_presence_matrix(self, comparison_results_dir: str):
        """
        Export a unified edge presence matrix with all thresholds expanded horizontally.
        
        Creates edge_presence_matrix.csv with columns:
        - edge_key, source, target
        - For each dataset and threshold: presence marker (✔️/❌)
        - For each dataset and threshold: weight
        - Conservation summary
        
        This provides a single-file view of how edges are affected by threshold changes.
        """
        dataset_names = self.parameters.get_dataset_names()
        thresholds = self.parameters.thresholds
        
        # Collect all unique edges across all thresholds
        all_edges = {}
        
        for threshold in thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            
            available = [d for d in dataset_names if d in aligned.columns]
            
            for edge_key, row in aligned.iterrows():
                if edge_key not in all_edges:
                    # Parse source/target from edge key
                    if ' -> ' in str(edge_key):
                        parts = str(edge_key).split(' -> ')
                        source = parts[0]
                        target = parts[1] if len(parts) > 1 else ''
                    else:
                        source = str(edge_key)
                        target = ''
                    
                    all_edges[edge_key] = {
                        'edge_key': edge_key,
                        'source': source,
                        'target': target,
                    }
                
                # Add threshold-specific presence and weight for each dataset
                for dataset in available:
                    safe_name = self.parameters._sanitize_name(dataset)
                    weight = row[dataset] if dataset in row else 0
                    
                    # Presence marker: N3 — consistent 1/0 (mixed True/0
                    # produced unparseable CSV columns)
                    pres_col = f'{safe_name}_t{threshold}'
                    all_edges[edge_key][pres_col] = int(weight > 0)
                    
                    # Weight column: weight_dataset_threshold
                    weight_col = f'w_{safe_name}_t{threshold}'
                    all_edges[edge_key][weight_col] = weight if weight > 0 else ''
        
        if not all_edges:
            return
        
        # Build DataFrame
        rows = list(all_edges.values())
        presence_df = pd.DataFrame(rows)
        
        # Add summary columns - count thresholds where edge is present per dataset
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            # Count how many thresholds this edge appears in for this dataset
            presence_cols = [f'{safe_name}_t{t}' for t in thresholds if f'{safe_name}_t{t}' in presence_df.columns]
            if presence_cols:
                presence_df[f'{safe_name}_count'] = presence_df[presence_cols].apply(
                    lambda x: sum(1 for v in x if v == True), axis=1
                )
        
        # Add total conservation count (edge present in any dataset at lowest threshold)
        lowest_threshold = min(thresholds)
        available = [d for d in dataset_names if f'{self.parameters._sanitize_name(d)}_t{lowest_threshold}' in presence_df.columns]
        presence_cols = [f'{self.parameters._sanitize_name(d)}_t{lowest_threshold}' for d in available]
        if presence_cols:
            presence_df['conserved_at_lowest'] = presence_df[presence_cols].apply(
                lambda x: sum(1 for v in x if v == True), axis=1
            )

        # F6: asked vs applied — constant per-(dataset, threshold) columns
        # state the real cutoff (tau-raised or Edge-Budget floored). The
        # presence columns stay keyed by the ASKED threshold on purpose:
        # renaming them would break downstream name reconstruction (the
        # report key-findings build f'{safe}_t{t}' directly).
        applied_cols = []
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            for t in thresholds:
                applied, pruned, floor, _applied_source = \
                    self._applied_state_for(dataset, t)
                col = f'applied_{safe_name}_t{t}'
                pruned_col = f'pruned_{safe_name}_t{t}'
                if col not in presence_df.columns:
                    presence_df[col] = int(applied) if applied is not None else ''
                    presence_df[pruned_col] = int(bool(pruned))
                    applied_cols.extend([col, pruned_col])
        
        # Reorder columns: edge info first, then by threshold
        col_order = ['edge_key', 'source', 'target']
        
        # Add presence columns grouped by threshold
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                pres_col = f'{safe_name}_t{threshold}'
                if pres_col in presence_df.columns:
                    col_order.append(pres_col)
        
        # Add weight columns grouped by threshold
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                weight_col = f'w_{safe_name}_t{threshold}'
                if weight_col in presence_df.columns:
                    col_order.append(weight_col)
        
        # Add summary columns
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            count_col = f'{safe_name}_count'
            if count_col in presence_df.columns:
                col_order.append(count_col)
        
        if 'conserved_at_lowest' in presence_df.columns:
            col_order.append('conserved_at_lowest')
        # F6: applied/pruned state columns trail the per-threshold block.
        col_order.extend([c for c in applied_cols if c in presence_df.columns])
        
        # Filter and reorder
        col_order = [c for c in col_order if c in presence_df.columns]
        presence_df = presence_df[col_order]
        
        # Sort by conservation at lowest threshold, then alphabetically
        if 'conserved_at_lowest' in presence_df.columns:
            presence_df = presence_df.sort_values(['conserved_at_lowest', 'edge_key'], ascending=[False, True])
        
        # Save
        self._save_csv(presence_df, os.path.join(comparison_results_dir, "edge_presence_matrix.csv"))
        self._log(f"Saved: edge_presence_matrix.csv (unified, {len(presence_df)} edges, {len(thresholds)} thresholds)")

    def _export_merged_unique_connections(self, comparison_results_dir: str):
        """
        Export merged unique connections - one file per dataset with all thresholds.
        
        Instead of separate unique_to_{dataset}_minsyn_{threshold}.csv files,
        creates a single unique_to_{dataset}.csv with a threshold column.
        
        Columns include:
        - edge_key, source, target
        - threshold (synapse cutoff level)
        - weight
        - Additional context columns
        """
        dataset_names = self.parameters.get_dataset_names()
        
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            merged_unique = []
            
            for threshold in self.parameters.thresholds:
                unique = self.get_unique_connections(threshold)
                if safe_name not in unique and dataset not in unique:
                    continue
                
                # Get unique_df - check safe_name first, then dataset name
                # Cannot use `or` with DataFrames due to ambiguity
                if safe_name in unique:
                    unique_df = unique[safe_name]
                elif dataset in unique:
                    unique_df = unique[dataset]
                else:
                    unique_df = pd.DataFrame()
                
                if unique_df.empty:
                    continue
                
                for _, row in unique_df.iterrows():
                    # Get edge info from row or index
                    edge_key = row.get('edge_key', row.name if hasattr(row, 'name') else '')
                    if not edge_key:
                        # Try to construct from source/target
                        source = row.get('source_type', row.get('source', row.get('type_pre', '')))
                        target = row.get('target_type', row.get('target', row.get('type_post', '')))
                        edge_key = f"{source} -> {target}"
                    else:
                        # Parse from edge_key
                        if ' -> ' in str(edge_key):
                            parts = str(edge_key).split(' -> ')
                            source = parts[0]
                            target = parts[1] if len(parts) > 1 else ''
                        else:
                            source = str(edge_key)
                            target = ''
                    
                    weight = row.get('weight', row.get(safe_name, row.get(dataset, 0)))
                    
                    merged_unique.append({
                        'edge_key': edge_key,
                        'source': source,
                        'target': target,
                        'threshold': threshold,
                        'weight': weight,
                        'unique_to': safe_name,
                    })
            
            if merged_unique:
                merged_df = pd.DataFrame(merged_unique)
                merged_df = merged_df.sort_values(['threshold', 'weight'], ascending=[True, False])
                self._save_csv(merged_df, os.path.join(comparison_results_dir, f"unique_to_{safe_name}.csv"))
                self._log(f"Saved: unique_to_{safe_name}.csv ({len(merged_df)} unique edges)")

    def _find_any_applied_output_folder(self, dataset: str) -> Optional[str]:
        """Smallest applied ``minsyn_*`` folder for a dataset, from disk.

        Fallback for neuron-count export when the resolved applied folder
        is missing (legacy runs, renamed folders). Reads each folder's own
        ``all_attributes.json`` ``applied_threshold`` — never the folder
        name — and returns the folder with the smallest applied value.
        """
        base = os.path.join(
            self.parameters.full_output_path, 'dataset_data',
            self.parameters._sanitize_name(dataset))
        if not os.path.isdir(base):
            return None
        candidates = []
        for name in os.listdir(base):
            if not name.startswith('minsyn_'):
                continue
            folder = os.path.join(base, name)
            if not os.path.isdir(folder):
                continue
            applied = None
            attrs_path = os.path.join(folder, 'all_attributes.json')
            if os.path.exists(attrs_path):
                try:
                    with open(attrs_path, encoding='utf-8') as f:
                        attrs = json.load(f)
                    applied = attrs.get('applied_threshold')
                    if applied is None:
                        applied = attrs.get('min_synapse_num')
                except Exception:
                    applied = None
            if applied is None:
                # Last resort: the folder's numeric suffix.
                suffix = name[len('minsyn_'):].split('_')[0]
                applied = int(suffix) if suffix.isdigit() else None
            if applied is not None:
                candidates.append((int(applied), folder))
        if not candidates:
            return None
        return min(candidates)[1]

    def _export_neuron_counts_comparison(self, comparison_results_dir: str):
        """
        Export source/target neuron counts comparison across datasets.

        Creates:
        1. neuron_counts_summary.csv - Total counts per dataset (source/target)
        2. neuron_counts_by_type.csv - Count per neuron type per dataset
           (keyed by merge-policy group label when a query-anchored merge
           policy governs the run, so counts agree with the merged
           presence/similarity frames)
        3. neuron_counts_by_group.csv - Count per custom group per dataset (if custom groups exist)

        Data is loaded from source_neurons.csv and target_neurons.csv saved by FindAllPath.
        """
        dataset_names = self.parameters.get_dataset_names()
        merge_policy = self._merge_policy_or_none()

        def _count_key(dataset: str, type_val) -> str:
            """Group label for a raw type when the merge policy governs it,
            so neuron counts agree with the merged presence/similarity
            frames (plan acceptance: counts agree on group membership)."""
            raw = str(type_val)
            if merge_policy is not None:
                label = merge_policy.key_for(dataset, raw)
                if label:
                    return label
            return raw

        def _track_member(type_counts: Dict, count_key: str, dataset: str,
                          type_val, count: int) -> None:
            """Remember the per-(dataset, raw type) composition behind a
            merged group row, exported as the ``group_members`` column."""
            members = type_counts[count_key].setdefault('_members', {})
            per_dataset = members.setdefault(str(dataset), {})
            name = str(type_val)
            per_dataset[name] = per_dataset.get(name, 0) + int(count)

        # Collect neuron data from each dataset
        all_source_data = []
        all_target_data = []
        summary_data = []
        type_counts = {}  # type -> {dataset: count}
        group_counts = {}  # group -> {dataset: count}

        def _count_hemisphere(df: pd.DataFrame) -> Dict[str, int]:
            if df is None or df.empty:
                return {'L': 0, 'R': 0, 'U': 0}
            hemi_col = None
            for col in ['hemisphere', 'hemisphere_code', 'hemisphere_label', 'Soma side', 'soma_side']:
                if col in df.columns:
                    hemi_col = col
                    break
            counts = {'L': 0, 'R': 0, 'U': 0}
            if hemi_col:
                vals = df[hemi_col].astype(str).str.strip().str.upper()
                counts['L'] = int((vals == 'L').sum())
                counts['R'] = int((vals == 'R').sum())
                counts['U'] = int((~vals.isin(['L', 'R'])).sum())
                return counts
            if 'type' in df.columns:
                types = df['type'].astype(str)
                counts['L'] = int(types.str.endswith('_L').sum())
                counts['R'] = int(types.str.endswith('_R').sum())
                counts['U'] = int(types.str.endswith('_U').sum())
            return counts
        
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            
            # Load from the lowest threshold output for this dataset
            # (source/target neurons are normally the same across thresholds).
            # In combination mode ``parameters.thresholds`` is the global
            # union of all query cells.  Using its minimum here can point at a
            # directory that was never materialized for this dataset (for
            # example, looking for male-cns/minsyn_3 when its query cell is
            # minsyn_10), which silently turns valid counts into zeros.
            get_dataset_thresholds = getattr(
                self.parameters, 'get_thresholds_for_dataset', None)
            dataset_thresholds = (
                get_dataset_thresholds(dataset)
                if callable(get_dataset_thresholds) else [])
            if dataset_thresholds:
                requested_min = min(dataset_thresholds)
            elif self.parameters.thresholds:
                # Compatibility fallback for older parameter objects that
                # do not expose a per-dataset schedule.
                requested_min = min(self.parameters.thresholds)
            else:
                requested_min = None

            # Resolve the MATERIALIZED applied folder, not the requested
            # minimum: Feature-G tau-collapse means the lowest requested
            # threshold usually has no folder of its own (its data lives in
            # the applied folder), which silently produced all-zero counts.
            applied_threshold = None
            if requested_min is not None:
                applied_threshold = self.get_applied_folder(
                    dataset, requested_min)
            if applied_threshold is None:
                applied_list = self.get_applied_thresholds(dataset)
                applied_threshold = applied_list[0] if applied_list else None

            # N1: FNC writes source/target_neurons.csv at the minsyn folder
            # ROOT; older runs had them under data_details/ — try both.
            # Resolve disk-aware so the new grammar folder names are found.
            dataset_output_path = (
                self._resolve_dataset_output_path(dataset, requested_min)
                if requested_min is not None else ''
            )
            source_candidates = [
                os.path.join(dataset_output_path, 'source_neurons.csv'),
                os.path.join(dataset_output_path, 'data_details', 'source_neurons.csv'),
            ]
            target_candidates = [
                os.path.join(dataset_output_path, 'target_neurons.csv'),
                os.path.join(dataset_output_path, 'data_details', 'target_neurons.csv'),
            ]
            source_file = next((p for p in source_candidates if os.path.exists(p)), source_candidates[0])
            target_file = next((p for p in target_candidates if os.path.exists(p)), target_candidates[0])
            if not os.path.exists(source_file) and not os.path.exists(
                    target_file):
                # Fallback scan: any existing applied folder (smallest
                # applied provenance) rather than defaulting to zero.
                fallback = self._find_any_applied_output_folder(dataset)
                if fallback:
                    dataset_output_path = fallback
                    source_file = os.path.join(fallback, 'source_neurons.csv')
                    target_file = os.path.join(fallback, 'target_neurons.csv')
                    self._log(
                        f"  Neuron counts: resolved {dataset} via fallback "
                        f"folder {os.path.basename(fallback)}")
                else:
                    self._log(
                        f"  Warning: no source/target neuron files found for "
                        f"{dataset} (requested_min={requested_min}, "
                        f"applied={applied_threshold}); counts will be 0.")
            
            source_count = 0
            target_count = 0
            source_df = None
            target_df = None
            source_hemi = {'L': 0, 'R': 0, 'U': 0}
            target_hemi = {'L': 0, 'R': 0, 'U': 0}
            
            # Load source neurons
            if os.path.exists(source_file):
                try:
                    source_df = self._read_csv(source_file, dtype={'bodyId': str})
                    source_count = len(source_df)
                    source_hemi = _count_hemisphere(source_df)
                    
                    # Count by type
                    if 'type' in source_df.columns:
                        for type_val in source_df['type'].dropna().unique():
                            type_cnt = len(source_df[source_df['type'] == type_val])
                            count_key = _count_key(dataset, type_val)
                            if count_key not in type_counts:
                                type_counts[count_key] = {'role': 'source'}
                            count_col = f'{safe_name}_source'
                            type_counts[count_key][count_col] = (
                                type_counts[count_key].get(count_col, 0)
                                + type_cnt)
                            if merge_policy is not None:
                                _track_member(type_counts, count_key,
                                              dataset, type_val, type_cnt)
                    
                    # Count by custom group
                    if 'custom_group' in source_df.columns:
                        for group_val in source_df['custom_group'].dropna().unique():
                            group_cnt = len(source_df[source_df['custom_group'] == group_val])
                            if group_val not in group_counts:
                                group_counts[group_val] = {'role': 'source'}
                            group_counts[group_val][f'{safe_name}_source'] = group_cnt
                            
                except Exception as e:
                    self._log(f"Warning: Could not load source neurons for {dataset}: {e}")
            
            # Load target neurons
            if os.path.exists(target_file):
                try:
                    target_df = self._read_csv(target_file, dtype={'bodyId': str})
                    target_count = len(target_df)
                    target_hemi = _count_hemisphere(target_df)
                    
                    # Count by type
                    if 'type' in target_df.columns:
                        for type_val in target_df['type'].dropna().unique():
                            type_cnt = len(target_df[target_df['type'] == type_val])
                            count_key = _count_key(dataset, type_val)
                            if count_key not in type_counts:
                                type_counts[count_key] = {'role': 'target'}
                            count_col = f'{safe_name}_target'
                            type_counts[count_key][count_col] = (
                                type_counts[count_key].get(count_col, 0)
                                + type_cnt)
                            if merge_policy is not None:
                                _track_member(type_counts, count_key,
                                              dataset, type_val, type_cnt)
                    
                    # Count by custom group  
                    if 'custom_group' in target_df.columns:
                        for group_val in target_df['custom_group'].dropna().unique():
                            group_cnt = len(target_df[target_df['custom_group'] == group_val])
                            if group_val not in group_counts:
                                group_counts[group_val] = {'role': 'target'}
                            group_counts[group_val][f'{safe_name}_target'] = group_cnt
                            
                except Exception as e:
                    self._log(f"Warning: Could not load target neurons for {dataset}: {e}")
            
            # Summary row
            summary_data.append({
                'dataset': safe_name,
                'source_count': source_count,
                'target_count': target_count,
                'total_neurons': source_count + target_count,
                'source_L': source_hemi['L'],
                'source_R': source_hemi['R'],
                'source_U': source_hemi['U'],
                'target_L': target_hemi['L'],
                'target_R': target_hemi['R'],
                'target_U': target_hemi['U'],
                'total_L': source_hemi['L'] + target_hemi['L'],
                'total_R': source_hemi['R'] + target_hemi['R'],
                'total_U': source_hemi['U'] + target_hemi['U'],
                'source_types': source_df['type'].nunique() if source_df is not None and 'type' in source_df.columns else 0,
                'target_types': target_df['type'].nunique() if target_df is not None and 'type' in target_df.columns else 0,
            })
        
        # Save summary CSV
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            self._save_csv(summary_df, os.path.join(comparison_results_dir, "neuron_counts_summary.csv"))
            self._log(f"Saved: neuron_counts_summary.csv ({len(summary_df)} datasets)")
        
        # Save type counts CSV (presence matrix style)
        if type_counts:
            type_rows = []
            for type_val, counts in type_counts.items():
                counts = dict(counts)
                members = counts.pop('_members', None)
                row = {'type': type_val}
                row.update(counts)
                if members:
                    # Only MERGED groups carry a composition note — a group
                    # whose every dataset shows the same single raw name is
                    # not merged, and the breakdown would be noise.
                    distinct_names = {
                        name
                        for per_dataset in members.values()
                        for name in per_dataset
                    }
                    if len(distinct_names) > 1:
                        row['group_members'] = '; '.join(
                            f"{ds}: " + ', '.join(
                                f'{member_name}({cnt})'
                                for member_name, cnt in sorted(per_ds.items()))
                            for ds, per_ds in sorted(members.items()))
                type_rows.append(row)
            
            type_df = pd.DataFrame(type_rows)
            if 'group_members' in type_df.columns:
                # Keep the composition note as the LAST column, after the
                # per-dataset count columns.
                type_df = type_df[[c for c in type_df.columns
                                   if c != 'group_members']
                                  + ['group_members']]
            # Sort by type name
            type_df = type_df.sort_values('type')
            self._save_csv(type_df, os.path.join(comparison_results_dir, "neuron_counts_by_type.csv"))
            self._log(f"Saved: neuron_counts_by_type.csv ({len(type_df)} types)")
            
            # Store for HTML report
            self._neuron_type_counts = type_df
        
        # Save group counts CSV (if any groups exist)
        if group_counts:
            group_rows = []
            for group_val, counts in group_counts.items():
                row = {'custom_group': group_val}
                row.update(counts)
                group_rows.append(row)
            
            group_df = pd.DataFrame(group_rows)
            group_df = group_df.sort_values('custom_group')
            self._save_csv(group_df, os.path.join(comparison_results_dir, "neuron_counts_by_group.csv"))
            self._log(f"Saved: neuron_counts_by_group.csv ({len(group_df)} groups)")
            
            # Store for HTML report
            self._neuron_group_counts = group_df
        
        # Store summary for HTML report
        self._neuron_counts_summary = pd.DataFrame(summary_data) if summary_data else pd.DataFrame()

    def _export_unified_path_presence_matrix(self, comparison_results_dir: str):
        """
        To-Do List 5 Item 1: Export unified path presence matrix with all thresholds horizontally.
        
        Similar to edge_presence_matrix.csv but for multi-hop paths, with columns:
        - path_key, source, target, hops, intermediates
        - For each dataset and threshold: presence marker (✔️/❌)
        - For each dataset and threshold: weight
        - For each dataset: hop weights as [w1, w2, ...]
        - Conservation summary
        """
        import ast
        
        dataset_names = self.parameters.get_dataset_names()
        thresholds = self.parameters.thresholds
        
        # Collect all paths across all thresholds
        all_paths = {}  # path_key -> {base_info, threshold_data}
        
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                
                # Find path data file
                dataset_output_path = self.parameters.get_dataset_output_path(dataset, threshold)
                
                path_files_to_try = [
                    os.path.join(dataset_output_path, f"minsyn_{threshold}_data_original_paths.csv"),
                ]
                
                if os.path.exists(dataset_output_path):
                    for f in os.listdir(dataset_output_path):
                        if f.endswith('_allpaths_type.csv'):
                            path_files_to_try.append(os.path.join(dataset_output_path, f))
                
                path_df = None
                for path_file in path_files_to_try:
                    if os.path.exists(path_file):
                        try:
                            path_df = self._read_csv(path_file)
                            break
                        except Exception:
                            continue
                
                if path_df is None or path_df.empty:
                    continue
                
                # Determine format: path column vs source/target columns
                path_col = 'path' if 'path' in path_df.columns else 'path_str' if 'path_str' in path_df.columns else None
                has_source_target_format = 'source' in path_df.columns and 'target' in path_df.columns
                
                if path_col is None and not has_source_target_format:
                    continue
                
                # Extract path information
                for _, row in path_df.iterrows():
                    # Handle source/target format (e.g., from data_original_paths.csv)
                    if path_col is None and has_source_target_format:
                        source_node = str(row.get('source', ''))
                        target_node = str(row.get('target', ''))
                        if not source_node or source_node == 'nan' or not target_node or target_node == 'nan':
                            continue
                        path_nodes = [source_node, target_node]
                    else:
                        # Handle path/path_str format
                        path_str = str(row.get('path', row.get('path_str', '')))
                        if not path_str or path_str == 'nan':
                            continue
                        
                        # Parse path string
                        if '->' in path_str:
                            path_nodes = [n.strip() for n in path_str.split('->')]
                        elif path_str.startswith('['):
                            try:
                                path_nodes = ast.literal_eval(path_str)
                            except Exception:
                                continue
                        else:
                            continue
                    
                    if len(path_nodes) < 2:
                        continue
                    
                    # Build path key using canonical names for cross-dataset merging
                    canonical_key, display_key = self._build_path_key_with_mapping(path_nodes, dataset)
                    
                    # Get canonical node names
                    canonical_nodes = [self._get_canonical_type(node, dataset) for node in path_nodes]
                    source = canonical_nodes[0]
                    target = canonical_nodes[-1]
                    intermediates = canonical_nodes[1:-1] if len(canonical_nodes) > 2 else []
                    
                    # Use canonical_key for merging
                    path_key = canonical_key
                    
                    # Initialize path data (use display names for output)
                    if path_key not in all_paths:
                        all_paths[path_key] = {
                            'path_key': display_key,
                            'source': self._get_display_type(source) if self.parameters.auto_type_mapping else source,
                            'target': self._get_display_type(target) if self.parameters.auto_type_mapping else target,
                            'hops': len(path_nodes) - 1,
                            'intermediates': ' → '.join([self._get_display_type(i) for i in intermediates]) if intermediates else '',
                        }
                    
                    # Add threshold-specific data
                    weight = row.get('min_weight', row.get('weight', 1))
                    if pd.isna(weight):
                        weight = 1
                    
                    # Presence and weight columns (True/0 for CSV readability)
                    pres_col = f'{safe_name}_t{threshold}'
                    weight_col = f'w_{safe_name}_t{threshold}'
                    all_paths[path_key][pres_col] = True
                    all_paths[path_key][weight_col] = float(weight)
                    
                    # Parse hop weights
                    weights_str = str(row.get('weights', row.get('hop_weights', '')))
                    if weights_str and weights_str != 'nan':
                        hop_weights_col = f'hop_{safe_name}_t{threshold}'
                        if weights_str.startswith('['):
                            all_paths[path_key][hop_weights_col] = weights_str
                        elif ',' in weights_str:
                            try:
                                hw = [float(w.strip()) for w in weights_str.split(',')]
                                all_paths[path_key][hop_weights_col] = f"[{', '.join(str(int(w)) for w in hw)}]"
                            except Exception:
                                pass
        
        if not all_paths:
            self._log("No path data available for unified path presence matrix")
            return
        
        # Build DataFrame
        rows = list(all_paths.values())
        path_df = pd.DataFrame(rows)
        
        # Fill missing presence markers with 0 (absent)
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                pres_col = f'{safe_name}_t{threshold}'
                if pres_col in path_df.columns:
                    path_df[pres_col] = path_df[pres_col].fillna(0)
                else:
                    path_df[pres_col] = 0
        
        # Add conservation count at lowest threshold
        lowest_threshold = min(thresholds)
        presence_cols = [f'{self.parameters._sanitize_name(d)}_t{lowest_threshold}' for d in dataset_names]
        presence_cols = [c for c in presence_cols if c in path_df.columns]
        if presence_cols:
            path_df['conserved_at_lowest'] = path_df[presence_cols].apply(
                lambda x: sum(1 for v in x if v == True), axis=1
            )
        
        # Reorder columns
        col_order = ['path_key', 'source', 'target', 'hops', 'intermediates']
        
        # Add presence columns by threshold
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                pres_col = f'{safe_name}_t{threshold}'
                if pres_col in path_df.columns:
                    col_order.append(pres_col)
        
        # Add weight columns by threshold
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                weight_col = f'w_{safe_name}_t{threshold}'
                if weight_col in path_df.columns:
                    col_order.append(weight_col)
        
        # Add hop weight columns by threshold  
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                hop_col = f'hop_{safe_name}_t{threshold}'
                if hop_col in path_df.columns:
                    col_order.append(hop_col)
        
        if 'conserved_at_lowest' in path_df.columns:
            col_order.append('conserved_at_lowest')
        
        col_order = [c for c in col_order if c in path_df.columns]
        path_df = path_df[col_order]
        
        # Sort by conservation
        if 'conserved_at_lowest' in path_df.columns:
            path_df = path_df.sort_values(['conserved_at_lowest', 'path_key'], ascending=[False, True])
        
        # Save
        self._save_csv(path_df, os.path.join(comparison_results_dir, "path_presence_matrix.csv"))
        self._log(f"Saved: path_presence_matrix.csv (unified, {len(path_df)} paths, {len(thresholds)} thresholds)")
        
        # Update comparison report with path presence matrix for visualizations
        if self.comparison_report is not None:
            self.comparison_report['path_presence_matrix'] = path_df

    def _export_type_resolution_union(self, comparison_results_dir: str):
        """Export the union type-resolution table (type_coverage.py).

        One row per (query, appeared type, dataset): how the type resolves
        in that dataset and why it is absent when it is — the resolver's
        verdict for the full union, not only the types a dataset's own
        pathfinding happened to recruit.
        """
        try:
            coverage = self._type_coverage()
            from .type_coverage import coverage_rows
            rows = coverage_rows(self, coverage)
        except Exception as exc:  # noqa: BLE001
            self._log(f"Warning: type resolution union export failed: {exc}")
            return
        if not rows:
            return
        output_path = os.path.join(
            comparison_results_dir, 'type_resolution_union.csv')
        self._save_csv(pd.DataFrame(rows), output_path)
        self._log_file(output_path, "Type resolution union")

    def _export_presence_matrix(self, comparison_results_dir: str, threshold: Any, silent: bool = False):
        """
        Export edge and path presence matrices showing conservation across datasets.
        
        Creates unified tables with:
        - ✔️/❌ presence markers per dataset
        - Weights per dataset
        - Conservation count (number of datasets with edge)
        - Conservation metrics (CV, max_weight)
        
        Args:
            comparison_results_dir: Directory to save output files
            threshold: Weight threshold for analysis
            silent: If True, suppress per-file logging
        """
        query = (
            self._query_record(threshold)
            if self.parameters.threshold_mode == 'combinations'
            else None
        )
        query_id = query.get('id') if query else None
        query_label = query.get('label') if query else None
        query_thresholds = query.get('thresholds', {}) if query else {}
        dataset_names = self.parameters.get_dataset_names()
        aligned = (
            self.get_aligned_data_for_query(query)
            if query else self.get_aligned_data(threshold)
        )
        
        if aligned.empty:
            self._log(f"No aligned data for presence matrix at threshold {threshold}")
            return
        
        # Get top edges union from all datasets
        top_n = self.parameters.top_edges
        
        if top_n > 0:
            top_edges_union = set()
            
            for dataset in dataset_names:
                if dataset in aligned.columns:
                    dataset_top = set(aligned.nlargest(top_n, dataset).index)
                    top_edges_union.update(dataset_top)
            
            # Limit total rows to reasonable number (2x top_edges)
            max_rows = top_n * 2
            if len(top_edges_union) > max_rows:
                # Sort by max weight across all datasets
                available = [d for d in dataset_names if d in aligned.columns]
                aligned['_max_weight'] = aligned[available].max(axis=1)
                top_edges_union = set(aligned.nlargest(max_rows, '_max_weight').index)
                aligned = aligned.drop(columns=['_max_weight'])
            
            # Filter to top edges
            matrix_df = aligned.loc[aligned.index.isin(top_edges_union)].copy()
        else:
            # Include all edges if top_edges <= 0
            matrix_df = aligned.copy()
        
        if matrix_df.empty:
            return
        
        # Build presence matrix using vectorized operations (much faster than iterrows)
        available = [d for d in dataset_names if d in matrix_df.columns]
        
        # Ensure index is a flat string index (not MultiIndex)
        if isinstance(matrix_df.index, pd.MultiIndex):
            # Convert MultiIndex to string format "source -> target"
            edge_keys = pd.Series([f"{idx[0]} -> {idx[1]}" for idx in matrix_df.index], index=matrix_df.index)
        else:
            edge_keys = matrix_df.index.astype(str)
        # ``Index.astype(str)`` stays an Index, and ``.str.split(expand=True)``
        # on an Index returns a MultiIndex — which silently fell into the
        # fallback below and wrote the full edge key into source_type (with
        # an empty target_type).  Normalize to a Series so the split is a
        # real DataFrame in both branches.
        if not isinstance(edge_keys, pd.Series):
            edge_keys = pd.Series(edge_keys, index=matrix_df.index)
        
        # Parse edge keys to source/target using vectorized string operations
        # Use try/except to handle edge cases where split returns unexpected types
        try:
            split_keys = edge_keys.str.split(' -> ', n=1, expand=True)
            # Ensure split_keys is a DataFrame with at least 2 columns
            if isinstance(split_keys, pd.DataFrame):
                source_types = split_keys[0].fillna(edge_keys)
                target_types = split_keys[1].fillna('') if 1 in split_keys.columns else pd.Series('', index=matrix_df.index)
            else:
                # Fallback: split didn't return DataFrame (edge case)
                source_types = edge_keys
                target_types = pd.Series('', index=matrix_df.index)
        except Exception:
            # Ultimate fallback
            source_types = edge_keys
            target_types = pd.Series('', index=matrix_df.index)
        
        # Start building the presence DataFrame
        presence_df = pd.DataFrame({
            'edge_key': edge_keys,
            'source_type': source_types.values,
            'target_type': target_types.values,
        }, index=matrix_df.index)
        
        # Build safe name mapping
        safe_names = {d: self.parameters._sanitize_name(d) for d in available}
        
        # Add presence markers and weight columns for each dataset (vectorized)
        for dataset in available:
            safe_name = safe_names[dataset]
            weights = matrix_df[dataset]
            is_present = weights > 0

            # Presence marker (True/0 for CSV readability)
            presence_df[safe_name] = is_present.map({True: True, False: 0})

            # Weight column (show weight if present, else empty string)
            presence_df[f'weight_{safe_name}'] = weights.where(is_present, '')

        # Per-endpoint union-resolution status columns (type_coverage.py):
        # the coverage status of each endpoint TYPE in that dataset, so a
        # False presence cell is explainable (below threshold / not in
        # dataset / unmapped / ...).  Empty when the pass has no entry.
        try:
            coverage = self._type_coverage_for_query(
                query_id if query_id else threshold)
        except Exception:  # noqa: BLE001
            coverage = {}

        def _endpoint_status(name: Any, dataset: str) -> str:
            entry = coverage.get((str(name).split('(')[0], dataset))
            return entry.status if entry is not None else ''

        if coverage:
            for dataset in available:
                safe_name = safe_names[dataset]
                presence_df[f'source_status_{safe_name}'] = [
                    _endpoint_status(name, dataset) for name in source_types
                ]
                presence_df[f'target_status_{safe_name}'] = [
                    _endpoint_status(name, dataset) for name in target_types
                ]
        
        # Calculate conservation count (number of datasets with edge > 0)
        presence_cols = [safe_names[d] for d in available]
        # Convert True/0 to 1/0 for counting
        presence_df['conservation_count'] = (matrix_df[available] > 0).sum(axis=1)
        
        # Calculate statistics (vectorized)
        weight_values = matrix_df[available].copy()
        # Replace 0 with NaN for statistics (so we only consider present edges)
        weight_values = weight_values.replace(0, np.nan)
        
        presence_df['max_weight'] = weight_values.max(axis=1)
        presence_df['avg_weight'] = weight_values.mean(axis=1)
        
        # Calculate CV only where we have more than 1 present dataset
        has_multiple = (weight_values.notna().sum(axis=1) > 1)
        cv_values = weight_values.std(axis=1) / weight_values.mean(axis=1)
        presence_df['weight_cv'] = cv_values.round(3).where(has_multiple, '')
        
        # Replace NaN with empty string for display
        presence_df['max_weight'] = presence_df['max_weight'].fillna('')
        presence_df['avg_weight'] = presence_df['avg_weight'].fillna('')
        
        if presence_df.empty:
            return
        
        # Order columns: edge_key, source, target, conservation_count, presence markers, weights, stats
        col_order = ['edge_key', 'source_type', 'target_type', 'conservation_count']
        
        # Add presence marker columns
        for dataset in available:
            safe_name = self.parameters._sanitize_name(dataset)
            if safe_name in presence_df.columns:
                col_order.append(safe_name)
        
        # Add weight columns
        for dataset in available:
            safe_name = self.parameters._sanitize_name(dataset)
            weight_col = f'weight_{safe_name}'
            if weight_col in presence_df.columns:
                col_order.append(weight_col)

        # Add endpoint coverage-status columns right after the weights
        for dataset in available:
            safe_name = self.parameters._sanitize_name(dataset)
            for role in ('source_status', 'target_status'):
                col = f'{role}_{safe_name}'
                if col in presence_df.columns:
                    col_order.append(col)
        
        # Add statistics
        col_order.extend(['max_weight', 'avg_weight', 'weight_cv'])
        
        # Ensure all columns exist
        col_order = [c for c in col_order if c in presence_df.columns]
        presence_df = presence_df[col_order]
        
        # Sort by conservation count (desc) then max weight (desc)
        presence_df = presence_df.sort_values(
            ['conservation_count', 'max_weight'],
            ascending=[False, False]
        )
        
        if query_id:
            presence_df.insert(0, 'query_id', query_id)
            presence_df.insert(1, 'query_label', query_label)
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                presence_df.insert(
                    len(presence_df.columns),
                    f'threshold_{safe_name}',
                    query_thresholds.get(dataset, ''),
                )
            safe_query_id = self._safe_query_filename_id(query_id)
            output_name = f"edge_presence_matrix_query_{safe_query_id}.csv"
        else:
            output_name = f"edge_presence_matrix_minsyn_{threshold}.csv"

        # Save edge presence matrix
        self._save_csv(presence_df, os.path.join(comparison_results_dir, output_name))
        if not silent:
            self._log(f"Saved: {output_name} ({len(presence_df)} edges)")
        
        # Also save a threshold-independent version at the middle threshold
        mid_threshold = self.parameters.thresholds[len(self.parameters.thresholds) // 2]
        if not query_id and threshold == mid_threshold:
            self._save_csv(presence_df, os.path.join(comparison_results_dir, "edge_presence_matrix.csv"))
            if not silent:
                self._log("Saved: edge_presence_matrix.csv (default)")
    
    def _export_path_presence_matrix(self, comparison_results_dir: str, threshold: Any, silent: bool = False):
        """
        Export path presence matrix showing multi-hop path conservation across datasets.
        
        Similar to edge presence matrix but for source → intermediate → target paths.
        Creates unified tables with:
        - ✔️/❌ presence markers per dataset
        - Path structure (source → inter1 → inter2 → target)
        - Hop weights as [w1, w2, ...] for each dataset
        - Conservation metrics
        
        Args:
            comparison_results_dir: Directory to save output files
            threshold: Weight threshold for analysis
            silent: If True, suppress per-file logging
        """
        import ast
        
        query = (
            self._query_record(threshold)
            if self.parameters.threshold_mode == 'combinations'
            else None
        )
        query_id = query.get('id') if query else None
        query_label = query.get('label') if query else None
        query_thresholds = query.get('thresholds', {}) if query else {}
        dataset_names = self.parameters.get_dataset_names()
        
        # Limit paths to prevent hanging on large datasets
        max_paths_per_dataset = 5000  # Safety limit (reduced from 10000)
        
        # Collect paths from path CSV files (not connection data)
        path_data = {}  # path_key -> {dataset: True/False}
        path_details = {}  # path_key -> {source, intermediates, target, weights, hop_weights}
        
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            dataset_threshold = (
                query_thresholds.get(dataset)
                if query else threshold
            )
            if dataset_threshold is None:
                continue
            
            # Find path data file: minsyn_X_data_original_paths.csv or {source}_to_{target}_allpaths_type.csv
            dataset_output_path = self.parameters.get_dataset_output_path(
                dataset, dataset_threshold)
            
            # Try multiple path file patterns
            path_files_to_try = [
                os.path.join(
                    dataset_output_path,
                    f"minsyn_{dataset_threshold}_data_original_paths.csv",
                ),
            ]
            
            # Also check for source_to_target_allpaths_type.csv pattern
            if os.path.exists(dataset_output_path):
                for f in os.listdir(dataset_output_path):
                    if f.endswith('_allpaths_type.csv'):
                        path_files_to_try.append(os.path.join(dataset_output_path, f))
            
            path_df = None
            for path_file in path_files_to_try:
                if os.path.exists(path_file):
                    try:
                        path_df = self._read_csv(path_file)
                        if not silent:
                            self._log(f"Loaded path data from {os.path.basename(path_file)} for {dataset}")
                        break
                    except Exception as e:
                        if not silent:
                            self._log(f"Warning: Could not read {path_file}: {e}")
            
            if path_df is None or path_df.empty:
                if not silent:
                    self._log(f"No path data found for {dataset} at threshold {threshold}")
                continue
            
            # Extract path information from DataFrame using vectorized operations where possible
            # Expected columns: 
            # Format 1 (allpaths_type.csv): path/path_str, weights, min_weight, length
            # Format 2 (data_original_paths.csv): source, target, weight, weights, layer
            
            # Determine format: check for 'path' or 'path_str' column vs 'source'+'target' columns
            path_col = 'path' if 'path' in path_df.columns else 'path_str' if 'path_str' in path_df.columns else None
            has_source_target_format = 'source' in path_df.columns and 'target' in path_df.columns
            
            if path_col is None and not has_source_target_format:
                if not silent:
                    self._log(f"Path file for {dataset} has no 'path' or 'source/target' columns")
                continue
            
            # Handle source/target format (e.g., from data_original_paths.csv)
            if path_col is None and has_source_target_format:
                # Create path strings from source/target columns
                valid_mask = path_df['source'].notna() & path_df['target'].notna()
                valid_paths = path_df[valid_mask].copy()
                
                if valid_paths.empty:
                    continue
                    
                # Limit paths to prevent hanging on large datasets
                if len(valid_paths) > max_paths_per_dataset:
                    if 'weight' in valid_paths.columns:
                        valid_paths = valid_paths.nlargest(max_paths_per_dataset, 'weight')
                    else:
                        valid_paths = valid_paths.head(max_paths_per_dataset)
                    if not silent:
                        self._log(f"  Limiting to top {max_paths_per_dataset} paths for {dataset}")
                
                # Process source-target format directly
                for idx, row in valid_paths.iterrows():
                    source_node = str(row['source'])
                    target_node = str(row['target'])
                    path_nodes = [source_node, target_node]
                    
                    # Build path key using canonical names for cross-dataset merging
                    canonical_key, display_key = self._build_path_key_with_mapping(path_nodes, dataset)
                    canonical_nodes = [self._get_canonical_type(node, dataset) for node in path_nodes]
                    source = canonical_nodes[0]
                    target = canonical_nodes[-1]
                    path_key = canonical_key
                    
                    # Initialize path data if not exists
                    if path_key not in path_data:
                        path_data[path_key] = {'_display_key': display_key}
                        path_details[path_key] = {
                            'source': source,
                            'target': target,
                            'intermediates': [],
                            'weights': {},
                            'hop_weights': {}
                        }
                    
                    # Mark as present
                    path_data[path_key][safe_name] = True
                    
                    # Parse hop weights from weights column
                    hop_weights_list = []
                    weights_str = str(row.get('weights', ''))
                    if weights_str and weights_str != 'nan':
                        if weights_str.startswith('['):
                            try:
                                hop_weights_list = ast.literal_eval(weights_str)
                            except Exception:
                                pass
                        elif ',' in weights_str:
                            try:
                                hop_weights_list = [float(w.strip()) for w in weights_str.split(',')]
                            except Exception:
                                pass
                    
                    if hop_weights_list:
                        path_details[path_key]['hop_weights'][safe_name] = hop_weights_list
                    
                    # Get weight value
                    weight = row.get('weight', 1)
                    if pd.isna(weight):
                        weight = hop_weights_list[0] if hop_weights_list else 1
                    
                    if safe_name not in path_details[path_key]['weights']:
                        path_details[path_key]['weights'][safe_name] = []
                    path_details[path_key]['weights'][safe_name].append(float(weight))
                
                continue  # Done with this dataset, skip the path_col logic below
            
            # Original path_col format handling (allpaths_type.csv with 'path' or 'path_str' column)
            # Vectorized: filter out null/empty paths
            valid_mask = path_df[path_col].notna() & (path_df[path_col].astype(str) != 'nan') & (path_df[path_col].astype(str) != '')
            valid_paths = path_df[valid_mask].copy()
            
            if valid_paths.empty:
                continue
            
            # Limit paths to prevent hanging on large datasets
            if len(valid_paths) > max_paths_per_dataset:
                # Sort by min_weight if available and take top paths
                if 'min_weight' in valid_paths.columns:
                    valid_paths = valid_paths.nlargest(max_paths_per_dataset, 'min_weight')
                else:
                    valid_paths = valid_paths.head(max_paths_per_dataset)
                if not silent:
                    self._log(f"  Limiting to top {max_paths_per_dataset} paths for {dataset}")
            
            # Process paths - we need to iterate here due to complex parsing, but limit to valid rows only
            for idx, row in valid_paths.iterrows():
                path_str = str(row[path_col])
                
                # Parse path string to extract nodes
                # Handle both "A->B->C" and "['A', 'B', 'C']" formats
                if '->' in path_str:
                    path_nodes = [n.strip() for n in path_str.split('->')]
                elif path_str.startswith('['):
                    # Parse list format like "['aMe12', 'KCg-d', 'PPL101']"
                    try:
                        path_nodes = ast.literal_eval(path_str)
                    except Exception:
                        continue
                else:
                    continue
                
                if len(path_nodes) < 2:
                    continue
                
                # Build path key using canonical names for cross-dataset merging
                # and display names for the output
                canonical_key, display_key = self._build_path_key_with_mapping(path_nodes, dataset)
                
                # Get canonical node names for source/intermediates/target
                canonical_nodes = [self._get_canonical_type(node, dataset) for node in path_nodes]
                source = canonical_nodes[0]
                target = canonical_nodes[-1]
                intermediates = canonical_nodes[1:-1] if len(canonical_nodes) > 2 else []
                
                # Use canonical_key for data merging (consistent across datasets)
                path_key = canonical_key
                
                # Initialize path data if not exists
                if path_key not in path_data:
                    path_data[path_key] = {'_display_key': display_key}
                    path_details[path_key] = {
                        'source': source,
                        'target': target,
                        'intermediates': intermediates,
                        'weights': {},
                        'hop_weights': {}  # Store individual hop weights
                    }
                
                # Mark as present
                path_data[path_key][safe_name] = True
                
                # Parse hop weights from weights column (e.g., "[10, 5]" or "10,5")
                hop_weights_list = []
                weights_str = str(row.get('weights', row.get('hop_weights', '')))
                if weights_str and weights_str != 'nan':
                    if weights_str.startswith('['):
                        try:
                            hop_weights_list = ast.literal_eval(weights_str)
                        except Exception:
                            pass
                    elif ',' in weights_str:
                        try:
                            hop_weights_list = [float(w.strip()) for w in weights_str.split(',')]
                        except Exception:
                            pass
                
                # Store hop weights
                if hop_weights_list:
                    path_details[path_key]['hop_weights'][safe_name] = hop_weights_list
                
                # Record min weight as the path weight
                weight = row.get('min_weight', row.get('weight', 1))
                if pd.isna(weight):
                    if hop_weights_list:
                        weight = min(hop_weights_list)
                    else:
                        weight = 1
                
                if safe_name not in path_details[path_key]['weights']:
                    path_details[path_key]['weights'][safe_name] = []
                path_details[path_key]['weights'][safe_name].append(float(weight))
        
        if not path_data:
            if not silent:
                self._log(f"No path data for path presence matrix at threshold {threshold}")
            return
        
        # Build presence matrix rows
        rows = []
        available_datasets = [self.parameters._sanitize_name(d) for d in dataset_names]
        
        for path_key, presence in path_data.items():
            details = path_details[path_key]
            
            # Use display_key for human-readable output (shows type variants)
            display_key = presence.get('_display_key', path_key)
            
            row_data = {
                'path_key': display_key,
                'source': self._get_display_type(details['source']) if self.parameters.auto_type_mapping else details['source'],
                'target': self._get_display_type(details['target']) if self.parameters.auto_type_mapping else details['target'],
                'hops': len(details['intermediates']) + 1,
                'intermediates': ' → '.join([self._get_display_type(i) for i in details['intermediates']]) if details['intermediates'] else '',
            }
            
            # Add presence markers, weights, and hop weights
            conservation_count = 0
            all_weights = []
            
            for safe_name in available_datasets:
                is_present = presence.get(safe_name, False)
                row_data[safe_name] = True if is_present else 0  # True/0 for CSV readability
                
                if is_present:
                    conservation_count += 1
                    weights = details['weights'].get(safe_name, [])
                    hop_weights = details['hop_weights'].get(safe_name, [])
                    
                    if weights:
                        avg_weight = np.mean(weights)
                        row_data[f'weight_{safe_name}'] = round(avg_weight, 2)
                        all_weights.extend(weights)
                    else:
                        row_data[f'weight_{safe_name}'] = ''
                    
                    # Add hop weights as formatted string -w1-w2- with dashes
                    if hop_weights:
                        row_data[f'hop_weights_{safe_name}'] = '-' + '-'.join(str(int(w)) for w in hop_weights) + '-'
                    else:
                        row_data[f'hop_weights_{safe_name}'] = ''
                else:
                    row_data[f'weight_{safe_name}'] = ''
                    row_data[f'hop_weights_{safe_name}'] = ''
            
            row_data['conservation_count'] = conservation_count
            
            # Calculate statistics
            if all_weights:
                row_data['max_weight'] = round(max(all_weights), 2)
                row_data['avg_weight'] = round(np.mean(all_weights), 2)
                if len(all_weights) > 1:
                    row_data['weight_cv'] = round(np.std(all_weights) / np.mean(all_weights), 3)
                else:
                    row_data['weight_cv'] = ''
            else:
                row_data['max_weight'] = ''
                row_data['avg_weight'] = ''
                row_data['weight_cv'] = ''
            
            rows.append(row_data)
        
        if not rows:
            return
        
        # Create DataFrame and sort
        path_presence_df = pd.DataFrame(rows)
        
        # Reorder columns for better readability
        col_order = ['path_key', 'source', 'target', 'hops', 'intermediates', 'conservation_count']
        
        # Add presence columns
        for safe_name in available_datasets:
            if safe_name in path_presence_df.columns:
                col_order.append(safe_name)
        
        # Add weight columns  
        for safe_name in available_datasets:
            weight_col = f'weight_{safe_name}'
            if weight_col in path_presence_df.columns:
                col_order.append(weight_col)
        
        # Add hop weights columns
        for safe_name in available_datasets:
            hop_col = f'hop_weights_{safe_name}'
            if hop_col in path_presence_df.columns:
                col_order.append(hop_col)
        
        # Add statistics
        col_order.extend(['max_weight', 'avg_weight', 'weight_cv'])
        col_order = [c for c in col_order if c in path_presence_df.columns]
        path_presence_df = path_presence_df[col_order]
        
        path_presence_df = path_presence_df.sort_values(
            ['conservation_count', 'max_weight'],
            ascending=[False, False]
        )
        
        # Limit to top paths only when a positive report-row cap is configured.
        # Zero/negative values mean include all rows, matching the edge
        # presence matrix and the top-edge metric helpers.
        if self.parameters.top_edges > 0:
            max_paths = self.parameters.top_edges * 2
            if len(path_presence_df) > max_paths:
                path_presence_df = path_presence_df.head(max_paths)
        
        if query_id:
            path_presence_df.insert(0, 'query_id', query_id)
            path_presence_df.insert(1, 'query_label', query_label)
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                path_presence_df.insert(
                    len(path_presence_df.columns),
                    f'threshold_{safe_name}',
                    query_thresholds.get(dataset, ''),
                )
            safe_query_id = self._safe_query_filename_id(query_id)
            output_name = f"path_presence_matrix_query_{safe_query_id}.csv"
        else:
            output_name = f"path_presence_matrix_minsyn_{threshold}.csv"

        # Save
        self._save_csv(path_presence_df, os.path.join(comparison_results_dir, output_name))
        if not silent:
            self._log(f"Saved: {output_name} ({len(path_presence_df)} paths)")
        
        # Save default version at middle threshold
        mid_threshold = self.parameters.thresholds[len(self.parameters.thresholds) // 2]
        if not query_id and threshold == mid_threshold:
            self._save_csv(path_presence_df, os.path.join(comparison_results_dir, "path_presence_matrix.csv"))
            if not silent:
                self._log("Saved: path_presence_matrix.csv (default)")
    
    def _export_motif_analysis(self, comparison_results_dir: str, threshold: Any):
        """
        Export network motif analysis for each dataset.
        
        Detects and compares common network motifs:
        - Feedforward loops (A→B→C, A→C)
        - Feedback loops (A→B→A)
        - Fan-in patterns (multiple inputs to one node)
        - Fan-out patterns (one node with multiple outputs)
        - Reciprocal connections
        
        Args:
            comparison_results_dir: Directory to save output files
            threshold: Weight threshold for analysis
        """
        query = (
            self._query_record(threshold)
            if self.parameters.threshold_mode == 'combinations'
            else None
        )
        query_id = query.get('id') if query else None
        query_thresholds = query.get('thresholds', {}) if query else {}
        dataset_names = self.parameters.get_dataset_names()
        motif_data = []
        
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            
            # Build graph from connection data
            dataset_threshold = (
                query_thresholds.get(dataset)
                if query else threshold
            )
            df = self.raw_results.get(dataset, {}).get(
                dataset_threshold, pd.DataFrame())
            if df.empty:
                continue
            
            # Extract edges using vectorized operations
            source_col = 'type_pre' if 'type_pre' in df.columns else 'source'
            target_col = 'type_post' if 'type_post' in df.columns else 'target'
            
            # Filter valid rows
            valid_mask = (
                df[source_col].notna() & 
                df[target_col].notna() & 
                (df[source_col].astype(str) != 'nan') & 
                (df[target_col].astype(str) != 'nan')
            )
            valid_df = df[valid_mask]
            
            # Build edge set and adjacency structures
            sources = valid_df[source_col].astype(str).values
            targets = valid_df[target_col].astype(str).values
            
            edge_set = set(zip(sources, targets))
            nodes = set(sources) | set(targets)
            
            # Build adjacency dict for fast neighbor lookup
            out_neighbors = {}  # node -> set of outgoing neighbors
            for s, t in edge_set:
                if s not in out_neighbors:
                    out_neighbors[s] = set()
                out_neighbors[s].add(t)
            
            # Calculate motif counts
            feedforward_loops = 0
            feedback_loops = 0
            reciprocal_connections = 0
            
            # Fan-in/fan-out analysis using vectorized counting
            from collections import Counter
            out_degree = dict(Counter(sources))
            in_degree = dict(Counter(targets))
            
            # Find reciprocal connections - only count each pair once
            for (s, t) in edge_set:
                if s < t and (t, s) in edge_set:  # Only count when s < t to avoid double counting
                    reciprocal_connections += 1
                elif s == t:  # Self-loop (edge to itself)
                    pass  # Don't count self-loops as reciprocal
                elif s > t and (t, s) in edge_set:
                    pass  # Skip - already counted
            
            # Find feedforward loops (A→B→C where A→C also exists)
            # Use adjacency dict for O(E) instead of O(E*N)
            for a in out_neighbors:
                a_neighbors = out_neighbors.get(a, set())
                for b in a_neighbors:
                    b_neighbors = out_neighbors.get(b, set())
                    # Check which C nodes (neighbors of B) are also neighbors of A
                    common = a_neighbors & b_neighbors
                    feedforward_loops += len(common - {a})  # Exclude A itself
            
            # Find feedback loops (A→B→A cycles) - same as reciprocal
            feedback_loops = reciprocal_connections
            
            # Calculate hub metrics
            max_out_degree = max(out_degree.values()) if out_degree else 0
            max_in_degree = max(in_degree.values()) if in_degree else 0
            avg_out_degree = np.mean(list(out_degree.values())) if out_degree else 0
            avg_in_degree = np.mean(list(in_degree.values())) if in_degree else 0
            
            # Find fan-in/fan-out hubs
            fan_out_hubs = [n for n, d in out_degree.items() if d >= 3]  # Nodes with 3+ outputs
            fan_in_hubs = [n for n, d in in_degree.items() if d >= 3]  # Nodes with 3+ inputs
            
            motif_data.append({
                'dataset': safe_name,
                'threshold': dataset_threshold,
                **({
                    'query_id': query_id,
                    'query_label': query.get('label', query_id),
                    'threshold_mode': 'combinations',
                } if query else {}),
                'total_nodes': len(nodes),
                'total_edges': len(edge_set),
                'feedforward_loops': feedforward_loops,
                'feedback_loops': feedback_loops,
                'reciprocal_connections': reciprocal_connections,
                'fan_out_hubs': len(fan_out_hubs),
                'fan_in_hubs': len(fan_in_hubs),
                'max_out_degree': max_out_degree,
                'max_in_degree': max_in_degree,
                'avg_out_degree': round(avg_out_degree, 2),
                'avg_in_degree': round(avg_in_degree, 2),
                'density': round(len(edge_set) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0, 4),
            })
        
        if not motif_data:
            self._log(f"No motif data available at threshold {threshold}")
            return motif_data  # Return for later merging
        
        return motif_data  # Return for merging in export_cross_dataset_comparisons

    def _generate_visualizations(self, output_dir: str):
        """
        Generate and save all matplotlib visualizations.
        
        Creates visualization PNG files in the comparison_visualizations/ subfolder at base level.
        Also generates interactive HTML heatmaps using VisualizePath.
        Uses cached similarities to avoid redundant calculations.
        
        Args:
            output_dir: Base output directory
        """
        try:
            from .visualizations import ComparisonVisualizer
        except ImportError:
            self._log("Warning: ComparisonVisualizer not available, skipping visualizations")
            return
        
        # Save visualizations to comparison_visualizations/ at base level (not inside comparison_results/)
        vis_dir = os.path.join(output_dir, "comparison_visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        if self.parameters.threshold_mode == 'combinations':
            queries = self.get_threshold_queries()
            # A combination query is a row-wise comparison point.  Render
            # every row, never the middle row of the raw threshold union.
            visualization_thresholds = [query['id'] for query in queries]
            selected_query = queries[len(queries) // 2] if queries else None
            mid_threshold = selected_query['id'] if selected_query else None
            aligned = (
                self.get_aligned_data_for_query(selected_query)
                if selected_query else pd.DataFrame())
            visualization_results = {
                dataset: {
                    query['id']: self.raw_results.get(dataset, {}).get(
                        query['thresholds'][dataset], pd.DataFrame())
                    for query in queries
                }
                for dataset in self.parameters.get_dataset_names()
            }
            align_func = self.get_aligned_data_for_query
            similarity_func = lambda query_id: self._similarity_cache.get(
                query_id, pd.DataFrame())
            path_data_func = lambda query_id: self._get_path_data_for_query(
                self._query_record(query_id))
            # Ratio/probability filtering is disabled for pathfinding
            # comparisons.  Passing None also prevents stale by_ratio/
            # by_probability folders from being emitted by the generic
            # visualizer.
            ratio_data_func = None
            prob_data_func = None
            path_presence = pd.DataFrame()
            point_metadata_func = lambda query_id: self._comparison_point_metadata(
                query_id)
            point_label_func = lambda query_id: self._comparison_point_label(
                query_id)
            point_stem_func = lambda query_id: self._comparison_point(
                query_id).file_stem
        else:
            mid_threshold = self.parameters.thresholds[
                len(self.parameters.thresholds) // 2
            ]
            aligned = self.get_aligned_data(mid_threshold)
            visualization_thresholds = self._analysis_thresholds()
            visualization_results = self.get_mapped_results()
            align_func = self.get_aligned_data
            similarity_func = self.get_cached_similarities
            path_data_func = self._get_path_data_for_threshold
            # Ratio/probability filtering is disabled for pathfinding
            # comparisons in BOTH modes (parity with the combinations branch
            # above).  Passing None keeps by_ratio/by_probability folders
            # from being emitted for Standard runs as well.
            ratio_data_func = None
            prob_data_func = None
            point_metadata_func = None
            point_label_func = None
            point_stem_func = None
        dataset_names = self.parameters.get_dataset_names()
        
        # Get pairwise similarities if available
        pairwise_sim = pd.DataFrame()
        if self.comparison_report and 'pairwise_similarities' in self.comparison_report:
            pairwise_sim = self.comparison_report['pairwise_similarities']
            if pairwise_sim is None:
                pairwise_sim = pd.DataFrame()
        
        # Get path presence matrix if available
        path_presence = pd.DataFrame()
        if self.comparison_report and 'path_presence_matrix' in self.comparison_report:
            path_presence = self.comparison_report['path_presence_matrix']
        
        try:
            visualizer = ComparisonVisualizer(verbose=self.verbose)
            
            # Build nickname map from parameters
            dataset_names = self.parameters.get_dataset_names()
            nicknames = self.parameters.get_dataset_nicknames()
            nickname_map = dict(zip(dataset_names, nicknames))
            
            # Get type-mapped results for proper cross-dataset comparison
            # This ensures types like MeVPaMe1 (male-cns) and MTe46 (FAFB) are recognized as the same
            mapped_results = visualization_results
            
            # Generate all standard plots, passing cached similarity function
            # Feature G: render only the effective (non-duplicate) thresholds
            visualizer.save_all_plots(
                results=mapped_results,
                aligned_data=aligned,
                similarities=pairwise_sim,
                output_dir=vis_dir,
                thresholds=visualization_thresholds,
                align_func=align_func,
                similarity_func=similarity_func,
                current_threshold=mid_threshold,
                path_data_func=path_data_func,
                ratio_data_func=ratio_data_func,
                prob_data_func=prob_data_func,
                output_base_path=self.parameters.full_output_path,
                nickname_map=nickname_map,
                path_presence_matrix=path_presence,  # Pass path presence matrix for accurate path counts
                point_metadata_func=point_metadata_func,
                point_label_func=point_label_func,
                point_stem_func=point_stem_func,
                threshold_mode=getattr(self.parameters, 'threshold_mode', 'standard'),
                silent=True  # Suppress per-file messages, show summary instead
            )
            
            self._log_file(vis_dir, "Saved visualizations")
        except Exception as e:
            self._log(f"Warning: Failed to generate some visualizations: {e}")

        # Feature D: edge-density-vs-threshold curves from the alignment
        # prober's extended grid (costs nothing once the prober ran).
        density_df = getattr(self, '_alignment_density_df', None)
        if density_df is not None and not density_df.empty:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from .visualizations import ComparisonVisualizer as _CV
                fig = _CV(verbose=self.verbose).plot_edge_density_curves(
                    density_df,
                    nickname_map=nickname_map,
                    best_matches=getattr(self, '_alignment_best_df', None),
                    log_y=True,
                )
                out_path = os.path.join(
                    vis_dir, "edge_density_threshold_curves.png")
                fig.savefig(out_path, dpi=200, bbox_inches='tight')
                plt.close(fig)
                self._log_file(out_path, "Edge density vs threshold curves")
            except Exception as e:
                self._log(f"Warning: edge density curve plot failed: {e}")

        # Auto threshold-density alignment: two-panel query-scoped density
        # curves (plan §5 Phase E).
        auto_curves = getattr(self, '_density_curves_df', None)
        if auto_curves is not None and not auto_curves.empty:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from .visualizations import ComparisonVisualizer as _CV
                fig = _CV(verbose=self.verbose).plot_density_alignment_curves(
                    auto_curves,
                    windows=getattr(self, '_density_windows_df', None),
                    aligned=getattr(self, '_density_alignment_df', None),
                    nickname_map=nickname_map,
                    normalizer_label=(
                        'edges per searched node (E(t)/N)'
                        if getattr(self.parameters, 'density_normalizer',
                                   'per_node') == 'per_node'
                        else getattr(self.parameters, 'density_normalizer',
                                     'per_node')),
                )
                out_path = os.path.join(
                    vis_dir, "density_alignment_threshold_curves.png")
                fig.savefig(out_path, dpi=200, bbox_inches='tight')
                plt.close(fig)
                self._log_file(
                    out_path, "Density alignment vs threshold curves")
            except Exception as e:
                self._log(f"Warning: density alignment plot failed: {e}")

        # Feature C: typed-grid alignment distance heatmap.
        try:
            self._export_threshold_alignment_heatmap(vis_dir)
        except Exception as e:
            self._log(f"Warning: alignment heatmap failed: {e}")
        
        # Generate VisualizePath interactive heatmaps (no separate network files)
        self._generate_vispath_visualizations(vis_dir)

        # Generate the conserved-path network files consumed by the report.
        # The per-threshold helper already handles Standard and combinations;
        # without this orchestration call the report can only render a dash
        # even when the alignment contains conserved edges.
        try:
            self.visualize_conserved_paths_all_thresholds()
        except Exception as e:
            self._log(f"Warning: Failed to generate conserved path graphs: {e}")

        # Generate conserved reciprocal graphs when enabled
        if getattr(self.parameters, 'find_reciprocal', False):
            try:
                self.visualize_conserved_reciprocal_graph_all_thresholds()
            except Exception as e:
                self._log(f"Warning: Failed to generate conserved reciprocal graphs: {e}")
    
    def _get_path_data_for_threshold(self, threshold: Any) -> pd.DataFrame:
        """
        Get path min_weight data aligned across all datasets for a given threshold.
        
        Reads from dataset_data/{dataset}/minsyn_{threshold}/*_original_paths.csv
        or *_allpaths_type.csv and extracts min_weight values, aligning paths across datasets.
        
        Args:
            threshold: The threshold level
            
        Returns:
            DataFrame with path index and dataset columns containing min_weight
        """
        query_map = threshold.get('thresholds', {}) if isinstance(
            threshold, dict) else None
        scalar_threshold = None if query_map is not None else threshold
        dataset_names = self.parameters.get_dataset_names()
        all_path_data = {}
        
        for dataset_name in dataset_names:
            dataset_threshold = (
                query_map.get(dataset_name)
                if query_map is not None else scalar_threshold
            )
            if dataset_threshold is None:
                continue
            dataset_output_path = self._resolve_dataset_output_path(
                dataset_name, dataset_threshold)
            
            # Try multiple path file patterns
            path_files_to_try = [
                os.path.join(
                    dataset_output_path,
                    f'minsyn_{dataset_threshold}_data_original_paths.csv',
                ),
            ]
            
            # Also check for source_to_target_allpaths_type.csv pattern
            if os.path.exists(dataset_output_path):
                for f in os.listdir(dataset_output_path):
                    if f.endswith('_allpaths_type.csv'):
                        path_files_to_try.append(os.path.join(dataset_output_path, f))
            
            # Try to read from available files
            df = None
            for path_file in path_files_to_try:
                if os.path.exists(path_file):
                    try:
                        df = self._read_csv(path_file)
                        break
                    except Exception as e:
                        self._log(f"Warning: Could not read {path_file}: {e}")
            
            if df is not None:
                # Handle two formats:
                # Format 1 (allpaths_type.csv): 'path' and 'min_weight' columns
                # Format 2 (data_original_paths.csv): 'source', 'target', and 'weight' columns
                has_path_format = 'path' in df.columns and 'min_weight' in df.columns
                has_source_target_format = 'source' in df.columns and 'target' in df.columns and 'weight' in df.columns
                
                if has_path_format:
                    for _, row in df.iterrows():
                        original_path_key = row['path']
                        min_weight = row['min_weight']
                        
                        # Apply type mapping to path key if auto_type_mapping is enabled
                        if self.parameters.auto_type_mapping and self.parameters._auto_type_mapper:
                            # Parse path nodes
                            if '->' in str(original_path_key):
                                path_nodes = [n.strip() for n in str(original_path_key).split('->')]
                            elif ' → ' in str(original_path_key):
                                path_nodes = [n.strip() for n in str(original_path_key).split(' → ')]
                            else:
                                path_nodes = [str(original_path_key)]
                            
                            # Build canonical and display keys
                            canonical_key, display_key = self._build_path_key_with_mapping(path_nodes, dataset_name)
                            path_key = canonical_key  # Use canonical key for merging
                        else:
                            path_key = original_path_key
                            display_key = original_path_key
                        
                        if path_key not in all_path_data:
                            all_path_data[path_key] = {'_display_key': display_key}
                        all_path_data[path_key][dataset_name] = min_weight
                
                elif has_source_target_format:
                    # Handle source/target format from data_original_paths.csv
                    for _, row in df.iterrows():
                        source = str(row['source'])
                        target = str(row['target'])
                        weight = row['weight']
                        
                        if pd.isna(source) or pd.isna(target) or source == 'nan' or target == 'nan':
                            continue
                        
                        path_nodes = [source, target]
                        
                        # Apply type mapping if enabled
                        if self.parameters.auto_type_mapping and self.parameters._auto_type_mapper:
                            canonical_key, display_key = self._build_path_key_with_mapping(path_nodes, dataset_name)
                            path_key = canonical_key
                        else:
                            path_key = f"{source} → {target}"
                            display_key = path_key
                        
                        if path_key not in all_path_data:
                            all_path_data[path_key] = {'_display_key': display_key}
                        all_path_data[path_key][dataset_name] = weight
        
        if not all_path_data:
            return pd.DataFrame()
        
        # Build result DataFrame using display keys for index
        result_rows = []
        for canonical_key, data in all_path_data.items():
            display_key = data.pop('_display_key', canonical_key)
            row_data = {d: data.get(d, 0) for d in dataset_names}
            result_rows.append((display_key, row_data))
        
        if not result_rows:
            return pd.DataFrame()
        
        result_df = pd.DataFrame([r[1] for r in result_rows], index=[r[0] for r in result_rows])
        return result_df.fillna(0)

    def _get_path_data_for_query(self, query: Dict[str, Any]) -> pd.DataFrame:
        """Query-aware wrapper used by advanced similarity metrics."""
        return self._get_path_data_for_threshold(query)
    
    def _get_path_hop_weights_for_threshold(
            self, threshold: Any) -> Dict[str, Dict[str, List[float]]]:
        """Get hop weights for a scalar threshold or a query threshold map.

        Query reports pass the complete query object so each dataset is read
        from its own requested ``minsyn_<threshold>`` directory.  The source
        files exist in two layouts: a path/weights table and a
        source/target/weights table.  Both are normalized into the same path
        key consumed by the report renderer.
        """
        import ast

        query_map = (threshold.get('thresholds', {})
                     if isinstance(threshold, dict) else None)
        scalar_threshold = None if query_map is not None else threshold
        dataset_names = self.parameters.get_dataset_names()
        all_hop_weights = {}

        def _parse_weights(value) -> List[float]:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return []
            if isinstance(value, (list, tuple, np.ndarray)):
                raw_values = list(value)
            else:
                text = str(value).strip()
                if not text or text.lower() == 'nan':
                    return []
                raw_values = None
                try:
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, (list, tuple)):
                        raw_values = list(parsed)
                except (ValueError, SyntaxError):
                    pass
                if raw_values is None:
                    raw_values = text.strip('[]()').split(',')
            weights = []
            for item in raw_values:
                try:
                    weights.append(float(item))
                except (TypeError, ValueError):
                    return []
            return weights

        for dataset_name in dataset_names:
            dataset_threshold = (
                query_map.get(dataset_name)
                if query_map is not None else scalar_threshold
            )
            if dataset_threshold is None:
                continue
            safe_name = self.parameters._sanitize_name(dataset_name)
            dataset_output_path = self._resolve_dataset_output_path(
                dataset_name, dataset_threshold)

            path_files_to_try = [
                os.path.join(
                    dataset_output_path,
                    f'minsyn_{dataset_threshold}_data_original_paths.csv'),
            ]
            if os.path.exists(dataset_output_path):
                for filename in os.listdir(dataset_output_path):
                    if filename.endswith('_allpaths_type.csv'):
                        path_files_to_try.append(
                            os.path.join(dataset_output_path, filename))

            df = None
            for path_file in path_files_to_try:
                if os.path.exists(path_file):
                    try:
                        df = self._read_csv(path_file)
                        break
                    except Exception as exc:
                        self._log(
                            f"Warning: Could not read {path_file}: {exc}")
            if df is None:
                continue

            has_path = 'path' in df.columns
            has_endpoints = {'source', 'target'} <= set(df.columns)
            weight_column = next(
                (column for column in ('weights', 'hop_weights')
                 if column in df.columns), None)
            if weight_column is None or not (has_path or has_endpoints):
                continue

            for _, row in df.iterrows():
                if has_path and not pd.isna(row.get('path')):
                    original_path_key = row.get('path')
                elif has_endpoints:
                    source = row.get('source')
                    target = row.get('target')
                    if pd.isna(source) or pd.isna(target):
                        continue
                    original_path_key = f"{source} → {target}"
                else:
                    continue
                hop_weights = _parse_weights(row.get(weight_column))
                if not hop_weights:
                    continue

                original_path_key = str(original_path_key)
                if (self.parameters.auto_type_mapping
                        and self.parameters._auto_type_mapper):
                    if '->' in original_path_key:
                        path_nodes = [
                            node.strip()
                            for node in original_path_key.split('->')]
                    elif ' → ' in original_path_key:
                        path_nodes = [
                            node.strip()
                            for node in original_path_key.split(' → ')]
                    else:
                        path_nodes = [original_path_key]
                    path_key, _ = self._build_path_key_with_mapping(
                        path_nodes, dataset_name)
                else:
                    path_key = original_path_key

                all_hop_weights.setdefault(path_key, {})[safe_name] = \
                    hop_weights

        return all_hop_weights
    
    def _get_ratio_data_for_threshold(self, threshold: int) -> pd.DataFrame:
        """
        Get min_ratio data aligned across all datasets for a given threshold.
        
        Reads from dataset_data/{dataset}/minsyn_{threshold}/*_original_paths.csv
        or *_allpaths_type.csv and extracts min_ratio values.
        
        Args:
            threshold: The threshold level
            
        Returns:
            DataFrame with path index and dataset columns containing min_ratio
        """
        dataset_names = self.parameters.get_dataset_names()
        all_ratio_data = {}
        
        for dataset_name in dataset_names:
            dataset_output_path = self._resolve_dataset_output_path(
                dataset_name, threshold)
            
            # Try multiple path file patterns
            path_files_to_try = [
                os.path.join(dataset_output_path, f'minsyn_{threshold}_data_original_paths.csv'),
            ]
            
            # Also check for source_to_target_allpaths_type.csv pattern
            if os.path.exists(dataset_output_path):
                for f in os.listdir(dataset_output_path):
                    if f.endswith('_allpaths_type.csv'):
                        path_files_to_try.append(os.path.join(dataset_output_path, f))
            
            # Try to read from available files
            df = None
            for path_file in path_files_to_try:
                if os.path.exists(path_file):
                    try:
                        df = self._read_csv(path_file)
                        break
                    except Exception as e:
                        self._log(f"Warning: Could not read {path_file}: {e}")
            
            if df is not None and 'path' in df.columns and 'min_ratio' in df.columns:
                for _, row in df.iterrows():
                    original_path_key = row['path']
                    min_ratio = row['min_ratio']
                    
                    # Apply type mapping to path key if auto_type_mapping is enabled
                    if self.parameters.auto_type_mapping and self.parameters._auto_type_mapper:
                        # Parse path nodes
                        if '->' in str(original_path_key):
                            path_nodes = [n.strip() for n in str(original_path_key).split('->')]
                        elif ' → ' in str(original_path_key):
                            path_nodes = [n.strip() for n in str(original_path_key).split(' → ')]
                        else:
                            path_nodes = [str(original_path_key)]
                        
                        # Build canonical and display keys
                        canonical_key, display_key = self._build_path_key_with_mapping(path_nodes, dataset_name)
                        path_key = canonical_key  # Use canonical key for merging
                    else:
                        path_key = original_path_key
                        display_key = original_path_key
                    
                    if path_key not in all_ratio_data:
                        all_ratio_data[path_key] = {'_display_key': display_key}
                    all_ratio_data[path_key][dataset_name] = min_ratio
        
        if not all_ratio_data:
            return pd.DataFrame()
        
        # Build result DataFrame using display keys for index
        result_rows = []
        for canonical_key, data in all_ratio_data.items():
            display_key = data.pop('_display_key', canonical_key)
            row_data = {d: data.get(d, 0) for d in dataset_names}
            result_rows.append((display_key, row_data))
        
        if not result_rows:
            return pd.DataFrame()
        
        result_df = pd.DataFrame([r[1] for r in result_rows], index=[r[0] for r in result_rows])
        return result_df.fillna(0)
    
    def _get_prob_data_for_threshold(self, threshold: int) -> pd.DataFrame:
        """
        Get traversal probability (path_prob) data aligned across datasets for a threshold.
        
        Reads from dataset_data/{dataset}/minsyn_{threshold}/*_original_paths.csv
        or *_allpaths_type.csv and extracts path_prob values.
        
        Args:
            threshold: The threshold level
            
        Returns:
            DataFrame with path index and dataset columns containing path_prob
        """
        dataset_names = self.parameters.get_dataset_names()
        all_prob_data = {}
        
        for dataset_name in dataset_names:
            dataset_output_path = self._resolve_dataset_output_path(
                dataset_name, threshold)
            
            # Try multiple path file patterns
            path_files_to_try = [
                os.path.join(dataset_output_path, f'minsyn_{threshold}_data_original_paths.csv'),
            ]
            
            # Also check for source_to_target_allpaths_type.csv pattern
            if os.path.exists(dataset_output_path):
                for f in os.listdir(dataset_output_path):
                    if f.endswith('_allpaths_type.csv'):
                        path_files_to_try.append(os.path.join(dataset_output_path, f))
            
            # Try to read from available files
            df = None
            for path_file in path_files_to_try:
                if os.path.exists(path_file):
                    try:
                        df = self._read_csv(path_file)
                        break
                    except Exception as e:
                        self._log(f"Warning: Could not read {path_file}: {e}")
            
            if df is not None and 'path' in df.columns and 'path_prob' in df.columns:
                for _, row in df.iterrows():
                    original_path_key = row['path']
                    path_prob = row['path_prob']
                    
                    # Apply type mapping to path key if auto_type_mapping is enabled
                    if self.parameters.auto_type_mapping and self.parameters._auto_type_mapper:
                        # Parse path nodes
                        if '->' in str(original_path_key):
                            path_nodes = [n.strip() for n in str(original_path_key).split('->')]
                        elif ' → ' in str(original_path_key):
                            path_nodes = [n.strip() for n in str(original_path_key).split(' → ')]
                        else:
                            path_nodes = [str(original_path_key)]
                        
                        # Build canonical and display keys
                        canonical_key, display_key = self._build_path_key_with_mapping(path_nodes, dataset_name)
                        path_key = canonical_key  # Use canonical key for merging
                    else:
                        path_key = original_path_key
                        display_key = original_path_key
                    
                    if path_key not in all_prob_data:
                        all_prob_data[path_key] = {'_display_key': display_key}
                    all_prob_data[path_key][dataset_name] = path_prob
        
        if not all_prob_data:
            return pd.DataFrame()
        
        # Build result DataFrame using display keys for index
        result_rows = []
        for canonical_key, data in all_prob_data.items():
            display_key = data.pop('_display_key', canonical_key)
            row_data = {d: data.get(d, 0) for d in dataset_names}
            result_rows.append((display_key, row_data))
        
        if not result_rows:
            return pd.DataFrame()
        
        result_df = pd.DataFrame([r[1] for r in result_rows], index=[r[0] for r in result_rows])
        return result_df.fillna(0)

    def _get_edge_ratio_data_for_threshold(self, threshold: Any) -> pd.DataFrame:
        """
        Get edge-level connection_ratio data aligned across datasets for a threshold.
        
        Reads from:
        1. dataset_data/{dataset}/minsyn_{threshold}/connections_edge.csv (edge mode)
        2. dataset_data/{dataset}/minsyn_{threshold}/data_details/connection_type.csv (path mode fallback)
        
        connection_ratio = w_ij / W_j (edge weight / total post-synaptic sites)
        
        Args:
            threshold: The threshold level
            
        Returns:
            DataFrame with edge index and dataset columns containing connection_ratio
        """
        dataset_names = self.parameters.get_dataset_names()
        all_ratio_data = {}
        
        for dataset_name in dataset_names:
            dataset_threshold = self._threshold_for_dataset(
                threshold, dataset_name)
            if dataset_threshold is None:
                continue
            dataset_output_path = self._resolve_dataset_output_path(
                dataset_name, dataset_threshold)
            
            df = None
            
            # Try to read connections_edge.csv first (edge mode output)
            conn_file = os.path.join(dataset_output_path, 'connections_edge.csv')
            file_exists = os.path.exists(conn_file)
            if file_exists:
                try:
                    # Check file is not empty before reading
                    if os.path.getsize(conn_file) > 0:
                        df = self._read_csv(conn_file)
                except pd.errors.EmptyDataError:
                    pass  # File is empty or has no columns, try fallback
                except Exception:
                    pass  # Other errors, try fallback
            
            # Fallback to data_details/connection_type.csv (path mode output)
            if df is None or df.empty or 'connection_ratio' not in df.columns:
                conn_type_file = os.path.join(dataset_output_path, 'data_details', 'connection_type.csv')
                if os.path.exists(conn_type_file):
                    try:
                        if os.path.getsize(conn_type_file) > 0:
                            df = self._read_csv(conn_type_file)
                    except pd.errors.EmptyDataError:
                        pass  # File is empty
                    except Exception:
                        pass  # Other errors
            
            if df is None or df.empty or 'connection_ratio' not in df.columns:
                continue
            
            # Determine edge columns
            if 'std_label_pre' in df.columns and 'std_label_post' in df.columns:
                pre_col, post_col = 'std_label_pre', 'std_label_post'
            elif 'type_pre' in df.columns and 'type_post' in df.columns:
                pre_col, post_col = 'type_pre', 'type_post'
            else:
                pre_col, post_col = 'bodyId_pre', 'bodyId_post'
            
            # Aggregate by edge - use mean for connection_ratio
            for _, row in df.iterrows():
                pre_type = str(row[pre_col])
                post_type = str(row[post_col])
                
                # Apply type mapping if auto_type_mapping is enabled
                if self.parameters.auto_type_mapping and self.parameters._auto_type_mapper:
                    canonical_pre = self._get_canonical_type(pre_type, dataset_name)
                    canonical_post = self._get_canonical_type(post_type, dataset_name)
                    # _get_display_type takes canonical name, not (type, dataset)
                    display_pre = self._get_display_type(canonical_pre)
                    display_post = self._get_display_type(canonical_post)
                    edge_key = f"{canonical_pre} -> {canonical_post}"
                    display_edge_key = f"{display_pre} -> {display_post}"
                else:
                    edge_key = f"{pre_type} -> {post_type}"
                    display_edge_key = edge_key
                
                ratio_val = row['connection_ratio']
                if pd.notna(ratio_val):
                    if edge_key not in all_ratio_data:
                        all_ratio_data[edge_key] = {'_display_key': display_edge_key}
                    # Store ratio, will be averaged later if multiple edges
                    if dataset_name not in all_ratio_data[edge_key]:
                        all_ratio_data[edge_key][dataset_name] = []
                    all_ratio_data[edge_key][dataset_name].append(ratio_val)
        
        if not all_ratio_data:
            return pd.DataFrame()
        
        # Convert lists to averages and build result DataFrame
        dataset_names = self.parameters.get_dataset_names()
        result_rows = []
        for edge_key, data in all_ratio_data.items():
            display_key = data.pop('_display_key', edge_key)
            row_data = {}
            for ds in dataset_names:
                vals = data.get(ds, [])
                if isinstance(vals, list) and vals:
                    row_data[ds] = sum(vals) / len(vals)
                else:
                    row_data[ds] = 0.0
            result_rows.append((display_key, row_data))
        
        if not result_rows:
            return pd.DataFrame()
        
        result_df = pd.DataFrame([r[1] for r in result_rows], index=[r[0] for r in result_rows])
        return result_df.fillna(0)

    def _get_edge_nt_details(self, threshold: Any) -> Dict[str, Dict[str, str]]:
        """
        Get NT type per edge per dataset at a threshold.
        
        Returns:
            Dict[canonical_edge_key, Dict[dataset_name, nt_type]]
        """
        dataset_names = self.parameters.get_dataset_names()
        nt_by_edge: Dict[str, Dict[str, str]] = {} # canonical_key -> {dataset: nt}

        for dataset_name in dataset_names:
            dataset_threshold = self._threshold_for_dataset(
                threshold, dataset_name)
            if dataset_threshold is None:
                continue
            safe_name = self.parameters._sanitize_name(dataset_name)
            
            # Try reciprocal first, then standard
            # We want the most specific NT info available
            dataset_output_path = os.path.join(
                self._resolve_dataset_output_path(
                    dataset_name, dataset_threshold),
                'find_reciprocal'
            )
            conn_file = os.path.join(dataset_output_path, 'reciprocal_connection_type.csv')
            
            df = None
            if os.path.exists(conn_file) and os.path.getsize(conn_file) > 0:
                try:
                    df = self._read_csv(conn_file)
                except Exception:
                    df = None

            # Fallback to standard connections
            if df is None or df.empty:
                dataset_output_path = self._resolve_dataset_output_path(
                    dataset_name, dataset_threshold)
                conn_file = os.path.join(dataset_output_path, 'connections_edge.csv')
                if os.path.exists(conn_file) and os.path.getsize(conn_file) > 0:
                    try:
                        df = self._read_csv(conn_file)
                    except Exception:
                        df = None

            if (df is None or df.empty):
                # Fallback to data_details
                dataset_output_path = self._resolve_dataset_output_path(
                    dataset_name, dataset_threshold)
                fallback = os.path.join(dataset_output_path, 'data_details', 'connection_type.csv')
                if os.path.exists(fallback) and os.path.getsize(fallback) > 0:
                    try:
                        df = self._read_csv(fallback)
                    except Exception:
                        df = None

            if df is None or df.empty:
                continue

            if 'std_label_pre' in df.columns and 'std_label_post' in df.columns:
                pre_col, post_col = 'std_label_pre', 'std_label_post'
            elif 'type_pre' in df.columns and 'type_post' in df.columns:
                pre_col, post_col = 'type_pre', 'type_post'
            elif 'bodyId_pre' in df.columns and 'bodyId_post' in df.columns:
                pre_col, post_col = 'bodyId_pre', 'bodyId_post'
            else:
                continue

            if 'nt_type_pre' in df.columns:
                nt_col = 'nt_type_pre'
            elif 'nt_type' in df.columns:
                nt_col = 'nt_type'
            else:
                continue

            for _, row in df.iterrows():
                pre_type = str(row[pre_col])
                post_type = str(row[post_col])
                nt_val = row.get(nt_col, None)
                if pd.isna(nt_val):
                    continue
                nt_str = str(nt_val).strip()
                if not nt_str:
                    continue
                
                # Normalize NT string
                nt_upper = nt_str.upper()
                nt_str = self._NT_NORMALIZATION_MAP.get(nt_upper, nt_str.lower())

                if self.parameters.auto_type_mapping and self.parameters._auto_type_mapper:
                    canonical_pre = self._get_canonical_type(pre_type, dataset_name)
                    canonical_post = self._get_canonical_type(post_type, dataset_name)
                else:
                    canonical_pre = pre_type
                    canonical_post = post_type

                edge_key = f"{canonical_pre} -> {canonical_post}"
                if edge_key not in nt_by_edge:
                    nt_by_edge[edge_key] = {}
                
                # Check if we already have a value for this dataset (avoid overwrite if duplicates, or just take first)
                if dataset_name not in nt_by_edge[edge_key]:
                    nt_by_edge[edge_key][dataset_name] = nt_str
        
        return nt_by_edge

    def _get_reciprocal_edge_ratio_data_for_threshold(self, threshold: Any) -> pd.DataFrame:
        """
        Get edge-level connection_ratio data from reciprocal outputs for a threshold.

        Reads from:
        dataset_data/{dataset}/minsyn_{threshold}/find_reciprocal/reciprocal_connection_type.csv
        """
        dataset_names = self.parameters.get_dataset_names()
        all_ratio_data = {}

        for dataset_name in dataset_names:
            dataset_threshold = self._threshold_for_dataset(
                threshold, dataset_name)
            if dataset_threshold is None:
                continue
            safe_name = self.parameters._sanitize_name(dataset_name)
            dataset_output_path = os.path.join(
                self._resolve_dataset_output_path(
                    dataset_name, dataset_threshold),
                'find_reciprocal'
            )

            df = None
            conn_file = os.path.join(dataset_output_path, 'reciprocal_connection_type.csv')
            if os.path.exists(conn_file) and os.path.getsize(conn_file) > 0:
                try:
                    df = self._read_csv(conn_file)
                except Exception:
                    df = None

            if df is None or df.empty or 'connection_ratio' not in df.columns:
                continue

            if 'type_pre' in df.columns and 'type_post' in df.columns:
                df['edge_key'] = df['type_pre'].astype(str) + ' -> ' + df['type_post'].astype(str)
            else:
                continue

            ratio_series = df.groupby('edge_key')['connection_ratio'].mean()
            all_ratio_data[dataset_name] = ratio_series

        if not all_ratio_data:
            return pd.DataFrame()

        result_df = pd.DataFrame(all_ratio_data).fillna(0)
        return result_df

    def _generate_vispath_visualizations(self, vis_dir: str):
        """
        Generate interactive HTML visualizations using VisualizePath.
        
        Note: Interactive heatmaps have been removed (redundant with PNG visualizations
        and comparison_report.html). Network visualizations are embedded directly
        in comparison_report.html using Cytoscape.js.
        
        This method is kept for future extension but currently is a no-op.
        
        Args:
            vis_dir: Directory to save visualization files
        """
        # Note: _generate_vispath_heatmaps and heatmaps/ folder removed as redundant
        # All heatmap needs are served by PNG visualizations in visualizations/ folder
        pass

    def _create_combined_network_html(self, conn_rows: List[Dict], datasets: List[str],
                                       threshold: int, output_path: str, title: str):
        """
        Create a combined network HTML with all datasets side by side.
        
        Uses vis.js with dataset-based coloring to distinguish connections.
        """
        import json
        
        # Build nodes and edges
        nodes = []
        edges = []
        node_ids = {}
        node_counter = 0
        
        # Color palette for datasets
        colors = ['#3b82f6', '#8b5cf6', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4']
        dataset_colors = {ds: colors[i % len(colors)] for i, ds in enumerate(datasets)}
        
        for row in conn_rows:
            source = row['source']
            target = row['target']
            weight = row['weight']
            dataset = row.get('dataset', 'unknown')
            
            # Add source node
            if source not in node_ids:
                node_ids[source] = node_counter
                nodes.append({
                    'id': node_counter, 
                    'label': source, 
                    'group': 'source',
                    'title': f'{source}'
                })
                node_counter += 1
            
            # Add target node  
            if target not in node_ids:
                node_ids[target] = node_counter
                nodes.append({
                    'id': node_counter, 
                    'label': target, 
                    'group': 'target',
                    'title': f'{target}'
                })
                node_counter += 1
            
            # Add edge with dataset color
            edge_color = dataset_colors.get(self.parameters._sanitize_name(dataset), '#64748b')
            edges.append({
                'from': node_ids[source],
                'to': node_ids[target],
                'value': int(weight),
                'title': f"{source} → {target}: {int(weight)} ({dataset})",
                'arrows': 'to',
                'color': {'color': edge_color, 'highlight': edge_color}
            })
        
        # Create legend HTML
        legend_items = ''.join([
            f'<span style="color:{dataset_colors.get(self.parameters._sanitize_name(ds), "#64748b")}; margin-right:15px;">● {self.parameters._sanitize_name(ds)}</span>'
            for ds in datasets
        ])
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        h1 {{ color: #2563eb; }}
        .legend {{ margin: 10px 0; font-size: 14px; }}
        #network {{ width: 100%; height: 700px; border: 1px solid #e2e8f0; border-radius: 8px; }}
        .stats {{ background: #f8fafc; padding: 10px 15px; border-radius: 8px; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="stats">
        <strong>{len(nodes)}</strong> neurons | <strong>{len(edges)}</strong> connections | 
        <strong>{len(datasets)}</strong> datasets
    </div>
    <div class="legend">{legend_items}</div>
    <div id="network"></div>
    <script>
        const nodes = new vis.DataSet({json.dumps(nodes)});
        const edges = new vis.DataSet({json.dumps(edges)});
        const container = document.getElementById('network');
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
            nodes: {{
                shape: 'dot',
                size: 20,
                font: {{ size: 12 }},
                borderWidth: 2
            }},
            edges: {{
                width: 2,
                arrows: {{ to: {{ enabled: true, scaleFactor: 0.8 }} }},
                smooth: {{ type: 'curvedCW', roundness: 0.15 }}
            }},
            groups: {{
                source: {{ color: {{ background: '#3b82f6', border: '#1d4ed8' }} }},
                target: {{ color: {{ background: '#8b5cf6', border: '#6d28d9' }} }},
                intermediate: {{ color: {{ background: '#22c55e', border: '#15803d' }} }}
            }},
            layout: {{
                hierarchical: {{
                    enabled: true,
                    direction: 'LR',
                    sortMethod: 'directed',
                    levelSeparation: 200,
                    nodeSpacing: 80
                }}
            }},
            physics: {{ enabled: false }},
            interaction: {{ hover: true, tooltipDelay: 100 }}
        }};
        new vis.Network(container, data, options);
    </script>
</body>
</html>'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        self._log(f"Saved: {os.path.basename(output_path)}")

    def _create_threshold_comparison_network_html(self, conn_rows: List[Dict], 
                                                   dataset: str, thresholds: List[int],
                                                   output_path: str, title: str):
        """
        Create a network HTML comparing all thresholds for a single dataset.
        
        Uses edge width/opacity to show threshold sensitivity.
        """
        import json
        
        # Aggregate connections across thresholds
        # For each edge, track which thresholds it appears at and max weight
        edge_data = {}
        for row in conn_rows:
            source = row['source']
            target = row['target']
            weight = row['weight']
            threshold = row['threshold']
            
            key = (source, target)
            if key not in edge_data:
                edge_data[key] = {'thresholds': [], 'weights': [], 'max_weight': 0}
            edge_data[key]['thresholds'].append(threshold)
            edge_data[key]['weights'].append(weight)
            edge_data[key]['max_weight'] = max(edge_data[key]['max_weight'], weight)
        
        # Build nodes and edges
        nodes = []
        edges = []
        node_ids = {}
        node_counter = 0
        
        # Color by threshold count (more thresholds = more "conserved")
        num_thresholds = len(thresholds)
        
        for (source, target), data in edge_data.items():
            # Add source node
            if source not in node_ids:
                node_ids[source] = node_counter
                nodes.append({
                    'id': node_counter, 
                    'label': source, 
                    'group': 'source',
                    'title': f'{source}'
                })
                node_counter += 1
            
            # Add target node
            if target not in node_ids:
                node_ids[target] = node_counter
                nodes.append({
                    'id': node_counter, 
                    'label': target, 
                    'group': 'target',
                    'title': f'{target}'
                })
                node_counter += 1
            
            # Edge color based on threshold conservation
            t_count = len(data['thresholds'])
            if t_count == num_thresholds:
                color = '#22c55e'  # Green: in all thresholds
            elif t_count >= num_thresholds * 0.5:
                color = '#f59e0b'  # Orange: in half
            else:
                color = '#ef4444'  # Red: in few
            
            t_list = ', '.join(str(t) for t in sorted(data['thresholds']))
            edges.append({
                'from': node_ids[source],
                'to': node_ids[target],
                'value': int(data['max_weight']),
                'title': f"{source} → {target}\\nMax weight: {int(data['max_weight'])}\\nThresholds: {t_list}",
                'arrows': 'to',
                'color': {'color': color, 'highlight': color}
            })
        
        # Legend
        legend_html = '''
        <span style="color:#22c55e; margin-right:15px;">● All thresholds</span>
        <span style="color:#f59e0b; margin-right:15px;">● Half thresholds</span>
        <span style="color:#ef4444; margin-right:15px;">● Few thresholds</span>
        '''
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        h1 {{ color: #2563eb; }}
        .legend {{ margin: 10px 0; font-size: 14px; }}
        #network {{ width: 100%; height: 700px; border: 1px solid #e2e8f0; border-radius: 8px; }}
        .stats {{ background: #f8fafc; padding: 10px 15px; border-radius: 8px; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="stats">
        <strong>{len(nodes)}</strong> neurons | <strong>{len(edges)}</strong> unique connections | 
        Thresholds: <strong>{', '.join(str(t) for t in thresholds)}</strong>
    </div>
    <div class="legend">{legend_html}</div>
    <div id="network"></div>
    <script>
        const nodes = new vis.DataSet({json.dumps(nodes)});
        const edges = new vis.DataSet({json.dumps(edges)});
        const container = document.getElementById('network');
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
            nodes: {{
                shape: 'dot',
                size: 20,
                font: {{ size: 12 }},
                borderWidth: 2
            }},
            edges: {{
                width: 2,
                arrows: {{ to: {{ enabled: true, scaleFactor: 0.8 }} }},
                smooth: {{ type: 'curvedCW', roundness: 0.15 }}
            }},
            groups: {{
                source: {{ color: {{ background: '#3b82f6', border: '#1d4ed8' }} }},
                target: {{ color: {{ background: '#8b5cf6', border: '#6d28d9' }} }},
                intermediate: {{ color: {{ background: '#22c55e', border: '#15803d' }} }}
            }},
            layout: {{
                hierarchical: {{
                    enabled: true,
                    direction: 'LR',
                    sortMethod: 'directed',
                    levelSeparation: 200,
                    nodeSpacing: 80
                }}
            }},
            physics: {{ enabled: false }},
            interaction: {{ hover: true, tooltipDelay: 100 }}
        }};
        new vis.Network(container, data, options);
    </script>
</body>
</html>'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        self._log(f"Saved: {os.path.basename(output_path)}")

    # =========================================================================
    # Connectivity Profile Comparison
    # =========================================================================
    
    def direct_comparison(
        self,
        neurons_a: Optional[Union[str, int, List[Union[str, int]]]] = None,
        neurons_b: Optional[Union[str, int, List[Union[str, int]]]] = None,
        dataset_a: Optional[str] = None,
        dataset_b: Optional[str] = None,
        direction: Optional[str] = None,
        comparison_mode: str = 'type',
        output_dir: Optional[str] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Direct comparison of specific neurons between datasets.
        
        This is a convenience method for directly comparing neurons by type name
        or bodyId. Uses ProfileComparator.direct_comparison() under the hood.
        
        If neurons_a/neurons_b are not provided, uses source_neurons/target_neurons
        from ComparisonParameters.
        
        Args:
            neurons_a: Neurons to compare from dataset_a (default: params.source_neurons)
            neurons_b: Neurons to compare from dataset_b (default: params.target_neurons)
            dataset_a: First dataset (default: first in params.datasets)
            dataset_b: Second dataset (default: second in params.datasets)
            direction: 'upstream', 'downstream', or 'both' (default: params.verification_direction)
            comparison_mode: 'type' (aggregate) or 'bodyid' (individual)
            output_dir: Where to save results (default: params output folder)
            save_results: If True, save results to CSV
        
        Returns:
            Dict with 'results' DataFrame and 'summary' statistics
        
        Example:
            >>> analyzer = ComparisonAnalyzer(params)
            >>> # Compare specific types
            >>> results = analyzer.direct_comparison('aMe12', 'aMe12')
            >>> # Or use defaults from params
            >>> results = analyzer.direct_comparison()
        """
        from .profile_comparator import ProfileComparator
        from .connectivity_profiler import ConnectivityProfiler, ProfilerConfig
        
        p = self.parameters
        
        # Resolve defaults
        datasets = p.get_dataset_names()
        ds_a = dataset_a or (datasets[0] if len(datasets) > 0 else None)
        ds_b = dataset_b or (datasets[1] if len(datasets) > 1 else ds_a)
        direction = direction or p.verification_direction
        
        # Use source/target neurons if not specified
        if neurons_a is None:
            neurons_a = p.source_neurons
        if neurons_b is None:
            neurons_b = p.target_neurons
        
        # Resolve output directory
        if output_dir is None:
            output_dir = os.path.join(p.full_output_path, "direct_comparison")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create profiler
        config = ProfilerConfig(
            top_k_bodyid=p.verification_top_k,
            top_m_type=p.verification_top_m,
            min_synapse_threshold=p.verification_min_synapse_threshold,
            include_untyped_partners=p.verification_include_untyped,
            use_cache=True
        )
        
        profiler = ConnectivityProfiler(
            datasets=[ds_a, ds_b],
            config=config,
            token=p.resolve_token(),
            verbose=self.verbose
        )
        
        # Run direct comparison
        self._log(f"Running direct comparison: {neurons_a} ({ds_a}) vs {neurons_b} ({ds_b})")
        
        results = ProfileComparator.direct_comparison(
            neurons_a=neurons_a,
            neurons_b=neurons_b,
            dataset_a=ds_a,
            dataset_b=ds_b,
            profiler=profiler,
            direction=direction,
            comparison_mode=comparison_mode,
            label_mapper=self.label_mapper,
            type_mapper=(self.parameters._auto_type_mapper
                         if self.parameters.auto_type_mapping else None),
            score_weights=p.verification_score_weights,
            verbose=self.verbose
        )
        
        # Save results
        if save_results and not results['results'].empty:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(output_dir, f'direct_comparison_{timestamp}.csv')
            self._save_csv(results['results'], filepath)
            self._log(f"Saved: {filepath}")
            results['output_file'] = filepath
        
        return results
    
    def connectivity_profile_comparison(
        self,
        output_dir: Optional[str] = None,
        neuron_types: Optional[List[str]] = None,
        direction: Optional[str] = None,
        comparison_mode: Optional[str] = None,
        include_visualizations: Optional[bool] = None,
        top_k: Optional[int] = None,
        top_m: Optional[int] = None,
        min_synapse_threshold: Optional[int] = None,
        include_untyped_partners: Optional[bool] = None,
        score_weights: Optional[Dict[str, float]] = None,
        _skip_html_regeneration: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Reconstructed connectivity profile comparison.
        
        Workflow:
        1) For each neuron type, run pairwise direct comparisons across all dataset pairs
           using ProfileComparator.direct_comparison (same-label only).
        2) Merge pairwise results into a comparison matrix (type x dataset-pair).
        3) Compute per-type average similarity across all datasets.
        4) Save CSVs and an optional HTML + heatmap visualization.
        
        Only the essential arguments are kept; unused legacy flags were removed.
        """
        # Defaults from parameters
        p = self.parameters
        direction = direction if direction is not None else p.verification_direction
        comparison_mode = comparison_mode if comparison_mode is not None else p.verification_mode
        include_visualizations = include_visualizations if include_visualizations is not None else p.verification_include_visualizations
        top_k = top_k if top_k is not None else p.verification_top_k
        top_m = top_m if top_m is not None else p.verification_top_m
        min_synapse_threshold = min_synapse_threshold if min_synapse_threshold is not None else p.verification_min_synapse_threshold
        include_untyped_partners = include_untyped_partners if include_untyped_partners is not None else p.verification_include_untyped
        score_weights = score_weights if score_weights is not None else p.verification_score_weights

        # Visuals can be suppressed either by caller or via the internal skip flag
        visuals_enabled = bool(include_visualizations and not _skip_html_regeneration)

        # Validate datasets
        dataset_names = self.parameters.get_dataset_names()
        if len(dataset_names) < 2:
            self._log("Need at least 2 datasets for profile comparison")
            return {}

        # Resolve neuron types
        if neuron_types is None:
            src_types, tgt_types, inter_types = self._extract_types_from_results()
            ordered = list(src_types) + [t for t in tgt_types if t not in src_types]
            seen = set(ordered)
            ordered += [t for t in inter_types if t not in seen]
            neuron_types = ordered
        neuron_types = list(dict.fromkeys(neuron_types))  # dedupe, keep order

        if not neuron_types:
            self._log("No neuron types available for profile comparison")
            return {}

        if comparison_mode not in ['loose', 'strict']:
            self._log(f"Warning: Invalid comparison_mode '{comparison_mode}', using 'loose'")
            comparison_mode = 'loose'

        # Output dir
        if output_dir is None:
            output_dir = os.path.join(self.parameters.full_output_path, "connectivity_profile_comparison")
        os.makedirs(output_dir, exist_ok=True)

        # Profiler
        try:
            from .connectivity_profiler import ConnectivityProfiler, ProfilerConfig
            from .profile_comparator import ProfileComparator
        except ImportError as e:
            self._log(f"Warning: Connectivity profile modules not available: {e}")
            return {}

        profiler = ConnectivityProfiler(
            datasets=dataset_names,
            config=ProfilerConfig(
                top_k_bodyid=top_k,
                top_m_type=top_m,
                min_synapse_threshold=min_synapse_threshold,
                include_untyped_partners=include_untyped_partners,
                use_cache=True,
            ),
            token=self.parameters.resolve_token(),
            verbose=self.verbose,
        )

        # Pairwise comparisons
        pairwise_records: List[Dict[str, Any]] = []
        matrix_store: Dict[str, Dict[str, float]] = {t: {} for t in neuron_types}

        for ds_a, ds_b in combinations(dataset_names, 2):
            pair_label = f"{ds_a} vs {ds_b}"
            self._log(f"Comparing {len(neuron_types)} types: {pair_label}")
            try:
                res = ProfileComparator.direct_comparison(
                    neurons_a=neuron_types,
                    neurons_b=neuron_types,
                    dataset_a=ds_a,
                    dataset_b=ds_b,
                    profiler=profiler,
                    direction=direction,
                    comparison_mode=comparison_mode,
                    label_mapper=self.label_mapper,
                    type_mapper=(self.parameters._auto_type_mapper
                                 if self.parameters.auto_type_mapping else None),
                    score_weights=score_weights,
                    top_k=top_k,
                    top_m=top_m,
                    min_synapse_threshold=min_synapse_threshold,
                    include_untyped_partners=include_untyped_partners,
                    same_label_only=True,
                    verbose=self.verbose,
                )
            except Exception as e:
                self._log(f"Direct comparison failed for {pair_label}: {e}")
                continue

            # Prefer type_summary; fallback to results
            type_df = res.get('type_summary')
            if type_df is None or (hasattr(type_df, 'empty') and type_df.empty):
                type_df = res.get('results', pd.DataFrame())

            if type_df is None or type_df.empty:
                self._log(f"No results for {pair_label}")
                continue

            # Collect all available metrics
            metric_map = {
                'avg_rank_corr': 'rank_corr', 'rank_corr': 'rank_corr',
                'avg_rank_union': 'rank_union', 'rank_union': 'rank_union',
                'avg_cosine': 'cosine', 'cosine': 'cosine',
                'avg_jaccard': 'jaccard', 'jaccard': 'jaccard'
            }
            
            # Initialize stores for each metric if not exists
            if not hasattr(self, '_matrix_stores'):
                self._matrix_stores = {m: {t: {} for t in neuron_types} for m in set(metric_map.values())}

            found_any = False
            for col, canonical in metric_map.items():
                if col in type_df.columns:
                    for _, row in type_df.iterrows():
                        ntype = row.get('neuron_type') or row.get('type') or row.get('pair') or row.get('type_a')
                        if pd.isna(ntype):
                            continue
                        val = row.get(col)
                        if pd.isna(val):
                            continue
                        self._matrix_stores[canonical].setdefault(str(ntype), {})[pair_label] = float(val)
                        
                        # Add to pairwise records (only once per canonical metric per pair)
                        # We might overwrite if multiple cols map to same canonical, but that's fine (prefer last/best?)
                        # Actually, let's just append all and filter later or just keep it simple
                        pairwise_records.append({
                            'neuron_type': str(ntype),
                            'dataset_a': ds_a,
                            'dataset_b': ds_b,
                            'metric': canonical,
                            'value': float(val),
                        })
                    found_any = True
            
            if not found_any:
                self._log(f"No similarity columns found for {pair_label}")

        if not pairwise_records:
            self._log("No pairwise comparison results produced")
            return {}

        # Build matrix DataFrames for each metric
        matrices = {}
        pair_cols = [f"{a} vs {b}" for a, b in combinations(dataset_names, 2)]
        
        for metric, store in self._matrix_stores.items():
            # Check if store has any data
            has_data = any(store.values())
            if not has_data:
                continue
                
            df = pd.DataFrame(store).T.reset_index().rename(columns={'index': 'neuron_type'})
            existing_cols = [c for c in pair_cols if c in df.columns]
            if not existing_cols:
                continue
                
            df = df[['neuron_type'] + existing_cols]
            matrices[metric] = df

        if not matrices:
            self._log("No valid matrices could be built")
            return {}

        # Use 'rank_corr' as primary for sorting/display when present
        primary_metric = 'rank_corr' if 'rank_corr' in matrices else list(matrices.keys())[0]
        
        # Create summary DataFrame with averages for ALL metrics
        # Start with all unique neuron types across all matrices
        all_types = set()
        for df in matrices.values():
            all_types.update(df['neuron_type'].tolist())
        
        summary_df = pd.DataFrame({'neuron_type': sorted(list(all_types))})
        
        # Calculate and merge averages for each metric
        for metric, df in matrices.items():
            value_cols = [c for c in df.columns if c != 'neuron_type']
            # Calculate mean for this metric
            avg_series = df.set_index('neuron_type')[value_cols].mean(axis=1, skipna=True)
            avg_df = avg_series.reset_index().rename(columns={0: f'avg_{metric}'})
            summary_df = pd.merge(summary_df, avg_df, on='neuron_type', how='left')
            
        # Add n_pairs count (using primary metric)
        primary_df = matrices[primary_metric]
        primary_cols = [c for c in primary_df.columns if c != 'neuron_type']
        count_series = primary_df.set_index('neuron_type')[primary_cols].count(axis=1)
        count_df = count_series.reset_index().rename(columns={0: 'n_pairs'})
        summary_df = pd.merge(summary_df, count_df, on='neuron_type', how='left')
        
        # Sort by primary metric average
        if f'avg_{primary_metric}' in summary_df.columns:
            summary_df = summary_df.sort_values(f'avg_{primary_metric}', ascending=False)

        pairwise_df = pd.DataFrame(pairwise_records)

        # Save outputs
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # Save all matrices
        for metric, df in matrices.items():
            path = os.path.join(output_dir, f'comparison_matrix_{metric}_{timestamp}.csv')
            self._save_csv(df, path)
            
        summary_path = os.path.join(output_dir, f'comparison_summary_{timestamp}.csv')
        pairwise_path = os.path.join(output_dir, f'pairwise_results_{timestamp}.csv')
        self._save_csv(summary_df, summary_path)
        self._save_csv(pairwise_df, pairwise_path)
        self._log(f"Saved matrices, summary, and pairwise results to {output_dir}")

        report_path = None
        heatmap_paths = {}

        # Optional visualization
        if visuals_enabled:
            try:
                import matplotlib.pyplot as plt
                
                for metric, df in matrices.items():
                    cols = [c for c in df.columns if c != 'neuron_type']
                    if not cols:
                        continue
                        
                    fig, ax = plt.subplots(figsize=(max(6, len(cols)*0.8), max(6, len(df)*0.25)))
                    heatmap_data = df.set_index('neuron_type')[cols]
                    
                    # Determine range and colormap based on data
                    data_min = heatmap_data.min().min()
                    if data_min < 0:
                        # Use diverging colormap for data with negative values (e.g. correlations)
                        cmap = 'RdBu_r'
                        vmin = -1
                        vmax = 1
                    else:
                        # Use sequential colormap for positive-only data (e.g. Jaccard)
                        cmap = 'viridis'
                        vmin = 0
                        vmax = 1
                        
                    im = ax.imshow(heatmap_data.fillna(np.nan), aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_xticks(range(len(cols)))
                    ax.set_xticklabels(cols, rotation=45, ha='right')
                    ax.set_yticks(range(len(heatmap_data.index)))
                    ax.set_yticklabels(heatmap_data.index)
                    ax.set_title(f'Connectivity Profile Similarity ({metric})')
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    
                    path = os.path.join(output_dir, f'comparison_heatmap_{metric}_{timestamp}.png')
                    fig.tight_layout()
                    fig.savefig(path, dpi=200)
                    plt.close(fig)
                    heatmap_paths[metric] = path
                    self._log(f"Saved heatmap: {path}")
            except Exception as e:
                self._log(f"Heatmap generation failed: {e}")

            # Interactive Heatmap
            interactive_path = None
            try:
                interactive_path = os.path.join(output_dir, f'comparison_interactive_{timestamp}.html')
                interactive_matrices = {}
                for metric, df in matrices.items():
                    if 'neuron_type' in df.columns:
                        interactive_matrices[metric] = df.set_index('neuron_type')
                    else:
                        interactive_matrices[metric] = df.copy()
                
                generate_interactive_heatmap(
                    interactive_matrices, 
                    interactive_path, 
                    title=f"Connectivity Profile Comparison ({timestamp})",
                    showfig=False,
                    verbose=self.verbose
                )
                self._log(f"Saved interactive heatmap: {interactive_path}")
            except Exception as e:
                self._log(f"Interactive heatmap generation failed: {e}")

            # HTML report
            try:
                report_path = os.path.join(output_dir, f'comparison_report_{timestamp}.html')
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write("<html><head><title>Connectivity Profile Comparison</title>")
                    f.write("<style>body{font-family:sans-serif; margin:20px;} table{border-collapse:collapse; width:100%;} th,td{border:1px solid #ddd; padding:8px; text-align:left;} th{background-color:#f2f2f2;} img{max-width:100%; height:auto; margin-bottom:20px;}</style>")
                    f.write("</head><body>")
                    f.write("<h2>Connectivity Profile Comparison</h2>")
                    
                    if interactive_path:
                        f.write(f"<p><a href='{os.path.basename(interactive_path)}' target='_blank' style='font-size:16px; font-weight:bold; color:#4CAF50;'>👉 Open Interactive Heatmap</a></p>")
                    
                    f.write("<h3>Summary (Primary Metric: " + primary_metric + ")</h3>")
                    f.write(summary_df.to_html(index=False))
                    
                    # Heatmaps section
                    f.write("<h3>Similarity Heatmaps</h3>")
                    for metric, path in heatmap_paths.items():
                        f.write(f"<h4>{metric}</h4>")
                        f.write(f"<p><img src='{os.path.basename(path)}'/></p>")
                    
                    # Matrices section
                    f.write("<h3>Similarity Matrices</h3>")
                    for metric, df in matrices.items():
                        f.write(f"<h4>{metric}</h4>")
                        f.write(df.to_html(index=False))
                        
                    f.write("<h3>Pairwise Records</h3>")
                    f.write(pairwise_df.head(2000).to_html(index=False))
                    f.write("</body></html>")
                self._log(f"Saved HTML report: {report_path}")
            except Exception as e:
                self._log(f"HTML report generation failed: {e}")

        return {
            'matrix': matrices.get(primary_metric),
            'matrices': matrices,
            'summary': summary_df,
            'pairwise_results': pairwise_df,
            'comparison_mode': comparison_mode,
            'report_path': report_path,
            'heatmap_path': heatmap_paths.get(primary_metric),
            'heatmap_paths': heatmap_paths,
            'include_visualizations': visuals_enabled,
        }
    
    # Alias for backward compatibility
    def run_connectivity_profile_verification(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Alias for connectivity_profile_comparison() for backward compatibility.
        
        Deprecated: Use connectivity_profile_comparison() instead.
        """
        # Remove arguments that were removed from the main method
        kwargs.pop('parallel', None)
        kwargs.pop('max_workers', None)
        return self.connectivity_profile_comparison(**kwargs)
    

    def _extract_types_from_results(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Extract source, target, and intermediate neuron types from comparison results.
        
        Returns:
            Tuple of (source_types, target_types, intermediate_types)
        """
        source_types = set()
        target_types = set()
        intermediate_types = set()
        
        # Get from parameters
        source_neurons = self.parameters.source_neurons
        target_neurons = self.parameters.target_neurons
        
        # Process source neurons - extract concrete types if possible
        for sn in source_neurons:
            if isinstance(sn, str) and '.*' not in sn and '*' not in sn:
                source_types.add(sn)
        
        # Process target neurons
        for tn in target_neurons:
            if isinstance(tn, str) and '.*' not in tn and '*' not in tn:
                target_types.add(tn)
        
        # Extract from raw results if available
        for dataset_name, threshold_results in self.raw_results.items():
            for threshold, df in threshold_results.items():
                if df.empty:
                    continue
                
                # Source types from type_pre column
                if 'type_pre' in df.columns:
                    for t in df['type_pre'].dropna().unique():
                        t_str = str(t)
                        if t_str and t_str != 'nan':
                            # Check if it matches source pattern
                            for sn in source_neurons:
                                if self._type_matches_pattern(t_str, sn):
                                    source_types.add(t_str)
                                    break
                            else:
                                # Not a source, might be intermediate
                                for tn in target_neurons:
                                    if self._type_matches_pattern(t_str, tn):
                                        target_types.add(t_str)
                                        break
                                else:
                                    intermediate_types.add(t_str)
                
                # Target types from type_post column
                if 'type_post' in df.columns:
                    for t in df['type_post'].dropna().unique():
                        t_str = str(t)
                        if t_str and t_str != 'nan':
                            for tn in target_neurons:
                                if self._type_matches_pattern(t_str, tn):
                                    target_types.add(t_str)
                                    break
                            else:
                                # Check if source
                                for sn in source_neurons:
                                    if self._type_matches_pattern(t_str, sn):
                                        source_types.add(t_str)
                                        break
                                else:
                                    intermediate_types.add(t_str)
        
        return list(source_types), list(target_types), list(intermediate_types)
    
    def _type_matches_pattern(self, type_name: str, pattern: Union[str, int]) -> bool:
        """Check if type_name matches the pattern (supports regex patterns)."""
        import re
        
        if isinstance(pattern, int):
            return str(pattern) == type_name
        
        if '.*' in pattern or '*' in pattern:
            regex_pattern = _wildcard_pattern_to_regex(pattern)
            try:
                return bool(re.match(f'^{regex_pattern}$', type_name))
            except re.error:
                return pattern == type_name
        
        return pattern == type_name
    
    # =========================================================================
    # Conserved Path Visualization
    # =========================================================================
    
    def visualize_conserved_paths(
        self,
        threshold: Optional[Any] = None,
        trim_dead_ends: bool = True,
        output_folder: Optional[str] = None,
        showfig: bool = False,
        network_layout: str = 'hierarchical',
        edge_width_scale: str = 'log',
        **vispath_kwargs
    ) -> Optional[str]:
        """
        Visualize conserved edges/paths across all datasets using VisualizePath.
        
        Creates standalone network visualizations showing only edges that are
        conserved (present in all datasets), with synapse strengths from each
        dataset shown in the hover labels.
        
        Args:
            threshold: Weight threshold to use. If None, uses middle threshold.
            trim_dead_ends: If True, removes nodes that don't connect source to target.
                          Edges leading to dead-ends are removed.
            output_folder: Output folder for visualizations. If None, uses
                          {comparison_output}/comparison_visualizations/conserved_paths/
            showfig: If True, opens the visualization in browser.
            network_layout: Layout algorithm ('hierarchical', 'spring', 'circular').
            edge_width_scale: Edge width scaling ('log', 'linear', 'sqrt', 'none').
            **vispath_kwargs: Additional keyword arguments for VisualizePath.
            
        Returns:
            Path to the generated HTML file, or None if no conserved edges found.
            
        Example:
            >>> analyzer = ComparisonAnalyzer(params)
            >>> analyzer.run_comparison()
            >>> analyzer.visualize_conserved_paths(threshold=5, trim_dead_ends=True)
        """
        # Import VisualizePath
        try:
            from vispath_pkg import VisualizePath
        except ImportError:
            self._log("Warning: vispath_pkg not available. Cannot generate conserved path visualization.")
            return None
        
        # Ensure comparison has been run
        if not self.raw_results:
            self._log("Warning: No comparison results. Run run_comparison() first.")
            return None
        
        dataset_names = self.parameters.get_dataset_names()
        dataset_nickname_map = self.parameters.get_nickname_map()
        
        query = None
        if self.parameters.threshold_mode == 'combinations':
            if threshold is None:
                queries = self.get_threshold_queries()
                threshold = queries[len(queries) // 2] if queries else None
            if threshold is not None:
                query = self._query_record(threshold)

        # Determine threshold
        if threshold is None:
            threshold = self.parameters.thresholds[len(self.parameters.thresholds) // 2]

        display_threshold = query.get('id') if query else threshold
        self._log(
            f"Generating conserved path visualization @ threshold="
            f"{display_threshold}...")

        # Get aligned data
        aligned = (self.get_aligned_data_for_query(query)
                   if query else self.get_aligned_data(threshold))
        if aligned.empty:
            self._log("Warning: No aligned data at this threshold.")
            return None
        
        # Find conserved edges (present in ALL datasets)
        available_ds = [d for d in dataset_names if d in aligned.columns]
        if not available_ds:
            self._log("Warning: No datasets with aligned data.")
            return None
        
        # Mask for edges present in all datasets
        mask_all = (aligned[available_ds] > 0).all(axis=1)
        conserved_edges = aligned[mask_all].copy()
        
        if conserved_edges.empty:
            self._log("Warning: No conserved edges found at this threshold.")
            return None
        
        self._log(f"  Found {len(conserved_edges)} conserved edges")
        
        # Get ratio and probability data for enhanced hover labels
        ratio_data = self._get_edge_ratio_data_for_threshold(
            query or threshold)
        # Note: prob_data is path-level, not edge-level. We'll include it if edge matches
        
        # Get NT details for all edges
        nt_details = self._get_edge_nt_details(query or threshold)
        
        # Build edge list with weights from each dataset
        edge_list = []
        edge_labels = {}  # {(source, target): {dataset: {'weight': w, 'ratio': r}, ...}}
        
        for edge_key, row in conserved_edges.iterrows():
            # Parse source/target from edge key
            if ' -> ' in str(edge_key):
                parts = str(edge_key).split(' -> ')
                source = parts[0].strip()
                target = parts[1].strip() if len(parts) > 1 else ''
            else:
                continue
            
            # Get weights and ratios from all datasets
            # Format for vispath edge_labels: {key: value} where key-value pairs are shown in tooltip
            edge_info = {}  # Will contain: {'MCNS weight': 5, 'MCNS ratio': 0.01, ...}
            avg_weight = 0
            
            # Get NT info for this edge if available
            edge_nts = nt_details.get(str(edge_key), {})
            unique_nts = set(v for v in edge_nts.values() if v)
            nt_consensus = next(iter(unique_nts)) if len(unique_nts) == 1 else None
            
            for dataset in available_ds:
                weight = row[dataset]
                if weight > 0:
                    # Use the collision-aware nickname map so two releases
                    # from the same dataset family remain distinguishable.
                    nickname = dataset_nickname_map.get(
                        dataset, self.parameters._sanitize_name(dataset)
                    )
                    
                    # Add weight for this dataset
                    edge_info[f'{nickname} wt'] = int(weight)
                    
                    # Try to get ratio data for this edge
                    if not ratio_data.empty and dataset in ratio_data.columns:
                        # Look for matching edge in ratio_data
                        edge_key_for_ratio = str(edge_key)
                        if edge_key_for_ratio in ratio_data.index:
                            ratio_val = ratio_data.loc[edge_key_for_ratio, dataset]
                            # Handle case where loc returns a Series (duplicate indices)
                            if isinstance(ratio_val, pd.Series):
                                ratio_val = ratio_val.iloc[0]
                            if ratio_val > 0:
                                edge_info[f'{nickname} ratio'] = round(ratio_val, 4)
                    
                    # Add NT info for this dataset if available
                    if dataset in edge_nts:
                        edge_info[f'{nickname} nt'] = edge_nts[dataset]
                    
                    avg_weight += weight
            
            avg_weight = avg_weight / len(available_ds) if available_ds else 0
            
            edge_data = {
                'source': source,
                'target': target,
                'weight': avg_weight,  # Use average weight for visualization
            }
            if nt_consensus:
                edge_data['nt_type'] = nt_consensus
                
            edge_list.append(edge_data)
            edge_labels[(source, target)] = edge_info
        
        if not edge_list:
            self._log("Warning: No valid edges to visualize.")
            return None
        
        # Convert to DataFrame
        edges_df = pd.DataFrame(edge_list)
        
        # Enable NT coloring if we have NT types
        if 'nt_type' in edges_df.columns and not edges_df['nt_type'].isna().all():
            if 'color_edges_by_nt' not in vispath_kwargs:
                vispath_kwargs['color_edges_by_nt'] = True

        # Helper to extract canonical name from display name
        def get_canonical_name(display_name: str) -> str:
            if '(' in display_name:
                return display_name.split('(')[0].strip()
            return display_name
        
        # Transform node labels to display format with dataset-specific names
        # Format: {canonical}({alt1}/{alt2}) for types that differ across datasets
        type_mapper = self.parameters._auto_type_mapper if self.parameters.auto_type_mapping else None
        display_name_map = {}  # {canonical_name: display_name}
        node_dataset_info = {}  # {display_name: {code: name_in_that_dataset}} for hover labels
        dataset_legend = {}  # {short_code: full_dataset_name} for legend
        
        if type_mapper:
            # Get all unique nodes and compute display names + dataset info for hover
            all_unique_nodes = set(edges_df['source'].unique()) | set(edges_df['target'].unique())
            for node in all_unique_nodes:
                display_name, ds_info = type_mapper.get_display_name_with_dataset_info(node, dataset_names)
                if display_name != node:
                    display_name_map[node] = display_name
                if ds_info:
                    node_dataset_info[display_name] = ds_info
            
            # Get dataset legend info
            dataset_legend = type_mapper.get_all_dataset_short_codes(dataset_names)
            
            # Apply display name transformation to edges_df
            if display_name_map:
                edges_df['source'] = edges_df['source'].apply(lambda x: display_name_map.get(x, x))
                edges_df['target'] = edges_df['target'].apply(lambda x: display_name_map.get(x, x))
                
                # Update edge_labels keys to use display names
                new_edge_labels = {}
                for (src, tgt), info in edge_labels.items():
                    new_src = display_name_map.get(src, src)
                    new_tgt = display_name_map.get(tgt, tgt)
                    new_edge_labels[(new_src, new_tgt)] = info
                edge_labels = new_edge_labels
                
                self._log(f"  Applied display names to {len(display_name_map)} nodes with cross-dataset name differences")

        # Identify source and target nodes from parameters
        # CRITICAL: Include ALL mapped type names across all datasets, not just original query types
        source_patterns = set(self.parameters._ensure_flat_list(self.parameters.source_neurons))
        target_patterns = set(self.parameters._ensure_flat_list(self.parameters.target_neurons))
        
        # If auto_type_mapping is enabled, also include resolved type names for each dataset
        if self.parameters.auto_type_mapping:
            for dataset in dataset_names:
                src_resolved = self.parameters.get_source_neurons_for_dataset(dataset)
                source_patterns.update(src_resolved)
                tgt_resolved = self.parameters.get_target_neurons_for_dataset(dataset)
                target_patterns.update(tgt_resolved)
        
        # Convert back to list for matching function
        source_patterns = list(source_patterns)
        target_patterns = list(target_patterns)
        
        # Get all unique nodes
        all_nodes = set(edges_df['source'].unique()) | set(edges_df['target'].unique())
        
        # Helper to extract base name without hemisphere suffix
        def get_base_name(label: str) -> str:
            """Extract base name from label, removing hemisphere suffix like _L, _R, _U."""
            base = get_canonical_name(label)
            if base.endswith(('_L', '_R', '_U')):
                return base[:-2]
            return base
        
        # Classify nodes as source, target, or intermediate
        import re
        separate_hemispheres = bool(getattr(self.parameters, 'separate_hemispheres', False))
        
        def matches_patterns(node: str, patterns: list) -> bool:
            """Check if node matches any pattern. Handles merged display names and hemisphere suffixes."""
            # Get canonical name for matching (handles merged display names)
            canonical = get_canonical_name(node)
            base = get_base_name(node)
            # Check the full label, canonical name, and base name
            names_to_check = list(set([node, canonical, base]))
            
            for name in names_to_check:
                for pattern in patterns:
                    if isinstance(pattern, str):
                        # Handle regex patterns
                        if '.*' in pattern or '*' in pattern:
                            regex = _wildcard_pattern_to_regex(pattern)
                            if re.match(f'^{regex}$', name, re.IGNORECASE):
                                return True
                        elif name.lower() == pattern.lower():
                            return True
                        # If separating hemispheres, allow exact base name to match suffixed labels
                        elif separate_hemispheres and '.*' not in pattern and '*' not in pattern:
                            if name.lower().startswith(pattern.lower() + '_'):
                                suffix = name[len(pattern) + 1:]
                                if suffix.upper() in ('L', 'R', 'U'):
                                    return True
            return False
        
        source_nodes = {n for n in all_nodes if matches_patterns(n, source_patterns)}
        target_nodes = {n for n in all_nodes if matches_patterns(n, target_patterns)}
        intermediate_nodes = all_nodes - source_nodes - target_nodes
        
        # Trim dead-ends if requested
        if trim_dead_ends and source_nodes and target_nodes:
            self._log("  Trimming dead-end nodes...")
            
            # Build adjacency for reachability analysis
            from collections import defaultdict
            forward_adj = defaultdict(set)  # node -> downstream nodes
            backward_adj = defaultdict(set)  # node -> upstream nodes
            
            for _, row in edges_df.iterrows():
                forward_adj[row['source']].add(row['target'])
                backward_adj[row['target']].add(row['source'])
            
            # Find nodes reachable from sources (forward)
            reachable_from_source = set()
            from collections import deque
            queue = deque(source_nodes)
            while queue:
                node = queue.popleft()
                if node in reachable_from_source:
                    continue
                reachable_from_source.add(node)
                for next_node in forward_adj[node]:
                    if next_node not in reachable_from_source:
                        queue.append(next_node)
            
            # Find nodes that can reach targets (backward)
            can_reach_target = set()
            queue = deque(target_nodes)
            while queue:
                node = queue.popleft()
                if node in can_reach_target:
                    continue
                can_reach_target.add(node)
                for prev_node in backward_adj[node]:
                    if prev_node not in can_reach_target:
                        queue.append(prev_node)
            
            # Keep only nodes that are on paths from source to target
            valid_nodes = reachable_from_source & can_reach_target
            
            # Filter edges to only include valid nodes
            edges_df = edges_df[
                edges_df['source'].isin(valid_nodes) & 
                edges_df['target'].isin(valid_nodes)
            ].copy()
            
            # Filter edge labels
            edge_labels = {
                k: v for k, v in edge_labels.items() 
                if k[0] in valid_nodes and k[1] in valid_nodes
            }
            
            removed_count = len(all_nodes) - len(valid_nodes)
            if removed_count > 0:
                self._log(f"  Removed {removed_count} dead-end nodes, {len(edges_df)} edges remaining")
        
        if edges_df.empty:
            self._log("Warning: No edges remaining after trimming dead-ends.")
            return None
        
        # Setup output path (at root level, parallel to comparison_report.html)
        if output_folder is None:
            output_folder = os.path.join(
                self.parameters.full_output_path,
                "conserved_paths"
            )
        os.makedirs(output_folder, exist_ok=True)
        
        filename_threshold = (
            self._safe_query_filename_id(display_threshold)
            if query else display_threshold
        )
        # B7: query-id files drop the legacy literal `t` prefix (a query
        # slug already self-identifies); standard numeric thresholds keep
        # the historical `t19` form.
        base_filename = (
            f"conserved_network_{filename_threshold}"
            if query else f"conserved_network_t{filename_threshold}"
        )

        self._log(f"  Creating VisualizePath visualization with {len(edges_df)} edges...")
        
        # Create VisualizePath with conserved edges
        vp = VisualizePath(
            path_file=edges_df,
            output_folder=output_folder,
            showfig=showfig,
            network_layout=network_layout,
            edge_width_scale=edge_width_scale,
            edge_labels=edge_labels,  # Multi-dataset synapse strengths
            dataset_legend=dataset_legend,  # Dataset short code legend for display names
            node_dataset_info=node_dataset_info,  # Node-level dataset info for hover labels
            verbose=self.verbose,
            separate_hemispheres=self.parameters.separate_hemispheres,
            **vispath_kwargs
        )
        
        # Override base filename
        vp.base_filename = base_filename
        
        # Build network and create visualization
        vp.build_network()
        
        # Set node types for coloring
        for node in vp.G_network.nodes():
            if node in source_nodes:
                vp.G_network.nodes[node]['node_type'] = 'source'
            elif node in target_nodes:
                vp.G_network.nodes[node]['node_type'] = 'target'
            else:
                vp.G_network.nodes[node]['node_type'] = 'intermediate'
        
        # Generate network visualization
        output_path = vp.create_network()
        output_path_heatmap = vp.create_heatmap()
        
        self._log_file(output_path, "Saved conserved path visualization")
        
        return output_path
    
    def visualize_conserved_paths_all_thresholds(
        self,
        trim_dead_ends: bool = True,
        output_folder: Optional[str] = None,
        showfig: bool = False,
        **vispath_kwargs
    ) -> List[str]:
        """
        Generate conserved path visualizations for all thresholds.
        
        Creates a subfolder 'conserved_paths/' containing one network 
        visualization per threshold, each showing only edges conserved
        across all datasets with multi-dataset synapse strengths.
        
        Args:
            trim_dead_ends: If True, removes dead-end nodes.
            output_folder: Output folder for visualizations. If None, uses
                          {comparison_output}/comparison_visualizations/conserved_paths/
            showfig: If True, opens visualizations in browser.
            **vispath_kwargs: Additional keyword arguments for VisualizePath.
            
        Returns:
            List of paths to generated HTML files.
        """
        # Set up output folder for all thresholds (at root level, parallel to comparison_report.html)
        if output_folder is None:
            output_folder = os.path.join(
                self.parameters.full_output_path,
                "conserved_paths"
            )
        os.makedirs(output_folder, exist_ok=True)
        
        if self.parameters.threshold_mode == 'combinations':
            thresholds_to_render = self.get_threshold_queries()
        else:
            thresholds_to_render = self.parameters.thresholds
        self._log(
            f"Generating conserved path visualizations for "
            f"{len(thresholds_to_render)} threshold queries...")
        
        output_paths = []
        
        for threshold in thresholds_to_render:
            result = self.visualize_conserved_paths(
                threshold=threshold,
                trim_dead_ends=trim_dead_ends,
                output_folder=output_folder,  # Use shared folder
                showfig=showfig,
                **vispath_kwargs
            )
            if result:
                output_paths.append(result)
        
        if output_paths:
            self._log(f"  Generated {len(output_paths)} conserved path visualizations in: conserved_paths/")
        
        return output_paths

    # =========================================================================
    # Conserved Reciprocal Graph Visualization
    # =========================================================================

    def visualize_conserved_reciprocal_graph(
        self,
        threshold: Optional[Any] = None,
        trim_dead_ends: bool = True,
        output_folder: Optional[str] = None,
        showfig: bool = False,
        network_layout: str = 'hierarchical',
        edge_width_scale: str = 'log',
        **vispath_kwargs
    ) -> Optional[str]:
        """Visualize conserved edges from reciprocal graphs across datasets."""
        try:
            from vispath_pkg import VisualizePath
        except ImportError:
            self._log("Warning: vispath_pkg not available. Cannot generate conserved reciprocal visualization.")
            return None

        if not self.raw_results:
            self._log("Warning: No comparison results. Run run_comparison() first.")
            return None

        dataset_names = self.parameters.get_dataset_names()
        dataset_nickname_map = self.parameters.get_nickname_map()

        query = None
        if self.parameters.threshold_mode == 'combinations':
            if threshold is None:
                queries = self.get_threshold_queries()
                threshold = queries[len(queries) // 2] if queries else None
            if threshold is not None:
                query = self._query_record(threshold)

        if threshold is None:
            threshold = self.parameters.thresholds[len(self.parameters.thresholds) // 2]

        display_threshold = query.get('id') if query else threshold
        self._log(
            f"Generating conserved reciprocal graph @ threshold="
            f"{display_threshold}...")

        aligned = (self.get_aligned_data_for_network(query)
                   if query else self.get_aligned_data_for_network(threshold))
        if aligned.empty:
            self._log("Warning: No aligned reciprocal data at this threshold.")
            return None

        available_ds = [d for d in dataset_names if d in aligned.columns]
        if not available_ds:
            self._log("Warning: No datasets with aligned reciprocal data.")
            return None

        mask_all = (aligned[available_ds] > 0).all(axis=1)
        conserved_edges = aligned[mask_all].copy()
        if conserved_edges.empty:
            self._log("Warning: No conserved reciprocal edges found at this threshold.")
            return None

        self._log(f"  Found {len(conserved_edges)} conserved reciprocal edges")

        ratio_data = self._get_reciprocal_edge_ratio_data_for_threshold(
            query or threshold)

        edge_list = []
        edge_labels = {}

        for edge_key, row in conserved_edges.iterrows():
            if ' -> ' in str(edge_key):
                parts = str(edge_key).split(' -> ')
                source = parts[0].strip()
                target = parts[1].strip() if len(parts) > 1 else ''
            else:
                continue

            edge_info = {}
            avg_weight = 0
            for dataset in available_ds:
                weight = row[dataset]
                if weight > 0:
                    # Keep release-qualified aliases unique in reciprocal
                    # graph hover labels as well.
                    nickname = dataset_nickname_map.get(
                        dataset, self.parameters._sanitize_name(dataset)
                    )
                    edge_info[f'{nickname} wt'] = int(weight)

                    if not ratio_data.empty and dataset in ratio_data.columns:
                        edge_key_for_ratio = str(edge_key)
                        if edge_key_for_ratio in ratio_data.index:
                            ratio_val = ratio_data.loc[edge_key_for_ratio, dataset]
                            if isinstance(ratio_val, pd.Series):
                                ratio_val = ratio_val.iloc[0]
                            if ratio_val > 0:
                                edge_info[f'{nickname} ratio'] = round(ratio_val, 4)

                    avg_weight += weight

            avg_weight = avg_weight / len(available_ds) if available_ds else 0

            edge_list.append({'source': source, 'target': target, 'weight': avg_weight})
            edge_labels[(source, target)] = edge_info

        if not edge_list:
            self._log("Warning: No valid edges to visualize.")
            return None

        edges_df = pd.DataFrame(edge_list)

        source_patterns = set(self.parameters._ensure_flat_list(self.parameters.source_neurons))
        target_patterns = set(self.parameters._ensure_flat_list(self.parameters.target_neurons))
        if self.parameters.auto_type_mapping:
            for dataset in dataset_names:
                source_patterns.update(self.parameters.get_source_neurons_for_dataset(dataset))
                target_patterns.update(self.parameters.get_target_neurons_for_dataset(dataset))
        source_patterns = list(source_patterns)
        target_patterns = list(target_patterns)

        all_nodes = set(edges_df['source'].unique()) | set(edges_df['target'].unique())

        import re
        separate_hemispheres = bool(getattr(self.parameters, 'separate_hemispheres', False))

        def get_canonical_name(display_name: str) -> str:
            """Extract canonical name from display name with potential annotations."""
            if '(' in display_name:
                return display_name.split('(')[0].strip()
            return display_name.strip()

        def get_base_name(label: str) -> str:
            base = get_canonical_name(label)
            if base.endswith(('_L', '_R', '_U')):
                return base[:-2]
            return base

        def matches_patterns(node: str, patterns: list) -> bool:
            canonical = get_canonical_name(node)
            base = get_base_name(node)
            names_to_check = list(set([node, canonical, base]))
            for name in names_to_check:
                for pattern in patterns:
                    if isinstance(pattern, str):
                        if '.*' in pattern or '*' in pattern:
                            regex = _wildcard_pattern_to_regex(pattern)
                            if re.match(f'^{regex}$', name, re.IGNORECASE):
                                return True
                        elif name.lower() == pattern.lower():
                            return True
                        elif separate_hemispheres and '.*' not in pattern and '*' not in pattern:
                            if name.lower().startswith(pattern.lower() + '_'):
                                suffix = name[len(pattern) + 1:]
                                if suffix.upper() in ('L', 'R', 'U'):
                                    return True
            return False

        source_nodes = {n for n in all_nodes if matches_patterns(n, source_patterns)}
        target_nodes = {n for n in all_nodes if matches_patterns(n, target_patterns)}
        intermediate_nodes = all_nodes - source_nodes - target_nodes

        if trim_dead_ends and source_nodes and target_nodes:
            from collections import defaultdict
            forward_adj = defaultdict(set)
            backward_adj = defaultdict(set)
            for _, row in edges_df.iterrows():
                forward_adj[row['source']].add(row['target'])
                backward_adj[row['target']].add(row['source'])

            reachable_from_source = set()
            from collections import deque
            queue = deque(source_nodes)
            while queue:
                node = queue.popleft()
                if node in reachable_from_source:
                    continue
                reachable_from_source.add(node)
                for next_node in forward_adj[node]:
                    if next_node not in reachable_from_source:
                        queue.append(next_node)

            can_reach_target = set()
            queue = deque(target_nodes)
            while queue:
                node = queue.popleft()
                if node in can_reach_target:
                    continue
                can_reach_target.add(node)
                for prev_node in backward_adj[node]:
                    if prev_node not in can_reach_target:
                        queue.append(prev_node)

            valid_nodes = reachable_from_source & can_reach_target
            edges_df = edges_df[
                edges_df['source'].isin(valid_nodes) &
                edges_df['target'].isin(valid_nodes)
            ].copy()
            edge_labels = {
                k: v for k, v in edge_labels.items()
                if k[0] in valid_nodes and k[1] in valid_nodes
            }

        if edges_df.empty:
            self._log("Warning: No edges remaining after trimming dead-ends.")
            return None

        if output_folder is None:
            output_folder = os.path.join(
                self.parameters.full_output_path,
                "conserved_reciprocal_graph"
            )
        os.makedirs(output_folder, exist_ok=True)

        filename_threshold = (
            self._safe_query_filename_id(display_threshold)
            if query else display_threshold
        )
        base_filename = (
            f"conserved_reciprocal_{filename_threshold}"
            if query else f"conserved_reciprocal_t{filename_threshold}"
        )

        vp = VisualizePath(
            path_file=edges_df,
            output_folder=output_folder,
            showfig=showfig,
            network_layout=network_layout,
            edge_width_scale=edge_width_scale,
            edge_labels=edge_labels,
            verbose=self.verbose,
            separate_hemispheres=self.parameters.separate_hemispheres,
            **vispath_kwargs
        )

        vp.base_filename = base_filename
        vp.build_network()

        for node in vp.G_network.nodes():
            if node in source_nodes:
                vp.G_network.nodes[node]['node_type'] = 'source'
            elif node in target_nodes:
                vp.G_network.nodes[node]['node_type'] = 'target'
            else:
                vp.G_network.nodes[node]['node_type'] = 'intermediate'

        output_path = vp.create_network()
        self._log_file(output_path, "Saved conserved reciprocal graph")

        return output_path

    def visualize_conserved_reciprocal_graph_all_thresholds(
        self,
        trim_dead_ends: bool = True,
        output_folder: Optional[str] = None,
        showfig: bool = False,
        **vispath_kwargs
    ) -> List[str]:
        """Generate conserved reciprocal graph visualizations for all thresholds."""
        if output_folder is None:
            output_folder = os.path.join(
                self.parameters.full_output_path,
                "conserved_reciprocal_graph"
            )
        os.makedirs(output_folder, exist_ok=True)

        if self.parameters.threshold_mode == 'combinations':
            thresholds_to_render = self.get_threshold_queries()
        else:
            thresholds_to_render = self.parameters.thresholds
        self._log(
            f"Generating conserved reciprocal graphs for "
            f"{len(thresholds_to_render)} threshold queries...")

        output_paths = []
        for threshold in thresholds_to_render:
            result = self.visualize_conserved_reciprocal_graph(
                threshold=threshold,
                trim_dead_ends=trim_dead_ends,
                output_folder=output_folder,
                showfig=showfig,
                **vispath_kwargs
            )
            if result:
                output_paths.append(result)

        if output_paths:
            self._log(f"  Generated {len(output_paths)} conserved reciprocal graphs in: conserved_reciprocal_graph/")

        return output_paths
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_dataset_config(self, name: str) -> Optional[DatasetConfig]:
        """Get dataset configuration by name."""
        return self._dataset_configs.get(name)
    
    def clear_results(self):
        """Clear all cached results."""
        self.raw_results.clear()
        self.aligned_results.clear()
        self.query_aligned_results.clear()
        self._network_aligned_cache.clear()
        self._similarity_cache.clear()
        self.comparison_report = None
        self._log("Results cleared")
    
    def set_label_mapper(self, mapper: LabelMapper):
        """
        Set or update label mapper.
        
        Args:
            mapper: LabelMapper instance
        """
        self.label_mapper = mapper
        # Clear aligned results since labels may change
        self.aligned_results.clear()
        self.query_aligned_results.clear()
        self._network_aligned_cache.clear()
        self._similarity_cache.clear()
        self._log("Label mapper updated")
    
    def generate_html_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate an interactive HTML report with Plotly charts.
        
        Creates a self-contained HTML file with:
        - Executive summary with key findings
        - Interactive path count charts
        - Edge weight heatmap
        - Conservation analysis tables
        - Dataset comparison dashboard
        
        Args:
            output_path: Path to save the HTML report.  When omitted, the
                compatibility default is ``comparison_results/comparison_report.html``;
                :meth:`export_results` passes the run-root path so the
                exported report is alongside ``parameters.json`` and the
                plain-text report.
            
        Returns:
            Path to the generated HTML file
        """
        # Ensure comparison has been run
        if self.comparison_report is None:
            self.run_comparison_analysis()
        
        if output_path is None:
            output_path = os.path.join(
                self.parameters.full_output_path,
                "comparison_results",
                "comparison_report.html"
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Try to import plotly, fall back to basic HTML if not available
        try:
            import plotly  # noqa: F401
            has_plotly = True
        except ImportError:
            has_plotly = False
            self._log("Warning: Plotly not installed. Generating basic HTML report.")
        
        html_content = self._generate_html_content(has_plotly)

        if getattr(self.parameters, 'report_layout', 'tabbed') == 'both':
            # Side-by-side comparison mode: also write the legacy
            # single-page report next to the tabbed one.
            try:
                legacy_content = self._generate_legacy_html_content(has_plotly)
                legacy_path = str(
                    Path(output_path).with_name(
                        'comparison_report_legacy.html'))
                with open(legacy_path, 'w', encoding='utf-8') as f:
                    f.write(legacy_content)
                self._log(f"Legacy HTML report saved to: {legacy_path}")
            except Exception as exc:  # noqa: BLE001 — comparison extra only
                self._log(f"Warning: could not write the legacy report: {exc}")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self._log(f"HTML report saved to: {output_path}")
        return output_path
    
    def _generate_html_content(self, has_plotly: bool = True) -> str:
        """Generate the HTML content for the report — the tabbed layout by
        default (``report_layout='tabbed'``, wrapping the legacy
        generator's output via ``comparison/report_tabbed.py``); the
        historical single-page report with ``report_layout='legacy'``."""
        legacy = self._generate_legacy_html_content(has_plotly)
        return self._apply_report_layout(legacy)

    def _generate_legacy_html_content(self, has_plotly: bool = True) -> str:
        """Generate the legacy single-page report content (the reference
        implementation, unchanged).  The tabbed layout wraps this via
        :meth:`_generate_html_content`."""
        from .html_report_generator import generate_html_report

        if self.parameters.threshold_mode == 'combinations':
            # Custom combinations use the same full report shell as Standard
            # comparisons, but all sections consume explicit query points.
            legacy = generate_html_report(
                analyzer=self,
                dataset_names=self.parameters.get_dataset_names(),
                thresholds=[],
                mode_specific_note=self._generate_mode_specific_note(),
                path_count_data=[],
                key_findings_per_threshold={},
                comparison_points=self.get_comparison_points(),
            )
            return legacy
        else:
            dataset_names = self.parameters.get_dataset_names()
            # Feature G: per-threshold report sections render the effective
            # thresholds only — τ-collapse duplicates are marked in the
            # sensitivity export instead of rendered twice.
            thresholds = self._analysis_thresholds()

            # Generate mode-specific note
            mode_specific_note = self._generate_mode_specific_note()

            # Collect path count data for charts
            path_count_data = []
            for dataset in dataset_names:
                for threshold in thresholds:
                    df = self.raw_results.get(dataset, {}).get(threshold, pd.DataFrame())
                    count = len(df) if not df.empty else 0
                    path_count_data.append({
                    'dataset': self.parameters._sanitize_name(dataset),
                    'threshold': threshold,
                    'count': count
                })
        
        # Get key findings per threshold and add path stats
        key_findings_per_threshold = self.comparison_report.get('key_findings_per_threshold', {})
        
        # Get path presence matrix for accurate path counts
        path_presence_matrix = self.comparison_report.get('path_presence_matrix', pd.DataFrame())
        
        # Add path statistics to key findings using path presence matrix
        for threshold in thresholds:
            if not path_presence_matrix.empty:
                # Use path presence matrix (sanitized names with _t{threshold} suffix)
                total_paths = 0
                common_paths = 0
                
                # Find columns for this threshold
                cols_for_threshold = []
                for d in dataset_names:
                    safe_name = self.parameters._sanitize_name(d)
                    col_name = f'{safe_name}_t{threshold}'
                    if col_name in path_presence_matrix.columns:
                        cols_for_threshold.append(col_name)
                
                if cols_for_threshold:
                    # Count paths present in at least one dataset at this threshold
                    is_any = pd.Series(False, index=path_presence_matrix.index)
                    for col in cols_for_threshold:
                        vals = path_presence_matrix[col]
                        if vals.dtype == object:
                            is_any |= ((vals == 'True') | (vals == True))
                        elif vals.dtype == bool:
                            is_any |= vals
                        else:
                            is_any |= (vals > 0)
                    total_paths = int(is_any.sum())
                    
                    # Count common paths (present in ALL datasets at this threshold)
                    if len(cols_for_threshold) == len(dataset_names):
                        is_common = pd.Series(True, index=path_presence_matrix.index)
                        for col in cols_for_threshold:
                            vals = path_presence_matrix[col]
                            if vals.dtype == object:
                                is_common &= ((vals == 'True') | (vals == True))
                            elif vals.dtype == bool:
                                is_common &= vals
                            else:
                                is_common &= (vals > 0)
                        common_paths = int(is_common.sum())
                
                if threshold in key_findings_per_threshold:
                    key_findings_per_threshold[threshold]['total_paths'] = total_paths
                    key_findings_per_threshold[threshold]['common_paths'] = common_paths
                    if total_paths > 0:
                        key_findings_per_threshold[threshold]['path_conservation_rate'] = common_paths / total_paths
        
        # Generate HTML using the new generator
        legacy = generate_html_report(
            analyzer=self,
            dataset_names=dataset_names,
            thresholds=thresholds,
            mode_specific_note=mode_specific_note,
            path_count_data=path_count_data,
            key_findings_per_threshold=key_findings_per_threshold
        )
        return legacy

    def _apply_report_layout(self, legacy: str) -> str:
        """Apply the ``report_layout`` setting: transform the legacy
        single-page HTML into the tabbed layout ('tabbed', the default)
        or pass it through ('legacy')."""
        layout = getattr(self.parameters, 'report_layout', 'tabbed')
        if layout == 'legacy':
            return legacy
        try:
            from .report_tabbed import build_tabbed_report
            return build_tabbed_report(self, legacy)
        except Exception as exc:  # noqa: BLE001 — layout is cosmetic
            self._log(f"Warning: tabbed report layout failed ({exc}); "
                      "falling back to the legacy single-page report")
            return legacy

    def _generate_combination_html_content(self) -> str:
        """Backward-compatible entry point for the full query report.

        Older integrations called this private method directly.  Keep that
        call safe, but route it through the same full report shell used by
        :meth:`_generate_html_content`; combination mode must not fall back to
        the historical compact table.
        """
        from .html_report_generator import generate_html_report
        return generate_html_report(
            analyzer=self,
            dataset_names=self.parameters.get_dataset_names(),
            thresholds=[],
            mode_specific_note=self._generate_mode_specific_note(),
            path_count_data=[],
            key_findings_per_threshold={},
            comparison_points=self.get_comparison_points(),
        )

        # Kept below only as historical source context for downstream forks;
        # the return above is intentional and makes the private compatibility
        # path obey the current report contract.
        import html

        datasets = self.parameters.get_dataset_names()
        queries = self.get_threshold_queries()
        esc = lambda value: html.escape(str(value))

        def fmt(value):
            if value is None:
                return '—'
            if isinstance(value, float):
                return f'{value:g}'
            return str(value)

        query_rows = []
        provenance_rows = []
        similarity_rows = []
        for query in queries:
            query_id = query['id']
            aligned = self.get_aligned_data_for_query(query)
            available = [ds for ds in datasets if ds in aligned.columns]
            total_edges = len(aligned)
            common_edges = int((aligned[available] > 0).all(axis=1).sum()) \
                if available else 0
            path_count = common_paths = 0
            try:
                path_data = self._get_path_data_for_query(query)
                path_count = len(path_data)
                path_available = [ds for ds in datasets
                                  if ds in path_data.columns]
                common_paths = int(
                    (path_data[path_available] > 0).all(axis=1).sum()
                ) if path_available else 0
            except Exception:
                pass
            query_rows.append((query, total_edges, common_edges,
                               path_count, common_paths))
            for ds in datasets:
                threshold = int(query['thresholds'][ds])
                row = self._path_provenance_row(ds, threshold)
                provenance_rows.append((query, ds, row))

        similarities = (self.comparison_report or {}).get(
            'threshold_similarities', pd.DataFrame())
        if isinstance(similarities, pd.DataFrame) and not similarities.empty:
            for _, row in similarities.iterrows():
                similarity_rows.append(row)

        parts = [
            '<!doctype html><html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            '<title>DROCAT Cross-Dataset Comparison</title>',
            '<style>body{font-family:system-ui,sans-serif;margin:2rem;color:#1f2937}'
            'table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.9rem}'
            'th,td{border:1px solid #d1d5db;padding:.4rem;text-align:left}'
            'th{background:#f3f4f6}.card{padding:1rem;border:1px solid #d1d5db;'
            'border-radius:.5rem;margin:1rem 0}.note{color:#4b5563}',
            '</style></head><body>',
            '<h1>Cross-Dataset Comparison</h1>',
            '<p>Advanced threshold combinations: each query row uses one '
            'requested threshold per dataset. The raw threshold union is not '
            'a comparison axis.</p>',
            '<div class="card"><strong>Datasets:</strong> '
            + esc(', '.join(datasets)) + '</div>',
            '<h2>Threshold queries</h2>',
            '<table><tr><th>Query</th><th>Label</th>'
            + ''.join(f'<th>{esc(ds)} requested</th>' for ds in datasets)
            + '<th>Total edges</th><th>Common edges</th>'
              '<th>Total paths</th><th>Common paths</th></tr>',
        ]
        for query, total_edges, common_edges, path_count, common_paths \
                in query_rows:
            parts.append(
                '<tr><td>' + esc(query['id']) + '</td><td>'
                + esc(query.get('label', query['id'])) + '</td>'
                + ''.join(
                    f'<td>{esc(query["thresholds"].get(ds))}</td>'
                    for ds in datasets)
                + f'<td>{total_edges}</td><td>{common_edges}</td>'
                  f'<td>{path_count}</td><td>{common_paths}</td></tr>')
        parts.append('</table>')

        parts.extend([
            '<h2>Applied threshold and bottleneck provenance</h2>',
            '<p class="note">Applied is the canonical equivalent Min '
            'Synapse Count for the materialized output. tau is the '
            'StrongestFirst landing/collapse bound. w0/w1 are the Edge '
            'Budget floor/landing, w2 is the strongest dropped bottleneck, '
            'and W* is the strongest retained bottleneck.</p>',
            '<table><tr><th>Query</th><th>Dataset</th><th>Requested</th>'
            '<th>Applied</th><th>Source</th><th>SF budget</th>'
            '<th>SF bite</th><th>tau</th><th>Edge budget</th>'
            '<th>Edge budget applied</th><th>w0</th><th>w1</th><th>w2</th>'
            '<th>W*</th>'
            '<th>Complete</th></tr>',
        ])
        for query, ds, row in provenance_rows:
            values = (
                query['id'], ds, row.get('requested_threshold'),
                row.get('applied_threshold'), row.get('applied_threshold_source'),
                row.get('strongest_first_budget'),
                row.get('strongest_first_budget_bitten'), row.get('tau'),
                row.get('edge_budget'), row.get('edge_budget_applied'),
                row.get('edge_weight_floor'),
                row.get('edge_budget_landing'),
                row.get('strongest_dropped_bottleneck'),
                row.get('strongest_retained_bottleneck'),
                row.get('paths_complete'),
            )
            parts.append('<tr>' + ''.join(
                f'<td>{esc(fmt(value))}</td>' for value in values
            ) + '</tr>')
        parts.append('</table>')

        parts.extend([
            '<h2>Pairwise similarities</h2>',
            '<table><tr><th>Query</th><th>Dataset 1</th><th>Dataset 2</th>'
            '<th>Jaccard</th><th>Ruzicka</th><th>Weight correlation</th>'
            '<th>Common edges</th></tr>',
        ])
        for row in similarity_rows:
            parts.append('<tr>' + ''.join(
                f'<td>{esc(fmt(row.get(key)))}</td>' for key in (
                    'query_id', 'dataset_1', 'dataset_2',
                    'jaccard_similarity', 'ruzicka_similarity',
                    'pearson_correlation', 'common_edges')
            ) + '</tr>')
        if not similarity_rows:
            parts.append('<tr><td colspan="7">No similarity rows found.</td></tr>')
        parts.extend([
            '</table>',
            '<p class="note">Machine-readable query join: '
            'comparison_results/threshold_combinations.csv. Presence and '
            'path tables are exported as one file per query id.</p>',
            '</body></html>',
        ])
        return ''.join(parts)


def quick_compare(
    datasets: List[Union[str, DatasetConfig]],
    source_neurons: List[Union[str, int]],
    target_neurons: List[Union[str, int]],
    max_interlayer: int = 2,
    thresholds: Optional[List[int]] = None,
    label_mapper: Optional[LabelMapper] = None,
    output_folder: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for quick cross-dataset comparison.
    
    Args:
        datasets: List of dataset identifiers (strings) or DatasetConfig objects
        source_neurons: Source neuron types/patterns (shared across all datasets)
        target_neurons: Target neuron types/patterns (shared across all datasets)
        max_interlayer: Maximum interlayer hops (default: 2)
        thresholds: List of weight thresholds (default: [1, 3, 5, 10, 20])
        label_mapper: Optional LabelMapper for standardization
        output_folder: Optional output directory
        verbose: Print progress
        
    Returns:
        Comparison results dictionary
        
    Example:
        >>> results = quick_compare(
        ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
        ...     source_neurons=['MBON14.*_R'],
        ...     target_neurons=['KCg-d.*_R'],
        ...     max_interlayer=2,
        ...     thresholds=[1, 5, 10]
        ... )
    """
    if thresholds is None:
        thresholds = [1, 3, 5, 10, 20]
    
    params = ComparisonParameters(
        datasets=datasets,
        source_neurons=source_neurons,
        target_neurons=target_neurons,
        max_interlayer=max_interlayer,
        thresholds=thresholds,
        output_folder=output_folder or '.',
    )
    
    analyzer = ComparisonAnalyzer(params, label_mapper, verbose)
    results = analyzer.run_comparison()
    
    if output_folder:
        analyzer.export_results()
    
    return results
