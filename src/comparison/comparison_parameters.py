"""
ComparisonParameters - Global comparison settings for cross-dataset analysis.

This module defines the ComparisonParameters dataclass that holds all global settings
for running cross-dataset comparisons. ComparisonParameters is the PRIMARY ENTRY POINT
for comparison workflows - all necessary parameters are defined here.

Workflow:
    1. Create ComparisonParameters with datasets (strings or DatasetConfig), neurons, and settings
    2. Pass to ComparisonAnalyzer for execution
    3. Optionally provide LabelMapper for cross-dataset label standardization
"""

from dataclasses import dataclass, field
from typing import List, Union, Optional, Any, Dict, Tuple, TYPE_CHECKING
from datetime import datetime
import json
import os
import warnings

try:
    from ..flywire_ids import is_fafb_dataset, is_local_connectome_dataset
except ImportError:  # pragma: no cover - direct package imports
    from flywire_ids import is_fafb_dataset, is_local_connectome_dataset

try:
    from ..utils.naming_utils import canonical_dataset_name
except ImportError:  # pragma: no cover - direct package imports
    from utils.naming_utils import canonical_dataset_name

try:
    from ..utils.naming_utils import make_unique_dataset_labels
except ImportError:  # pragma: no cover - supports direct ``comparison`` imports
    from utils.naming_utils import make_unique_dataset_labels

if TYPE_CHECKING:
    from .label_mapper import LabelMapper


@dataclass
class ComparisonParameters:
    """
    Global comparison settings for cross-dataset analysis.
    
    This is the PRIMARY ENTRY POINT for comparison workflows. All necessary parameters
    are defined here, including datasets, source/target neurons, and analysis settings.
    
    Attributes:
        datasets (List[str | DatasetConfig]): List of dataset identifiers 
            (e.g., 'hemibrain:v1.2.1', 'male-cns:v0.9') or DatasetConfig objects.
            At least 2 datasets required for comparison (or 1 with allow_single_dataset=True).
            
        datasets_nickname (List[str], optional): Short display names for datasets in
            visualizations. If provided, must match length of datasets list.
            Example: ['Hemi', 'Male', 'FAFB'] for shorter labels in charts/networks.
            
        source_neurons (List[str | int] | List[List] | Dict | LabelMapper | None): Source neuron 
            types, bodyIds, or patterns to analyze. Supports:
            - Simple list: ['MBON14.*_R', 'PPL101.*_R']
            - Nested list (groups): [['MBON14.*_R'], ['MBON06.*_R']]
            - Dict filter: {'contains': 'DN'}  # Types containing 'DN'
            - Dict filter: {'startswith': ['aMe', 'Mi']}  # Types starting with prefixes
            - Dict filter: {'endswith': '_R'}  # Types ending with suffix
            - Dict filter: {'regex': r'DN[a-z]\d+'}  # Regex pattern
            - Dict filter: {'contains': 'DN', 'endswith': '_R'}  # Combined (AND)
            - LabelMapper: for dataset-specific neuron mapping
            - None: All neurons in the dataset (including untyped, not recommended)
            - Empty list []: All neurons having a given type (excludes untyped neurons)
            
        target_neurons (List[str | int] | List[List] | Dict | LabelMapper | None): Target neuron
            types, bodyIds, or patterns. Same format options as source_neurons.
            
        max_interlayer (int): Maximum number of intermediate hops for path finding.
            Default: 2. Shared across ALL datasets for fair comparison.
            
        thresholds (List[int]): Minimum synapse count cutoffs for filtering.
            Default: [1, 3, 5, 10, 20]. Results generated at each threshold level.
            
        source_labels (str | List[str]): Custom label(s) for source neuron group(s)
            in output files and visualizations.
            
        target_labels (str | List[str]): Custom label(s) for target neuron group(s)
            in output files and visualizations.
            
        top_edges (int): Number of top edges/paths retained in comparison
            report tables (top-edge comparison, overlap, and presence matrices).
            It does not trim the pathfinding graph or set the per-visualization
            drawn-edge cap, though it can affect report-derived summary plots.
            Default: -1 (include all report rows; the UI passes an explicit value).
            
        comparison_mode (str): Analysis mode - 'path' or 'edge'.
            - 'path': Discover edges through source-to-target paths (may filter
              strong edges if path intermediates are weak)
            - 'edge': Query all edges independently (preserves all strong edges)
            Default: 'path'
            
        output_folder (str): Base directory for all output files. Default: '.'
        
        saveas (str, optional): Custom folder name for output. If None, generates
            timestamped name like 'comparison_results_20251127_120000'.
            
        token (str): NeuPrint authentication token. Empty string loads from
            NEUPRINT_APPLICATION_TOKEN environment variable.
            
        allow_single_dataset (bool): Allow single dataset mode for threshold
            sensitivity analysis. Default: True.
    
    Example:
        >>> # Basic comparison workflow
        >>> params = ComparisonParameters(
        ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
        ...     datasets_nickname=['Hemi', 'Male'],  # Short display names
        ...     source_neurons=['MBON14.*_R'],
        ...     target_neurons=['KCg-d.*_R', 'PPL101.*_R'],
        ...     max_interlayer=2,
        ...     thresholds=[1, 3, 5, 10],
        ...     output_folder='./comparison_output',
        ...     saveas='mbon14_comparison'
        ... )
        >>> analyzer = ComparisonAnalyzer(params)
        >>> analyzer.run()
        >>>
        >>> # With grouped neurons (for separate analysis per group)
        >>> params = ComparisonParameters(
        ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
        ...     source_neurons=[['MBON14.*_R'], ['MBON06.*_R']],  # 2 source groups
        ...     target_neurons=[['KCg-d.*_R'], ['PPL101.*_R']],   # 2 target groups
        ...     source_labels=['MBON14_grp', 'MBON06_grp'],
        ...     target_labels=['KCg-d_grp', 'PPL101_grp'],
        ... )
        >>>
        >>> # With dict-based filters (flexible neuron selection)
        >>> params = ComparisonParameters(
        ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
        ...     source_neurons={'contains': 'DN'},        # All descending neurons
        ...     target_neurons={'startswith': ['LC', 'LPLC']},  # LC and LPLC types
        ...     source_labels='DN_neurons',
        ...     target_labels='LC_neurons',
        ... )
        >>>
        >>> # Dict filter with AND logic
        >>> params = ComparisonParameters(
        ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
        ...     source_neurons={'contains': 'DN', 'endswith': '_R'},  # DN types ending in _R
        ...     target_neurons={'regex': r'MBON\\d+'},  # MBON followed by numbers
        ... )
    """
    
    # Primary configuration (REQUIRED)
    datasets: List[Union[str, Any]] = field(default_factory=list)
    """List of dataset identifiers (e.g., 'hemibrain:v1.2.1', 'male-cns:v0.9') or DatasetConfig objects.
    At least 2 datasets required for cross-dataset comparison."""
    
    datasets_nickname: Optional[List[str]] = None
    """Short display names for datasets in visualizations (e.g., ['Hemi', 'Male', 'FAFB']).
    If provided, must match length of datasets list. Falls back to sanitized dataset names if None."""
    
    source_neurons: Union[List[Union[str, int]], List[List[Union[str, int]]], Dict, Any, None] = field(default_factory=list)
    """Source neuron types, bodyIds, or regex patterns. Supports:
    - Simple list: ['MBON14.*_R', 'PPL101.*_R']
    - Grouped list: [['MBON14.*_R'], ['MBON06.*_R']] for separate group analysis
    - Dict filter: {'contains': 'DN'}, {'startswith': 'aMe'}, {'endswith': '_R'}, {'regex': 'pattern'}
    - Dict combined: {'contains': 'DN', 'endswith': '_R'} (AND logic)
    - LabelMapper object for dataset-specific neuron mapping
    - None: All neurons in the dataset (not recommended for large datasets)
    - Empty list []: All neurons having a given type (excludes untyped neurons)"""
    
    target_neurons: Union[List[Union[str, int]], List[List[Union[str, int]]], Dict, Any, None] = field(default_factory=list)
    """Target neuron types, bodyIds, or regex patterns. Same format options as source_neurons.
    - None: All neurons in the dataset (not recommended for large datasets)
    - Empty list []: All neurons having a given type (excludes untyped neurons)"""
    
    max_interlayer: int = 2
    """Maximum interlayer hops for path finding (shared across ALL datasets).
    In shortest path mode (path_mode='shortest') this is an EXACT depth
    bound: paths are capped at max_interlayer + 1 edges (0 = direct
    connections only; the UI defaults to 8). Set a high value (e.g. 99)
    for effectively unlimited search — simple paths cannot exceed the
    neuron count, so a high bound is never reached in practice."""
    
    # Analysis settings
    thresholds: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    """Min synapse count thresholds for standard comparisons.

    In ``threshold_mode='standard'`` each value is a same-threshold query
    applied to every selected dataset.  In ``threshold_mode='combinations'``
    this field is a derived, sorted union of the values used by the explicit
    query rows and is retained for raw-run scheduling/backward compatibility;
    it is not a comparison identity.
    """

    threshold_mode: str = 'standard'
    """Threshold query mode: ``'standard'``, ``'combinations'`` or ``'auto'``.

    Standard mode expands each scalar in ``thresholds`` to one query shared by
    all selected datasets.  Combination mode uses one complete row in
    ``threshold_combinations`` per query, with one requested threshold per
    selected dataset.  Auto mode (plan §4.8) measures the density windows
    from one bootstrap run and installs density-aligned combination rows.
    """

    threshold_auto: bool = False
    """True once ``threshold_mode='auto'`` has measured and installed its
    aligned rows (the mode then reads ``'combinations'``; this flag keeps
    the auto origin available for labeling and warnings)."""

    threshold_combinations: Optional[List[Dict[str, Any]]] = None
    """Explicit cross-dataset threshold query rows.

    Each row has ``id``/``label`` (optional) and a ``thresholds`` mapping from
    every selected dataset name to exactly one positive integer.  Combination
    mode requires at least two selected datasets.  This is a query matrix,
    not a per-dataset threshold schedule.
    """

    threshold_dataset_order: Optional[List[str]] = None
    """Stable dataset-column order used by advanced threshold combinations."""

    dataset_thresholds: Optional[Dict[str, List[int]]] = None
    """Deprecated legacy vertical threshold schedules.

    This field is accepted only to read older saved configurations.  New UI
    payloads and exports use ``threshold_mode`` and
    ``threshold_combinations``.  When supplied directly, the legacy schedule
    remains available through ``get_thresholds_for_dataset`` so historical
    scripts do not silently change meaning; new code should use
    ``get_threshold_queries``.
    """
    
    source_labels: Union[str, List[str]] = ''
    """Unified label(s) for source group(s) - string or list matching group count"""
    
    target_labels: Union[str, List[str]] = ''
    """Unified label(s) for target group(s) - string or list matching group count"""
    
    intermediate_labels: Union[str, List[str]] = ''
    """Unified label(s) for intermediate group(s) - string or list matching group count"""
    
    top_edges: int = -1
    """Top-edge/report-row cap for cross-dataset analysis tables.
    Positive values select the strongest rows; 0 or a negative value includes
    all rows. This is independent of graph trimming and the per-visualization
    drawing limit, though it can affect report-derived summary plots."""

    graph_edge_limit_bodyid: Optional[int] = None
    """Edge Budget cap for the bodyId-level graph in path runs.
    In ``all`` mode a positive value may apply a lossy weight floor when the
    lossless-pruned cone exceeds the cap. In ``shortest`` mode the setting is
    ignored: the graph is never floored because trimming can remove the only
    shortest route and inflate reported distances. None uses the per-mode
    default; 0 disables the Edge Budget."""

    max_paths_bodyid: Optional[int] = None
    """Path budget for the StrongestFirst enumerator (the default
    pathfinding): enumeration emits intact paths strongest-first and stops
    at the budget, draining ties, so the result is exactly "all intact
    paths with bottleneck >= the reported tau". None = per-mode default
    (1,000,000 in both path modes). A legacy unbounded enumerator is only
    available to script/API callers that explicitly select a legacy
    algorithm with no budget; the UI always uses StrongestFirst."""

    replay_paths: bool = True
    """Feature F (threshold-alignment spec §9): in path mode ('all'),
    enumerate ONCE at the lowest per-dataset threshold and materialize every
    higher threshold from the bottleneck-annotated path set instead of
    re-running FindAllPath per threshold. Outputs are identical (paths at
    threshold t are exactly the t0 paths whose minimum edge weight >= t);
    disable to force the legacy per-threshold enumeration. Shortest mode is
    never replayed (min-hop sets are not nested across thresholds)."""

    auto_extend_thresholds: bool = False
    """F7: when a run's effective tau collapses asked thresholds (every
    point below tau is a duplicate), extend each dataset's threshold list
    with k x tau_ref points (tau_ref = max per-dataset tau, k = 2, 3, ...)
    while the point stays <= 2x the max asked threshold — the schedule is
    GLOBAL so the expanded points stay shared across datasets. Default
    off; the effective-threshold banner suggests it on collapse."""

    capture_density: bool = False
    """Threshold-density capture (plan §4.1): persist each dataset's
    query-scoped, classified cone arrays (path bottlenecks + edge weights
    with typed/untyped/debris endpoint classes) and ``density_meta.json``
    into ``dataset_data/{dataset}/_density/``. The comparison driver turns
    this on for every delegated pathfinding run in every threshold mode
    (curves are exported in standard, combinations, and auto); the flag is
    an opt-out for direct API callers. Capture only runs in FindAllPath
    ('all' mode)."""

    density_normalizer: str = 'per_node'
    """Normalizer for the auto-mode density y-axis (plan §4.3): the primary
    ``'per_node'`` = E(t)/N with a t-independent N (annotated searched
    nodes, falling back to searched nodes). Diagnostic-only alternatives:
    ``'raw'`` (bare E(t)), ``'per_source'`` (E(t)/n_source_neurons),
    ``'cone'`` (E(t)/relevant-cone edges at applied)."""

    drop_untyped: bool = True
    """Drop edges touching untyped neurons from the cross-dataset
    results (default on). A neuron is untyped when its resolved type
    label is empty / Unknown / its own bodyId (the fallback label);
    such edges can never match across datasets. The dropped rows are
    exported to comparison_results/untyped_dropped_records.csv and the
    per-run counts appended to user_warning_notes.txt."""

    report_layout: str = 'legacy'
    """HTML report layout: ``'legacy'`` (default) renders the historical
    single-page report unchanged; ``'tabbed'`` opts in to the new
    page-tabbed report (per-combo pages, split per-threshold /
    per-density plots — ``comparison/report_tabbed.py`` wrapping the
    legacy generator's output; still under refinement); ``'both'``
    writes the tabbed report as ``comparison_report.html`` AND the
    legacy single-page report as ``comparison_report_legacy.html`` for
    side-by-side comparison."""

    edgeN_limit: int = 500
    """Visualization Edge Limit passed to the FindAllPath visualizations.
    It only caps edges drawn per network/Sankey/heatmap visualization; it does
    not trim the analysis graph or alter path discovery."""
    
    overall_label_mapper: Optional[Any] = None
    """LabelMapper object or configuration for defining source, target, and intermediate neuron labels."""

    overall_mapping_json: Optional[str] = None
    """Path to a JSON file with source/target/intermediate mappings (LabelMapper format).
    Convenient for UI runs: the file path is serializable, unlike a LabelMapper object."""

    source_mapping_file: Optional[str] = None
    """Path to a CSV/JSON source-mapping file (LabelMapper format)."""

    target_mapping_file: Optional[str] = None
    """Path to a CSV/JSON target-mapping file (LabelMapper format)."""

    intermediate_mapping_file: Optional[str] = None
    """Path to a CSV/JSON intermediate-mapping file (LabelMapper format)."""
    
    verbose: bool = True
    """Whether to print initialization summary and progress updates."""
    
    # Connectivity Profile Verification Settings
    # These are used by run_connectivity_profile_verification() when called separately
    verification_direction: str = 'both'
    """Direction for profile verification: 'upstream', 'downstream', or 'both'"""
    
    verification_mode: str = 'loose'
    """Verification comparison mode: 'strict' or 'loose'.
    - 'strict': Only compare explicitly typed partners
    - 'loose': Include untyped partners in comparison"""
    
    verification_top_k: int = 15
    """Number of top partners to include per type for profile comparison"""
    
    verification_top_m: int = 5
    """Minimum synapse count for top partners (0 = no filter)"""
    
    verification_min_synapse_threshold: int = 3
    """Minimum synapse count for edges to include in verification"""
    
    verification_include_untyped: bool = True
    """Include untyped partners in profile comparison"""
    
    verification_min_common_partners: int = 3
    """Minimum common partners required for valid comparison"""
    
    verification_score_weights: Dict[str, float] = field(default_factory=lambda: {'jaccard': 0.50, 'rank': 0.50})
    """Weights for combining Jaccard and rank correlation scores"""
    
    verification_include_partner_details: bool = True
    """Include detailed partner information in verification output"""
    
    verification_include_visualizations: bool = True
    """Generate visualization files for verification results"""
    
    # Parallel processing settings
    parallel: bool = True
    """Enable parallel processing for batch operations (verification, similarity matrix)"""
    
    max_workers: Optional[int] = None
    """Maximum number of parallel workers. None = auto (min(32, cpu_count + 4))"""
    
    # Comparison mode
    comparison_mode: str = 'path'
    """Comparison mode: 'path' (default) or 'edge'.
    
    - 'path': Uses FindAllPath() to discover edges through paths from source to target.
              Edges are only included if they appear on valid paths. This may cause
              strong edges to be filtered if they only exist on paths with weak
              intermediate edges below the threshold.
              
    - 'edge': Queries all edges between relevant neuron types at the lowest threshold,
              then filters edges independently at each threshold level. This preserves
              all strong edges regardless of path context, providing more accurate
              edge presence comparison across datasets.
    """

    pathfinding: str = 'StrongestFirst'
    """Pathfinding algorithm to use in FindAllPath (names match the
    algorithms):
    - 'StrongestFirst' (default, 2026-09-04): budgeted best-first on the
      path bottleneck - emits intact paths strongest-first; with the path
      budget (max_paths_bodyid) reached, output = all intact paths with
      bottleneck >= the reported tau. Without a budget bite the emitted
      SET equals the complete enumerators.
    - 'MemoizedDFS': Memoized DFS (forward) - complete enumeration
      (unordered); the previous default
    - 'DFS': Memoized DFS (backward) - best when targets are few
    - 'MeetInMiddle': Meet-in-the-middle DFS - fastest at shallow depths
    - 'DP': Backward Reachability (DP) - robust, low memory
    - 'Bidirectional': Bidirectional BFS - shortest paths first, but stores
      full layer trees (highest memory)
    Ignored when path_mode='shortest' (shortest enumeration uses a fixed
    BFS-distance-guided algorithm).
    """
    
    path_mode: str = 'all'
    """Path enumeration mode for the per-dataset runs:
    - 'all': FindAllPath() — every path within max_interlayer (default).
    - 'shortest': FindShortestPath() — only the per-(source, target)
      minimum-hop paths (all ties kept). max_interlayer is an EXACT depth
      bound there (0 = direct connections only; 8 = UI default; set a high
      value like 99 for effectively unlimited search). Discovery stops
      early once all targets are found.
    Applies to both comparison modes ('path' and 'edge').
    """
    
    search_columns: str = 'auto'
    """Which columns to search when resolving source/target neuron names in
    each dataset: 'auto' (default) searches all columns with priority
    bodyId -> type -> instance -> other string columns (e.g. flywireType,
    hemibrainType, mancType); 'type', 'instance' and 'bodyId' restrict the
    search to that single column."""
    
    # Output settings
    output_folder: str = '.'
    """Base folder for all output files"""
    
    saveas: Optional[str] = None
    """Custom folder name (auto-generates with timestamp if None)"""
    
    # Authentication
    token: str = ''
    """NeuPrint token. Empty string = load from NEUPRINT_APPLICATION_TOKEN env var"""
    
    # Internal fixed settings (NOT user-configurable for comparison consistency)
    _use_cache: bool = field(default=True, repr=False)
    """Internal: caching enabled for performance"""
    
    cache_only: bool = False
    """If True, operate in offline mode using only local cache without connecting to NeuPrint server.
    Useful when:
    - The server is unavailable but local cache has all needed data
    - The dataset is no longer available on the server (e.g., deprecated datasets)
    - Working offline with previously cached data
    If cache is insufficient for the query, an error will be raised.
    If False (default), attempts server connection first, falls back to cache-only if connection fails
    AND cache appears sufficient."""
    
    _min_ratio: float = field(default=0.0, repr=False)
    """Internal: ratio filtering done post-hoc"""
    
    _min_prob: float = field(default=0.0, repr=False)
    """Internal: probability filtering done post-hoc"""
    
    _output_format: str = field(default='csv', repr=False)
    """Internal: output format fixed to CSV for comparison"""
    
    allow_single_dataset: bool = True
    """Allow single dataset for threshold sensitivity analysis only"""

    skip_bodyId: bool = True
    """Skip bodyId-level results for speed. Default True: cross-dataset
    comparison aggregates by (mapped) type, and skipping the bodyId-level
    exports cuts multi-GB output files (bodyId connMatrix CSVs, raw path
    lists) to ~MB scale with type-level outputs untouched."""
    """If True, skip bodyId-level data saving, visualization, and calculations.
    Useful for large-scale analyses where only type-level data is needed."""

    find_reciprocal: bool = False
    """If True, FindAllPath enriches the path graph by finding all direct connections
    among nodes in the path graph and saves results in a find_reciprocal subfolder."""

    separate_hemispheres: bool = False
    """Whether to separate left/right hemisphere neurons in type/custom-group aggregation.
    When True, type/custom_group labels are suffixed with _L/_R/_U based on hemisphere info.
    Note: Even when False, hemisphere analysis (symmetry_analysis, keep_only_hemisphere_conserved_connections)
    still works by extracting hemisphere info from type labels that have _L/_R/_U suffixes."""

    symmetry_analysis: bool = True
    """If True, compute hemisphere symmetry analysis per dataset/threshold (symmetric datasets only).
    Analysis extracts hemisphere info from type labels (_L/_R/_U suffixes) and compares
    L vs R connections. Works regardless of separate_hemispheres setting."""

    keep_only_hemisphere_conserved_connections: bool = False
    """If True, keep only edges that are conserved between hemispheres in cross-dataset
    comparison. An edge is conserved if both it and its mirror counterpart exist
    (e.g., A_L->B_L paired with A_R->B_R). This filtering is applied to edges that have
    hemisphere suffixes (_L/_R/_U) in their type labels - edges without hemisphere info
    are kept as-is. Works regardless of separate_hemispheres setting."""

    max_edges_for_metrics: Optional[int] = 50000
    """Maximum number of edges to compute similarity metrics for.
    If the aligned data has more edges than this threshold, expensive metrics
    computation (Jaccard, Ruzicka, correlation, etc.) will be skipped for that threshold.
    
    Default: 50000 (handles most comparisons in reasonable time)
    Set to None to compute for all edges regardless of count.
    For very fast runs, use 5000-10000.
    For complete analysis on smaller datasets, use None."""

    force_API_fetching: bool = False
    """If True, use CAVE API to fetch FlyWire (FAFB) data instead of local files.
    This fetches connection data directly from the CAVE API for more up-to-date data.
    Note: Only applies to FAFB datasets. BANC does not support API fetching."""

    auto_type_mapping: bool = True
    """Enable automatic cross-dataset type mapping using male-cns neuron_df.
    
    When enabled, uses the type mapping columns from male-cns (flywireType, hemibrainType, mancType)
    to automatically resolve equivalent neuron types across datasets. This allows comparing neurons
    by their biological identity even when type names differ (e.g., MeVPLo2 in male-cns = MTe07 in flywire).
    These columns define comparison/search mappings; they do not overwrite a
    source dataset's native ``type`` label in a transformed query overlay.
    For example, an MCNS ``SMP227`` query remains labeled ``SMP227`` in the
    target scene even when its ``flywireType`` value is a ``CB*`` name.
    
    Features:
    - Auto-loads mappings from male-cns_v1_0_allneurons_neuron_df.csv
    - Handles 1-to-1 type mappings automatically
    - Warns about N-to-1 mappings that could cause incorrect aggregation
    - Priority order: male-cns > flywire > manc > hemibrain > optic-lobe
    - LabelMapper has higher priority (manual mappings override auto-mapping)
    
    Note: Requires male-cns dataset to be initialized first for mapping file to exist."""

    source_dataset: Optional[str] = None
    """Optional explicit namespace for source/target type queries.

    When set (for example ``male-cns:v0.9``), automatic type mapping keeps
    that release as the queried source and can apply a registered release
    alias downstream.  ``None`` preserves the historical priority-based
    source detection.
    """

    def __post_init__(self):
        """Validate and process parameters after initialization."""
        from .label_mapper import LabelMapper
        
        # Ensure datasets is a list
        if isinstance(self.datasets, str):
            self.datasets = [self.datasets]

        # 1. Enforce Exclusivity: overall_label_mapper vs source/target LabelMappers
        has_overall = self.overall_label_mapper is not None
        is_source_mapper = isinstance(self.source_neurons, LabelMapper)
        is_target_mapper = isinstance(self.target_neurons, LabelMapper)

        # File-based mappings (UI runs pass serializable paths): build the
        # LabelMapper here so the analyzer only ever sees overall_label_mapper.
        # These act as a mapping OVERLAY: explicit source/target neuron
        # queries stay (the mapper only renames matching neurons), unlike
        # overall_label_mapper which fully defines the neuron sets.
        self._mapping_overlay = False
        self._source_mapper = None
        self._target_mapper = None
        has_mapping_files = any([
            self.overall_mapping_json,
            self.source_mapping_file,
            self.target_mapping_file,
            self.intermediate_mapping_file,
        ])
        if has_mapping_files:
            if has_overall or is_source_mapper or is_target_mapper:
                raise ValueError(
                    "Ambiguous LabelMapper configuration: mapping file parameters "
                    "(overall_mapping_json / source_mapping_file / target_mapping_file / "
                    "intermediate_mapping_file) cannot be combined with an explicit mapper."
                )
            self.overall_label_mapper = LabelMapper(
                overall_mapping_json=self.overall_mapping_json,
                source_mapping_file=self.source_mapping_file,
                target_mapping_file=self.target_mapping_file,
                intermediate_mapping_file=self.intermediate_mapping_file,
            )
            self._mapping_overlay = True
        if has_overall and (is_source_mapper or is_target_mapper):
            raise ValueError(
                "Ambiguous LabelMapper configuration: 'overall_label_mapper' cannot be used "
                "simultaneously with 'source_neurons' or 'target_neurons' as LabelMappers. "
                "Please provide either a single overall mapper OR specific source/target mappers."
            )
        
        # 2. Handle overall_label_mapper logic (skipped in overlay mode,
        # where the mapper renames matching neurons on top of explicit
        # source/target queries instead of defining them)
        if has_overall and not self._mapping_overlay:
            # If source_neurons provided alongside overall_label_mapper, raise error
            if self.source_neurons:
                raise ValueError(
                    "Ambiguous configuration: 'source_neurons' cannot be provided when "
                    "'overall_label_mapper' is used. The mapper defines the source neurons."
                )
            # If target_neurons provided alongside overall_label_mapper, raise error
            if self.target_neurons:
                raise ValueError(
                    "Ambiguous configuration: 'target_neurons' cannot be provided when "
                    "'overall_label_mapper' is used. The mapper defines the target neurons."
                )
            
            # Extract source/target info from overall_label_mapper
            # We don't populate self.source_neurons list here because it's dataset-specific.
            # Instead, we rely on the mapper being present.
            
            # Auto-populate labels if not provided
            if not self.source_labels:
                self.source_labels = self.overall_label_mapper.get_all_std_labels('source')
            if not self.target_labels:
                self.target_labels = self.overall_label_mapper.get_all_std_labels('target')
            if not self.intermediate_labels:
                self.intermediate_labels = self.overall_label_mapper.get_all_std_labels('intermediate')
                
            # Set internal mapper references
            self._source_mapper = self.overall_label_mapper
            self._target_mapper = self.overall_label_mapper

        # 3. Merge Logic: If source/target are mappers (and no overall), merge into overall_label_mapper
        elif is_source_mapper or is_target_mapper:
            self.overall_label_mapper = LabelMapper()
            
            if is_source_mapper:
                self.overall_label_mapper.merge(self.source_neurons)
                self._source_mapper = self.source_neurons
                self.source_neurons = []  # Clear list, rely on mapper
                
                if not self.source_labels:
                    self.source_labels = self._source_mapper.get_all_std_labels('source')

            if is_target_mapper:
                self.overall_label_mapper.merge(self.target_neurons)
                self._target_mapper = self.target_neurons
                self.target_neurons = []  # Clear list, rely on mapper
                
                if not self.target_labels:
                    self.target_labels = self._target_mapper.get_all_std_labels('target')
        
        # 4. Handle string/list inputs (if not mappers)
        else:
            if isinstance(self.source_neurons, str):
                self.source_neurons = [self.source_neurons]
            elif not self.source_neurons:
                self.source_neurons = []
                
            if isinstance(self.target_neurons, str):
                self.target_neurons = [self.target_neurons]
            elif not self.target_neurons:
                self.target_neurons = []
            
            # Initialize mapper references to None if not set
            self._source_mapper = None
            self._target_mapper = None
        
        # Ensure source_labels is a list
        if isinstance(self.source_labels, str) and self.source_labels:
            self.source_labels = [self.source_labels]
        elif not self.source_labels:
            self.source_labels = []
        
        # Ensure target_labels is a list
        if isinstance(self.target_labels, str) and self.target_labels:
            self.target_labels = [self.target_labels]
        elif not self.target_labels:
            self.target_labels = []
            
        # Ensure intermediate_labels is a list
        if isinstance(self.intermediate_labels, str) and self.intermediate_labels:
            self.intermediate_labels = [self.intermediate_labels]
        elif not self.intermediate_labels:
            self.intermediate_labels = []
        
        # Normalize the threshold query contract.  This runs before the
        # remaining validation because dataset names are already available
        # from ``self.datasets`` and the derived raw-run schedule is needed by
        # output-directory creation and the analyzer.
        self._normalize_threshold_configuration()
        
        # Hemisphere analysis validation and enforcement
        # When separate_hemispheres=True, always enable symmetry_analysis
        if self.separate_hemispheres:
            if not self.symmetry_analysis:
                self.symmetry_analysis = True
                if self.verbose:
                    print("\033[36mℹ️  symmetry_analysis automatically enabled (separate_hemispheres=True)\033[0m")
        else:
            # When separate_hemispheres=False, warn and disable keep_only_hemisphere_conserved_connections
            if self.keep_only_hemisphere_conserved_connections:
                print("\033[33m⚠️  Warning: keep_only_hemisphere_conserved_connections=True requires separate_hemispheres=True.\n"
                      "   Setting keep_only_hemisphere_conserved_connections=False.\033[0m")
                self.keep_only_hemisphere_conserved_connections = False
            # Symmetry analysis also requires separate_hemispheres
            if self.symmetry_analysis:
                if self.verbose:
                    print("\033[33m⚠️  Note: symmetry_analysis requires separate_hemispheres=True to produce meaningful results.\n"
                          "   Setting symmetry_analysis=False.\033[0m")
                self.symmetry_analysis = False
        
        # Validate comparison_mode
        valid_modes = ['path', 'edge']
        if self.comparison_mode not in valid_modes:
            raise ValueError(f"comparison_mode must be one of {valid_modes}, got: {self.comparison_mode}")
        
        # Validate path_mode
        valid_path_modes = ['all', 'shortest']
        if self.path_mode not in valid_path_modes:
            raise ValueError(f"path_mode must be one of {valid_path_modes}, got: {self.path_mode}")
        
        # Initialize auto type mapper (stores CrossDatasetTypeMapper instance if enabled)
        self._auto_type_mapper = None
        if self.auto_type_mapping:
            self._initialize_auto_type_mapping()
        
        # Validate minimum requirements
        if len(self.datasets) < 1:
            raise ValueError("At least 1 dataset is required")
        if len(self.datasets) < 2 and not self.allow_single_dataset:
            raise ValueError(
                "At least 2 datasets are required for cross-dataset comparison. "
                "Set allow_single_dataset=True for single-dataset threshold sensitivity analysis."
            )

        # Warn about FAFB hemisphere annotation when mixed datasets are used
        dataset_names = self.get_dataset_names()
        has_fafb = any(is_fafb_dataset(str(ds)) for ds in dataset_names)
        # BANC has the same hemisphere convention as the local FAFB release;
        # only true NeuPrint datasets trigger the reversal warning.
        has_neuprint = any(
            not is_local_connectome_dataset(str(ds)) for ds in dataset_names
        )
        if has_fafb and has_neuprint and len(dataset_names) > 1:
            print("\033[33m⚠️  FAFB hemisphere labels are reversed relative to NeuPrint datasets.\n"
                  "   Interpret L/R comparisons across FAFB vs NeuPrint with caution.\033[0m")

        # Print initialization summary
        if self.verbose:
            print("\n=== ComparisonParameters Initialization Summary ===")
            print(f"Datasets Included ({len(self.datasets)}):")
            display_nicknames = self.get_dataset_nicknames()
            for i, ds in enumerate(self.datasets):
                nickname = display_nicknames[i] if i < len(display_nicknames) else "N/A"
                print(f"  - {ds} (Nickname: {nickname})")
                
            print(f"\nSource Neurons:")
            if self.overall_label_mapper:
                print(f"  Defined by LabelMapper (Labels: {self.source_labels})")
            elif self.source_neurons is None:
                print(f"  None (All neurons in each dataset, including untyped)")
            elif isinstance(self.source_neurons, list) and len(self.source_neurons) == 0:
                print(f"  [] (All typed neurons in each dataset)")
            else:
                print(f"  {self.source_neurons}")
                # Show auto type mapping results for source neurons
                if self.auto_type_mapping and self._auto_type_mapper:
                    self._print_neuron_mapping_summary(
                        self._ensure_flat_list(self.source_neurons),
                        dataset_names,
                        indent="  "
                    )
                 
            print(f"\nTarget Neurons:")
            if self.overall_label_mapper:
                print(f"  Defined by LabelMapper (Labels: {self.target_labels})")
            elif self.target_neurons is None:
                print(f"  None (All neurons in each dataset, including untyped)")
            elif isinstance(self.target_neurons, list) and len(self.target_neurons) == 0:
                print(f"  [] (All typed neurons in each dataset)")
            else:
                print(f"  {self.target_neurons}")
                # Show auto type mapping results for target neurons
                if self.auto_type_mapping and self._auto_type_mapper:
                    self._print_neuron_mapping_summary(
                        self._ensure_flat_list(self.target_neurons),
                        dataset_names,
                        indent="  "
                    )
            
            if self.intermediate_labels:
                print(f"\nIntermediate Neurons:")
                if self.overall_label_mapper:
                    print(f"  Defined by LabelMapper (Labels: {self.intermediate_labels})")
                else:
                    # If intermediate labels provided but no mapper, just print labels
                    print(f"  Labels: {self.intermediate_labels}")
                 
            print(f"\nLabel Mapper Provided: {'Yes' if self.overall_label_mapper else 'No'}")
            print(f"Auto Type Mapping: {'Enabled' if self.auto_type_mapping else 'Disabled'}")
            if self.auto_type_mapping and self._auto_type_mapper:
                print(f"  └─ Type mapper loaded successfully")
            print(f"Comparison Mode: {self.comparison_mode}")
            print(f"Path Enumeration: {self.path_mode}" +
                  (" (per-pair shortest paths only; Max Layers is an exact depth bound, "
                   "0 = direct connections only)"
                   if self.path_mode == 'shortest' else ""))
            print(f"Separate Hemispheres: {self.separate_hemispheres}")
            print(f"Hemisphere Symmetry Analysis: {self.symmetry_analysis}")
            print(f"Keep Only Hemisphere-Conserved: {self.keep_only_hemisphere_conserved_connections}")
            print(f"Find Reciprocal: {self.find_reciprocal}")
            print("===================================================\n")
        
        if self.threshold_mode == 'combinations':
            if not self.threshold_combinations:
                raise ValueError(
                    "At least one threshold combination row is required "
                    "when threshold_mode='combinations'"
                )
        elif not self.thresholds and self.threshold_mode != 'auto':
            # Auto mode synthesizes its schedule from the measured density
            # windows; an empty threshold list is valid (bootstrap floor
            # defaults to Min Synapse Count 3).
            raise ValueError("At least one threshold is required")
        
        # Handle None and empty list for source/target neurons (similar to coana.py FindNeuronConnection)
        # None = all neurons in dataset (not recommended)
        # [] = all typed neurons (neurons with non-null type)
        self._source_is_all_neurons = self.source_neurons is None
        self._source_is_all_typed = (isinstance(self.source_neurons, list) and 
                                     len(self.source_neurons) == 0 and 
                                     not self._source_mapper and 
                                     not self.overall_label_mapper)
        self._target_is_all_neurons = self.target_neurons is None
        self._target_is_all_typed = (isinstance(self.target_neurons, list) and 
                                     len(self.target_neurons) == 0 and 
                                     not self._target_mapper and 
                                     not self.overall_label_mapper)
        
        # Print warnings for large dataset queries
        if self._source_is_all_neurons or self._target_is_all_neurons:
            print('\033[33m⚠️  Warning: Using None for neurons will query ALL neurons in dataset (including untyped).\n'
                  '   This is not recommended for large datasets. Consider using [] for all typed neurons instead.\033[0m')
        
        if self._source_is_all_typed:
            print('\033[36mℹ️  Source neurons: [] (all typed neurons in each dataset)\033[0m')
        if self._target_is_all_typed:
            print('\033[36mℹ️  Target neurons: [] (all typed neurons in each dataset)\033[0m')
        
        # Cache the timestamp for consistent output_name
        self._cached_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 5. Verify LabelMapper consistency (if used). Skipped in overlay
        # mode: a mapping overlay may legitimately cover only some datasets
        # and never defines the source/target sets.
        if self.overall_label_mapper and not self._mapping_overlay and not self.overall_label_mapper.is_empty:
            # Get dataset names from parameters
            param_datasets = set()
            for ds in self.datasets:
                # Handle DatasetConfig objects or strings
                ds_name = ds.dataset if hasattr(ds, 'dataset') else str(ds)
                param_datasets.add(ds_name)
            
            # Get datasets from mapper
            source_datasets = set()
            if self._source_mapper:
                for label_map in self._source_mapper._source_mapping.values():
                    source_datasets.update(label_map.keys())
            
            target_datasets = set()
            if self._target_mapper:
                for label_map in self._target_mapper._target_mapping.values():
                    target_datasets.update(label_map.keys())
            
            mapper_datasets = source_datasets.union(target_datasets)
            
            # Check 1: Datasets in LabelMapper should match ComparisonParameters.datasets
            missing_in_mapper = param_datasets - mapper_datasets
            if missing_in_mapper:
                print(f"\033[93mWarning: Datasets defined in parameters but missing from LabelMapper: {missing_in_mapper}\033[0m")
                
            # Check 2: Source vs Target consistency in Mapper
            if source_datasets != target_datasets:
                 print(f"\033[93mWarning: Inconsistent datasets between source and target mappings in LabelMapper.\033[0m")
                 if source_datasets - target_datasets:
                     print(f"\033[93m  Source only: {source_datasets - target_datasets}\033[0m")
                 if target_datasets - source_datasets:
                     print(f"\033[93m  Target only: {target_datasets - source_datasets}\033[0m")
    
    @property
    def run_timestamp(self) -> str:
        """Get the cached timestamp string for this run."""
        return self._cached_timestamp

    @staticmethod
    def _coerce_threshold_value(value: Any, field_name: str) -> int:
        """Coerce one threshold value and reject ambiguous/non-positive input."""
        if isinstance(value, bool) or value is None:
            raise ValueError(f"{field_name} must contain positive integers")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} must contain positive integers; got {value!r}"
            ) from exc
        try:
            if float(value) != float(parsed):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} must contain whole positive integers; got {value!r}"
            ) from exc
        if parsed <= 0:
            raise ValueError(
                f"{field_name} must contain positive integers; got {parsed}"
            )
        return parsed

    @classmethod
    def _clean_threshold_list(cls, values: Any, field_name: str) -> List[int]:
        """Normalize a user/API threshold list without changing row meaning."""
        if values is None:
            return []
        if isinstance(values, (str, bytes)):
            values = [part for part in str(values).replace(',', ' ').split() if part]
        try:
            return sorted({
                cls._coerce_threshold_value(value, field_name)
                for value in values
                if value is not None and str(value).strip() != ''
            })
        except TypeError as exc:
            raise ValueError(f"{field_name} must be a list of thresholds") from exc

    def _normalize_threshold_configuration(self) -> None:
        """Normalize standard and row-wise threshold query configuration.

        ``dataset_thresholds`` is deliberately handled as a legacy mode.  It
        is not converted into combination rows by position because unequal
        old lists have no reliable pairing semantics.  New configuration is
        always represented by complete query rows.
        """
        valid_modes = {'standard', 'combinations', 'auto'}
        mode = str(self.threshold_mode or 'standard').strip().lower()
        if self.threshold_combinations is not None and mode == 'standard':
            mode = 'combinations'
        if self.dataset_thresholds and self.threshold_combinations is not None:
            raise ValueError(
                "Use threshold_combinations or legacy dataset_thresholds, not both"
            )

        known_order = list(self.get_dataset_names())
        provided_order = self.threshold_dataset_order
        if provided_order is None:
            dataset_order = known_order
        else:
            dataset_order = list(dict.fromkeys(str(ds) for ds in provided_order))
            if set(dataset_order) != set(known_order) or len(dataset_order) != len(known_order):
                raise ValueError(
                    "threshold_dataset_order must contain each selected dataset "
                    "exactly once"
                )
        self.threshold_dataset_order = dataset_order

        if mode == 'combinations' and len(dataset_order) < 2:
            raise ValueError(
                "threshold_mode='combinations' requires at least two "
                "selected datasets"
            )
        if mode == 'auto' and len(dataset_order) < 2:
            raise ValueError(
                "threshold_mode='auto' requires at least two selected datasets"
            )

        # The old field is kept as an explicit compatibility mode.  It is
        # normalized exactly as historical callers expect and is not emitted
        # by new standard/combination payloads.
        if self.dataset_thresholds:
            self.threshold_mode = 'legacy_vertical'
            self._global_thresholds = self._clean_threshold_list(
                self.thresholds, 'thresholds')
            normalized: Dict[str, List[int]] = {}
            known = set(known_order)
            for ds, values in self.dataset_thresholds.items():
                if ds not in known:
                    if self.verbose:
                        print(
                            f"\033[33m⚠️  dataset_thresholds: ignoring unknown "
                            f"dataset '{ds}' (not in the run selection)\033[0m"
                        )
                    continue
                cleaned = self._clean_threshold_list(
                    values, f"dataset_thresholds[{ds}]")
                if cleaned:
                    normalized[ds] = cleaned
            self.dataset_thresholds = normalized or None
            union = set(self._global_thresholds)
            for values in normalized.values():
                union.update(values)
            self.thresholds = sorted(union)
            self.threshold_combinations = None
            return

        self.dataset_thresholds = None
        if mode not in valid_modes:
            raise ValueError(
                f"threshold_mode must be one of {sorted(valid_modes)}, got: {mode}"
            )
        self.threshold_mode = mode
        self.thresholds = self._clean_threshold_list(self.thresholds, 'thresholds')

        if mode == 'standard':
            if self.threshold_combinations:
                raise ValueError(
                    "threshold_combinations is only valid with "
                    "threshold_mode='combinations'"
                )
            self.threshold_combinations = None
            return

        if mode == 'auto':
            # Auto mode's rows are MEASURED from a bootstrap run and
            # installed later by the driver (thresholds/combinations are
            # synthesized once the per-dataset density windows exist).
            if self.threshold_combinations:
                raise ValueError(
                    "threshold_combinations is only valid with "
                    "threshold_mode='combinations'"
                )
            self.threshold_combinations = None
            self.thresholds = self._clean_threshold_list(
                self.thresholds, 'thresholds')
            return

        rows = self.threshold_combinations or []
        normalized_rows: List[Dict[str, Any]] = []
        seen_ids = set()
        seen_maps = set()
        union = set()
        for index, raw_row in enumerate(rows, start=1):
            if not isinstance(raw_row, dict):
                raise ValueError(
                    f"threshold_combinations row {index} must be an object"
                )
            raw_values = (
                raw_row.get('thresholds')
                or raw_row.get('thresholds_by_dataset')
                or raw_row.get('requested_thresholds')
            )
            if not isinstance(raw_values, dict):
                raise ValueError(
                    f"threshold_combinations row {index} must provide a "
                    "thresholds mapping"
                )
            unknown = sorted(set(raw_values) - set(dataset_order))
            missing = [ds for ds in dataset_order if ds not in raw_values]
            if unknown:
                raise ValueError(
                    f"threshold_combinations row {index} has unknown datasets: "
                    f"{', '.join(unknown)}"
                )
            if missing:
                raise ValueError(
                    f"threshold_combinations row {index} is missing thresholds "
                    f"for: {', '.join(missing)}"
                )
            values = {
                ds: self._coerce_threshold_value(
                    raw_values[ds],
                    f"threshold_combinations row {index} [{ds}]",
                )
                for ds in dataset_order
            }
            row_id = str(raw_row.get('id') or f'combo_{index:03d}').strip()
            if not row_id:
                row_id = f'combo_{index:03d}'
            if row_id in seen_ids:
                raise ValueError(
                    f"threshold_combinations contains duplicate id '{row_id}'"
                )
            signature = tuple(values[ds] for ds in dataset_order)
            if signature in seen_maps:
                raise ValueError(
                    f"threshold_combinations contains duplicate threshold row "
                    f"at index {index}"
                )
            seen_ids.add(row_id)
            seen_maps.add(signature)
            union.update(values.values())
            row_mode = str(raw_row.get('row_mode') or '').strip().lower()
            normalized_rows.append({
                'id': row_id,
                'label': str(raw_row.get('label') or f'Combination {index}'),
                'thresholds': values,
                'row_mode': row_mode or None,
            })
        self.threshold_combinations = normalized_rows
        self.thresholds = sorted(union)

    def get_threshold_queries(self) -> List[Dict[str, Any]]:
        """Return normalized cross-dataset threshold queries in row order.

        A query always has one requested threshold per selected dataset.  In
        standard mode each scalar threshold becomes a same-threshold query.
        Combination mode returns the explicit table rows.  The deprecated
        legacy vertical mode returns standard-shaped rows only as a safe
        compatibility view; its historical per-dataset schedule remains
        available through :meth:`get_thresholds_for_dataset`.
        """
        order = list(self.threshold_dataset_order or self.get_dataset_names())
        if self.threshold_mode == 'auto' and self.threshold_combinations:
            # Auto mode: rows were measured and installed by the driver.
            return [
                {
                    'id': row['id'],
                    'label': row.get('label', row['id']),
                    'thresholds': {ds: int(row['thresholds'][ds]) for ds in order},
                    'dataset_order': list(order),
                    'row_mode': row.get('row_mode'),
                }
                for row in (self.threshold_combinations or [])
            ]
        if self.threshold_mode == 'combinations':
            return [
                {
                    'id': row['id'],
                    'label': row.get('label', row['id']),
                    'thresholds': {ds: int(row['thresholds'][ds]) for ds in order},
                    'dataset_order': list(order),
                    'row_mode': row.get('row_mode'),
                }
                for row in (self.threshold_combinations or [])
            ]
        if self.threshold_mode == 'legacy_vertical':
            return [
                {
                    'id': f'threshold_{threshold}',
                    'label': f'N={threshold}',
                    'thresholds': {ds: int(threshold) for ds in order},
                    'dataset_order': list(order),
                }
                for threshold in self.thresholds
            ]
        return [
            {
                'id': f'threshold_{threshold}',
                'label': f'N={threshold}',
                'thresholds': {ds: int(threshold) for ds in order},
                'dataset_order': list(order),
            }
            for threshold in self.thresholds
        ]

    def get_unique_threshold_jobs(self) -> List[Tuple[str, int]]:
        """Return deduplicated raw jobs required by the threshold queries."""
        jobs = {
            (dataset, int(query['thresholds'][dataset]))
            for query in self.get_threshold_queries()
            for dataset in (self.threshold_dataset_order or self.get_dataset_names())
        }
        return sorted(jobs, key=lambda item: (
            (self.threshold_dataset_order or self.get_dataset_names()).index(item[0]),
            item[1],
        ))

    def install_auto_combinations(self, rows: List[Dict[str, Any]]) -> None:
        """Install measured auto-mode alignment rows (plan §4.8).

        The auto driver measures the per-dataset density windows, builds the
        vertical/horizontal rows, and hands them here. They are stored as
        ordinary ``threshold_combinations`` rows and the mode switches to
        ``'combinations'`` so the rest of the pipeline (query-keyed
        summaries, exports, report) reuses the already-supported engine with
        no special casing. ``threshold_auto`` stays True so auto-specific
        labeling/warnings remain available. Each row's origin is recorded as
        ``row_mode`` ('vertical' | 'horizontal', inferred from the B7 id
        prefixes ``threshold=`` / ``aligned_density=`` and, for legacy
        runs, ``aligned_v_*`` / ``aligned_h_*``) so the report can
        separate the two analyses.
        """
        if not rows:
            raise ValueError(
                "install_auto_combinations requires at least one row")
        installed = []
        for i, r in enumerate(rows, start=1):
            row_id = str(r.get('id') or f"auto_{i:03d}")
            row_mode = str(r.get('row_mode') or '').strip().lower()
            if not row_mode:
                # B7 ids first (`threshold={N}` / `aligned_density={level}`),
                # then the legacy `aligned_v_*` / `aligned_h_*` prefixes.
                if row_id.startswith('threshold='):
                    row_mode = 'vertical'
                elif row_id.startswith('aligned_density='):
                    row_mode = 'horizontal'
                elif row_id.startswith('aligned_v'):
                    row_mode = 'vertical'
                elif row_id.startswith('aligned_h'):
                    row_mode = 'horizontal'
            installed.append({
                'id': row_id,
                'label': str(r.get('label') or r.get('id') or f"auto_{i:03d}"),
                'thresholds': dict(r['thresholds']),
                'row_mode': row_mode or None,
            })
        # Display/manifest order (plan round-4 F-5): verticals first with
        # ascending thresholds, then horizontal rows by density level
        # DESCENDING so the query axis follows thresholds increasing
        # (density falls as the Min Synapse Count rises).
        def _level(row):
            try:
                return float(str(row['id']).split('=', 1)[1])
            except (ValueError, IndexError):
                return float('-inf')
        verticals = [r for r in installed if r.get('row_mode') == 'vertical']
        horizontals = sorted(
            (r for r in installed if r.get('row_mode') == 'horizontal'),
            key=_level, reverse=True)
        others = [r for r in installed
                  if r.get('row_mode') not in ('vertical', 'horizontal')]
        installed = verticals + horizontals + others
        self.threshold_combinations = installed
        self.threshold_mode = 'combinations'
        self.threshold_auto = True
        self._normalize_threshold_configuration()

    def get_thresholds_for_dataset(self, dataset: str) -> List[int]:
        """Return the derived raw-run schedule for one dataset.

        In combination mode this is the unique set of cell values used by
        that dataset across all query rows.  It is an execution/cache helper,
        not the comparison semantic.  The legacy vertical behavior is kept
        only when ``dataset_thresholds`` was supplied by an older caller.
        """
        if self.threshold_mode in ('combinations', 'auto'):
            return sorted({
                int(row['thresholds'][dataset])
                for row in (self.threshold_combinations or [])
                if dataset in row.get('thresholds', {})
            }) or (list(self.thresholds)
                   if self.threshold_mode == 'auto' else [])
        if self.dataset_thresholds:
            override = self.dataset_thresholds.get(dataset)
            if override:
                return list(override)
        if self.dataset_thresholds and getattr(self, '_global_thresholds', None):
            return list(self._global_thresholds)
        return list(self.thresholds)

    
    # Dataset single-character abbreviation mapping
    # Keys are lowercased for case-insensitive matching
    DATASET_SHORT_CODES = {
        'male-cns': 'M',
        'mcns': 'M',
        'fafb': 'F',
        'flywire_fafb': 'F',
        'flywire-fafb': 'F',
        'banc': 'B',
        'flywire_banc': 'B',
        'flywire-banc': 'B',
        'hemibrain': 'H',
        'hemi': 'H',
        'optic-lobe': 'O',
        'optic_lobe': 'O',
        'manc': 'C',
        'mushroombody': 'U',
    }
    
    # Dataset display nickname mapping for plot axes/legends
    # Keys are lowercased for case-insensitive matching
    # These are the default short names used in visualizations when no nickname is provided
    DEFAULT_DISPLAY_NICKNAMES = {
        'male-cns': 'MCNS',
        'mcns': 'MCNS',
        'male_cns': 'MCNS',
        'fafb': 'FAFB',
        'flywire_fafb': 'FAFB',
        'flywire-fafb': 'FAFB',
        'banc': 'BANC',
        'flywire_banc': 'BANC',
        'flywire-banc': 'BANC',
        'hemibrain': 'HEMI',
        'hemi': 'HEMI',
        'optic-lobe': 'OL',
        'optic_lobe': 'OL',
        'opticlobe': 'OL',
        'manc': 'MANC',
        'mushroombody': 'MB',
        'mushroom_body': 'MB',
        'mushroom-body': 'MB',
        'larva': 'LARVA',
        'l1em': 'L1EM',
        'central-brain': 'CB',
        'central_brain': 'CB',
        'vnc': 'VNC',
    }
    
    def _get_neuron_abbreviation(self, neurons: Union[List, Dict, Any], labels: List[str] = None) -> str:
        """
        Generate a short abbreviation for neuron list, similar to FindNeuronConnection's naming.
        
        Logic:
        - If labels are provided, use the first label
        - If LabelMapper, use first source/target label from the mapper
        - If dict filter: generate name from filter operators (e.g., 'DN_R' for contains+endswith)
        - If list with single item: use that item (remove .* patterns)
        - If list with multiple items: use first item + '_etc'
        - If empty: return 'ALL'
        
        Args:
            neurons: Neuron list, nested list, dict filter, or LabelMapper
            labels: Optional custom labels
            
        Returns:
            Abbreviated name string
        """
        from .label_mapper import LabelMapper
        
        # Priority 1: Use provided labels
        if labels and len(labels) > 0:
            if len(labels) == 1:
                return str(labels[0]).replace('.*', '')
            else:
                return str(labels[0]).replace('.*', '') + '_etc'
        
        # Priority 2: Handle LabelMapper
        if isinstance(neurons, LabelMapper):
            std_labels = neurons.get_all_std_labels('source') or neurons.get_all_std_labels('target')
            if std_labels:
                if len(std_labels) == 1:
                    return str(std_labels[0]).replace('.*', '')
                else:
                    return str(std_labels[0]).replace('.*', '') + '_etc'
        
        # Priority 3: Handle dict-based filter
        if isinstance(neurons, dict):
            # Build abbreviation from filter values
            parts = []
            for operator in ['contains', 'startswith', 'endswith', 'regex', 'exact']:
                if operator in neurons:
                    values = neurons[operator]
                    if not isinstance(values, list):
                        values = [values]
                    # Use first value, clean it up
                    if values:
                        part = str(values[0]).replace('.*', '').replace(r'\d+', '').replace(r'\d', '')
                        # Truncate long patterns
                        if len(part) > 10:
                            part = part[:10]
                        parts.append(part)
                        if len(values) > 1:
                            parts.append('etc')
                        break  # Use first operator found
            
            if parts:
                return '_'.join(parts)
            return 'filter'
        
        # Priority 4: Handle list
        if not neurons:
            return 'ALL'
        
        # Flatten nested lists to get first element
        flat_list = self._ensure_flat_list(neurons) if hasattr(self, '_ensure_flat_list') else neurons
        if isinstance(flat_list, list) and len(flat_list) > 0:
            first_item = flat_list[0]
            # Handle nested lists - get the actual first neuron
            while isinstance(first_item, list) and len(first_item) > 0:
                first_item = first_item[0]
            
            # Convert to string and clean up
            name = str(first_item).replace('.*', '')
            
            # Add '_etc' if multiple items
            if len(flat_list) > 1:
                name += '_etc'
            return name
        
        return 'ALL'
    
    def _get_dataset_short_codes(self) -> str:
        """
        Generate combined single-character codes for all datasets.
        
        Uses DATASET_SHORT_CODES mapping. For example:
        - ['male-cns:v0.9', 'flywire_FAFB_v783', 'banc_v626'] -> 'MFB'
        - ['hemibrain:v1.2.1', 'male-cns:v0.9'] -> 'HM'
        
        Returns:
            Combined compact dataset codes. Release suffixes are included when
            selected datasets share the same family code.
        """
        dataset_names = self.get_dataset_names()
        base_codes = []

        for ds_name in dataset_names:
            # Get dataset name string
            # Normalize: lowercase, remove version info
            ds_lower = ds_name.lower()
            # Remove version suffixes like :v1.2.1, _v783, etc.
            import re
            ds_clean = re.sub(r'[:_]v?\d+[\d._]*$', '', ds_lower)
            ds_clean = ds_clean.replace('_', '-')  # Normalize underscores
            
            # Look up in mapping
            code = None
            for key, val in self.DATASET_SHORT_CODES.items():
                if key in ds_clean or ds_clean in key:
                    code = val
                    break
            
            # Fallback: use first character uppercase
            if code is None:
                code = ds_clean[0].upper() if ds_clean else 'X'

            base_codes.append(code)

        # Keep compact one-character codes for distinct dataset families, but
        # append the release when two selected datasets share a family code.
        # This prevents e.g. ``MM`` from hiding which male-cns release ran.
        unique_codes = make_unique_dataset_labels(dataset_names, base_codes)
        return ''.join(unique_codes)
    
    def get_display_nickname(self, dataset_name: str) -> str:
        """
        Get a short display nickname for a dataset for use in plot labels/axes.
        
        Uses DEFAULT_DISPLAY_NICKNAMES mapping. For example:
        - 'male-cns:v0.9' -> 'MCNS'
        - 'flywire_FAFB_v783' -> 'FAFB'
        - 'hemibrain:v1.2.1' -> 'HEMI'
        
        If no mapping found, returns a sanitized short name.
        
        Args:
            dataset_name: Full dataset name or identifier
            
        Returns:
            Short display nickname (typically 2-5 characters)
        """
        import re
        
        # Normalize: lowercase, remove version info
        ds_lower = dataset_name.lower()
        # Remove version suffixes like :v1.2.1, _v783, etc.
        ds_clean = re.sub(r'[:_]v?\d+[\d._]*$', '', ds_lower)
        ds_clean = ds_clean.replace('_', '-')  # Normalize underscores
        
        # Look up in mapping - try various forms
        for key, nickname in self.DEFAULT_DISPLAY_NICKNAMES.items():
            if key in ds_clean or ds_clean in key:
                return nickname
        
        # Fallback: create short name from first word, uppercase, max 6 chars
        # e.g., 'unknown-dataset' -> 'UNKNOW'
        short = ds_clean.split('-')[0].upper()[:6]
        return short if short else dataset_name[:6].upper()
    
    def get_nickname_map(self) -> Dict[str, str]:
        """
        Build a nickname map for all datasets using DEFAULT_DISPLAY_NICKNAMES.
        
        This is a convenience method for visualization functions that need
        a mapping from full dataset names to short display names.
        
        If datasets_nickname is provided, those are used. Otherwise,
        nicknames are auto-generated using DEFAULT_DISPLAY_NICKNAMES.
        
        Returns:
            Dict mapping dataset names to display nicknames
        """
        dataset_names = self.get_dataset_names()
        
        return {
            name: nickname
            for name, nickname in zip(dataset_names, self.get_dataset_nicknames())
        }
    
    @property
    def output_name(self) -> str:
        """
        Get the output folder name.

        Format: cross-dataset_{source_abbr}_to_{target_abbr}_{dataset_codes}_{timestamp}

        Examples:
        - cross-dataset_aMe12_to_PPL101_MFB_20251120_143025
        - cross-dataset_KC_etc_to_MBON_etc_HM_20251120_143025

        Returns:
            saveas value if provided, otherwise auto-generated name with timestamp
        """
        if self.saveas:
            return self.saveas
        
        # Get source abbreviation
        source_abbr = self._get_neuron_abbreviation(
            self.source_neurons if self.source_neurons else self._source_mapper,
            self.source_labels
        )
        
        # Get target abbreviation
        target_abbr = self._get_neuron_abbreviation(
            self.target_neurons if self.target_neurons else self._target_mapper,
            self.target_labels
        )
        
        # Get dataset codes
        dataset_codes = self._get_dataset_short_codes()
        
        return f"cross-dataset_{source_abbr}_to_{target_abbr}_{dataset_codes}_{self.run_timestamp}"
    
    @property
    def full_output_path(self) -> str:
        """
        Get the full path to the output folder.
        
        Returns:
            Absolute path combining output_folder and output_name
        """
        return os.path.abspath(os.path.join(self.output_folder, self.output_name))
    
    @property
    def dataset_data_path(self) -> str:
        """Get path to dataset_data subfolder."""
        return os.path.join(self.full_output_path, 'dataset_data')
    
    @property
    def comparison_results_path(self) -> str:
        """Get path to comparison_results subfolder."""
        return os.path.join(self.full_output_path, 'comparison_results')
    
    @property
    def visualizations_path(self) -> str:
        """Get path to visualizations subfolder."""
        return os.path.join(self.comparison_results_path, 'visualizations')
    
    def get_dataset_names(self) -> List[str]:
        """
        Get dataset identifier strings from datasets list.
        
        Handles both string datasets and DatasetConfig objects.
        
        Returns:
            List of dataset identifier strings
        """
        names = []
        for ds in self.datasets:
            if isinstance(ds, str):
                names.append(ds)
            else:
                # Assume DatasetConfig with dataset attribute
                names.append(getattr(ds, 'dataset', str(ds)))
        return names
    
    def get_dataset_nicknames(self) -> List[str]:
        """
        Get short nicknames for datasets for visualization labels.
        
        Returns datasets_nickname if provided, otherwise returns auto-generated
        display nicknames from DEFAULT_DISPLAY_NICKNAMES mapping.
        
        Returns:
            List of short dataset labels for visualizations
        """
        dataset_names = self.get_dataset_names()
        if self.datasets_nickname and len(self.datasets_nickname) == len(dataset_names):
            base_nicknames = list(self.datasets_nickname)
        else:
            base_nicknames = [self.get_display_nickname(n) for n in dataset_names]

        # A family nickname is intentionally concise, but it must become
        # release-aware when the selection contains the same family twice.
        return make_unique_dataset_labels(dataset_names, base_nicknames)
    
    def _resolved_query_types(self, tokens: List[Any], dataset: str,
                              role: str,
                              drop_unresolved: bool = False
                              ) -> Optional[List[Any]]:
        """Dataset query list from the structured resolver records, or None
        when the resolver cannot handle them (no auto mapping / mapper).

        Phase 3 of plan-cross-dataset-query-resolution-samename-taxonomy:
        plain type tokens resolve to the union of the dataset's record
        targets — taxonomy expansions, taxonomy member-mapping unions,
        same-name identity names, unique renames and licensed split
        branches — while bodyIds, patterns and every non-taxonomy case
        keep the legacy pass-through verbatim.  With ``drop_unresolved``
        (the filtered getters) records without licensed targets — echoes,
        conflicts, evidence-only relations, unmatched tokens — are dropped
        instead of passing through.
        """
        if not self.auto_type_mapping or not self._auto_type_mapper:
            return None
        if tokens is None or (isinstance(tokens, list) and not tokens):
            return tokens
        type_tokens: List[str] = []
        passthrough: List[Any] = []
        for tok in tokens:
            if (isinstance(tok, str) and tok.strip()
                    and '*' not in tok and not tok.strip().isdigit()):
                # Normalize whitespace so a stray 'L2 ' resolves (and
                # dedupes) as 'L2' rather than dropping out of the list.
                normalized = tok.strip()
                if normalized not in type_tokens:
                    type_tokens.append(normalized)
            else:
                passthrough.append(tok)
        try:
            from .query_resolver import (
                DatasetTaxonomyResolver, resolve_query_tokens,
            )
        except ImportError:  # pragma: no cover
            from query_resolver import (
                DatasetTaxonomyResolver, resolve_query_tokens,
            )

        cache = getattr(self, '_query_type_resolution_cache', None)
        if cache is None:
            cache = self._query_type_resolution_cache = {}
        key = (role, tuple(type_tokens))
        if key not in cache:
            # One shared taxonomy resolver per parameters: the reduced
            # dataset frames load once and are reused across tokens/roles.
            taxonomy_resolver = getattr(self, '_taxonomy_resolver', None)
            if taxonomy_resolver is None:
                taxonomy_resolver = DatasetTaxonomyResolver(
                    self._auto_type_mapper).resolve
                self._taxonomy_resolver = taxonomy_resolver
            cache[key] = resolve_query_tokens(
                type_tokens, self.get_dataset_names(),
                self._auto_type_mapper, role=role,
                source_dataset=self.source_dataset,
                taxonomy_resolver=taxonomy_resolver)
        records = cache[key]
        drop_statuses = {'same_name_fallback', 'conflict', 'unmatched',
                         'evidence_only'} if drop_unresolved else set()
        resolved: List[str] = []
        for rec in records:
            if rec.get('dataset') != dataset:
                continue
            if rec.get('status') in drop_statuses:
                continue
            resolved.extend(str(t) for t in rec.get('target_types') or [])
        seen: set = set()
        out: List[Any] = []
        for item in list(passthrough) + resolved:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    def get_source_neurons_for_dataset(self, dataset: str) -> List[Union[str, int]]:
        """
        Get source neurons for a specific dataset.

        Resolution priority:
        1. LabelMapper (explicit mappings)
        2. Auto type mapping (from male-cns neuron_df)
        3. Shared source_neurons list (same names across all datasets)

        Args:
            dataset: Dataset identifier

        Returns:
            List of source neuron types/patterns/bodyIds
        """
        # Priority 1: LabelMapper
        if self._source_mapper is not None:
            # Get neurons from LabelMapper for this dataset
            return self._source_mapper.get_all_neurons_for_dataset(dataset, 'source')

        # Priority 2: Auto type mapping
        if self.auto_type_mapping and self._auto_type_mapper:
            tokens = self._ensure_flat_list(self.source_neurons)
            resolved = self._resolved_query_types(tokens, dataset, 'source')
            if resolved is not None:
                return resolved
            return self._resolve_neurons_with_auto_mapping(tokens, dataset)

        # Priority 3: Shared list
        return self._ensure_flat_list(self.source_neurons)
    
    def get_target_neurons_for_dataset(self, dataset: str) -> List[Union[str, int]]:
        """
        Get target neurons for a specific dataset.
        
        Resolution priority:
        1. LabelMapper (explicit mappings)
        2. Auto type mapping (from male-cns neuron_df)
        3. Shared target_neurons list (same names across all datasets)
        
        Args:
            dataset: Dataset identifier
            
        Returns:
            List of target neuron types/patterns/bodyIds
        """
        # Priority 1: LabelMapper
        if self._target_mapper is not None:
            return self._target_mapper.get_all_neurons_for_dataset(dataset, 'target')

        # Priority 2: Auto type mapping
        if self.auto_type_mapping and self._auto_type_mapper:
            tokens = self._ensure_flat_list(self.target_neurons)
            resolved = self._resolved_query_types(tokens, dataset, 'target')
            if resolved is not None:
                return resolved
            return self._resolve_neurons_with_auto_mapping(tokens, dataset)

        # Priority 3: Shared list
        return self._ensure_flat_list(self.target_neurons)
    
    def get_source_groups(self) -> List[List[Union[str, int]]]:
        """
        Get source neurons as a list of groups.
        
        Returns:
            List of neuron groups (for grouped analysis)
        """
        return self._ensure_grouped(self.source_neurons)
    
    def get_target_groups(self) -> List[List[Union[str, int]]]:
        """
        Get target neurons as a list of groups.
        
        Returns:
            List of neuron groups (for grouped analysis)
        """
        return self._ensure_grouped(self.target_neurons)
    
    def _ensure_flat_list(self, neurons) -> List[Union[str, int]]:
        """Flatten nested list of neurons to single list.
        
        Handles special cases:
        - None: Returns None (signals all neurons in dataset)
        - Empty list []: Returns empty list (signals all typed neurons)
        """
        # None = all neurons, preserve this signal
        if neurons is None:
            return None
        # Empty list = all typed neurons, preserve this signal  
        if isinstance(neurons, list) and len(neurons) == 0:
            return []
        
        flat = []
        for item in neurons:
            if isinstance(item, list):
                flat.extend(self._ensure_flat_list(item) or [])
            else:
                flat.append(item)
        return flat
    
    def _ensure_grouped(self, neurons) -> List[List[Union[str, int]]]:
        """
        Ensure neurons are in grouped format (list of lists).
        
        Handles special cases:
        - None: Returns [[]] (signals all neurons for single group)
        - Empty list []: Returns [[]] (signals all typed neurons for single group)
        """
        # None or empty list = single group with all neurons
        if neurons is None or (isinstance(neurons, list) and len(neurons) == 0):
            return [[]]
        
        # Check if first element is a list (already grouped)
        if isinstance(neurons[0], list):
            return neurons
        else:
            # Wrap in single group
            return [neurons]
    
    def get_dataset_output_path(self, dataset: str, threshold: int) -> str:
        """
        Get output path for a specific dataset and threshold.

        Honors the applied-folder naming grammar once a run has populated
        ``_applied_folder_lookup`` (populated by the analyzer's reconcile
        step): a requested threshold that was pruned/aliased resolves to the
        folder that actually holds its data, and an applied value resolves
        to its grammar name (e.g. ``minsyn_19_applied_floor``). Before any
        run meta exists it is the plain ``minsyn_{threshold}`` builder.
        
        Args:
            dataset: Dataset identifier
            threshold: Synapse threshold
            
        Returns:
            Path to dataset/threshold output folder
        """
        safe_name = self._sanitize_name(dataset)
        lookup = getattr(self, '_applied_folder_lookup', None) or {}
        folder_name = lookup.get((dataset, int(threshold)))
        if folder_name:
            return os.path.join(self.dataset_data_path, safe_name, folder_name)
        return os.path.join(self.dataset_data_path, safe_name, f'minsyn_{threshold}')

    def set_applied_folder_lookup(self, dataset: str, threshold: int,
                                  folder_name: str) -> None:
        """Register the on-disk folder name for one (dataset, threshold)."""
        if not hasattr(self, '_applied_folder_lookup') \
                or self._applied_folder_lookup is None:
            self._applied_folder_lookup = {}
        self._applied_folder_lookup[(dataset, int(threshold))] = folder_name

    def applied_folder_name(self, applied: int, requested_thresholds=None,
                            is_floor: bool = False) -> str:
        """Physical folder name for a materialized threshold.

        Grammar:
        - **collapse floor** (``is_floor=True``): a level one or more other
          requested thresholds alias to. Always
          ``minsyn_{applied}_applied_floor`` — including when the floor value
          coincides with a requested level (that requested level is served by
          this folder, so no ``_equal_...`` variant and no marker).
        - **genuine requested run** (``is_floor=False``): the bare
          ``minsyn_{applied}``.

        ``applied_folder_name_candidates`` lists both forms for disk lookup.
        """
        applied = int(applied)
        if is_floor:
            return f'minsyn_{applied}_applied_floor'
        return f'minsyn_{applied}'

    def applied_folder_name_candidates(self, value: int) -> list:
        """Legacy/current folder names that may hold an applied threshold."""
        value = int(value)
        return [
            f'minsyn_{value}',
            f'minsyn_{value}_applied_floor',
            f'minsyn_{value}_equal_applied_floor',
        ]

    def skipped_folder_name(self, requested: int) -> str:
        """Marker folder name for a requested threshold with no materialized
        output of its own."""
        return f'minsyn_{int(requested)}_skipped'

    def get_applied_output_path(self, dataset: str, applied: int,
                                requested_thresholds) -> str:
        safe_name = self._sanitize_name(dataset)
        return os.path.join(
            self.dataset_data_path, safe_name,
            self.applied_folder_name(applied, requested_thresholds))

    def get_skipped_output_path(self, dataset: str, requested: int) -> str:
        safe_name = self._sanitize_name(dataset)
        return os.path.join(
            self.dataset_data_path, safe_name,
            self.skipped_folder_name(requested))
    
    def _sanitize_name(self, name: str) -> str:
        """Convert name to filesystem-safe format.
        
        Replaces ':' and '.' with '_' but preserves '-' for readability.
        Examples:
            'hemibrain:v1.2.1' -> 'hemibrain_v1_2_1'
            'male-cns:v0.9' -> 'male-cns_v0_9'
            'flywire_FAFB_v783' -> 'flywire_FAFB_v783'
        """
        return canonical_dataset_name(name).replace(':', '_').replace('.', '_')
    
    def create_output_directories(self) -> None:
        """Create all necessary output directories."""
        # Main output folder
        os.makedirs(self.full_output_path, exist_ok=True)

        # Dataset data root only.  Per-threshold folders are created by the
        # pathfinding run under the applied-threshold grammar; pre-creating
        # requested-named folders here would leave empty mislabeled
        # directories behind after a collapse.
        for dataset in self.get_dataset_names():
            os.makedirs(
                os.path.join(self.dataset_data_path,
                             self._sanitize_name(dataset)),
                exist_ok=True)
        
        # Comparison results folder
        os.makedirs(self.comparison_results_path, exist_ok=True)
        
        # Visualizations folder
        os.makedirs(self.visualizations_path, exist_ok=True)
    
    def resolve_token(self) -> str:
        """
        Resolve authentication token.
        
        Returns:
            Token from parameter, or from NEUPRINT_APPLICATION_TOKEN env var
        """
        if self.token:
            return self.token
        return os.environ.get('NEUPRINT_APPLICATION_TOKEN', '')
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation suitable for parameters.json
            
        Note: When auto_type_mapping=True, the source_groups and target_groups
        will contain the resolved types per dataset with unmapped types removed.
        """
        # Build source/target groups per dataset (with unmapped types filtered out for export)
        source_groups = {}
        target_groups = {}
        
        for dataset in self.get_dataset_names():
            # For export, use remove_unmapped=True to filter out types that don't exist
            source_groups[dataset] = self.get_source_neurons_for_dataset_filtered(dataset)
            target_groups[dataset] = self.get_target_neurons_for_dataset_filtered(dataset)
        
        return {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '2.3',
                'run_timestamp': self.run_timestamp,
                'description': 'ComparisonParameters for cross-dataset analysis'
            },
            
            # Dataset configuration
            'datasets': self.get_dataset_names(),
            'datasets_nickname': self.get_dataset_nicknames(),
            'source_dataset': self.source_dataset,
            
            # Neuron configuration
            'source_neurons': self._ensure_flat_list(self.source_neurons),
            'target_neurons': self._ensure_flat_list(self.target_neurons),
            'source_groups': source_groups,
            'target_groups': target_groups,
            'source_labels': self.source_labels,
            'target_labels': self.target_labels,
            'intermediate_labels': self.intermediate_labels,
            
            # Analysis parameters
            'thresholds': self.thresholds,
            'threshold_mode': self.threshold_mode,
            'threshold_dataset_order': list(
                self.threshold_dataset_order or self.get_dataset_names()),
            'threshold_combinations': self.threshold_combinations,
            # Deprecated compatibility field.  New UI/configuration uses the
            # explicit mode and query rows above; legacy callers retain their
            # original schedule so old saved runs remain readable.
            'dataset_thresholds': self.dataset_thresholds,
            'max_interlayer': self.max_interlayer,
            'top_edges': self.top_edges,
            'graph_edge_limit_bodyid': self.graph_edge_limit_bodyid,
            'max_paths_bodyid': self.max_paths_bodyid,
            'replay_paths': self.replay_paths,
            # F7: auto-extend collapsed thresholds (k x tau_ref, capped at
            # 2x the max asked threshold) after the run reveals a collapse.
            'auto_extend_thresholds': self.auto_extend_thresholds,
            'capture_density': self.capture_density,
            'density_normalizer': self.density_normalizer,
            'threshold_auto': bool(getattr(self, 'threshold_auto', False)),
            'drop_untyped': self.drop_untyped,
            'report_layout': self.report_layout,
            'edgeN_limit': self.edgeN_limit,
            'comparison_mode': self.comparison_mode,
            'path_mode': self.path_mode,
            'pathfinding': self.pathfinding,
            
            # Auto type mapping
            'auto_type_mapping': self.auto_type_mapping,
            
            # Data fetching options
            'skip_bodyId': self.skip_bodyId,
            'find_reciprocal': self.find_reciprocal,
            'separate_hemispheres': self.separate_hemispheres,
            'symmetry_analysis': self.symmetry_analysis,
            'keep_only_hemisphere_conserved_connections': self.keep_only_hemisphere_conserved_connections,
            'max_edges_for_metrics': self.max_edges_for_metrics,
            'force_API_fetching': self.force_API_fetching,
            
            # Verification settings
            'verification_settings': {
                'direction': self.verification_direction,
                'mode': self.verification_mode,
                'top_k': self.verification_top_k,
                'top_m': self.verification_top_m,
                'min_synapse_threshold': self.verification_min_synapse_threshold,
                'include_untyped': self.verification_include_untyped,
                'min_common_partners': self.verification_min_common_partners,
                'score_weights': self.verification_score_weights,
                'include_partner_details': self.verification_include_partner_details,
                'include_visualizations': self.verification_include_visualizations
            },
            
            # Performance settings
            'performance_settings': {
                'parallel': self.parallel,
                'max_workers': self.max_workers,
                'use_cache': self._use_cache,
                'cache_only': self.cache_only
            },
            
            # Internal analysis settings
            'analysis_settings': {
                'min_ratio': self._min_ratio,
                'min_prob': self._min_prob,
                'output_format': self._output_format,
                'allow_single_dataset': self.allow_single_dataset,
                'verbose': self.verbose
            },
            
            # Output configuration
            'output_folder': self.output_folder,
            'saveas': self.saveas,
            'full_output_path': self.full_output_path
        }
    
    def get_source_neurons_for_dataset_filtered(self, dataset: str) -> List[Union[str, int]]:
        """
        Get source neurons for export with unmapped types removed.
        
        Same as get_source_neurons_for_dataset but removes types that don't exist
        in the target dataset when auto_type_mapping is enabled.
        
        Args:
            dataset: Dataset identifier
            
        Returns:
            List of source neuron types/patterns/bodyIds with unmapped types removed
        """
        # Priority 1: LabelMapper
        if self._source_mapper is not None:
            return self._source_mapper.get_all_neurons_for_dataset(dataset, 'source')
        
        # Priority 2: Auto type mapping (with unmapped filtering)
        if self.auto_type_mapping and self._auto_type_mapper:
            tokens = self._ensure_flat_list(self.source_neurons)
            resolved = self._resolved_query_types(
                tokens, dataset, 'source', drop_unresolved=True)
            if resolved is not None:
                return resolved
            return self._resolve_neurons_with_auto_mapping(
                tokens,
                dataset,
                remove_unmapped=True
            )

        # Priority 3: Shared list
        return self._ensure_flat_list(self.source_neurons)

    def get_target_neurons_for_dataset_filtered(self, dataset: str) -> List[Union[str, int]]:
        """
        Get target neurons for export with unmapped types removed.
        
        Same as get_target_neurons_for_dataset but removes types that don't exist
        in the target dataset when auto_type_mapping is enabled.
        
        Args:
            dataset: Dataset identifier
            
        Returns:
            List of target neuron types/patterns/bodyIds with unmapped types removed
        """
        # Priority 1: LabelMapper
        if self._target_mapper is not None:
            return self._target_mapper.get_all_neurons_for_dataset(dataset, 'target')
        
        # Priority 2: Auto type mapping (with unmapped filtering)
        if self.auto_type_mapping and self._auto_type_mapper:
            tokens = self._ensure_flat_list(self.target_neurons)
            resolved = self._resolved_query_types(
                tokens, dataset, 'target', drop_unresolved=True)
            if resolved is not None:
                return resolved
            return self._resolve_neurons_with_auto_mapping(
                tokens,
                dataset,
                remove_unmapped=True
            )

        # Priority 3: Shared list
        return self._ensure_flat_list(self.target_neurons)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ComparisonParameters':
        """
        Create ComparisonParameters from dictionary.
        
        Args:
            data: Dictionary with parameter data (e.g., from parameters.json)
            
        Returns:
            ComparisonParameters instance
        """
        # The pre-query schema stored a list of thresholds per dataset under
        # ``dataset_thresholds`` and had no mode/query identity.  A saved
        # payload from that schema is only safely convertible when every
        # selected dataset has an equally long list: position i then defines
        # one complete cross-dataset query row.  Do this migration here (at
        # the serialization boundary) while retaining the direct-constructor
        # compatibility mode used by older Python callers.
        legacy_dataset_thresholds = data.get('dataset_thresholds')
        has_threshold_mode = (
            'threshold_mode' in data
            and data.get('threshold_mode') not in (None, '')
        )
        has_explicit_combinations = data.get('threshold_combinations') not in (
            None, [])
        migrated_threshold_mode = data.get('threshold_mode', 'standard')
        migrated_threshold_combinations = data.get('threshold_combinations')
        migrated_dataset_thresholds = legacy_dataset_thresholds
        migrated_thresholds = data.get('thresholds', [1, 3, 5, 10, 20])
        migrated_threshold_order = data.get('threshold_dataset_order')
        if (legacy_dataset_thresholds and not has_threshold_mode
                and not has_explicit_combinations):
            raw_datasets = data.get('datasets', [])
            if isinstance(raw_datasets, str):
                raw_datasets = [raw_datasets]
            dataset_order = [
                ds.dataset if hasattr(ds, 'dataset') else str(ds)
                for ds in (raw_datasets or [])
            ]
            if not dataset_order:
                raise ValueError(
                    "Cannot migrate legacy dataset_thresholds without selected datasets"
                )
            if not isinstance(legacy_dataset_thresholds, dict):
                raise ValueError(
                    "Legacy dataset_thresholds must be a dataset-to-list mapping"
                )
            unknown = sorted(set(legacy_dataset_thresholds) - set(dataset_order))
            missing = [
                dataset for dataset in dataset_order
                if dataset not in legacy_dataset_thresholds
            ]
            if unknown or missing:
                detail = []
                if unknown:
                    detail.append(f"unknown datasets: {', '.join(unknown)}")
                if missing:
                    detail.append(
                        f"missing datasets: {', '.join(missing)}")
                raise ValueError(
                    "Legacy dataset_thresholds cannot be migrated safely; "
                    + "; ".join(detail)
                    + ". Provide explicit threshold_combinations instead."
                )

            cleaned_by_dataset = {
                dataset: cls._clean_threshold_list(
                    legacy_dataset_thresholds[dataset],
                    f"dataset_thresholds[{dataset}]")
                for dataset in dataset_order
            }
            lengths = {len(values) for values in cleaned_by_dataset.values()}
            if len(lengths) != 1:
                raise ValueError(
                    "Legacy dataset_thresholds lists have unequal lengths; "
                    "provide explicit threshold_combinations with one complete "
                    "row per query instead."
                )
            row_count = lengths.pop()
            if row_count == 0:
                raise ValueError(
                    "Legacy dataset_thresholds contains no threshold rows; "
                    "provide explicit threshold_combinations instead."
                )
            migrated_threshold_order = dataset_order
            if len(dataset_order) < 2:
                # A one-dataset legacy schedule is equivalent to the
                # standard scalar threshold list, not a cross-dataset query.
                migrated_threshold_mode = 'standard'
                migrated_threshold_combinations = None
                migrated_thresholds = cleaned_by_dataset[dataset_order[0]]
                migration_message = (
                    "Migrated legacy dataset_thresholds to standard thresholds "
                    "because Custom combination requires at least two datasets."
                )
            else:
                migrated_threshold_mode = 'combinations'
                migrated_threshold_combinations = [
                    {
                        'id': f'combo_{index:03d}',
                        'label': f'Combination {index}',
                        'thresholds': {
                            dataset: cleaned_by_dataset[dataset][index - 1]
                            for dataset in dataset_order
                        },
                    }
                    for index in range(1, row_count + 1)
                ]
                migration_message = (
                    "Migrated legacy dataset_thresholds positionally to "
                    "threshold_mode='combinations'. Review the generated query "
                    "rows; unequal or incomplete legacy lists are rejected."
                )
            migrated_dataset_thresholds = None
            warnings.warn(
                migration_message,
                UserWarning,
                stacklevel=2,
            )

        # Extract nested settings with defaults
        verification = data.get('verification_settings', {})
        performance = data.get('performance_settings', {})
        
        return cls(
            # Dataset configuration
            datasets=data.get('datasets', []),
            datasets_nickname=data.get('datasets_nickname'),
            source_dataset=data.get('source_dataset'),
            
            # Neuron configuration
            source_neurons=data.get('source_neurons', []),
            target_neurons=data.get('target_neurons', []),
            source_labels=data.get('source_labels', []),
            target_labels=data.get('target_labels', []),
            
            # Analysis parameters
            max_interlayer=data.get('max_interlayer', 2),
            thresholds=migrated_thresholds,
            threshold_mode=migrated_threshold_mode,
            threshold_dataset_order=migrated_threshold_order,
            threshold_combinations=migrated_threshold_combinations,
            dataset_thresholds=migrated_dataset_thresholds,
            top_edges=data.get('top_edges', 50),
            graph_edge_limit_bodyid=data.get('graph_edge_limit_bodyid', None),
            max_paths_bodyid=data.get('max_paths_bodyid', None),
            replay_paths=data.get('replay_paths', True),
            auto_extend_thresholds=data.get('auto_extend_thresholds', False),
            capture_density=data.get('capture_density', False),
            density_normalizer=data.get('density_normalizer', 'per_node'),
            threshold_auto=data.get('threshold_auto', False),
            drop_untyped=data.get('drop_untyped', True),
            edgeN_limit=data.get('edgeN_limit', 500),
            comparison_mode=data.get('comparison_mode', 'path'),
            path_mode=data.get('path_mode', 'all'),
            
            # Verification settings
            verification_direction=verification.get('direction', 'both'),
            verification_mode=verification.get('mode', 'loose'),
            verification_top_k=verification.get('top_k', 5),
            verification_top_m=verification.get('top_m', 0),
            verification_min_synapse_threshold=verification.get('min_synapse_threshold', 3),
            verification_include_untyped=verification.get('include_untyped', True),
            verification_min_common_partners=verification.get('min_common_partners', 3),
            verification_score_weights=verification.get('score_weights', {'jaccard': 0.50, 'rank': 0.50}),
            verification_include_partner_details=verification.get('include_partner_details', True),
            verification_include_visualizations=verification.get('include_visualizations', True),
            
            # Performance settings
            parallel=performance.get('parallel', True),
            max_workers=performance.get('max_workers'),
            
            # Hemisphere / reciprocity flags (plan round-4 fix: from_dict
            # silently dropped these, so a pinned resume/re-export downgraded
            # a hemispheric run to non-hemispheric).
            separate_hemispheres=data.get('separate_hemispheres', False),
            symmetry_analysis=data.get('symmetry_analysis', False),
            keep_only_hemisphere_conserved_connections=data.get(
                'keep_only_hemisphere_conserved_connections', False),
            find_reciprocal=data.get('find_reciprocal', False),
            skip_bodyId=data.get('skip_bodyId', True),

            # Output configuration
            output_folder=data.get('output_folder', '.'),
            saveas=data.get('saveas'),
            token=data.get('token', ''),
        )

    @classmethod
    def for_run_folder(cls, run_dir: str) -> 'ComparisonParameters':
        """Build parameters pinned to an EXISTING run folder (plan R4).

        ``ComparisonParameters.from_dict`` re-derives a fresh timestamped
        output folder, so re-exporting or resuming an old run via a script
        silently creates a new folder. This classmethod reads the run's
        ``parameters.json`` and pins ``full_output_path`` to ``run_dir``
        (via ``output_folder`` = parent + ``saveas`` = folder name), so a
        follow-up ``run_comparison(skip_existing=True)`` +
        ``export_results()`` resumes/re-exports IN PLACE.

        Args:
            run_dir: Path to the existing run folder (must contain
                ``parameters.json``).

        Returns:
            ComparisonParameters instance whose ``full_output_path`` is
            exactly ``run_dir``.
        """
        run_dir = os.path.abspath(run_dir)
        params_path = os.path.join(run_dir, 'parameters.json')
        if not os.path.exists(params_path):
            raise FileNotFoundError(
                f"Not a run folder (missing parameters.json): {run_dir}")
        with open(params_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        params = cls.from_dict(data)
        params.output_folder = os.path.dirname(run_dir)
        params.saveas = os.path.basename(run_dir)
        return params
    
    def _initialize_auto_type_mapping(self) -> None:
        """
        Initialize auto type mapping from male-cns neuron_df.

        Uses the process-wide CrossDatasetTypeMapper singleton
        (``get_type_mapper``) — the SAME mapper instance the Type Mapping
        panel, homolog finding, and the neuron-index viewer use — so every
        surface shares one load state and one set of mapper indexes.
        Mapping warnings will be shown during the initialization summary
        print.

        Nuances of the shared instance: an already-built singleton ignores
        ``workspace_path`` (one repository = one workspace), and the
        singleton is constructed quiet — this class's own summary print
        reports the mapping state.  A mapper whose load fails keeps
        ``_auto_type_mapper`` None (identical to the previous
        new-instance contract); the failure reason stays visible via
        ``mapper.last_load_error`` / ``get_type_mapper_state()``.
        """
        from .cross_dataset_type_mapper import get_type_mapper
        import os

        # Determine workspace path
        # Candidates are derived from the file location and the current
        # working directory only, so the lookup stays portable across machines.
        workspace_candidates = [
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # From this file
            os.getcwd(),
        ]

        workspace_path = None
        for candidate in workspace_candidates:
            # the auto mapping source is the male-cns v1.0 neuron info
            neuron_df_path = os.path.join(
                candidate, 'datasets', 'male-cns_v1_0',
                'male-cns_v1_0_allneurons_neuron_df.csv'
            )
            if os.path.exists(neuron_df_path):
                workspace_path = candidate
                break

        if workspace_path is None:
            if self.verbose:
                print("\n⚠️ Auto type mapping: Could not find male-cns neuron_df file.")
                print("   Initialize male-cns dataset first, or disable auto_type_mapping.")
            self._auto_type_mapper = None
            return

        # Shared singleton: same mapper instance as the panel / homolog /
        # viewer surfaces.  get_type_mapper() triggers load(); an
        # unhealthy mapper is reported as disabled here.
        mapper = get_type_mapper(workspace_path=workspace_path)
        if mapper is None or not mapper._loaded:
            self._auto_type_mapper = None
            return
        self._auto_type_mapper = mapper
    
    def _resolve_neurons_with_auto_mapping(
        self, 
        neurons: List[Union[str, int]], 
        dataset: str,
        remove_unmapped: bool = False
    ) -> List[Union[str, int]]:
        """
        Resolve neuron type names using auto type mapping.
        
        Args:
            neurons: List of neuron types/patterns/bodyIds (or None/empty list for all neurons)
            dataset: Target dataset to resolve for
            remove_unmapped: If True, remove types that don't have a mapping in the target
                           dataset (instead of passing them through). Default False for 
                           backward compatibility with querying.
            
        Returns:
            List of resolved neuron types for the target dataset,
            or None/empty list for special "all neurons" cases
        """
        # Handle special cases for all neurons/all typed neurons
        if neurons is None or (isinstance(neurons, list) and len(neurons) == 0):
            return neurons  # Pass through None or [] as-is for downstream handling
            
        if not self._auto_type_mapper:
            return neurons
        
        resolved = []
        for neuron in neurons:
            # Skip non-string identifiers (bodyIds)
            if not isinstance(neuron, str):
                resolved.append(neuron)
                continue
            
            # Skip regex patterns - pass through as-is
            if '*' in neuron or ('.' in neuron and '.*' in neuron):
                resolved.append(neuron)
                continue
            
            # Try to find mapping (shared validity resolver: only an
            # unambiguous one-target case renames; splits/conflicts behave
            # like the legacy None return)
            source_ds = (
                self.source_dataset
                or self._auto_type_mapper.detect_type_source(neuron)
            )
            if source_ds:
                from .type_resolver import (
                    STATUS_VALID_SPLIT, resolve_valid_targets,
                )
                res = resolve_valid_targets(
                    self._auto_type_mapper, neuron, source_ds, dataset)
                if res.status == STATUS_VALID_SPLIT and res.target_types:
                    # A valid split has real per-branch evidence: query all
                    # licensed branches instead of silently keeping the raw
                    # name (which would match nothing in this dataset).
                    resolved.extend(str(t) for t in res.target_types)
                elif res.equivalence_key:
                    resolved.append(res.equivalence_key)
                elif not remove_unmapped:
                    # No mapping found, use original (might be missing in
                    # target dataset). This is the low-confidence same-name
                    # fallback surfaced by the report's Query Resolution
                    # section and user_warning_notes.
                    resolved.append(neuron)
                # If remove_unmapped=True and no mapping found, skip the type
            elif not remove_unmapped:
                # Type not found in any known dataset, pass through
                resolved.append(neuron)
            # If remove_unmapped=True and type not found, skip it
        
        return resolved
    
    def _print_neuron_mapping_summary(
        self,
        neurons: List[Union[str, int]],
        datasets: List[str],
        indent: str = "  ",
    ) -> None:
        """
        Print mapping summary for source/target neurons.
        
        Shows:
        - Per-dataset resolved types (only if different from query)
        - N-to-1 warnings with recommendation
        - 1-to-N warnings with recommendation
        """
        if not self._auto_type_mapper:
            return
        
        # Get mapping summary
        display_nicknames = self.get_dataset_nicknames()
        summary = self._auto_type_mapper.get_source_target_mapping_summary(
            [n for n in neurons if isinstance(n, str)],
            datasets,
        )
        
        # Print different mappings
        if summary['different_mappings']:
            print(f"{indent}Auto-mapped types:")
            for type_name, mappings in summary['different_mappings']:
                # Show: type_name → dataset1:mapped1, dataset2:mapped2
                mapping_strs = []
                for ds, mapped in mappings.items():
                    if mapped != type_name:
                        # Use nickname if available
                        ds_idx = datasets.index(ds) if ds in datasets else -1
                        ds_label = display_nicknames[ds_idx] if (
                            ds_idx >= 0 and ds_idx < len(display_nicknames)
                        ) else ds
                        mapping_strs.append(f"{ds_label}:{mapped}")
                if mapping_strs:
                    print(f"{indent}  • {type_name} → {', '.join(mapping_strs)}")
        
        # Print N-to-1 warnings
        if summary['n_to_1_warnings']:
            print(f"\n{indent}⚠️  N-to-1 type mapping detected:")
            for type_name, src_ds, tgt_ds, conflict_types in summary['n_to_1_warnings']:
                # Determine which dataset is "A" (N types) and which is "B" (1 type)
                # The source_type in the conflict is the "1" side
                conflict_types_str = ", ".join(sorted(conflict_types))
                if type_name in conflict_types:
                    # User queried a type from the "N" side
                    # Recommend using the "1" type from target dataset
                    print(f"{indent}  • '{type_name}' is one of N types ({conflict_types_str}) in {tgt_ds}")
                    print(f"{indent}    that all map to 1 type in {src_ds}.")
                    print(f"{indent}    💡 Recommend: Query by the {src_ds} type name to compare aggregated results.")
                else:
                    # User queried the "1" type
                    print(f"{indent}  • '{type_name}' in {src_ds} maps to multiple types in {tgt_ds}:")
                    print(f"{indent}    ({conflict_types_str})")
                    print(f"{indent}    💡 Results from {tgt_ds} will aggregate these {len(conflict_types)} types.")
        
        # Print 1-to-N warnings
        if summary['one_to_n_warnings']:
            print(f"\n{indent}⚠️  1-to-N type mapping detected:")
            for type_name, src_ds, tgt_ds, split_types in summary['one_to_n_warnings']:
                split_types_str = ", ".join(sorted(split_types))
                print(f"{indent}  • '{type_name}' in {src_ds} splits into {len(split_types)} types in {tgt_ds}:")
                print(f"{indent}    ({split_types_str})")
                print(f"{indent}    💡 Results from {tgt_ds} will aggregate these {len(split_types)} types.")
                print(f"{indent}    To avoid aggregation, use custom LabelMapper or set auto_type_mapping=False.")

        # Always tell the user to double check automatic mappings
        if (summary['different_mappings'] or summary['n_to_1_warnings']
                or summary['one_to_n_warnings']):
            print(f"\n{indent}⚠️  These auto type mappings were applied automatically - "
                  f"please double check them before interpreting cross-dataset results.")
    
    def get_auto_type_mapper(self) -> Optional[Any]:
        """
        Get the auto type mapper instance if enabled.
        
        Returns:
            CrossDatasetTypeMapper instance or None
        """
        return self._auto_type_mapper
    
    def export_auto_mapping(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        Export the auto type mapping used in this comparison.
        
        Args:
            output_path: Path to save CSV file. If None, saves to output folder.
            
        Returns:
            Path to saved file, or None if auto mapping not enabled.
        """
        if not self._auto_type_mapper:
            return None
        
        if output_path is None:
            output_path = os.path.join(
                self.full_output_path, 
                'auto_type_mapping.csv'
            )
        
        self._auto_type_mapper.export_mapping(output_path)
        return output_path
    
    def __repr__(self) -> str:
        datasets_str = self.get_dataset_names()
        return (f"ComparisonParameters(datasets={datasets_str}, "
                f"sources={len(self._ensure_flat_list(self.source_neurons))}, "
                f"targets={len(self._ensure_flat_list(self.target_neurons))}, "
                f"max_interlayer={self.max_interlayer}, "
                f"path_mode={self.path_mode}, "
                f"thresholds={self.thresholds})")


def canonical_source_rank(dataset: str) -> int:
    """Canonical-row source priority (plan F, 2026-09-15).

    When a canonical type row is observed in several datasets, the report's
    Sources table picks the observation from the highest-priority dataset:
    ``male-cns`` first, then FAFB/FlyWire, then the other NeuPrint
    datasets (hemibrain, manc, optic-lobe, ...), and BANC last (its type
    annotations are the auto-transferred ones). Unknown datasets rank with
    the other-neuprint tier.
    """
    ds = str(dataset or '').lower()
    if 'male-cns' in ds or 'malecns' in ds or 'mcns' in ds:
        return 0
    if 'fafb' in ds or 'flywire' in ds:
        return 1
    if 'banc' in ds:
        return 3
    return 2
