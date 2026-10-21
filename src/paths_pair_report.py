"""Paths pair report — per-source-target-pair HTML report for pathfinding runs.

Plan: ``_plan/plan-paths-pair-report.md``.

Reads the paths tables a run already produced (``*_allpaths_type.csv`` —
Complete Paths, Shortest Paths, or the per-dataset ``minsyn_<N>/`` delegates
inside a cross-dataset run) and writes, additively into the run folder:

* ``paths_pair_breakdown/pair_breakdown_paths.csv`` — one row per path,
  uncapped, with pair/rank columns, the pair's source/target bodyId
  coverage (``n/N`` from the run's enrollment files), and everything the
  HTML caps reference.
* ``paths_pair_breakdown/pair_breakdown_intermediates.csv`` — one row per
  (pair, intermediate): how many paths use it, ``shared`` (>=2 paths of the
  same pair) vs ``unique`` (exactly 1), earliest hop position.
* ``path_report.html`` — self-contained report in the cross-dataset
  report's visual language (house CSS classes, tabbed router in the style
  of ``comparison/report_tabbed.py``; the vis-network library is vendored
  and inlined, so the interactive networks work offline — a static layered
  SVG renders if the library ever fails). Four pages:

  - **Overview** — run summary, hops histogram, pair table, artifact links.
  - **Global** — per-unit stats, the pair x unit path-count matrix (with
    hop-range tooltips), a pairs-by-unit-coverage histogram, and a global
    route network of the most-traversed edges.
  - **Analysis** — one page holding the mode x view destinations that this
    run actually produced (plan-corridor-refill-and-mode-tabs Q1/Option B):
    a **mode** row (``Complete`` / ``Fast (corridor)``) and a **view** row
    (``Pair Explorer`` / ``Refill``). Each row renders only when it has at
    least two destinations and is otherwise stated as a caption, so a
    single-mode run offers two destinations and a both-mode run four —
    never a control with one option.
    * ``Complete → Pair Explorer`` — Source and Target multi-selects render
      ONE pane for the selection: a single pair gets its own pane, several
      pairs get the UNION (one merged top-paths table with a Pair column and
      one merged layered network). Each network card has a tab strip — "All"
      plus one tab per path length in the drawn set — rebuilding the graph
      from only that length's rows. Rows carry the pair's source/target
      bodyId coverage; intermediates are colored shared (blue, >=2 paths of
      the pair) vs unique (gray, exactly 1).
    * ``Complete → Refill`` — the second path-building round as a second
      Pair Explorer, with a scope row (``Refill-asked`` | ``Refill-full``)
      and a ``Refill view`` row (``Complete at ...`` / ``Refill-carrying``).
    * ``Fast (corridor) → Pair Explorer`` — the lane's validated walks,
      same table grammar, tiered at the LANE's own asked threshold.
    * ``Fast (corridor) → Refill`` — the lane's synapse >= 1 second round
      (``route_full_*``), the same two views as the complete mode's.
  - **Data** — unit rollup (with per-delegate report links on
    cross-dataset runs), breakdown CSVs, paths tables, run artifacts, and
    run-level vispath/conserved-view links.

  The old top-level addresses (``#tab=explorer``, ``#tab=refill``,
  ``#tab=corridor``, ``#tab=corridor_refill``) still resolve: they are
  aliases into Analysis, so a cited link lands on the destination it names.
  A run whose ONLY paths are the corridor lane's (Fast Mode = ``fast``) gets
  this same report with the lane as its unit — never an empty state that
  describes validated walks as finding nothing.

  Cross-dataset runs additionally write a single-unit copy of the report
  into every ``dataset_data/<dataset>/<delegate>/`` folder.

Capping is viewport-only: every capped view states its cap and links to the
uncapped CSV beside the report. Ranking is bottleneck-first
(``min_weight`` desc, ``path`` asc; the path-probability tiebreak is
SUSPENDED 2026-10-06 — probability values are withheld due to
unverifiable fidelity and are neither exported nor rendered). The generator
overwrites only its own outputs and is byte-stable modulo the Generated
timestamp.
"""

from __future__ import annotations

import html as _html
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:  # house slug helper — same private-import precedent as report_tabbed.py
    from comparison.html_report_generator import _query_report_slug
except ImportError:  # pragma: no cover - direct src/ execution
    try:
        from html_report_generator import _query_report_slug  # type: ignore
    except ImportError:
        def _query_report_slug(value) -> str:
            slug = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value or 'query'))
            return slug.strip(' ._-') or 'query'


REPORT_NAME = 'path_report.html'
OUTPUT_DIR_NAME = 'paths_pair_breakdown'
PATHS_CSV_NAME = 'pair_breakdown_paths.csv'
INTERMEDIATES_CSV_NAME = 'pair_breakdown_intermediates.csv'
PATHS_SUFFIX = '_allpaths_type.csv'

DEFAULT_TOP_PER_LENGTH = 10
DEFAULT_GLOBAL_EDGES = 60
DEFAULT_GLOBAL_PAIRS = 50
# Data-tab presence census (F-EX-010's sibling item): the sidecar folders
# whose files the tab must link, and the names it must not repeat.
DATA_TAB_CENSUS_DIRS = (
    '', 'data_details', 'data_details/type_level_refill',
    'data_details/route_plan', 'route_scope_paths', 'comparison_results')
DATA_TAB_CENSUS_SKIP_NAMES = {REPORT_NAME, 'user_warning_notes.txt'}
DATA_TAB_CENSUS_CAP = 80

# Where a second-round dataset's inputs live inside a run folder
# (plan-corridor-refill-and-mode-tabs Phase 2). The Complete mode's refill
# mirrors into type_level_refill/ + data_details/type_level_refill/; the
# corridor lane keeps ONE namespace (route_scope_paths/) for its asked
# table and its synapse >= 1 second round, with the same column grammar.
DATASET_LAYOUTS = {
    'complete': {
        'paths': 'type_level_refill',
        'pairs': 'data_details/type_level_refill',
        'pairs_asked': 'refill_type_pairs.csv',
        'pairs_full': 'refill_full_type_pairs.csv',
        # One conditioned-denominator file serves both scopes: the complete
        # mode's full view suppresses its Adj column (synapse >= 1 conditions
        # nothing), so the asked-tier file is never read as a >= 1 denominator.
        'incoming': 'data_details/type_level_refill'
                    '/type_incoming_asked.csv',
        'incoming_full': 'data_details/type_level_refill'
                         '/type_incoming_asked.csv',
        'vispath': (('refill_type_network.html',
                     '🕸️ refill type network (vispath)'),
                    ('refill_bodyId_network.html',
                     '🕸️ refill bodyId network (vispath)')),
        # F7.2 gave the full-connection scope its OWN mirror pages, so the
        # synapse >= 1 view links those rather than the asked scope's picture
        # of a different mass (measured: a live run's report linked only the
        # asked pages while refill_full_type_*.html sat unlinked beside them).
        'vispath_full': (('refill_full_type_network.html',
                          '🕸️ refill-full type network (vispath)'),
                         ('refill_full_type_heatmap.html',
                          '🔥 refill-full type heatmap (vispath)')),
        # The complete mode's pair tables publish the pair mass as `weight`
        # for the emitted set; no per-hop universe readout is printed there,
        # so its explorers opt out of the column (an absent readout beats a
        # borrowed one).
        'universe': False},
    'corridor': {
        'paths': 'route_scope_paths',
        'pairs': 'route_scope_paths',
        'pairs_asked': 'route_connection_type.csv',
        'pairs_full': 'route_full_type_pairs.csv',
        # The lane's two scopes condition at DIFFERENT tiers, so each reads
        # its own denominator file: asked at the lane's asked threshold,
        # full at synapse >= 1 (where conditioned == plain).
        'incoming': 'route_scope_paths/route_type_incoming_asked.csv',
        'incoming_full': 'route_scope_paths/route_type_incoming.csv',
        'vispath': (('route_type_network.html',
                     '🕸️ lane type network (vispath)'),
                    ('route_type_heatmap.html',
                     '🔥 lane type heatmap (vispath)')),
        'vispath_full': (('route_full_type_network.html',
                          '🕸️ lane full-scope network (vispath)'),
                         ('route_full_type_heatmap.html',
                          '🔥 lane full-scope heatmap (vispath)')),
        # Both lane scopes publish the whole admitted type-pair weight beside
        # their ratios, so the corridor explorer prints `Universe type
        # weights` per hop (F4/Q3).
        'universe': True},
}
# path_prob removed from the rank keys (SUSPENDED 2026-10-06):
RANK_KEYS = ('min_weight', 'min_ratio', 'length')

# Folders that never hold a paths table unit (viz/analysis sidecars).
SKIP_DIRS = {
    'data_details', 'visualization', 'visualization_data',
    'bodyId_visualization', 'network_early', 'network_early_bodyId',
    'find_reciprocal', 'hemisphere_symmetry', 'custom_groups',
    'comparison_results', 'comparison_visualizations',
    'comparison_report_used_data', 'similarity_matrices', 'conserved_paths',
    'type_level_refill', '_density', 'cache', '__pycache__',
}

# House palette (html_report_generator._generate_html_header) plus the
# shared/unique pair; green/amber/gray conservation semantics are avoided.
COLOR_SOURCE = '#ef4444'
COLOR_TARGET = '#8b5cf6'
COLOR_SHARED = '#2563eb'
COLOR_UNIQUE = '#94a3b8'

# Vendored vis-network build (offline interactivity). Falls back to the
# house CDN tag when the asset file is missing.
_VIS_NETWORK_PATH = Path(__file__).parent / 'assets' / 'vis-network.min.js'


def _vis_network_script() -> str:
    """The vis-network library as an inline <script> block, delegated to
    the shared ``vendored_assets`` helper (plan-offline-html-exports) so
    every exporter shares one inline/escape implementation. Falls back to
    the module's own read (and then the CDN tag) when the helper is
    unavailable.
    """
    try:
        from vendored_assets import inline_vis_network as _inline
    except ImportError:  # pragma: no cover - direct src/ execution
        try:
            from src.vendored_assets import inline_vis_network as _inline
        except ImportError:
            _inline = None
    if _inline is not None:
        return _inline()
    try:
        lib = _VIS_NETWORK_PATH.read_text(encoding='utf-8')
    except OSError:
        return ('<script src="https://unpkg.com/vis-network/standalone/'
                'umd/vis-network.min.js"></script>')
    lib = re.sub(r'</(script)', r'<\\/\1', lib, flags=re.IGNORECASE)
    return '<script>' + lib + '</script>'


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

LANE_DIR = 'route_scope_paths'
TYPED_LANE_DIR = 'route_scope_paths_type_pruned'
LANE_PROV_NAME = 'route_provenance.json'
LANE_WALKS_NAME = 'route_paths_type.csv'


@dataclass
class Unit:
    """One folder holding a ``*_allpaths_type.csv`` paths table."""
    unit_id: str            # run-root-relative, 'dataset_data/' stripped; '' = run root
    label: str              # display label
    folder: Path
    csv_path: Path
    dataset: str
    threshold: str          # best-effort numeric minsyn value ('' when absent)
    sort_key: Tuple
    lane_dir: str = LANE_DIR  # Fast or typed lane namespace


def _threshold_of(unit_id: str) -> str:
    match = re.search(r'minsyn_(\d+)', unit_id)
    if match:
        return match.group(1)
    match = re.search(r'minratio_([0-9neg_]+)', unit_id)
    return match.group(1) if match else ''


def discover_units(run_dir: Path) -> List[Unit]:
    """Find every folder under *run_dir* holding a paths table.

    A unit is any folder containing a file with the exact suffix
    ``*_allpaths_type.csv`` (never the ``_excluded``/``_group`` variants;
    ``data_details`` and the other sidecar folders are not scanned). The
    unit label is the raw folder path relative to the run root with the
    ``dataset_data/`` prefix stripped, so ``minsyn_5_applied_floor`` stays
    intact; the numeric threshold is parsed best-effort for sorting only.
    """
    units: List[Unit] = []
    run_dir = Path(run_dir)
    for dirpath, dirnames, filenames in os.walk(run_dir):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith('.')
            and d != OUTPUT_DIR_NAME
        )
        csvs = sorted(f for f in filenames if f.endswith(PATHS_SUFFIX))
        if not csvs:
            continue
        folder = Path(dirpath)
        rel = folder.relative_to(run_dir).as_posix()
        unit_id = '' if rel == '.' else re.sub(r'^dataset_data/', '', rel)
        parts = unit_id.split('/') if unit_id else []
        dataset = parts[0] if len(parts) >= 2 else ''
        threshold = _threshold_of(unit_id)
        try:
            threshold_sort = float(threshold) if threshold else -1.0
        except ValueError:  # pragma: no cover - regex guarantees digits
            threshold_sort = -1.0
        units.append(Unit(
            unit_id=unit_id,
            label=unit_id if unit_id else '(run root)',
            folder=folder,
            csv_path=folder / csvs[0],
            dataset=dataset,
            threshold=threshold,
            sort_key=(dataset, threshold_sort, unit_id),
        ))
    units.sort(key=lambda u: u.sort_key)
    return units


def _lane_provenance(folder: Path, lane_dir: str = LANE_DIR) -> dict:
    """The lane's own provenance for a folder, or {} when it never ran.

    ``status == 'complete'`` is the gate: a refused lane writes provenance
    too, and a refusal is disclosure for the corridor CARD, not a dataset."""
    prov_path = Path(folder) / lane_dir / LANE_PROV_NAME
    if not prov_path.exists():
        return {}
    try:
        prov = json.loads(prov_path.read_text(encoding='utf-8'))
    except Exception:  # noqa: BLE001 - unreadable provenance == no lane
        return {}
    return prov if str(prov.get('status') or '') == 'complete' else {}


def lane_unit(run_dir: Path, meta: Dict[str, Any]) -> Optional[Unit]:
    """The unit shape of a run whose ONLY path table is the lane's.

    A standalone Fast run (``corridor_mode='fast'``) skips the Complete
    pipeline entirely, so it has no ``*_allpaths_type.csv`` and
    ``discover_units`` returns []. Its ``route_scope_paths/
    route_paths_type.csv`` is written in the same column order, so treating
    it as the run's paths table gives the lane a real report — Overview,
    pair tables, Explorer — instead of an empty state that describes a run
    with 7 validated walks as one that found nothing (F-EX-010's sibling,
    plan-corridor-refill-and-mode-tabs D2/G8)."""
    run_dir = Path(run_dir)
    for lane_dir in (LANE_DIR, TYPED_LANE_DIR):
        csv_path = run_dir / lane_dir / LANE_WALKS_NAME
        if not csv_path.exists() or not _lane_provenance(run_dir, lane_dir):
            continue
        return Unit(
            unit_id='',
            label=_root_display_label(
                run_dir, meta, run_kind_for(run_dir, meta)),
            folder=run_dir, csv_path=csv_path,
            dataset=str(meta.get('dataset') or ''), threshold='',
            sort_key=('', -1.0, ''),
            lane_dir=lane_dir,
        )
    return None


# ---------------------------------------------------------------------------
# run metadata (opportunistic — the CSVs are the only hard requirement)
# ---------------------------------------------------------------------------

def run_kind(run_name: str) -> str:
    if run_name.startswith('find-paths-shortest'):
        return 'Shortest Paths'
    if run_name.startswith('find-paths-complete'):
        return 'Complete Paths'
    if run_name.startswith('find-network'):
        # A Network run writes a 1-hop *_allpaths_type.csv precisely so the
        # Pair Explorer reads it (Phase B3) — but it is not a pathfinding run,
        # and labelling it 'Pathfinding' made the report describe a direct
        # connection as a searched route and quote depth/lane advice the tool
        # can never act on (plan 2026-10-16 C3).
        return 'Network'
    if run_name.startswith('cross-dataset'):
        return 'Cross-Dataset > Paths'
    return 'Pathfinding'


def _parameters_value(run_dir, key: str) -> str:
    """One `key:<spaces>value` line of the folder's own parameters.txt (''
    when absent). `_sniff_metadata` normalizes into its own vocabulary and
    drops the keys this needs, so the stamp is read from the bytes."""
    try:
        lines = (Path(run_dir) / 'parameters.txt').read_text(
            encoding='utf-8', errors='replace').splitlines()
    except OSError:
        return ''
    for line in lines:
        head, sep, tail = line.strip().partition(':')
        if sep and head.strip() == key:
            return tail.strip()
    return ''


def run_kind_for(run_dir, meta=None) -> str:
    """The tool that wrote THIS folder, falling back to what the folder says.

    `run_kind` reads the run-name grammar, which is how a folder is recognized
    by eye — but the group also writes folders under a user's `Save As` name,
    and those carried no kind at all: the report fell back to 'Pathfinding' and
    re-quoted the depth/lane advice a Network run cannot act on (the C3 fix was
    therefore only true of a NAMED folder). Every tool has stamped its own
    `path_mode` since plan 2026-10-16 Q4, so a name that settles nothing is
    answered from the folder. Applied only as a fallback: a named folder keeps
    its established reading, and a cross-dataset unit folder is never a root
    here (the report is generated for the `cross-dataset_*` root).
    """
    name = Path(run_dir).name
    kind = run_kind(name)
    if kind != 'Pathfinding':
        return kind
    return {
        'network': 'Network',
        'shortest': 'Shortest Paths',
        'all': 'Complete Paths',
    }.get(_parameters_value(run_dir, 'path_mode').strip().lower(), kind)


# Applied-threshold / budget provenance keys (the block pathfinding runs
# append to parameters.txt; same fields the cross-dataset report marks).
PROVENANCE_KEYS = (
    'requested_threshold', 'applied_threshold', 'applied_threshold_source',
    'strongest_first_budget', 'strongest_first_budget_bitten',
    'strongest_first_tau', 'tau_canonical', 'strongest_dropped_bottleneck',
    'edge_budget', 'edge_budget_applied', 'edge_budget_landing',
    'edge_weight_floor', 'strongest_retained_bottleneck', 'paths_complete',
    # Coverage early-stop knobs (shortest); the realized stop layer lives
    # in all_attributes.json (surfaced by the card/table renderers).
    'shortest source coverage', 'shortest target coverage',
)


def _sniff_metadata(run_dir: Path) -> Dict[str, Any]:
    """Best-effort run metadata; every field may come back missing."""
    meta: Dict[str, Any] = {}
    params_txt = run_dir / 'parameters.txt'
    if params_txt.exists():
        try:
            text = params_txt.read_text(encoding='utf-8', errors='replace')
        except OSError:
            text = ''
        for key, pattern in (
            ('min_synapse', r'min synapse number:\s*([^\s]+)'),
            ('applied_threshold', r'applied_threshold:\s*([^\s]+)'),
            ('min_ratio', r'min connection ratio:\s*([^\s]+)'),
            ('weight_basis', r'weight basis:\s*([^\s]+)'),
            ('run_mode', r'run mode:\s*([^\s]+)'),
            ('threshold_combination', r'threshold combination:\s*([^\s]+)'),
            ('max_interlayer', r'max interlayer:\s*([^\s]+)'),
            ('dataset', r'^dataset:\s*([^\s]+)'),
            ('filter_by', r'filter by:\s*([^\s]+)'),
            ('separate_hemispheres', r'separate hemispheres:\s*([^\s]+)'),
        ):
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                meta[key] = match.group(1)
        # applied-threshold / budget provenance block (explicitly marks
        # what the run actually used, incl. StrongestFirst budget/tau and
        # Edge Budget landing — the pathfinding analogue of the
        # cross-dataset report's provenance section)
        provenance = {}
        prov_pattern = re.compile(
            r'^(' + '|'.join(PROVENANCE_KEYS) + r'):\s*(.+)$', re.MULTILINE)
        for match in prov_pattern.finditer(text):
            provenance[match.group(1)] = match.group(2).strip()
        if provenance:
            meta['provenance'] = provenance
    if 'provenance' not in meta and (run_dir / 'dataset_data').is_dir():
        # Cross-dataset run root: no root-level parameters.txt/provenance
        # block — aggregate the per-delegate blocks so the report can show
        # one applied-threshold row per delegate.
        delegate_prov: Dict[str, Dict[str, str]] = {}
        line_re = re.compile(
            r'^(' + '|'.join(PROVENANCE_KEYS) + r'):\s*(.+)$', re.MULTILINE)
        for params_path in sorted(
                (run_dir / 'dataset_data').glob('*/*/parameters.txt')):
            unit_id = re.sub(
                r'^dataset_data/', '',
                params_path.relative_to(run_dir).as_posix()).replace(
                    '/parameters.txt', '')
            try:
                delegate_text = params_path.read_text(
                    encoding='utf-8', errors='replace')
            except OSError:
                continue
            parsed: Dict[str, str] = {}
            for match in line_re.finditer(delegate_text):
                parsed[match.group(1)] = match.group(2).strip()
            if parsed:
                delegate_prov[unit_id] = parsed
        if delegate_prov:
            meta['delegate_provenance'] = delegate_prov
    manifest = run_dir / 'run_manifest.json'
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding='utf-8'))
            params = data.get('parameters', {})
            meta.setdefault('datasets', data.get('datasets'))
            meta.setdefault('nicknames', data.get('nicknames'))
            meta.setdefault('path_mode', params.get('path_mode'))
            meta.setdefault('comparison_mode', params.get('comparison_mode'))
            meta.setdefault('thresholds', params.get('thresholds'))
            meta.setdefault('max_interlayer', params.get('max_interlayer'))
        except (OSError, ValueError):
            pass
    return meta


# ---------------------------------------------------------------------------
# per-unit breakdown
# ---------------------------------------------------------------------------

def _is_untyped_node(name: str) -> bool:
    """Untyped node label: empty/sentinel or the bodyId-fallback form
    (digit-only, optionally hemisphere-suffixed) — same predicate family
    as ``utils.label_utils.is_untyped_type_label``."""
    s = str(name).strip()
    if s.endswith(('_L', '_R', '_U')):
        s = s[:-2].strip()
    if not s or s.lower() in {'unknown', 'nan', 'none', 'null',
                              '<na>', '<null>'}:
        return True
    return s.isdigit()


def load_unit_paths(csv_path: Path, drop_untyped: bool = True) -> pd.DataFrame:
    """Load one paths CSV and derive the pair columns the report needs.

    ``drop_untyped`` (default on, 2026-10-09): rows whose ``path``
    traverses an untyped node — an empty/sentinel label or the
    bodyId-fallback form (digit-only) — are dropped, matching the
    comparison-level Drop Untyped Neurons semantics; a 2-hop path whose
    middle node is a bodyId string is discovery noise, not a pair."""
    df = pd.read_csv(csv_path, dtype={'path': str})
    if 'path' not in df.columns:
        raise ValueError(f'{csv_path}: missing required "path" column')
    df = df[df['path'].notna()].copy()
    if drop_untyped and not df.empty:
        nodes = df['path'].str.split('->')
        untyped = nodes.map(
            lambda segs: any(_is_untyped_node(s) for s in segs))
        if untyped.any():
            df = df.loc[~untyped].copy()
    nodes = df['path'].str.split('->')
    df['_source'] = nodes.str[0]
    df['_target'] = nodes.str[-1]
    df['_length'] = nodes.str.len() - 1
    if 'min_weight' not in df.columns:
        df['min_weight'] = float('nan')
    if 'min_ratio' not in df.columns:
        df['min_ratio'] = float('nan')
    if 'path_prob' not in df.columns:
        df['path_prob'] = float('nan')
    if 'weights' not in df.columns:
        df['weights'] = ''
    if 'ratios' not in df.columns:
        df['ratios'] = ''
    if 'coverage' not in df.columns:
        df['coverage'] = ''
    df['min_weight'] = pd.to_numeric(df['min_weight'], errors='coerce')
    df['min_ratio'] = pd.to_numeric(df['min_ratio'], errors='coerce')
    df['path_prob'] = pd.to_numeric(df['path_prob'], errors='coerce')
    return df


def _rank_sort(frame: pd.DataFrame, rank_by: str) -> pd.DataFrame:
    # The former ``path_prob`` tiebreak is SUSPENDED (2026-10-06,
    # unverifiable fidelity): ties resolve on the path string.
    ascending = [rank_by == 'length', True]
    return frame.sort_values(
        [rank_by, '_path_key'],
        ascending=ascending, na_position='last', kind='mergesort')


def build_unit_breakdown(
    frame: pd.DataFrame, rank_by: str,
    source_cov: Optional[Dict[str, str]] = None,
    target_cov: Optional[Dict[str, str]] = None,
    pair_bodyid: Optional[Dict[Tuple[str, str], Tuple[int, int]]] = None,
    pair_ratio: Optional[Dict[Tuple[str, str], float]] = None,
) -> Tuple[List[dict], List[dict], Dict[Tuple[str, str], dict]]:
    """Per-pair breakdown of one unit.

    Returns (path rows, intermediate rows, pair stats keyed
    ``(source, target)``). Rank is bottleneck-first within (pair, length);
    shared/unique classification is over the pair's FULL path set.

    Coverage is counted PER PATH and PER PAIR (2026-10-04): each path
    row carries the FIRST/LAST entries of its own per-node coverage list
    (distinct bodyIds realizing that path), and each pair stat carries
    the pair's DISTINCT endpoint bodyIds when bodyId-level output exists
    (``pair_bodyid``); without it the pair falls back to the type's
    query-scope value with ``cov_scope='type'`` (exact pairs say
    ``'pair'``).
    """
    source_cov = source_cov or {}
    target_cov = target_cov or {}
    pair_bodyid = pair_bodyid or {}
    frame = frame.copy()
    frame['_path_key'] = frame['path']
    ordered = _rank_sort(frame, rank_by)

    path_rows: List[dict] = []
    inter_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    inter_minhop: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(dict)
    pair_stats: Dict[Tuple[str, str], dict] = {}

    for (source, target), group in ordered.groupby(['_source', '_target'], sort=True):
        lengths = Counter(int(v) for v in group['_length'])
        pair_rows: List[dict] = []
        for length, length_group in group.groupby('_length', sort=True):
            for rank, (_, row) in enumerate(length_group.iterrows(), start=1):
                nodes = str(row['path']).split('->')
                for hop, node in enumerate(nodes[1:-1], start=1):
                    key = (source, target)
                    inter_counts[key][node] += 1
                    prior = inter_minhop[key].get(node)
                    if prior is None or hop < prior:
                        inter_minhop[key][node] = hop
                ratios = row.get('ratios', '')
                coverage = row.get('coverage', '')
                path_scov, path_tcov = _path_end_coverage(coverage)
                pair_rows.append({
                    'source': source,
                    'target': target,
                    'pair': f'{source}->{target}',
                    'rank_in_pair_length': rank,
                    'path': str(row['path']),
                    'weights': '' if pd.isna(row['weights']) else str(row['weights']),
                    'ratios': '' if pd.isna(ratios) else str(ratios),
                    'coverage': '' if pd.isna(coverage) else str(coverage),
                    'min_weight': row['min_weight'],
                    'min_ratio': row.get('min_ratio'),
                    'pair_min_ratio': _path_pair_ratio(
                        row['path'], pair_ratio or {}),
                    'length': int(length),
                    'source_bodyid_coverage': (
                        path_scov or source_cov.get(source, '')),
                    'target_bodyid_coverage': (
                        path_tcov or target_cov.get(target, '')),
                })
        exact = pair_bodyid.get((source, target))
        if exact is not None:
            stats_scov = (f'{exact[0]}/{_cov_denominator(source_cov.get(source))}'
                          if _cov_denominator(source_cov.get(source)) else '')
            stats_tcov = (f'{exact[1]}/{_cov_denominator(target_cov.get(target))}'
                          if _cov_denominator(target_cov.get(target)) else '')
            cov_scope = 'pair'
        else:
            stats_scov = source_cov.get(source, '')
            stats_tcov = target_cov.get(target, '')
            cov_scope = 'type'
        stats = {
            'paths': len(pair_rows),
            'lengths': dict(sorted(lengths.items())),
            'shared': sum(1 for v in inter_counts[(source, target)].values() if v >= 2),
            'unique': sum(1 for v in inter_counts[(source, target)].values() if v == 1),
            'scov': stats_scov,
            'tcov': stats_tcov,
            'cov_scope': cov_scope,
        }
        pair_stats[(source, target)] = stats
        path_rows.extend(pair_rows)

    inter_rows: List[dict] = []
    for (source, target), counts in inter_counts.items():
        for node, count in counts.items():
            inter_rows.append({
                'source': source,
                'target': target,
                'intermediate': node,
                'n_paths_using': int(count),
                'classification': 'shared' if count >= 2 else 'unique',
                'min_hop_position': int(inter_minhop[(source, target)][node]),
            })
    inter_rows.sort(key=lambda r: (r['source'], r['target'],
                                   -r['n_paths_using'], r['intermediate']))
    return path_rows, inter_rows, pair_stats


# ---------------------------------------------------------------------------
# payload builders
# ---------------------------------------------------------------------------

def _fmt_num(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '—'
    try:
        return f'{float(value):g}'
    except (TypeError, ValueError):
        return str(value)


def _ratio_list(ratios: str) -> Optional[List[float]]:
    try:
        parsed = json.loads(ratios)
        if isinstance(parsed, list):
            return [float(v) for v in parsed]
    except (TypeError, ValueError):
        pass
    return None


def _weight_list(weights: str) -> Optional[List[float]]:
    try:
        parsed = json.loads(weights)
        if isinstance(parsed, list):
            return [float(v) for v in parsed]
    except (TypeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# bodyId coverage (round 8b, user item 1: n/N for source and target only)
# ---------------------------------------------------------------------------

def _load_enrollment_coverage(
        folder: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Per-type bodyId coverage ``n/N`` for the pair pane (round-8b
    item 1): source = bodyIds with ``isInPath`` / all enrolled bodyIds of
    the type (``source_neurons.csv``); target = bodyIds reached
    (``Checked``) / all resolved bodyIds of the type
    (``target_neurons.csv``). Missing files yield empty strings."""
    folder = Path(folder)
    source_cov: Dict[str, str] = {}
    target_cov: Dict[str, str] = {}
    src_path = folder / 'source_neurons.csv'
    if src_path.exists():
        try:
            frame = pd.read_csv(src_path, usecols=['type', 'isInPath'])
            grouped = frame.groupby('type')['isInPath']
            for type_name, values in grouped:
                flags = values.fillna(False).astype(str).str.lower() == 'true'
                source_cov[str(type_name)] = f'{int(flags.sum())}/{len(flags)}'
        except Exception:  # noqa: BLE001 - coverage is best-effort
            pass
    tgt_path = folder / 'target_neurons.csv'
    if tgt_path.exists():
        try:
            frame = pd.read_csv(tgt_path, usecols=['type', 'Checked'])
            grouped = frame.groupby('type')['Checked']
            for type_name, values in grouped:
                flags = values.fillna(False).astype(str).str.lower() == 'true'
                target_cov[str(type_name)] = f'{int(flags.sum())}/{len(flags)}'
        except Exception:  # noqa: BLE001 - coverage is best-effort
            pass
    return source_cov, target_cov


_COV_ELEMENT_RE = re.compile(r'(\d+)\s*/\s*(\d+)')


def _path_end_coverage(coverage_value):
    """Per-path source/target bodyId coverage: the FIRST and LAST ``n/N``
    entries of the row's per-node coverage list (round 9) — distinct
    bodyIds realizing THIS path at its endpoints / the type's members in
    the run. ``(None, None)`` when the row carries no coverage list."""
    text = '' if coverage_value is None else str(coverage_value)
    entries = _COV_ELEMENT_RE.findall(text)
    if not entries:
        return None, None
    return (f'{entries[0][0]}/{entries[0][1]}',
            f'{entries[-1][0]}/{entries[-1][1]}')


def _cov_denominator(n_over_n: str) -> str:
    """The ``N`` of an ``n/N`` coverage string ('' when unparseable)."""
    m = _COV_ELEMENT_RE.search(str(n_over_n or ''))
    return m.group(2) if m else ''


def _load_query_coverage(folder) -> Dict[str, Tuple[int, int]]:
    """Query-scope bodyId coverage totals for one unit: source = bodyIds
    with ``isInPath`` / all enrolled; target = bodyIds ``Checked`` / all
    resolved. Best-effort — missing files simply omit the side."""
    folder = Path(folder)
    out: Dict[str, Tuple[int, int]] = {}
    try:
        frame = pd.read_csv(folder / 'source_neurons.csv',
                            usecols=['isInPath'])
        flags = (frame['isInPath'].fillna(False).astype(str).str.lower()
                 == 'true')
        out['source'] = (int(flags.sum()), len(flags))
    except Exception:  # noqa: BLE001 - coverage is best-effort
        pass
    try:
        frame = pd.read_csv(folder / 'target_neurons.csv',
                            usecols=['Checked'])
        flags = (frame['Checked'].fillna(False).astype(str).str.lower()
                 == 'true')
        out['target'] = (int(flags.sum()), len(flags))
    except Exception:  # noqa: BLE001
        pass
    return out


def _load_pair_level_ratios(folder
                            ) -> Tuple[Dict[Tuple[str, str], float], str]:
    """(type_pre, type_post) -> pair-level ratio from the run's own
    data_details/connection_type.csv (the ratio is conn_layer-invariant
    after the recompute). Prefers ``connection_ratio_adj`` — the adjusted
    pair-level ratio (deduped recurring-pair numerators / refill-aware)
    — falling back to plain ``connection_ratio``. Returns
    ``(mapping, source)`` with source in {'adjusted', 'plain', ''} so the
    Pair-ratio column can label which claim it shows."""
    out: Dict[Tuple[str, str], float] = {}
    source = ''
    try:
        conn_csv = Path(folder) / 'data_details' / 'connection_type.csv'
        if not conn_csv.exists():
            return out, source
        ct = pd.read_csv(conn_csv)
        col = None
        for candidate, tag in (('connection_ratio_adj', 'adjusted'),
                               ('connection_ratio', 'plain')):
            if {'type_pre', 'type_post', candidate} <= set(ct.columns):
                col, source = candidate, tag
                break
        if col is None:
            return out, source
        # Two row shapes share the table (2026-10-05 real-data audit,
        # heuristic corrected 2026-10-07): recurring pairs whose value is
        # a pair-level STAMP repeated identically on every conn_layer row
        # (the pair value is the row value; summing would overcount by
        # the row count), and SPLIT pairs carrying each depth's realized
        # share (row values differ; the pair value is their SUM — the
        # rows share the post type's denominator, so the shares add).
        # The discriminator is the VALUE column's own constancy, never
        # the weight shape: round-15 stamps the pair-level adjusted
        # ratio as a row-constant even when the per-depth weights differ
        # — keying on weights summed k identical stamps for exactly the
        # split-shaped pairs this loader was written for.
        grouped = ct.groupby(['type_pre', 'type_post'])
        for (pre, post), grp in grouped:
            rows_col = grp[col].dropna()
            if rows_col.empty:
                continue
            vals = rows_col.astype(float)
            split = (len(vals) > 1
                     and float(vals.max()) != float(vals.min()))
            value = (float(vals.sum()) if split
                     else float(vals.iloc[0]))
            # One precision per quantity across the report: every ratio an
            # export table publishes is stored at 6 decimals, so the
            # pair-level ratio read from connection_type.csv is quantized
            # the same way — otherwise the same pair renders 0.001329 in
            # one tab's payload and 0.0013286659682653 in another's.
            out[(str(pre), str(post))] = round(value, 6)
    except (OSError, ValueError):
        pass
    return out, source


def _path_pair_ratio(path_str, pair_ratio) -> Optional[float]:
    """Weakest PAIR-level mass share among a path's hops (None when the
    conn table carries none of the path's pairs — e.g. legacy runs)."""
    labs = str(path_str).split('->')
    vals = [pair_ratio.get((labs[i], labs[i + 1]))
            for i in range(len(labs) - 1)]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def _load_refill_pair_ratios(
        folder: Path, pairs_name: str = 'refill_type_pairs.csv',
        prefer_adj: bool = True,
        subdir: str = 'data_details/type_level_refill'
) -> Tuple[Dict[Tuple[str, str], float], str]:
    """(type_pre, type_post) -> refilled pair-level ratio for the Refill
    Explorer, from <subdir>/<pairs_name> — the complete mode's second round
    mirrors into data_details/type_level_refill/, the corridor lane keeps
    its own namespace (route_scope_paths/). Prefers
    ``refilled_connection_ratio_adj`` (the threshold-conditioned
    denominator) over plain ``refilled_connection_ratio`` — the same
    preference order as the main table's loader. The FULL scope
    (refill_full_type_pairs.csv) passes prefer_adj=False: its admission
    floor is synapse >= 1, where the conditioned denominator equals the
    plain all-inputs one, so the plain full ratio IS the adjusted value.
    Returns ``(mapping, source)`` with source in
    {'adjusted', 'plain', 'full-plain', ''}."""
    out: Dict[Tuple[str, str], float] = {}
    source = ''
    try:
        pairs_csv = Path(folder) / subdir / pairs_name
        if not pairs_csv.exists():
            return out, source
        frame = pd.read_csv(pairs_csv)
        col = None
        candidates = (('refilled_connection_ratio_adj', 'adjusted'),
                      ('refilled_connection_ratio', 'plain')) \
            if prefer_adj else \
            (('refilled_connection_ratio', 'full-plain'),)
        if pairs_name.startswith('route_connection_type'):
            # the corridor lane's asked-tier pair table names its shares
            # the way every other type-pair table does
            candidates = (('connection_ratio_adj', 'adjusted'),
                          ('connection_ratio', 'plain')) if prefer_adj \
                else (('connection_ratio', 'full-plain'),)
        for candidate, tag in candidates:
            if {'type_pre', 'type_post', candidate} <= set(frame.columns):
                col, source = candidate, tag
                break
        if col is None:
            return out, source
        for _, r in frame.iterrows():
            if pd.notna(r.get(col)):
                out[(str(r['type_pre']), str(r['type_post']))] = float(
                    r[col])
    except (OSError, ValueError):
        pass
    return out, source


def _load_pair_universe(
        folder: Path, pairs_name: str, subdir: str,
        columns: Tuple[str, ...] = ('universe_type_weights',)
) -> Dict[Tuple[str, str], float]:
    """(type_pre, type_post) -> the WHOLE admitted type-pair weight at this
    scope's tier, from <subdir>/<pairs_name> — the per-hop
    `Universe type weights` readout (F4/Q3, renamed from `universe pair
    mass`; the quantity is unchanged). Empty when the table publishes no
    such column, which is how a dataset opts out of the column rather than
    printing a wrong one."""
    out: Dict[Tuple[str, str], float] = {}
    try:
        frame = pd.read_csv(Path(folder) / subdir / pairs_name)
        col = next((c for c in columns
                    if {'type_pre', 'type_post', c} <= set(frame.columns)),
                   None)
        if col is None:
            return out
        for _, r in frame.iterrows():
            if pd.notna(r.get(col)):
                out[(str(r['type_pre']), str(r['type_post']))] = float(
                    r[col])
    except (OSError, ValueError):
        pass
    return out


def _hop_universe_list(path_str, universe: Dict[Tuple[str, str], float]) -> str:
    """Per-hop universe weights in the presence-matrix list form the explorer
    prints (`dashList` bolds the minimum, and an em dash marks a hop whose
    pair the table does not publish). Masses are integers when they are
    integers — a synapse count never needs a decimal tail."""
    labs = str(path_str).split('->')
    parts = []
    for i in range(len(labs) - 1):
        v = universe.get((labs[i], labs[i + 1]))
        if v is None:
            parts.append('\u2014')
        else:
            parts.append(str(int(v)) if float(v).is_integer()
                         else repr(round(float(v), 2)))
    return '[' + ', '.join(parts) + ']'


# ---------------------------------------------------------------------------
# Percent convention (plan §2.5, 2026-10-05 user directive): every ratio
# and probability the REPORT renders is xx.xx% (2 decimals). The breakdown
# CSV keeps raw values; only report payloads re-format.
# ---------------------------------------------------------------------------

def _fmt_pct(value) -> str:
    """One ratio/probability as ``xx.xx%`` (raw passthrough when NaN)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return '—' if value in (None, '') else str(value)
    if v != v:  # NaN
        return '—'
    return f'{100.0 * v:.2f}%'


_RATIO_LIST_RE = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')


def _format_ratio_list(text) -> str:
    """'[0.0263, 0.060]' -> '[2.63%, 6.00%]'; None entries (unresolvable
    post-type incoming, 2026-10-10) pass through positionally and render
    '—' via _fmt_pct (passthrough when the string carries no parseable
    numbers — legacy/hand-edited payloads)."""
    raw = '' if text is None else str(text)
    if not _RATIO_LIST_RE.search(raw):
        return raw
    parts = _RATIO_LIST_RE.findall(raw)
    # None entries are not numbers; findall would silently SHIFT the
    # remaining ratios left. Split on the list's own commas instead and
    # format each element positionally.
    inner = raw.strip()
    if inner.startswith('[') and inner.endswith(']'):
        inner = inner[1:-1]
    if 'None' in inner:
        elems = [e.strip() for e in inner.split(',') if e.strip()]
        return '[' + ', '.join(
            '—' if e == 'None' else _fmt_pct(float(e)) for e in elems) + ']'
    return '[' + ', '.join(_fmt_pct(p) for p in parts) + ']'


def _per_hop_adjusted_list(path_str, pair_ratio) -> str:
    """Per-hop ADJUSTED ratio list for one path: connection_ratio_adj of
    each (type_pre, type_post) hop from the loaded pair map, percent-
    formatted; hops the map lacks render '—'."""
    labs = str(path_str).split('->')
    parts = []
    for i in range(len(labs) - 1):
        v = (pair_ratio or {}).get((labs[i], labs[i + 1]))
        parts.append(_fmt_pct(v) if v is not None else '—')
    return '[' + ', '.join(parts) + ']' if parts else ''


def _load_type_incoming_conditioned(folder) -> Dict[str, float]:
    """Per post TYPE: the dataset-wide incoming mass CONDITIONED at the
    run's Min Synapse Count (data_details/type_incoming_conditioned.csv,
    written by the engine from the same fetch that produced the pair-level
    `connection_ratio_adj`). This is the denominator a CHAIN row's adjusted
    ratio divides by — the run's own guide states it ("the denominator is
    the dataset-wide incoming mass, inputs >= threshold for adjusted"), and
    at Min Synapse Count 1 it equals the all-inputs total, so adj == plain
    by construction.

    It is deliberately NOT `type_incoming_emitted.csv` (that file sums this
    run's emitted cone: measured PPL101 1,283 vs the dataset's 39,137, which
    inflated chain adjusted ratios ~30x and let them saturate at 100%).
    Empty when the run predates the file — the caller then falls back to the
    pair-level adjusted map, which is at least the documented
    `connection_ratio_adj`."""
    out: Dict[str, float] = {}
    path = Path(folder) / 'data_details' / 'type_incoming_conditioned.csv'
    try:
        frame = pd.read_csv(path)
        if {'type', 'incoming'} <= set(frame.columns):
            for _, r in frame.iterrows():
                if pd.notna(r['incoming']):
                    out[str(r['type'])] = float(r['incoming'])
    except (OSError, ValueError):
        pass
    return out


def _conditioned_threshold(folder) -> str:
    """The tier `data_details/type_incoming_conditioned.csv` was computed
    at, as the engine recorded it (its `threshold` column). The Adj header
    must name THAT number — the run's `applied_threshold` is a different
    cut (the emitted-set floor), and labelling the conditioned denominator
    with it showed tier 4 over a denominator conditioned at 3."""
    path = Path(folder) / 'data_details' / 'type_incoming_conditioned.csv'
    try:
        frame = pd.read_csv(path, usecols=['threshold'])
    except (OSError, ValueError):
        return ''
    for value in frame['threshold'].dropna().unique():
        text = str(value).strip()
        if text and text.lower() != 'none':
            return text.rstrip('0').rstrip('.') if '.' in text else text
    return ''


def _opt_cell(value) -> str:
    """A passthrough list cell that is simply absent when the source table
    has no such column (carrying tables carry emitted/total; the others do
    not) — never a fabricated dash."""
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except (TypeError, ValueError):
        pass
    return str(value)


def _realized_conditioned_list(path_str, weights, in_conditioned) -> str:
    """Per-hop ADJUSTED ratio list computed from the REALIZED hop
    masses over the CONDITIONED dataset-wide incoming denominator — the
    chain-derived measure the run's guide and the pair-level
    `connection_ratio_adj` both describe: numerator = this chain's realized
    hop mass, denominator = the post type's dataset incoming counting only
    edges at or above the run's Min Synapse Count. At Min Synapse Count 1
    that denominator IS the all-inputs one, so the list equals the plain
    Ratios list by construction. Hops whose post type lacks a conditioned
    denominator render '—'. Empty when ``in_conditioned`` is None
    (legacy run) — the caller falls back to the pair-level adj map."""
    if not in_conditioned:
        return ''
    labs = str(path_str).split('->')
    ws = _weight_list(weights) or []
    parts = []
    for i in range(len(labs) - 1):
        denom = in_conditioned.get(labs[i + 1])
        w = ws[i] if i < len(ws) else None
        if denom and w is not None:
            parts.append(_fmt_pct(w / denom))
        else:
            parts.append('—')
    return '[' + ', '.join(parts) + ']' if parts else ''


def _load_pair_bodyid_coverage(folder):
    """Distinct endpoint bodyIds per (source type, target type) pair,
    counted over the bodyId-level paths table (per-path, per-pair).

    Returns ``{}`` when the run skipped bodyId-level output
    (``skip_bodyId``) — pair-scope coverage then falls back to the
    type's query-scope value, labeled as such."""
    folder = Path(folder)
    bid_path = next(iter(sorted(folder.glob(
        '*_allpaths_bodyId_paths.csv'))), None)
    if bid_path is None:
        # skip_bodyId runs persist the per-pair distinct counts at RUN
        # time (the bodyId paths themselves are never saved) — read the
        # summary instead; equally exact for pair-scope coverage.
        summary = folder / 'data_details' / 'pair_bodyid_coverage.csv'
        if not summary.exists():
            return {}
        try:
            frame = pd.read_csv(summary)
            return {
                (str(r.source_type), str(r.target_type)):
                    (int(r.distinct_source_bodyids),
                     int(r.distinct_target_bodyids))
                for r in frame.itertuples(index=False)
            }
        except Exception:  # noqa: BLE001
            return {}
    try:
        frame = pd.read_csv(bid_path, usecols=['path'])
    except Exception:  # noqa: BLE001
        return {}
    type_maps = {}
    for side, fname in (('src', 'source_neurons.csv'),
                        ('tgt', 'target_neurons.csv')):
        try:
            f = pd.read_csv(folder / fname, usecols=['bodyId', 'type'])
            type_maps[side] = {
                str(b): str(t)
                for b, t in zip(f['bodyId'], f['type'])
            }
        except Exception:  # noqa: BLE001
            type_maps[side] = {}
    if not type_maps.get('src') or not type_maps.get('tgt'):
        return {}
    src_ids = defaultdict(set)
    tgt_ids = defaultdict(set)
    for chain in frame['path'].astype(str):
        nodes = chain.split('->')
        if len(nodes) < 2:
            continue
        source = type_maps['src'].get(nodes[0])
        target = type_maps['tgt'].get(nodes[-1])
        if source is None or target is None:
            continue
        src_ids[(source, target)].add(nodes[0])
        tgt_ids[(source, target)].add(nodes[-1])
    return {key: (len(src_ids[key]), len(tgt_ids[key]))
            for key in src_ids}


def _root_display_label(run_dir: Path, meta: Dict[str, Any],
                        fallback: str) -> str:
    """Informative label for a report's own run-root unit — '(run root)'
    is retired (round-8 item 5). Single-dataset runs label by dataset +
    min synapse; nested delegate reports fall back to their folder name
    (e.g. ``minsyn_3``) or the dataset alone; the run name is the last
    resort."""
    dataset = str(meta.get('dataset') or '').strip()
    min_syn = str(meta.get('min_synapse') or '').strip()
    if str(meta.get('weight_basis') or '').startswith('connection_ratio'):
        min_r = str(meta.get('min_ratio') or '').strip()
        if dataset and min_r:
            # Round-13: ratio+AND runs keep a synapse floor — label it.
            if (str(meta.get('threshold_combination') or '')
                    .strip().lower().startswith('and')
                    and min_syn and min_syn not in ('0', '1', 'None')):
                return (f'{dataset} · min ratio {min_r} '
                        f'· and synapse {min_syn}')
            return f'{dataset} · min ratio {min_r}'
    if dataset and min_syn and min_syn not in ('0', 'None'):
        min_r = str(meta.get('min_ratio') or '').strip()
        # Round-13: synapse+AND runs co-threshold by ratio — label it
        # (plain synapse runs carry min connection ratio 0.0).
        if (str(meta.get('threshold_combination') or '')
                .strip().lower().startswith('and')
                and min_r and min_r not in ('0', '0.0', 'None')):
            return (f'{dataset} · min synapse {min_syn} '
                    f'· and ratio {min_r}')
        return f'{dataset} · min synapse {min_syn}'
    if dataset:
        return dataset
    if run_dir.name.startswith('minsyn_'):
        return run_dir.name
    return fallback or run_dir.name


def build_viz(rows: Sequence[dict], inter_counts: Counter,
              inter_minhop: Dict[str, int]) -> dict:
    """Layered-DAG layout for the drawn (capped) path set.

    Columns: source, intermediates at their minimum full-set hop position,
    target. Within-column order is first-seen over the drawn rows, so the
    layout is deterministic. Node classes come from the FULL pair path set
    (shared >=2 paths, unique ==1).
    """
    max_len = max(len(str(r['path']).split('->')) for r in rows)
    ncols = max(max_len, 2)  # L0 runs keep source and target in separate columns
    node_cls: Dict[str, str] = {}
    node_col: Dict[str, int] = {}
    node_title: Dict[str, str] = {}

    def place(node: str, col: int, cls: str, title: str) -> None:
        if node not in node_col:
            node_col[node] = col
            node_cls[node] = cls
            node_title[node] = title

    edge_count: Counter = Counter()
    edge_weight: Dict[Tuple[str, str], float] = defaultdict(float)
    # round-8 item: per-edge ratio mass (sum of hop_ratio * hop_weight) —
    # the weight-weighted mean ratio pairs naturally with the summed
    # synapse weight in the hover label.
    edge_ratio_mass: Dict[Tuple[str, str], float] = defaultdict(float)
    edge_ratio_count: Counter = Counter()
    paths: List[List[str]] = []

    for row in rows:
        path = str(row['path'])
        ids = path.split('->')
        paths.append(ids)
        hop_weights = _weight_list(str(row.get('weights', '')))
        for index, node in enumerate(ids):
            if index == 0:
                place(node, 0, 'source', f'{node} — source')
            elif index == len(ids) - 1:
                place(node, ncols - 1, 'target', f'{node} — target')
            else:
                count = inter_counts.get(node, 0)
                cls = 'shared' if count >= 2 else 'unique'
                col = inter_minhop.get(node, index)
                col = min(col, ncols - 2)
                place(node, col, cls,
                      f'{node} — {cls} · {count} path'
                      f'{"s" if count != 1 else ""} · earliest hop {col}')
        hop_ratios = _ratio_list(str(row.get('ratios', '')))
        for index, (a, b) in enumerate(zip(ids, ids[1:])):
            edge_count[(a, b)] += 1
            hop_w = (hop_weights[index]
                     if hop_weights is not None and index < len(hop_weights)
                     else None)
            if hop_w is not None:
                edge_weight[(a, b)] += hop_w
            if hop_ratios is not None and index < len(hop_ratios):
                edge_ratio_mass[(a, b)] += (
                    hop_ratios[index] * (hop_w if hop_w else 1.0))
                edge_ratio_count[(a, b)] += 1

    # stable within-column rows: first-seen order over the drawn paths
    col_seen: Dict[int, int] = defaultdict(int)
    col_seen_order: Dict[Tuple[int, str], int] = {}
    for ids in paths:
        for node in ids:
            col = node_col[node]
            if (col, node) not in col_seen_order:
                col_seen_order[(col, node)] = col_seen[col]
                col_seen[col] += 1

    nodes = []
    for node, col in node_col.items():
        label = node if len(node) <= 18 else node[:17] + '…'
        nodes.append({
            'id': node, 'label': label, 'cls': node_cls[node],
            'col': col, 'row': col_seen_order[(col, node)],
            'title': node_title[node],
        })

    edges = []
    for (a, b), count in sorted(edge_count.items()):
        weight = edge_weight.get((a, b))
        mass = edge_ratio_mass.get((a, b))
        edges.append({
            'f': a, 't': b, 'c': count,
            'w': round(weight, 1) if weight else 0.0,
            'rw': (mass if mass is not None else None),
            'r': (mass / weight if mass is not None and weight else None),
        })

    return {
        'ncols': ncols,
        'nrows': max(col_seen.values()) if col_seen else 1,
        'nodes': nodes,
        'edges': edges,
    }


def build_pair_entry(
    unit: Unit, source: str, target: str, stats: dict,
    inter_counts: Counter, inter_minhop: Dict[str, int],
    drawn_rows: Sequence[dict],
) -> dict:
    lengths = stats['lengths']
    drawn_nodes = {n for row in drawn_rows for n in str(row['path']).split('->')}
    drawn_shared = sum(
        1 for n in drawn_nodes
        if n not in (source, target) and inter_counts.get(n, 0) >= 2)
    drawn_unique = sum(
        1 for n in drawn_nodes
        if n not in (source, target) and inter_counts.get(n, 0) == 1)
    return {
        'id': '',  # assigned by the assembler
        'unit': unit.unit_id,
        'unit_label': unit.label,
        'source': source,
        'target': target,
        'paths_total': stats['paths'],
        'n_lengths': len(lengths),
        'length_range': (f'{min(lengths)}–{max(lengths)}'
                         if lengths else '—'),
        'shared': stats['shared'],
        'unique': stats['unique'],
        'scov': stats.get('scov', ''),
        'tcov': stats.get('tcov', ''),
        'cov_scope': stats.get('cov_scope', 'type'),
        'drawn_shared': drawn_shared,
        'drawn_unique': drawn_unique,
        'viz': build_viz(drawn_rows, inter_counts, inter_minhop),
    }


def build_global(
    units: Sequence[Unit], frames: Dict[str, pd.DataFrame],
    global_pairs: int, global_edges: int,
    unit_source_cov: Optional[Dict[str, Dict[str, str]]] = None,
    unit_target_cov: Optional[Dict[str, Dict[str, str]]] = None,
    unit_pair_cov: Optional[Dict[str, Dict[Tuple[str, str],
                                           Tuple[str, str]]]] = None,
    unit_query_cov: Optional[Dict[str, Dict[str, Tuple[int, int]]]] = None,
) -> dict:
    """Run-wide (global) presentations, in the spirit of the cross-dataset
    report's summary sections: per-unit stats, the pair x unit path-count
    matrix (the global analog of the path presence matrix), the pairs-by-
    unit-coverage histogram, and a top-edges global route network.

    The network aggregates every path edge across all units; the drawing is
    capped to the ``global_edges`` most-traversed edges (deterministic
    tiebreak on the edge key). Node columns follow the same layered rule as
    the per-pair DAG: sources column 0, intermediates at their minimum
    intermediate hop, pure targets in the last column. A node is ``shared``
    (blue) when touched by >=2 distinct pairs, ``unique`` (gray) otherwise.
    """
    unit_rows = []
    for unit in units:
        frame = frames[unit.unit_id]
        weights = frame['min_weight'].dropna()
        lengths = frame['_length'].dropna().astype(int)
        unit_rows.append({
            'id': unit.unit_id, 'label': unit.label,
            'paths': int(len(frame)),
            'pairs': int(frame.groupby(['_source', '_target']).ngroups),
            'sources': int(frame['_source'].nunique()),
            'targets': int(frame['_target'].nunique()),
            'length_range': (f'{lengths.min()}–{lengths.max()}'
                             if len(lengths) else '—'),
            'median_weight': float(weights.median()) if len(weights) else None,
            'p90_weight': float(weights.quantile(0.9)) if len(weights) else None,
            'max_weight': float(weights.max()) if len(weights) else None,
            # Query-scope bodyId coverage (over the whole query's
            # enrollment — distinct from the per-pair / per-path numbers).
            'src_cov': (unit_query_cov or {}).get(unit.unit_id, {}).get(
                'source'),
            'tgt_cov': (unit_query_cov or {}).get(unit.unit_id, {}).get(
                'target'),
            # Per-TYPE breakdown for the Overview coverage card: the same
            # enrollment CSVs grouped by type ({type: 'n/N'} strings).
            'src_cov_by_type': dict(
                (unit_source_cov or {}).get(unit.unit_id) or {}),
            'tgt_cov_by_type': dict(
                (unit_target_cov or {}).get(unit.unit_id) or {}),
        })

    pair_cells: Dict[Tuple[str, str], Dict[str, int]] = {}
    pair_len_range: Dict[Tuple[str, str], List[int]] = {}
    pair_unit_range: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
    pair_cov_detail: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
    for unit in units:
        frame = frames[unit.unit_id]
        pair_cov = (unit_pair_cov or {}).get(unit.unit_id) or {}
        for (source, target), count in (
                frame.groupby(['_source', '_target']).size().items()):
            pair_cells.setdefault((source, target), {})[unit.unit_id] = int(count)
            # per-unit bodyId coverage detail (source n/N · target n/N),
            # PER PAIR when bodyId-level output exists (else the type's
            # query-scope value).
            exact = pair_cov.get((source, target))
            if exact is not None:
                pair_cov_detail[(source, target)][unit.unit_id] = (
                    f'{exact[0] or "—"} / {exact[1] or "—"}'
                    f' [{exact[2]}]')
        lengths_by_pair = frame.groupby(['_source', '_target'])['_length']
        for (source, target), lengths in lengths_by_pair:
            lo, hi = int(lengths.min()), int(lengths.max())
            span = pair_len_range.setdefault((source, target), [lo, hi])
            span[0], span[1] = min(span[0], lo), max(span[1], hi)
            pair_unit_range[(source, target)][unit.unit_id] = (
                f'{lo}–{hi}' if lo != hi else f'{lo}')
    matrix_rows = [
        {'source': s, 'target': t, 'cells': cells, 'total': sum(cells.values()),
         'cons': len(cells),
         'length_range': (f'{pair_len_range[(s, t)][0]}–{pair_len_range[(s, t)][1]}'
                          if (s, t) in pair_len_range else '—'),
         'ranges': pair_unit_range.get((s, t), {}),
         # bodyId n/N coverage (round 9): from the unit where the pair has
         # the most paths; cov_detail carries every unit's values
         'scov': '', 'tcov': '', 'cov_detail': pair_cov_detail.get((s, t), {})}
        for (s, t), cells in pair_cells.items()
    ]
    for row in matrix_rows:
        key = (row['source'], row['target'])
        detail = row['cov_detail']
        if detail:
            primary = max(detail, key=lambda u: pair_cells[key].get(u, 0))
            scov, _, tcov = detail[primary].partition(' / ')
            row['scov'] = scov.rsplit(' [', 1)[0]
            row['tcov'] = tcov.rsplit(' [', 1)[0]
            # Scope label: 'pair' = distinct bodyIds counted over this
            # pair's own paths (bodyId-level output present); 'type' =
            # the type's query-scope fallback (skip_bodyId runs).
            row['cov_scope'] = (
                'pair' if detail[primary].endswith('[pair]') else 'type')
    matrix_rows.sort(key=lambda r: (-r['cons'], -r['total'], r['source'], r['target']))
    cons_hist = Counter(r['cons'] for r in matrix_rows)
    pair_matrix = {
        'units': [u.unit_id for u in units],
        'total_pairs': len(matrix_rows),
        'shown': min(len(matrix_rows), global_pairs),
        'rows': matrix_rows[:global_pairs],
    }

    # ROUTE NETWORK (2026-10-07): ONE representative path per source-
    # target pair — the SHORTEST, then the STRONGEST (max min_weight)
    # within that length, then path asc for determinism. The network
    # shows each pair's backbone route; the full graph stays in the
    # linked vispath network html.
    rep_paths: List[Tuple[str, ...]] = []
    for unit in units:
        frame = frames[unit.unit_id]
        best = {}
        for _, row in frame.iterrows():
            ids = str(row['path']).split('->')
            key = (ids[0], ids[-1])
            length = len(ids) - 1
            mw = row['min_weight']
            mw = float('-inf') if pd.isna(mw) else float(mw)
            cur = best.get(key)
            # SHORTEST, then STRONGEST (max bottleneck), then path asc
            cand = (length, -mw, ids)
            if cur is None or cand < cur:
                best[key] = cand
        rep_paths.extend(v[2] for v in best.values())

    # F-EX-009: the run's REAL edge set. The drawing used to be built from
    # one representative route per source-target pair, so the panel claimed
    # "7 of 7 distinct edges" for a run whose 64 chains span 92 distinct type
    # edges over 38 nodes — the word "total" described the drawing, not the
    # run. Statistics now come from EVERY emitted chain; the representative
    # set stays available as a toggle, and each edge carries its synapse mass
    # so widths mean something (they used to mean traversal count alone).
    edge_count: Counter = Counter()          # representative traversals
    all_edge_paths: Counter = Counter()      # chains using the edge (all)
    all_edge_mass: Dict[Tuple[str, str], float] = defaultdict(float)
    min_hop_inter: Dict[str, int] = {}
    ever_source: Dict[str, bool] = {}
    ever_target: Dict[str, bool] = {}
    node_paths: Counter = Counter()
    node_pairs: Dict[str, set] = defaultdict(set)
    ncols = 2
    for ids in rep_paths:
        for a, b in zip(ids, ids[1:]):
            edge_count[(a, b)] += 1
    # every emitted chain, not one per pair. Roles and columns are read
    # from the SAME whole-run scope as the statistics: a relay that only
    # appears in a non-representative chain is still a relay, so it must
    # not be drawn in the source column.
    for unit in units:
        frame = frames[unit.unit_id]
        for _, prow in frame.iterrows():
            labels = [s.strip() for s in str(prow['path']).split('->')]
            if len(labels) < 2:
                continue
            masses = _weight_list(prow.get('weights')) or []
            hops = len(labels) - 1
            ncols = max(ncols, len(labels))
            source, target = labels[0], labels[-1]
            for hop, node in enumerate(labels):
                node_paths[node] += 1
                node_pairs[node].add((source, target))
                if hop == 0:
                    ever_source[node] = True
                elif hop == hops:
                    ever_target[node] = True
                else:
                    prior = min_hop_inter.get(node)
                    if prior is None or hop < prior:
                        min_hop_inter[node] = hop
            for i, (a, b) in enumerate(zip(labels, labels[1:])):
                all_edge_paths[(a, b)] += 1
                if i < len(masses):
                    all_edge_mass[(a, b)] += float(masses[i])

    def node_col(node: str) -> int:
        if node in min_hop_inter:
            return min(min_hop_inter[node], ncols - 2)
        if node in ever_target and node not in ever_source:
            return ncols - 1
        return 0

    _all_edges = sorted(
        set(all_edge_paths) | set(edge_count),
        key=lambda e: (-all_edge_mass.get(e, 0.0), -all_edge_paths[e], e))
    emitted_edge_total = len(_all_edges)
    emitted_node_total = len(node_paths)
    top_edges = [
        (e, all_edge_paths[e]) for e in _all_edges[:global_edges]]
    nodes_set = {n for edge in top_edges for n in edge[0]}
    by_col: Dict[int, List[str]] = defaultdict(list)
    for node in nodes_set:
        by_col[node_col(node)].append(node)
    nodes = []
    for col, members in by_col.items():
        members.sort(key=lambda n: (-node_paths[n], n))
        for row_index, node in enumerate(members):
            # Role precedence (house palette): source+target teal, source
            # red, pure target purple, otherwise shared/unique by pairs —
            # a target that also relays stays purple, not blue.
            if node in ever_source and node in ever_target:
                cls = 'both'
            elif node in ever_source:
                cls = 'source'
            elif node in ever_target:
                cls = 'target'
            else:
                cls = 'shared' if len(node_pairs[node]) >= 2 else 'unique'
            label = node if len(node) <= 18 else node[:17] + '…'
            nodes.append({
                'id': node, 'label': label, 'cls': cls, 'col': col,
                'row': row_index,
                'title': (f'{node} — {node_paths[node]} path'
                          f'{"s" if node_paths[node] != 1 else ""} · '
                          f'{len(node_pairs[node])} pair'
                          f'{"s" if len(node_pairs[node]) != 1 else ""}'),
            })
    return {
        'units': unit_rows,
        'pair_matrix': pair_matrix,
        'pair_units_hist': dict(sorted(cons_hist.items())),
        'network': {
            'ncols': ncols,
            'nrows': max((len(v) for v in by_col.values()), default=1),
            'nodes': nodes,
            'edges': [{'f': a, 't': b, 'c': count,
                       'w': round(all_edge_mass.get((a, b), 0.0), 2),
                       'rep': bool(edge_count.get((a, b), 0))}
                      for (a, b), count in top_edges],
        },
        'edge_stats': {'total': emitted_edge_total,
                       'shown': len(top_edges)},
        'node_stats': {'total': emitted_node_total,
                       'shown': len(nodes_set)},
    }


def build_payload(
    run_name: str, kind: str, meta: Dict[str, Any],
    units: Sequence[Unit], frames: Dict[str, pd.DataFrame],
    unit_inter: Dict[str, Dict[Tuple[str, str], Counter]],
    unit_inter_minhop: Dict[str, Dict[Tuple[str, str], Dict[str, int]]],
    unit_pair_stats: Dict[str, Dict[Tuple[str, str], dict]],
    unit_drawn: Dict[str, Dict[Tuple[str, str], List[dict]]],
    top_per_length: int, rank_by: str,
    vispath_links: Optional[List[Dict[str, str]]] = None,
    unit_pair_ratio_source: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Assemble the JSON payload the page renders on demand."""
    pairs: List[dict] = []

    for unit in units:
        for (source, target), stats in sorted(
                unit_pair_stats[unit.unit_id].items(),
                key=lambda item: (-item[1]['paths'], item[0])):
            counts = unit_inter[unit.unit_id][(source, target)]
            minhop = unit_inter_minhop[unit.unit_id][(source, target)]
            entry = build_pair_entry(
                unit, source, target, stats, counts, minhop,
                unit_drawn[unit.unit_id][(source, target)])
            pairs.append(entry)

    # unique ids for deep links
    used: Dict[str, int] = {}
    for entry in pairs:
        base = _query_report_slug(
            f"{entry['unit']}__{entry['source']}__{entry['target']}")
        entry['id'] = base
        if base in used:
            used[base] += 1
            entry['id'] = f'{base}_{used[base]}'
        else:
            used[base] = 1

    length_counts: Counter = Counter()
    for unit in units:
        frame = frames[unit.unit_id]
        for value, count in frame['_length'].value_counts().items():
            length_counts[int(value)] += int(count)

    return {
        'run': {
            'name': run_name,
            'kind': kind,
            # What the Analysis bar calls this run's destination. A Network
            # run browses a 1-hop direct-connection table, not a searched
            # path set, and modes is kind-blind — so before this the bar read
            # "Mode: Complete — the only mode this run produced" for a folder
            # that has no Complete mode at all (measured on a real
            # find-network run, 2026-10-16 review). Carried as DATA so the
            # claim is checkable from the payload, not only from JS text.
            'mode_labels': ({'complete': 'Direct connections (1 hop)'}
                            if kind == 'Network' else {}),
            'mode_note': ('this tool has no second mode: no Complete search, '
                          'no Fast (corridor) lane, no path budget and no tau'
                          if kind == 'Network' else ''),
            'shortest': (kind == 'Shortest Paths'
                         or str(meta.get('path_mode', '')).lower() == 'shortest'),
            'hemispheres': str(meta.get('separate_hemispheres', '')).lower()
            in ('true', '1', 'yes'),
            'meta': {k: v for k, v in meta.items() if v is not None},
            'top_per_length': top_per_length,
            'rank_by': rank_by,
            # Pair-ratio column source: 'adjusted' (connection_ratio_adj)
            # or 'plain' (connection_ratio); '' = no pair-level ratios.
            'pair_ratio_source': (unit_pair_ratio_source or {}).get(
                units[0].unit_id, '') if units else '',
            'vispath': vispath_links or [],
            'units': [
                {'id': u.unit_id, 'label': u.label, 'dataset': u.dataset,
                 'threshold': u.threshold, 'lane_dir': u.lane_dir}
                for u in units
            ],
        },
        'histogram': dict(sorted(length_counts.items())),
        'pairs': pairs,
    }


# ---------------------------------------------------------------------------
# breakdown CSV writers
# ---------------------------------------------------------------------------

# SUSPENDED (2026-10-06): the probability columns (``probabilities``,
# ``path_prob``) are withheld from the breakdown export — unverifiable
# fidelity.
PATHS_COLUMNS = [
    'dataset', 'threshold', 'unit', 'source', 'target', 'pair',
    'rank_in_pair_length', 'path', 'weights', 'ratios',
    'coverage', 'min_weight', 'min_ratio', 'pair_min_ratio',
    'length',
    'source_bodyid_coverage', 'target_bodyid_coverage', 'paths_in_pair',
]

INTERMEDIATES_COLUMNS = [
    'dataset', 'threshold', 'unit', 'source', 'target', 'intermediate',
    'n_paths_using', 'classification', 'min_hop_position',
]


def write_breakdown_csvs(
    out_dir: Path, units: Sequence[Unit],
    unit_path_rows: Dict[str, List[dict]],
    unit_inter_rows: Dict[str, List[dict]],
    unit_pair_stats: Dict[str, Dict[Tuple[str, str], dict]],
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths_path = out_dir / PATHS_CSV_NAME
    inter_path = out_dir / INTERMEDIATES_CSV_NAME

    all_path_rows: List[dict] = []
    for unit in units:
        for row in unit_path_rows[unit.unit_id]:
            record = {
                'dataset': unit.dataset,
                'threshold': unit.threshold,
                'unit': unit.unit_id,
                **row,
            }
            record['paths_in_pair'] = unit_pair_stats[unit.unit_id][
                (row['source'], row['target'])]['paths']
            all_path_rows.append(record)
    pd.DataFrame(all_path_rows, columns=PATHS_COLUMNS).to_csv(
        paths_path, index=False)

    all_inter_rows = [
        {'dataset': unit.dataset, 'threshold': unit.threshold,
         'unit': unit.unit_id, **row}
        for unit in units for row in unit_inter_rows[unit.unit_id]
    ]
    pd.DataFrame(all_inter_rows, columns=INTERMEDIATES_COLUMNS).to_csv(
        inter_path, index=False)
    return paths_path, inter_path


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _discover_vispath_links(run_dir: Path) -> List[Dict[str, str]]:
    """Relative links to the run's run-level network views (round-8 item
    1): vispath Network / Sankey / Heatmap HTMLs at the run root or under
    ``visualization/``, and cross-dataset conserved-path networks.
    Returned with display labels; missing files are simply absent."""
    links: List[Dict[str, str]] = []
    patterns = (
        ('Network_*.html', '🕸️ vispath network'),
        ('Sankey_*.html', '🌊 vispath Sankey'),
        ('Heatmap_*.html', '🔥 vispath heatmap'),
        ('conserved_paths/conserved_network_*_network.html',
         '🕸️ conserved-path network'),
    )
    seen = set()
    for sub in (run_dir, run_dir / 'visualization'):
        for pattern, label in patterns:
            for match in sorted(sub.glob(pattern)):
                rel = match.relative_to(run_dir).as_posix()
                if rel in seen:
                    continue
                seen.add(rel)
                links.append({'href': rel, 'label': label})
    return links


def _link_if_exists(path: Path, base_dir: Path, label: Optional[str] = None) -> str:
    """House ``_make_link`` convention: '-' when the target is missing."""
    if not path or not path.exists():
        return '-'
    rel = os.path.relpath(path, base_dir).replace(os.sep, '/')
    return (f'<a href="{_html.escape(rel, quote=True)}" target="_blank">'
            f'{_html.escape(label or "Open")}</a>')


REPORT_CSS = """
:root {
  --primary-color: #2563eb;
  --secondary-color: #64748b;
  --success-color: #22c55e;
  --warning-color: #f59e0b;
  --danger-color: #ef4444;
  --card-bg: #ffffff;
  --border-color: #e2e8f0;
  --bg-color: #f8fafc;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background-color: var(--bg-color);
  color: #1e293b;
  line-height: 1.6;
}
.container { max-width: 1600px; margin: 0 auto; padding: 20px; }
header {
  background: linear-gradient(135deg, var(--primary-color), #1d4ed8);
  color: white; padding: 30px; border-radius: 12px; margin-bottom: 24px;
}
header h1 { font-size: 2rem; margin-bottom: 10px; }
header p { opacity: 0.92; font-size: 0.95rem; }
.card {
  background: var(--card-bg); border-radius: 12px; padding: 24px;
  margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  border: 1px solid var(--border-color);
}
.card h3 {
  color: var(--primary-color); margin-bottom: 16px; font-size: 1.1rem;
  border-bottom: 2px solid var(--border-color); padding-bottom: 10px;
}
.card h4 { color: var(--secondary-color); margin: 12px 0 8px 0; font-size: 1rem; }
.grid { display: grid; gap: 20px; grid-template-columns: repeat(4, 1fr); }
.stat-box { background: var(--bg-color); padding: 16px; border-radius: 8px; text-align: center; }
.stat-box .value { font-size: 1.8rem; font-weight: bold; color: var(--primary-color); }
.stat-box .label { color: var(--secondary-color); font-size: 0.85rem; }
table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.85rem; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border-color); }
th { background: var(--bg-color); font-weight: 600; }
tr:hover { background: var(--bg-color); }
.sticky-table-container {
  max-height: 520px; overflow-y: auto; overflow-x: auto; position: relative;
}
.sticky-table-container thead th {
  position: sticky; top: 0; background: #f1f5f9; z-index: 10;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
/* short tables render full-height (no sliced rows at a 520px cut) */
.sticky-table-container.fit { max-height: none; }
/* wide matrices: first column pinned while the rest scrolls horizontally */
.matrix-first-col { position: sticky; left: 0; background: var(--card-bg); z-index: 5;
  box-shadow: 1px 0 0 var(--border-color); }
.sticky-table-container thead th.matrix-first-col { left: 0; background: #f1f5f9;
  z-index: 11; }
.badge { padding: 3px 8px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; }
.badge-success { background: #dcfce7; color: #166534; }
.badge-warning { background: #fef3c7; color: #92400e; }
.badge-danger { background: #fee2e2; color: #991b1b; }
.badge-info { background: #dbeafe; color: #1e40af; }
.presence-check { color: var(--success-color); }
.presence-cross { color: var(--danger-color); }
.cap-note { color: var(--secondary-color); font-size: 12px; margin-top: 8px; }
.cap-note a { color: var(--primary-color); }
.code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  background: var(--bg-color); padding: 1px 6px; border-radius: 4px; }
.page-tab-bar { display: flex; flex-wrap: wrap; gap: 4px; position: sticky;
  top: 0; background: var(--card-bg); z-index: 50; padding: 8px 4px;
  border-bottom: 2px solid var(--border-color); }
.page-tab-btn { border: 1px solid var(--border-color); background: #fff;
  padding: 6px 14px; border-radius: 16px; cursor: pointer; font-size: 0.92em; }
.page-tab-btn.active { background: #1d4ed8; color: #fff;
  border-color: #1d4ed8; font-weight: 600; }
/* mode/view rows: one labelled row per choice the page actually offers */
.mode-row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
  margin: 2px 0 8px; }
.mode-row-label { font-size: 11px; font-weight: 700; letter-spacing: .04em;
  text-transform: uppercase; color: var(--secondary-color); margin-right: 6px; }
.select-row { display: flex; gap: 18px; flex-wrap: wrap; align-items: flex-end; }
.select-row label { display: flex; flex-direction: column; font-size: 12px;
  color: var(--secondary-color); gap: 4px; font-weight: 600; }
.select-row select { padding: 8px 10px; border: 1px solid var(--border-color);
  border-radius: 6px; min-width: 240px; font-size: 14px; background: #fff;
  color: #1e293b; }
.hist { display: flex; align-items: flex-end; gap: 10px; height: 150px;
  padding: 24px 8px 26px; }
.hist .bar { width: 52px; position: relative;
  background: linear-gradient(180deg, #60a5fa, var(--primary-color));
  border-radius: 6px 6px 0 0; min-height: 2px; }
.hist .bar .v { position: absolute; top: -20px; width: 100%; text-align: center;
  font-size: 11px; color: var(--secondary-color); }
.hist .bar .l { position: absolute; bottom: -20px; width: 100%; text-align: center;
  font-size: 11px; color: var(--secondary-color); }
.legend { display: flex; gap: 16px; font-size: 12px; margin: 4px 0 10px;
  color: var(--secondary-color); flex-wrap: wrap; }
.viz-tabs { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px; }
.viz-tab-btn { border: 1px solid var(--border-color); background: #fff;
  padding: 3px 12px; border-radius: 12px; cursor: pointer; font-size: 12px; }
.viz-tab-btn:hover { background: var(--bg-color); }
.viz-tab-btn.active { background: var(--primary-color); color: #fff;
  border-color: var(--primary-color); font-weight: 600; }
.legend .chip { width: 14px; height: 14px; border-radius: 3px; display: inline-block;
  vertical-align: -2px; margin-right: 5px; }
.path-viz { width: 100%; height: auto; border: 1px solid var(--border-color);
  border-radius: 8px; background: #fff; }
.network-container { height: 520px; border: 1px solid var(--border-color);
  border-radius: 8px; background: #fff; margin-top: 8px; }
.node rect { stroke: #fff; stroke-width: 1.5; }
.node.source rect { fill: #ef4444; }
.node.target rect { fill: #8b5cf6; }
.node.both rect { fill: #14b8a6; }
.node.shared rect { fill: #2563eb; }
.node.unique rect { fill: #94a3b8; }
.node text { fill: #fff; font-size: 12px; pointer-events: none; }
.node.unique text { fill: #0f172a; }
.edge { fill: none; stroke: #64748b; opacity: 0.45; transition: opacity .1s; }
.edge.hl { stroke: var(--primary-color); opacity: 1; }
tr.group-row td { background: var(--bg-color); font-weight: 600;
  color: var(--secondary-color); font-size: 12px; }
tr[data-i] { cursor: default; }
.pair-head-badges { margin: 4px 0 10px; }
.pair-head-badges .badge { margin-right: 6px; }
@media (max-width: 768px) {
  .grid { grid-template-columns: 1fr 1fr; }
}
@media print {
  .page-tab-bar { display: none; }
  .card { break-inside: avoid; }
  .tab-content { display: block !important; }
}
"""


def _esc(value: Any) -> str:
    return _html.escape(str(value), quote=True)


def _load_refill_summary(run_dir) -> dict:
    """Type-level refill provenance when the records exist in-run."""
    if not run_dir:
        return {}
    path = Path(run_dir) / 'data_details' / 'type_level_refill' / \
        'refill_provenance.json'
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def _render_provenance_card(run: dict, run_dir=None) -> str:
    """Applied-threshold / budget provenance, marking explicitly what the
    run actually used — the pathfinding counterpart of the cross-dataset
    report's 'Applied Thresholds & Bottleneck Provenance' section.

    Cross-dataset run roots have no root-level provenance block; there the
    card renders the per-delegate provenance aggregated by
    ``_sniff_metadata`` (one applied-threshold row per delegate)."""
    prov = (run.get('meta') or {}).get('provenance')
    delegate_prov = (run.get('meta') or {}).get('delegate_provenance') or {}
    if not prov and not delegate_prov:
        return ''
    if not prov:
        return _render_delegate_provenance_table(delegate_prov, run_dir)
    rows = []

    def field(label, key, badge=False, badge_cls='badge-info'):
        value = prov.get(key)
        if value is None:
            return
        if badge:
            rows.append(
                f'<tr><td>{_esc(label)}</td>'
                f'<td><span class="badge {badge_cls}">{_esc(value)}</span></td></tr>')
        else:
            rows.append(f'<tr><td>{_esc(label)}</td><td>{_esc(value)}</td></tr>')

    meta_all = run.get('meta') or {}
    if str(meta_all.get('weight_basis') or '').startswith(
            'connection_ratio'):
        rows.append(
            '<tr><td>Weight basis</td>'
            '<td><span class="badge badge-info">connection_ratio '
            '(bodyId)</span> — thresholds, τ, w0/w1 and W* are in ratio '
            'units; synapse counts remain in every table '
            'column.</td></tr>')
        rows.append(
            '<tr><td>Two scopes</td><td>Weights, Ratios and Coverage on '
            'the path rows are <strong>path-level realized support</strong>: '
            'the distinct bodyId edges backing each hop at that exact '
            'position (ratios divide their mass by the post type\u2019s '
            'full-membership incoming mass), so every number is backed by '
            'the same edges as the coverage n. '
            '<strong>Pair-level</strong> readouts — the pair\u2019s total '
            'mass share and involved-members coverage — remain in '
            'data_details/connection_type.csv. The applied threshold acts '
            'per bodyId edge — the type-level min_ratio can fall below it '
            '(see ratio_synapse_map.csv for the implied per-neuron synapse '
            'cutoffs).</td></tr>')
    applied = prov.get('applied_threshold')
    source = prov.get('applied_threshold_source')
    if applied is not None:
        shown = f'applied threshold: {applied}'
        if source:
            shown += f'  ({source})'
        rows.append(
            f'<tr><td><strong>Threshold actually used</strong></td>'
            f'<td><span class="badge badge-warning">{_esc(shown)}</span></td></tr>')
        requested = prov.get('requested_threshold')
        if requested is not None:
            rows.append(
                f'<tr><td>Requested threshold</td><td>{_esc(requested)}</td></tr>')
    field('StrongestFirst budget', 'strongest_first_budget')
    bitten = prov.get('strongest_first_budget_bitten')
    if bitten is not None:
        cls = 'badge-danger' if bitten.lower() == 'true' else 'badge-success'
        field('StrongestFirst budget bitten', 'strongest_first_budget_bitten',
              badge=True, badge_cls=cls)
    field('StrongestFirst tau', 'strongest_first_tau')
    field('Canonical tau (reproduces output)', 'tau_canonical')
    field('Dropped bottleneck (w2)', 'strongest_dropped_bottleneck')
    field('Edge budget', 'edge_budget')
    field('Edge budget applied', 'edge_budget_applied')
    field('Edge budget landing (w1)', 'edge_budget_landing')
    field('Edge weight floor (w0)', 'edge_weight_floor')
    field('Retained bottleneck (W*)', 'strongest_retained_bottleneck')
    field('Shortest source coverage', 'shortest source coverage')
    field('Shortest target coverage', 'shortest target coverage')
    stop = _coverage_stop_summary(run_dir)
    if stop:
        rows.append(
            f'<tr><td><strong>Coverage stop</strong></td>'
            f'<td><span class="badge badge-info">{_esc(stop)}</span> — '
            'backward discovery stopped once the per-type requirements '
            'were met; deeper pairs within the depth bound were not '
            'searched (emitted pairs keep exact per-pair minimum '
            'hops).</td></tr>')
    # Phase 2 (plan-type-level-refill): when the auto hook (or the CLI
    # with --in-run) produced refill records, surface them beside the
    # provenance they answer to.
    refill = _load_refill_summary(run_dir)
    if refill:
        # F-EX-008 (gate L8): the Overview must not contradict the Refill
        # tab of the same report. It used to read the TOP-LEVEL (asked-scope)
        # counters only, so a run whose asked scope had nothing to recover
        # printed "0 bodyId edges / 0 synapses" while its FULL scope had
        # recovered 2,222 (L1w5) or 2,577 (L1r0.001) synapses — the Refill
        # tab quoted those numbers in its own header.
        _full = refill.get('full') or {}
        _asked_edges = int(refill.get('refill_edges') or 0)
        _asked_mass = int(refill.get('refill_weight_total') or 0)
        _full_edges = int(_full.get('refill_edges') or 0)
        _full_mass = int(_full.get('refill_weight_total') or 0)
        if _asked_edges or _asked_mass or _full_edges or _full_mass:
            bits = []
            if _asked_edges or _asked_mass:
                bits.append(f'asked scope: {_asked_edges:,} bodyId edges / '
                            f'{_asked_mass:,} synapses of budget-cut mass')
            if _full_edges or _full_mass:
                bits.append(f'full scope (synapse &ge; 1): {_full_edges:,} '
                            'sub-threshold bodyId edges / '
                            f'{_full_mass:,} synapses behind the emitted '
                            'pairs')
            rows.append(
                f'<tr><td><strong>Type-level refill</strong></td>'
                f'<td>{" &middot; ".join(bits)} recovered (see '
                f'data_details/type_level_refill/ and Path → Refill)'
                f'</td></tr>')
        else:
            rows.append(
                '<tr><td>Type-level refill</td><td>'
                f'{_esc(str(refill.get("status")))} '
                '&mdash; nothing to recover at either scope</td></tr>')
    complete = prov.get('paths_complete')
    if complete is not None:
        cls = 'badge-success' if complete.lower() == 'true' else 'badge-warning'
        rows.append(
            f'<tr><td>Paths complete</td>'
            f'<td><span class="badge {cls}">{_esc(complete)}</span></td></tr>')
    if not rows:
        return ''
    return (
        '<div class="card"><h3>⚙️ Applied threshold &amp; budget provenance'
        '</h3>'
        '<div class="sticky-table-container"><table><thead><tr>'
        '<th>Field</th><th>Value</th></tr></thead><tbody>'
        + ''.join(rows) + '</tbody></table></div>'
        '<p class="cap-note">Same provenance block as parameters.txt / '
        'all_attributes.json: the applied threshold is what the run '
        'actually used (it can differ from the requested one when the '
        'StrongestFirst path budget or the Edge Budget bit — the output '
        'equals a complete run at the canonical tau).</p></div>')


def _coverage_stop_summary(folder) -> str:
    """'layer 3 (source any + target 100%)' from all_attributes.json's
    shortest_discovery_diagnostics.coverage_stop — '' when absent."""
    try:
        attrs = json.loads(
            (Path(folder) / 'all_attributes.json').read_text(
                encoding='utf-8'))
        stop = (attrs.get('shortest_discovery_diagnostics')
                or {}).get('coverage_stop')
    except Exception:  # noqa: BLE001 - best-effort disclosure
        return ''
    if not stop:
        return ''

    def _fmt(value):
        if value is None:
            return 'unset'
        return 'any' if float(value) == 0.0 else f'{float(value):.0%}'

    req = stop.get('requirements') or {}
    return (f"layer {stop.get('stopped_at_layer')} "
            f"(source {_fmt(req.get('source'))} + "
            f"target {_fmt(req.get('target'))})")


def _render_delegate_provenance_table(delegate_prov: dict,
                                      run_dir=None) -> str:
    """Per-delegate applied-threshold table for cross-dataset run roots
    (round 9b): one row per delegate, aggregated by ``_sniff_metadata``.
    The coverage-stop column reads each delegate's own
    all_attributes.json diagnostics."""
    rows = []
    for unit_id, prov in sorted(delegate_prov.items()):
        cov_stop = (_coverage_stop_summary(
            Path(run_dir) / 'dataset_data' / unit_id)
            if run_dir else '')
        bitten = prov.get('strongest_first_budget_bitten', '')
        bitten_cls = ('badge-danger' if bitten.lower() == 'true'
                      else 'badge-success')
        complete = prov.get('paths_complete', '')
        complete_cls = ('badge-success' if complete.lower() == 'true'
                        else 'badge-warning')
        applied = prov.get('applied_threshold', '—')
        applied_source = prov.get('applied_threshold_source', '')
        rows.append(
            f'<tr><td>{_esc(unit_id)}</td>'
            f'<td>{_esc(prov.get("requested_threshold", "—"))}</td>'
            f'<td><span class="badge badge-warning">{_esc(applied)}</span></td>'
            f'<td{_applied_title_attr(applied_source)}>'
            f'{_esc(applied_source or "—")}</td>'
            f'<td><span class="badge {bitten_cls}">{_esc(bitten or "—")}</span></td>'
            f'<td>{_esc(prov.get("strongest_first_tau", "—"))}</td>'
            f'<td><span class="badge {complete_cls}">{_esc(complete or "—")}</span></td>'
            f'<td>{_esc(prov.get("shortest source coverage", "—"))}</td>'
            f'<td>{_esc(prov.get("shortest target coverage", "—"))}</td>'
            f'<td><span class="badge badge-info">'
            f'{_esc(cov_stop or "—")}</span></td>'
            f'</tr>')
    return (
        '<div class="card"><h3>⚙️ Applied thresholds (per delegate)</h3>'
        '<div class="sticky-table-container"><table><thead><tr>'
        '<th>Unit</th><th>Requested</th><th>Applied</th><th>Source</th>'
        '<th>SF budget bitten</th><th>Tau</th><th>Paths complete</th>'
        '<th>Cov src</th><th>Cov tgt</th><th>Coverage stop</th>'
        '</tr></thead><tbody>' + ''.join(rows) +
        '</tbody></table></div>'
        '<p class="cap-note">One row per cross-dataset delegate, from each '
        'delegate\u2019s own parameters.txt provenance block — the '
        'applied threshold is what that delegate actually used.</p></div>')


def _applied_title_attr(source: str) -> str:
    return (f' title="{_html.escape(source, quote=True)}"'
            if source else '')


def _refill_has_data(prov: dict) -> bool:
    """True when the run's refill records carry browsable data: real cut
    mass ('refilled'), the non-bitten ratio-mode complete-at-asked
    table ('complete_at_asked'), or the DUAL refill-full scope
    (synapse >= 1 — present on every all-mode run since 2026-10-08)."""
    return (str(prov.get('status') or '')
            in ('refilled', 'complete_at_asked')
            or bool(prov.get('full')))


def _render_per_type_coverage_table(unit: dict, show_label: bool) -> str:
    """Per-TYPE breakdown under the aggregate query-coverage line: the
    same enrollment CSVs grouped by type — source rows count bodyIds on
    any path over the type's enrolled members, target rows bodyIds
    reached over the type's resolved members. Worst coverage first
    within each role, so the gaps surface at the top."""
    rows = []
    for role, key in (('source', 'src_cov_by_type'),
                      ('target', 'tgt_cov_by_type')):
        parsed = []
        for type_name, n_over_n in (unit.get(key) or {}).items():
            m = _COV_ELEMENT_RE.search(str(n_over_n or ''))
            if not m:
                continue
            n, total = int(m.group(1)), int(m.group(2))
            pct = (100.0 * n / total) if total else 0.0
            parsed.append((type_name, n, total, pct))
        parsed.sort(key=lambda t: (t[3], t[0]))
        rows.extend((role,) + row for row in parsed)
    if not rows:
        return ''
    body = []
    for role, type_name, n, total, pct in rows:
        title = f'{n:,} of {total:,} bodyIds in the {role} enrollment'
        body.append(
            '<tr>'
            f'<td title="{_esc(title)}">{_esc(str(type_name))}</td>'
            f'<td>{role}</td>'
            f'<td title="{_esc(title)}">{n:,} / {total:,}</td>'
            f'<td>{pct:.1f}%</td></tr>')
    heading = 'Coverage by type'
    if show_label:
        heading += f' — {_esc(unit.get("label") or "")}'
    return (
        f'<h4 style="margin:14px 0 6px 0;">{heading}</h4>'
        '<div class="sticky-table-container"><table><thead><tr>'
        '<th>Type</th><th>Role</th><th>BodyIds</th><th>Covered</th>'
        '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>')


def _render_query_coverage_card(payload: dict) -> str:
    """Overall (query-scope) source/target bodyId coverage on the
    Overview tab — over the whole enrollment, IGNORING path pairs:
    sources = bodyIds on ANY path (isInPath) / enrolled; targets =
    bodyIds reached (Checked) / resolved. One line per unit (cross
    roots show every delegate), followed by the per-type breakdown of
    the same enrollment CSVs."""
    units = (payload.get('global') or {}).get('units') or []
    lines = []
    tables = []
    for unit in units:
        parts = [_esc(unit['label'])]
        for side, key, verb in (('source', 'src_cov', 'on paths'),
                                ('target', 'tgt_cov', 'reached')):
            cov = unit.get(key)
            if not cov:
                continue
            n, total = cov
            pct = f' ({100.0 * n / total:.1f}%)' if total else ''
            parts.append(f'{side}: <strong>{n:,}/{total:,}</strong>{pct} '
                         f'{verb}')
        if len(parts) > 1:
            lines.append('<div>· ' + ' — '.join(parts) + '</div>')
        table = _render_per_type_coverage_table(
            unit, show_label=len(units) > 1)
        if table:
            tables.append(table)
    if not lines:
        return ''
    return (
        '<div class="card"><h3>Overall bodyId coverage (query scope)</h3>'
        + ''.join(lines)
        + ''.join(tables)
        + '<p class="cap-note">Query-scope coverage over the WHOLE '
          'enrollment, independent of path pairs: source bodyIds lying on '
          'any emitted path (isInPath) over all enrolled; target bodyIds '
          'reached by any path (Checked) over all resolved. The per-type '
          'rows use the same denominators per type (enrolled / resolved '
          'members), worst coverage first. Per-pair and '
          'per-path coverage live in the Global tab and the Pair '
          'Explorer.</p></div>')


def _render_refill_card(run_dir: Path) -> str:
    """Type-level refill RESULTS, shown automatically when the run
    produced refill records (budget-cut mass recovered behind the
    emitted pairs): the provenance summary plus the top refilled type
    pairs with their refilled strength and ratio."""
    folder = Path(run_dir) / 'data_details' / 'type_level_refill'
    prov_path = folder / 'refill_provenance.json'
    if not prov_path.exists():
        return ''
    prov = {}
    try:
        prov = json.loads(prov_path.read_text(encoding='utf-8'))
    except Exception:  # noqa: BLE001
        return ''
    status = str(prov.get('status') or '')
    if status and status not in ('refilled', 'complete_at_asked'):
        return ''  # 'complete_at_asked' surfaces on the Refill view only
    pairs_csv = folder / 'refill_type_pairs.csv'
    complete_only = status == 'complete_at_asked'
    if complete_only:
        # the Overview zero-mass card stays hidden (approved plan) — the
        # Analysis -> Refill carries the complete-at-asked explorer
        return ''
    rows = []
    try:
        frame = pd.read_csv(pairs_csv)
        if status == 'refilled':
            frame = frame[frame['refill_weight'].fillna(0) > 0]
        frame = frame.sort_values('refill_weight', ascending=False).head(12)
        for _, r in frame.iterrows():
            rows.append(
                '<tr><td>' + _esc(str(r['type_pre'])) + '</td>'
                '<td>' + _esc(str(r['type_post'])) + '</td>'
                f'<td>{int(r.get("emitted_weight", 0) or 0):,}</td>'
                f'<td>+{int(r.get("refill_weight", 0) or 0):,}</td>'
                f'<td><strong>{int(r.get("refilled_total", 0) or 0):,}'
                '</strong></td></tr>')
    except Exception:  # noqa: BLE001 - table is best-effort
        rows = []
    edges = prov.get('refill_pair_count') or prov.get('refill_edges')
    summary_bits = []
    if edges is not None:
        summary_bits.append(f'{int(edges):,} bodyId edges')
    # The provenance's refill_weight_total is in the RUN's weight units
    # (ratio mass on ratio runs); the type-pairs CSV is synapse mass —
    # prefer the CSV sum so the label is always honest.
    syn_total = None
    try:
        syn_total = int(pd.read_csv(pairs_csv)['refill_weight'].fillna(0)
                        .sum())
    except Exception:  # noqa: BLE001
        syn_total = None
    if syn_total:
        summary_bits.append(f'{syn_total:,} synapses of budget-cut mass')
    # R3 unfired residual: >=asked emitted-pair mass that fired on NO
    # chain (off-route); disclosed, not silently dropped.
    unfired = None
    try:
        if 'unfired_residual' in pd.read_csv(pairs_csv, nrows=1).columns:
            unfired = int(pd.read_csv(
                pairs_csv)['unfired_residual'].fillna(0).sum())
    except Exception:  # noqa: BLE001
        unfired = None
    if unfired:
        summary_bits.append(f'{unfired:,} synapses unfired (off-route)')
    summary = ('Recovered ' + ' / '.join(summary_bits)
               if summary_bits else 'Refill records present')
    # Per-column tier labels (fix 2026-10-10, user-caught): EMITTED is
    # the applied cut, but the REFILL and REFILLED TOTAL columns live at
    # the ASKED tier (the refill-asked scope re-runs at the asked
    # threshold) — labeling them with the applied cut claimed refill
    # mass at 'synapse >= 4' when it is asked-tier (>= 3) mass.
    cutoff = prov.get('applied_threshold')
    # MODE-AWARE (2026-10-07): on ratio-basis runs the applied threshold
    # is a RATIO — labeling it "synapse >= 0.01" is wrong; the mass
    # columns' cutoff there is the forced min synapse (1), with the
    # ratio tier named separately.
    ratio_basis = str(prov.get('weight_basis') or '').startswith(
        'connection_ratio')

    def _tier_lbl(threshold):
        try:
            val = float(threshold)
        except (TypeError, ValueError):
            return 'synapse'
        if ratio_basis:
            min_syn = prov.get('min_synapse')
            try:
                min_syn = float(min_syn)
            except (TypeError, ValueError):
                min_syn = 1.0
            return (f'synapse \u2265 {min_syn:g}'
                    + f', ratio \u2265 {val:g}')
        return f'synapse \u2265 {val:g}'

    cutoff_lbl = _tier_lbl(cutoff)
    asked_lbl = _tier_lbl(prov.get('requested_threshold'))
    _th_titles = {
        'Pre type': 'The presynaptic type of the pair.',
        'Post type': 'The postsynaptic type of the pair.',
        f'Emitted ({cutoff_lbl})': 'The pair\u2019s synapse mass at the '
            'applied cutoff \u2014 what connection_type.csv counts.',
        f'Refill ({asked_lbl})': 'Recovered mass behind the applied cut, '
            're-derived at the ASKED threshold on this pair, riding '
            'in-scope routes only \u2014 a certified lower bound of the '
            'complete-at-asked refill.',
        f'Refilled total ({asked_lbl})': 'Emitted + Refill \u2014 the '
            'complete-at-asked strength of the pair.',
    }
    table_html = (
        '<div class="sticky-table-container"><table><thead><tr>'
        '<th>Pre type</th><th>Post type</th>'
        f'<th title="{_esc(_th_titles[f"Emitted ({cutoff_lbl})"])}">'
        f'Emitted ({cutoff_lbl})</th>'
        f'<th title="{_esc(_th_titles[f"Refill ({asked_lbl})"])}">'
        f'Refill ({asked_lbl})</th>'
        f'<th title="{_esc(_th_titles[f"Refilled total ({asked_lbl})"])}">'
        f'Refilled total ({asked_lbl})</th>'
        '</tr></thead><tbody>'
        + ''.join(rows) + '</tbody></table></div>') if rows else ''
    return (
        '<div class="card"><h3>Type-level refill (budget-cut mass '
        'recovered)</h3>'
        f'<p>{_esc(summary)} — recovered behind the emitted pairs (see '
        'data_details/type_level_refill/; the refilled strength is a '
        'certified lower bound of the complete-at-asked refill).</p>'
        + table_html
        + '<p class="cap-note">refilled_total = emitted + refill mass; '
          'the refill is re-derived at the ASKED threshold '
          f'({asked_lbl}) while emitted carries the applied cut '
          f'({cutoff_lbl}). Full tables: the Path → Refill view and '
          'data_details/type_level_refill/; the standalone '
          '<code>type_level_refill/</code> folder mirrors the refilled '
          'data in the connection-table format with its own '
          'heatmap/network. What "Adjusted ratios" and the refill mean: '
          'see the exported '
          '<a href="_UserGuide_please_read_me.html" target="_blank">'
          'user guide</a> (Ratios, adjusted ratios, and the refill).</p>'
        '</div>') if (summary_bits or rows) else ''


def _render_overview(payload: dict, run_dir: Path,
                     breakdown_paths: Tuple[Path, Path]) -> str:
    run = payload['run']
    pairs = payload['pairs']
    lane_info = payload.get('lane') or {}
    # A standalone Fast run reports the lane's own chains, so the Overview
    # must not read as a Complete run that emitted paths (D2/G8): the same
    # numbers, named for the mode that produced them.
    #
    # A Network run counts 1-hop rows, not paths (review 2026-10-16): the
    # Overview here is the only place a reader sees the run's headline number,
    # and calling direct connections 'Paths' — or announcing 'No paths found'
    # for a tool that has no path search — describes an artifact the tool
    # never produces. Same kind-aware wording as the Analysis bar and the
    # empty-state report.
    _network = run.get('kind') == 'Network'
    if lane_info.get('lane_only'):
        paths_label = 'Validated lane chains (all units)'
    elif _network:
        paths_label = 'Direct connections (all units)'
    else:
        paths_label = 'Paths (all units)'
    empty_note = ('No direct connections found in this run.'
                  if _network else 'No paths found in this run.')
    # A lane that attempted and refused gets its own card, because Option B
    # only mounts a corridor page when the corridor HAS data — without this
    # the refusal reason would be nowhere in the report.
    _refused = lane_info.get('refusals') or []
    # (an f-string expression cannot span lines on this interpreter, so the
    # per-unit rows are built outside the concatenation)
    _refused_items = ''.join(
        '<li><b>{}</b> &mdash; {}: {}</li>'.format(
            _esc(r.get('label') or r.get('unit') or 'Run'),
            _esc(r.get('status')), _esc(r.get('reason')))
        for r in _refused)
    refusal_note = ('' if not _refused else
                    '<div class="card"><h3>Fast mode (corridor) &mdash; '
                    'did not run</h3><p style="color:var(--secondary-color);'
                    'font-size:0.9em;">The lane was selected for this run but '
                    'produced no tables, so it offers no destination under Path. The '
                    'reason is its own, per unit:</p><ul>'
                    + _refused_items
                    + '</ul><p class="cap-note">Re-run the pass on this '
                      'folder with <code>scripts/CorridorDeepDive.py '
                      '&lt;run_dir&gt;</code> to see whether it runs '
                      'now.</p></div>')
    _lane_units = lane_info.get('units') or []
    _is_typed_lane = any(bool(u.get('type_pruned')) for u in _lane_units)
    _lane_dir_name = ((_lane_units[0].get('lane_dir')
                       if _lane_units else None) or LANE_DIR)
    mode_note = ('' if not lane_info.get('lane_only') else
                 '<div class="card"><h3>Run mode: '
                 + ('Type-pruned complete only' if _is_typed_lane
                    else 'Fast (corridor) only') + '</h3>'
                 '<p style="color:var(--secondary-color);font-size:0.9em;">'
                 'This run skipped the Complete pipeline (discovery, '
                 'enumeration, materialization, refill) by request, so its '
                 'only path table is the lane\u2019s '
                 '<code>' + _lane_dir_name + '/route_paths_type.csv</code>. '
                 'Every number below counts those validated chains. Browse '
                 'them per source &times; target under <b>Analysis</b> '
                 '(' + ('Type-pruned complete' if _is_typed_lane
                        else 'Fast (corridor)')
                 + ' &rarr; Pair Explorer), and the same '
                 'chains re-valued at synapse &ge; 1 under '
                 + ('Type-pruned complete' if _is_typed_lane
                    else 'Fast (corridor)')
                 + ' &rarr; Refill.</p></div>')
    _typed_card = ''
    if lane_info.get('lane_only') and (
            Path(run_dir) / TYPED_LANE_DIR
            / 'type_prune_manifest.json').exists():
        _typed_card, _ = _render_type_pruned_section(Path(run_dir), 'Run')
    total_paths = sum(p['paths_total'] for p in pairs)
    largest = max(pairs, key=lambda p: (p['paths_total'], p['source'], p['target'])) \
        if pairs else None
    coverage_card = _render_query_coverage_card(payload)
    refill_card = _render_refill_card(run_dir)

    stats_html = f"""
        <div class="grid">
            <div class="stat-box"><div class="value">{len(run['units'])}</div>
                <div class="label">Units</div></div>
            <div class="stat-box"><div class="value">{len(pairs)}</div>
                <div class="label">Source–target pairs</div></div>
            <div class="stat-box"><div class="value">{total_paths:,}</div>
                <div class="label">{paths_label}</div></div>
            <div class="stat-box"><div class="value">{largest['paths_total']:,}</div>
                <div class="label">Largest pair — {_esc(largest['source'])} → {_esc(largest['target'])}</div></div>
        </div>""" if pairs else f'<p>{empty_note}</p>'

    hist = payload['histogram']
    if hist:
        peak = max(hist.values())
        bars = ''.join(
            f'<div class="bar" style="height:{max(2, round(100 * count / peak))}px">'
            f'<span class="v">{count:,}</span><span class="l">{length} hops</span></div>'
            for length, count in hist.items())
        hist_html = f'<div class="card"><h3>Hops histogram</h3><div class="hist">{bars}</div></div>'
    else:
        hist_html = ''

    unit_labels = {u['id']: u['label'] for u in run['units']}
    rows = []
    for pair in sorted(pairs, key=lambda p: (-p['paths_total'], p['source'], p['target'])):
        cov_scope_title = (
            'title="distinct endpoint bodyIds over this pair'+"'"+'s paths"'
            if pair.get('cov_scope') == 'pair'
            else 'title="type query-scope (run skipped bodyId output)"')
        rows.append(
            f'<tr data-pair="{_esc(pair["id"])}">'
            f'<td>{_esc(unit_labels.get(pair["unit"], pair["unit"]))}</td>'
            f'<td><strong>{_esc(pair["source"])}</strong></td>'
            f'<td><strong>{_esc(pair["target"])}</strong></td>'
            f'<td>{pair["paths_total"]:,}</td>'
            f'<td {cov_scope_title}>{_esc(pair.get("scov") or "—")}</td>'
            f'<td {cov_scope_title}>{_esc(pair.get("tcov") or "—")}</td>'
            f'<td>{pair["n_lengths"]} ({_esc(pair["length_range"])})</td>'
            f'<td>{pair["shared"]:,} shared · {pair["unique"]:,} unique</td>'
            f'<td><a href="#" onclick="TAB.gotoPair(\'{_esc(pair["id"])}\');'
            f'return false">Open</a></td></tr>')
    pair_table = ''
    if rows:
        pair_table = (
            '<div class="card"><h3>Pairs</h3>'
            '<p style="color:var(--secondary-color);font-size:0.9em;">Sorted by '
            'path count. Src/Tgt coverage is counted per pair (distinct '
            "endpoint bodyIds over the pair's own paths; hover says when "
            'a run skipped bodyId output and the type query-scope value '
            'is shown instead). “Open” jumps to the Pair Explorer with '
            'the source and target selects set.</p>'
            '<div class="sticky-table-container"><table><thead><tr>'
            '<th>Unit</th><th>Source</th><th>Target</th><th>Paths</th>'
            '<th>Src on paths</th><th>Tgt reached</th>'
            '<th>Lengths</th><th>Intermediates</th><th></th>'
            '</tr></thead><tbody>' + ''.join(rows) +
            '</tbody></table></div></div>')

    artifact_names = [
        'parameters.txt', 'all_attributes.json', 'parameters.json',
        'run_manifest.json', 'effective_thresholds.json',
        'source_neurons.csv', 'target_neurons.csv',
        'comparison_report.html', 'comparison_report.txt',
        'user_warning_notes.txt', 'pathfinding_provenance.csv',
    ]
    links = []
    for name in artifact_names:
        link = _link_if_exists(run_dir / name, run_dir, name)
        if link != '-':
            links.append(link)
    for pattern in ('comparison_results/path_presence_matrix_*.csv',
                    'Network_*.html', 'Sankey_*.html', 'Heatmap_*.html',
                    'visualization/Network_*.html', 'visualization/Sankey_*.html',
                    'visualization/Heatmap_*.html'):
        matches = sorted(run_dir.glob(pattern))
        for match in matches[:8]:
            rel = os.path.relpath(match, run_dir).replace(os.sep, '/')
            links.append(f'<a href="{_esc(rel)}" target="_blank">{_esc(rel)}</a>')
        if len(matches) > 8:
            links.append(f'<span class="cap-note">… (+{len(matches) - 8} more '
                         f'{_esc(pattern)})</span>')
    for path in breakdown_paths:
        links.append(_link_if_exists(path, run_dir, str(path.relative_to(run_dir))))
    artifacts = (
        '<div class="card"><h3>Artifacts</h3><p style="font-size:0.9em;">'
        + ' · '.join(links) + '</p></div>') if links else ''

    return (f'<div class="card"><h3>Run</h3>'
            f'<p><strong>{_esc(run["kind"])}</strong> — {_esc(run["name"])}'
            + ('<span class="badge badge-info" style="margin-left:8px;">Path '
               'Enumeration: shortest</span>' if run['shortest'] else '')
            + ('<span class="badge badge-warning" style="margin-left:8px;">'
               '🧠 Hemisphere-aware run — type names carry _L/_R/_U suffixes'
               '</span>' if run['hemispheres'] else '')
            + '</p>'
            + ('<p class="cap-note">Shortest path mode: results contain only '
               'the per-(source, target) minimum-hop paths (all ties kept); '
               'longer alternative routes are excluded. Note: minimum hops '
               'are per bodyId pair, so a type-level pair can still span '
               'several lengths.</p>' if run['shortest'] else '')
            + f'<p style="color:var(--secondary-color);font-size:0.9em;">'
            f'{_esc(_meta_summary(run))}</p>'
            f'<p class="cap-note">Caps: top-{run["top_per_length"]} paths per '
            f'(pair, length) · rank by {_esc(run["rank_by"])}. Every capped '
            f'view links to the uncapped CSV beside this report.</p></div>'
            + mode_note + _typed_card + refusal_note +
            f'<div class="card"><h3>Summary</h3>{stats_html}</div>'
            + hist_html + coverage_card + refill_card
            + _render_provenance_card(run, run_dir) + pair_table + artifacts)


def _meta_summary(run: dict) -> str:
    """Raw (unescaped) one-line metadata summary; callers escape once."""
    meta = run['meta']
    parts = []
    for label, key in (('dataset(s)', 'datasets'), ('nicknames', 'nicknames'),
                       ('mode', 'comparison_mode'), ('path enumeration', 'path_mode'),
                       ('thresholds', 'thresholds'), ('min synapse', 'min_synapse'),
                       ('max interlayer', 'max_interlayer')):
        value = meta.get(key)
        if value:
            parts.append(f'{label}: {value}')
    return ' | '.join(parts) if parts else 'no run metadata found'


def _run_file_census(run_dir: Path, skip) -> List[Tuple[str, List[Path]]]:
    """Every artifact file the run actually wrote in its own sidecar
    folders, grouped by folder (F-EX-010's sibling item: the Data tab used
    to list a fixed handful of names, so 24-25 files that existed were
    linked nowhere). Presence-driven by construction — a folder that was
    never written contributes nothing. Cross-dataset delegates keep their
    own sidecars and are covered by the nested reports."""
    groups: List[Tuple[str, List[Path]]] = []
    for rel in DATA_TAB_CENSUS_DIRS:
        base = (run_dir / rel) if rel else run_dir
        if not base.is_dir():
            continue
        files = [p for p in sorted(base.iterdir())
                 if p.is_file() and not p.name.startswith('.')
                 and p not in skip and p.name not in DATA_TAB_CENSUS_SKIP_NAMES]
        if files:
            groups.append((rel or '.', files))
    return groups


def _render_data_tab(payload: dict, run_dir: Path) -> str:
    run = payload['run']
    per_unit: Dict[str, dict] = {
        u['id']: {'pairs': 0, 'paths': 0} for u in run['units']}
    for pair in payload['pairs']:
        agg = per_unit[pair['unit']]
        agg['pairs'] += 1
        agg['paths'] += pair['paths_total']
    length_ranges = {u['id']: u.get('length_range', '—')
                     for u in payload['global']['units']}
    rows = []
    for unit in run['units']:
        agg = per_unit[unit['id']]
        report_link = '-'
        if unit['dataset']:
            nested = (run_dir / 'dataset_data' / unit['id'] / REPORT_NAME)
            report_link = _link_if_exists(nested, run_dir, 'Open')
        rows.append(
            f'<tr><td>{_esc(unit["label"])}</td><td>{_esc(unit["dataset"] or "—")}</td>'
            f'<td>{_esc(unit["threshold"] or "—")}</td>'
            f'<td>{_esc(length_ranges.get(unit["id"], "—"))}</td>'
            f'<td>{agg["pairs"]}</td><td>{agg["paths"]:,}</td>'
            f'<td>{report_link}</td></tr>')
    table = (
        '<div class="card"><h3>Units</h3>'
        '<div class="sticky-table-container"><table><thead><tr>'
        '<th>Unit</th><th>Dataset</th><th>Threshold</th><th>Lengths</th>'
        '<th>Pairs</th><th>Paths</th><th>Pair report</th></tr></thead><tbody>'
        + ''.join(rows) + '</tbody></table></div>'
        '<p class="cap-note">Unit labels are the raw folder names relative to '
        'the run root (dataset_data/ stripped); thresholds parsed best-effort '
        'for sorting only. Cross-dataset delegates each carry their own '
        'single-unit <code>path_report.html</code> inside their folder.</p></div>')

    csv_links = []
    linked = set()
    for unit in run['units']:
        csv_path = next(iter(sorted(unit_csv_candidates(run_dir, unit))), None)
        if csv_path is None:
            continue
        linked.add(csv_path)
        rel = os.path.relpath(csv_path, run_dir).replace(os.sep, '/')
        csv_links.append(f'<a href="{_esc(rel)}" target="_blank">{_esc(unit["label"])}</a>')
    paths_card = (
        '<div class="card"><h3>Paths tables</h3><p style="font-size:0.9em;">'
        + ' · '.join(csv_links) + '</p></div>') if csv_links else ''

    breakdown_card = (
        '<div class="card"><h3>Breakdown CSVs (lossless)</h3>'
        '<p style="font-size:0.9em;">'
        '<a href="paths_pair_breakdown/pair_breakdown_paths.csv" target="_blank">'
        'pair_breakdown_paths.csv</a> — one row per path with pair, rank, '
        'weights and bodyid_coverage · '
        '<a href="paths_pair_breakdown/pair_breakdown_intermediates.csv" '
        'target="_blank">pair_breakdown_intermediates.csv</a> — shared/unique '
        'per (pair, intermediate)</p></div>')

    artifact_names = [
        'parameters.txt', 'all_attributes.json', 'parameters.json',
        'run_manifest.json', 'effective_thresholds.json',
        'source_neurons.csv', 'target_neurons.csv',
        'comparison_report.html', 'comparison_report.txt',
        'user_warning_notes.txt',
        'comparison_results/pathfinding_provenance.csv',
        'comparison_results/path_count_comparison.csv',
    ]
    artifact_links = []
    for name in artifact_names:
        path = run_dir / name
        link = _link_if_exists(path, run_dir, name)
        if link != '-':
            linked.add(path)
            artifact_links.append(link)
    artifacts_card = (
        '<div class="card"><h3>Run artifacts</h3><p style="font-size:0.9em;">'
        + ' · '.join(artifact_links) + '</p></div>') if artifact_links else ''

    vispath_links = []
    for item in run.get('vispath', []):
        vispath_links.append(
            f'<a href="{_esc(item["href"])}" target="_blank">'
            f'{_esc(item["label"])}</a>')
    vispath_card = (
        '<div class="card"><h3>vispath views (run-level)</h3>'
        '<p style="font-size:0.9em;">' + ' · '.join(vispath_links) + '</p>'
        '<p class="cap-note">The vispath visualizations cover the whole run; '
        'per-pair layered maps live in the Pair Explorer.</p></div>'
    ) if vispath_links else ''

    census_groups = _run_file_census(run_dir, linked)
    census_total = sum(len(files) for _, files in census_groups)
    census_parts, shown = [], 0
    for rel, files in census_groups:
        room = DATA_TAB_CENSUS_CAP - shown
        if room <= 0:
            break
        take = files[:room]
        shown += len(take)
        labels = []
        for path in take:
            href = os.path.relpath(path, run_dir).replace(os.sep, '/')
            name = path.name if rel == '.' else f'{rel}/{path.name}'
            labels.append(f'<a href="{_html.escape(href, quote=True)}" '
                          f'target="_blank">{_esc(name)}</a>')
        census_parts.append(
            f'<h4>{_esc("(run root)" if rel == "." else rel + "/")}</h4>'
            f'<p style="font-size:0.9em;">' + ' · '.join(labels) + '</p>')
    census_card = ''
    if census_parts:
        more = (f'<p class="cap-note">… {census_total - shown:,} more file(s) '
                f'in these folders not linked (cap {DATA_TAB_CENSUS_CAP}).</p>'
                if census_total > shown else '')
        census_card = (
            '<div class="card"><h3>Every sidecar this run wrote</h3>'
            + ''.join(census_parts) + more
            + '<p class="cap-note">Presence-driven: a folder the run never '
            'wrote contributes nothing, so no link here points at a missing '
            'file. Column meaning for each of these is in the run guide '
            '(_UserGuide_please_read_me.html) and docs/OUTPUT_FILES.md. '
            'Cross-dataset delegates keep their own sidecars beside their '
            'nested report.</p></div>')

    return (table + breakdown_card + paths_card + artifacts_card
            + census_card + vispath_card)


def _render_global_tab(payload: dict) -> str:
    """Run-wide presentations, cross-dataset-analysis style: per-unit
    stats, the pair x unit path-count matrix, the pairs-by-unit-coverage
    histogram, and the top-edges global route network (SVG rendered by JS
    from the payload)."""
    glob = payload['global']
    run = payload['run']
    unit_labels = {u['id']: u['label'] for u in run['units']}

    def _cov_cell(pair_tuple, title):
        if not pair_tuple:
            return '<td>—</td>'
        n, total = pair_tuple
        pct = (f' ({100.0 * n / total:.1f}%)' if total else '')
        return f'<td title="{title}">{n:,}/{total:,}{pct}</td>'

    unit_rows = []
    for unit in glob['units']:
        unit_rows.append(
            f'<tr><td>{_esc(unit["label"])}</td>'
            f'<td>{unit["paths"]:,}</td><td>{unit["pairs"]}</td>'
            f'<td>{unit["sources"]}</td><td>{unit["targets"]}</td>'
            f'<td>{_fmt_num(unit["median_weight"])}</td>'
            f'<td>{_fmt_num(unit["p90_weight"])}</td>'
            f'<td>{_fmt_num(unit["max_weight"])}</td>'
            + _cov_cell(unit.get('src_cov'),
                        'Query-scope: source bodyIds on any path (isInPath)'
                        ' / all enrolled bodyIds of the query')
            + _cov_cell(unit.get('tgt_cov'),
                        'Query-scope: target bodyIds reached (Checked) /'
                        ' all resolved bodyIds of the query')
            + '</tr>')
    query_cov_lines = []
    for unit in glob['units']:
        parts = [_esc(unit['label'])]
        if unit.get('src_cov'):
            n, total = unit['src_cov']
            pct = f' ({100.0 * n / total:.1f}%)' if total else ''
            parts.append(f'sources {n:,}/{total:,}{pct}')
        if unit.get('tgt_cov'):
            n, total = unit['tgt_cov']
            pct = f' ({100.0 * n / total:.1f}%)' if total else ''
            parts.append(f'targets {n:,}/{total:,}{pct}')
        if len(parts) > 1:
            query_cov_lines.append(
                '<div>· ' + ' — '.join(parts) + '</div>')
    query_cov_card = (
        '<div class="card"><h3>Query bodyId coverage</h3>'
        + (''.join(query_cov_lines) if query_cov_lines
           else '<p class="cap-note">Enrollment files unavailable.</p>')
        + '<p class="cap-note">Query-scope: source = bodyIds on ANY path '
          '(isInPath) / all enrolled; target = bodyIds reached (Checked) / '
          'all resolved — over the whole query. The Pair × unit matrix '
          'below and the Pair Explorer carry the per-pair and per-path '
          'numbers (per-pair distinct counting needs bodyId-level output; '
          "skip_bodyId runs show the type's query-scope value there, "
          'labeled in the hover).</p></div>') if query_cov_lines else ''
    units_card = (
        '<div class="card"><h3>Units</h3>'
        '<div class="sticky-table-container"><table><thead><tr>'
        '<th>Unit</th><th>Paths</th><th>Pairs</th><th>Sources</th>'
        '<th>Targets</th><th>Median min-weight</th><th>P90</th><th>Max</th>'
        '<th title="Query-scope: on any path / enrolled">Src on paths</th>'
        '<th title="Query-scope: reached / resolved">Tgt reached</th>'
        '</tr></thead><tbody>' + ''.join(unit_rows) +
        '</tbody></table></div>'
        '<p class="cap-note">Weights are per-path bottlenecks (min_weight); '
        'P90 is the 90th percentile. Min synapse thresholds make these '
        'non-comparable across units — compare shapes, not levels. Ratio-basis units (minratio_ folders) ARE cross-dataset comparable: the threshold is a fraction of each post neuron\'s total input.</p></div>')

    units_card = query_cov_card + units_card
    matrix = glob['pair_matrix']
    n_units = len(matrix['units'])
    # short matrices render full-height so no rows hide behind a scroll cut
    container_cls = ('sticky-table-container fit'
                     if len(matrix['rows']) <= 20 else 'sticky-table-container')
    matrix_rows = []
    for row in matrix['rows']:
        cells = []
        for unit_id in matrix['units']:
            count = row['cells'].get(unit_id)
            if count is None:
                cells.append('<td>—</td>')
            else:
                unit_range = (row.get('ranges') or {}).get(unit_id)
                title = (f' title="hops {unit_range}"' if unit_range else '')
                cells.append(f'<td{title}><strong>{count:,}</strong></td>')
        detail = row.get('cov_detail') or {}
        cov_title = ''
        if detail:
            scope = ('per-pair distinct bodyIds'
                     if row.get('cov_scope') == 'pair'
                     else 'type query-scope (run skipped bodyId output)')
            cov_title = ' title="' + _esc(
                '; '.join(f'{u}: {v.rsplit(" [", 1)[0]}'
                          for u, v in sorted(detail.items()))
                + f' — {scope}') + '"'
        matrix_rows.append(
            f'<tr><td class="matrix-first-col"><strong>{_esc(row["source"])}</strong> → '
            f'<strong>{_esc(row["target"])}</strong></td>'
            + ''.join(cells)
            + f'<td{cov_title}><span class="badge badge-info">{_esc(row.get("scov") or "—")}</span></td>'
            f'<td{cov_title}><span class="badge badge-info">{_esc(row.get("tcov") or "—")}</span></td>'
            f'<td>{_esc(row.get("length_range", "—"))}</td>'
            f'<td>{row["total"]:,}</td></tr>')
    matrix_card = (
        '<div class="card"><h3>Pair × unit path counts</h3>'
        '<p style="color:var(--secondary-color);font-size:0.9em;">The global '
        'analog of the cross-dataset path presence matrix: pairs (matched by '
        'name across units) as rows, units as columns, cells = path counts '
        '(hover for the hop range within that unit). Sorted by unit '
        'coverage, then total paths. Source/Target coverage are counted '
        'PER PAIR — distinct endpoint bodyIds over that pair\u2019s own '
        'paths (requires bodyId-level output; runs that skipped it show '
        'the type\u2019s query-scope value — hover says which). Per-path '
        'numbers are on the Pair Explorer rows.</p>'
        f'<div class="{container_cls}"><table><thead><tr>'
        '<th class="matrix-first-col">Pair</th>'
        + ''.join(f'<th>{_esc(unit_labels.get(u, u))}</th>' for u in matrix['units'])
        + '<th>Source coverage</th><th>Target coverage</th>'
        + '<th>Lengths</th><th>Total</th></tr></thead><tbody>'
        + ''.join(matrix_rows) + '</tbody></table></div>')
    if matrix['shown'] < matrix['total_pairs']:
        matrix_card += (
            f'<p class="cap-note">Showing {matrix["shown"]} of '
            f'{matrix["total_pairs"]:,} pairs — the full list is in the '
            f'Overview pair table and '
            f'<a href="paths_pair_breakdown/pair_breakdown_paths.csv" '
            f'target="_blank">pair_breakdown_paths.csv</a> (lossless).</p>')
    matrix_card += '</div>'

    hist = glob['pair_units_hist']
    if hist and n_units > 1:
        peak = max(hist.values())
        bars = ''.join(
            f'<div class="bar" style="height:{max(2, round(100 * count / peak))}px">'
            f'<span class="v">{count}</span><span class="l">{k}/{n_units}</span></div>'
            for k, count in hist.items())
        hist_card = (
            '<div class="card"><h3>Pairs by unit coverage</h3>'
            f'<div class="hist">{bars}</div>'
            '<p class="cap-note">Bar label k/N = the pair appears in k of the '
            'N units (pair-level conservation distribution; top number = '
            'pair count). Single-unit runs omit this chart.</p></div>')
    else:
        hist_card = ''

    stats = glob['edge_stats']
    nstat = glob['node_stats']
    # full-graph link: the vispath network html(s) carry the COMPLETE
    # graph this backbone summarizes
    full_links = []
    for item in (run.get('vispath') or []):
        if 'network' in item.get('label', '').lower():
            full_links.append(
                f'<a href="{_esc(item["href"])}" target="_blank">'
                f'{_esc(item["label"])}</a>')
    full_note = (' Full graph: ' + ' · '.join(full_links) + '.') \
        if full_links else ''
    network_card = (
        '<div class="card"><h3>Global route network</h3>'
        '<div class="legend">'
        '<span><span class="chip" style="background:#ef4444"></span>source type</span>'
        '<span><span class="chip" style="background:#8b5cf6"></span>target type</span>'
        '<span><span class="chip" style="background:#14b8a6"></span>source &amp; target</span>'
        '<span><span class="chip" style="background:#2563eb"></span>intermediate in ≥2 pairs</span>'
        '<span><span class="chip" style="background:#94a3b8"></span>intermediate in 1 pair</span>'
        '</div>'
        '<div id="global-network"></div>'
        f'<p class="cap-note">Defaults to the run\u2019s emitted type edges: '
        f'{stats["shown"]:,} of {stats["total"]:,} distinct edges drawn '
        f'({nstat["shown"]:,} of {nstat["total"]:,} nodes), edge width '
        f'&prop; synapse mass; hover a node for its path/pair counts. The '
        f'\u201cRepresentative routes\u201d toggle draws one route per '
        f'source-target pair — the SHORTEST, then the strongest (max '
        f'bottleneck) within that length — with width &prop; traversals. '
        f'Every emitted edge and its mass also ship in '
        f'data_details/connection_type.csv.{full_note}</p></div>')

    return units_card + matrix_card + hist_card + network_card


def unit_csv_candidates(run_dir: Path, unit: dict) -> List[Path]:
    unit_id = unit['id']
    if not unit_id:
        return sorted(run_dir.glob(f'*{PATHS_SUFFIX}'))
    folder = run_dir / 'dataset_data' / unit_id
    if not folder.exists():
        folder = run_dir / unit_id
    return sorted(folder.glob(f'*{PATHS_SUFFIX}'))


def _render_corridor_section(folder: Path, label: str,
                             link_prefix: str = '',
                             lane_dir: str = LANE_DIR) -> Tuple[str, bool]:
    """One unit's corridor-lane disclosure: provenance, its own limits, the
    full-scope second round and the export links.

    The lane's WALK TABLE is not here: since Phase 2 the corridor's chains
    are browsed through the Explorer (per source &times; target, capped per
    length, with the same per-cell hover explanations as the Pair Explorer),
    and its second round through the corridor Refill view. What this card
    keeps is what a table cannot carry &mdash; why a lane refused, which
    classes filled their route share or spent their realization envelope,
    and which files hold the sample-level detail the report no longer
    prints (F-EX-007). Presence-
    driven: returns (html, found)."""
    folder = Path(folder)
    # Issue 2026-10-22 §2.1: fall back to the typed namespace when the
    # requested one does not exist (a typed-only run; a run can have both).
    if not (folder / lane_dir / 'route_provenance.json').exists():
        for cand in (TYPED_LANE_DIR, LANE_DIR):
            if (folder / cand / 'route_provenance.json').exists():
                lane_dir = cand
                break
    prov_path = folder / lane_dir / 'route_provenance.json'
    if not prov_path.exists():
        return '', False
    prov = {}
    try:
        prov = json.loads(prov_path.read_text(encoding='utf-8'))
    except Exception:  # noqa: BLE001
        return '', False
    if str(prov.get('status') or '') != 'complete':
        reason = _esc(str(prov.get('reason') or 'not generated'))
        return ('<div class="card"><h3>' + _esc(label) + '</h3>'
                '<p>Corridor lane: not generated &mdash; ' + reason
                + '.</p></div>', True)
    _is_typed = isinstance(prov.get("type_prune"), dict)
    asked = prov.get('asked')
    ask_txt = ('?' if asked is None else str(asked))
    floors = [v.get('floor') for v in (prov.get('classes') or {}).values()
              if v.get('floor') is not None]
    gap = prov.get('gap_partition') or {}
    gap_txt = ''
    if gap:
        gap_txt = ('<p>Gap partition vs this run&#8217;s own type paths: '
                   + str(gap.get('main_paths', '&mdash;')) + ' main / '
                   + str(gap.get('corridor_walks', '&mdash;'))
                   + ' corridor walks / '
                   + str(gap.get('both', '&mdash;')) + ' shared / '
                   + str(gap.get('corridor_only', '&mdash;'))
                   + ' corridor-only.</p>')
    links = []
    for fname, lab in ((LANE_WALKS_NAME, 'validated walks'),
                       ('route_paths_bodyId.csv', 'bodyId paths'),
                       ('route_connection_type.csv', 'connection table'),
                       ('route_full_paths_type.csv',
                        'full-scope (synapse &ge; 1) chains'),
                       ('route_full_carrying_paths_type.csv',
                        'full-scope refill-carrying chains'),
                       ('route_full_type_pairs.csv',
                        'full-scope pair table'),
                       ('route_type_incoming_asked.csv',
                        'conditioned incoming (&ge; asked)'),
                       ('route_type_incoming.csv',
                        'incoming (synapse &ge; 1)'),
                       ('route_provenance.json', 'provenance'),
                       ('README.md', 'lane README')):
        if (folder / lane_dir / fname).exists():
            links.append('<a href="' + link_prefix + lane_dir + '/'
                         + fname + '">' + lab + '</a>')
    basis = _esc(str(prov.get('validation_basis') or 'lossless_prune'))
    floor_txt = ((str(min(floors)) + '..' + str(max(floors)))
                 if floors else '&mdash;')
    capped = ' &middot; <b>time-capped</b>' if prov.get('time_capped') else ''
    # F-EX-007: the lane's own limits, stated where the numbers are read.
    # The walk columns that SHOWED the sample are gone from the report
    # (F4/Q3), so the caveat has to name the file that still holds them.
    _cls = prov.get('classes') or {}
    _stops = Counter(str(v.get('stop') or '') for v in _cls.values()
                     if isinstance(v, dict))
    _budgeted = _stops.get('budget', 0)
    _ties = sum(int(v.get('ties_truncated') or 0) for v in _cls.values()
                if isinstance(v, dict))
    # Two currencies since 2026-10-16: the class budget is counted in
    # published TYPE-LEVEL ROUTES, the realization envelope in bodyId
    # witnesses. A class can spend the envelope without filling its share
    # (one fat corridor), so the sample caveat has to watch both.
    _enveloped = sum(
        1 for v in _cls.values() if isinstance(v, dict)
        and (v.get('realization_envelope') or 0) > 0
        and int(v.get('emitted') or 0) >= int(v['realization_envelope']))
    _quota = ', '.join(f'{k}&times;{v}' for k, v in sorted(_stops.items())
                       if k) or '&mdash;'
    _why = []
    if _budgeted:
        _why.append(f'<b>{_budgeted} class(es) filled their route share'
                    f' (stop=budget)</b>')
    if _enveloped:
        _why.append(f'<b>{_enveloped} class(es) spent their realization'
                    f' envelope</b> (every published route keeps one'
                    f' witness; the rest of a fat corridor&#8217;s tie mass'
                    f' is truncated)')
    sample_txt = ('' if not (_budgeted or _enveloped or _ties) else
                  ' &middot; ' + ', '.join(_why) + f' (stop reasons: {_quota}):'
                  ' their sampled witness mass, sampled path count and'
                  ' candidate UB/floor certificates live in'
                  ' <code>route_paths_type.csv</code> only &mdash; a SAMPLE,'
                  ' not a connection weight, and never the chain mass the'
                  ' Explorer prints'
                  + (f' &middot; {_ties} boundary tie(s) truncated'
                     if _ties else ''))
    # The lane's SECOND ROUND (Q2): the same chains re-valued at
    # synapse >= 1, so a Fast run answers the same scope pair a Complete run
    # does. Disclosed here even when it produced nothing.
    _full = prov.get('full') or {}
    full_txt = ''
    if _full:
        _fstat = str(_full.get('status') or '')
        if _fstat == 'failed':
            full_txt = ('<p><b>Full-scope second round: FAILED</b> &mdash; '
                        + _esc(str(_full.get('error') or ''))
                        + '. The lane&#8217;s asked-scope tables are '
                        'unaffected.</p>')
        elif _fstat in ('empty',):
            full_txt = ('<p><b>Full-scope second round</b> (synapse &ge; 1): '
                        'none of the lane&#8217;s chains realizes at the '
                        'floor.</p>')
        elif _fstat == 'nothing_below_tier':
            full_txt = ('<p><b>Full-scope second round</b> (synapse &ge; 1): '
                        + str(int(_full.get('chains') or 0))
                        + ' chains re-valued, none carries sub-threshold '
                        'mass &mdash; at this tier the lane already saw '
                        'every edge. Browse the complete set in the Refill '
                        'view.</p>')
        else:
            full_txt = ('<p><b>Full-scope second round</b> (synapse &ge; 1, '
                        'no tier): ' + str(int(_full.get('chains') or 0))
                        + ' lane chains re-valued, '
                        + str(int(_full.get('chains_carrying') or 0))
                        + ' carrying '
                        + f'{int(_full.get("refill_weight_total") or 0):,}'
                        + ' synapses below the lane&#8217;s asked tier '
                        + f'(&ge; {ask_txt}) &mdash; '
                        + str(int(_full.get('refill_edges') or 0))
                        + ' sub-threshold bodyId edges, in '
                        + str(_full.get('elapsed_s', '&mdash;')) + 's. '
                        + ('<b>' + str(int(_full.get('chains_revisit') or 0))
                           + '</b> chain(s) revisit an effective type label, '
                           'so their masses carry the disclosed '
                           'layered-walk superset semantics. '
                           if int(_full.get('chains_revisit') or 0) else '')
                        + '</p>')
    else:
        full_txt = ('<p><b>Full-scope second round</b>: this lane ran before '
                    'the synapse &ge; 1 re-value existed &mdash; regenerate '
                    'with <code>scripts/CorridorDeepDive.py &lt;run_dir&gt;'
                    '</code> to get the Refill view for the lane.</p>')
    _tp = prov.get('type_prune') if _is_typed else {}
    _typed_para = (
        '<p><b>Type-pruned COMPLETE lane</b>: '
        + _esc(str(_tp.get('basis') or '&mdash;')) + ' prune at '
        + _esc(str(_tp.get('min_ratio_adj') or '&mdash;')) + '; '
        + _esc(str(_tp.get('candidate_walks') or '&mdash;'))
        + ' candidate walks, ' + _esc(str(_tp.get('phantoms') or '&mdash;'))
        + ' phantoms; <b>type_pruned_complete</b>='
        + _esc(str(prov.get('type_pruned_complete'))) + ', '
        + '<b>filter_complete</b>='
        + _esc(str(_tp.get('filter_complete'))) + '. This is separate from '
        + 'the Complete lane&#8217;s <code>paths_complete</code>.</p>'
        if _is_typed else '')
    return ('<div class="card"><h3>' + _esc(label)
            + (' &mdash; Type-pruned complete lane' if _is_typed else '')
            + '</h3>' + _typed_para
            + '<p style="color:var(--secondary-color);font-size:0.9em;">'
            'Route-scoped pass at the ASKED threshold &mdash; same ratio '
            'conventions as the Refill view (plain = full-membership '
            'incoming; adj = &ge;-threshold incoming), independent scope '
            '(no refill columns). Validation basis: ' + basis
            + '. Weights and Coverage are <b>chain-level</b>: the distinct '
            'bodyId edges that can realize this hop of THIS chain at the '
            'lane\u2019s admitted tier, not a sample sum.</p>'
            '<p>Validated walks: ' + str(prov.get('validated_walks',
                                                  '&mdash;')) + ' over '
            + str(prov.get('scope_pairs', '&mdash;'))
            + ' scope type-pairs &middot; bodyId paths: '
            + str(prov.get('bodyId_paths', '&mdash;'))
            + ' &middot; class floors: ' + floor_txt
            + ' &middot; runtime: ' + str(prov.get('runtime_s', '&mdash;'))
            + 's' + capped + sample_txt + '</p>'
            + gap_txt + full_txt
            + '<p>' + ' &middot; '.join(links) + '</p></div>', True)


def _render_type_pruned_section(folder: Path, label: str,
                                link_prefix: str = '') -> Tuple[str, bool]:
    """Issue 2026-10-22: the typed lane's own card, read from
    ``route_scope_paths_type_pruned/`` (never the Fast namespace)."""
    folder = Path(folder)
    lane = folder / TYPED_LANE_DIR
    manifest_p = lane / "type_prune_manifest.json"
    prov_p = lane / "route_provenance.json"
    if not manifest_p.exists():
        return "", False
    try:
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
        prov = (json.loads(prov_p.read_text(encoding="utf-8"))
                if prov_p.exists() else {})
    except (OSError, ValueError):
        return "", False
    if str(prov.get("status") or "complete") != "complete":
        reason = _esc(str(prov.get("reason") or "not generated"))
        return ('<div class="card"><h3>' + _esc(label) + '</h3>'
                '<p>Type-pruned lane: not generated &mdash; ' + reason
                + '.</p></div>', True)

    def _v(key, default="&mdash;"):
        value = manifest.get(key)
        return default if value is None else _esc(str(value))

    pair_txt = (f'{_v("pairs_before")} &rarr; {_v("pairs_after")}')
    links = []
    for fname, lab in (
            ("route_paths_type.csv", "validated type walks"),
            ("route_paths_bodyId.csv", "bodyId paths"),
            ("route_connection_type.csv", "connection table"),
            ("route_type_pairs_pruned.csv", "pruned type graph"),
            ("route_full_paths_type.csv", "full-scope (synapse &ge; 1) chains"),
            ("route_full_type_pairs.csv", "full-scope pair table"),
            ("type_prune_manifest.json", "type-prune manifest"),
            ("route_provenance.json", "provenance"),
            ("README.md", "lane README")):
        if (lane / fname).exists():
            links.append('<a href="' + link_prefix + TYPED_LANE_DIR + "/"
                         + fname + '">' + lab + '</a>')
    complete = manifest.get("type_pruned_complete")
    filter_complete = manifest.get("filter_complete")
    html = (
        '<div class="card"><h3>' + _esc(label)
        + ' &mdash; Type-pruned complete lane</h3>'
        + '<p>Pruned the lossless discovery cone on '
        + _esc(str(manifest.get("basis") or "&mdash;")) + ' at '
        + _esc(str(manifest.get("min_ratio_adj") or "&mdash;"))
        + '; enumerated ' + _v("candidate_walks")
        + ' complete type walks (' + _v("revisit_policy")
        + ', max ' + _v("max_type_visits") + ' visits/type), validated '
        + _v("validated_walks") + ' at bodyId level, rejected '
        + _v("phantoms") + ' phantoms, applied the exact realized '
        + '<code>min_ratio_adj</code> gate to ' + _v("filtered_weak")
        + ' weak and exported ' + _v("accepted_paths") + ' paths.</p>'
        + '<p>Pairs ' + pair_txt + '; mass source '
        + _esc(str(manifest.get("mass_source") or "&mdash;"))
        + '; floor ' + _v("type_pruned_floor")
        + ' (Complete tau ' + _v("complete_tau") + '). '
        + '<b>type_pruned_complete</b>=' + _esc(str(complete))
        + ', <b>filter_complete</b>=' + _esc(str(filter_complete))
        + ' (' + _v("completeness_reason") + '). '
        + 'This is separate from the Complete lane\'s '
        + '<code>paths_complete</code>.</p>'
        + ('<p>' + ' &middot; '.join(links) + '</p>' if links else '')
        + '</div>')
    return html, True


def _render_corridor_tab(payload: dict, run_dir: Path) -> str:
    """The corridor lane cards (C2): presence-driven &mdash; rendered only when
    at least one unit carries route_scope_paths/; otherwise the honest
    how-to-enable hint. Independent of the refill datasets."""
    run = payload['run']
    units = run.get('units') or []
    sections = []
    found_any = False
    for unit in units:
        label = _esc(unit.get('label') or unit.get('id') or '')
        unit_rel = unit.get('id') or ''
        if unit.get('dataset'):
            folder = Path(run_dir) / 'dataset_data' / unit_rel
            link_prefix = 'dataset_data/' + unit_rel + '/'
        else:
            folder = Path(run_dir)
            link_prefix = ''
        section, found = _render_corridor_section(
            folder, label, link_prefix,
            lane_dir=unit.get('lane_dir') or LANE_DIR)
        if section:
            sections.append(section)
        found_any = found_any or found
        # A run with BOTH lanes gets the typed card beside the Fast one;
        # a typed-only unit's lane card already names the typed lane.
        if (folder / LANE_DIR / 'route_provenance.json').exists():
            typed_section, typed_found = _render_type_pruned_section(
                folder, label, link_prefix)
            if typed_section:
                sections.append(typed_section)
            found_any = found_any or typed_found
    if not sections and units:
        section, found = _render_corridor_section(Path(run_dir), 'Run')
        if section:
            sections.append(section)
        found_any = found_any or found
    if not found_any:
        # A Network run is NOT given a card here: this section mounts in the
        # corridor template only, and a Network run's modes never include
        # 'corridor' — so any text placed here for that tool is unreachable
        # (measured). Its absence is stated where the reader actually lands,
        # in the Analysis bar caption (run.mode_note).
        return ('<div class="card"><h3>Fast mode (corridor lane)</h3>'
                '<p style="color:var(--secondary-color);">No corridor '
                'artifacts under this run. Fast mode writes '
                '<code>route_scope_paths/</code> after deep runs '
                '(Fast Mode = both, or auto with Max Intermediate '
                'Layers &ge; 4); regenerate any run folder post-hoc with '
                '<code>scripts/CorridorDeepDive.py &lt;run_dir&gt;</code>.'
                '</p></div>')
    return ''.join(sections)


def _corridor_tier_text(lane: dict) -> str:
    """"at synapse &ge; N" for the lane's own asked tier(s) — read from the
    lane's provenance, never from the Complete run's thresholds (F-EX-012's
    one-key-two-meanings is the failure this avoids)."""
    tiers = sorted({str(u.get('asked')) for u in (lane.get('units') or [])
                    if u.get('asked') is not None}, key=float)
    if not tiers:
        return 'at the lane\u2019s asked threshold'
    return 'at synapse &ge; ' + ', '.join(tiers)


def _render_corridor_explorer_tab(payload: dict, run_dir: Path) -> str:
    """Analysis &rarr; Fast (corridor) &rarr; Pair Explorer (F3): the lane's
    validated walks browsed per source &times; target, in the same capped,
    per-cell-explained table grammar the Complete Pair Explorer uses.

    The lane's own disclosure card stands above the explorer, so the tier the
    numbers are valued at, the walk-budget stop reasons and the export links
    stay on the page that shows the numbers."""
    lane = payload.get('lane') or {}
    corridor = payload.get('corridor')
    # A ratio-basis lane conditions its adjusted denominator at the
    # min-synapse tier, which such a run forces to 1 — so its Adj. ratios
    # repeat Ratios and the explorer omits the column. The page that shows
    # the numbers has to say so (the same doctrine the Pair Explorer's
    # ratio-basis suppression follows).
    _bases = {str(u.get('weight_basis') or '') for u in (lane.get('units')
                                                         or [])}
    ratio_lane = any(b.startswith('connection_ratio') for b in _bases)
    _typed_lane = any(bool(u.get('type_pruned'))
                      for u in (lane.get('units') or []))
    parts = ['<div class="card"><h3>'
             + ('Type-pruned complete lane &mdash; validated chains'
                if _typed_lane else
                'Fast mode (corridor) &mdash; validated chains')
             + '</h3>'
             '<p style="color:var(--secondary-color);font-size:0.9em;">One '
             'validated type-level walk per row group, ' + _corridor_tier_text(
                 lane) + '. Weights and Coverage are <b>chain-level</b> (the '
             'distinct bodyId edges that can realize that hop of THAT chain '
             '&mdash; the forward+backward layered filter, the refill\u2019s '
             'recipe), and Ratios / Adj. ratios are their shares of the '
             'post type\u2019s incoming mass. The lane\u2019s sample-level '
             'columns (sampled witness mass, sampled path count, candidate '
             'UB / class floor, per-walk landing tau) stay in '
             '<code>route_paths_type.csv</code> and are not printed here; '
             'the second-round view of the same chains is the corridor '
             '<b>Refill</b> view.'
             + (' This lane is <b>ratio-basis</b>, so its conditioned '
                'denominator is the all-inputs one and Adj. ratios would '
                'duplicate Ratios &mdash; the column is omitted here and '
                'kept in route_paths_type.csv.' if ratio_lane else '')
             + '</p></div>']
    parts.append(_render_corridor_tab(payload, run_dir))
    if corridor and corridor.get('pairs'):
        parts.append(_render_explorer_select_card(
            'corridor-view-bar', 'c-', 'Select corridor pairs',
            'Ctrl/Cmd-click (shift-click for ranges) to select several '
            'sources and/or targets &mdash; the pane renders every selected '
            'source &times; target combination that the lane validated, '
            'stacked, capped per length like the Pair Explorer.',
            corridor.get('units') or [], show_scope_bar=False))
        parts.append('<div id="corridor-pane"></div>')
    return ''.join(parts)


def _render_corridor_refill_tab(payload: dict, run_dir: Path) -> str:
    """Analysis &rarr; Fast (corridor) &rarr; Refill: the lane's SECOND ROUND
    (Q2) — the same validated chains re-valued at synapse &ge; 1 through the
    refill's own layered filter, so a Fast run answers the same scope pair a
    Complete run does and can be read standalone.

    Only the full scope exists here: the lane has no budget that raises an
    applied threshold, so its sub-threshold band is <code>[1, asked)</code>
    and there is no asked-scope second round to switch to. The bar therefore
    states the scope as a caption instead of rendering one lit chip (G1)."""
    lane = payload.get('lane') or {}
    complete_full = payload.get('corridor_complete_full')
    carrying = payload.get('corridor_refill_full')
    any_pairs = ((complete_full and complete_full.get('pairs'))
                 or (carrying and carrying.get('pairs')))
    units_for_bar = ((complete_full or {}).get('units')
                     or (carrying or {}).get('units') or [])
    parts = ['<div class="card"><h3>Fast mode (corridor) &mdash; refill</h3>'
             '<p style="color:var(--secondary-color);font-size:0.9em;">The '
             'lane\u2019s chains re-valued at <b>synapse &ge; 1</b> (no '
             'tier) against the lane\u2019s own hop pairs, with the refill\u2019s '
             'forward+backward layered filter and its emitted / refill class '
             'split. Two views, same grammar as the Complete Refill: '
             '<b>Complete at full (synapse&ge;1)</b> &mdash; every lane chain '
             'with the per-hop emitted / refill / skipped split, and '
             '<b>Refill-carrying</b> &mdash; only the chains that carry mass '
             'below the lane\u2019s asked tier (' + _corridor_tier_text(lane)
             + '). At this floor the conditioned denominator IS the plain '
             'all-inputs one, so Adj. ratios would duplicate Ratios and the '
             'column is omitted (it is kept in the CSV where it differs). '
             'Denominators: <code>route_type_incoming.csv</code>.</p></div>']
    sections, found = [], False
    for unit in (payload['run'].get('units') or []):
        unit_rel = unit.get('id') or ''
        if unit.get('dataset'):
            folder = Path(run_dir) / 'dataset_data' / unit_rel
            prefix = f'dataset_data/{unit_rel}/'
        else:
            folder, prefix = Path(run_dir), ''
        section, hit = _render_corridor_section(
            folder, _esc(unit.get('label') or unit_rel or 'Run'), prefix,
            lane_dir=unit.get('lane_dir') or LANE_DIR)
        if section:
            sections.append(section)
        found = found or hit
    if not sections:
        section, hit = _render_corridor_section(Path(run_dir), 'Run')
        sections.append(section)
        found = found or hit
    parts.extend(sections)
    if any_pairs:
        parts.append(_render_explorer_select_card(
            'corridor-refill-mode-bar', 'cr-',
            'Select corridor refill pairs',
            'Ctrl/Cmd-click (shift-click for ranges) to select several '
            'sources and/or targets &mdash; the pane renders every selected '
            'source &times; target combination that carries full-scope mass, '
            'stacked, capped per length like the Pair Explorer.',
            units_for_bar))
        parts.append('<div id="corridor-refill-pane"></div>')
    elif found:
        parts.append(
            '<div class="card"><p class="cap-note">No corridor Refill table '
            'in this run folder: the lane\u2019s synapse &ge; 1 second round '
            'wrote no <code>route_full_paths_type.csv</code> (it predates '
            'that round, or no chain realizes at the floor). Regenerate with '
            '<code>scripts/CorridorDeepDive.py &lt;run_dir&gt;</code>.</p>'
            '</div>')
    return ''.join(parts)


def _render_refill_tab(payload: dict, run_dir: Path) -> str:

    """The standalone Refill page: the SECOND path-building round (the
    asked-threshold re-enumeration within the involved-types restricted
    graph) presented as a SECOND Pair Explorer over the refilled
    type-level paths — selects → per-pair panes → capped
    "Top paths per length" tables, exactly the Pair Explorer layout
    (plan round 2026-10-06, user correction). Always present. States:
    records + paths table → the explorer; records without the paths
    table (legacy run) → summary + export links + a regeneration note;
    no records → the honest "budget did not bite" hint."""
    run = payload['run']
    refill = payload.get('refill')
    refill_complete = payload.get('refill_complete')
    # DUAL REFILL (2026-10-08): the refill-full scope (synapse >= 1, no
    # tier) — full-only runs (non-bitten synapse) render here too.
    refill_full = payload.get('refill_full')
    complete_full = payload.get('complete_full')
    units = run.get('units') or []
    sections = []
    has_records = False
    for unit in units:
        label = _esc(unit.get('label') or unit.get('id') or '')
        unit_rel = unit.get('id') or ''
        if unit.get('dataset'):
            folder = Path(run_dir) / 'dataset_data' / unit_rel
            link_prefix = f'dataset_data/{unit_rel}/'
        else:
            folder = Path(run_dir)
            link_prefix = ''
        section, found = _render_refill_section(folder, label, link_prefix)
        sections.append(section)
        has_records = has_records or found
    if not sections:
        section, found = _render_refill_section(Path(run_dir), 'Run')
        sections.append(section)
        has_records = has_records or found
    parts = [
        '<div class="card"><h3>Type-level refill</h3>'
        '<p style="color:var(--secondary-color);font-size:0.9em;">The '
        'budget-cut mass recovered behind the emitted type pairs — one '
        'more round of path building at the ASKED threshold within the '
        'involved-types restricted graph, presented as a second Pair '
        'Explorer with TWO views: <b>Complete at asked</b> (every '
        'asked-threshold chain — emitted + refill + skipped per hop; the '
        'default) and <b>Refill-carrying</b> (only the chains that carry '
        'recovered mass). refilled_total = emitted + refill mass; ratios '
        'use percent form; where the run shows an adjusted variant its '
        'denominator is the dataset incoming conditioned at the '
        'Min-Synapse tier (a ratio-basis run conditions at 1, so adj equals '
        'plain there and one column shows). <b>DUAL REFILL:</b> the '
        'scope switch above the selects flips between <b>Refill-asked</b> '
        '(the run\'s own thresholding) and <b>Refill-full</b> (synapse '
        '>= 1, no tier — the full-connection view of the SAME emitted '
        'chains; produced for every run).</p></div>'
    ]
    parts.extend(sections)
    has_tables = ((refill and refill.get('pairs'))
                  or (refill_complete and refill_complete.get('pairs'))
                  or (refill_full and refill_full.get('pairs'))
                  or (complete_full and complete_full.get('pairs')))
    if has_tables:
        parts.append(_render_refill_select_card(
            refill, refill_complete))
        parts.append('<div id="refill-pane"></div>')
    elif has_records:
        parts.append(
            '<div class="card"><p class="cap-note">This run has refill '
            'records but no refilled type-level paths table '
            '(<code>type_level_refill/refill_paths_type.csv</code>) — it '
            'predates the refill path explorer. Regenerate with '
            '<code>scripts/TypeLevelRefill.py &lt;run_folder&gt; '
            '--connections ... --neuron-table ... --in-run</code> to '
            'browse the recovered routes here.</p></div>')
    return ''.join(parts)


def _render_explorer_select_card(bar_id: str, prefix: str, title: str,
                                note: str, units_for_bar: Sequence[dict],
                                show_scope_bar: bool = True) -> str:
    """A Pair Explorer's control card: the mode/scope bar the page JS fills,
    the multi-unit select when the dataset spans units, and the source /
    target multi-selects. One implementation for every explorer (Pair
    Explorer, Refill, corridor chains, corridor refill), because Q5 is
    "mirror the complete mode exactly" — the grammar must not drift.

    `prefix` namespaces the element ids so several explorers can coexist on
    one page (`r-` for the refill, `c-`/`cr-` for the lane)."""
    multi = len(units_for_bar) > 1
    unit_select = (f'<label>Unit<select id="{prefix}sel-unit"></select></label>'
                   if multi else '')
    bar = (f'<div id="{bar_id}" class="button-group" '
           f'style="margin-bottom:10px;"></div>'
           if show_scope_bar else '')
    return (
        f'<div class="card"><h3>{title}</h3>' + bar +
        '<div class="select-row">'
        + unit_select +
        f'<label>Source<select id="{prefix}sel-source" multiple size="6">'
        '</select></label>'
        f'<label>Target<select id="{prefix}sel-target" multiple size="6">'
        '</select></label>'
        '</div>'
        f'<p class="cap-note">{note}</p></div>')


def _render_refill_select_card(refill: Optional[dict],
                               refill_complete: Optional[dict]) -> str:
    """The refill Select-pairs card: same layout as the Pair Explorer's,
    with r-prefixed element ids bound to RefillExplorer. The mode bar
    (RefillModes in the page JS) switches between the complete-at-asked
    and refill-carrying datasets; the JS builds the buttons itself."""
    units_for_bar = ((refill_complete or {}).get('units')
                     or (refill or {}).get('units') or [])
    return _render_explorer_select_card(
        'refill-mode-bar', 'r-', 'Select refilled pairs',
        'Ctrl/Cmd-click (shift-click for ranges) to select several sources '
        'and/or targets &mdash; the pane renders every selected source '
        '&times; target combination that has refilled paths, stacked. The '
        'capped tables rank the refilled type paths exactly like the Pair '
        'Explorer ranks the emitted ones (top per length, bottleneck-first).',
        units_for_bar)


def _render_refill_section(folder: Path, label: str,
                           link_prefix: str = '') -> Tuple[str, bool]:
    """One unit's recovered-mass summary + export links. Returns
    (html, has_records)."""
    folder = Path(folder)
    prov_path = folder / 'data_details' / 'type_level_refill' / (
        'refill_provenance.json')
    prov = {}
    if prov_path.exists():
        try:
            prov = json.loads(prov_path.read_text(encoding='utf-8'))
        except Exception:  # noqa: BLE001
            prov = {}
    pairs_csv = folder / 'data_details' / 'type_level_refill' / (
        'refill_type_pairs.csv')

    head = (f'<div class="card"><h3>{label}</h3>')

    _full_block = prov.get('full') or {}
    if (str(prov.get('status') or '') not in ('refilled',
                                              'complete_at_asked')
            and _full_block):
        # DUAL REFILL: a full-only run (non-bitten synapse) — the full
        # scope summary instead of the no-records hint.
        _rv = int(_full_block.get('chains_revisit') or 0)
        _rv_txt = ('' if not _rv else
                   ' <b>' + f'{_rv:,}' + '</b> emitted chain(s) revisit an '
                   'effective type label — their full-scope masses carry '
                   'the disclosed layered-walk superset semantics (a body '
                   'may occupy two positions), not simple-path masses. ')
        return (head
                + '<p><b>Refill-full</b> (synapse &gt;= 1, no tier — the '
                'full-connection scope): the budget did not bite on this '
                'run, so there is no refill-asked data; the full scope is '
                'the deliverable — '
                f'{int(_full_block.get("chains_complete") or 0):,} emitted '
                'chains re-valued at synapse>=1 '
                f'({_full_block.get("refill_edges", 0):,} sub-threshold '
                f'bodyId edges / '
                f'{int(_full_block.get("refill_weight_total") or 0):,} '
                'synapses recovered). '
                + _rv_txt +
                'Browse it in the explorer below '
                '(Refill-full scope).</p></div>', False)
    if str(prov.get('status') or '') not in ('refilled',
                                             'complete_at_asked') \
            or not pairs_csv.exists():
        applied = asked = None
        attrs_path = folder / 'all_attributes.json'
        if attrs_path.exists():
            try:
                attrs = json.loads(attrs_path.read_text(encoding='utf-8'))
                applied = attrs.get('applied_threshold')
                asked = attrs.get('requested_threshold')
            except Exception:  # noqa: BLE001
                pass
        detail = ''
        if applied is not None and asked is not None:
            detail = (f' This run: applied {applied} ≤ asked {asked}.')
        return (head
                + '<p class="cap-note">No refill records — the budget did '
                  'not bite: the emitted pairs are the complete set at '
                  'this threshold; there is no cut mass to recover.'
                + detail + '</p></div>', False)

    complete_only = prov.get('status') == 'complete_at_asked'
    edges = prov.get('refill_pair_count') or prov.get('refill_edges')
    syn_total = None
    try:
        syn_total = int(pd.read_csv(pairs_csv)['refill_weight'].fillna(0)
                        .sum())
    except Exception:  # noqa: BLE001
        syn_total = None
    summary_bits = []
    if complete_only:
        summary_bits.append('complete-at-asked view — the full '
                            'asked-threshold chain set (no cut mass)')
    if edges:
        summary_bits.append(f'{int(edges):,} bodyId edges')
    if syn_total:
        summary_bits.append(f'{syn_total:,} synapses of budget-cut mass')
    has_adj = False
    try:
        has_adj = 'refilled_connection_ratio_adj' in pd.read_csv(
            pairs_csv, nrows=1).columns
    except Exception:  # noqa: BLE001
        has_adj = False
    adj_note = ('' if has_adj else
                ' (adjusted refilled ratios arrive with runs from the '
                'updated engine; this run predates the column)')

    # -- exports + standalone-folder artifacts (the per-connection pair
    # table and the bodyId detail are exported files only).
    data_rel = f'{link_prefix}data_details/type_level_refill'
    export_links = [
        (f'{data_rel}/refill_type_pairs.csv',
         'refill_type_pairs.csv — full per-connection type-pairs table'),
        (f'{data_rel}/refill_bodyId_pairs.csv',
         'refill_bodyId_pairs.csv — full bodyId detail'),
        (f'{data_rel}/refill_provenance.json', 'refill_provenance.json'),
    ]
    standalone_links = [
        (f'{link_prefix}type_level_refill/complete_at_asked_paths_type.csv',
         'complete_at_asked_paths_type.csv — complete-at-asked chains '
         'with the emitted/refill/skipped per-hop split'),
        (f'{link_prefix}type_level_refill/refill_paths_type.csv',
         'refill_paths_type.csv — refilled type paths (allpaths format)'),
        (f'{link_prefix}type_level_refill/refill_connection_type.csv',
         'refill_connection_type.csv — refill mass, connection-table '
         'format'),
        (f'{link_prefix}type_level_refill/refilled_connection_type.csv',
         'refilled_connection_type.csv — complete-at-asked '
         '(emitted + refill)'),
        (f'{link_prefix}type_level_refill/refilled_connection_bodyId.csv',
         'refilled_connection_bodyId.csv — complete-at-asked bodyId edges'),
        (f'{link_prefix}type_level_refill/refill_type_heatmap.html',
         '🔥 refill type heatmap'),
        (f'{link_prefix}type_level_refill/refill_type_network.html',
         '🕸️ refill type network'),
        (f'{link_prefix}type_level_refill/refill_bodyId_heatmap.html',
         '🔥 refill bodyId heatmap'),
        (f'{link_prefix}type_level_refill/refill_bodyId_network.html',
         '🕸️ refill bodyId network'),
    ]

    def _links_html(pairs):
        found = []
        for href, text in pairs:
            if (folder / href[len(link_prefix):]).exists():
                found.append(f'<a href="{_esc(href)}" target="_blank">'
                             f'{_esc(text)}</a>')
        return found

    export_html = _links_html(export_links)
    standalone_html = _links_html(standalone_links)
    export_note = ''
    if export_html:
        export_note = ('<p class="cap-note">Exported data: '
                       + ' · '.join(export_html) + '</p>')
    standalone_note = ''
    if standalone_html:
        standalone_note = (
            '<p class="cap-note">Standalone folder '
            f'<code>{_esc(link_prefix)}type_level_refill/</code> — the '
            'refilled data mirrored in the path/connection data format '
            'with its own visualizations: '
            + ' · '.join(standalone_html) + '</p>')

    if complete_only:
        summary_p = ('<p>Complete-at-asked view — the full '
                     'asked-threshold chain set (no cut mass: the budget '
                     'did not bite on this run).' + adj_note + '</p>')
    else:
        summary_p = ('<p>Recovered ' + ' / '.join(summary_bits)
                     + ' — recovered behind the emitted pairs; a certified '
                     'lower bound of the complete-at-asked refill.'
                     + adj_note + '</p>')

    return (head
            + summary_p
            + export_note
            + standalone_note
            + '</div>', True)


def _render_explorer_template(run: dict) -> str:
    multi = len(run['units']) > 1
    unit_select = ''
    if multi:
        unit_select = ('<label>Unit<select id="sel-unit"></select></label>')
    return (
        '<div class="card"><h3>Select pairs</h3>'
        '<div class="select-row">'
        + unit_select +
        '<label>Source<select id="sel-source" multiple size="6"></select></label>'
        '<label>Target<select id="sel-target" multiple size="6"></select></label>'
        '</div>'
        '<p class="cap-note">Ctrl/Cmd-click (shift-click for ranges) to '
        'select several sources and/or targets — the pane renders every '
        'selected source × target combination that exists in the unit, '
        'stacked. No per-pair tabs, so any number of types/groups stays '
        'navigable.</p></div>'
        '<div id="pair-pane"></div>')


def _render_page_shell(payload_json: str, overview_html: str,
                       global_html: str, data_html: str, explorer_html: str,
                       run: dict, generated_at: str,
                       refill_html: str = '',
                       corridor_html: str = '',
                       corridor_refill_html: str = '',
                       guide_link: str = '') -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
__VIS_NETWORK__
<style>__CSS__</style>
</head>
<body>
<div class="container">
<header>
  <h1>🛤️ Paths Pair Report</h1>
  <p>__KIND__ — __RUNAME__</p>
  <p>__META__</p>
  <p style="margin-top:2px;">__GUIDELINK__</p>
  <p style="opacity:0.8;">Generated: __GENERATED__ · capped views; full data in
  paths_pair_breakdown/ beside this file</p>
</header>
<div class="page-tab-bar" role="tablist">
  <button class="page-tab-btn" data-page="overview" role="tab"
    onclick="TAB.show('overview')">Overview</button>
  <button class="page-tab-btn" data-page="global" role="tab"
    onclick="TAB.show('global')">Global</button>
  <button class="page-tab-btn" data-page="analysis" role="tab"
    onclick="TAB.show('analysis')">Path</button>
  <button class="page-tab-btn" data-page="data" role="tab"
    onclick="TAB.show('data')">Data</button>
</div>
<div id="page-content"></div>
<template id="tpl-overview">__OVERVIEW__</template>
<template id="tpl-global">__GLOBAL__</template>
<template id="tpl-analysis">
  <div class="card">
    <div id="analysis-bar"></div>
  </div>
  <div id="analysis-dest"></div>
</template>
<template id="tpl-explorer">__EXPLORER__</template>
<template id="tpl-data">__DATA__</template>
<template id="tpl-refill">__REFILL__</template>
<template id="tpl-corridor">__CORRIDOR__</template>
<template id="tpl-corridor_refill">__CORRIDOR_REFILL__</template>
</div>
<script type="application/json" id="pair-data">__PAYLOAD__</script>
<script>
__JS__
</script>
</body>
</html>""".replace('__TITLE__', _esc(f'Paths Pair Report — {run["name"]}')) \
           .replace('__VIS_NETWORK__', _vis_network_script()) \
           .replace('__CSS__', REPORT_CSS) \
           .replace('__KIND__', _esc(run['kind'])) \
           .replace('__RUNAME__', _esc(run['name'])) \
           .replace('__META__', _esc(_meta_summary(run))) \
           .replace('__GUIDELINK__', guide_link) \
           .replace('__GENERATED__', _esc(generated_at)) \
           .replace('__OVERVIEW__', overview_html) \
           .replace('__GLOBAL__', global_html) \
           .replace('__EXPLORER__', explorer_html) \
           .replace('__DATA__', data_html) \
           .replace('__REFILL__', refill_html) \
           .replace('__CORRIDOR__', corridor_html) \
           .replace('__CORRIDOR_REFILL__', corridor_refill_html) \
           .replace('__PAYLOAD__', payload_json) \
           .replace('__JS__', REPORT_JS)


REPORT_JS = r"""
(function() {
  'use strict';
  var DATA = JSON.parse(document.getElementById('pair-data').textContent);
  var UNIT_LABEL = {};
  DATA.run.units.forEach(function(u) { UNIT_LABEL[u.id] = u.label; });
  var LAYOUT = { COLW: 240, NW: 170, NH: 26, GY: 10, PAD: 16 };
  var ROLE_COLORS = { source: '#ef4444', target: '#8b5cf6', both: '#14b8a6',
    shared: '#2563eb', unique: '#94a3b8' };
  var NET_FALLBACK = '⚠️ vis-network could not initialize — showing the '
    + 'static layered map instead.';
  var PENDING_NETS = [];

  // Interactive vis-network rendering, house cross-dataset-report style
  // (dot-free box nodes, hover/drag/zoom, physics off). Pair networks pass
  // explicit layered x/y positions; the GLOBAL network uses vis-network's
  // hierarchical LR layout — with short paths most nodes land in one
  // layer and manual positions would crush into a vertical stack.
  // Returns false when the CDN is unreachable so callers keep the static
  // SVG fallback visible.
  function buildVisNetwork(container, viz, hierarchical) {
    if (typeof vis === 'undefined' || !container) { return false; }
    var nodes = new vis.DataSet(viz.nodes.map(function(n) {
      var item = { id: n.id, label: n.label, title: n.title,
        shape: 'box', shapeProperties: { borderRadius: 6 },
        color: { background: ROLE_COLORS[n.cls] || '#64748b',
                 border: '#ffffff',
                 highlight: { background: ROLE_COLORS[n.cls] || '#64748b',
                              border: '#1e293b' } },
        font: { color: n.cls === 'unique' ? '#0f172a' : '#ffffff', size: 11 } };
      if (!hierarchical) {
        item.x = n.col * LAYOUT.COLW;
        item.y = -n.row * (LAYOUT.NH + LAYOUT.GY);
      }
      return item;
    }));
    var edges = new vis.DataSet(viz.edges.map(function(e) {
      return { from: e.f, to: e.t,
        title: (e.t_count != null ? e.t_count : e.c) + ' paths'
          + (e.w ? ' · Σw = ' + fmt(e.w) : '')
          + (function() { var r = (e.rw && e.w) ? e.rw / e.w : e.r;
              return (r != null && isFinite(r))
                ? ' · mean ratio = ' + fmt(r) : ''; })(),
        width: 1 + 2 * Math.log2(e.c),
        color: { color: '#64748b', highlight: '#2563eb', hover: '#2563eb' },
        arrows: { to: { enabled: true, scaleFactor: 0.4 } },
        smooth: { enabled: true, type: 'cubicBezier', roundness: 0.5 } };
    }));
    var options = { interaction: { hover: true, dragView: true, zoomView: true },
                    physics: { enabled: false } };
    if (hierarchical) {
      options.layout = { hierarchical: { enabled: true, direction: 'LR',
        sortMethod: 'directed', levelSeparation: 260, nodeSpacing: 45 } };
    }
    var network = new vis.Network(container, { nodes: nodes, edges: edges }, options);
    network.fit({ animation: false });
    return true;
  }
  function flushNetworks() {
    PENDING_NETS.forEach(function(item) {
      if (buildVisNetwork(item.wrap, item.viz, item.hier)) {
        item.svg.style.display = 'none';
      } else {
        item.wrap.innerHTML = '';
        item.wrap.appendChild(elt('p', 'cap-note', NET_FALLBACK));
      }
    });
    PENDING_NETS = [];
  }

  function elt(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }
  function fmt(v) {
    if (v === null || v === undefined) { return '—'; }
    var n = Number(v);
    if (isNaN(n)) { return String(v); }
    return String(n);
  }
  function badge(text, cls) { return elt('span', 'badge ' + cls, text); }

  // ---- control-bar rows, one implementation for every switch -------------
  // A row is a label plus its chips; the CALLER decides whether the row
  // exists at all (the >= 2 destinations rule below), and a hidden row is
  // always replaced by a caption, so the page never states less than it
  // knows. Used by the Path bar and by both mode/scope switches.
  function barRow(bar, labelText) {
    var wrap = document.createElement('div');
    wrap.className = 'mode-row';
    wrap.appendChild(elt('span', 'mode-row-label', labelText));
    bar.appendChild(wrap);
    return wrap;
  }
  function barCaption(bar, text) {
    bar.appendChild(elt('div', 'cap-note', text));
  }
  function barChip(parent, label, onPick) {
    var b = document.createElement('button');
    b.type = 'button'; b.className = 'page-tab-btn';
    b.textContent = label;
    b.addEventListener('click', onPick);
    parent.appendChild(b);
    return b;
  }

  // ---- page router (house report_tabbed pattern) ----
  // One Analysis page holds the four destinations the old bar spread over
  // three buttons (Q1 / Option B): mode x view. A single-mode run therefore
  // shows two destinations, a both-mode run four.
  var SECTION = {
    'complete|explorer': {
      tpl: 'explorer', mount: function() { Explorer.mount(); } },
    'complete|refill': {
      tpl: 'refill', mount: function() { RefillModes.init(); } },
    'corridor|explorer': {
      tpl: 'corridor', mount: function() { CorridorExplorer.mount(); } },
    'corridor|refill': {
      tpl: 'corridor_refill', mount: function() {
        CorridorRefillModes.init(); } }
  };
  var MODE_NAME = {complete: 'Complete', corridor: 'Fast (corridor)'};
  var VIEW_NAME = {explorer: 'Pair Explorer', refill: 'Refill'};
  // The old top-level buttons are still valid addresses (#tab=explorer,
  // #tab=refill, #tab=corridor): a cited link must land on the destination
  // it names, not on Overview.
  var TAB_ALIAS = {explorer: ['complete', 'explorer'],
                   refill: ['complete', 'refill'],
                   corridor: ['corridor', 'explorer'],
                   corridor_refill: ['corridor', 'refill']};

  var Analysis = {
    mode: null,
    view: null,
    modes: function() {
      var views = DATA.run.views || {};
      var list = DATA.run.modes || ['complete'];
      return list.filter(function(m) {
        return (views[m] || []).length > 0; });
    },
    views: function(mode) {
      return ((DATA.run.views || {})[mode] || []).filter(function(v) {
        return !!SECTION[mode + '|' + v]; });
    },
    render: function() {
      var self = this;
      var modes = this.modes();
      var dest = document.getElementById('analysis-dest');
      var bar = document.getElementById('analysis-bar');
      if (!modes.length || !dest) { return; }
      if (modes.indexOf(this.mode) < 0) { this.mode = modes[0]; }
      var views = this.views(this.mode);
      if (!views.length) { return; }
      if (views.indexOf(this.view) < 0) { this.view = views[0]; }
      if (bar) {
        bar.innerHTML = '';
        var RUN_C = DATA.run || {};
      var labelOf = function(m) {
        // the run's own label for this destination when it has one (a Network
        // run's 1-hop table is not a 'Complete' search), else the shared name
        return (RUN_C.mode_labels || {})[m] || MODE_NAME[m] || m;
      };
      if (modes.length > 1) {
          var modeRow = barRow(bar, 'Mode');
          modes.forEach(function(m) {
            var b = barChip(modeRow, labelOf(m), function() {
              self.mode = m; self.view = null; self.render(); });
            b.dataset.mode = m;
            b.classList.toggle('active', self.mode === m);
          });
        } else {
          barCaption(bar, 'Mode: ' + labelOf(this.mode) + ' \u2014 '
            + (RUN_C.mode_note || 'the only mode this run produced') + '.');
        }
        if (views.length > 1) {
          var viewRow = barRow(bar, 'View');
          views.forEach(function(v) {
            var b = barChip(viewRow, VIEW_NAME[v] || v, function() {
              self.view = v; self.render(); });
            b.dataset.view = v;
            b.classList.toggle('active', self.view === v);
          });
        } else {
          barCaption(bar, 'View: ' + (VIEW_NAME[this.view] || this.view)
            + ' \u2014 the only view this mode produced.');
        }
      }
      var sec = SECTION[this.mode + '|' + this.view];
      dest.innerHTML = '';
      dest.appendChild(document.getElementById('tpl-' + sec.tpl)
                       .content.cloneNode(true));
      sec.mount();
      try {
        history.replaceState(null, '', '#tab=analysis&mode=' + this.mode
                                    + '&view=' + this.view);
      } catch (e) {}
    }
  };

  var TAB = {
    show: function(page) {
      // rename 2026-10-15: the visible tab is 'Path', the page id stays
      // 'analysis' (an address every citation and alias already uses).
      if (page === 'path') { page = 'analysis'; }
      var alias = TAB_ALIAS[page];
      if (alias) { Analysis.mode = alias[0]; Analysis.view = alias[1]; }
      var name = alias ? 'analysis' : page;
      var tpl = document.getElementById('tpl-' + name);
      var container = document.getElementById('page-content');
      if (!tpl || !container) { return; }
      container.innerHTML = '';
      container.appendChild(tpl.content.cloneNode(true));
      document.querySelectorAll('.page-tab-btn').forEach(function(b) {
        b.classList.toggle('active', b.dataset.page === name); });
      if (name === 'analysis') { Analysis.render(); }
      if (name === 'global') { GlobalNet.render(); }
      try { history.replaceState(null, '', '#tab=' + name); } catch (e) {}
    },
    gotoPair: function(pairId) {
      this.show('explorer');
      try {
        history.replaceState(null, '', '#tab=explorer&pair=' + pairId);
      } catch (e) {}
      Explorer.selectPair(pairId);
    }
  };
  window.TAB = TAB;

  // ---- shared SVG DAG builder (pair panes + global network) ----
  function makeSvg(viz) {
    var NS = 'http://www.w3.org/2000/svg';
    var COLW = LAYOUT.COLW, NW = LAYOUT.NW, NH = LAYOUT.NH, GY = LAYOUT.GY,
        PAD = LAYOUT.PAD;
    var W = PAD * 2 + viz.ncols * COLW;
    var H = PAD * 2 + Math.max(1, viz.nrows) * (NH + GY);
    var svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    svg.setAttribute('class', 'path-viz');
    svg.setAttribute('preserveAspectRatio', 'xMinYMin meet');

    var pos = {};
    viz.nodes.forEach(function(n) {
      pos[n.id] = {
        x: PAD + n.col * COLW,
        y: PAD + n.row * (NH + GY),
        col: n.col
      };
    });

    var edgeEls = {};
    var maxC = 1;
    viz.edges.forEach(function(e) { if (e.c > maxC) { maxC = e.c; } });
    var maxLog = Math.log2(1 + maxC);
    viz.edges.forEach(function(e) {
      var from = pos[e.f], to = pos[e.t];
      var x1 = from.x + NW, y1 = from.y + NH / 2;
      var x2 = to.x, y2 = to.y + NH / 2;
      var d;
      if (from.col === to.col) {
        var bulge = 70;
        d = 'M ' + x1 + ' ' + y1 +
            ' C ' + (x1 + bulge) + ' ' + (y1 - 30) +
            ', ' + (x2 + bulge) + ' ' + (y2 - 30) + ', ' + x2 + ' ' + y2;
      } else {
        var dx = Math.max(40, (x2 - x1) * 0.45);
        d = 'M ' + x1 + ' ' + y1 +
            ' C ' + (x1 + dx) + ' ' + y1 + ', ' + (x2 - dx) + ' ' + y2 +
            ', ' + x2 + ' ' + y2;
      }
      var pathEl = document.createElementNS(NS, 'path');
      pathEl.setAttribute('d', d);
      pathEl.setAttribute('class', 'edge');
      // round-8 item 2: normalize within the drawn set (1..6 px) so a
      // 9,000-count edge no longer renders at 27 px next to a 1,000-count one
      pathEl.setAttribute('stroke-width',
        1 + 5 * Math.log2(1 + e.c) / maxLog);
      var title = document.createElementNS(NS, 'title');
      title.textContent = (e.t_count != null ? e.t_count : e.c) + ' paths'
        + (e.w ? ' · Σw = ' + fmt(e.w) : '')
        + (function() { var r = (e.rw && e.w) ? e.rw / e.w : e.r;
            return (r != null && isFinite(r))
              ? ' · mean ratio = ' + fmt(r) : ''; })();
      pathEl.appendChild(title);
      svg.appendChild(pathEl);
      edgeEls[e.f + '->' + e.t] = pathEl;
    });

    viz.nodes.forEach(function(n) {
      var g = document.createElementNS(NS, 'g');
      g.setAttribute('class', 'node ' + n.cls);
      var rect = document.createElementNS(NS, 'rect');
      rect.setAttribute('x', pos[n.id].x);
      rect.setAttribute('y', pos[n.id].y);
      rect.setAttribute('width', NW);
      rect.setAttribute('height', NH);
      rect.setAttribute('rx', 6);
      g.appendChild(rect);
      var text = document.createElementNS(NS, 'text');
      text.setAttribute('x', pos[n.id].x + NW / 2);
      text.setAttribute('y', pos[n.id].y + NH / 2 + 4);
      text.setAttribute('text-anchor', 'middle');
      text.textContent = n.label;
      g.appendChild(text);
      var title = document.createElementNS(NS, 'title');
      title.textContent = n.title;
      g.appendChild(title);
      svg.appendChild(g);
    });

    return { svg: svg, edgeEls: edgeEls };
  }

  // Round-8b item 2: merge the selected pairs' per-pair viz payloads into
  // ONE union layout — nodes keyed by id (source col 0, target last col,
  // intermediates at their minimum drawn column), edges summed, rows
  // first-seen over the union table's row order so row-hover stays aligned.
  function mergeViz(pairs, rowPaths) {
    var ncols = 2;
    pairs.forEach(function(p) { ncols = Math.max(ncols, p.viz.ncols); });
    var info = {};
    pairs.forEach(function(p) {
      p.viz.nodes.forEach(function(n) {
        var rec = info[n.id] = info[n.id] ||
          { minCol: ncols - 2, pairCount: 0, sharedInAny: false,
            isSource: false, isTarget: false };
        if (n.cls === 'source') { rec.isSource = true; }
        else if (n.cls === 'target') { rec.isTarget = true; }
        else {
          rec.minCol = Math.min(rec.minCol, n.col);
          if (n.cls === 'shared') { rec.sharedInAny = true; }
        }
        rec.pairCount += 1;
      });
    });
    var colOf = {}, rowOf = {}, colSeen = {};
    Object.keys(info).forEach(function(id) {
      var rec = info[id];
      if (rec.isSource) { colOf[id] = 0; }
      else if (rec.isTarget) { colOf[id] = ncols - 1; }
      else { colOf[id] = Math.min(rec.minCol, ncols - 2); }
    });
    rowPaths.forEach(function(path) {
      path.split('->').forEach(function(node) {
        var col = colOf[node];
        if (rowOf[node] === undefined) {
          rowOf[node] = colSeen[col] || 0;
          colSeen[col] = (colSeen[col] || 0) + 1;
        }
      });
    });
    var edgeAgg = {};
    pairs.forEach(function(p) {
      p.viz.edges.forEach(function(e) {
        var key = e.f + '->' + e.t;
        var rec = edgeAgg[key] = edgeAgg[key]
          || { f: e.f, t: e.t, c: 0, w: 0, rw: 0 };
        rec.c += e.c; rec.w += (e.w || 0); rec.rw += (e.rw || 0);
      });
    });
    var drawn_shared = 0, drawn_unique = 0;
    var nodes = Object.keys(info).map(function(id) {
      var rec = info[id];
      var cls;
      if (rec.isSource && rec.isTarget) { cls = 'both'; }
      else if (rec.isSource) { cls = 'source'; }
      else if (rec.isTarget) { cls = 'target'; }
      else {
        cls = (rec.sharedInAny || rec.pairCount >= 2) ? 'shared' : 'unique';
        if (cls === 'shared') { drawn_shared += 1; } else { drawn_unique += 1; }
      }
      var label = id.length <= 18 ? id : id.slice(0, 17) + '…';
      return { id: id, label: label, cls: cls, col: colOf[id],
        row: rowOf[id] !== undefined ? rowOf[id] : 0,
        title: id + ' — drawn in ' + rec.pairCount + ' of '
          + pairs.length + ' selected pairs' };
    });
    return {
      ncols: ncols,
      nrows: Math.max(1, Math.max.apply(null, [1].concat(
        Object.keys(colSeen).map(function(c) { return colSeen[c]; })))),
      nodes: nodes,
      edges: Object.keys(edgeAgg).map(function(k) { return edgeAgg[k]; }),
      drawn_shared: drawn_shared,
      drawn_unique: drawn_unique,
    };
  }

  // Round-8c item 2: build a layered viz from capped table rows alone
  // (path + hop weights), used for the per-length network tabs — mirrors
  // the Python build_viz rules at reduced scope (classification within
  // the given rows).
  function buildVizFromRows(rows) {
    var counts = {}, minHop = {};
    var paths = rows.map(function(r) { return String(r.path).split('->'); });
    var ncols = 2;
    paths.forEach(function(ids) { ncols = Math.max(ncols, ids.length); });
    paths.forEach(function(ids) {
      for (var i = 1; i + 1 < ids.length; i++) {
        var n = ids[i];
        counts[n] = (counts[n] || 0) + 1;
        if (minHop[n] === undefined || i < minHop[n]) { minHop[n] = i; }
      }
    });
    var nodeCol = {}, nodeCls = {}, nodeTitle = {};
    paths.forEach(function(ids) {
      ids.forEach(function(node, i) {
        if (nodeCol[node] !== undefined) { return; }
        if (i === 0) {
          nodeCol[node] = 0; nodeCls[node] = 'source';
          nodeTitle[node] = node + ' — source';
        } else if (i === ids.length - 1) {
          nodeCol[node] = ncols - 1; nodeCls[node] = 'target';
          nodeTitle[node] = node + ' — target';
        } else {
          nodeCol[node] = Math.min(minHop[node], ncols - 2);
          nodeCls[node] = counts[node] >= 2 ? 'shared' : 'unique';
          nodeTitle[node] = node + ' — ' + nodeCls[node] + ' · '
            + counts[node] + ' path' + (counts[node] !== 1 ? 's' : '');
        }
      });
    });
    var rowOf = {}, colSeen = {};
    paths.forEach(function(ids) {
      ids.forEach(function(node) {
        if (rowOf[node] === undefined) {
          var col = nodeCol[node];
          rowOf[node] = colSeen[col] || 0;
          colSeen[col] = (colSeen[col] || 0) + 1;
        }
      });
    });
    var edgeAgg = {};
    rows.forEach(function(r) {
      var ids = String(r.path).split('->');
      var hw = null;
      try {
        var parsed = JSON.parse(r.weights);
        if (parsed instanceof Array) { hw = parsed; }
      } catch (e) { /* weights stay unmerged */ }
      for (var k = 0; k + 1 < ids.length; k++) {
        var key = ids[k] + '->' + ids[k + 1];
        var rec = edgeAgg[key] = edgeAgg[key] ||
          { f: ids[k], t: ids[k + 1], c: 0, w: 0 };
        rec.c += 1;
        if (hw && k < hw.length) { rec.w += Number(hw[k]) || 0; }
      }
    });
    var drawn_shared = 0, drawn_unique = 0;
    var nodes = Object.keys(nodeCol).map(function(id) {
      var cls = nodeCls[id];
      if (cls === 'shared') { drawn_shared += 1; }
      if (cls === 'unique') { drawn_unique += 1; }
      var label = id.length <= 18 ? id : id.slice(0, 17) + '…';
      return { id: id, label: label, cls: cls, col: nodeCol[id],
        row: rowOf[id] || 0, title: nodeTitle[id] };
    });
    return {
      ncols: ncols,
      nrows: Math.max(1, Math.max.apply(null, [1].concat(
        Object.keys(colSeen).map(function(c) { return colSeen[c]; })))),
      nodes: nodes,
      edges: Object.keys(edgeAgg).map(function(k) { return edgeAgg[k]; }),
      drawn_shared: drawn_shared,
      drawn_unique: drawn_unique,
    };
  }

  // ---- global network (Global tab) ----
  // F-EX-009: the default view draws the run's EMITTED edge set (each edge
  // weighted by its synapse mass, which is what widths now mean), with the
  // one-route-per-pair skeleton kept as a toggle. The old default drew four
  // representative routes and then announced "7 of 7 distinct edges" for a
  // run that emitted 92.
  var GlobalNet = {
    mode: 'all',
    render: function() {
      var holder = document.getElementById('global-network');
      if (!holder || holder.dataset.done || !DATA.global) { return; }
      holder.dataset.done = '1';
      var self = this;
      var bar = elt('div', 'button-group');
      [['all', 'All emitted edges'],
       ['rep', 'Representative routes']].forEach(function(item) {
        var b = document.createElement('button');
        b.type = 'button'; b.className = 'page-tab-btn';
        b.textContent = item[1];
        if (self.mode === item[0]) { b.classList.add('active'); }
        b.addEventListener('click', function() {
          self.mode = item[0];
          bar.querySelectorAll('button').forEach(function(x) {
            x.classList.remove('active'); });
          b.classList.add('active');
          self.draw(holder);
        });
        bar.appendChild(b);
      });
      holder.appendChild(bar);
      this.view = elt('div');
      holder.appendChild(this.view);
      this.draw(holder);
    },
    draw: function(holder) {
      var viz = DATA.global.network;
      var src = (viz.edges || []);
      var edges = this.mode === 'rep'
        ? src.filter(function(e) { return e.rep; }) : src;
      // STYLE PARITY with the Analysis networks (user, 2026-10-15): those
      // draw width ∝ log2(COUNT) over the SAME layered geometry, and the
      // global view used to disagree on both counts — vis-network laid it
      // out hierarchically (so the live view differed from its own SVG
      // fallback) and the all-edges view fed the synapse mass into the
      // width, so a few fat ribbons dominated the picture. The mass stays
      // the hover's subject; the thickness now follows the same law as
      // every other network on the page — the raw count, which on this
      // graph is already a small number (a rescale here would have mapped
      // a uniform graph's every edge to the maximum width instead).
      edges = edges.map(function(e) {
        return { f: e.f, t: e.t, c: e.c, t_count: e.c, w: e.w, r: e.r };
      });
      var v2 = { nodes: viz.nodes, edges: edges,
                 ncols: viz.ncols, nrows: viz.nrows };
      this.view.innerHTML = '';
      var svgWrap = elt('div');
      svgWrap.appendChild(makeSvg(v2).svg);
      this.view.appendChild(svgWrap);
      var netWrap = elt('div', 'network-container');
      this.view.appendChild(netWrap);
      // hier: false — the SAME explicit column positions the SVG fallback
      // and the Analysis panes use, so the interactive view can no longer
      // disagree with the static one.
      PENDING_NETS.push({ wrap: netWrap, svg: svgWrap, viz: v2,
                          hier: false });
      setTimeout(flushNetworks, 0);
      var note = elt('div');
      note.style.opacity = '0.8';
      note.innerHTML = (this.mode === 'rep'
        ? 'One representative route per source-target pair; edge width '
          + '&prop; traversal count (log-scaled), hover lists the mass.'
        : 'Every emitted type edge in the drawn cap; edge width '
          + '&prop; chain count (log-scaled), hover lists the synapse mass. '
          + 'Same geometry and scaling as the Pair Explorer networks.');
      this.view.appendChild(note);
    }
  };

  // ---- pair explorer ----
  // Explorer factory: the SAME explorer machinery powers the Pair
  // The Pair Explorer view (main paths) and the Refill view (the second
  // path-building round's refilled type paths) — only the data array,
  // element ids and run context differ per instance.
  function makeExplorer(cfg) {
    var P = cfg.data;
    var RUN = cfg.run;
    var UNIT_LABEL = {};
    // rebuilt on every mount: mode switches (RefillModes.set) swap
    // cfg.units after construction
    function refreshUnitLabels() {
      UNIT_LABEL = {};
      (cfg.units || []).forEach(function(u) {
        UNIT_LABEL[u.id] = u.label; });
    }
    refreshUnitLabels();
    return {
    // TAB.show re-clones the page template on EVERY visit, so mount() must
    // fully re-bind the fresh selects and re-render each time — an
    // early-return guard here left the pane empty whenever the user
    // switched away and back (fixed 2026-10-02).
    byUnit: null,
    sel: null,
    mount: function() {
      this.byUnit = {};
      refreshUnitLabels();
      var self = this;
      cfg.data.forEach(function(p) {
        (self.byUnit[p.unit] = self.byUnit[p.unit] || []).push(p); });
      this.unitSel = document.getElementById(cfg.unitSelId);
      this.sourceSel = document.getElementById(cfg.sourceSelId);
      this.targetSel = document.getElementById(cfg.targetSelId);
      if (!this.sourceSel || !this.targetSel) { return; }
      var units = cfg.units.map(function(u) { return u.id; });
      var self = this;
      if (this.unitSel) {
        units.forEach(function(uid) {
          var opt = document.createElement('option');
          opt.value = uid; opt.textContent = UNIT_LABEL[uid];
          self.unitSel.appendChild(opt);
        });
        this.unitSel.addEventListener('change', function() {
          self.fillSources(); self.rememberSel(); });
      }
      this.sourceSel.addEventListener('change', function() {
        self.fillTargets(); self.rememberSel(); });
      this.targetSel.addEventListener('change', function() {
        self.render(); self.rememberSel(); });
      // A re-mount keeps what the user selected: switching the refill VIEW
      // (or scope, or mode) changes which data is on screen, never the
      // source/target picks (F6 — mount() used to jump to the biggest pair
      // every time, collapsing a multi-selection to one pair).
      if (this.sel && this.restoreSel()) { return; }
      var biggest = cfg.data.reduce(function(a, b) {
        return (b.paths_total > a.paths_total) ? b : a; }, cfg.data[0]);
      if (biggest) {
        this.setPair(biggest.id);
      } else {
        this.fillSources();
      }
      this.rememberSel();
    },
    rememberSel: function() {
      if (!this.sourceSel) { return; }
      this.sel = {
        unit: this.unitSel ? this.unitSel.value : null,
        sources: this.selectedValues(this.sourceSel),
        targets: this.selectedValues(this.targetSel) };
    },
    restoreSel: function() {
      var s = this.sel, self = this, hit = 0;
      if (this.unitSel && s.unit) {
        if (!Array.prototype.some.call(this.unitSel.options, function(o) {
              if (o.value !== s.unit) { return false; }
              o.selected = true; return true; })) { return false; }
      }
      this.fillSources();
      Array.prototype.forEach.call(this.sourceSel.options, function(o) {
        var want = s.sources.indexOf(o.value) >= 0;
        o.selected = want; if (want) { hit += 1; }
      });
      if (!hit) { return false; }
      this.fillTargets();
      var thit = 0;
      Array.prototype.forEach.call(this.targetSel.options, function(o) {
        var want = s.targets.indexOf(o.value) >= 0;
        o.selected = want; if (want) { thit += 1; }
      });
      if (!thit && this.targetSel.options.length) {
        this.targetSel.options[0].selected = true;
      }
      this.render();
      return true;
    },
    currentUnit: function() {
      return this.unitSel ? this.unitSel.value : (cfg.units[0] || {}).id;
    },
    fillSources: function() {
      var unit = this.currentUnit();
      var agg = {};
      (this.byUnit[unit] || []).forEach(function(p) {
        agg[p.source] = (agg[p.source] || 0) + p.paths_total; });
      var names = Object.keys(agg).sort(function(a, b) {
        return agg[b] - agg[a] || (a < b ? -1 : 1); });
      var sel = this.sourceSel;
      var previous = this.selectedValues(sel);
      sel.innerHTML = '';
      names.forEach(function(name) {
        var opt = document.createElement('option');
        opt.value = name; opt.textContent = name + ' (' + agg[name] + ' paths)';
        if (previous.indexOf(name) !== -1) { opt.selected = true; }
        sel.appendChild(opt);
      });
      if (sel.selectedOptions.length === 0 && sel.options.length > 0) {
        sel.options[0].selected = true;  // never leave the explorer empty
      }
      this.fillTargets();
    },
    fillTargets: function() {
      var unit = this.currentUnit();
      var sources = this.selectedValues(this.sourceSel);
      var wanted = {};
      sources.forEach(function(s) { wanted[s] = true; });
      var agg = {};
      (this.byUnit[unit] || []).forEach(function(p) {
        if (wanted[p.source]) {
          agg[p.target] = (agg[p.target] || 0) + p.paths_total;
        }
      });
      var names = Object.keys(agg).sort(function(a, b) {
        return agg[b] - agg[a] || (a < b ? -1 : 1); });
      var sel = this.targetSel;
      var previous = this.selectedValues(sel);
      sel.innerHTML = '';
      names.forEach(function(name) {
        var opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name + ' (' + agg[name] + ' paths)';
        if (previous.indexOf(name) !== -1) { opt.selected = true; }
        sel.appendChild(opt);
      });
      if (sel.selectedOptions.length === 0 && sel.options.length > 0) {
        sel.options[0].selected = true;  // never leave the explorer empty
      }
      this.render();
    },
    setPair: function(pairId) {
      var pair = cfg.data.filter(function(p) { return p.id === pairId; })[0];
      if (!pair) { return; }
      if (this.unitSel) { this.unitSel.value = pair.unit; }
      this.fillSources();
      this.sourceSel.value = pair.source;
      this.fillTargets();
      this.targetSel.value = pair.target;
      this.render();
      this.rememberSel();
    },
    selectPair: function(pairId) { this.setPair(pairId); },
    selectedValues: function(sel) {
      return Array.prototype.slice.call(sel.selectedOptions).map(function(o) {
        return o.value; });
    },
    currentPairs: function() {
      var unit = this.currentUnit();
      var sources = this.selectedValues(this.sourceSel);
      var targets = this.selectedValues(this.targetSel);
      var wanted = {};
      sources.forEach(function(s) { wanted[s] = true; });
      var out = [];
      (this.byUnit[unit] || []).forEach(function(p) {
        if (wanted[p.source] && targets.indexOf(p.target) !== -1) {
          out.push(p);
        }
      });
      out.sort(function(a, b) { return b.paths_total - a.paths_total; });
      return out;
    },

    render: function() {
      var pane = document.getElementById(cfg.paneId);
      if (!pane) { return; }
      pane.innerHTML = '';
      var pairs = this.currentPairs();
      if (!pairs || pairs.length === 0) { return; }
      // Round-8b item 2: a multi-selection renders ONE pane whose table
      // and network are the UNION of the selected pairs — not standalone
      // sections.
      if (pairs.length === 1) {
        pane.appendChild(this.buildPane(pairs[0]));
      } else {
        pane.appendChild(this.buildUnionPane(pairs));
      }
      setTimeout(flushNetworks, 0);
    },

    buildPane: function(p) {
      var pane = elt('div');
      // header
      var head = elt('div', 'card');
      var title = elt('h3', null, p.source + ' → ' + p.target);
      head.appendChild(title);
      var badges = elt('div', 'pair-head-badges');
      badges.appendChild(badge(p.paths_total + ' paths', 'badge-info'));
      badges.appendChild(badge(p.n_lengths + ' lengths (' + p.length_range + ')', 'badge-info'));
      badges.appendChild(badge(p.shared + ' shared · ' + p.unique + ' unique intermediates', 'badge-success'));
      badges.appendChild(badge('unit: ' + UNIT_LABEL[p.unit], 'badge-warning'));
      head.appendChild(badges);
      if (RUN.shortest) {
        head.appendChild(elt('p', 'cap-note',
          'Shortest mode: only the minimum-hop paths per bodyId pair (ties '
          + 'kept); a type-level pair can still span several lengths, so '
          + 'the per-length top-' + RUN.top_per_length
          + ' cap applies normally.'));
      }
      pane.appendChild(head);

      // capped table + network (single pair: table/viz from its payload)
      pane.appendChild(this.renderCappedTable([p]));
      var rows = [];
      p.table.groups.forEach(function(g) {
        g.rows.forEach(function(r) { rows.push(r); });
      });
      var rowPaths = rows.map(function(r) { return r.path; });
      pane.appendChild(this.renderNetworkCard(rows, p.viz, rowPaths, {
        netId: p.id,
        allCaption: 'Drawn: the capped top-paths set (' + p.drawn_shared
          + ' shared / ' + p.drawn_unique + ' unique intermediates drawn). '
          + 'Classification from the full pair path set: ' + p.shared
          + ' shared / ' + p.unique + ' unique. Edge width scales within '
          + 'the drawn set.'}));

      // links
      var foot = elt('p', 'cap-note');
      foot.appendChild(document.createTextNode('Full list: '));
      var a = elt('a', null, cfg.csvLabel);
      a.href = cfg.csvHref;
      a.target = '_blank';
      foot.appendChild(a);
      if (cfg.filterPairs) {
        foot.appendChild(document.createTextNode(' — filter '));
        foot.appendChild(elt('span', 'code',
          'pair = ' + p.source + '->' + p.target));
        foot.appendChild(document.createTextNode(' and '));
        foot.appendChild(elt('span', 'code',
          'unit = ' + (UNIT_LABEL[p.unit] || p.unit)));
      }
      (RUN.vispath || []).forEach(function(item) {
        foot.appendChild(document.createTextNode(' · '));
        var v = elt('a', null, item.label);
        v.href = item.href; v.target = '_blank';
        foot.appendChild(v);
      });
      pane.appendChild(foot);
      return pane;
    },

    // Round-8b item 2: ONE pane for a multi-selection — the table and the
    // network are the UNION of the selected pairs.
    buildUnionPane: function(pairs) {
      var pane = elt('div');
      var head = elt('div', 'card');
      head.appendChild(elt('h3', null,
        pairs.length + ' selected pairs — union'));
      var badges = elt('div', 'pair-head-badges');
      var totalPaths = 0;
      pairs.forEach(function(p) { totalPaths += p.paths_total; });
      badges.appendChild(badge(totalPaths + ' paths', 'badge-info'));
      badges.appendChild(badge(pairs.length + ' pairs', 'badge-info'));
      var totShared = 0, totUnique = 0;
      pairs.forEach(function(p) { totShared += p.shared; totUnique += p.unique; });
      badges.appendChild(badge(totShared + ' shared · ' + totUnique
        + ' unique intermediates (full sets)', 'badge-success'));
      head.appendChild(badges);
      var names = pairs.map(function(p) { return p.source + '→' + p.target; });
      head.appendChild(elt('p', 'cap-note', names.slice(0, 8).join(', ')
        + (names.length > 8 ? ' … (+' + (names.length - 8) + ' more)' : '')));
      pane.appendChild(head);

      // per-length tab rows: {len, path, weights} flattened across the
      // selected pairs' capped tables
      var rows = [];
      pairs.forEach(function(p) {
        p.table.groups.forEach(function(g) {
          g.rows.forEach(function(r) {
            rows.push({ len: g.len, path: r.path, weights: r.weights });
          });
        });
      });
      var merged = mergeViz(pairs, rows.map(function(r) { return r.path; }));
      pane.appendChild(this.renderCappedTable(pairs));
      pane.appendChild(this.renderNetworkCard(rows, merged,
        rows.map(function(r) { return r.path; }), {
        netId: 'union',
        allCaption: 'Drawn: the UNION of the selected pairs\u2019 capped paths ('
          + merged.drawn_shared + ' shared / ' + merged.drawn_unique
          + ' unique intermediates drawn; shared = on \u22652 drawn paths or '
          + 'shared within any selected pair). Edge counts and widths are '
          + 'merged across pairs.'}));

      var foot = elt('p', 'cap-note');
      foot.appendChild(document.createTextNode('Full list: '));
      var a = elt('a', null, cfg.csvLabel);
      a.href = cfg.csvHref;
      a.target = '_blank';
      foot.appendChild(a);
      if (cfg.filterPairs) {
        foot.appendChild(document.createTextNode(' — filter '));
        var filterText = 'pair in (' + names.slice(0, 6).join(', ')
          + (names.length > 6 ? ', …' : '') + ')';
        foot.appendChild(elt('span', 'code',
          filterText.replace(/→/g, '->')));
        foot.appendChild(document.createTextNode(' and '));
        foot.appendChild(elt('span', 'code',
          'unit = ' + (UNIT_LABEL[pairs[0].unit] || pairs[0].unit)));
      }
      // The mirror links belong to BOTH footer builders: the union pane had
      // its own footer without them, so selecting more than one pair dropped
      // every vispath link (found by the CDP gate, 2026-10-15).
      (RUN.vispath || []).forEach(function(item) {
        foot.appendChild(document.createTextNode(' \u00b7 '));
        var v = elt('a', null, item.label);
        v.href = item.href; v.target = '_blank';
        foot.appendChild(v);
      });
      pane.appendChild(foot);
      return pane;
    },

    // Round 8b: one builder for 1..N pairs. With N>1 the table is the
    // UNION — rows from every selected pair (Pair column added), grouped
    // by length ascending, re-ranked by min_weight desc within each
    // length; caps stay per (pair, length).
    renderCappedTable: function(pairs) {
      var card = elt('div', 'card');
      card.appendChild(elt('h3', null, pairs.length > 1
        ? 'Top paths per length (union)' : 'Top paths per length'));
      var groupsByLen = {};
      // The header's "of N" counts CAPACITY per (pair, length), so it is
      // accumulated once per GROUP — summing the per-row copy over-counted
      // by the number of shown rows (fixed 2026-10-15: a 5-of-12 group read
      // "of 60").
      var capacityByLen = {};
      pairs.forEach(function(p) {
        p.table.groups.forEach(function(g) {
          var bucket = groupsByLen[g.len] = groupsByLen[g.len] || [];
          capacityByLen[g.len] = (capacityByLen[g.len] || 0) + g.total;
          g.rows.forEach(function(r) {
            // EVERY field a cell reads must be carried here: this mapper
            // rebuilds the rows for the pane, and the 2026-10-15 defect was
            // a dropped `uni`/`emitted`/`pr_rank` (empty Universe column,
            // "—" in Emitted, display rank under an R1 header) plus `total`
            // colliding with the group's capacity. `total_list` is the
            // row's own per-hop list; the group capacity is counted once per
            // group in `capacityByLen`, never stored on a row.
            bucket.push({pair: p.source + '→' + p.target, rank: r.rank,
              pr_rank: r.pr_rank,
              path: r.path, len: g.len, mw: r.mw, mr: r.mr, pr: r.pr,
              weights: r.weights, ratios: r.ratios, pradj: r.pradj,
              emitted: r.emitted, total_list: r.total, uni: r.uni,
              coverage: r.coverage});
          });
        });
      });
      var lengths = Object.keys(groupsByLen).map(Number).sort(function(a, b) {
        return a - b; });
      // COLUMN SPEC (plan §2.1, 2026-10-05): headers, cells, sort keys
      // and hovers ALL derive from one ordered spec — a header/cell
      // drift is structurally impossible (the round-8c union-column fix,
      // generalized after the 2026-10-05 mismatch the user screenshotted:
      // the conditional ratio columns had landed in different positions
      // in the two hand-kept lists).
      var union = pairs.length > 1;
      var ratioRun = RUN.rank_by === 'min_ratio';
      var hasAdj = pairs.some(function(p) {
        return (p.table.groups || []).some(function(g) {
          return (g.rows || []).some(function(r) { return r.pradj; });
        });
      });
      // Percent convention (§2.5): every ratio/probability renders
      // xx.xx%; a value that rounds to 0.00% keeps the raw form in the
      // title so no precision silently vanishes.
      function fmtPct(v) {
        if (v == null || v !== v) { return '—'; }
        return (100 * v).toFixed(2) + '%';
      }

      function tdc(v, title) {
        var td = elt('td', null, v);
        if (title) { td.title = title; }
        return td;
      }
      // Presence-matrix dash format (cross-dataset paths-tab style):
      // -v1-v2- with the BOTTLENECK (minimum) bolded. Ratios below
      // 0.01% render as (<0.01%).
      function dashList(listStr, fmt) {
        var vals = String(listStr || '').replace(/^\[/, '')
          .replace(/\]$/, '').split(',').map(function(x) {
            return x.trim(); })
          .filter(function(x) { return x !== ''; });
        if (!vals.length) { return '—'; }
        var nums = vals.map(function(v) {
          var n = parseFloat(v);
          return isNaN(n) ? null
            : (v.indexOf('%') !== -1 ? n / 100 : n);
        });
        var present = nums.filter(function(n) { return n !== null; });
        var mn = present.length ? Math.min.apply(null, present) : null;
        var parts = vals.map(function(v, i) {
          var shown = fmt ? fmt(v, nums[i]) : v;
          if (nums[i] !== null && nums[i] === mn) {
            return '<strong>' + shown + '</strong>';
          }
          return shown;
        });
        return '-' + parts.join('-') + '-';
      }
      function dashTd(listStr, fmt, title) {
        var td = elt('td');
        td.innerHTML = dashList(listStr, fmt);
        if (title) { td.title = title; }
        return td;
      }
      function ratioDashFmt(v, n) {
        if (n === null) { return '—'; }
        return n < 0.0001 ? '(<0.01%)' : fmtPct(n);
      }
      function adjHeader(minSyn) {
        return minSyn
          ? 'Adj. ratios (inputs \u2265 ' + minSyn + ' synapses)'
          : 'Adj. ratios (threshold-conditioned inputs)';
      }
      var hasSplit = !!(RUN.has_split && (pairs || []).some(function(p) {
        return (p.table.groups || []).some(function(g) {
          return (g.rows || []).some(function(r) { return r.total; });
        });
      }));
      // The Weights header names the tier the masses are admitted at. A
      // run whose metadata records no threshold (a sniff that found no
      // Min Synapse Count, a hand-made folder) must not print
      // "synapse ≥ undefined" — a number the reader cannot check is worse
      // than no number, and the same law made adjHeader fall back to words.
      var wTier = RUN.ratio_basis ? (RUN.min_synapse || 1)
        : (RUN.applied || RUN.min_synapse);
      var COLS = [
        { h: '#', cell: function(row, i) {
            // Plan rank where the table carries it (the corridor's chains:
            // acceptance rank by candidate upper bound, R1 2026-10-15);
            // the display rank within this (pair, length) group otherwise.
            // The hover states WHICH one the number is, per cell, so the
            // same header never means two things silently.
            var td = elt('td', null, row.pr_rank != null ? row.pr_rank : i + 1);
            td.title = row.pr_rank != null
              ? 'Acceptance rank by candidate upper bound (F-PF-002: an UB '
                + 'rank, not a measured route strength) — this walk\u2019s '
                + 'plan_rank in the lane\u2019s route_paths_type.csv.'
              : 'Display rank within this (pair, length) group, by the '
                + 'ranking key this report was generated with.';
            return td; } },
        union && { h: 'Pair', cell: function(row) {
            return elt('td', null, row.pair); } },
        { h: 'Path', cell: function(row) {
            var td = elt('td');
            td.appendChild(elt('strong', null, row.path));
            return td; } },
        { h: 'Len', cell: function(row) { return elt('td', null, row.len); } },
        { h: RUN.weights_label || (wTier
            ? ('Weights (synapse \u2265 ' + wTier + ')')
            : 'Weights (admitted synapse mass)'),
          title: 'Per-hop realized synapse mass, -w1-w2- (presence-matrix '
            + 'format; the BOTTLENECK — weakest hop — is bolded): the '
            + 'distinct bodyId edges backing each hop at that position — '
            + 'the same edges behind the Coverage n, NOT the pair\u2019s '
            + 'cross-depth total.'
            + (RUN.weights_label ? ' This table states its SUBJECT: the '
                + 'mass here is the sub-threshold band only; the emitted '
                + 'and total lists sit beside it.' : ''),
          cell: function(row) {
            return dashTd(row.weights, null,
              'Per-hop realized synapse mass; bold = the bottleneck.'); } },
        { h: 'Ratios (all inputs)',
          title: 'Per-hop REALIZED share on every weight basis, -r1-r2- '
            + '(presence-matrix format; the BOTTLENECK — weakest hop — '
            + 'is bolded; shares below 0.01% show (<0.01%)): the mass of '
            + 'the bodyId edges backing that hop at that position / the '
            + 'post type\u2019s full-membership incoming mass — ALL '
            + 'inputs (Min Synapse Count = 1), the same canonical '
            + 'denominator as data_details/connection_type.csv.',
          cell: function(row) {
            return dashTd(row.ratios, ratioDashFmt,
              'Per-hop realized share; bold = the bottleneck; '
              + '(<0.01%) = below 0.01%.'); } },
        !RUN.ratio_basis && !RUN.suppress_adj && hasAdj && { h: adjHeader(RUN.adj_tier || RUN.min_synapse),
          title: 'Per-hop ADJUSTED ratio list, -r1-r2- (presence-matrix '
            + 'format; the weakest entry is bolded; shares below 0.01% '
            + 'show (<0.01%)): derived from THIS chain \u2014 the mass of '
            + 'the bodyId edges backing that hop at that position / the '
            + 'post type\u2019s DATASET incoming counting only edges at or '
            + 'above the run\u2019s Min Synapse Count (the threshold-'
            + 'conditioned denominator behind connection_ratio_adj in '
            + 'data_details/connection_type.csv; always \u2265 Ratios; '
            + 'equals them when N \u2264 1, where the conditioning cuts '
            + 'nothing). Hops whose post type lacks a conditioned '
            + 'denominator render \u2014 in place. The adjusted bottleneck '
            + 'of this chain is the weakest entry of this list.',
          cell: function(row) {
            return dashTd(row.pradj, ratioDashFmt,
              'Per-hop adjusted share; bold = the weakest entry; '
              + '(<0.01%) = below 0.01%.'); } },
        hasSplit && { h: 'Emitted (context)',
          title: 'The EMITTED-class mass of the same hops (the mass already '
            + 'above the run\u2019s applied cut) \u2014 context column, not '
            + 'the subject of this table. Emitted + the sub-threshold '
            + 'Weights list = Total.',
          cell: function(row) { return dashTd(row.emitted, null,
            'Emitted-class hop mass; bold = the weakest hop.'); } },
        hasSplit && { h: 'Total (emitted + sub-threshold)',
          title: 'The full realized mass of these hops at this scope — '
            + 'emitted plus the sub-threshold band. Reconciles with the '
            + '"Complete at ..." view of the same chains.',
          cell: function(row) { return dashTd(row.total_list, null,
            'Total realized hop mass; bold = the weakest hop.'); } },
        RUN.has_universe && { h: 'Universe type weights',
          title: 'Per hop: the WHOLE admitted type-pair mass this chain is '
            + 'drawn from at this scope\u2019s tier, in the same -a-b- hop '
            + 'order \u2014 the type pair\u2019s aggregate weight over ALL '
            + 'its bodyId edges, not this chain\u2019s share of it. Every '
            + 'Weights entry above is a LOWER BOUND of the entry here; '
            + 'neither list is summed across hops. Source: the pair table '
            + 'column universe_type_weights. \u2014 = a hop the pair table '
            + 'does not publish.',
          cell: function(row) {
            return dashTd(row.uni, null,
              'Whole admitted type-pair mass per hop; this chain\u2019s hop '
              + 'mass is a lower bound of it.'); } },
        { h: 'Coverage',
          title: 'Per-node coverage, n1/N1, n2/N2, ...: distinct bodyIds '
            + 'realizing this exact sequence at that position / the '
            + 'type\u2019s population in the run\u2019s discovered '
            + 'network (first entry = source coverage, last = target '
            + 'coverage).',
          cell: function(row) {
            var plain = String(row.coverage || '')
              .replace(/^\[/, '').replace(/\]$/, '');
            return tdc(plain || '—',
              'Per-node bodyId coverage: distinct bodyIds on the emitted '
              + 'paths at that position / the type\u2019s members in the '
              + 'run\u2019s discovered network.'); } },
      ].filter(Boolean);
      var wrap = elt('div', 'sticky-table-container');
      var table = elt('table');
      var thead = elt('thead');
      var hr = elt('tr');
      COLS.forEach(function(col) {
        var th = elt('th', null, col.h);
        if (col.title) { th.title = col.title; }
        hr.appendChild(th); });
      thead.appendChild(hr); table.appendChild(thead);
      var tbody = elt('tbody');
      var rowPaths = [];
      var colSpan = COLS.length;
      lengths.forEach(function(len) {
        var rows = groupsByLen[len].slice().sort(function(a, b) {
          var ka = ratioRun ? (a.mr || 0) : (a.mw || 0);
          var kb = ratioRun ? (b.mr || 0) : (b.mw || 0);
          return kb - ka || (a.path < b.path ? -1 : 1); });
        var gr = elt('tr', 'group-row');
        var gtd = elt('td', null, len + ' hops — ' + rows.length
          + ' shown (' + (union ? 'union, top '
            + RUN.top_per_length + ' per pair' : 'top ' + rows.length)
          + ' of ' + (capacityByLen[len] || rows.length) + ')');
        gtd.colSpan = colSpan;
        gr.appendChild(gtd);
        tbody.appendChild(gr);
        rows.forEach(function(row, i) {
          var tr = elt('tr');
          tr.dataset.path = row.path;
          // §2.1: cells come from the SAME spec as the headers — one
          // loop, one order, always aligned.
          COLS.forEach(function(col) {
            tr.appendChild(col.cell(row, i));
          });
          tbody.appendChild(tr);
          rowPaths.push(row.path);
        });
      });
      table.appendChild(tbody);
      wrap.appendChild(table);
      card.appendChild(wrap);
      var totalShown = rowPaths.length;
      var totalAll = 0;
      pairs.forEach(function(p) { totalAll += p.paths_total; });
      var note = elt('p', 'cap-note');
      note.textContent = 'Showing ' + totalShown + ' of ' + totalAll
        + ' paths (top-' + RUN.top_per_length + ' per pair and length, '
        + 'rank by ' + RUN.rank_by
        + (pairs.length > 1
           ? ', union re-ranked by '
             + (ratioRun ? 'min ratio' : 'min weight')
             + ' within each length' : '')
        + '). Hover a row to highlight its route in the network below.';
      card.appendChild(note);
      return card;
    },

    // Round-8c item 2: the network card carries a tab strip — "All" plus
    // one tab per path length present in the drawn set — each tab
    // rebuilding the layered viz from only that length's capped rows.
    renderNetworkCard: function(rows, baseViz, baseRowPaths, opts) {
      var card = elt('div', 'card');
      card.appendChild(elt('h3', null, 'Network'));
      var byLen = {};
      var lengths = [];
      rows.forEach(function(r) {
        if (byLen[r.len] === undefined) { byLen[r.len] = []; lengths.push(r.len); }
        byLen[r.len].push(r);
      });
      lengths.sort(function(a, b) { return a - b; });

      var legend = elt('div', 'legend');
      var items = [['#ef4444', 'source'], ['#8b5cf6', 'target'],
        ['#2563eb', 'shared intermediate (≥2 paths)'],
        ['#94a3b8', 'unique intermediate (1 path)']];
      items.forEach(function(item) {
        var span = elt('span');
        var chip = elt('span', 'chip');
        chip.style.background = item[0];
        span.appendChild(chip);
        span.appendChild(document.createTextNode(item[1]));
        legend.appendChild(span);
      });
      card.appendChild(legend);

      var tabStrip = elt('div', 'viz-tabs');
      var vizArea = elt('div');
      card.appendChild(tabStrip);
      card.appendChild(vizArea);

      var current = { edgeEls: {} };
      function mountViz(viz, caption) {
        vizArea.innerHTML = '';
        var built = makeSvg(viz);
        current.edgeEls = built.edgeEls;
        var svgWrap = elt('div');
        svgWrap.appendChild(built.svg);
        vizArea.appendChild(svgWrap);
        var netWrap = elt('div', 'network-container');
        netWrap.id = 'net-' + opts.netId;
        vizArea.appendChild(netWrap);
        PENDING_NETS.push({ wrap: netWrap, svg: svgWrap, viz: viz });
        vizArea.appendChild(elt('p', 'cap-note', caption));
      }
      function bindHover() {
        var pane = document.getElementById(cfg.paneId);
        if (!pane) { return; }
        pane.querySelectorAll('tr[data-path]').forEach(function(tr) {
          if (tr.dataset.hoverBound) { return; }
          tr.dataset.hoverBound = '1';
          var ids = (tr.dataset.path || '').split('->');
          if (ids.length < 2) { return; }
          var keys = [];
          for (var k = 0; k + 1 < ids.length; k++) {
            keys.push(ids[k] + '->' + ids[k + 1]);
          }
          tr.addEventListener('mouseenter', function() {
            keys.forEach(function(key) {
              var el = current.edgeEls[key];
              if (el) { el.classList.add('hl'); }
            });
          });
          tr.addEventListener('mouseleave', function() {
            keys.forEach(function(key) {
              var el = current.edgeEls[key];
              if (el) { el.classList.remove('hl'); }
            });
          });
        });
      }
      function activate(key) {
        Array.prototype.forEach.call(tabStrip.children, function(b) {
          b.classList.toggle('active', b.dataset.key === String(key));
        });
        if (key === 'all') {
          mountViz(baseViz, opts.allCaption);
        } else {
          var grp = byLen[key] || [];
          mountViz(buildVizFromRows(grp),
            key + '-hop paths only — ' + grp.length + ' drawn; shared = on '
            + '≥2 of them. Hover a matching table row to highlight its route.');
        }
        bindHover();
        setTimeout(flushNetworks, 0);
      }
      var tabs = [['all', 'All (' + rows.length + ')']];
      lengths.forEach(function(len) {
        tabs.push([String(len), len + ' hops (' + byLen[len].length + ')']);
      });
      tabs.forEach(function(item) {
        var b = elt('button', 'viz-tab-btn', item[1]);
        b.dataset.key = item[0];
        b.type = 'button';
        b.addEventListener('click', function() { activate(item[0]); });
        tabStrip.appendChild(b);
      });
      activate('all');
      return card;
    }
    };
  }
  // Threshold labels for the ratio headers: the run's own Min Synapse
  // Count (from the sniffed metadata).
  var MIN_SYN = DATA.run.meta ? DATA.run.meta.min_synapse : null;
  // ratio-basis runs: the applied threshold is a RATIO, not a synapse
  // cutoff; the Weights scope is the forced min synapse (1) and the
  // adjusted-ratio column is removed (adj == plain by construction).
  var RATIO_BASIS = !!(DATA.run.meta
    && String(DATA.run.meta.weight_basis || '')
      .startsWith('connection_ratio'));
  // the APPLIED threshold (the cut that defined the emitted set) — the
  // Pair Explorer's Weights scope; refill views are asked-scoped
  var APPLIED = DATA.run.meta ? DATA.run.meta.applied_threshold : null;
  // ADJ_TIER: the threshold recorded inside type_incoming_conditioned.csv,
  // i.e. the tier the Adj denominator was actually conditioned at. The
  // header must name this, not the applied cut (F-EX-012's sibling).
  var ADJ_TIER = DATA.run.meta ? (DATA.run.meta.adj_tier || null) : null;
  var Explorer = makeExplorer({
    data: DATA.pairs, units: DATA.run.units,
    run: {rank_by: DATA.run.rank_by, min_synapse: MIN_SYN,
          applied: APPLIED, adj_tier: ADJ_TIER, ratio_basis: RATIO_BASIS,
          pair_ratio_source: DATA.run.pair_ratio_source,
          shortest: DATA.run.shortest,
          top_per_length: DATA.run.top_per_length,
          vispath: DATA.run.vispath || []},
    unitSelId: 'sel-unit', sourceSelId: 'sel-source',
    targetSelId: 'sel-target', paneId: 'pair-pane',
    csvLabel: 'pair_breakdown_paths.csv',
    csvHref: 'paths_pair_breakdown/pair_breakdown_paths.csv',
    filterPairs: true });
  // The Refill view's mode switch (plan-refill-complete-at-asked-paths
  // one explorer, TWO datasets — the COMPLETE-AT-ASKED paths (default)
  // and the REFILL-carrying paths. A mode switch mutates the shared cfg
  // and re-mounts; makeExplorer itself stays untouched (the main
  // Explorer shares it).
  var REFILL_CFG = {
    data: [], units: [],
    run: {
      rank_by: DATA.run.rank_by, pair_ratio_source: '',
      min_synapse: MIN_SYN, applied: APPLIED, adj_tier: ADJ_TIER,
      ratio_basis: RATIO_BASIS,
      shortest: DATA.run.shortest,
      top_per_length: DATA.run.top_per_length, vispath: []
    },
    unitSelId: 'r-sel-unit', sourceSelId: 'r-sel-source',
    targetSelId: 'r-sel-target', paneId: 'refill-pane',
    csvLabel: '', csvHref: '', filterPairs: false };
  var RefillExplorer = makeExplorer(REFILL_CFG);
  // ---- the corridor's OWN explorer (F3 / Q3) ------------------------------
  // The lane's validated walks browsed per source x target with the SAME
  // capped, per-cell-explained table grammar the Complete Pair Explorer uses.
  // Its tier comes from the LANE's own provenance: the Weights header says
  // what the lane admitted, never the Complete run's applied cut
  // (F-EX-012's one-key-two-meanings is the defect this rules out).
  var LANE0 = (((DATA.lane || {}).units) || [])[0] || {};
  var LANE_ASKED = (LANE0.asked === null || LANE0.asked === undefined
                    || LANE0.asked === '')
    ? null : Number(LANE0.asked);
  var LANE_RATIO = String(LANE0.weight_basis || '')
    .indexOf('connection_ratio') === 0;
  var CORRIDOR_CFG = {
    data: (DATA.corridor || {}).pairs || [],
    units: (DATA.corridor || {}).units || [],
    run: {
      rank_by: DATA.run.rank_by,
      pair_ratio_source: (DATA.corridor || {}).pair_ratio_source || '',
      min_synapse: LANE_ASKED, applied: LANE_ASKED, adj_tier: LANE_ASKED,
      ratio_basis: LANE_RATIO,
      // at the synapse >= 1 floor the conditioned denominator IS the plain
      // all-inputs one, so an Adj column would duplicate Ratios (G5)
      suppress_adj: LANE_RATIO || (LANE_ASKED !== null && LANE_ASKED <= 1),
      has_universe: !!(DATA.corridor || {}).has_universe,
      shortest: DATA.run.shortest,
      top_per_length: DATA.run.top_per_length,
      vispath: (DATA.corridor || {}).vispath || [] },
    unitSelId: 'c-sel-unit', sourceSelId: 'c-sel-source',
    targetSelId: 'c-sel-target', paneId: 'corridor-pane',
    csvLabel: 'route_paths_type.csv',
    csvHref: (DATA.corridor && DATA.corridor.csvHref)
             || 'route_scope_paths/route_paths_type.csv',
    filterPairs: false };
  var CorridorExplorer = makeExplorer(CORRIDOR_CFG);
  var CORRIDOR_REFILL_CFG = {
    data: [], units: [],
    run: {
      rank_by: DATA.run.rank_by, pair_ratio_source: '',
      min_synapse: MIN_SYN, applied: APPLIED, adj_tier: ADJ_TIER,
      ratio_basis: RATIO_BASIS,
      shortest: DATA.run.shortest,
      top_per_length: DATA.run.top_per_length, vispath: [] },
    unitSelId: 'cr-sel-unit', sourceSelId: 'cr-sel-source',
    targetSelId: 'cr-sel-target', paneId: 'corridor-refill-pane',
    csvLabel: '', csvHref: '', filterPairs: false };
  var CorridorRefillExplorer = makeExplorer(CORRIDOR_REFILL_CFG);

  // ---- ONE mode/scope/view switch machine (Q5: mirror the complete mode) --
  // The Complete Refill and the corridor Refill share the switch: sets built
  // by the instance, a scope row and a view row that render only when they
  // have >= 2 destinations (G1), a caption that names the choice whenever a
  // row is hidden, and a mount that preserves the selection (F6). Two copies
  // of this logic is how the two modes' grammars would drift apart.
  function makeModeSwitch(spec) {
    return {
      sets: {},
      scope: spec.initialScope || 'asked',
      view: null,
      init: function() {
        var sw = this;
        this.sets = spec.build();
        var keys = Object.keys(this.sets);
        function isFull(key) { return !!spec.full[key]; }
        var hasAsked = keys.some(function(k) { return !isFull(k); });
        var hasFull = keys.some(function(k) { return isFull(k); });
        // ---- resolve the effective scope BEFORE the bar is built (D1) ----
        // Filtering the view buttons against a stale scope used to leave a
        // full-only run with ONE already-active scope chip and NO view
        // buttons, so its Refill-carrying view was unreachable (measured in
        // headless Chrome).
        if (this.scope === 'full' && !hasFull) { this.scope = 'asked'; }
        if (this.scope === 'asked' && !hasAsked && hasFull) {
          this.scope = 'full';
        }
        function inScope(key) {
          return sw.scope === 'full' ? isFull(key) : !isFull(key);
        }
        var views = spec.views.filter(function(item) {
          return sw.sets[item[0]] && inScope(item[0]); });
        var bar = document.getElementById(spec.barId);
        if (bar) {
          bar.innerHTML = '';
          // A control row renders only with >= 2 destinations; the current
          // choice is then stated as a caption, so hiding a row never hides
          // information (barRow / barChip / barCaption are the Analysis
          // bar's own helpers — one implementation for every switch).
          if (hasAsked && hasFull) {
            var scopeRow = barRow(bar, 'Scope');
            [['full', spec.scopeLabels.full],
             ['asked', spec.scopeLabels.asked]].forEach(function(item) {
              var b = barChip(scopeRow, item[1], function() {
                sw.scope = item[0]; sw.init(); });
              b.dataset.scope = item[0];
              b.classList.toggle('active', sw.scope === item[0]);
            });
          } else {
            barCaption(bar, 'Scope: ' + spec.scopeCaptions[this.scope]
              + ' \u2014 the only scope this run produced.');
          }
          if (views.length > 1) {
            var viewRow = barRow(bar, spec.viewRowLabel || 'View');
            views.forEach(function(item) {
              var b = barChip(viewRow, item[1], function() {
                sw.set(item[0]); });
              b.dataset.mode = item[0];
            });
          } else if (views.length === 1) {
            barCaption(bar, 'View: ' + views[0][1]
              + ' \u2014 the only view this scope produced.');
          }
        }
        // Default view: the one already chosen when it survives the scope
        // switch, else the scope's own first preference.
        var first = (this.view && this.sets[this.view] && inScope(this.view))
          ? this.view : null;
        if (!first) {
          var wanted = spec.defaults[this.scope] || [];
          for (var i = 0; i < wanted.length; i += 1) {
            if (this.sets[wanted[i]] && inScope(wanted[i])) {
              first = wanted[i]; break;
            }
          }
        }
        this.set(first);
      },
      set: function(mode) {
        if (!mode || !this.sets[mode]) { return; }
        this.view = mode;
        var sw = this;
        var s = this.sets[mode];
        var cfg = spec.cfg;
        var ov = s.ov || {};
        cfg.data = s.data; cfg.units = s.units;
        cfg.run.pair_ratio_source = s.pair_ratio_source;
        cfg.run.vispath = s.vispath || [];
        cfg.csvLabel = s.csvLabel; cfg.csvHref = s.csvHref;
        // Scope tier labels: each table states the tier ITS denominators are
        // conditioned at — asked scope at the ASKED threshold, full scope at
        // synapse >= 1 — never the pair explorer's applied cut.
        cfg.run.applied = ('applied' in ov) ? ov.applied : APPLIED;
        cfg.run.min_synapse = ('min_synapse' in ov) ? ov.min_synapse : MIN_SYN;
        cfg.run.ratio_basis = ('ratio_basis' in ov)
          ? ov.ratio_basis : RATIO_BASIS;
        cfg.run.suppress_adj = ('suppress_adj' in ov)
          ? ov.suppress_adj : false;
        cfg.run.adj_tier = ('adj_tier' in ov) ? ov.adj_tier : ADJ_TIER;
        // F-EX-002(ii): a carrying view states its subject in the Weights
        // column and shows the emitted/total context beside it. Cleared for
        // the other views so switching never leaves a stale label.
        cfg.run.weights_label = ('weights_label' in ov)
          ? ov.weights_label : null;
        cfg.run.has_split = !!s.split;
        // `Universe type weights` renders only for a dataset whose pair table
        // publishes the whole admitted type-pair mass (F4/Q3).
        cfg.run.has_universe = !!s.has_universe;
        var bar = document.getElementById(spec.barId);
        if (bar) {
          // Each row marks its own kind: a scope chip is lit by the scope, a
          // view chip by the view (one pass over both used to light two
          // views at once).
          bar.querySelectorAll('button[data-mode]').forEach(function(b) {
            b.classList.toggle('active', b.dataset.mode === mode); });
          bar.querySelectorAll('button[data-scope]').forEach(function(b) {
            b.classList.toggle('active', b.dataset.scope === sw.scope); });
        }
        spec.explorer.mount();
      }
    };
  }

  var RefillModes = makeModeSwitch({
    barId: 'refill-mode-bar', cfg: REFILL_CFG, explorer: RefillExplorer,
    initialScope: 'asked',
    // The Path bar owns the 'View' row (Pair Explorer / Refill); this
    // row picks a view WITHIN the Refill page, so the page never shows two
    // rows answering to the same word.
    viewRowLabel: 'Refill view',
    scopeLabels: {full: 'Refill-full (synapse>=1)', asked: 'Refill-asked'},
    scopeCaptions: {
      full: 'Refill-full (synapse >= 1)',
      asked: 'Refill-asked (the run\u2019s own thresholding)'},
    full: {complete_full: true, refill_full: true},
    views: [['complete', 'Complete at asked'],
            ['refill', 'Refill-carrying'],
            ['complete_full', 'Complete at full (synapse>=1)'],
            ['refill_full', 'Refill-carrying (sub-threshold)']],
    defaults: {asked: ['complete', 'refill'],
               full: ['complete_full', 'refill_full']},
    build: function() {
      var sets = {};
      // Scope ASKED (the run's own thresholding; budget-cut recovery).
      if (DATA.refill_complete && DATA.refill_complete.pairs &&
          DATA.refill_complete.pairs.length) {
        sets.complete = {
          data: DATA.refill_complete.pairs,
          units: DATA.refill_complete.units,
          pair_ratio_source: DATA.refill_complete.pair_ratio_source,
          csvLabel: 'complete_at_asked_paths_type.csv',
          csvHref: 'type_level_refill/complete_at_asked_paths_type.csv',
          vispath: (DATA.refill_complete.vispath || []),
          has_universe: !!DATA.refill_complete.has_universe,
          // scope tier labels: the ASKED threshold (never the applied
          // cut — the refill tables are re-valued at the asked tier)
          ov: { applied: null } };
      }
      if (DATA.refill && DATA.refill.pairs && DATA.refill.pairs.length) {
        sets.refill = {
          data: DATA.refill.pairs, units: DATA.refill.units,
          pair_ratio_source: DATA.refill.pair_ratio_source,
          csvLabel: 'refill_paths_type.csv',
          csvHref: 'type_level_refill/refill_paths_type.csv',
          vispath: (DATA.refill.vispath || []),
          has_universe: !!DATA.refill.has_universe,
          ov: { applied: null, weights_label:
                  'Weights — sub-threshold (refill)' },
          split: true };
      }
      // Scope FULL (DUAL REFILL 2026-10-08: synapse >= 1, no tier — the
      // full-connection view of the SAME emitted chains).
      if (DATA.complete_full && DATA.complete_full.pairs &&
          DATA.complete_full.pairs.length) {
        sets.complete_full = {
          data: DATA.complete_full.pairs,
          units: DATA.complete_full.units,
          pair_ratio_source: DATA.complete_full.pair_ratio_source,
          csvLabel: 'refill_full_paths_type.csv',
          csvHref: 'type_level_refill/refill_full_paths_type.csv',
          vispath: (DATA.complete_full.vispath || []),
          has_universe: !!DATA.complete_full.has_universe,
          ov: { applied: null, min_synapse: 1, ratio_basis: false,
                suppress_adj: true } };
      }
      if (DATA.refill_full && DATA.refill_full.pairs &&
          DATA.refill_full.pairs.length) {
        sets.refill_full = {
          data: DATA.refill_full.pairs, units: DATA.refill_full.units,
          pair_ratio_source: DATA.refill_full.pair_ratio_source,
          csvLabel: 'refill_full_carrying_paths_type.csv',
          csvHref: 'type_level_refill/refill_full_carrying_paths_type.csv',
          vispath: (DATA.refill_full.vispath || []),
          has_universe: !!DATA.refill_full.has_universe,
          ov: { applied: null, min_synapse: 1, ratio_basis: false,
                suppress_adj: true,
                weights_label:
                  'Weights — sub-threshold (synapse >= 1)' },
          split: true };
      }
      return sets;
    }});

  var CorridorRefillModes = makeModeSwitch({
    barId: 'corridor-refill-mode-bar', cfg: CORRIDOR_REFILL_CFG,
    explorer: CorridorRefillExplorer,
    initialScope: 'full',
    viewRowLabel: 'Refill view',
    scopeLabels: {full: 'Refill-full (synapse>=1)', asked: 'Refill-asked'},
    scopeCaptions: {
      full: 'Refill-full (synapse >= 1) — the lane\u2019s only second round '
        + '(it has no budget that raises an applied threshold)',
      asked: 'Refill-asked'},
    full: {corridor_complete_full: true, corridor_refill_full: true},
    views: [['corridor_complete_full', 'Complete at full (synapse>=1)'],
            ['corridor_refill_full', 'Refill-carrying (sub-threshold)']],
    defaults: {full: ['corridor_complete_full', 'corridor_refill_full'],
               asked: []},
    build: function() {
      var sets = {};
      if (DATA.corridor_complete_full && DATA.corridor_complete_full.pairs &&
          DATA.corridor_complete_full.pairs.length) {
        sets.corridor_complete_full = {
          data: DATA.corridor_complete_full.pairs,
          units: DATA.corridor_complete_full.units,
          pair_ratio_source: DATA.corridor_complete_full.pair_ratio_source,
          csvLabel: 'route_full_paths_type.csv',
          csvHref: (DATA.corridor_complete_full
                    && DATA.corridor_complete_full.csvHref)
                   || 'route_scope_paths/route_full_paths_type.csv',
          vispath: (DATA.corridor_complete_full.vispath || []),
          has_universe: !!DATA.corridor_complete_full.has_universe,
          ov: { applied: null, min_synapse: 1, ratio_basis: false,
                suppress_adj: true } };
      }
      if (DATA.corridor_refill_full && DATA.corridor_refill_full.pairs &&
          DATA.corridor_refill_full.pairs.length) {
        sets.corridor_refill_full = {
          data: DATA.corridor_refill_full.pairs,
          units: DATA.corridor_refill_full.units,
          pair_ratio_source: DATA.corridor_refill_full.pair_ratio_source,
          csvLabel: 'route_full_carrying_paths_type.csv',
          csvHref: (DATA.corridor_refill_full
                    && DATA.corridor_refill_full.csvHref)
                   || 'route_scope_paths/route_full_carrying_paths_type.csv',
          vispath: (DATA.corridor_refill_full.vispath || []),
          has_universe: !!DATA.corridor_refill_full.has_universe,
          ov: { applied: null, min_synapse: 1, ratio_basis: false,
                suppress_adj: true,
                weights_label:
                  'Weights — sub-threshold (synapse >= 1)' },
          split: true };
      }
      return sets;
    }});

  // ---- boot: parse the deep link, else Overview ----
  function initial() {
    var hash = location.hash || '';
    var params = {};
    hash.replace(/^#/, '').split('&').forEach(function(part) {
      if (!part) { return; }
      var kv = part.split('=');
      params[kv[0]] = decodeURIComponent(kv.slice(1).join('='));
    });
    var page = params.tab || 'overview';
    // The Overview fallback is about a run with NOTHING to show, not about
    // the Complete table being empty: a standalone Fast run has no Complete
    // pairs by design and must still open on its own Analysis page (G8).
    if (page === 'analysis' && params.mode) { Analysis.mode = params.mode; }
    if (page === 'analysis' && params.view) { Analysis.view = params.view; }
    if (Analysis.modes().length === 0 && !DATA.pairs.length) {
      page = 'overview';
    }
    TAB.show(page);
    if (params.pair) {
      var onExplorer = (page === 'explorer')
        || (page === 'analysis' && Analysis.view === 'explorer');
      if (onExplorer) { Explorer.selectPair(params.pair); }
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initial);
  } else {
    initial();
  }
})();
"""


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def _build_unit_drawn(
    frame: pd.DataFrame, rank_by: str, top_per_length: int,
) -> Dict[Tuple[str, str], List[dict]]:
    """Per pair: the capped drawn rows (top-N per length, Rank order)."""
    work = frame.copy()
    work['_path_key'] = work['path']
    ordered = _rank_sort(work, rank_by)
    drawn: Dict[Tuple[str, str], List[dict]] = {}
    for (source, target), group in ordered.groupby(['_source', '_target'], sort=True):
        rows: List[dict] = []
        for _, length_group in group.groupby('_length', sort=True):
            take = length_group.head(top_per_length)
            for _, row in take.iterrows():
                rows.append({
                    'path': str(row['path']),
                    'weights': '' if pd.isna(row['weights']) else str(row['weights']),
                    'ratios': '' if pd.isna(row.get('ratios')) else str(row['ratios']),
                    'min_weight': row['min_weight'],
                    'rank': 0,
                })
        drawn[(source, target)] = rows
    return drawn


def _attach_table_groups(
    frame: pd.DataFrame, rank_by: str, top_per_length: int,
    pairs: List[dict],
    source_cov: Optional[Dict[str, str]] = None,
    target_cov: Optional[Dict[str, str]] = None,
    pair_cov: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
    pair_ratio: Optional[Dict[Tuple[str, str], float]] = None,
    in_conditioned: Optional[Dict[str, float]] = None,
    pair_universe: Optional[Dict[str, Dict[Tuple[str, str], float]]] = None,
) -> None:
    """Fill each pair entry's capped table groups (ranked rows per length),
    each row carrying PER-PATH ``scov``/``tcov`` bodyId n/N strings — the
    first/last entries of the row's own per-node coverage list (distinct
    bodyIds realizing THAT path), falling back to the pair's coverage
    when a row has no list."""
    source_cov = source_cov or {}
    target_cov = target_cov or {}
    pair_cov = pair_cov or {}
    pair_universe = pair_universe or {}
    work = frame.copy()
    work['_path_key'] = work['path']
    ordered = _rank_sort(work, rank_by)
    by_pair: Dict[Tuple[str, str], dict] = {}
    for (source, target), group in ordered.groupby(['_source', '_target'], sort=True):
        pair_entry = pair_cov.get(
            (source, target),
            (source_cov.get(source, ''), target_cov.get(target, '')))
        pair_scov, pair_tcov = pair_entry[0], pair_entry[1]
        groups = []
        for length, length_group in group.groupby('_length', sort=True):
            take = length_group.head(top_per_length)
            rows = []
            for rank, (_, row) in enumerate(take.iterrows(), start=1):
                self_cov = ('' if pd.isna(row.get('coverage'))
                            else str(row['coverage']))
                rows.append({
                    'rank': rank,
                    # plan_rank (R1, 2026-10-15): the lane's acceptance rank
                    # by candidate UPPER BOUND, kept as its own field so the
                    # `#` column can print it where the table has it (the
                    # corridor chains) while the other tables keep the
                    # within-group display rank.
                    'pr_rank': (None if pd.isna(row.get('plan_rank'))
                                else int(row['plan_rank'])),
                    'path': str(row['path']),
                    'len': int(length),
                    'mw': None if pd.isna(row['min_weight']) else float(row['min_weight']),
                    'mr': (None if pd.isna(row.get('min_ratio'))
                           else float(row['min_ratio'])),
                    'pr': _path_pair_ratio(row['path'], pair_ratio or {}),
                    'weights': '' if pd.isna(row['weights']) else str(row['weights']),
                    # F-EX-002(ii): the CARRYING tables state their subject
                    # (the sub-threshold mass) and carry the context beside
                    # them, so the emitted/total lists ride along when the
                    # table has them. The Weights column then gets labelled
                    # for what it holds instead of showing a total under a
                    # "sub-threshold" name.
                    'emitted': _opt_cell(row.get('emitted_weights')),
                    'total': _opt_cell(row.get('total_weights')),
                    # Percent convention (§2.5): the report payload shows
                    # xx.xx% lists; the breakdown CSV keeps raw strings.
                    'ratios': _format_ratio_list(
                        '' if pd.isna(row['ratios'])
                        else str(row['ratios'])),
                    'pradj': (
                        _realized_conditioned_list(
                            row['path'], row.get('weights'), in_conditioned)
                        if in_conditioned
                        else _per_hop_adjusted_list(
                            row['path'], pair_ratio or {})),
                    # `Universe type weights` (F4/Q3): per-hop whole
                    # type-pair mass, from the layout's pair table. '' when
                    # the dataset publishes no such column — the explorer
                    # then prints no such column either.
                    'uni': (_hop_universe_list(row['path'], pair_universe)
                            if pair_universe else ''),
                    # 2026-10-07: the per-path scov/tcov fields are
                    # gone — the coverage list IS the coverage display
                    # (n1/N1>n2/N2>... via the JS cell); rows without
                    # their own list fall back to the pair-scope
                    # endpoints (source>target).
                    'coverage': (
                        self_cov if self_cov else
                        (f'{pair_scov}>{pair_tcov}'
                         if (pair_scov or pair_tcov) else '')),
                })
            groups.append({'len': int(length), 'total': len(length_group),
                           'rows': rows})
        by_pair[(source, target)] = groups
    for entry in pairs:
        entry['table'] = {'groups': by_pair[(entry['source'], entry['target'])]}


def _generated_stamp() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def has_paths_tables(run_dir) -> bool:
    """True when any ``*_allpaths_type.csv`` paths table exists under
    *run_dir* (the only hard requirement of the generator). Cheap
    pre-check so callers can skip non-pathfinding runs."""
    try:
        return any(Path(run_dir).rglob(f'*{PATHS_SUFFIX}'))
    except OSError:
        return False


# F-EX-010 (2026-10-14): a run that finds nothing stops before the export
# block, so it leaves no paths table — and until now no report either. Its
# only documents (the user guide, user_warning_notes.txt) cited artifacts
# that were never written, and nothing in the folder stated WHY. An
# empty-state report says so in the run's own words.
EMPTY_STATE_ABSENT = (
    'data_details/', 'paths_pair_breakdown/', 'route_scope_paths/',
    'type_level_refill/')


def is_pathfinding_run(run_dir) -> bool:
    """True for a pathfinding run folder even when it holds no paths table.

    Recognized by the run-name grammar (Complete / Shortest / cross-dataset)
    plus the artifacts an early exit still writes.
    """
    run_dir = Path(run_dir)
    if run_kind_for(run_dir) == 'Pathfinding':
        return False
    return ((run_dir / 'parameters.txt').exists()
            or (run_dir / 'dataset_data').is_dir())


def _enrollment_counts(folder: Path) -> Tuple[int, int, int, int]:
    """(source bodyIds, source types, target bodyIds, target types) as
    enrolled by the run — the enrollment CSVs survive a zero-path exit."""
    out = []
    for name, type_col in (('source_neurons.csv', 'type'),
                           ('target_neurons.csv', 'type')):
        path = folder / name
        try:
            frame = pd.read_csv(path)
        except (OSError, ValueError):
            out.extend((0, 0))
            continue
        types = (frame[type_col].dropna().astype(str).str.strip().nunique()
                 if type_col in frame.columns else 0)
        out.extend((len(frame), int(types)))
    return tuple(out)


def _write_empty_state_report(run_dir: Path, meta: Dict[str, Any],
                              log=None) -> Path:
    """Render ``path_report.html`` for a run that emitted no paths table."""
    say = log or (lambda *a, **k: None)
    run_dir = Path(run_dir)
    kind = run_kind_for(run_dir, meta)
    # A Network run is not a search that came up empty: it is the induced
    # sub-network of the query, so an empty result means the queried neurons
    # simply share no direct edge at this threshold. Describing that with the
    # path run's "raise Max Interlayer" advice is advice that cannot apply
    # (plan 2026-10-16 C3: the folder grammar was unrecognized, so a
    # zero-connection Network run got no report at all).
    is_network = kind == 'Network'
    text = ''
    params_txt = run_dir / 'parameters.txt'
    if params_txt.exists():
        try:
            text = params_txt.read_text(encoding='utf-8', errors='replace')
        except OSError:
            text = ''

    def _param(key: str, default: str = 'not recorded') -> str:
        match = re.search(rf'^{re.escape(key)}:\s*(.+)$', text, re.MULTILINE)
        return match.group(1).strip() if match else default

    basis = str(meta.get('weight_basis') or '')
    ratio_basis = basis.startswith('connection_ratio')
    thr_label = 'connection ratio ≥' if ratio_basis else 'Min Synapse Count ≥'
    thr_value = (meta.get('min_ratio') if ratio_basis
                 else meta.get('min_synapse')) or '—'
    # the second gate is named only when the run really combined them; a
    # ratio run's Min Synapse 1 is the forced floor, not an AND threshold
    _and = (str(meta.get('threshold_combination') or '') == 'and'
            and ratio_basis and meta.get('min_synapse'))
    prov = meta.get('provenance') or {}
    sb, st, tb, tt = _enrollment_counts(run_dir)
    present = sorted(p.name for p in run_dir.iterdir() if p.is_file())
    absent = [name for name in EMPTY_STATE_ABSENT
              if not (run_dir / name).exists()]
    # D2 (plan-corridor-refill-and-mode-tabs): a Fast-only run has NO
    # Complete-lane table by design — its deliverable is the corridor lane.
    # The empty state must not claim "nothing was found" for such a folder.
    lane_prov: Dict[str, Any] = {}
    for _lane in (LANE_DIR, TYPED_LANE_DIR):
        try:
            lane_prov = json.loads(
                (run_dir / _lane / 'route_provenance.json').read_text(
                    encoding='utf-8'))
            break
        except (OSError, ValueError):
            continue
    lane_ok = str(lane_prov.get('status') or '') == 'complete'
    lane_walks = lane_prov.get('validated_walks')

    rows = [
        ('Threshold that produced nothing', f'{thr_label} {_esc(thr_value)}'
         + (f' AND Min Synapse Count ≥ {_esc(_and)}' if _and else '')),
        ('Weight basis', _esc(basis or 'not recorded')),
        ('Max interlayer', _esc(meta.get('max_interlayer') or '—')),
        ('Dataset', _esc(meta.get('dataset') or _param('dataset'))),
        ('Query', _esc(f'{_param("source name")} → {_param("target name")}')),
        ('Source neurons enrolled', f'{sb} bodyIds / {st} type labels'),
        ('Target neurons enrolled', f'{tb} bodyIds / {tt} type labels'),
        ('Filter by', _esc(meta.get('filter_by') or '—')),
        ('Run mode', _esc(meta.get('run_mode') or _param('run mode'))),
        ('Run date', _esc(_param('run date'))),
    ]
    stat_boxes = ''.join(
        f'<div class="stat-box"><div class="value">{_esc(v)}</div>'
        f'<div class="label">{_esc(k)}</div></div>'
        for k, v in (('direct connections' if is_network else 'paths emitted',
                      0),
                     ('source type labels', st),
                     ('target type labels', tt),
                     ('applied tau', prov.get('strongest_first_tau')
                      or 'not reached')
                     if not is_network else
                     ('graph capped', 'never (no path budget)')))
    prov_rows = ''.join(
        f'<tr><td>{_esc(k)}</td><td>{_esc(prov[k])}</td></tr>'
        for k in PROVENANCE_KEYS if k in prov)
    absent_html = ''.join(f'<li><span class="code">{_esc(n)}</span></li>'
                          for n in absent)
    present_html = ''.join(f'<li><span class="code">{_esc(n)}</span></li>'
                           for n in present)
    lane_section = ''
    if lane_ok:
        try:
            lane_section, _found = _render_corridor_section(
                run_dir, 'Fast (corridor) lane',
                lane_dir=(TYPED_LANE_DIR
                          if (Path(run_dir) / TYPED_LANE_DIR
                              / 'route_provenance.json').exists()
                          and not (Path(run_dir) / LANE_DIR
                                   / 'route_provenance.json').exists()
                          else LANE_DIR))
        except Exception:  # noqa: BLE001 - the lane read is additive here
            lane_section = ''
    headline = ('Complete mode produced no paths' if lane_ok
                else ('No direct connections among the queried neurons'
                      if is_network else 'This run found no paths'))
    intro = (
        f'This folder ran in <b>Fast only</b> mode: the Complete pipeline '
        f'(discovery, enumeration, materialization, refill) was skipped by '
        f'selection, so there is no allpaths table to tabulate here. The '
        f'corridor lane DID produce {lane_walks} validated walk'
        f'{"s" if str(lane_walks) != "1" else ""} — listed below.'
        if lane_ok else
        ('A Network run keeps only connections whose BOTH endpoints are in '
         'the queried set, so an empty result means the neurons you asked '
         'about share no direct edge at this threshold — intermediate '
         'neurons are never involved here. The run\'s own settings are the '
         'explanation:' if is_network else
         'Path enumeration stopped before any path was written, so there is '
         'nothing to tabulate. The run\'s own settings are the explanation:'))
    body = (
        f'<div class="card"><h3>{headline}</h3>'
        f'<p>{intro}</p>'
        f'<div class="grid" style="grid-template-columns:repeat(4,1fr);">'
        f'{stat_boxes}</div>'
        f'<table><tbody>'
        + ''.join(f'<tr><th>{_esc(k)}</th><td>{v}</td></tr>'
                  for k, v in rows)
        + '</tbody></table>'
        '<p class="cap-note">Readouts are the run\'s recorded parameters, not '
        'a re-computation: the engine writes them before enumerating, so a '
        'zero-path folder still carries them.</p></div>'
        + lane_section +
        '<div class="card"><h3>Applied-threshold provenance</h3>'
        + (f'<table><thead><tr><th>key</th><th>value</th></tr></thead>'
           f'<tbody>{prov_rows}</tbody></table>' if prov_rows else
           '<p>No provenance block was recorded — StrongestFirst never '
           'reached a tau, because no source could reach a target.</p>')
        + '</div>'
        f'<div class="card"><h3>Why the other documents point at missing files</h3>'
        f'<p><span class="badge badge-warning">Not written</span> these '
        f'folders and files do not exist for this run, even where '
        f'<span class="code">user_warning_notes.txt</span> or the run guide '
        f'mentions them:</p><ul>{absent_html}</ul>'
        f'<p><span class="badge badge-info">Present</span></p>'
        f'<ul>{present_html}</ul></div>'
        f'<div class="card"><h3>{"Why Complete mode found nothing" if lane_ok else "What usually moves this"}</h3><ul>'
        + (f'<li>Lower the threshold — a ratio gate of '
           f'{_esc(thr_value)} asks every hop to carry that share of the post '
           f'neuron\'s all-post incoming mass.</li>' if ratio_basis else
           # the synapse run has no ratio gate at all: naming one here (and
           # calling its synapse count a "share") was a claim about a knob the
           # run never used
           f'<li>Lower Min Synapse Count — a gate of '
           f'{_esc(thr_value)} synapses asks every hop to be backed by at '
           f'least that much measured mass between the pair.</li>')
        + ('<li>Add the neurons that connect them: this tool shows only '
           '<i>direct</i> connections, so a set that is disjoint on paper '
           'may still be densely connected through intermediates — run '
           'Complete Paths (or Shortest Paths) over the same sets.</li>'
           if is_network else
           '<li>Raise Max Interlayer so a longer chain can connect the '
           'sets.</li>') +
        f'<li>Check <span class="code">user_warning_notes.txt</span>: '
        f'<span class="code">drop_untyped=True</span> removes edges touching '
        f'untyped neurons, and an untyped source or target stays enrolled in '
        f'the CSVs while its incident edges are gone.</li>'
        + ('' if is_network else
           '<li>Relax the Edge Budget / re-enable "Limit Graph Edges" if the '
           'graph was capped before the search.</li>')
        + f'</ul></div>')
    html_text = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f'<title>{_esc(f"Paths Pair Report — {run_dir.name}")}</title>\n'
        f'<style>{REPORT_CSS}</style>\n</head>\n<body>\n<div class="container">\n'
        f'<header><h1>🛤️ Paths Pair Report</h1>'
        f'<p>{_esc(kind)} — {_esc(run_dir.name)}</p>'
        f'<p>{_esc(_meta_summary({"meta": meta, "name": run_dir.name}))}</p>'
        f'<p style="opacity:0.8;">Generated: {_esc(_generated_stamp())} · '
        f'{"Fast-lane-only run (no Complete-mode paths table)" if lane_ok else "empty state (no paths table in this folder)"}</p></header>\n'
        f'{body}\n</div>\n</body>\n</html>\n')
    report = run_dir / REPORT_NAME
    report.write_text(html_text, encoding='utf-8')
    say(f'.{"lane-only" if lane_ok else "empty-state"} report: {report}')
    return report


def generate_paths_pair_report(
    run_dir, top_per_length: int = DEFAULT_TOP_PER_LENGTH,
    rank_by: str = 'min_weight', log=None,
    global_pairs: int = DEFAULT_GLOBAL_PAIRS,
    global_edges: int = DEFAULT_GLOBAL_EDGES,
    root_report: bool = True,
) -> Path:
    """Generate the pair report + breakdown CSVs for one run folder.

    Writes ONLY ``path_report.html`` and ``paths_pair_breakdown/`` into
    the run folder; every other file is left untouched. Returns the
    report path, or the nested-reports directory when
    ``root_report=False`` (cross-dataset layout, 2026-10-11 user
    directive: the cross root keeps only ``comparison_report.html`` —
    the per-delegate pair reports live beside each delegate in
    ``dataset_data/<dataset>/<delegate>/``; the nested generation below
    still runs and the root render is skipped).
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f'run folder not found: {run_dir}')
    say = log or (lambda *a, **k: None)

    meta = _sniff_metadata(run_dir)
    if (rank_by == 'min_weight'
            and str(meta.get('weight_basis') or '')
            .startswith('connection_ratio')):
        # Ratio-basis run: min_ratio IS the strength definition (the
        # value StrongestFirst ordered and tau cut on).
        rank_by = 'min_ratio'
    units = discover_units(run_dir)
    # A standalone Fast run has no Complete paths table at all — only the
    # lane's. Give it the lane as its unit instead of an empty state (D2).
    lane_only = not units
    if lane_only:
        root_unit = lane_unit(run_dir, meta)
        if root_unit is not None:
            units = [root_unit]
    if not units:
        if is_pathfinding_run(run_dir) and root_report:
            # F-EX-010: a zero-path run is the run's own result, and the
            # folder must say so instead of staying silent.
            return _write_empty_state_report(run_dir, meta, log=say)
        raise FileNotFoundError(
            f'no *_allpaths_type.csv paths table found under {run_dir}')
    # "(run root)" is retired: the report's own root unit gets an
    # informative dataset/threshold label (round-8 item 5).
    for unit in units:
        if not unit.unit_id:
            unit.label = _root_display_label(run_dir, meta,
                                             run_kind_for(run_dir, meta))
    say(f'.units: {len(units)}')

    frames: Dict[str, pd.DataFrame] = {}
    unit_path_rows: Dict[str, List[dict]] = {}
    unit_inter_rows: Dict[str, List[dict]] = {}
    unit_pair_stats: Dict[str, Dict[Tuple[str, str], dict]] = {}
    unit_inter_counts: Dict[str, Dict[Tuple[str, str], Counter]] = {}
    unit_inter_minhop: Dict[str, Dict[Tuple[str, str], Dict[str, int]]] = {}
    unit_drawn: Dict[str, Dict[Tuple[str, str], List[dict]]] = {}
    unit_source_cov: Dict[str, Dict[str, str]] = {}
    unit_target_cov: Dict[str, Dict[str, str]] = {}
    unit_pair_cov: Dict[str, Dict[Tuple[str, str], Tuple[str, str]]] = {}
    unit_query_cov: Dict[str, Dict[str, Tuple[int, int]]] = {}
    unit_pair_ratio_source: Dict[str, str] = {}
    unit_pair_level_ratio: Dict[str, Dict[Tuple[str, str], float]] = {}
    unit_in_conditioned: Dict[str, Dict[str, float]] = {}

    for unit in units:
        frame = load_unit_paths(unit.csv_path)
        frames[unit.unit_id] = frame
        source_cov, target_cov = _load_enrollment_coverage(unit.folder)
        unit_source_cov[unit.unit_id] = source_cov
        unit_target_cov[unit.unit_id] = target_cov
        unit_query_cov[unit.unit_id] = _load_query_coverage(unit.folder)
        unit_in_conditioned[unit.unit_id] = _load_type_incoming_conditioned(
            unit.folder)
        if not meta.get('adj_tier'):
            # F-EX-012's header sibling: label the Adj column with the
            # denominator's own recorded tier, never the applied cut.
            meta['adj_tier'] = _conditioned_threshold(unit.folder)
        pair_bodyid = _load_pair_bodyid_coverage(unit.folder)
        pair_level_ratio, pair_ratio_source = _load_pair_level_ratios(
            unit.folder)
        unit_pair_ratio_source[unit.unit_id] = pair_ratio_source
        unit_pair_level_ratio[unit.unit_id] = pair_level_ratio
        path_rows, inter_rows, pair_stats = build_unit_breakdown(
            frame, rank_by, source_cov=source_cov, target_cov=target_cov,
            pair_bodyid=pair_bodyid, pair_ratio=pair_level_ratio)
        unit_pair_cov[unit.unit_id] = {
            key: (stats.get('scov', ''), stats.get('tcov', ''),
                  stats.get('cov_scope', 'type'))
            for key, stats in pair_stats.items()
        }
        unit_path_rows[unit.unit_id] = path_rows
        unit_inter_rows[unit.unit_id] = inter_rows
        unit_pair_stats[unit.unit_id] = pair_stats
        counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        minhop: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(dict)
        for row in inter_rows:
            key = (row['source'], row['target'])
            counts[key][row['intermediate']] = row['n_paths_using']
            minhop[key][row['intermediate']] = row['min_hop_position']
        unit_inter_counts[unit.unit_id] = counts
        unit_inter_minhop[unit.unit_id] = minhop
        unit_drawn[unit.unit_id] = _build_unit_drawn(
            frame, rank_by, top_per_length)
        say(f'  · {unit.label}: {len(frame)} paths, '
            f'{len(pair_stats)} pairs')

    out_dir = run_dir / OUTPUT_DIR_NAME
    breakdown_paths = write_breakdown_csvs(
        out_dir, units, unit_path_rows, unit_inter_rows, unit_pair_stats)

    vispath_links = _discover_vispath_links(run_dir)
    payload = build_payload(
        run_dir.name, run_kind_for(run_dir, meta), meta,
        units, frames, unit_inter_counts, unit_inter_minhop,
        unit_pair_stats, unit_drawn, top_per_length, rank_by,
        vispath_links=vispath_links,
        unit_pair_ratio_source=unit_pair_ratio_source)
    _attach_table_groups_multi(units, frames, payload, rank_by, top_per_length,
                               unit_source_cov, unit_target_cov,
                               unit_pair_cov, unit_pair_level_ratio,
                               unit_in_conditioned=unit_in_conditioned)
    payload['global'] = build_global(
        units, frames, global_pairs, global_edges,
        unit_source_cov=unit_source_cov, unit_target_cov=unit_target_cov,
        unit_pair_cov=unit_pair_cov, unit_query_cov=unit_query_cov)

    # -- Refill datasets = the second path-building round. TWO tables
    # when the run's mirror carries both (R1, plan
    # plan-refill-complete-at-asked-paths): the REFILL-carrying paths
    # (refill_paths_type.csv) and the COMPLETE-AT-ASKED paths
    # (complete_at_asked_paths_type.csv — every asked-threshold chain
    # with the emitted/refill/skipped per-hop split). Both run through
    # the SAME machinery as the main tables; the Refill view mounts a
    # second Pair Explorer with a mode toggle (complete = default, D2).

    def _build_refill_dataset(kind: str, filename: str,
                              layout: str = 'complete',
                              lane_dir: Optional[str] = None) -> Optional[dict]:
        """One capped Explorer dataset built from a second-round table.

        `layout` says which namespace the round lives in: the Complete
        mode's refill mirrors into ``type_level_refill/`` +
        ``data_details/type_level_refill/``, while the corridor lane keeps
        ONE namespace (``route_scope_paths/``) for its asked table and its
        synapse >= 1 second round (Q2/Q5 — same grammar, lane namespace).
        """
        lay = DATASET_LAYOUTS[layout]
        if lane_dir and layout == 'corridor':
            # Issue 2026-10-22 §2.1: the typed lane owns a separate
            # namespace; a typed-only run's Explorer must read it.
            lay = dict(lay)
            lay['paths'] = lane_dir
            lay['pairs'] = lane_dir
            lay['incoming'] = f'{lane_dir}/route_type_incoming_asked.csv'
            lay['incoming_full'] = f'{lane_dir}/route_type_incoming.csv'
        frames_k: Dict[str, pd.DataFrame] = {}
        units_k: List[Unit] = []
        ratio_k: Dict[str, Dict[Tuple[str, str], float]] = {}
        ratio_src_k: Dict[str, str] = {}
        # DUAL REFILL: the FULL scope (synapse >= 1) reads its OWN pair
        # table with the plain full ratio — the asked table's adjusted
        # values are threshold-conditioned at the run's min synapse and
        # are wrong under the full scope (user-caught 2026-10-09).
        full_kind = kind.endswith('_full') or kind.endswith('_full_asked')
        pairs_name = (lay['pairs_full'] if full_kind
                      else lay['pairs_asked'])
        # Scope also decides WHICH denominator file and WHICH mirror pages
        # the set reads: the corridor conditions its asked view at the lane's
        # own tier and its synapse >= 1 view at the floor (the complete
        # layout points both at one file, which is what its single mirror
        # already assumes).
        incoming_rel = lay['incoming_full' if full_kind else 'incoming']
        vispath_spec = lay['vispath_full' if full_kind else 'vispath']
        uni_k: Dict[str, Dict[Tuple[str, str], float]] = {}
        asked_in_k: Dict[str, Dict[str, float]] = {}
        for unit in units:
            csv_path = unit.folder / lay['paths'] / filename
            if not csv_path.exists():
                continue
            try:
                rframe = load_unit_paths(csv_path)
            except Exception as exc:  # noqa: BLE001 - refill is additive
                say(f'  ! {kind} paths for {unit.label} unreadable: {exc}')
                continue
            if rframe.empty:
                continue
            frames_k[unit.unit_id] = rframe
            units_k.append(unit)
            ratio_map, ratio_src = _load_refill_pair_ratios(
                unit.folder, pairs_name=pairs_name,
                prefer_adj=not full_kind, subdir=lay['pairs'])
            ratio_k[unit.unit_id] = ratio_map
            ratio_src_k[unit.unit_id] = ratio_src
            # `Universe type weights` (F4/Q3): the whole admitted type-pair
            # mass beside the chain's own hop mass, from the same pair table
            # the ratios come from. Only the layouts that publish the column
            # get the readout.
            if lay.get('universe'):
                uni_k[unit.unit_id] = _load_pair_universe(
                    unit.folder, pairs_name=pairs_name, subdir=lay['pairs'])
            # CHAIN-derived Adj (fix 2026-10-10): the >=threshold incoming
            # per type, exported by the refill — the conditioned
            # denominator the realized hop masses divide by.
            asked_csv = unit.folder / incoming_rel
            if asked_csv.exists():
                try:
                    asked_in_k[unit.unit_id] = {
                        str(r['type_post']): float(r['incoming_threshold'])
                        for _, r in pd.read_csv(asked_csv).iterrows()}
                except Exception:  # noqa: BLE001 - additive
                    pass
        if not units_k:
            return None
        pairs_k: List[dict] = []
        for unit in units_k:
            rframe = frames_k[unit.unit_id]
            r_rows, r_inter, r_stats = build_unit_breakdown(
                rframe, rank_by,
                source_cov=unit_source_cov.get(unit.unit_id) or {},
                target_cov=unit_target_cov.get(unit.unit_id) or {},
                pair_ratio=ratio_k.get(unit.unit_id))
            r_drawn = _build_unit_drawn(rframe, rank_by, top_per_length)
            r_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
            r_minhop: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
                dict)
            for row in r_inter:
                key = (row['source'], row['target'])
                r_counts[key][row['intermediate']] = row['n_paths_using']
                r_minhop[key][row['intermediate']] = row['min_hop_position']
            for (source, target), stats in sorted(
                    r_stats.items(),
                    key=lambda item: (-item[1]['paths'], item[0])):
                # Keyed by THIS pair — `key` is the r_inter loop's stale
                # leftover and is unbound whenever a unit has no
                # intermediates rows (1-hop tables, network runs).
                pair_key = (source, target)
                pairs_k.append(build_pair_entry(
                    unit, source, target, stats, r_counts[pair_key],
                    r_minhop[pair_key], r_drawn[(source, target)]))
        used_ids: Dict[str, int] = {}
        for entry in pairs_k:
            base = _query_report_slug(
                f"{entry['unit']}__{entry['source']}__{entry['target']}")
            entry['id'] = base
            if base in used_ids:
                used_ids[base] += 1
                entry['id'] = f'{base}_{used_ids[base]}'
            else:
                used_ids[base] = 1
        for unit in units_k:
            entries = [e for e in pairs_k if e['unit'] == unit.unit_id]
            if entries:
                _attach_table_groups(
                    frames_k[unit.unit_id], rank_by, top_per_length,
                    entries,
                    source_cov=unit_source_cov.get(unit.unit_id) or {},
                    target_cov=unit_target_cov.get(unit.unit_id) or {},
                    pair_ratio=ratio_k.get(unit.unit_id),
                    in_conditioned=asked_in_k.get(unit.unit_id),
                    pair_universe=uni_k.get(unit.unit_id))
        # vispath network links for the pane footer (the mirror folder's
        # per-level vispath networks cover every pair)
        vis_links: List[Dict[str, str]] = []
        seen_hrefs: set = set()
        for unit in units_k:
            prefix = (f'dataset_data/{unit.unit_id}/'
                      if unit.dataset else '')
            for fname, label in vispath_spec:
                href = f'{prefix}{lay["paths"]}/{fname}'
                if (unit.folder / lay['paths'] / fname).exists() \
                        and href not in seen_hrefs:
                    seen_hrefs.add(href)
                    vis_links.append({'href': href, 'label': label})
        # The run's OWN visualization pages (user, 2026-10-15): the lane's
        # mirror pages are skipped while these exist, because they draw a
        # strict SUPERSET of the lane's scope in the same style — so a both-
        # modes run links them here rather than duplicating the picture.
        for unit in units_k:
            prefix = (f'dataset_data/{unit.unit_id}/'
                      if unit.dataset else '')
            viz_dir = unit.folder / 'visualization'
            if not viz_dir.is_dir():
                continue
            for pattern, label in (
                    ('Network_*.html', '🕸️ run Network (all pairs, NT-coloured)'),
                    ('Heatmap_*.html', '🔥 run Heatmap (all pairs)')):
                for hit in sorted(viz_dir.glob(pattern))[:1]:
                    href = f'{prefix}visualization/{hit.name}'
                    if href not in seen_hrefs:
                        seen_hrefs.add(href)
                        vis_links.append({'href': href, 'label': label})

        return {
            'units': [{'id': u.unit_id, 'label': u.label,
                       'dataset': u.dataset, 'threshold': u.threshold}
                      for u in units_k],
            'pairs': pairs_k,
            'pair_ratio_source': next(
                (ratio_src_k.get(u.unit_id, '') for u in units_k
                 if ratio_src_k.get(u.unit_id)), ''),
            'has_universe': any(uni_k.values()),
            'vispath': vis_links,
        }

    refill_payload = _build_refill_dataset(
        'refill', 'refill_paths_type.csv')
    refill_complete_payload = _build_refill_dataset(
        'complete', 'complete_at_asked_paths_type.csv')
    if refill_complete_payload is not None:
        # completeness stamp: the AND over every contributing unit's own
        # provenance (a cross root is complete only when all delegates are)
        stamps = []
        for unit in units:
            prov_path = unit.folder / 'data_details' / 'type_level_refill' / (
                'refill_provenance.json')
            if not prov_path.exists():
                continue
            try:
                stamps.append(json.loads(
                    prov_path.read_text(encoding='utf-8')).get(
                    'complete_at_asked'))
            except Exception:  # noqa: BLE001
                stamps.append(None)
        refill_complete_payload['complete_at_asked'] = (
            all(s is True for s in stamps) if stamps else None)

    # DUAL REFILL (2026-10-08): the refill-full scope (synapse >= 1, no
    # tier) — carrying = chains with sub-threshold edges at full scope;
    # complete = every emitted chain re-valued at full scope.
    refill_full_payload = _build_refill_dataset(
        'refill_full', 'refill_full_carrying_paths_type.csv')
    complete_full_payload = _build_refill_dataset(
        'complete_full', 'refill_full_paths_type.csv')

    # -- Corridor (Fast) datasets = the lane's own two rounds (Q2/Q5: the
    # SAME grammar as the complete mode, in the lane's namespace). The asked
    # view is the lane's validated walks — the per-pair breakdown the corridor
    # tab never had (F3); the full views are the lane's synapse >= 1 second
    # round (`route_full_*`, Phase 1).
    _fast_lane = any(
        (u.folder / LANE_DIR / LANE_WALKS_NAME).exists() for u in units)
    _typed_lane = any(
        (u.folder / TYPED_LANE_DIR / LANE_WALKS_NAME).exists()
        for u in units)
    _corridor_lane = (LANE_DIR if _fast_lane or not _typed_lane
                      else TYPED_LANE_DIR)
    corridor_payload = _build_refill_dataset(
        'corridor', LANE_WALKS_NAME, layout='corridor',
        lane_dir=_corridor_lane)
    corridor_complete_full_payload = _build_refill_dataset(
        'corridor_complete_full', 'route_full_paths_type.csv',
        layout='corridor', lane_dir=_corridor_lane)
    corridor_refill_full_payload = _build_refill_dataset(
        'corridor_refill_full', 'route_full_carrying_paths_type.csv',
        layout='corridor', lane_dir=_corridor_lane)
    for _p, _name in (
            (corridor_payload, LANE_WALKS_NAME),
            (corridor_complete_full_payload, 'route_full_paths_type.csv'),
            (corridor_refill_full_payload,
             'route_full_carrying_paths_type.csv')):
        if _p is not None:
            _p['csvHref'] = f'{_corridor_lane}/{_name}'

    payload['refill'] = refill_payload
    payload['refill_complete'] = refill_complete_payload
    payload['refill_full'] = refill_full_payload
    payload['complete_full'] = complete_full_payload
    payload['corridor'] = corridor_payload
    payload['corridor_complete_full'] = corridor_complete_full_payload
    payload['corridor_refill_full'] = corridor_refill_full_payload

    def _has_pairs(key: str) -> bool:
        block = payload.get(key)
        return bool(block and block.get('pairs'))

    # Which Analysis destinations this run actually produced. The bar renders
    # a row only when it has >= 2 destinations, so this list is the report's
    # own presence law, not a guess from the folder name (G1).
    modes = []
    if payload.get('pairs') and not lane_only:
        modes.append('complete')
    if _has_pairs('corridor') or _has_pairs('corridor_complete_full') \
            or _has_pairs('corridor_refill_full'):
        modes.append('corridor')
    payload['run']['modes'] = modes
    views = {
        'complete': ([] if lane_only else
                     [v for v, ok in (
                         ('explorer', bool(payload.get('pairs'))),
                         ('refill', _has_pairs('refill')
                                  or _has_pairs('refill_complete')
                                  or _has_pairs('refill_full')
                                  or _has_pairs('complete_full'))) if ok]),
        'corridor': [v for v, ok in (
            ('explorer', _has_pairs('corridor')),
            ('refill', _has_pairs('corridor_complete_full')
                     or _has_pairs('corridor_refill_full'))) if ok],
    }
    payload['run']['views'] = views
    # The lane's own provenance, for the captions that must name the tier a
    # corridor view is valued at (never inferred from a Complete threshold).
    lane_prov = []
    refused_prov = []
    for unit in units:
        prov = _lane_provenance(unit.folder, unit.lane_dir)
        if prov:
            lane_prov.append({
                'unit': unit.unit_id, 'label': unit.label,
                'asked': prov.get('asked'),
                'validated_walks': prov.get('validated_walks'),
                'scope_pairs': prov.get('scope_pairs'),
                'runtime_s': prov.get('runtime_s'),
                'time_capped': prov.get('time_capped'),
                'weight_basis': prov.get('weight_basis'),
                'run_mode': prov.get('run_mode') or prov.get('corridor_mode'),
                'lane_dir': unit.lane_dir,
                'type_pruned': bool(prov.get('type_prune')),
                'type_pruned_complete': prov.get('type_pruned_complete'),
                'full': prov.get('full') or {}})
            continue
        # A lane that ATTEMPTED and refused still has a fact to report.
        # Under Option B the corridor page only exists when the lane has
        # data, so a refusal would otherwise vanish from the report — and
        # "Fast Mode = both but the lane never ran" is exactly what a reader
        # needs to see (D2's disclosure duty, plan §9 risk 2).
        raw = unit.folder / unit.lane_dir / LANE_PROV_NAME
        if raw.exists():
            try:
                rp = json.loads(raw.read_text(encoding='utf-8'))
            except Exception:  # noqa: BLE001 - unreadable == nothing to say
                continue
            refused_prov.append({
                'unit': unit.unit_id, 'label': unit.label,
                'status': str(rp.get('status') or 'refused'),
                'reason': str(rp.get('reason') or 'not generated')})
    payload['lane'] = {'present': bool(lane_prov), 'units': lane_prov,
                       'refusals': refused_prov, 'lane_only': lane_only}

    payload_json = json.dumps(payload, ensure_ascii=True).replace('</', '<\\/')

    # Cross-dataset runs also embed a single-unit report INTO every
    # dataset_data/<dataset>/<delegate>/ folder (user request 2026-10-02):
    # browsing a delegate beside its own parameters.txt/all_attributes.json
    # gets its own self-consistent report with working relative links.
    # Nested generation happens BEFORE the root render so the root's Data
    # tab can link the per-delegate reports via _link_if_exists.
    nested_reports: List[Path] = []
    if any(u.dataset for u in units):
        for unit in units:
            if not unit.dataset:
                continue
            try:
                nested_reports.append(generate_paths_pair_report(
                    unit.folder, top_per_length=top_per_length,
                    rank_by=rank_by,
                    global_pairs=global_pairs, global_edges=global_edges,
                    log=None))
            except Exception as exc:  # noqa: BLE001 - best-effort extras
                say(f'  ! nested report for {unit.label} skipped: {exc}')
        if nested_reports:
            say(f'.nested reports: {len(nested_reports)} delegate folders')

    # link the exported user guide when it exists (the detailed
    # explanation of adjusted ratios and the refill lives there)
    guide_file = run_dir / '_UserGuide_please_read_me.html'
    guide_link = ''
    if guide_file.exists():
        guide_link = (' &middot; <a href="_UserGuide_please_read_me.html" '
                      'target="_blank" style="color:#ffd54f">\U0001f4d6 '
                      'user guide (ratios, adjusted ratios, refill '
                      'explained)</a>')
    html_str = _render_page_shell(
        payload_json,
        _render_overview(payload, run_dir, breakdown_paths),
        _render_global_tab(payload),
        _render_data_tab(payload, run_dir),
        _render_explorer_template(payload['run']),
        payload['run'],
        _generated_stamp(),
        refill_html=_render_refill_tab(payload, run_dir),
        corridor_html=_render_corridor_explorer_tab(payload, run_dir),
        corridor_refill_html=_render_corridor_refill_tab(payload, run_dir),
        guide_link=guide_link,
    )
    if not root_report:
        say('.root report skipped (cross-dataset layout — per-delegate '
            'reports carry the pair views)')
        return run_dir
    report_path = run_dir / REPORT_NAME
    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write(html_str)
    say(f'.report: {report_path}')
    return report_path


def _attach_table_groups_multi(
    units: Sequence[Unit], frames: Dict[str, pd.DataFrame],
    payload: dict, rank_by: str, top_per_length: int,
    unit_source_cov: Optional[Dict[str, Dict[str, str]]] = None,
    unit_target_cov: Optional[Dict[str, Dict[str, str]]] = None,
    unit_pair_cov: Optional[Dict[str, Dict[Tuple[str, str],
                                           Tuple[str, str]]]] = None,
    unit_pair_ratio: Optional[Dict[str, Dict[Tuple[str, str],
                                             float]]] = None,
    unit_in_conditioned: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    by_unit: Dict[str, List[dict]] = {u.unit_id: [] for u in units}
    for entry in payload['pairs']:
        by_unit[entry['unit']].append(entry)
    for unit in units:
        entries = by_unit[unit.unit_id]
        if entries:
            _attach_table_groups(
                frames[unit.unit_id], rank_by, top_per_length, entries,
                source_cov=(unit_source_cov or {}).get(unit.unit_id),
                target_cov=(unit_target_cov or {}).get(unit.unit_id),
                pair_cov=(unit_pair_cov or {}).get(unit.unit_id),
                pair_ratio=(unit_pair_ratio or {}).get(unit.unit_id),
                in_conditioned=(unit_in_conditioned or {}).get(unit.unit_id))
