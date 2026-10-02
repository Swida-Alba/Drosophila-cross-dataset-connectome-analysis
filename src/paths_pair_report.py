"""Paths pair report — per-source-target-pair HTML report for pathfinding runs.

Plan: ``_plan/plan-paths-pair-report.md``.

Reads the paths tables a run already produced (``*_allpaths_type.csv`` —
Complete Paths, Shortest Paths, or the per-dataset ``minsyn_<N>/`` delegates
inside a cross-dataset run) and writes, additively into the run folder:

* ``paths_pair_breakdown/pair_breakdown_paths.csv`` — one row per path,
  uncapped, with the pair/rank columns the HTML caps reference.
* ``paths_pair_breakdown/pair_breakdown_intermediates.csv`` — one row per
  (pair, intermediate): how many paths use it, ``shared`` (>=2 paths of the
  same pair) vs ``unique`` (exactly 1), earliest hop position.
* ``path_report.html`` — self-contained zero-CDN report in the
  cross-dataset report's visual language (house CSS classes, tabbed router
  from ``comparison/report_tabbed.py``). Four pages: Overview / Global
  (cross-dataset-analysis-style run-wide presentations: per-unit stats,
  the pair × unit path-count matrix, unit-coverage histogram, and a
  top-edges global route network) / Pair Explorer (two selection boxes —
  Source and Target — render one pair pane on demand) / Data. Each pair
  pane carries a per-pair paths presence matrix (paths x units, house
  presence-table conventions), the capped per-length top-10 table, and an
  inline-SVG layered DAG coloring shared (blue) vs unique (gray)
  intermediates.

Capping is viewport-only: every capped view states its cap and links to the
uncapped CSV beside the report. Ranking is bottleneck-first
(``min_weight`` desc, ``path_prob`` desc, ``path`` asc). The generator
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
RANK_KEYS = ('min_weight', 'path_prob', 'length')

# Folders that never hold a paths table unit (viz/analysis sidecars).
SKIP_DIRS = {
    'data_details', 'visualization', 'visualization_data',
    'bodyId_visualization', 'network_early', 'network_early_bodyId',
    'find_reciprocal', 'hemisphere_symmetry', 'custom_groups',
    'comparison_results', 'comparison_visualizations',
    'comparison_report_used_data', 'similarity_matrices', 'conserved_paths',
    '_density', 'cache', '__pycache__',
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
    """The vis-network library as an inline <script> block.

    Inlines the vendored ``src/assets/vis-network.min.js`` so the
    interactive networks work fully offline. HTML only terminates a
    script block on the sequence ``</script``, so exactly that sequence
    is neutralized (``<\\/script`` — an identity escape inside JS
    strings/regexes); a blanket ``</`` escape would corrupt regex
    literals ("Invalid regular expression flags"). Falls back to the
    house CDN tag when the vendored asset is missing.
    """
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


def _threshold_of(unit_id: str) -> str:
    match = re.search(r'minsyn_(\d+)', unit_id)
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


# ---------------------------------------------------------------------------
# run metadata (opportunistic — the CSVs are the only hard requirement)
# ---------------------------------------------------------------------------

def run_kind(run_name: str) -> str:
    if run_name.startswith('find-paths-shortest'):
        return 'Shortest Paths'
    if run_name.startswith('find-paths-complete'):
        return 'Complete Paths'
    if run_name.startswith('cross-dataset'):
        return 'Cross-Dataset > Paths'
    return 'Pathfinding'


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
            ('max_interlayer', r'max interlayer:\s*([^\s]+)'),
            ('dataset', r'^dataset:\s*([^\s]+)'),
            ('filter_by', r'filter by:\s*([^\s]+)'),
            ('separate_hemispheres', r'separate hemispheres:\s*([^\s]+)'),
        ):
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                meta[key] = match.group(1)
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

def load_unit_paths(csv_path: Path) -> pd.DataFrame:
    """Load one paths CSV and derive the pair columns the report needs."""
    df = pd.read_csv(csv_path, dtype={'path': str})
    if 'path' not in df.columns:
        raise ValueError(f'{csv_path}: missing required "path" column')
    df = df[df['path'].notna()].copy()
    nodes = df['path'].str.split('->')
    df['_source'] = nodes.str[0]
    df['_target'] = nodes.str[-1]
    df['_length'] = nodes.str.len() - 1
    if 'min_weight' not in df.columns:
        df['min_weight'] = float('nan')
    if 'path_prob' not in df.columns:
        df['path_prob'] = float('nan')
    if 'weights' not in df.columns:
        df['weights'] = ''
    df['min_weight'] = pd.to_numeric(df['min_weight'], errors='coerce')
    df['path_prob'] = pd.to_numeric(df['path_prob'], errors='coerce')
    return df


def _rank_sort(frame: pd.DataFrame, rank_by: str) -> pd.DataFrame:
    ascending = [rank_by == 'length', False, True]
    return frame.sort_values(
        [rank_by, 'path_prob', '_path_key'],
        ascending=ascending, na_position='last', kind='mergesort')


def build_unit_breakdown(
    frame: pd.DataFrame, rank_by: str,
    source_cov: Optional[Dict[str, str]] = None,
    target_cov: Optional[Dict[str, str]] = None,
) -> Tuple[List[dict], List[dict], Dict[Tuple[str, str], dict]]:
    """Per-pair breakdown of one unit.

    Returns (path rows, intermediate rows, pair stats keyed
    ``(source, target)``). Rank is bottleneck-first within (pair, length);
    shared/unique classification is over the pair's FULL path set. Each
    path row carries its pair's ``source_cov``/``target_cov`` bodyId
    n/N strings (round-8b item 1).
    """
    source_cov = source_cov or {}
    target_cov = target_cov or {}
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
                probabilities = row.get('probabilities', '')
                ratios = row.get('ratios', '')
                pair_rows.append({
                    'source': source,
                    'target': target,
                    'pair': f'{source}->{target}',
                    'rank_in_pair_length': rank,
                    'path': str(row['path']),
                    'weights': '' if pd.isna(row['weights']) else str(row['weights']),
                    'probabilities': '' if pd.isna(probabilities) else str(probabilities),
                    'ratios': '' if pd.isna(ratios) else str(ratios),
                    'min_weight': row['min_weight'],
                    'path_prob': row['path_prob'],
                    'length': int(length),
                    'source_bodyid_coverage': source_cov.get(source, ''),
                    'target_bodyid_coverage': target_cov.get(target, ''),
                })
        stats = {
            'paths': len(pair_rows),
            'lengths': dict(sorted(lengths.items())),
            'shared': sum(1 for v in inter_counts[(source, target)].values() if v >= 2),
            'unique': sum(1 for v in inter_counts[(source, target)].values() if v == 1),
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


def _root_display_label(run_dir: Path, meta: Dict[str, Any],
                        fallback: str) -> str:
    """Informative label for a report's own run-root unit — '(run root)'
    is retired (round-8 item 5). Single-dataset runs label by dataset +
    min synapse; nested delegate reports fall back to their folder name
    (e.g. ``minsyn_3``) or the dataset alone; the run name is the last
    resort."""
    dataset = str(meta.get('dataset') or '').strip()
    min_syn = str(meta.get('min_synapse') or '').strip()
    if dataset and min_syn and min_syn not in ('0', 'None'):
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
        for index, (a, b) in enumerate(zip(ids, ids[1:])):
            edge_count[(a, b)] += 1
            if hop_weights is not None and index < len(hop_weights):
                edge_weight[(a, b)] += hop_weights[index]

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
        edges.append({
            'f': a, 't': b, 'c': count,
            'w': round(weight, 1) if weight else 0.0,
        })

    return {
        'ncols': ncols,
        'nrows': max(col_seen.values()) if col_seen else 1,
        'nodes': nodes,
        'edges': edges,
        'paths': paths,
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
        'drawn_shared': drawn_shared,
        'drawn_unique': drawn_unique,
        'viz': build_viz(drawn_rows, inter_counts, inter_minhop),
    }


def build_global(
    units: Sequence[Unit], frames: Dict[str, pd.DataFrame],
    global_pairs: int, global_edges: int,
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
        })

    pair_cells: Dict[Tuple[str, str], Dict[str, int]] = {}
    pair_len_range: Dict[Tuple[str, str], List[int]] = {}
    pair_unit_range: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
    for unit in units:
        frame = frames[unit.unit_id]
        for (source, target), count in (
                frame.groupby(['_source', '_target']).size().items()):
            pair_cells.setdefault((source, target), {})[unit.unit_id] = int(count)
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
         'ranges': pair_unit_range.get((s, t), {})}
        for (s, t), cells in pair_cells.items()
    ]
    matrix_rows.sort(key=lambda r: (-r['cons'], -r['total'], r['source'], r['target']))
    cons_hist = Counter(r['cons'] for r in matrix_rows)
    pair_matrix = {
        'units': [u.unit_id for u in units],
        'total_pairs': len(matrix_rows),
        'shown': min(len(matrix_rows), global_pairs),
        'rows': matrix_rows[:global_pairs],
    }

    edge_count: Counter = Counter()
    min_hop_inter: Dict[str, int] = {}
    ever_source: Dict[str, bool] = {}
    ever_target: Dict[str, bool] = {}
    node_paths: Counter = Counter()
    node_pairs: Dict[str, set] = defaultdict(set)
    ncols = 2
    for unit in units:
        frame = frames[unit.unit_id]
        for row in frame.itertuples(index=False):
            ids = str(row.path).split('->')
            ncols = max(ncols, len(ids))
            hops = len(ids) - 1
            source, target = ids[0], ids[-1]
            for hop, node in enumerate(ids):
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
            for a, b in zip(ids, ids[1:]):
                edge_count[(a, b)] += 1

    def node_col(node: str) -> int:
        if node in min_hop_inter:
            return min(min_hop_inter[node], ncols - 2)
        if node in ever_target and node not in ever_source:
            return ncols - 1
        return 0

    top_edges = sorted(edge_count.items(), key=lambda kv: (-kv[1], kv[0]))
    top_edges = top_edges[:global_edges]
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
            'edges': [{'f': a, 't': b, 'c': count} for (a, b), count in top_edges],
        },
        'edge_stats': {'total': len(edge_count), 'shown': len(top_edges)},
        'node_stats': {'total': len(node_paths), 'shown': len(nodes_set)},
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
            'shortest': (kind == 'Shortest Paths'
                         or str(meta.get('path_mode', '')).lower() == 'shortest'),
            'hemispheres': str(meta.get('separate_hemispheres', '')).lower()
            in ('true', '1', 'yes'),
            'meta': {k: v for k, v in meta.items() if v is not None},
            'top_per_length': top_per_length,
            'rank_by': rank_by,
            'vispath': vispath_links or [],
            'units': [
                {'id': u.unit_id, 'label': u.label, 'dataset': u.dataset,
                 'threshold': u.threshold}
                for u in units
            ],
        },
        'histogram': dict(sorted(length_counts.items())),
        'pairs': pairs,
    }


# ---------------------------------------------------------------------------
# breakdown CSV writers
# ---------------------------------------------------------------------------

PATHS_COLUMNS = [
    'dataset', 'threshold', 'unit', 'source', 'target', 'pair',
    'rank_in_pair_length', 'path', 'weights', 'probabilities', 'ratios',
    'min_weight', 'path_prob', 'length', 'source_bodyid_coverage',
    'target_bodyid_coverage', 'paths_in_pair',
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


def _render_overview(payload: dict, run_dir: Path,
                     breakdown_paths: Tuple[Path, Path]) -> str:
    run = payload['run']
    pairs = payload['pairs']
    total_paths = sum(p['paths_total'] for p in pairs)
    largest = max(pairs, key=lambda p: (p['paths_total'], p['source'], p['target'])) \
        if pairs else None

    stats_html = f"""
        <div class="grid">
            <div class="stat-box"><div class="value">{len(run['units'])}</div>
                <div class="label">Units</div></div>
            <div class="stat-box"><div class="value">{len(pairs)}</div>
                <div class="label">Source–target pairs</div></div>
            <div class="stat-box"><div class="value">{total_paths:,}</div>
                <div class="label">Paths (all units)</div></div>
            <div class="stat-box"><div class="value">{largest['paths_total']:,}</div>
                <div class="label">Largest pair — {_esc(largest['source'])} → {_esc(largest['target'])}</div></div>
        </div>""" if pairs else '<p>No paths found in this run.</p>'

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
        rows.append(
            f'<tr data-pair="{_esc(pair["id"])}">'
            f'<td>{_esc(unit_labels.get(pair["unit"], pair["unit"]))}</td>'
            f'<td><strong>{_esc(pair["source"])}</strong></td>'
            f'<td><strong>{_esc(pair["target"])}</strong></td>'
            f'<td>{pair["paths_total"]:,}</td>'
            f'<td>{pair["n_lengths"]} ({_esc(pair["length_range"])})</td>'
            f'<td>{pair["shared"]:,} shared · {pair["unique"]:,} unique</td>'
            f'<td><a href="#" onclick="TAB.gotoPair(\'{_esc(pair["id"])}\');'
            f'return false">Open</a></td></tr>')
    pair_table = ''
    if rows:
        pair_table = (
            '<div class="card"><h3>Pairs</h3>'
            '<p style="color:var(--secondary-color);font-size:0.9em;">Sorted by '
            'path count. “Open” jumps to the Pair Explorer with the source and '
            'target selects set.</p>'
            '<div class="sticky-table-container"><table><thead><tr>'
            '<th>Unit</th><th>Source</th><th>Target</th><th>Paths</th>'
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
            f'<div class="card"><h3>Summary</h3>{stats_html}</div>'
            + hist_html + pair_table + artifacts)


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
    for unit in run['units']:
        csv_path = next(iter(sorted(unit_csv_candidates(run_dir, unit))), None)
        if csv_path is None:
            continue
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
        link = _link_if_exists(run_dir / name, run_dir, name)
        if link != '-':
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

    return table + breakdown_card + paths_card + artifacts_card + vispath_card


def _render_global_tab(payload: dict) -> str:
    """Run-wide presentations, cross-dataset-analysis style: per-unit
    stats, the pair x unit path-count matrix, the pairs-by-unit-coverage
    histogram, and the top-edges global route network (SVG rendered by JS
    from the payload)."""
    glob = payload['global']
    run = payload['run']
    unit_labels = {u['id']: u['label'] for u in run['units']}

    unit_rows = []
    for unit in glob['units']:
        unit_rows.append(
            f'<tr><td>{_esc(unit["label"])}</td>'
            f'<td>{unit["paths"]:,}</td><td>{unit["pairs"]}</td>'
            f'<td>{unit["sources"]}</td><td>{unit["targets"]}</td>'
            f'<td>{_fmt_num(unit["median_weight"])}</td>'
            f'<td>{_fmt_num(unit["p90_weight"])}</td>'
            f'<td>{_fmt_num(unit["max_weight"])}</td></tr>')
    units_card = (
        '<div class="card"><h3>Units</h3>'
        '<div class="sticky-table-container"><table><thead><tr>'
        '<th>Unit</th><th>Paths</th><th>Pairs</th><th>Sources</th>'
        '<th>Targets</th><th>Median min-weight</th><th>P90</th><th>Max</th>'
        '</tr></thead><tbody>' + ''.join(unit_rows) +
        '</tbody></table></div>'
        '<p class="cap-note">Weights are per-path bottlenecks (min_weight); '
        'P90 is the 90th percentile. Min synapse thresholds make these '
        'non-comparable across units — compare shapes, not levels.</p></div>')

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
        badge_cls = ('badge-success' if row['cons'] == n_units
                     else 'badge-warning' if row['cons'] > 1 else 'badge-danger')
        matrix_rows.append(
            f'<tr><td class="matrix-first-col"><strong>{_esc(row["source"])}</strong> → '
            f'<strong>{_esc(row["target"])}</strong></td>'
            + ''.join(cells)
            + f'<td><span class="badge {badge_cls}">{row["cons"]}/{n_units}</span></td>'
            f'<td>{_esc(row.get("length_range", "—"))}</td>'
            f'<td>{row["total"]:,}</td></tr>')
    matrix_card = (
        '<div class="card"><h3>Pair × unit path counts</h3>'
        '<p style="color:var(--secondary-color);font-size:0.9em;">The global '
        'analog of the cross-dataset path presence matrix: pairs (matched by '
        'name across units) as rows, units as columns, cells = path counts '
        '(hover for the hop range within that unit). Sorted by unit '
        'coverage, then total paths.</p>'
        f'<div class="{container_cls}"><table><thead><tr>'
        '<th class="matrix-first-col">Pair</th>'
        + ''.join(f'<th>{_esc(unit_labels.get(u, u))}</th>' for u in matrix['units'])
        + '<th>Coverage</th><th>Lengths</th><th>Total</th></tr></thead><tbody>'
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
        f'<p class="cap-note">Showing the top {stats["shown"]} of '
        f'{stats["total"]:,} distinct edges by traversal count '
        f'({nstat["shown"]} of {nstat["total"]:,} nodes). Edge width ∝ '
        f'traversals; hover a node for its path/pair counts. The full edge '
        f'set is derivable from pair_breakdown_paths.csv.</p></div>')

    return units_card + matrix_card + hist_card + network_card


def unit_csv_candidates(run_dir: Path, unit: dict) -> List[Path]:
    unit_id = unit['id']
    if not unit_id:
        return sorted(run_dir.glob(f'*{PATHS_SUFFIX}'))
    folder = run_dir / 'dataset_data' / unit_id
    if not folder.exists():
        folder = run_dir / unit_id
    return sorted(folder.glob(f'*{PATHS_SUFFIX}'))


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
                       run: dict, generated_at: str) -> str:
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
  <p style="opacity:0.8;">Generated: __GENERATED__ · capped views; full data in
  paths_pair_breakdown/ beside this file</p>
</header>
<div class="page-tab-bar" role="tablist">
  <button class="page-tab-btn" data-page="overview" role="tab"
    onclick="TAB.show('overview')">Overview</button>
  <button class="page-tab-btn" data-page="global" role="tab"
    onclick="TAB.show('global')">Global</button>
  <button class="page-tab-btn" data-page="explorer" role="tab"
    onclick="TAB.show('explorer')">Pair Explorer</button>
  <button class="page-tab-btn" data-page="data" role="tab"
    onclick="TAB.show('data')">Data</button>
</div>
<div id="page-content"></div>
<template id="tpl-overview">__OVERVIEW__</template>
<template id="tpl-global">__GLOBAL__</template>
<template id="tpl-explorer">__EXPLORER__</template>
<template id="tpl-data">__DATA__</template>
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
           .replace('__GENERATED__', _esc(generated_at)) \
           .replace('__OVERVIEW__', overview_html) \
           .replace('__GLOBAL__', global_html) \
           .replace('__EXPLORER__', explorer_html) \
           .replace('__DATA__', data_html) \
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
        title: e.c + ' paths' + (e.w ? ' · Σw = ' + fmt(e.w) : ''),
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

  // ---- page router (house report_tabbed pattern) ----
  var TAB = {
    show: function(page) {
      var tpl = document.getElementById('tpl-' + page);
      var container = document.getElementById('page-content');
      if (!tpl || !container) { return; }
      container.innerHTML = '';
      container.appendChild(tpl.content.cloneNode(true));
      document.querySelectorAll('.page-tab-btn').forEach(function(b) {
        b.classList.toggle('active', b.dataset.page === page); });
      if (page === 'explorer') { Explorer.mount(); }
      if (page === 'global') { GlobalNet.render(); }
      try { history.replaceState(null, '', '#tab=' + page); } catch (e) {}
    },
    gotoPair: function(pairId) {
      this.show('explorer');
      try { history.replaceState(null, '', '#tab=explorer&pair=' + pairId); } catch (e) {}
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
      title.textContent = e.c + ' paths'
        + (e.w ? ' · Σw = ' + fmt(e.w) : '');
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
        var rec = edgeAgg[key] = edgeAgg[key] || { f: e.f, t: e.t, c: 0, w: 0 };
        rec.c += e.c; rec.w += (e.w || 0);
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

  // ---- global network (Global tab) ----
  var GlobalNet = {
    render: function() {
      var holder = document.getElementById('global-network');
      if (!holder || holder.dataset.done || !DATA.global) { return; }
      holder.dataset.done = '1';
      var viz = DATA.global.network;
      var svgWrap = elt('div');
      svgWrap.appendChild(makeSvg(viz).svg);
      holder.appendChild(svgWrap);
      var netWrap = elt('div', 'network-container');
      holder.appendChild(netWrap);
      PENDING_NETS.push({ wrap: netWrap, svg: svgWrap, viz: viz, hier: true });
      setTimeout(flushNetworks, 0);
    }
  };

  // ---- pair explorer ----
  var Explorer = {
    // TAB.show re-clones the page template on EVERY visit, so mount() must
    // fully re-bind the fresh selects and re-render each time — an
    // early-return guard here left the pane empty whenever the user
    // switched away and back (fixed 2026-10-02).
    byUnit: null,
    mount: function() {
      this.byUnit = {};
      DATA.pairs.forEach(function(p) {
        (Explorer.byUnit[p.unit] = Explorer.byUnit[p.unit] || []).push(p); });
      this.unitSel = document.getElementById('sel-unit');
      this.sourceSel = document.getElementById('sel-source');
      this.targetSel = document.getElementById('sel-target');
      var units = DATA.run.units.map(function(u) { return u.id; });
      var self = this;
      if (this.unitSel) {
        units.forEach(function(uid) {
          var opt = document.createElement('option');
          opt.value = uid; opt.textContent = UNIT_LABEL[uid];
          self.unitSel.appendChild(opt);
        });
        this.unitSel.addEventListener('change', function() {
          self.fillSources(); });
      }
      this.sourceSel.addEventListener('change', function() { self.fillTargets(); });
      this.targetSel.addEventListener('change', function() { self.render(); });
      var biggest = DATA.pairs.reduce(function(a, b) {
        return (b.paths_total > a.paths_total) ? b : a; }, DATA.pairs[0]);
      if (biggest) {
        this.setPair(biggest.id);
      } else {
        this.fillSources();
      }
    },
    currentUnit: function() {
      return this.unitSel ? this.unitSel.value : DATA.run.units[0].id;
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
      var pair = DATA.pairs.filter(function(p) { return p.id === pairId; })[0];
      if (!pair) { return; }
      if (this.unitSel) { this.unitSel.value = pair.unit; }
      this.fillSources();
      this.sourceSel.value = pair.source;
      this.fillTargets();
      this.targetSel.value = pair.target;
      this.render();
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
      var pane = document.getElementById('pair-pane');
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
      if (DATA.run.shortest) {
        head.appendChild(elt('p', 'cap-note',
          'Shortest mode: only the minimum-hop paths per bodyId pair (ties '
          + 'kept); a type-level pair can still span several lengths, so '
          + 'the per-length top-' + DATA.run.top_per_length
          + ' cap applies normally.'));
      }
      pane.appendChild(head);

      // capped table + network (single pair: table/viz from its payload)
      pane.appendChild(this.renderCappedTable([p]));
      var rowPaths = [];
      p.table.groups.forEach(function(g) {
        g.rows.forEach(function(r) { rowPaths.push(r.path); });
      });
      pane.appendChild(this.renderVizCard(p.viz, rowPaths, {
        netId: p.id,
        caption: 'Drawn: the capped top-paths set (' + p.drawn_shared
          + ' shared / ' + p.drawn_unique + ' unique intermediates drawn). '
          + 'Classification from the full pair path set: ' + p.shared
          + ' shared / ' + p.unique + ' unique. Edge width scales within '
          + 'the drawn set.'}));

      // links
      var foot = elt('p', 'cap-note');
      foot.appendChild(document.createTextNode('Full list: '));
      var a = elt('a', null, 'pair_breakdown_paths.csv');
      a.href = 'paths_pair_breakdown/pair_breakdown_paths.csv';
      a.target = '_blank';
      foot.appendChild(a);
      foot.appendChild(document.createTextNode(' — filter '));
      foot.appendChild(elt('span', 'code', 'pair = ' + p.source + '->' + p.target));
      foot.appendChild(document.createTextNode(' and '));
      foot.appendChild(elt('span', 'code', 'unit = ' + (UNIT_LABEL[p.unit] || p.unit)));
      (DATA.run.vispath || []).forEach(function(item) {
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

      var merged = mergeViz(pairs, this.unionRowPaths(pairs));
      var rowPaths = this.unionRowPaths(pairs);
      pane.appendChild(this.renderCappedTable(pairs));
      pane.appendChild(this.renderVizCard(merged, rowPaths, {
        netId: 'union',
        caption: 'Drawn: the UNION of the selected pairs\u2019 capped paths ('
          + merged.drawn_shared + ' shared / ' + merged.drawn_unique
          + ' unique intermediates drawn; shared = on \u22652 drawn paths or '
          + 'shared within any selected pair). Edge counts and widths are '
          + 'merged across pairs.'}));

      var foot = elt('p', 'cap-note');
      foot.appendChild(document.createTextNode('Full list: '));
      var a = elt('a', null, 'pair_breakdown_paths.csv');
      a.href = 'paths_pair_breakdown/pair_breakdown_paths.csv';
      a.target = '_blank';
      foot.appendChild(a);
      foot.appendChild(document.createTextNode(' — filter '));
      var filterText = 'pair in (' + names.slice(0, 6).join(', ')
        + (names.length > 6 ? ', …' : '') + ')';
      foot.appendChild(elt('span', 'code',
        filterText.replace(/→/g, '->')));
      foot.appendChild(document.createTextNode(' and '));
      foot.appendChild(elt('span', 'code',
        'unit = ' + (UNIT_LABEL[pairs[0].unit] || pairs[0].unit)));
      pane.appendChild(foot);
      return pane;
    },

    // The union table's row order: grouped by length ascending, then
    // bottleneck (min_weight) desc across the whole selection.
    unionRowPaths: function(pairs) {
      var all = [];
      pairs.forEach(function(p) {
        p.table.groups.forEach(function(g) {
          g.rows.forEach(function(r) { all.push(r); });
        });
      });
      all.sort(function(a, b) {
        return a.len - b.len || b.mw - a.mw || (a.path < b.path ? -1 : 1);
      });
      return all.map(function(r) { return r.path; });
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
      pairs.forEach(function(p) {
        p.table.groups.forEach(function(g) {
          var bucket = groupsByLen[g.len] = groupsByLen[g.len] || [];
          g.rows.forEach(function(r) {
            bucket.push({pair: p.source + '→' + p.target, rank: r.rank,
              path: r.path, len: g.len, mw: r.mw, pp: r.pp,
              weights: r.weights, scov: r.scov, tcov: r.tcov,
              total: g.total});
          });
        });
      });
      var lengths = Object.keys(groupsByLen).map(Number).sort(function(a, b) {
        return a - b; });
      var wrap = elt('div', 'sticky-table-container');
      var table = elt('table');
      var thead = elt('thead');
      var hr = elt('tr');
      ['#', 'Pair', 'Path', 'Len', 'Min weight', 'Path prob', 'Weights',
       'Source bIds', 'Target bIds'].forEach(function(t) {
        hr.appendChild(elt('th', null, t)); });
      thead.appendChild(hr); table.appendChild(thead);
      var tbody = elt('tbody');
      var rowPaths = [];
      lengths.forEach(function(len) {
        var rows = groupsByLen[len].slice().sort(function(a, b) {
          return (b.mw || 0) - (a.mw || 0) || (a.path < b.path ? -1 : 1); });
        var totals = {};
        rows.forEach(function(r) {
          totals[r.pair] = Math.max(totals[r.pair] || 0, r.total); });
        var gr = elt('tr', 'group-row');
        var gtd = elt('td', null, len + ' hops — ' + rows.length
          + ' shown (' + (pairs.length > 1 ? 'union, top '
            + DATA.run.top_per_length + ' per pair' : 'top ' + rows.length)
          + ' of ' + rows.reduce(function(acc, r) {
            return acc + r.total; }, 0) + ')');
        gtd.colSpan = 9;
        gr.appendChild(gtd);
        tbody.appendChild(gr);
        rows.forEach(function(row, i) {
          var tr = elt('tr');
          tr.dataset.i = rowPaths.length;
          tr.appendChild(elt('td', null, i + 1));
          if (pairs.length > 1) { tr.appendChild(elt('td', null, row.pair)); }
          var pathTd = elt('td');
          pathTd.appendChild(elt('strong', null, row.path));
          tr.appendChild(pathTd);
          tr.appendChild(elt('td', null, row.len));
          tr.appendChild(elt('td', null, fmt(row.mw)));
          tr.appendChild(elt('td', null, fmt(row.pp)));
          tr.appendChild(elt('td', null, row.weights));
          var scovTd = elt('td', null, row.scov || '—');
          scovTd.title = 'Source bodyIds on paths (isInPath) / enrolled — '
            + 'from source_neurons.csv';
          tr.appendChild(scovTd);
          var tcovTd = elt('td', null, row.tcov || '—');
          tcovTd.title = 'Target bodyIds reached (Checked) / resolved — '
            + 'from target_neurons.csv';
          tr.appendChild(tcovTd);
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
        + ' paths (top-' + DATA.run.top_per_length + ' per pair and length, '
        + 'rank by ' + DATA.run.rank_by
        + (pairs.length > 1 ? ', union re-ranked by min weight within each length' : '')
        + '). Hover a row to highlight its route in the network below.';
      card.appendChild(note);
      card.dataset.rowPaths = JSON.stringify(rowPaths);
      return card;
    },

    renderVizCard: function(viz, rowPaths, opts) {
      var card = elt('div', 'card');
      card.appendChild(elt('h3', null, 'Network'));
      var legend = elt('div', 'legend');
      var items = [['#ef4444', 'source'], ['#8b5cf6', 'target']];
      var hasBoth = viz.nodes.some(function(n) { return n.cls === 'both'; });
      if (hasBoth) { items.push(['#14b8a6', 'source & target']); }
      items.push(['#2563eb', 'shared intermediate (≥2 paths)'],
        ['#94a3b8', 'unique intermediate (1 path)']);
      items.forEach(function(item) {
        var span = elt('span');
        var chip = elt('span', 'chip');
        chip.style.background = item[0];
        span.appendChild(chip);
        span.appendChild(document.createTextNode(item[1]));
        legend.appendChild(span);
      });
      card.appendChild(legend);

      var built = makeSvg(viz);
      var svg = built.svg;
      var edgeEls = built.edgeEls;
      var svgWrap = elt('div');
      svgWrap.appendChild(svg);
      card.appendChild(svgWrap);
      var netWrap = elt('div', 'network-container');
      netWrap.id = 'net-' + opts.netId;
      card.appendChild(netWrap);
      PENDING_NETS.push({ wrap: netWrap, svg: svgWrap, viz: viz,
        hier: viz === (DATA.global && DATA.global.network) });
      var caption = elt('p', 'cap-note');
      caption.textContent = opts.caption;
      card.appendChild(caption);

      // row-hover → highlight that path's edges (keys from the row path)
      setTimeout(function() {
        var pane = document.getElementById('pair-pane');
        if (!pane) { return; }
        pane.querySelectorAll('tr[data-i]').forEach(function(tr) {
          var ids = (rowPaths[Number(tr.dataset.i)] || '').split('->');
          if (ids.length < 2) { return; }
          var keys = [];
          for (var k = 0; k + 1 < ids.length; k++) {
            keys.push(ids[k] + '->' + ids[k + 1]);
          }
          tr.addEventListener('mouseenter', function() {
            keys.forEach(function(key) {
              var el = edgeEls[key];
              if (el) { el.classList.add('hl'); }
            });
          });
          tr.addEventListener('mouseleave', function() {
            keys.forEach(function(key) {
              var el = edgeEls[key];
              if (el) { el.classList.remove('hl'); }
            });
          });
        });
      }, 0);

      return card;
    }
  };

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
    if (DATA.pairs.length === 0) { page = 'overview'; }
    TAB.show(page === 'explorer' ? 'explorer' : page);
    if (page === 'explorer' && params.pair) {
      Explorer.selectPair(params.pair);
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
                    'min_weight': row['min_weight'],
                    'path_prob': row['path_prob'],
                    'rank': 0,
                })
        drawn[(source, target)] = rows
    return drawn


def _attach_table_groups(
    frame: pd.DataFrame, rank_by: str, top_per_length: int,
    pairs: List[dict],
    source_cov: Optional[Dict[str, str]] = None,
    target_cov: Optional[Dict[str, str]] = None,
) -> None:
    """Fill each pair entry's capped table groups (ranked rows per length),
    each row carrying its pair's ``scov``/``tcov`` bodyId n/N strings
    (round-8b item 1)."""
    source_cov = source_cov or {}
    target_cov = target_cov or {}
    work = frame.copy()
    work['_path_key'] = work['path']
    ordered = _rank_sort(work, rank_by)
    by_pair: Dict[Tuple[str, str], dict] = {}
    for (source, target), group in ordered.groupby(['_source', '_target'], sort=True):
        groups = []
        for length, length_group in group.groupby('_length', sort=True):
            take = length_group.head(top_per_length)
            rows = []
            for rank, (_, row) in enumerate(take.iterrows(), start=1):
                rows.append({
                    'rank': rank,
                    'path': str(row['path']),
                    'len': int(length),
                    'mw': None if pd.isna(row['min_weight']) else float(row['min_weight']),
                    'pp': None if pd.isna(row['path_prob']) else float(row['path_prob']),
                    'weights': '' if pd.isna(row['weights']) else str(row['weights']),
                    'scov': source_cov.get(source, ''),
                    'tcov': target_cov.get(target, ''),
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


def generate_paths_pair_report(
    run_dir, top_per_length: int = DEFAULT_TOP_PER_LENGTH,
    rank_by: str = 'min_weight', log=None,
    global_pairs: int = DEFAULT_GLOBAL_PAIRS,
    global_edges: int = DEFAULT_GLOBAL_EDGES,
) -> Path:
    """Generate the pair report + breakdown CSVs for one run folder.

    Writes ONLY ``path_report.html`` and ``paths_pair_breakdown/`` into
    the run folder; every other file is left untouched. Returns the
    report path.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(f'run folder not found: {run_dir}')
    say = log or (lambda *a, **k: None)

    meta = _sniff_metadata(run_dir)
    units = discover_units(run_dir)
    if not units:
        raise FileNotFoundError(
            f'no *_allpaths_type.csv paths table found under {run_dir}')
    # "(run root)" is retired: the report's own root unit gets an
    # informative dataset/threshold label (round-8 item 5).
    for unit in units:
        if not unit.unit_id:
            unit.label = _root_display_label(run_dir, meta, run_kind(run_dir.name))
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

    for unit in units:
        frame = load_unit_paths(unit.csv_path)
        frames[unit.unit_id] = frame
        source_cov, target_cov = _load_enrollment_coverage(unit.folder)
        unit_source_cov[unit.unit_id] = source_cov
        unit_target_cov[unit.unit_id] = target_cov
        path_rows, inter_rows, pair_stats = build_unit_breakdown(
            frame, rank_by, source_cov=source_cov, target_cov=target_cov)
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
        run_dir.name, run_kind(run_dir.name), meta,
        units, frames, unit_inter_counts, unit_inter_minhop,
        unit_pair_stats, unit_drawn, top_per_length, rank_by,
        vispath_links=vispath_links)
    _attach_table_groups_multi(units, frames, payload, rank_by, top_per_length,
                               unit_source_cov, unit_target_cov)
    payload['global'] = build_global(units, frames, global_pairs, global_edges)

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

    html_str = _render_page_shell(
        payload_json,
        _render_overview(payload, run_dir, breakdown_paths),
        _render_global_tab(payload),
        _render_data_tab(payload, run_dir),
        _render_explorer_template(payload['run']),
        payload['run'],
        _generated_stamp(),
    )
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
                target_cov=(unit_target_cov or {}).get(unit.unit_id))
