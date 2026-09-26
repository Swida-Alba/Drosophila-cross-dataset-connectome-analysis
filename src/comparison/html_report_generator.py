"""
HTML Report Generator for Cross-Dataset Comparison.

Generates a static HTML report organized by content type (not by threshold).
Structure:
- Overview & Key Findings
- Edge Count Charts (all thresholds)
- Similarity Matrices (all thresholds)  
- Networks (side-by-side per threshold)
- Edge Presence Matrices (per threshold)
- Path Presence Matrices (per threshold)
- Conservation Analysis (per threshold)
- Statistics Tables (per threshold)
"""

import json
import os
import html
import html as html_module
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple

try:
    from utils.threshold_state import applied_threshold_provenance
except ImportError:  # pragma: no cover - direct package imports
    from src.utils.threshold_state import applied_threshold_provenance

try:
    from utils.naming_utils import split_hemi_suffix
except ImportError:  # pragma: no cover - direct package imports
    from src.utils.naming_utils import split_hemi_suffix

try:
    from .type_coverage import absence_note as _coverage_absence_note
    from .type_coverage import node_coverage_line as _coverage_node_line
except ImportError:  # pragma: no cover - direct package imports
    try:
        from src.comparison.type_coverage import (
            absence_note as _coverage_absence_note,
            node_coverage_line as _coverage_node_line,
        )
    except ImportError:  # the card/hover annotations degrade to plain text
        _coverage_absence_note = None
        _coverage_node_line = None


def _make_link(path: str, base_dir: str) -> str:
    """Render an HTML link to a result file relative to the run's output folder."""
    if not path or not os.path.exists(path):
        return '-'
    rel_path = os.path.relpath(path, base_dir).replace(os.sep, '/')
    return f'<a href="{rel_path}" target="_blank">Open</a>'


def get_canonical_name(display_name: str) -> str:
    """Strip a merged-display suffix ('MBON14 (merged)') from a label."""
    if '(' in display_name:
        return display_name.split('(')[0].strip()
    return display_name


def get_base_name(name: str) -> str:
    """Strip a hemisphere suffix ('_L'/'_R'/'_U') from a label."""
    if name.endswith('_L') or name.endswith('_R') or name.endswith('_U'):
        return name[:-2]
    return name


def matches_patterns(label: str, patterns: set) -> bool:
    """Match a label against source/target patterns (regex, glob, or exact)."""
    import re
    # Get canonical name for matching (handles merged display names)
    canonical = get_canonical_name(label)

    # Get base name without hemisphere suffix (for separate_hemispheres mode)
    base_name = get_base_name(canonical)

    # Check label, canonical name, and base name (without suffix)
    names_to_check = list(set([label, canonical, base_name]))

    for name in names_to_check:
        for pattern in patterns:
            # Handle regex patterns (containing .* or other regex chars)
            # vs simple glob patterns (containing just *)
            if '.*' in pattern:
                # Already a regex pattern (e.g., "aMe.*"), use directly
                regex_pattern = pattern
            elif '*' in pattern:
                # Simple glob pattern (e.g., "aMe*"), convert * to .*
                # Escape special regex chars except *
                regex_pattern = re.escape(pattern).replace(r'\*', '.*')
            else:
                # Exact match pattern, escape for regex
                regex_pattern = re.escape(pattern)

            if re.match(f'^{regex_pattern}$', name, re.IGNORECASE):
                return True
    return False


def generate_html_report(
    analyzer,
    dataset_names: List[str],
    thresholds: List[int],
    mode_specific_note: str,
    path_count_data: List[Dict],
    key_findings_per_threshold: Dict[int, Dict],
    comparison_points=None,
) -> str:
    """
    Generate a static HTML report organized by content type.
    """
    if comparison_points:
        return _generate_query_html_report(
            analyzer=analyzer,
            dataset_names=dataset_names,
            comparison_points=comparison_points,
            mode_specific_note=mode_specific_note,
        )
    html_parts = []
    
    # Get nicknames for display (shorter names)
    nicknames = analyzer.parameters.get_dataset_nicknames()
    nickname_map = {d: nicknames[i] for i, d in enumerate(dataset_names)}
    
    # HTML Header with CSS
    html_parts.append(_generate_html_header())
    
    # Report header
    html_parts.append(_generate_report_header(analyzer, dataset_names, thresholds, mode_specific_note, nickname_map))

    # Concern 2 (report-fixes): in-page banner — applied minimal
    # thresholds per dataset (Feature G τ collapse + lossy floor marker)
    html_parts.append(_generate_applied_threshold_banner(analyzer, dataset_names))

    # Comparability callout (plan §5A): warn when budget pruning leaves too
    # few like-for-like thresholds.
    html_parts.append(_generate_dataset_coverage_callout(
        analyzer, dataset_names, nickname_map))
    html_parts.append(_generate_comparability_callout(analyzer, nickname_map))

    # Requested -> applied mapping note (issue #1/#6): sections label points
    # by requested threshold; this shows what each actually compared.
    html_parts.append(_generate_applied_threshold_map_note(
        analyzer, dataset_names, nickname_map))

    # Table of Contents
    html_parts.append(_generate_toc(
        thresholds,
        include_type_mapping=bool(getattr(
            analyzer.parameters, 'auto_type_mapping', False)),
        include_alignment=bool(
            getattr(analyzer, '_alignment_best_df', None) is not None
            or getattr(analyzer, '_alignment_density_df', None) is not None),
        include_auto_density=bool(
            getattr(analyzer, '_density_curves_df', None) is not None
            or getattr(analyzer, '_density_windows_df', None) is not None)))
    
    # 1. Summary Section
    html_parts.append(_generate_summary_section(
        analyzer, dataset_names, thresholds, path_count_data, 
        key_findings_per_threshold, nickname_map
    ))
    
    # 1.5. Neuron Counts Section
    html_parts.append(_generate_neuron_counts_section(analyzer, dataset_names, nickname_map))

    # 1.55. Query Resolution Section (input token -> dataset)
    html_parts.append(_generate_query_resolution_section(
        analyzer, dataset_names, nickname_map))

    # 1.6. Type Mapping Section (§4 of the report-fixes plan)
    html_parts.append(_generate_type_mapping_section(analyzer, dataset_names))

    # 1.75. Hemisphere Symmetry Section
    html_parts.append(_generate_hemisphere_symmetry_section(analyzer, dataset_names, thresholds, nickname_map))

    # 1.8. Threshold Alignment Section (density-equivalent thresholds)
    html_parts.append(_generate_threshold_alignment_section(
        analyzer, dataset_names, nickname_map))
    
    # 2. Similarity Matrices Section
    html_parts.append(_generate_similarity_section(analyzer, dataset_names, thresholds, nickname_map))
    
    # 3. Networks Section
    html_parts.append(_generate_networks_section(analyzer, dataset_names, thresholds, nickname_map))

    # 3.5. Reciprocal Visualizations Section (REMOVED - per-dataset reciprocal links are now in Networks section)
    # html_parts.append(_generate_reciprocal_visualizations_section(analyzer, dataset_names, thresholds, nickname_map))
    
    # 4. Edge Presence Matrices Section
    html_parts.append(_generate_edge_matrices_section(analyzer, dataset_names, thresholds, nickname_map))
    
    # 5. Path Presence Matrices Section
    html_parts.append(_generate_path_matrices_section(analyzer, dataset_names, thresholds, nickname_map))
    
    # 6. Conservation Analysis Section
    html_parts.append(_generate_conservation_section(analyzer, dataset_names, thresholds, 
                                                      key_findings_per_threshold, nickname_map))
    
    # 6.5. Dataset Overlap Matrices Section
    html_parts.append(_generate_overlap_matrices_section(analyzer, dataset_names, thresholds, nickname_map))
    
    # 7. Statistics Section
    html_parts.append(_generate_statistics_section(analyzer, dataset_names, thresholds, nickname_map))
    
    # Footer
    html_parts.append(_generate_footer())
    
    return ''.join(html_parts)


def _query_report_slug(value) -> str:
    """Return the same safe query slug used by comparison CSV exports."""
    import re
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value or 'query'))
    return slug.strip(' ._-') or 'query'


def _query_report_value(value) -> str:
    """Format a scalar for the query report without exposing NaN text."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '—'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        return f'{value:g}'
    return str(value)


def _point_heading(query_id: str, query_label: str) -> str:
    """Deduplicated "id — label" heading for one comparison point.

    B7 horizontal labels embed the id as their prefix (``aligned_density=0.5
    (MCNS 10, ...)``), so a naive ``f"{id} — {label}"`` heading prints the
    same level twice. When the label already starts with the id, the label
    alone carries the full information."""
    qid = str(query_id)
    label = str(query_label)
    if label == qid or label.startswith(qid + ' '):
        return label
    return f'{qid} — {label}'


def _query_report_cell(value) -> str:
    return html.escape(_query_report_value(value))


def _query_report_save_used_data(analyzer, dataframe: pd.DataFrame, filename: str):
    """Save a query-scoped source table used by the HTML report."""
    if dataframe is None:
        return
    output_root = getattr(analyzer.parameters, 'full_output_path', None)
    if not output_root:
        return
    used_dir = os.path.join(output_root, 'comparison_report_used_data')
    os.makedirs(used_dir, exist_ok=True)
    try:
        dataframe.to_csv(os.path.join(used_dir, filename), index=False)
    except Exception:
        # Report rendering should remain available when an optional table has
        # an unusual index or an object that pandas cannot serialize.
        pass


def _query_report_edge_table(aligned: pd.DataFrame, dataset_names, nickname_map,
                             caption: str) -> str:
    """Render one bounded edge table for a query point."""
    if aligned is None or aligned.empty:
        return f'<div class="card"><h3>{html.escape(caption)}</h3><p>No edge data available.</p></div>'
    available = [dataset for dataset in dataset_names if dataset in aligned.columns]
    if not available:
        return f'<div class="card"><h3>{html.escape(caption)}</h3><p>No dataset columns available.</p></div>'
    data = aligned.copy()
    data['_conservation'] = (data[available] > 0).sum(axis=1)
    data['_weight_total'] = data[available].sum(axis=1)
    data = data.sort_values(['_conservation', '_weight_total'], ascending=False).head(50)
    parts = [f'<div class="card"><h3>{html.escape(caption)}</h3>',
             '<div class="sticky-table-container"><table><thead><tr><th>Edge</th>']
    parts.extend(f'<th>{html.escape(nickname_map.get(ds, ds))}</th>' for ds in available)
    parts.append('<th>Conservation</th></tr></thead><tbody>')
    for key, row in data.iterrows():
        count = int(sum(row.get(ds, 0) > 0 for ds in available))
        badge = 'badge-success' if count == len(available) else (
            'badge-warning' if count > 1 else 'badge-danger')
        parts.append(f'<tr><td><strong>{html.escape(str(key))}</strong></td>')
        for ds in available:
            weight = row.get(ds, 0)
            cell = ('<span class="presence-check">✔️</span> '
                    + _query_report_cell(weight)) if weight > 0 else (
                        '<span class="presence-cross">❌</span>')
            parts.append(f'<td>{cell}</td>')
        parts.append(f'<td><span class="badge {badge}">{count}/{len(available)}</span></td></tr>')
    parts.append('</tbody></table></div></div>')
    return ''.join(parts)


def _query_report_path_table(path_data: pd.DataFrame, dataset_names, nickname_map,
                             caption: str) -> str:
    """Render one bounded path table for a query point.

    Keep the query report's path table aligned with the Standard report: the
    path key is followed by a ``Len`` column containing the number of edges
    (hops) in the path before the per-dataset weights.
    """
    if path_data is None or path_data.empty:
        return f'<div class="card"><h3>{html.escape(caption)}</h3><p>No path data available.</p></div>'
    available = [dataset for dataset in dataset_names if dataset in path_data.columns]
    if not available:
        return f'<div class="card"><h3>{html.escape(caption)}</h3><p>No dataset columns available.</p></div>'
    data = path_data.copy()
    data['_conservation'] = (data[available] > 0).sum(axis=1)
    data['_weight_total'] = data[available].sum(axis=1)
    data = data.sort_values(['_conservation', '_weight_total'], ascending=False).head(50)
    parts = [f'<div class="card"><h3>{html.escape(caption)}</h3>',
             '<div class="sticky-table-container"><table><thead><tr><th>Path</th><th>Len</th>']
    parts.extend(f'<th>{html.escape(nickname_map.get(ds, ds))}</th>' for ds in available)
    parts.append('<th>Conservation</th></tr></thead><tbody>')
    for key, row in data.iterrows():
        count = int(sum(row.get(ds, 0) > 0 for ds in available))
        badge = 'badge-success' if count == len(available) else (
            'badge-warning' if count > 1 else 'badge-danger')
        # Match the Standard report's definition: path length is the number
        # of edges, i.e. the number of arrows in the normalized key.
        path_length = _path_length_from_key(key)
        parts.append(
            f'<tr><td><strong>{html.escape(str(key))}</strong></td>'
            f'<td>{path_length}</td>')
        for ds in available:
            weight = row.get(ds, 0)
            cell = ('<span class="presence-check">✔️</span> '
                    + _query_report_cell(weight)) if weight > 0 else (
                        '<span class="presence-cross">❌</span>')
            parts.append(f'<td>{cell}</td>')
        parts.append(f'<td><span class="badge {badge}">{count}/{len(available)}</span></td></tr>')
    parts.append('</tbody></table></div></div>')
    return ''.join(parts)


def _query_report_overlap_table(matrix, labels, title: str) -> str:
    parts = [f'<div class="card"><h3>{html.escape(title)}</h3><table><thead><tr><th>From / in</th>']
    parts.extend(f'<th>{html.escape(str(label))}</th>' for label in labels)
    parts.append('</tr></thead><tbody>')
    for label, row in zip(labels, matrix):
        parts.append(f'<tr><th>{html.escape(str(label))}</th>')
        parts.extend(f'<td>{_query_report_cell(value)}</td>' for value in row)
        parts.append('</tr>')
    parts.append('</tbody></table></div>')
    return ''.join(parts)


def _query_report_network_card(aligned: pd.DataFrame, dataset_names,
                               nickname_map, query_id: str, query_label: str) -> str:
    """Render an interactive vis-network card for one query row."""
    safe = _query_report_slug(query_id)
    if aligned is None or aligned.empty:
        return f'<div class="card"><h3>{html.escape(query_label)}</h3><p>No network edges available.</p></div>'
    node_ids = {}
    nodes = []
    edges = []
    source_nodes = set()
    target_nodes = set()
    available = [ds for ds in dataset_names if ds in aligned.columns]

    def node_id(label):
        label = str(label)
        if label not in node_ids:
            node_ids[label] = len(node_ids) + 1
            nodes.append({'id': node_ids[label], 'label': label})
        return node_ids[label]

    for edge_index, (edge_key, row) in enumerate(aligned.head(1000).iterrows(), start=1):
        text = str(edge_key)
        if ' -> ' not in text:
            continue
        source, target = text.split(' -> ', 1)
        source_nodes.add(source)
        target_nodes.add(target)
        present = [ds for ds in available if row.get(ds, 0) > 0]
        if not present:
            continue
        source_id = node_id(source)
        target_id = node_id(target)
        count = len(present)
        color = '#22c55e' if count == len(available) else (
            '#f59e0b' if count > 1 else '#94a3b8')
        weight_text = ', '.join(
            f'{nickname_map.get(ds, ds)}={_query_report_value(row.get(ds, 0))}'
            for ds in present)
        edges.append({
            'id': edge_index,
            'from': source_id,
            'to': target_id,
            'label': weight_text,
            'title': weight_text,
            'color': color,
            'arrows': 'to',
        })
    for node in nodes:
        if node['label'] in source_nodes and node['label'] in target_nodes:
            node['color'] = '#14b8a6'
        elif node['label'] in source_nodes:
            node['color'] = '#ef4444'
        elif node['label'] in target_nodes:
            node['color'] = '#8b5cf6'
        else:
            node['color'] = '#3b82f6'
    node_json = json.dumps(nodes)
    edge_json = json.dumps(edges)
    return f'''
        <div class="card">
            <h3>{html.escape(query_label)}</h3>
            <p style="color:var(--secondary-color);">Green = conserved, amber = partial, gray = dataset-specific. The network is capped at 1,000 displayed edges; the CSV links below are lossless for the exported comparison table.</p>
            <div id="query_network_{safe}" style="height:520px;border:1px solid var(--border-color);border-radius:8px;"></div>
            <script>
                (function() {{
                    const nodes = new vis.DataSet({node_json});
                    const edges = new vis.DataSet({edge_json});
                    const container = document.getElementById('query_network_{safe}');
                    if (container && typeof vis !== 'undefined') {{
                        new vis.Network(container, {{nodes:nodes, edges:edges}}, {{
                            interaction: {{hover:true}},
                            physics: {{enabled:false}},
                            nodes: {{shape:'dot', size:18, font:{{size:12}}}},
                            edges: {{smooth:false, font:{{size:9, align:'middle'}}}}
                        }});
                    }}
                }})();
            </script>
        </div>'''


def _generate_query_html_report(analyzer, dataset_names, comparison_points,
                                mode_specific_note: str) -> str:
    """Generate the full report shell for row-wise threshold combinations.

    This section-for-section reuses the Standard report's generators so both
    modes share the same presentation and controls. Every section is keyed by
    ``query_id`` and every dataset cell is read from that query's threshold
    map. The raw threshold union is never treated as a comparison axis.
    """
    queries = analyzer.get_threshold_queries()
    query_by_id = {str(query.get('id') or query.get('query_id')): query
                   for query in queries}
    nicknames = analyzer.parameters.get_dataset_nicknames()
    nickname_map = {dataset: nicknames[index]
                    for index, dataset in enumerate(dataset_names)}
    output_root = getattr(analyzer.parameters, 'full_output_path', '')
    results_dir = os.path.join(output_root, 'comparison_results')
    used_dir = os.path.join(output_root, 'comparison_report_used_data')
    esc = html.escape

    point_rows = []
    provenance_rows = []
    similarity_rows = []
    aligned_by_id = {}
    path_by_id = {}

    # Auto-mode separation schema: vertical rows are the PER-THRESHOLD
    # analysis (one identical Min Synapse Count for every dataset — the
    # like-for-like spine) and horizontal rows are the COMBINATION analysis
    # (per-dataset density-equalizing thresholds — the density-matched
    # envelope). Verticals order first; every query carries its group so
    # each report section can render the two analyses separately.
    _auto = bool(getattr(analyzer.parameters, 'threshold_auto', False))

    def _row_mode_of(query):
        m = str(query.get('row_mode') or '').strip().lower()
        if m in ('vertical', 'horizontal'):
            return m
        qid = str(query.get('id') or '')
        # B7 ids first, then the legacy prefixes.
        if qid.startswith('threshold='):
            return 'vertical'
        if qid.startswith('aligned_density='):
            return 'horizontal'
        if qid.startswith('aligned_v'):
            return 'vertical'
        if qid.startswith('aligned_h'):
            return 'horizontal'
        return ''

    QUERY_GROUP_META = {
        'vertical': (
            'Same-threshold queries — per-threshold analysis',
            'Every dataset is read at one identical Min Synapse Count '
            '(the like-for-like spine).'),
        'horizontal': (
            'Density-matched queries — combination analysis',
            'Each dataset is read at its own density-equalizing threshold '
            '(the density-matched envelope; E(t)/N is matched, so '
            'thresholds differ by dataset).'),
    }
    query_groups = []
    if _auto:
        for _gmode in ('vertical', 'horizontal'):
            _grows = [q for q in queries if _row_mode_of(q) == _gmode]
            if _grows:
                _title, _desc = QUERY_GROUP_META[_gmode]
                query_groups.append((_gmode, _title, _desc, _grows))
        _grouped = {id(q) for _, _, _, g in query_groups for q in g}
        _rest = [q for q in queries if id(q) not in _grouped]
        if _rest:
            query_groups.append(('', '', '', _rest))
        queries = [q for _, _, _, g in query_groups for q in g]

    def _query_group_banner(gmode):
        if not gmode or gmode not in QUERY_GROUP_META:
            return ''
        title, desc = QUERY_GROUP_META[gmode]
        if gmode == 'vertical':
            style, bg, mark = '#1d4ed8', '#eff6ff', '🟦'
        else:
            style, bg, mark = '#b45309', '#fffbeb', '🟨'
        return (f'<div style="border:1px solid {style}; background:{bg}; '
                'padding:8px 14px; border-radius:8px; margin:6px 0 14px 0;">'
                f'<strong style="color:{style};">{mark} {html.escape(title)}'
                '</strong><br><span style="font-size:0.88em;">'
                f'{html.escape(desc)}</span></div>')

    def _tab_sep_factory():
        seen = set()
        def sep(row):
            g = row.get('row_mode') or ''
            if not g or g in seen:
                return ''
            seen.add(g)
            if g == 'vertical':
                return ('<span style="align-self:center;font-size:0.78rem;'
                        'font-weight:700;color:#1d4ed8;padding:0 6px;'
                        'white-space:nowrap;">🟦 Same-threshold</span>')
            return ('<span style="align-self:center;font-size:0.78rem;'
                    'font-weight:700;color:#b45309;padding:0 6px;'
                    'white-space:nowrap;">🟨 Density-matched</span>')
        return sep

    def _iter_grouped(rows):
        """Yield ``(group_banner_or_'', row)`` — a banner on the first row
        of each auto query group so sections render the two analyses
        (per-threshold / density-matched) as visually separated blocks."""
        seen = set()
        for row in rows:
            g = row.get('row_mode') or ''
            banner = ''
            if g and g not in seen:
                seen.add(g)
                banner = _query_group_banner(g)
            yield banner, row

    for query in queries:
        query_id = str(query.get('id') or query.get('query_id'))
        query_label = str(query.get('label') or query_id)
        row_mode = _row_mode_of(query)
        group_name = (QUERY_GROUP_META[row_mode][0] if row_mode else '')
        aligned = analyzer.get_aligned_data_for_query(query)
        path_data = analyzer._get_path_data_for_query(query)
        aligned_by_id[query_id] = aligned
        path_by_id[query_id] = path_data
        available = [ds for ds in dataset_names if ds in aligned.columns]
        total_edges = int(len(aligned))
        common_edges = int((aligned[available] > 0).all(axis=1).sum()) if available else 0
        path_available = [ds for ds in dataset_names if ds in path_data.columns]
        total_paths = int(len(path_data))
        common_paths = int((path_data[path_available] > 0).all(axis=1).sum()) if path_available else 0
        requested = dict(query.get('thresholds') or {})
        point_rows.append({
            'query_id': query_id,
            'query_label': query_label,
            'requested_thresholds': requested,
            'total_edges': total_edges,
            'common_edges': common_edges,
            'edge_rate': common_edges / total_edges if total_edges else 0,
            'total_paths': total_paths,
            'common_paths': common_paths,
            'path_rate': common_paths / total_paths if total_paths else 0,
            'row_mode': row_mode,
            'group_name': group_name,
        })
        for dataset in dataset_names:
            requested_threshold = int(requested[dataset])
            provenance = analyzer._path_provenance_row(dataset, requested_threshold)
            row = dict(provenance)
            row.update({
                'query_id': query_id,
                'query_label': query_label,
                'query_index': queries.index(query) + 1,
                'threshold_mode': 'combinations',
                '_group': group_name,
            })
            provenance_rows.append(row)

        similarity = (getattr(analyzer, '_similarity_cache', {}) or {}).get(
            query_id, pd.DataFrame())
        if similarity is None or similarity.empty:
            report_similarity = (analyzer.comparison_report or {}).get(
                'threshold_similarities', pd.DataFrame())
            if isinstance(report_similarity, pd.DataFrame) and 'query_id' in report_similarity.columns:
                similarity = report_similarity[report_similarity['query_id'].astype(str) == query_id]
        if isinstance(similarity, pd.DataFrame) and not similarity.empty:
            similarity = similarity.copy()
            similarity['query_id'] = query_id
            similarity['query_label'] = query_label
            similarity_rows.extend(similarity.to_dict('records'))

    provenance_df = pd.DataFrame(provenance_rows)
    similarity_df = pd.DataFrame(similarity_rows)
    _query_report_save_used_data(analyzer, provenance_df,
                                 'provenance_by_query.csv')
    _query_report_save_used_data(analyzer, similarity_df,
                                 'similarity_by_query.csv')
    try:
        manifest_path = os.path.join(results_dir, 'threshold_combinations.csv')
        manifest_df = pd.read_csv(manifest_path) if os.path.exists(manifest_path) else pd.DataFrame()
        _query_report_save_used_data(analyzer, manifest_df, 'threshold_combinations.csv')
    except Exception:
        pass

    parts = [_generate_html_header()]
    from datetime import datetime
    params = analyzer.parameters
    auto_mode = bool(getattr(params, 'threshold_auto', False))
    mode_label = ('auto (density-aligned)' if auto_mode
                  else 'custom combinations')
    dataset_display = ', '.join(esc(nickname_map.get(ds, ds)) for ds in dataset_names)
    hemi_badge = ''
    if getattr(params, 'separate_hemispheres', False):
        hemi_badge = ('<p style="margin:6px 0 0 0;"><span style="background:#ede9fe; color:#6d28d9; '
                      'border:1px solid #7c3aed; border-radius:10px; padding:2px 10px; font-size:0.9em;">'
                      '🧠 Hemisphere-aware run — type names carry _L/_R/_U suffixes</span></p>')
    parts.append(f'''
        <header>
            <h1>📊 Cross-Dataset Comparison Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Datasets: {dataset_display}</p>
            <p>Threshold mode: <strong>{mode_label}</strong> | Queries: <strong>{len(queries)}</strong> | Mode: <strong>{esc(getattr(params, 'comparison_mode', 'path'))}</strong> | Path Enumeration: <strong>{esc(getattr(params, 'path_mode', 'all'))}</strong></p>
            {hemi_badge}
            <p>Each query is one comparison point. Its requested threshold is allowed to differ by dataset; the raw execution union is not a comparison axis.</p>
        </header>
        {mode_specific_note}
    ''')
    # The applied-threshold banner is merged into the Summary section (the
    # Key Findings table carries requested→applied per dataset and a
    # collapsed provenance block), so the report no longer opens with a
    # standalone banner table.
    parts.append(_generate_dataset_coverage_callout(
        analyzer, dataset_names, nickname_map))
    parts.append(_generate_comparability_callout(analyzer, nickname_map))
    parts.append(_generate_toc(
        [row['query_id'] for row in point_rows],
        include_provenance=True,
        include_type_mapping=bool(getattr(params, 'auto_type_mapping', False)),
        include_auto_density=bool(
            getattr(analyzer, '_density_curves_df', None) is not None
            or getattr(analyzer, '_density_windows_df', None) is not None)))

    # Summary: key findings plus the four Standard summary charts (edge
    # counts, total weight, average connection ratio, average traversal
    # probability), each keyed by query. The former standalone
    # applied-threshold banner and 🎯 provenance section are merged in here:
    # each dataset cell shows requested→applied, and the full bottleneck
    # provenance rides below the table as one collapsed block.
    applied_by_qd = {}
    for prov_row in provenance_rows:
        applied_by_qd[(str(prov_row.get('query_id')),
                       str(prov_row.get('dataset')))] = prov_row

    def _threshold_prov_cell(row, dataset):
        requested = row['requested_thresholds'].get(dataset)
        prov = applied_by_qd.get((str(row['query_id']), str(dataset)))
        applied = prov.get('applied_threshold') if prov else None
        if applied is None:
            return esc(str(requested))
        source = str(prov.get('applied_threshold_source') or '')
        mark = ' *' if source and source != 'requested' else ''
        return (f'<span title="applied {esc(str(applied))} '
                f'(source: {esc(source)})">'
                f'{esc(str(requested))}&rarr;{esc(str(applied))}{mark}</span>')

    parts.append('<div id="summary" class="section"><div class="section-header">📋 Summary &amp; Key Findings</div><div class="section-content">')
    parts.append('<p class="card">Ratio and traversal-probability filtering is disabled for pathfinding comparisons. Counts below are keyed by query row; each dataset cell shows <strong>requested&rarr;applied</strong> Min Synapse Count (an asterisk marks an applied value resolved by the strongest-first budget rather than the request). The full bottleneck provenance is collapsed beneath the table.</p>')
    parts.append('<div class="card"><h3>Key Findings by Query</h3><div class="sticky-table-container"><table><thead><tr><th>Query</th><th>Label</th><th>Group</th>')
    parts.extend(f'<th>{esc(nickname_map.get(ds, ds))} req&rarr;app</th>' for ds in dataset_names)
    parts.append('<th>Total edges</th><th>Common edges</th><th>Edge conservation</th><th>Total paths</th><th>Common paths</th><th>Path conservation</th></tr></thead><tbody>')
    for row in point_rows:
        if row.get('row_mode') == 'vertical':
            group_badge = ('<span class="badge" style="background:#eff6ff;'
                           'color:#1d4ed8;">V · per-threshold</span>')
        elif row.get('row_mode') == 'horizontal':
            group_badge = ('<span class="badge" style="background:#fffbeb;'
                           'color:#b45309;">H · density-matched</span>')
        else:
            group_badge = ''
        parts.append('<tr><td><strong>' + esc(row['query_id']) + '</strong></td><td>' + esc(row['query_label']) + f'</td><td>{group_badge}</td>')
        parts.extend(f'<td>{_threshold_prov_cell(row, ds)}</td>' for ds in dataset_names)
        parts.append(''.join([
            f'<td>{row["total_edges"]}</td><td>{row["common_edges"]}</td><td>{row["edge_rate"] * 100:.1f}%</td>',
            f'<td>{row["total_paths"]}</td><td>{row["common_paths"]}</td><td>{row["path_rate"] * 100:.1f}%</td></tr>',
        ]))
    parts.append('</tbody></table></div>')
    # Merged provenance block (former 🎯 section): the banner table's
    # per-dataset×threshold rows plus the machine-readable join, collapsed
    # so the summary stays scannable. The anchor id keeps the TOC deep link.
    banner_html = _generate_applied_threshold_banner(analyzer, dataset_names)
    parts.append('<details id="threshold-provenance" style="margin-top:12px;">'
                 '<summary style="cursor:pointer; font-weight:600; color: var(--primary-color);">'
                 '🎯 Applied thresholds &amp; bottleneck provenance</summary>'
                 '<div style="margin-top:8px;">'
                 + (banner_html or '<p class="note">No bottleneck provenance recorded for this run.</p>')
                 + '<p class="note">Machine-readable join: '
                 + _make_link(os.path.join(results_dir, 'threshold_combinations.csv'), output_root)
                 + ' and ' + _make_link(os.path.join(results_dir, 'pathfinding_provenance.csv'), output_root)
                 + '.</p></div></details>')
    parts.append('</div>')

    # Chart data, including per-query average connection ratio and traversal
    # probability. Each dataset cell is read from the raw run at that
    # dataset's own requested threshold.
    chart_counts = []
    chart_weights = []
    avg_ratio_data = []
    avg_prob_data = []
    for query in queries:
        query_id = str(query.get('id') or query.get('query_id'))
        query_label = str(query.get('label') or query_id)
        for dataset in dataset_names:
            threshold = int((query.get('thresholds') or {})[dataset])
            df = analyzer.raw_results.get(dataset, {}).get(threshold, pd.DataFrame())
            chart_counts.append({'query_id': query_id, 'query_label': query_label, 'dataset': nickname_map.get(dataset, dataset), 'count': int(len(df))})
            chart_weights.append({'query_id': query_id, 'query_label': query_label, 'dataset': nickname_map.get(dataset, dataset), 'weight': float(df['weight'].sum()) if not df.empty and 'weight' in df.columns else 0})
            avg_ratio = 0.0
            avg_prob = 0.0
            try:
                ratio_data = analyzer._get_edge_ratio_data_for_threshold(threshold)
                if ratio_data is not None and not ratio_data.empty and dataset in ratio_data.columns:
                    non_zero = ratio_data[dataset][ratio_data[dataset] > 0]
                    if len(non_zero) > 0:
                        avg_ratio = float(non_zero.mean())
                        if pd.isna(avg_ratio):
                            avg_ratio = 0.0
            except Exception:
                avg_ratio = 0.0
            try:
                prob_data = analyzer._get_prob_data_for_threshold(threshold)
                if prob_data is not None and not prob_data.empty and dataset in prob_data.columns:
                    non_zero = prob_data[dataset][prob_data[dataset] > 0]
                    if len(non_zero) > 0:
                        avg_prob = float(non_zero.mean())
                        if pd.isna(avg_prob):
                            avg_prob = 0.0
            except Exception:
                avg_prob = 0.0
            avg_ratio_data.append({'query_id': query_id, 'query_label': query_label, 'dataset': nickname_map.get(dataset, dataset), 'ratio': avg_ratio})
            avg_prob_data.append({'query_id': query_id, 'query_label': query_label, 'dataset': nickname_map.get(dataset, dataset), 'prob': avg_prob})
    _query_report_save_used_data(analyzer, pd.DataFrame(chart_counts), 'edge_count_data_by_query.csv')
    _query_report_save_used_data(analyzer, pd.DataFrame(chart_weights), 'total_weight_data_by_query.csv')
    _query_report_save_used_data(analyzer, pd.DataFrame(avg_ratio_data), 'avg_ratio_data_by_query.csv')
    _query_report_save_used_data(analyzer, pd.DataFrame(avg_prob_data), 'avg_prob_data_by_query.csv')
    labels_json = json.dumps([row['query_id'] for row in point_rows])
    display_labels_json = json.dumps([row['query_label'] for row in point_rows])
    datasets_json = json.dumps([nickname_map.get(ds, ds) for ds in dataset_names])
    counts_json = json.dumps(chart_counts, default=str)
    weights_json = json.dumps(chart_weights, default=str)
    ratio_json = json.dumps(avg_ratio_data, default=str)
    prob_json = json.dumps(avg_prob_data, default=str)
    parts.append("""
        <div class="card"><h3>Edges, Weight, Ratio and Probability Across All Queries</h3>
        <style>
          .query-split { display: flex; flex-wrap: wrap; gap: 30px; }
          .query-split-col { flex: 1 1 44%; min-width: 0; }
          .query-split-col h4 { color: #1d4ed8; margin: 4px 0 12px 0; }
        </style>
        <div class="query-split">
            <div class="query-split-col" id="querySplitColV">
                <h4>Per-threshold analysis (vertical rows)</h4>
                <div id="queryEdgeCountChartV" class="chart-container"></div>
                <div id="queryTotalWeightChartV" class="chart-container"></div>
                <div id="queryAvgRatioChartV" class="chart-container"></div>
                <div id="queryAvgProbChartV" class="chart-container"></div>
            </div>
            <div class="query-split-col" id="querySplitColH">
                <h4>Density-matched analysis (horizontal rows)</h4>
                <div id="queryEdgeCountChartH" class="chart-container"></div>
                <div id="queryTotalWeightChartH" class="chart-container"></div>
                <div id="queryAvgRatioChartH" class="chart-container"></div>
                <div id="queryAvgProbChartH" class="chart-container"></div>
            </div>
            <div class="query-split-col" id="querySplitColO" style="display:none;">
                <h4>Other queries</h4>
                <div id="queryEdgeCountChartO" class="chart-container"></div>
                <div id="queryTotalWeightChartO" class="chart-container"></div>
                <div id="queryAvgRatioChartO" class="chart-container"></div>
                <div id="queryAvgProbChartO" class="chart-container"></div>
            </div>
        </div>
        <script>
        (function() {
            const ids = __IDS__, datasets = __DATASETS__;
            const rowsBy = {
                counts: __COUNTS__, weights: __WEIGHTS__,
                ratios: __RATIOS__, probs: __PROBS__,
            };
            const metrics = [
                ['counts',  'count',  'queryEdgeCountChart',  'Edge count'],
                ['weights', 'weight', 'queryTotalWeightChart', 'Total edge weight'],
                ['ratios',  'ratio',  'queryAvgRatioChart',   'Avg connection ratio (w_ij / W_j)'],
                ['probs',   'prob',   'queryAvgProbChart',    'Avg traversal probability'],
            ];
            const groups = [
                ['V', 'threshold=', 'querySplitColV'],
                ['H', 'aligned_density=', 'querySplitColH'],
                ['O', '', 'querySplitColO'],
            ];
            groups.forEach(function(g) {
                var tag = g[0], prefix = g[1], colId = g[2];
                var labels = prefix ? ids.filter(function(id) {
                    return id.indexOf(prefix) === 0; }) : ids.filter(function(id) {
                    return id.indexOf('threshold=') !== 0 &&
                           id.indexOf('aligned_density=') !== 0; });
                if (!labels.length) {
                    var col = document.getElementById(colId);
                    if (col) col.style.display = 'none';
                    return;
                }
                metrics.forEach(function(m) {
                    var rows = rowsBy[m[0]], field = m[1], yTitle = m[3];
                    var div = m[2] + tag;
                    var traces = datasets.map(function(ds) {
                        return {
                            name: ds, x: labels,
                            y: labels.map(function(id) {
                                var r = rows.find(function(v) {
                                    return v.dataset === ds && v.query_id === id; });
                                return r ? Number(r[field]) : 0; }),
                            type: 'scatter', mode: 'lines+markers',
                            connectgaps: false
                        }; });
                    Plotly.newPlot(div, traces, {
                        xaxis: { title: 'Query' },
                        yaxis: { title: yTitle }
                    }, { responsive: true });
                });
            });
        })();
        </script>
    """.replace('__IDS__', labels_json).replace('__DATASETS__', datasets_json).replace('__COUNTS__', counts_json).replace('__WEIGHTS__', weights_json).replace('__RATIOS__', ratio_json).replace('__PROBS__', prob_json))
    parts.append('</div></div>')
    parts.append(_generate_neuron_counts_section(analyzer, dataset_names, nickname_map))
    # The Quick Navigation links #query-resolution — the section is
    # threshold-agnostic (per-token resolver records), so the combination
    # report must render it too, not just the Standard report.
    parts.append(_generate_query_resolution_section(
        analyzer, dataset_names, nickname_map))
    parts.append(_generate_type_mapping_section(analyzer, dataset_names))

    # Hemisphere symmetry: point-aware. When the feature is enabled, render
    # the Standard per-dataset summary table once per query using each
    # dataset's requested-threshold run; otherwise show the same warning
    # card as the Standard report.
    separate_hemispheres = bool(getattr(params, 'separate_hemispheres', False))
    symmetry_analysis = bool(getattr(params, 'symmetry_analysis', False))
    parts.append('<div id="hemisphere-symmetry" class="section"><div class="section-header">🪞 Hemisphere Symmetry</div><div class="section-content">')
    if symmetry_analysis and separate_hemispheres:
        parts.append('<p class="card">Hemisphere symmetry summaries per dataset and query (ipsilateral vs contralateral). Each dataset is read at its own requested threshold.</p>')
        for group_banner, query in _iter_grouped(queries):
            if group_banner:
                parts.append(group_banner)
            query_id = str(query.get('id') or query.get('query_id'))
            query_label = str(query.get('label') or query_id)
            summaries = analyzer.get_hemisphere_symmetry_summaries_for_query(query)
            parts.append(f'<div class="card"><h3>{esc(_point_heading(query_id, query_label))}</h3>')
            if not summaries:
                parts.append('<p style="color:#999; text-align:center;">No hemisphere symmetry summaries found for this query.</p></div>')
                continue
            parts.append('<table><thead><tr>'
                         '<th>Dataset</th>'
                         '<th>Ipsi Jaccard</th><th>Contra Jaccard</th>'
                         '<th>Ipsi Conserved/Union</th><th>Contra Conserved/Union</th>'
                         '<th>Types Conserved/Union</th><th>Counts L/R</th>'
                         '</tr></thead><tbody>')
            for dataset in dataset_names:
                summary = summaries.get(dataset)
                if not summary:
                    continue
                ipsi = summary.get('ipsi', {})
                contra = summary.get('contra', {})
                types = summary.get('neuron_types', {})
                counts = summary.get('hemisphere_counts', {}).get('total', {})
                ipsi_cons = f"{ipsi.get('conserved', 0)}/{ipsi.get('union', 0)}"
                contra_cons = f"{contra.get('conserved', 0)}/{contra.get('union', 0)}"
                types_cons = f"{types.get('types_conserved', 0)}/{types.get('types_union', 0)}"
                lr_counts = f"{counts.get('L', 0)}/{counts.get('R', 0)}"
                parts.append(
                    f'<tr><td><strong>{esc(nickname_map.get(dataset, dataset))}</strong></td>'
                    f'<td>{float(ipsi.get("jaccard", 0)):.3f}</td>'
                    f'<td>{float(contra.get("jaccard", 0)):.3f}</td>'
                    f'<td>{esc(ipsi_cons)}</td>'
                    f'<td>{esc(contra_cons)}</td>'
                    f'<td>{esc(types_cons)}</td>'
                    f'<td>{esc(lr_counts)}</td></tr>')
            parts.append('</tbody></table></div>')
    else:
        # Name the exact missing piece: the summary tables need BOTH the
        # per-hemisphere suffixing (separate_hemispheres) and the symmetry
        # summaries (symmetry_analysis).
        missing = []
        if not separate_hemispheres:
            missing.append('<code>separate_hemispheres=True</code> (adds the '
                           '_L/_R/_U type suffixes and per-hemisphere outputs)')
        if not symmetry_analysis:
            missing.append('<code>symmetry_analysis=True</code> (computes the '
                           'per-query Ipsi/Contra summaries)')
        parts.append('<div class="card" style="background: #fef3c7; border: 1px solid #f59e0b;"><p style="color: #92400e; margin: 0;">'
                     '<strong>⚠️ Hemisphere analysis unavailable:</strong> not enabled for this comparison. Set '
                     + ' and '.join(missing) + ' in ComparisonParameters '
                     '(hemisphere symmetry is query-independent to enable) and re-run to '
                     'populate the per-query Ipsi/Contra tables.</p></div>')
    parts.append('</div></div>')

    # Similarity: the Standard four-metric heatmap card per query plus the
    # data links (the per-pair numeric table was dropped — the heatmaps
    # annotate every cell, and the full pair-level values live in the CSVs).
    # Both consume the exact same query-keyed similarity frame that feeds
    # the similarity_by_query.csv used-data export.
    parts.append('<div id="similarity" class="section"><div class="section-header">🔢 Similarity Matrices</div><div class="section-content"><p class="card">Pairwise metrics are calculated independently for every query row. Hover any heatmap cell for the exact value; the full pair table is in the linked CSVs.</p>')
    for group_banner, point_row in _iter_grouped(point_rows):
        if group_banner:
            parts.append(group_banner)
        query_id = point_row['query_id']
        query_label = point_row['query_label']
        safe = _query_report_slug(query_id)
        qdf = similarity_df[similarity_df.get('query_id', pd.Series(dtype=str)).astype(str) == query_id] if not similarity_df.empty and 'query_id' in similarity_df.columns else pd.DataFrame()
        matrices = _similarity_matrices_from_frame(qdf, dataset_names)
        parts.append(_similarity_heatmap_card(
            safe, _point_heading(query_id, query_label),
            [nickname_map.get(ds, ds) for ds in dataset_names], matrices))
        parts.append(_similarity_detail_table(qdf, dataset_names, nickname_map))
        parts.append('<p class="note">Used data: ' + _make_link(os.path.join(used_dir, 'similarity_by_query.csv'), output_root) + f' | query export: {_make_link(os.path.join(output_root, "similarity_matrices", f"similarity_query_{safe}.csv"), output_root)}</p>')
    parts.append('</div></div>')

    # Networks: the exact Standard section generator, driven by query keys
    # and query-aware data getters. The query is the primary tab axis and
    # the dataset view shows one dataset across all queries. Auto-mode
    # groups render as two separated sections (per-threshold vs
    # density-matched).
    try:
        _grouped_rows = [
            (gmode, [r for r in point_rows if r.get('row_mode') == gmode])
            for gmode in ('vertical', 'horizontal')]
        _grouped_rows = [(m, rs) for m, rs in _grouped_rows if rs]
        if not _grouped_rows:
            _grouped_rows = [('', point_rows)]
        for _gi, (_gmode, _grows) in enumerate(_grouped_rows):
            _sec_id = 'networks' if len(_grouped_rows) == 1 \
                else f'networks-{_gmode or _gi}'
            _note = ''
            if _gmode:
                _title, _desc = QUERY_GROUP_META[_gmode]
                _note = _query_group_banner(_gmode)
            parts.append(_generate_networks_section(
                analyzer, dataset_names, [], nickname_map,
                point_keys=[row['query_id'] for row in _grows],
                point_labels=[_point_heading(row['query_id'],
                                             row['query_label'])
                              for row in _grows],
                aligned_network_getter=lambda k: analyzer.get_aligned_data_for_network(
                    query_by_id.get(k, k)),
                aligned_getter=lambda k: analyzer.get_aligned_data_for_query(
                    query_by_id.get(k, k)),
                path_getter=lambda k: analyzer._get_path_data_for_query(
                    query_by_id.get(k, k)),
                mode_labels=('Query', 'Dataset'),
                tab_label_prefix='',
                key_noun_plural='queries',
                key_noun_singular='query',
                section_id=_sec_id,
                section_note=_note))
    except Exception as e:
        parts.append(f'<div id="networks" class="section"><div class="section-header">🕸️ Network Visualizations</div><div class="section-content"><div class="card"><p>Network rendering failed: {esc(str(e))}</p></div></div></div>')

    def render_query_matrix_section(section_id, section_header, intro, table_builder, export_pattern,
                                    rows=None, group_banner=''):
        """Query tabs + per-dataset-across-queries view for one matrix type.

        ``rows``/``group_banner`` let auto mode render the per-threshold
        (vertical) and density-matched (horizontal) analyses as separate
        sections with distinct DOM scopes."""
        rows = point_rows if rows is None else rows
        section_attr = html.escape(str(section_id), quote=True)
        root_json = json.dumps(str(section_id))
        by_query_id = f'{section_id}_by_query'
        by_dataset_id = f'{section_id}_by_dataset'
        dataset_keys = {}
        used_dataset_keys = set()
        for dataset in dataset_names:
            base_key = _js_ident(nickname_map.get(dataset, dataset))
            key = base_key
            suffix = 2
            while key in used_dataset_keys:
                key = f'{base_key}_{suffix}'
                suffix += 1
            used_dataset_keys.add(key)
            dataset_keys[dataset] = key

        out = [f'<div id="{section_attr}" class="section"><div class="section-header">{section_header}</div><div class="section-content">']
        out.append(f'<p class="card">{intro}</p>')
        if group_banner:
            out.append(group_banner)
        out.append('<div style="margin-bottom: 15px;"><span style="font-weight: 600; margin-right: 10px;">View by:</span>'
                   '<button class="tab-btn active" data-matrix-mode="query">Query</button>'
                   '<button class="tab-btn" data-matrix-mode="dataset">Dataset</button></div>')
        # By-query view
        out.append(f'<div id="{by_query_id}" class="tabs"><div class="tab-buttons">')
        for i, row in enumerate(rows):
            active = 'active' if i == 0 else ''
            query_key = _query_report_slug(row["query_id"])
            out.append(
                f'<button class="tab-btn {active}" '
                f'data-matrix-query-key="{html.escape(query_key, quote=True)}">'
                f'{esc(row["query_id"])}</button>')
        out.append('</div>')
        for i, row in enumerate(rows):
            query_id = row['query_id']
            safe = _query_report_slug(query_id)
            active = 'active' if i == 0 else ''
            out.append(f'<div id="{section_id}_query_tab_{html.escape(safe, quote=True)}" class="tab-content {active}">')
            caption = (f"{_point_heading(query_id, row['query_label'])} "
                       f"(requested: "
                       + ', '.join(f"{nickname_map.get(ds, ds)}={row['requested_thresholds'].get(ds)}"
                                   for ds in dataset_names) + ')')
            out.append(table_builder(query_id, caption))
            out.append('<p class="note">Export: ' + _make_link(os.path.join(results_dir, export_pattern.format(safe=safe)), output_root) + '</p>')
            out.append('</div>')
        out.append('</div>')
        # By-dataset view: one dataset across all queries
        out.append(f'<div id="{by_dataset_id}" class="tabs" style="display: none;"><div class="tab-buttons">')
        for i, ds in enumerate(dataset_names):
            active = 'active' if i == 0 else ''
            nick = nickname_map.get(ds, ds)
            out.append(
                f'<button class="tab-btn {active}" '
                f'data-matrix-dataset-key="{html.escape(dataset_keys[ds], quote=True)}">'
                f'{esc(nick)}</button>')
        out.append('</div>')
        for i, ds in enumerate(dataset_names):
            nick = nickname_map.get(ds, ds)
            active = 'active' if i == 0 else ''
            dataset_key = dataset_keys[ds]
            out.append(f'<div id="{section_id}_dataset_tab_{html.escape(dataset_key, quote=True)}" class="tab-content {active}">')
            out.append(_query_dataset_across_points_table(
                point_rows, aligned_by_id, path_by_id, ds, dataset_names,
                nickname_map, section_id))
            out.append('</div>')
        out.append('</div>')
        out.append(f'''
        <script>
            (function() {{
                const root = document.getElementById({root_json});
                if (!root) return;
                const byQuery = document.getElementById({json.dumps(by_query_id)});
                const byDataset = document.getElementById({json.dumps(by_dataset_id)});
                const setActive = (buttons, selected) => {{
                    buttons.forEach(button => button.classList.toggle('active', button === selected));
                }};
                const switchMode = (mode, button) => {{
                    setActive(root.querySelectorAll('[data-matrix-mode]'), button);
                    byQuery.style.display = mode === 'query' ? 'block' : 'none';
                    byDataset.style.display = mode === 'dataset' ? 'block' : 'none';
                }};
                const showQuery = (key, button) => {{
                    byQuery.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                    const panel = document.getElementById({json.dumps(section_id + '_query_tab_')} + key);
                    if (panel) panel.classList.add('active');
                    setActive(byQuery.querySelectorAll('.tab-btn'), button);
                }};
                const showDataset = (key, button) => {{
                    byDataset.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                    const panel = document.getElementById({json.dumps(section_id + '_dataset_tab_')} + key);
                    if (panel) panel.classList.add('active');
                    setActive(byDataset.querySelectorAll('.tab-btn'), button);
                }};
                root.querySelectorAll('[data-matrix-mode]').forEach(button =>
                    button.addEventListener('click', () => switchMode(button.dataset.matrixMode, button)));
                root.querySelectorAll('[data-matrix-query-key]').forEach(button =>
                    button.addEventListener('click', () => showQuery(button.dataset.matrixQueryKey, button)));
                root.querySelectorAll('[data-matrix-dataset-key]').forEach(button =>
                    button.addEventListener('click', () => showDataset(button.dataset.matrixDatasetKey, button)));
            }})();
        </script>
        ''')
        out.append('</div></div>')
        return ''.join(out)

    # Auto-mode separation: the per-threshold (vertical) and
    # density-matched (horizontal) analyses render as separate sections
    # with their own tabs, mirroring the Standard vs combination analyses.
    def _auto_matrix_groups():
        groups = [
            (gmode, [r for r in point_rows if r.get('row_mode') == gmode])
            for gmode in ('vertical', 'horizontal')]
        groups = [(m, rs) for m, rs in groups if rs]
        if not groups:
            groups = [('', point_rows)]
        return groups

    for _gmode, _grows in _auto_matrix_groups():
        _suffix = '' if len(_auto_matrix_groups()) == 1 else \
            ('_v' if _gmode == 'vertical' else '_h')
        _banner = _query_group_banner(_gmode)
        _gtitle, _gdesc = (QUERY_GROUP_META.get(_gmode) or ('', ''))
        _header_suffix = f' — {_gtitle}' if _gmode else ''
        parts.append(render_query_matrix_section(
            f'edge-matrices{_suffix}', f'🔗 Edge Presence Matrices{_header_suffix}',
            'Presence is read from each query\'s own aligned edge matrix; the dataset view shows one dataset across all queries.',
            lambda query_id, caption: _generate_presence_table(
                aligned_by_id.get(query_id, pd.DataFrame()), dataset_names,
                nickname_map, caption_override=caption),
            'edge_presence_matrix_query_{safe}.csv',
            rows=_grows, group_banner=_banner))

    for _gmode, _grows in _auto_matrix_groups():
        _suffix = '' if len(_auto_matrix_groups()) == 1 else \
            ('_v' if _gmode == 'vertical' else '_h')
        _banner = _query_group_banner(_gmode)
        _gtitle, _gdesc = (QUERY_GROUP_META.get(_gmode) or ('', ''))
        _header_suffix = f' — {_gtitle}' if _gmode else ''
        parts.append(render_query_matrix_section(
            f'path-matrices{_suffix}', f'🛤️ Path Presence Matrices{_header_suffix}',
            'Paths follow the Standard report\'s Len column (number of hops); the dataset view shows one dataset across all queries.',
            lambda query_id, caption: _generate_path_presence_table(
                analyzer, path_by_id.get(query_id, pd.DataFrame()), dataset_names,
                nickname_map, threshold=query_by_id.get(query_id),
                caption_override=caption),
            'path_presence_matrix_query_{safe}.csv',
            rows=_grows, group_banner=_banner))

    # Conservation: per-query distribution donuts plus the per-query
    # conserved-graph exports. Query order is a display order, not a
    # monotone threshold trend, so there is deliberately no trend chart.
    conservation_colors = [
        '#22c55e',  # All datasets (green)
        '#84cc16',  # N-1 (lime)
        '#eab308',  # N-2 (yellow)
        '#f97316',  # N-3 (orange)
        '#ef4444',  # N-4 (red)
        '#94a3b8',  # unique (gray)
    ]
    n_datasets = len(dataset_names)
    parts.append('<div id="conservation" class="section"><div class="section-header">🏆 Conservation Analysis</div><div class="section-content">')
    parts.append(f'<p class="card">Edge and path conservation across all {n_datasets} datasets, per query row. Shows the distribution of how many datasets each edge/path appears in.</p>')
    # Same auto-fill grid as the Standard per-threshold cards: donut cards
    # pack two-to-three per row instead of stretching full width.
    _donut_grid = ('display: grid; grid-template-columns: '
                   'repeat(auto-fill, minmax(380px, 1fr)); gap: 20px;')
    parts.append(f'<div style="{_donut_grid}">')
    for group_banner, row in _iter_grouped(point_rows):
        if group_banner:
            parts.append('</div>' + group_banner + f'<div style="{_donut_grid}">')
        query_id = row['query_id']
        query_label = row['query_label']
        safe = _query_report_slug(query_id)
        aligned = aligned_by_id.get(query_id, pd.DataFrame())
        path_data = path_by_id.get(query_id, pd.DataFrame())
        available = [ds for ds in dataset_names if ds in aligned.columns]
        edge_counts = {}
        if available and not aligned.empty:
            counts_per_edge = (aligned[available] > 0).sum(axis=1)
            edge_counts = counts_per_edge[counts_per_edge > 0].value_counts().to_dict()
        path_counts = {}
        path_available = [ds for ds in dataset_names if ds in path_data.columns]
        if path_available and not path_data.empty:
            counts_per_path = (path_data[path_available] > 0).sum(axis=1)
            path_counts = counts_per_path[counts_per_path > 0].value_counts().to_dict()

        edge_values = []
        edge_labels = []
        edge_colors = []
        for count in range(n_datasets, 0, -1):
            if count in edge_counts and edge_counts[count] > 0:
                edge_values.append(edge_counts[count])
                if count == n_datasets:
                    edge_labels.append(f'All {n_datasets} datasets')
                elif count == 1:
                    edge_labels.append('Unique (1)')
                else:
                    edge_labels.append(f'In {count} datasets')
                edge_colors.append(conservation_colors[min(n_datasets - count, len(conservation_colors) - 1)])
        path_values = []
        path_labels = []
        path_colors = []
        for count in range(n_datasets, 0, -1):
            if count in path_counts and path_counts[count] > 0:
                path_values.append(path_counts[count])
                if count == n_datasets:
                    path_labels.append(f'All {n_datasets} datasets')
                elif count == 1:
                    path_labels.append('Unique (1)')
                else:
                    path_labels.append(f'In {count} datasets')
                path_colors.append(conservation_colors[min(n_datasets - count, len(conservation_colors) - 1)])

        parts.append(_render_conservation_donut_card(
            safe, f"Conservation at {_point_heading(query_id, query_label)}",
            edge_values, edge_labels, edge_colors,
            path_values, path_labels, path_colors,
            row['total_edges'], row['common_edges'],
            row['total_paths'], row['common_paths']))
    parts.append('</div>')
    parts.append('''<div class="card" style="margin-top: 30px;"><h3>Conserved Graph Visualizations</h3><table><thead><tr><th>Query</th><th>Conserved Paths</th><th>Conserved Reciprocal Graph</th></tr></thead><tbody>''')
    _seen_groups = set()
    for row in point_rows:
        g = row.get('row_mode') or ''
        if g and g not in _seen_groups:
            _seen_groups.add(g)
            _title, _ = QUERY_GROUP_META[g]
            _bg = '#eff6ff' if g == 'vertical' else '#fffbeb'
            _mark = '🟦' if g == 'vertical' else '🟨'
            parts.append(f'<tr><td colspan="3" style="background:{_bg};">'
                         f'<strong>{_mark} {html.escape(_title)}</strong></td></tr>')
        query_id = row['query_id']
        query_file_id = _query_report_slug(query_id)
        # B7 query-id files drop the legacy literal `t` prefix; legacy
        # run folders keep their `t{slug}` files, so prefer whichever
        # exists and fall back to the B7 name.
        conserved_path_file = os.path.join(
            output_root, 'conserved_paths',
            f'conserved_network_{query_file_id}_network.html')
        if not os.path.exists(conserved_path_file):
            conserved_path_file = os.path.join(
                output_root, 'conserved_paths',
                f'conserved_network_t{query_file_id}_network.html')
        conserved_recip_file = os.path.join(
            output_root, 'conserved_reciprocal_graph',
            f'conserved_reciprocal_{query_file_id}_network.html')
        if not os.path.exists(conserved_recip_file):
            conserved_recip_file = os.path.join(
                output_root, 'conserved_reciprocal_graph',
                f'conserved_reciprocal_t{query_file_id}_network.html')
        parts.append(
            f'<tr><td><strong>{esc(query_id)}</strong></td>'
            f'<td>{_make_link(conserved_path_file, output_root)}</td>'
            f'<td>{_make_link(conserved_recip_file, output_root)}</td></tr>')
    parts.append('</tbody></table></div></div></div>')

    # Dataset overlap: the shared count/proportion heatmap card per query.
    parts.append('<div id="overlap-matrices" class="section"><div class="section-header">🔀 Dataset Overlap Matrices</div><div class="section-content">')
    parts.append('<p class="card">Asymmetric overlap matrices per query. Cell (row, col) shows how many edges/paths from the <strong>row</strong> dataset are also found in the <strong>column</strong> dataset. Diagonal = total count per dataset.</p>')
    parts.append('<div class="tabs"><div class="tab-buttons">')
    _overlap_sep = _tab_sep_factory()
    for i, row in enumerate(point_rows):
        parts.append(_overlap_sep(row))
        active = 'active' if i == 0 else ''
        safe = _query_report_slug(row['query_id'])
        parts.append(
            f'<button class="tab-btn {active}" '
            f'data-overlap-dom-key="{html.escape(safe, quote=True)}" '
            f'onclick="showOverlapTab({_html_js_arg(safe)}, this)">'
            f'{esc(row["query_id"])}</button>')
    parts.append('</div>')
    for i, row in enumerate(point_rows):
        query_id = row['query_id']
        safe = _query_report_slug(query_id)
        aligned = aligned_by_id.get(query_id, pd.DataFrame())
        path_data = path_by_id.get(query_id, pd.DataFrame())
        n = len(dataset_names)
        edge_matrix = [[0 for _ in range(n)] for _ in range(n)]
        for i1, d1 in enumerate(dataset_names):
            edges_in_d1 = set(aligned.index[aligned[d1] > 0]) if d1 in aligned.columns else set()
            edge_matrix[i1][i1] = len(edges_in_d1)
            for i2, d2 in enumerate(dataset_names):
                if i1 != i2:
                    edges_in_d2 = set(aligned.index[aligned[d2] > 0]) if d2 in aligned.columns else set()
                    edge_matrix[i1][i2] = len(edges_in_d1 & edges_in_d2)
        path_matrix = [[0 for _ in range(n)] for _ in range(n)]
        path_available = [ds for ds in dataset_names if ds in path_data.columns]
        if path_available and not path_data.empty:
            for i1, d1 in enumerate(dataset_names):
                paths_in_d1 = set(path_data.index[path_data[d1] > 0]) if d1 in path_data.columns else set()
                path_matrix[i1][i1] = len(paths_in_d1)
                for i2, d2 in enumerate(dataset_names):
                    if i1 != i2:
                        paths_in_d2 = set(path_data.index[path_data[d2] > 0]) if d2 in path_data.columns else set()
                        path_matrix[i1][i2] = len(paths_in_d1 & paths_in_d2)
        parts.append(_render_overlap_point_card(
            safe, f"Dataset Overlap for {_point_heading(query_id, row['query_label'])}",
            [nickname_map.get(ds, ds) for ds in dataset_names],
            edge_matrix, path_matrix, active=(i == 0)))
    parts.append('''</div>
        <script>
            function showOverlapTab(key, button) {
                document.querySelectorAll('#overlap-matrices .tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('#overlap-matrices .tab-btn').forEach(el => el.classList.remove('active'));
                const domKey = button && button.dataset.overlapDomKey;
                const panel = domKey ? document.getElementById('overlap_tab_' + domKey) : null;
                if (!panel) return;
                panel.classList.add('active');
                if (button) button.classList.add('active');
            }
        </script>''')
    parts.append('</div></div>')

    # Statistics: the shared per-point statistics blocks (identical row
    # semantics to Standard) per query tab, with requested/applied
    # provenance, plus the 2x2 similarity-trends plot on the query axis.
    parts.append('<div id="statistics" class="section"><div class="section-header">📉 Statistics</div><div class="section-content">')
    parts.append('<p class="card">Detailed statistics per dataset and query. Per-Dataset Statistics use the same definition as the Standard report (aligned edges &gt;0, mean over positive weights); the note above each tab adds each dataset\'s requested threshold.</p>')
    parts.append('<div class="tabs"><div class="tab-buttons">')
    _stats_sep = _tab_sep_factory()
    for i, row in enumerate(point_rows):
        parts.append(_stats_sep(row))
        active = 'active' if i == 0 else ''
        safe = _query_report_slug(row['query_id'])
        parts.append(
            f'<button class="tab-btn {active}" '
            f'data-stats-dom-key="{html.escape(safe, quote=True)}" '
            f'onclick="showStatsTab({_html_js_arg(safe)}, this)">'
            f'{esc(row["query_id"])}</button>')
    parts.append('</div>')
    for i, row in enumerate(point_rows):
        query_id = row['query_id']
        safe = _query_report_slug(query_id)
        active = 'active' if i == 0 else ''
        parts.append(f'<div id="stats_tab_{safe}" class="tab-content {active}">')
        prov_line = ', '.join(
            f"{nickname_map.get(ds, ds)}: requested {row['requested_thresholds'].get(ds)}"
            for ds in dataset_names)
        parts.append(f'<p class="note">{esc(_point_heading(query_id, row["query_label"]))} | {esc(prov_line)}</p>')
        parts.append(_stats_blocks_html(
            aligned_by_id.get(query_id, pd.DataFrame()), dataset_names,
            nickname_map,
            f"({_point_heading(query_id, row['query_label'])})"))
        parts.append('</div>')
    parts.append('''</div>
        <script>
            function showStatsTab(key, button) {
                document.querySelectorAll('#statistics .tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('#statistics .tab-btn').forEach(el => el.classList.remove('active'));
                const domKey = button && button.dataset.statsDomKey;
                const panel = domKey ? document.getElementById('stats_tab_' + domKey) : null;
                if (!panel) return;
                panel.classList.add('active');
                if (button) button.classList.add('active');
            }
        </script>''')
    try:
        parts.append(_generate_similarity_trends_2x2_plot(
            analyzer, dataset_names, [], nickname_map,
            point_keys=[row['query_id'] for row in point_rows],
            point_labels=[f"{row['query_id']} — {row['query_label']}" for row in point_rows],
            point_similarities={
                row['query_id']:
                    similarity_df[similarity_df.get('query_id', pd.Series(dtype=str)).astype(str) == row['query_id']]
                for row in point_rows},
            axis_title='Query (display order)',
            card_title='Similarity Trends Across Query Rows'))
    except Exception as e:
        parts.append(f'<div class="card"><p>Similarity trends plot failed: {esc(str(e))}</p></div>')
    parts.append('</div></div>')

    # Auto threshold-density alignment (plan §5 Phase E): present in the
    # combination report too, because auto mode installs its aligned rows as
    # a combination schedule and routes here.
    parts.append(_generate_auto_density_alignment_section(
        analyzer, dataset_names, nickname_map))

    parts.append(_generate_footer())
    return ''.join(parts)


def _query_dataset_across_points_table(point_rows, aligned_by_id, path_by_id,
                                       dataset, dataset_names, nickname_map,
                                       section_id):
    """Render one dataset's presence/weights across all query rows."""
    esc = html.escape
    nick = nickname_map.get(dataset, dataset)
    is_path = section_id == 'path-matrices'
    header = 'Path' if is_path else 'Edge'
    by_key = {}
    for row in point_rows:
        data = (path_by_id if is_path else aligned_by_id).get(row['query_id'], pd.DataFrame())
        if dataset in data.columns and not data.empty:
            present = data.index[data[dataset] > 0]
            for key in present:
                weight = data.loc[key, dataset]
                if hasattr(weight, 'iloc'):
                    weight = weight.iloc[0]
                by_key.setdefault(key, {})[row['query_id']] = weight
    if not by_key:
        return (f'<p style="color:#999; text-align:center;">No connections for '
                f'{esc(nick)} at any query.</p>')

    ordered = sorted(by_key.items(), key=lambda item: str(item[0]))
    parts = [f'<div style="margin-bottom: 8px; color: var(--primary-color); font-weight: 600;">Dataset: {esc(nick)} (all queries)</div>']
    parts.append('<div class="sticky-table-container" style="overflow-x: auto;"><table><thead><tr>')
    parts.append(f'<th>{esc(header)}</th>')
    if is_path:
        parts.append('<th>Len</th>')
    parts.extend(f'<th>{esc(row["query_id"])}</th>' for row in point_rows)
    parts.append('<th>Queries present</th></tr></thead><tbody>')
    for key, per_query in ordered:
        count = len(per_query)
        badge = ('badge-success' if count == len(point_rows)
                 else 'badge-warning' if count > 1 else 'badge-danger')
        cells = f'<tr><td><strong>{esc(str(key))}</strong></td>'
        if is_path:
            cells += f'<td>{_path_length_from_key(key)}</td>'
        for row in point_rows:
            weight = per_query.get(row['query_id'])
            if weight is not None and weight > 0:
                cells += (f'<td><span class="presence-check">✔️</span> '
                          f'{_query_report_cell(weight)}</td>')
            else:
                cells += '<td><span class="presence-cross">❌</span></td>'
        cells += f'<td><span class="badge {badge}">{count}/{len(point_rows)}</span></td></tr>'
        parts.append(cells)
    parts.append('</tbody></table></div>')
    return ''.join(parts)


def _generate_html_header() -> str:
    """Generate HTML head with CSS styles."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cross-Dataset Comparison Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script>
        // CDN fallback guard: show a clear banner instead of a silent blank canvas
        if (typeof vis === 'undefined') {
            window.addEventListener('DOMContentLoaded', function() {
                const section = document.getElementById('networks');
                if (section && !section.querySelector('.cdn-error')) {
                    const div = document.createElement('div');
                    div.className = 'cdn-error';
                    div.style.cssText = 'background:#fffbeb;border:1px solid #f59e0b;border-radius:8px;padding:12px 14px;color:#92400e;font-size:13px;margin-bottom:15px;';
                    div.textContent = '⚠️ vis-network failed to load (CDN unreachable). Check your internet connection and reload this page.';
                    section.prepend(div);
                }
            });
        }
    </script>
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #64748b;
            --success-color: #22c55e;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --card-bg: #ffffff;
            --border-color: #e2e8f0;
            --conserved-color: #22c55e;
            --partial-color: #f59e0b;
            --unique-color: #94a3b8;
        }
        /* Suspects hover popover (plan-cross-dataset-report-mapping-grid-
           and-role-tables Item 4): opens on hover/focus, absolutely
           positioned so the table never re-flows or scrolls, and stays
           open while the pointer is over badge or panel so the rival
           table's text stays selectable and copiable. */
        .suspects-wrap { position: relative; display: inline-block; z-index: 5; }
        .suspects-wrap .suspects-trigger { cursor: help; }
        .suspects-wrap .suspects-pop {
            display: none; position: absolute; top: 100%; left: 0; z-index: 60;
            background: #ffffff; border: 1px solid var(--border-color);
            border-radius: 8px; padding: 10px 12px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.18);
            min-width: 420px; max-width: 620px; font-size: 0.8em; color: #333;
            text-align: left; white-space: normal; cursor: auto;
        }
        .suspects-wrap:hover .suspects-pop,
        .suspects-wrap:focus-within .suspects-pop { display: block; }
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
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
        }
        header h1 { font-size: 2rem; margin-bottom: 10px; }
        header p { opacity: 0.9; }
        .section { margin: 40px 0; }
        .section-header {
            background: linear-gradient(90deg, var(--primary-color), #60a5fa);
            color: white;
            padding: 15px 25px;
            border-radius: 12px 12px 0 0;
            font-size: 1.3rem;
            font-weight: 600;
        }
        .section-content {
            background: var(--card-bg);
            border-radius: 0 0 12px 12px;
            padding: 24px;
            border: 1px solid var(--border-color);
            border-top: none;
        }
        .card {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--border-color);
        }
        .card h3 {
            color: var(--primary-color);
            margin-bottom: 16px;
            font-size: 1.1rem;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
        }
        .card h4 {
            color: var(--secondary-color);
            margin: 12px 0 8px 0;
            font-size: 1rem;
        }
        .grid { display: grid; gap: 20px; }
        .grid-2 { grid-template-columns: repeat(2, 1fr); }
        .grid-3 { grid-template-columns: repeat(3, 1fr); }
        .stat-box {
            background: var(--bg-color);
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-box .value { font-size: 1.8rem; font-weight: bold; color: var(--primary-color); }
        .stat-box .label { color: var(--secondary-color); font-size: 0.85rem; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 0.85rem;
        }
        th, td {
            padding: 8px 10px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        th { background: var(--bg-color); font-weight: 600; }
        tr:hover { background: var(--bg-color); }
        /* Sticky header for scrollable tables */
        .sticky-table-container {
            /* ~20-row scroll viewport on every data table (plan round-4 F-6) */
            max-height: 520px;
            overflow-y: auto;
            overflow-x: auto;
            position: relative;
        }
        .sticky-table-container thead th {
            position: sticky;
            top: 0;
            background: #f1f5f9;
            z-index: 10;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .badge {
            padding: 3px 8px;
            border-radius: 20px;
            font-size: 0.7rem;
            font-weight: 600;
        }
        .badge-success { background: #dcfce7; color: #166534; }
        .badge-warning { background: #fef3c7; color: #92400e; }
        .badge-danger { background: #fee2e2; color: #991b1b; }
        .presence-check { color: var(--success-color); }
        .presence-cross { color: var(--danger-color); }
        .toc {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            border: 1px solid var(--border-color);
        }
        .toc h2 { margin-bottom: 15px; color: var(--primary-color); }
        .toc ul { list-style: none; display: flex; flex-wrap: wrap; gap: 10px; }
        .toc a {
            display: inline-block;
            padding: 8px 16px;
            background: var(--bg-color);
            color: var(--primary-color);
            text-decoration: none;
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .toc a:hover { background: var(--primary-color); color: white; }
        .print-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 12px 24px;
            background: var(--primary-color);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 1000;
        }
        .print-btn:hover { background: #1d4ed8; }
        .tabs { margin-top: 15px; }
        .tab-buttons { display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 15px; }
        .tab-btn {
            padding: 8px 16px;
            border: 1px solid var(--border-color);
            background: var(--bg-color);
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
        }
        .tab-btn:hover { background: #e2e8f0; }
        .tab-btn.active { background: var(--primary-color); color: white; border-color: var(--primary-color); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .network-legend {
            display: flex;
            gap: 20px;
            margin-bottom: 10px;
            font-size: 0.9rem;
        }
        .legend-item { display: flex; align-items: center; gap: 5px; }
        .legend-color {
            width: 20px;
            height: 4px;
            border-radius: 2px;
        }
        .chart-container { min-height: 350px; }
        @media print {
            .print-btn { display: none; }
            .card { break-inside: avoid; }
            .section { break-before: page; }
            .tab-content { display: block !important; }
            .tab-buttons { display: none; }
        }
        @media (max-width: 768px) {
            .grid-2, .grid-3 { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
"""


def _threshold_display_label(analyzer, threshold, dataset_names) -> str:
    """``Threshold = t`` with the applied values appended when they differ."""
    try:
        values = []
        for ds in dataset_names:
            if threshold not in analyzer.parameters.get_thresholds_for_dataset(ds):
                continue
            view = analyzer.get_threshold_view(ds, int(threshold))
            values.append(view.get('applied_threshold'))
        if values and (len(set(str(v) for v in values)) > 1
                       or str(values[0]) != str(threshold)):
            return (f'Threshold = {threshold} '
                    f'(applied {"/".join(str(v) for v in values)})')
    except Exception:
        pass
    return f'Threshold = {threshold}'


def _threshold_tab_label(analyzer, threshold, dataset_names,
                         nickname_map) -> str:
    """Tab label for a requested threshold that names its applied value(s).

    Sections still key their points by the requested threshold (the run's
    identity), but a reader must not assume every dataset ran at it. The
    label appends the applied set, e.g. ``t = 5 (applied 19/6/17)``, and
    marks fully-aliased requests.
    """
    base = f't = {threshold}'
    try:
        values = []
        aliased = 0
        for ds in dataset_names:
            if threshold not in analyzer.parameters.get_thresholds_for_dataset(ds):
                continue
            view = analyzer.get_threshold_view(ds, int(threshold))
            values.append(view.get('applied_threshold'))
            if view.get('status') == 'aliased':
                aliased += 1
        if not values:
            return base
        applied = '/'.join(str(v) for v in values)
        if aliased == len(values):
            return f'{base} (aliased → {applied})'
        if len(set(str(v) for v in values)) > 1 or \
                str(threshold) != str(values[0]):
            return f'{base} (applied {applied})'
    except Exception:
        pass
    return base


def _generate_dataset_coverage_callout(analyzer, dataset_names,
                                       nickname_map) -> str:
    """Loud advisory when a configured dataset produced no data (plan B).

    A silent fetch failure used to drop a dataset out of every analysis
    while headers and provenance still listed it. This callout names the
    no-data datasets up front so the reader never misreads a 3-dataset
    comparison as 4-dataset.
    """
    try:
        coverage = analyzer.dataset_coverage()
    except Exception:
        return ''
    no_data = [ds for ds in dataset_names
               if coverage.get(ds, {}).get('status') != 'ok']
    if not no_data:
        return ''
    items = ', '.join(
        f"<strong>{html.escape(str(nickname_map.get(ds, ds)))}</strong> "
        f"({html.escape(str(ds))})" for ds in no_data)
    return f"""
        <div class="section" style="border-left:6px solid #b91c1c; background:#fef2f2; padding:12px 16px; margin:12px 0;">
            <strong style="color:#b91c1c;">⚠️ Dataset coverage warning</strong>
            <div style="margin-top:6px;">
                {items} produced <strong>no data</strong> in this run.
                Every analysis below ran on the remaining datasets only —
                headers and threshold provenance may still list the missing
                dataset(s). Check <code>dataset_data/&lt;dataset&gt;/run_log.txt</code>
                and the run manifest's <code>dataset_coverage</code> block.
            </div>
        </div>
"""


def _generate_applied_threshold_banner(analyzer, dataset_names: List[str]) -> str:
    """Render the cross-dataset threshold/bottleneck provenance table.

    ``applied_threshold`` is the canonical equivalent threshold for the
    materialized output.  It is intentionally displayed beside the
    StrongestFirst landing ``tau`` and the Edge Budget values so a landing
    tau is never mistaken for the applied threshold.  The old collapse title
    is retained for readers/tests that already recognize it.
    """
    meta_map = getattr(analyzer, '_path_run_meta', {}) or {}
    if not meta_map:
        return ''

    def value(value):
        if value is None:
            return '—'
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, float):
            return f'{value:g}'
        return str(value)

    rows = []
    has_provenance = False
    for ds in dataset_names:
        thresholds = sorted(
            threshold for dataset, threshold in meta_map
            if dataset == ds)
        for threshold in thresholds:
            meta = meta_map.get((ds, threshold), {}) or {}
            # Real post-fix runs carry the canonical field even when no
            # budget bit.  Legacy fake/old metadata with no applied state
            # should not create a meaningless banner.
            if not any([
                'applied_threshold' in meta,
                meta.get('tau') is not None,
                meta.get('strongest_first_tau') is not None,
                meta.get('budget_bitten'),
                meta.get('strongest_first_budget_bitten'),
                meta.get('edge_weight_floor') is not None,
            ]):
                continue
            has_provenance = True
            if hasattr(analyzer, '_path_provenance_row'):
                row = analyzer._path_provenance_row(ds, threshold)
            else:
                state = dict(meta)
                row = applied_threshold_provenance(
                    requested_threshold=state.get(
                        'requested_threshold', threshold),
                    strongest_first_tau=state.get(
                        'strongest_first_tau', state.get('tau')),
                    strongest_first_budget_bitten=state.get(
                        'strongest_first_budget_bitten',
                        state.get('budget_bitten', False)),
                    strongest_dropped_bottleneck=state.get(
                        'strongest_dropped_bottleneck'),
                    tau_canonical=state.get('tau_canonical'),
                    edge_weight_floor=state.get('edge_weight_floor'),
                    edge_budget_landing=state.get('edge_budget_landing'),
                    edge_budget=state.get('edge_budget'),
                    strongest_retained_bottleneck=state.get(
                        'strongest_retained_bottleneck'),
                )
                row['tau'] = row.get('strongest_first_tau')
            # Issue #7: the Edge Budget landing tier (w1) is only meaningful
            # when the floor actually bound this threshold; render it as — for
            # untouched rows even though the graph-level floor still exists.
            if not row.get('edge_budget_applied'):
                row = dict(row)
                row['edge_budget_landing'] = None
            rows.append(
                '<tr>' + ''.join(
                    f'<td>{html.escape(value(cell))}</td>'
                    for cell in (
                        ds,
                        threshold,
                        row.get('requested_threshold'),
                        row.get('applied_threshold'),
                        row.get('applied_threshold_source'),
                        row.get('strongest_first_budget'),
                        row.get('strongest_first_budget_bitten'),
                        row.get('tau', row.get('strongest_first_tau')),
                        row.get('edge_budget'),
                        row.get('edge_budget_applied'),
                        row.get('edge_weight_floor'),
                        row.get('edge_budget_landing'),
                        row.get('strongest_dropped_bottleneck'),
                        row.get('strongest_retained_bottleneck'),
                        row.get('paths_complete'),
                        (f"{row.get('untyped_dropped_fraction'):.3f}"
                         if row.get('untyped_dropped_fraction') is not None
                         else None),
                    )) + '</tr>')
    if not has_provenance:
        return ''
    # The first table cell is the threshold label; retain a compact table for
    # the report while keeping every value needed to interpret the run.
    header = ('<th>Dataset</th><th>Threshold</th><th>Requested</th>'
              '<th>Applied</th><th>Source</th><th>SF budget</th>'
              '<th>SF bite</th><th>tau</th><th>Edge budget</th>'
              '<th>Edge floor applied</th><th>w0</th><th>w1</th>'
              '<th>w2</th><th>W*</th><th>Complete</th><th>Untyped dropped</th>')
    return (
        '<div style="margin:14px 0; padding:10px 14px; border:1px solid #d9a441;'
        ' background:#fdf6e3; border-radius:6px; font-size:0.95em;">'
        '<strong>Applied minimal thresholds / threshold provenance:</strong>'
        '<table style="margin-top:6px; border-collapse:collapse;">'
        f'<tr>{header}</tr>{"".join(rows)}</table>'
        '<div style="margin-top:6px; color:#666;">Applied is the canonical '
        'equivalent Min Synapse Count for the materialized output. tau is the '
        'StrongestFirst landing/collapse bound; SF budget is the effective '
        'path budget and Edge budget is the graph edge cap; w0 is the Edge Budget floor, '
        'w1 its landing tier, w2 the strongest dropped bottleneck, and W* the '
        'strongest retained bottleneck. A skipped row aliases the applied folder.'
        '</div>'
        '</div>')


def _generate_comparability_callout(analyzer, nickname_map: Dict[str, str]) -> str:
    """Prominent advisory when budget pruning leaves too few like-for-like
    thresholds (plan §5A / issue #8)."""
    try:
        report = analyzer.comparability_report()
    except Exception:
        return ''
    warnings = list(report.get('warnings') or [])
    if not warnings:
        return ''
    level = report.get('warning_level', 'warning')
    color = '#b91c1c' if level == 'critical' else '#b45309'
    bg = '#fef2f2' if level == 'critical' else '#fffbeb'
    title = ('No comparable thresholds' if level == 'critical'
             else 'Limited threshold comparability')
    common = report.get('common_materialized') or []
    common_txt = ', '.join(str(t) for t in common) if common else '—'
    per_ds = []
    for ds, values in (report.get('materialized_thresholds') or {}).items():
        label = nickname_map.get(ds, ds)
        per_ds.append(f"{label}: {', '.join(str(v) for v in values) or '—'}")
    return f"""
        <div class="section" style="border-left:6px solid {color}; background:{bg}; padding:12px 16px; margin:12px 0;">
            <strong style="color:{color};">⚠️ {title}</strong>
            <ul style="margin:8px 0 4px 18px;">
                {''.join(f'<li>{html_module.escape(w)}</li>' for w in warnings)}
            </ul>
            <div style="font-size:0.9em; color:#555;">
                Comparable across all datasets: <strong>{html_module.escape(common_txt)}</strong>.
                Materialized per dataset — {'; '.join(html_module.escape(p) for p in per_ds)}.
                <br/>Prefer the density-aligned comparison (see Threshold Alignment)
                when the shared grid collapses.
            </div>
        </div>
"""


def _generate_applied_threshold_map_note(analyzer, dataset_names,
                                         nickname_map) -> str:
    """Header note mapping each requested threshold to its applied value."""
    try:
        applied_map = analyzer.get_applied_threshold_map()
    except Exception:
        return ''
    if not applied_map:
        return ''
    rows = []
    for requested, per_ds in applied_map.items():
        cells = []
        for ds in dataset_names:
            info = per_ds.get(ds)
            if not info:
                cells.append('—')
                continue
            applied = info.get('applied')
            if info.get('status') == 'aliased':
                cells.append(f'{applied} (aliased)')
            else:
                cells.append(str(applied))
        rows.append(
            f"<tr><td><strong>t={html_module.escape(str(requested))}</strong></td>"
            + ''.join(f'<td>{html_module.escape(c)}</td>' for c in cells)
            + '</tr>')
    head = ''.join(f'<th>{html_module.escape(nickname_map.get(d, d))}</th>'
                   for d in dataset_names)
    return f"""
        <div class="section-content" style="margin:8px 0;">
            <details><summary style="cursor:pointer;">Requested → applied thresholds</summary>
            <table style="margin-top:8px;"><thead><tr><th>Requested</th>{head}</tr></thead>
            <tbody>{''.join(rows)}</tbody></table>
            <p style="font-size:0.85em;color:#666;">Sections label points by the
            requested threshold; this table shows the applied (materialized)
            threshold actually compared in each dataset. "aliased" means the
            row reuses another threshold's materialized set.</p>
            </details>
        </div>
"""


def _generate_threshold_alignment_section(analyzer, dataset_names,
                                          nickname_map) -> str:
    """Embed the threshold-alignment result in the report (plan §7A).

    Renders the best-match table (applied anchors, alias provenance, match
    status), the density curves, and a like-for-like vs density-aligned
    note. Reads the analyzer's in-memory alignment frames (populated during
    export); returns '' when no alignment was computed.
    """
    best = getattr(analyzer, '_alignment_best_df', None)
    density = getattr(analyzer, '_alignment_density_df', None)
    auto_curves = getattr(analyzer, '_density_curves_df', None)
    if (best is None or getattr(best, 'empty', True)) and (
            density is None or getattr(density, 'empty', True)) and (
            auto_curves is None or getattr(auto_curves, 'empty', True)):
        return ''

    # Static renders from comparison_visualizations/ — these show up in
    # every medium.  The interactive Plotly figure only draws after JS, so
    # print/PDF capture of it comes out blank; prefer the PNG when present.
    vis_rel = 'comparison_visualizations'
    density_png = os.path.join(vis_rel, 'edge_density_threshold_curves.png')
    matrix_png = os.path.join(vis_rel, 'threshold_alignment_matrix.png')

    def _vis_exists(name):
        try:
            base = getattr(getattr(analyzer, 'parameters', None),
                           'full_output_path', None)
            return bool(base) and os.path.exists(os.path.join(base, name))
        except Exception:
            return False

    def _vis_img(name, alt):
        return (f'<img src="{html.escape(name)}" alt="{html.escape(alt)}" '
                'style="max-width:100%;height:auto;" loading="lazy">')

    parts = ['<div id="threshold-alignment" class="section">',
             '<div class="section-header">📐 Threshold Alignment '
             '(density-equivalent thresholds)</div>',
             '<div class="section-content">',
             '<p style="color: var(--secondary-color);">Density-equivalent '
             'thresholds across datasets, computed from distinct mapped type '
             'pairs with at least one edge at or above each threshold. '
             'Density equivalence is a matching hint, <strong>not</strong> '
             'biological equivalence.</p>']

    if best is not None and not getattr(best, 'empty', True):
        anchor = best[best.get('match_kind') == 'anchor'] \
            if 'match_kind' in best.columns else best
        cols = [c for c in (
            'reference_dataset', 'anchor_threshold', 'anchor_aliased_from',
            'target_dataset', 'best_t', 'best_t_range', 'match_status',
            'count_distance', 'within_tolerance', 'jaccard_at_best',
            'rank_similarity_at_best') if c in anchor.columns]
        parts.append('<div class="card"><h3>Best density matches</h3>'
                     '<div style="overflow-x:auto;"><table><thead><tr>'
                     + ''.join(f'<th>{html.escape(c)}</th>' for c in cols)
                     + '</tr></thead><tbody>')
        for _, row in anchor.head(100).iterrows():
            cells = []
            for c in cols:
                v = row.get(c)
                if isinstance(v, bool):
                    v = '✔️' if v else '❌'
                elif v is None or (isinstance(v, float) and v != v):
                    v = '—'
                cells.append(f'<td>{html.escape(str(v))}</td>')
            parts.append('<tr>' + ''.join(cells) + '</tr>')
        parts.append('</tbody></table></div>')
        parts.append('<p style="font-size:0.85em;color:#666;">'
                     '<code>match_status</code>: exact / within_tolerance / '
                     'outside_tolerance / target_density_below_range. '
                     '<code>best_t_range</code> is the plateau interval '
                     'achieving the count. <code>anchor_aliased_from</code> '
                     'lists requested thresholds that collapsed onto this '
                     'applied anchor.</p></div>')

    if density is not None and not getattr(density, 'empty', True):
        if _vis_exists(density_png):
            parts.append('<div class="card"><h3>Density curves</h3>'
                         + _vis_img(density_png,
                                    'Edge density vs threshold curves')
                         + '<p style="font-size:0.85em;color:#666;">'
                         'Left/right panels and per-dataset guides as in '
                         '<code>comparison_visualizations/'
                         'edge_density_threshold_curves.png</code>; typed '
                         '(materialized) thresholds are marked.</p></div>')
        else:
            # No static render available: keep the interactive Plotly
            # figure (screen-only; prints blank).
            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                for ds in dataset_names:
                    sub = density[density['dataset'] == ds]
                    if sub.empty:
                        continue
                    fig.add_trace(go.Scatter(
                        x=sub['threshold'], y=sub['pair_count'],
                        mode='lines', name=str(nickname_map.get(ds, ds))))
                fig.update_layout(height=340, margin=dict(l=40, r=10, t=30, b=40),
                                  xaxis_title='Threshold',
                                  yaxis_title='Distinct mapped type pairs',
                                  title='Edge density vs threshold')
                parts.append('<div class="card"><h3>Density curves</h3>'
                             + fig.to_html(full_html=False,
                                           include_plotlyjs=False)
                             + '</div>')
            except Exception:
                pass

    if _vis_exists(matrix_png):
        parts.append('<div class="card"><h3>Alignment distance matrix</h3>'
                     + _vis_img(matrix_png,
                                'Threshold alignment distance matrix')
                     + '<p style="font-size:0.85em;color:#666;">'
                     'Edge-count distance between each reference anchor '
                     '(applied threshold) and every target threshold; '
                     'green cells are the best matches above.</p></div>')

    parts.append('<p style="color:#666; font-size:0.9em;">Prefer the '
                 '<strong>density-aligned</strong> view above when the '
                 'like-for-like shared-threshold set collapses after budget '
                 'pruning.</p>')

    # --- Auto threshold-density alignment (plan §5 Phase E) ---
    parts.append(_generate_auto_density_alignment_section(
        analyzer, dataset_names, nickname_map))

    parts.append('</div></div>')
    return ''.join(parts)


def _auto_mode_resolution_banner(analyzer) -> str:
    """Callout when auto mode did not install its full aligned schedule.

    A degraded bootstrap used to leave the report reading like a deliberate
    Standard run (plan §auto-mode-density-regression P4, 2026-09-21), so the
    reason now appears in the section it affects.
    """
    status = getattr(analyzer, '_auto_bootstrap_status', None) or {}
    outcome = str(status.get('outcome') or '')
    if outcome not in ('degraded', 'verticals_only'):
        return ''
    bits = []
    uncaptured = [str(ds) for ds in (status.get('uncaptured') or [])]
    if uncaptured:
        bits.append('no density capture: ' + ', '.join(uncaptured))
    failures = status.get('failures') or {}
    for ds in sorted(failures):
        bits.append(f'{ds} bootstrap failed: {failures[ds]}')
    if status.get('reason'):
        bits.append(str(status['reason']))
    detail = html.escape(' · '.join(bits))
    if outcome == 'degraded':
        style = 'background:#fee2e2;border-left:4px solid #ef4444;'
        head = ('<strong>⚠️ Auto (density-aligned) mode did not resolve.'
                '</strong> No aligned schedule could be measured, so this run '
                'compared the requested thresholds as-is. The aligned-rows '
                'table, the per-density analysis sections and the alignment '
                'guides are absent for that reason, not by design.')
    else:
        style = 'background:#fff7ed;border-left:4px solid #b45309;'
        head = ('<strong>⚠️ Auto mode resolved partially.</strong> Only the '
                f'{status.get("vertical_rows", 0)} vertical (same-threshold) '
                'row(s) run as queries; the density-matched horizontal rows '
                'need every dataset’s curve and were not installed.')
    return ('<div style="' + style + 'padding:12px 16px;margin:8px 0;'
            'border-radius:0 6px 6px 0;">' + head
            + (f' <span style="color:#666;">{detail}</span>' if detail else '')
            + ' <span style="color:#666;">Also recorded in '
              'user_warning_notes.txt and run_manifest.json '
              '(auto_mode_status).</span></div>')


def _generate_auto_density_alignment_section(analyzer, dataset_names,
                                             nickname_map) -> str:
    """Density window/coverage table + two-panel curve (all modes, Phase E).

    Self-contained (used by both the Standard and query/combination report
    paths): reads the analyzer's in-memory density frames and renders the
    static PNG when present. Returns '' when no capture was produced.
    """
    curves = getattr(analyzer, '_density_curves_df', None)
    windows = getattr(analyzer, '_density_windows_df', None)
    aligned = getattr(analyzer, '_density_alignment_df', None)
    if (curves is None or getattr(curves, 'empty', True)) and (
            windows is None or getattr(windows, 'empty', True)):
        return ''
    vis_rel = 'comparison_visualizations'

    def _vis_exists(name):
        try:
            base = getattr(getattr(analyzer, 'parameters', None),
                           'full_output_path', None)
            return bool(base) and os.path.exists(os.path.join(base, name))
        except Exception:
            return False

    def _vis_img(name, alt):
        return (f'<img src="{html.escape(name)}" alt="{html.escape(alt)}" '
                'style="max-width:100%;height:auto;" loading="lazy">')

    parts = ['<div id="auto-density-alignment" class="section">'
             '<div class="section-header">📐 Density curves '
             '(query-scoped)</div><div class="section-content">',
             '<p class="card">Density curves are computed on the query\'s '
             '<strong>bodyId</strong> searched graph (cone) for every '
             'queried dataset, from its minimal available threshold to the '
             'measured ceiling; the compared matrices '
             'are <strong>type-level</strong> projections of the same search. '
             'A threshold here is a per-connection synapse count '
             '(Min Synapse Count).</p>']
    parts.append(_auto_mode_resolution_banner(analyzer))
    if windows is not None and not getattr(windows, 'empty', True):
        cols = [c for c in ('dataset', 'applied', 'w_start',
                            'w_star_stored', 'w_star_measured',
                            'w_star_mismatch', 'budget_bitten',
                            'paths_complete', 'path_complete_from',
                            'n_paths', 'n_edges', 'n_edges_active_basis',
                            'n_nodes', 'n_nodes_typed', 'n_nodes_untyped',
                            'n_nodes_debris') if c in windows.columns]
        body = ''.join(
            '<tr>' + ''.join(
                f'<td>{html.escape(str(row[c]))}</td>' for c in cols) + '</tr>'
            for _, row in windows.iterrows())
        head = ''.join(f'<th>{html.escape(str(c))}</th>' for c in cols)
        parts.append('<div class="card"><h3>Completeness windows</h3>'
                     '<div style="overflow-x:auto;"><table><thead><tr>'
                     + head + '</tr></thead><tbody>' + body
                     + '</tbody></table></div>'
                     '<p style="font-size:0.85em;color:#666;">Per-dataset '
                     'window [w_start, w_star_measured] (Min Synapse Count). '
                     '<code>w_star_measured</code> is the measured retained '
                     'ceiling (max enumerated path bottleneck); '
                     '<code>w_star_stored</code> is the provenance value '
                     '(flagged when it disagrees).</p></div>')
    # Interactive two-panel density curves (mirrors the static PNG the
    # analyzer still exports, but hover/zoom-able). Falls back to the PNG
    # only when no in-memory curve data exists.
    curves_html = ''
    if curves is not None and not getattr(curves, 'empty', True):
        curves_html = _density_curves_plotly_card(
            curves, aligned, windows, nickname_map)
    if curves_html:
        parts.append(curves_html)
    else:
        png = os.path.join(vis_rel, 'density_alignment_threshold_curves.png')
        if _vis_exists(png):
            parts.append('<div class="card"><h3>Density curves</h3>'
                         + _vis_img(png,
                                    'Density alignment vs threshold curves')
                         + '<p style="font-size:0.85em;color:#666;">Top: '
                         'normalized edge density E(t)/N with the aligned '
                         'density levels; bottom: enumerated path count '
                         '(diagnostic only — hub-inflated). Each dataset is '
                         'drawn over its own domain.</p></div>')
    if aligned is not None and not getattr(aligned, 'empty', True):
        ds_cols = [d for d in dataset_names if d in aligned.columns]
        cols = ['mode', 'level_continuous', 'level_normalized',
                'degenerate', 'clamped', 'max_abs_deviation'] + ds_cols
        cols = [c for c in cols if c in aligned.columns]
        has_partial = ('partial_datasets' in aligned.columns
                       and aligned['partial_datasets'].notna().any())
        if has_partial:
            cols.append('partial_datasets')

        # Which aligned rows this run actually installed as queries. Export
        # computes horizontal rows over the captured datasets only, so a
        # partially-resolved auto mode can publish density-matched rows that
        # were never run; the card must not claim otherwise (plan
        # §auto-mode-density-regression P3, 2026-09-21).
        installed_cells = set()
        for _q in (getattr(analyzer.parameters, 'threshold_combinations',
                           None) or []):
            _cells = (_q.get('thresholds') or {}) if isinstance(_q, dict) else {}
            try:
                installed_cells.add(tuple(sorted(
                    (str(ds), int(v)) for ds, v in _cells.items())))
            except (TypeError, ValueError):
                continue

        def _row_key(row):
            try:
                return tuple(sorted(
                    (str(ds), int(row[ds])) for ds in dataset_names))
            except (TypeError, ValueError, KeyError):
                return None

        advisory_ids = []
        if 'id' in aligned.columns:
            for _, _row in aligned.iterrows():
                if pd.isna(_row['id']):
                    continue
                if _row_key(_row) not in installed_cells:
                    advisory_ids.append(html.escape(str(_row['id'])))

        def _cell(row, c):
            v = row[c]
            try:
                if v is None or (isinstance(v, float) and v != v):
                    return ''  # None / NaN -> blank, never the literal 'nan'
            except (TypeError, ValueError):
                pass
            return html.escape(str(v))

        body = ''.join(
            '<tr>' + ''.join(f'<td>{_cell(row, c)}</td>' for c in cols)
            + '</tr>'
            for _, row in aligned.iterrows())
        head = ''.join(f'<th>{html.escape(str(c))}</th>' for c in cols)
        parts.append('<div class="card"><h3>Aligned threshold rows</h3>'
                     '<div style="overflow-x:auto;"><table><thead><tr>'
                     + head + '</tr></thead><tbody>' + body
                     + '</tbody></table></div>'
                     + ('<p style="font-size:0.85em;color:#b45309;">'
                        '⚠️ Partial rows: <code>partial_datasets</code> '
                        'names the datasets excluded for missing density '
                        'captures; those rows are computed over the '
                        'captured datasets only.</p>' if has_partial else '')
                     + '<p style="font-size:0.85em;color:#666;">Vertical rows '
                     'share one integer threshold across datasets; '
                     'horizontal rows share a normalized density. When both '
                     'coexist, read the <strong>vertical rows as the '
                     'like-for-like spine</strong> (one identical threshold '
                     'for every dataset) and the <strong>horizontal rows as '
                     'the density-matched envelope</strong> (per-dataset '
                     'thresholds equalizing E(t)/N, meaningful even where '
                     'no shared complete threshold exists). These rows are '
                     + ('also runnable as a combination query.'
                      if not advisory_ids else
                      'installed as this run\'s queries except the advisory '
                      'row(s): ' + ', '.join(advisory_ids[:12])
                      + ('…' if len(advisory_ids) > 12 else '') + '.')
                     + '</p></div>')
    parts.append('</div></div>')
    return ''.join(parts)


def _density_curves_plotly_card(curves, aligned, windows,
                                nickname_map: Dict[str, str]) -> str:
    """Interactive two-panel density-vs-threshold chart (query-scoped).

    Top panel: normalized edge density E(t)/N per dataset with dotted
    guides at the aligned density levels (horizontal rows). Bottom panel:
    enumerated path count (diagnostic — hub-inflated). Each dataset is
    drawn over its own threshold domain, as in the static PNG this card
    replaces.
    """
    try:
        series = []
        for ds, group in curves.groupby('dataset'):
            group = group.sort_values('threshold')
            thresholds = [int(t) for t in group['threshold'].tolist()]
            density = [None if pd.isna(v) else round(float(v), 6)
                       for v in group['density'].tolist()]
            path_counts = [int(v) for v in group['path_count'].tolist()]
            if not thresholds:
                continue
            materialized = set()
            if 'is_materialized' in group.columns:
                materialized = {int(t) for t, m in zip(
                    group['threshold'], group['is_materialized']) if bool(m)}
            w_start = w_star = None
            if windows is not None and not getattr(windows, 'empty', True):
                wrow = windows[windows['dataset'] == ds]
                if not wrow.empty:
                    lo = wrow.iloc[0].get('w_start')
                    hi = wrow.iloc[0].get('w_star_measured')
                    w_start = float(lo) if pd.notna(lo) else None
                    w_star = float(hi) if pd.notna(hi) else None
            series.append({
                'name': str((nickname_map or {}).get(ds, ds)),
                'thresholds': thresholds,
                'density': density,
                'pathCount': path_counts,
                'materialized': sorted(materialized),
                'wStart': w_start,
                'wStar': w_star,
            })
    except Exception:
        return ''
    if not series:
        return ''
    vguides = []
    hguides = []
    try:
        if aligned is not None and not getattr(aligned, 'empty', True):
            for _, row in aligned.iterrows():
                mode = str(row.get('mode') or '')
                row_id = str(row.get('id') or row.get('label') or mode)
                lvl_c = row.get('level_continuous')
                lvl_n = row.get('level_normalized')
                if 'vertical' in mode and lvl_c is not None \
                        and pd.notna(lvl_c):
                    vguides.append({
                        'x': round(float(lvl_c), 6),
                        'label': f'{row_id} — vertical (per-threshold) row'})
                elif 'horizontal' in mode and lvl_n is not None \
                        and pd.notna(lvl_n):
                    hguides.append({
                        'y': round(float(lvl_n), 6),
                        'label': f'{row_id} — horizontal (density-matched) row'})
    except Exception:
        vguides, hguides = [], []
    payload = json.dumps({'series': series, 'vguides': vguides,
                          'hguides': hguides}, allow_nan=False)
    dom_prefix = 'densityCurves'
    return f'''
        <div class="card"><h3>Density curves</h3>
        <div id="{dom_prefix}_density" style="width:100%; height:380px;"></div>
        <div id="{dom_prefix}_paths" style="width:100%; height:320px;"></div>
        <p style="font-size:0.85em;color:#666;">Top: normalized edge
        density E(t)/N. Grey dashed vertical lines mark the thresholds of
        the vertical (per-threshold) analysis rows; grey dotted horizontal
        lines mark the density levels of the horizontal (density-matched)
        rows — hover either guide for its row id. Bottom: enumerated path
        count (diagnostic only — hub-inflated). Each dataset is drawn over
        its own domain [w_start, w_star_measured] (light vertical window
        guides); hover any point for the value and whether that threshold
        was materialized; drag to zoom.</p></div>
        <script>
        (function() {{
            const payload = {payload};
            const colorCycle = ['#1d4ed8', '#b45309', '#7c3aed',
                                '#0f766e', '#be185d', '#4d7c0f'];
            const base = {{
                mode: 'lines+markers', connectgaps: false,
                line: {{ width: 2 }}, marker: {{ size: 5 }}
            }};
            const densityTraces = [];
            const pathTraces = [];
            payload.series.forEach(function(s, i) {{
                const color = colorCycle[i % colorCycle.length];
                const densityHover = s.thresholds.map(function(t, j) {{
                    const d = (s.density[j] === null) ? '—' : s.density[j];
                    const mat = s.materialized.indexOf(t) >= 0
                        ? ' · materialized' : '';
                    return s.name + ' · t=' + t + ' · E(t)/N=' + d + mat;
                }});
                const pathHover = s.thresholds.map(function(t, j) {{
                    return s.name + ' · t=' + t + ' · paths=' + s.pathCount[j];
                }});
                densityTraces.push(Object.assign({{}}, base, {{
                    name: s.name, x: s.thresholds, y: s.density,
                    marker: {{ size: 5, color: color }},
                    line: {{ width: 2, color: color }},
                    text: densityHover,
                    hovertemplate: '%{{text}}<extra></extra>'
                }}));
                pathTraces.push(Object.assign({{}}, base, {{
                    name: s.name, x: s.thresholds, y: s.pathCount,
                    marker: {{ size: 5, color: color }},
                    line: {{ width: 2, color: color, dash: 'dot' }},
                    text: pathHover,
                    hovertemplate: '%{{text}}<extra></extra>'
                }}));
            }});
            const xMin = Math.min.apply(null, payload.series.map(
                function(s) {{ return s.thresholds[0]; }}));
            const xMax = Math.max.apply(null, payload.series.map(
                function(s) {{ return s.thresholds[s.thresholds.length - 1]; }}));
            // Horizontal guides (density-matched rows) as traces so they
            // hover; vertical guides (per-threshold rows) as shapes plus an
            // invisible fat hover strip on the density panel.
            const yVals = [];
            payload.series.forEach(function(s) {{
                s.density.forEach(function(v) {{ if (v !== null) yVals.push(v); }});
            }});
            const yLo = Math.min.apply(null, yVals);
            const yHi = Math.max.apply(null, yVals);
            const vShapes = [];
            payload.vguides.forEach(function(g) {{
                vShapes.push({{
                    type: 'line', xref: 'x', yref: 'paper',
                    x0: g.x, x1: g.x, y0: 0, y1: 1,
                    line: {{ color: '#6b7280', width: 1.2, dash: 'dash' }}
                }});
                densityTraces.push({{
                    x: [g.x, g.x], y: [yLo, yHi], mode: 'lines',
                    line: {{ width: 10, color: 'rgba(0,0,0,0)' }},
                    hovertemplate: g.label
                        + ' · Min Synapse Count=' + g.x
                        + '<extra></extra>', showlegend: false
                }});
            }});
            payload.hguides.forEach(function(g) {{
                densityTraces.push({{
                    x: [xMin, xMax], y: [g.y, g.y], mode: 'lines',
                    line: {{ dash: 'dot', color: '#9ca3af', width: 1.5 }},
                    hovertemplate: g.label + ' · ρ=' + g.y
                        + '<extra></extra>', showlegend: false
                }});
            }});
            const shapes = [];
            payload.series.forEach(function(s) {{
                [s.wStart, s.wStar].forEach(function(v) {{
                    if (v === null) return;
                    shapes.push({{
                        type: 'line', xref: 'x', yref: 'paper',
                        x0: v, x1: v, y0: 0, y1: 1,
                        line: {{ color: '#d1d5db', width: 1 }}
                    }});
                }});
            }});
            const legend = {{ orientation: 'h', y: 1.12,
                             x: 0.5, xanchor: 'center' }};
            Plotly.newPlot('{dom_prefix}_density', densityTraces, {{
                margin: {{ l: 70, r: 20, t: 30, b: 40 }},
                xaxis: {{ title: 'Min Synapse Count' }},
                yaxis: {{ title: 'Edge density E(t)/N' }},
                legend: legend, shapes: shapes, hovermode: 'x unified'
            }}, {{ responsive: true }});
            Plotly.newPlot('{dom_prefix}_paths', pathTraces, {{
                margin: {{ l: 70, r: 20, t: 10, b: 40 }},
                xaxis: {{ title: 'Min Synapse Count' }},
                yaxis: {{ title: 'Enumerated paths (diagnostic)' }},
                legend: {{ orientation: 'h', y: -0.3,
                          x: 0.5, xanchor: 'center' }},
                hovermode: 'x unified'
            }}, {{ responsive: true }});
        }})();
        </script>
'''


def _query_resolution_badge(status: str, evidence: str = '') -> str:
    """Coloured badge for one query-resolution status (plan §2.4).

    Identity records are qualified by their evidence state: green when a
    cross-dataset relation confirms the identity, amber when the relation
    names another counterpart (identity kept, counter-evidence shown), grey
    when only the name itself supports it.
    """
    styles = {
        'same_name_identity': ('#15803d', '#f0fdf4', '✓'),
        'taxonomy': ('#1d4ed8', '#eff6ff', 'T'),
        'taxonomy_mapped': ('#1d4ed8', '#eff6ff', 'T→'),
        'mapped': ('#15803d', '#f0fdf4', '→'),
        'bridged': ('#15803d', '#f0fdf4', '⇢'),
        'valid_split': ('#7c3aed', '#f5f3ff', '⑃'),
        'evidence_only': ('#b45309', '#fffbeb', '…'),
        'same_name_fallback': ('#b45309', '#fffbeb', '≈'),
        'conflict': ('#b91c1c', '#fef2f2', '✗'),
        'unmapped': ('#6b7280', '#f9fafb', '∅'),
        'body_id': ('#15803d', '#f0fdf4', '#'),
        'group': ('#15803d', '#f0fdf4', 'G'),
        'pattern': ('#15803d', '#f0fdf4', '*'),
    }
    color, bg, mark = styles.get(status, ('#6b7280', '#f9fafb', ''))
    title = ''
    if status == 'same_name_identity':
        if evidence == 'contradicted':
            color, bg = '#b45309', '#fffbeb'
            title = 'identity kept; curated relation names another ' \
                    'counterpart'
        elif evidence == 'none':
            color, bg = '#6b7280', '#f9fafb'
            title = 'identity backed only by the name itself'
        else:
            title = 'identity confirmed by a cross-dataset relation'
    return (f'<span style="background:{bg};color:{color};'
            f'border:1px solid {color};border-radius:10px;'
            f'padding:1px 8px;font-size:0.85em;white-space:nowrap;" '
            f'title="{html.escape(title)}">{html.escape(mark)} '
            f'{html.escape(status)}</span>')


def _query_resolution_targets(targets: List[str], limit: int = 8) -> str:
    """Render a resolved-target cell, collapsing long member lists."""
    targets = [str(t) for t in (targets or [])]
    if not targets:
        return '—'
    if len(targets) <= limit:
        return html.escape(', '.join(targets))
    shown = ', '.join(targets[:limit])
    more = len(targets) - limit
    return (f'<span title="{html.escape(", ".join(targets))}">'
            f'{html.escape(shown)} … <em>(+{more} more)</em></span>')


def _generate_query_resolution_section(analyzer, dataset_names,
                                       nickname_map) -> str:
    """Standalone input-query resolver section (plan §7E, rev 2)."""
    try:
        records = analyzer.resolve_query_inputs()
    except Exception:
        records = []
    if not records:
        return ''

    same_name = [r for r in records if r.get('status') == 'same_name_fallback']
    conflicts = [r for r in records if r.get('status') == 'conflict']
    contradicted = [r for r in records
                    if r.get('status') == 'same_name_identity'
                    and r.get('evidence') == 'contradicted']
    # Coverage asymmetry: a token that resolved in some datasets but not all.
    by_token: Dict[str, set] = {}
    for r in records:
        by_token.setdefault(r['token'], set())
        if r.get('status') not in ('unmapped',):
            by_token[r['token']].add(r['dataset'])
    partial = {t: ds for t, ds in by_token.items()
               if 0 < len(ds) < len(dataset_names)}

    parts = ['<div id="query-resolution" class="section">',
             '<div class="section-header">🧭 Query Resolution '
             '(input → dataset)</div>',
             '<div class="section-content">',
             '<p style="color: var(--secondary-color);">How each input token '
             'was resolved against each dataset, using the same backend as '
             'the auto type mapper. A native same-name type resolves to '
             'itself first (<code>same_name_identity</code>), annotated by '
             'the cross-dataset relation: <span style="color:#15803d;">'
             'confirmed</span>, <span style="color:#b45309;">contradicted'
             '</span> (identity kept, counter-evidence shown) or grey '
             '(name only). Taxonomy values expand to their member types '
             '(<code>taxonomy</code>) and bridge into the other datasets '
             'through member mapping (<code>taxonomy_mapped</code>). '
             '<code>same_name_fallback</code> — a name echo with no native '
             'membership and no relation — remains the lowest-confidence '
             'pairing and should be reviewed.</p>']

    if same_name:
        tokens = ', '.join(sorted({r['token'] for r in same_name}))
        parts.append(
            '<div style="border-left:6px solid #b45309; background:#fffbeb; '
            'padding:10px 14px; margin:8px 0;">'
            '<strong>⚠️ Low-confidence same-name matches:</strong> '
            f'{html.escape(tokens)} were matched by identical name alone '
            '(not native to that dataset and without mapped evidence). If '
            'these are not genuine cross-dataset equivalents, restrict the '
            'query or supply a mapping.</div>')
    if contradicted:
        def _contra_text(r):
            chain = str(r.get('evidence_chain') or '')
            chain = chain.replace('curated counterpart: ', '')
            ds = nickname_map.get(r.get('dataset'), str(r.get('dataset')))
            return (f"{html.escape(str(r['token']))} vs "
                    f"{html.escape(chain)} ({html.escape(ds)})")
        detail = '; '.join(
            _contra_text(r) for r in sorted(
                contradicted, key=lambda r: (r['token'], r['dataset'])))
        parts.append(
            '<div style="border-left:6px solid #b45309; background:#fffbeb; '
            'padding:10px 14px; margin:8px 0;">'
            '<strong>⚠️ Same-name identities with counter-evidence:</strong> '
            f'{detail}. The query keeps the native name; check whether the '
            'curated counterpart matters for your interpretation.</div>')
    if conflicts:
        tokens = ', '.join(sorted({r['token'] for r in conflicts}))
        parts.append(
            '<div style="border-left:6px solid #b91c1c; background:#fef2f2; '
            'padding:10px 14px; margin:8px 0;">'
            '<strong>❌ Unresolved conflicts:</strong> '
            f'{html.escape(tokens)} have conflicting mappings and no automatic '
            'target.</div>')
    if partial:
        parts.append(
            '<div style="border-left:6px solid #0e7490; background:#ecfeff; '
            'padding:10px 14px; margin:8px 0;">'
            '<strong>ℹ️ Partial coverage:</strong> '
            + '; '.join(f'{html.escape(t)} resolved in '
                        f'{len(ds)}/{len(dataset_names)} datasets'
                        for t, ds in sorted(partial.items()))
            + '. Datasets without the counterpart cannot contribute to that '
            'row.</div>')

    parts.append('<div style="overflow-x:auto;"><table><thead><tr>'
                 '<th>Token</th><th>Role</th><th>Dataset</th><th>Method</th>'
                 '<th>Status</th><th>Evidence</th><th>Confidence</th>'
                 '<th>Resolved</th><th>Note</th></tr></thead><tbody>')
    for r in records:
        parts.append(
            '<tr>'
            f'<td><strong>{html.escape(str(r.get("token")))}</strong></td>'
            f'<td>{html.escape(str(r.get("role")))}</td>'
            f'<td>{html.escape(nickname_map.get(r.get("dataset"), str(r.get("dataset"))))}</td>'
            f'<td>{html.escape(str(r.get("method")))}</td>'
            f'<td>{_query_resolution_badge(str(r.get("status")), str(r.get("evidence") or ""))}</td>'
            f'<td>{html.escape(str(r.get("evidence") or "—"))}</td>'
            f'<td>{html.escape(str(r.get("confidence")))}</td>'
            f'<td>{_query_resolution_targets(r.get("target_types"))}</td>'
            f'<td style="font-size:0.9em;color:#666;">{html.escape(str(r.get("note") or ""))}</td>'
            '</tr>')
    parts.append('</tbody></table></div></div></div>')
    return ''.join(parts)


def _generate_report_header(analyzer, dataset_names: List[str], thresholds: List[int],
                            mode_specific_note: str, nickname_map: Dict[str, str]) -> str:
    """Generate the report header section."""
    from datetime import datetime
    
    params = analyzer.parameters
    thresholds_str = ', '.join(str(t) for t in thresholds)
    datasets_display = ', '.join(nickname_map[d] for d in dataset_names)
    
    # Path enumeration display: in shortest mode a Max Interlayer of 0 means
    # unlimited search depth (discovery stops when all targets are found).
    path_mode_display = getattr(params, 'path_mode', 'all')
    if path_mode_display == 'shortest' and params.max_interlayer <= 0:
        max_interlayer_display = 'unlimited (shortest mode)'
    else:
        max_interlayer_display = str(params.max_interlayer)
    shortest_note = (
        '<p><em>Shortest path mode: results contain only the per-(source, target) '
        'minimum-hop paths (all ties kept); longer alternative routes are excluded.</em></p>'
        if path_mode_display == 'shortest' else ''
    )
    
    hemi_badge = (
        '<p style="margin:6px 0 0 0;"><span style="background:#ede9fe; color:#6d28d9; '
        'border:1px solid #7c3aed; border-radius:10px; padding:2px 10px; font-size:0.9em;">'
        '🧠 Hemisphere-aware run — type names carry _L/_R/_U suffixes</span></p>'
        if getattr(params, 'separate_hemispheres', False) else ''
    )
    
    return f"""
        <header>
            <h1>📊 Cross-Dataset Comparison Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Datasets: {datasets_display}</p>
            <p>Thresholds: {thresholds_str} | Mode: <strong>{params.comparison_mode}</strong> | Path Enumeration: <strong>{path_mode_display}</strong> | Max Interlayer: {max_interlayer_display}</p>
            {hemi_badge}
            {shortest_note}
        </header>
        {mode_specific_note}
    """


def _generate_toc(thresholds: List[int],
                  include_provenance: bool = False,
                  include_type_mapping: bool = False,
                  include_alignment: bool = False,
                  include_auto_density: bool = False) -> str:
    """Generate the shared table of contents.

    Both report modes render the same section set, so the TOC lists every
    rendered section. ``include_provenance`` adds the Custom combination
    mode's ``threshold-provenance`` entry, ``include_type_mapping`` the
    type-mapping entry (rendered only when auto type mapping is on),
    ``include_alignment`` the threshold-alignment entry and
    ``include_auto_density`` the auto density-alignment entry. The
    ``thresholds`` argument is retained for call compatibility and is not
    rendered.
    """
    provenance_entry = ('<li><a href="#threshold-provenance">🎯 Applied '
                        'Thresholds &amp; Bottleneck Provenance</a></li>\n'
                        if include_provenance else '')
    type_mapping_entry = ('<li><a href="#type-mapping">🏷️ Type Mapping'
                          '</a></li>\n' if include_type_mapping else '')
    alignment_entry = ('<li><a href="#threshold-alignment">📐 Threshold '
                       'Alignment</a></li>\n' if include_alignment else '')
    auto_density_entry = ('<li><a href="#auto-density-alignment">📐 Density '
                          'curves (query-scoped)</a></li>\n'
                          if include_auto_density else '')
    return f"""
        <div class="toc">
            <h2>📑 Quick Navigation</h2>
            <ul>
                <li><a href="#summary">📋 Summary & Key Findings</a></li>
                <li><a href="#neuron-counts">🧬 Neuron Counts Comparison</a></li>
                <li><a href="#query-resolution">🧭 Query Resolution</a></li>
                {type_mapping_entry}{provenance_entry}{alignment_entry}{auto_density_entry}<li><a href="#hemisphere-symmetry">🪞 Hemisphere Symmetry</a></li>
                <li><a href="#similarity">🔢 Similarity Matrices</a></li>
                <li><a href="#networks">🕸️ Network Visualizations</a></li>
                <li><a href="#edge-matrices">🔗 Edge Presence Matrices</a></li>
                <li><a href="#path-matrices">🛤️ Path Presence Matrices</a></li>
                <li><a href="#conservation">🏆 Conservation Analysis</a></li>
                <li><a href="#overlap-matrices">🔀 Dataset Overlap Matrices</a></li>
                <li><a href="#statistics">📉 Statistics</a></li>
            </ul>
        </div>
"""


def _generate_summary_section(analyzer, dataset_names: List[str], thresholds: List[int],
                               path_count_data: List[Dict], key_findings: Dict,
                               nickname_map: Dict[str, str]) -> str:
    """Generate summary section with overview and charts."""
    html_parts = []

    # Requested -> applied map so rows/CSVs stay honest about what each
    # threshold actually compared (issue #6).
    try:
        applied_label = analyzer.get_applied_threshold_map()
    except Exception:
        applied_label = {}

    def _applied_txt(t):
        per_ds = applied_label.get(str(t)) or {}
        if not per_ds:
            return ''
        return '/'.join(str((per_ds.get(d) or {}).get('applied'))
                        for d in dataset_names if d in per_ds)
    
    html_parts.append("""
        <div id="summary" class="section">
            <div class="section-header">📋 Summary & Key Findings</div>
            <div class="section-content">
    """)
    html_parts.append("""
                <div class="card">
                    <h3>Key Findings by Threshold</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Threshold</th>
                                <th>Applied</th>
                                <th>Total Edges</th>
                                <th>Common Edges</th>
                                <th>Edge Conservation</th>
                                <th>Total Paths</th>
                                <th>Common Paths</th>
                                <th>Path Conservation</th>
                            </tr>
                        </thead>
                        <tbody>
    """)
    for t in thresholds:
        kf = key_findings.get(t, {})
        total_edges = kf.get('total_edges', 0)
        common_edges = kf.get('common_edges', 0)
        edge_rate = (common_edges / total_edges * 100) if total_edges > 0 else 0
        total_paths = kf.get('total_paths', 0)
        common_paths = kf.get('common_paths', 0)
        path_rate = (common_paths / total_paths * 100) if total_paths > 0 else 0
        applied_txt = _applied_txt(t) or '—'
        
        html_parts.append(f'<tr><td><strong>t={t}</strong></td><td>{applied_txt}</td><td>{total_edges}</td><td>{common_edges}</td><td>{edge_rate:.1f}%</td><td>{total_paths}</td><td>{common_paths}</td><td>{path_rate:.1f}%</td></tr>')
    
    html_parts.append('</tbody></table></div>')
    
    # Edge counts chart
    corrected_data = []
    for d in dataset_names:
        nick = nickname_map[d]
        for t in thresholds:
            aligned = analyzer.get_aligned_data(t)
            count = int((aligned[d] > 0).sum()) if not aligned.empty and d in aligned.columns else 0
            corrected_data.append({
                'dataset': nick, 'threshold': t, 'count': count,
                'applied_threshold': (applied_label.get(str(t), {})
                                      .get(d, {}) or {}).get('applied')})
    
    data_json = json.dumps(corrected_data)
    nicknames_json = json.dumps([nickname_map[d] for d in dataset_names])
    thresholds_json = json.dumps(thresholds)
    
    # Create folder to save used data for debugging/verification
    used_data_dir = os.path.join(analyzer.parameters.full_output_path, 'comparison_report_used_data')
    os.makedirs(used_data_dir, exist_ok=True)

    # Calculate total weights, avg edge ratio (connection_ratio), and avg traversal probability
    total_weight_data = []
    avg_ratio_data = []
    avg_prob_data = []
    
    # Pre-load and save ratio data for each threshold
    ratio_data_cache = {}
    for t in thresholds:
        try:
            ratio_data = analyzer._get_edge_ratio_data_for_threshold(t)
            if not ratio_data.empty:
                ratio_data.to_csv(os.path.join(used_data_dir, f'ratio_data_t{t}.csv'))
                ratio_data_cache[t] = ratio_data
        except Exception as e:
            print(f"[HTML Report] Warning: Failed to get ratio data for threshold {t}: {e}")
    
    for d in dataset_names:
        nick = nickname_map[d]
        for t in thresholds:
            aligned = analyzer.get_aligned_data(t)
            total_w = int(aligned[d].sum()) if not aligned.empty and d in aligned.columns else 0
            applied_val = (applied_label.get(str(t), {}).get(d, {}) or {}).get('applied')
            total_weight_data.append({'dataset': nick, 'threshold': t,
                                      'applied_threshold': applied_val,
                                      'weight': total_w})
            
            # Get connection_ratio data (edge-level ratio = weight / post-synaptic sites)
            # Only average over edges that actually exist in this dataset
            try:
                ratio_data = ratio_data_cache.get(t, pd.DataFrame())
                if not ratio_data.empty and d in ratio_data.columns:
                    dataset_ratios = ratio_data[d]
                    non_zero_ratios = dataset_ratios[dataset_ratios > 0]
                    if len(non_zero_ratios) > 0:
                        avg_ratio = float(non_zero_ratios.mean())
                    else:
                        avg_ratio = 0.0
                    if pd.isna(avg_ratio):
                        avg_ratio = 0.0
                else:
                    avg_ratio = 0.0
            except Exception:
                avg_ratio = 0.0
            avg_ratio_data.append({'dataset': nick, 'threshold': t,
                                   'applied_threshold': applied_val,
                                   'ratio': avg_ratio})
            
            # Get traversal probability data if available
            # Only average over paths that actually exist in this dataset (ignore 0s from other datasets' paths)
            try:
                prob_data = analyzer._get_prob_data_for_threshold(t)
                if not prob_data.empty and d in prob_data.columns:
                    # Only include paths where this dataset has non-zero probability
                    # This ensures we don't dilute the average with 0s from other datasets' paths
                    dataset_probs = prob_data[d]
                    non_zero_probs = dataset_probs[dataset_probs > 0]
                    if len(non_zero_probs) > 0:
                        avg_prob = float(non_zero_probs.mean())
                    else:
                        avg_prob = 0.0
                    if pd.isna(avg_prob):
                        avg_prob = 0.0
                else:
                    avg_prob = 0.0
            except Exception as e:
                # The value lands in used_data/avg_prob_data.csv; a silent
                # 0.0 would read as "no probability" rather than "failed".
                print(f"[HTML Report] Warning: Failed to get probability "
                      f"data for {nick} at threshold {t}: {e}")
                avg_prob = 0.0
            avg_prob_data.append({'dataset': nick, 'threshold': t, 'prob': avg_prob})
    
    # Save all chart data to files for verification
    pd.DataFrame(corrected_data).to_csv(os.path.join(used_data_dir, 'edge_count_data.csv'), index=False)
    pd.DataFrame(total_weight_data).to_csv(os.path.join(used_data_dir, 'total_weight_data.csv'), index=False)
    pd.DataFrame(avg_ratio_data).to_csv(os.path.join(used_data_dir, 'avg_ratio_data.csv'), index=False)
    pd.DataFrame(avg_prob_data).to_csv(os.path.join(used_data_dir, 'avg_prob_data.csv'), index=False)
    
    weight_data_json = json.dumps(total_weight_data)
    ratio_data_json = json.dumps(avg_ratio_data)
    prob_data_json = json.dumps(avg_prob_data)
    
    html_parts.append(f"""
                <div class="card">
                    <h3>Edge Counts Across All Thresholds</h3>
                    <div id="edgeCountChart" class="chart-container"></div>
                </div>
                <script>
                    (function() {{
                        const data = {data_json};
                        const datasets = {nicknames_json};
                        const thresholds = {thresholds_json};
                        const traces = datasets.map(ds => {{
                            const yVals = thresholds.map(t => {{
                                const item = data.find(d => d.dataset === ds && d.threshold === t);
                                return item ? item.count : 0;
                            }});
                            return {{
                                name: ds,
                                x: thresholds.map(t => 'T=' + t),
                                y: yVals,
                                type: 'bar',
                                text: yVals.map(v => v.toString()),
                                textposition: 'outside'
                            }};
                        }});
                        const allYVals = traces.flatMap(t => t.y);
                        const maxY = Math.max(...allYVals);
                        Plotly.newPlot('edgeCountChart', traces, {{
                            barmode: 'group',
                            xaxis: {{ title: 'Threshold' }},
                            yaxis: {{ title: 'Edge Count', range: [0, maxY * 1.15] }},
                            legend: {{ orientation: 'h', y: -0.15 }}
                        }}, {{responsive: true}});
                    }})();
                </script>
                
                <div class="card">
                    <h3>Total Edge Weight Across All Thresholds</h3>
                    <div id="totalWeightChart" class="chart-container"></div>
                </div>
                <script>
                    (function() {{
                        const data = {weight_data_json};
                        const datasets = {nicknames_json};
                        const thresholds = {thresholds_json};
                        const traces = datasets.map(ds => {{
                            const yVals = thresholds.map(t => {{
                                const item = data.find(d => d.dataset === ds && d.threshold === t);
                                return item ? item.weight : 0;
                            }});
                            return {{
                                name: ds,
                                x: thresholds.map(t => 'T=' + t),
                                y: yVals,
                                type: 'bar',
                                text: yVals.map(v => v.toString()),
                                textposition: 'outside'
                            }};
                        }});
                        const allYVals = traces.flatMap(t => t.y);
                        const maxY = Math.max(...allYVals);
                        Plotly.newPlot('totalWeightChart', traces, {{
                            barmode: 'group',
                            xaxis: {{ title: 'Threshold' }},
                            yaxis: {{ title: 'Total Edge Weight (Synapse Count)', range: [0, maxY * 1.15] }},
                            legend: {{ orientation: 'h', y: -0.15 }}
                        }}, {{responsive: true}});
                    }})();
                </script>
                
                <div class="card">
                    <h3>Average Connection Ratio Across All Thresholds</h3>
                    <div id="avgRatioChart" class="chart-container"></div>
                </div>
                <script>
                    (function() {{
                        const data = {ratio_data_json};
                        const datasets = {nicknames_json};
                        const thresholds = {thresholds_json};
                        const traces = datasets.map(ds => {{
                            const yVals = thresholds.map(t => {{
                                const item = data.find(d => d.dataset === ds && d.threshold === t);
                                return item ? item.ratio : 0;
                            }});
                            return {{
                                name: ds,
                                x: thresholds.map(t => 'T=' + t),
                                y: yVals,
                                type: 'bar',
                                text: yVals.map(v => v.toFixed(4)),
                                textposition: 'outside'
                            }};
                        }});
                        const allYVals = traces.flatMap(t => t.y);
                        const maxY = Math.max(...allYVals);
                        Plotly.newPlot('avgRatioChart', traces, {{
                            barmode: 'group',
                            xaxis: {{ title: 'Threshold' }},
                            yaxis: {{ title: 'Avg Connection Ratio (w_ij / W_j)', range: [0, maxY * 1.15] }},
                            legend: {{ orientation: 'h', y: -0.15 }}
                        }}, {{responsive: true}});
                    }})();
                </script>
                
                <div class="card">
                    <h3>Average Path Traversal Probability</h3>
                    <div id="avgProbChart" class="chart-container"></div>
                </div>
                <script>
                    (function() {{
                        const data = {prob_data_json};
                        const datasets = {nicknames_json};
                        const thresholds = {thresholds_json};
                        const traces = datasets.map(ds => {{
                            const yVals = thresholds.map(t => {{
                                const item = data.find(d => d.dataset === ds && d.threshold === t);
                                return item ? item.prob : 0;
                            }});
                            return {{
                                name: ds,
                                x: thresholds.map(t => 'T=' + t),
                                y: yVals,
                                type: 'bar',
                                text: yVals.map(v => v.toFixed(4)),
                                textposition: 'outside'
                            }};
                        }});
                        const allYVals = traces.flatMap(t => t.y);
                        const maxY = Math.max(...allYVals);
                        Plotly.newPlot('avgProbChart', traces, {{
                            barmode: 'group',
                            xaxis: {{ title: 'Threshold' }},
                            yaxis: {{ title: 'Avg Traversal Probability', range: [0, maxY * 1.15] }},
                            legend: {{ orientation: 'h', y: -0.15 }}
                        }}, {{responsive: true}});
                    }})();
                </script>
""")
    
    html_parts.append('</div></div>')
    return ''.join(html_parts)


def _generate_neuron_counts_section(analyzer, dataset_names: List[str], 
                                     nickname_map: Dict[str, str]) -> str:
    """Generate neuron counts comparison section with type mapping aggregation."""
    # Try to import type mapper for cross-dataset type name merging
    try:
        from .cross_dataset_type_mapper import get_type_mapper
        type_mapper = get_type_mapper()
        type_mapper.load()
        has_type_mapper = True
    except Exception:
        has_type_mapper = False
        type_mapper = None
    
    html_parts = []
    
    html_parts.append("""
        <div id="neuron-counts" class="section">
            <div class="section-header">🧬 Neuron Counts Comparison</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Comparison of source/target neuron existence across datasets. Shows which neuron types are present in each dataset.
                </p>
""")
    
    # Get neuron count data from analyzer
    summary_df = getattr(analyzer, '_neuron_counts_summary', None)
    type_df = getattr(analyzer, '_neuron_type_counts', None)
    group_df = getattr(analyzer, '_neuron_group_counts', None)
    
    # Summary table with bar chart
    if summary_df is not None and not summary_df.empty:
        html_parts.append('<div class="card"><h3>Summary: Total Neuron Counts</h3>')
        html_parts.append('<table><thead><tr><th>Dataset</th><th>Source Neurons</th><th>Target Neurons</th><th>Source Types</th><th>Target Types</th></tr></thead><tbody>')
        
        for _, row in summary_df.iterrows():
            html_parts.append(f'''<tr>
                <td><strong>{row.get("dataset", "")}</strong></td>
                <td>{row.get("source_count", 0)}</td>
                <td>{row.get("target_count", 0)}</td>
                <td>{row.get("source_types", 0)}</td>
                <td>{row.get("target_types", 0)}</td>
            </tr>''')
        
        html_parts.append('</tbody></table>')
        
        # Add bar chart for neuron counts - Grouped by Source/Target
        chart_data = []
        for _, row in summary_df.iterrows():
            chart_data.append({
                'dataset': row.get('dataset', ''),
                'source': row.get('source_count', 0),
                'target': row.get('target_count', 0)
            })
        
        chart_json = json.dumps(chart_data)
        html_parts.append(f'''
            <div id="neuronCountChart" class="chart-container" style="height: 300px;"></div>
            <script>
                (function() {{
                    const data = {chart_json};
                    
                    // Group by Source/Target
                    // Traces are datasets
                    const traces = [];
                    data.forEach(d => {{
                        traces.push({{
                            name: d.dataset,
                            x: ['Source Neurons', 'Target Neurons'],
                            y: [d.source, d.target],
                            type: 'bar'
                        }});
                    }});
                    
                    Plotly.newPlot('neuronCountChart', traces, {{
                        barmode: 'group',
                        xaxis: {{ title: 'Role' }},
                        yaxis: {{ title: 'Neuron Count' }},
                        legend: {{ orientation: 'v' }}
                    }}, {{responsive: true}});
                }})();
            </script>
        ''')
        html_parts.append('</div>')
    
    # Combined Type counts table and chart
    if type_df is not None and not type_df.empty:
        # Get all relevant columns
        source_cols = [c for c in type_df.columns if c not in ['type', 'role'] and 'source' in c.lower()]
        target_cols = [c for c in type_df.columns if c not in ['type', 'role'] and 'target' in c.lower()]
        all_cols = source_cols + target_cols
        
        # Grouping key: the grouped DISPLAY name (see the aggregation loop
        # below).  ``get_display_type_name`` routes through the mapper's
        # display-name alternatives, so licensed renames still coalesce
        # while a CONFLICTED type keeps a dataset-scoped label.
        def get_display_type_name(type_name: str) -> str:
            if not has_type_mapper or type_mapper is None:
                return type_name
            try:
                return type_mapper.get_display_name(type_name, dataset_names)
            except Exception:
                return type_name
        
        # Aggregate type counts by canonical name
        canonical_counts = {}  # {canonical_type: {col: total_count}}
        dataset_specific_types = {}  # {canonical_type: {col: set(original_types)}}
        canonical_to_display = {}  # {canonical_type: display_name_with_alternatives}

        # Group by the DISPLAY name, not the per-namespace canonical key:
        # ``canonical_merge_key`` canonicalizes within the auto-detected
        # namespace, so the MCNS and BANC members of ONE cell type can carry
        # different canonical keys while ``get_display_name`` collapses them
        # to the same label — those must render as a single row
        # (e.g. CL125(APDN3/LMTe01) with both datasets' counts).
        # B6: with a query-anchored merge policy, a governed type groups
        # under its POLICY LABEL (no scoped cosmetic form) — one group =
        # one count row, consistent with the merged frames.
        merge_policy = None
        try:
            merge_policy = analyzer._merge_policy_or_none()
        except Exception:
            merge_policy = None
        # Hemisphere-suffix awareness (plan R1-c): with
        # separate_hemispheres the raw types carry _L/_R/_U suffixes while
        # the policy keys base names — group them under
        # `<label><suffix>` so L and R stay distinct count rows that
        # still merge across datasets.
        hemi_aware = bool(getattr(
            getattr(analyzer, 'parameters', None), 'separate_hemispheres',
            False))
        raw_display: dict = {}
        for _, row in type_df.iterrows():
            orig_type = row.get('type', '')
            if orig_type:
                if merge_policy is not None:
                    policy_label = merge_policy.label_for_name(orig_type)
                    if policy_label is None and hemi_aware:
                        base_t, suffix = split_hemi_suffix(str(orig_type))
                        if suffix:
                            policy_label = merge_policy.label_for_name(base_t)
                            if policy_label is not None:
                                raw_display[orig_type] = (
                                    f'{policy_label}{suffix}')
                                continue
                    if policy_label is not None:
                        raw_display[orig_type] = policy_label
                        continue
                raw_display[orig_type] = get_display_type_name(orig_type)

        for _, row in type_df.iterrows():
            orig_type = row.get('type', '')
            if not orig_type:
                continue

            # The display name contains the most info (alternatives) and is
            # shared by every namespace variant of the same cell type.
            group_key = raw_display.get(orig_type, orig_type)
            canonical_to_display[group_key] = group_key

            # Initialize group entry if needed
            if group_key not in canonical_counts:
                canonical_counts[group_key] = {col: 0 for col in all_cols}
                dataset_specific_types[group_key] = {col: set() for col in all_cols}

            # Aggregate counts
            for col in all_cols:
                val = row.get(col, 0)
                if pd.notna(val) and val > 0:
                    canonical_counts[group_key][col] += int(val)
                    dataset_specific_types[group_key][col].add(orig_type)
        
        # Filter out entries with zero total count
        canonical_counts = {
            k: v for k, v in canonical_counts.items() 
            if sum(v.values()) > 0
        }
        
        if canonical_counts:
            html_parts.append('<div class="card"><h3>Neuron Counts by Type (Source & Target)</h3>')
            html_parts.append('<p style="color: var(--secondary-color); font-size: 0.9em; margin-bottom: 10px;">Neuron counts grouped by canonical type name (by query-anchored merge-policy group label when a policy governs the run). Dataset-specific type names shown in parentheses if they differ.</p>')
            
            # Sort by total count
            sorted_types = sorted(
                canonical_counts.keys(),
                key=lambda t: sum(canonical_counts[t].values()),
                reverse=True
            )

            def _role_order(cols):
                """Types ordered by THIS role's totals, EXCLUDING types with
                a zero total for the role (plan F-1, user refinement 2nd
                notice): the source chart/table shows only types actually
                present as sources; the target chart/table only types
                present as targets."""
                def total(t):
                    return sum(canonical_counts[t].get(c, 0) for c in cols)
                present = [t for t in canonical_counts if total(t) > 0]
                return sorted(present, key=total, reverse=True)

            source_sorted_types = _role_order(source_cols)
            target_sorted_types = _role_order(target_cols)
            # Top 50 per role for the charts; tables show all in role order.
            top_source_types = source_sorted_types[:50]
            top_target_types = target_sorted_types[:50]

            # Split Source / Target: two side-by-side bar charts and two
            # tables (user refinement 2026-09-15). Colors are assigned per
            # DATASET index so both charts (and the shared chip legend)
            # agree without a per-chart legend (issue 5b).
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

            def _build_traces(cols, types_list):
                traces = []
                for i, col in enumerate(cols):
                    display_col = (col.replace('_source', ' (Source)')
                                   .replace('_target', ' (Target)')
                                   .replace('_v', ':v').replace('_', ' '))
                    traces.append({
                        'name': display_col,
                        'x': [canonical_to_display.get(t, t)
                              for t in types_list],
                        'y': [canonical_counts[t].get(col, 0)
                              for t in types_list],
                        'type': 'bar',
                        'marker': {'color': colors[i % len(colors)]},
                        'showlegend': False,
                    })
                return traces

            def _row_cells(canonical, col):
                val = canonical_counts[canonical].get(col, 0)
                if val == 0:
                    return '<td class="absent">-</td>'
                orig_types = dataset_specific_types[canonical].get(col, set())
                cell_content = f"{int(val)}"
                if len(orig_types) == 1:
                    specific_type = list(orig_types)[0]
                    specific_base = (specific_type.replace('_L', '')
                                     .replace('_R', ''))
                    canonical_base = canonical.replace('_L', '').replace('_R', '')
                    if specific_base != canonical_base \
                            and specific_type not in canonical:
                        cell_content += (
                            f" <span style='font-size:0.8em; color:gray'>"
                            f"({specific_type})</span>")
                return f'<td class="present">{cell_content}</td>'

            def _table_html(cols, role_label, ordered_types):
                header_cells = ''.join(
                    '<th>' + html_module.escape(
                        col.replace('_source', ' (Source)')
                        .replace('_target', ' (Target)')
                        .replace('_v', ':v').replace('_', ' ')) + '</th>'
                    for col in cols)
                body_rows = []
                for canonical in ordered_types:
                    display_name = canonical_to_display.get(canonical, canonical)
                    member_note = ''
                    if type_df is not None and 'group_members' in type_df.columns:
                        match = type_df.loc[
                            type_df['type'] == canonical, 'group_members']
                        if (not match.empty and pd.notna(match.iloc[0])
                                and str(match.iloc[0]).strip()):
                            member_note = (
                                '<br><span style="color:var(--secondary-color);'
                                'font-size:0.8em;">'
                                + html_module.escape(str(match.iloc[0]))
                                + '</span>')
                    cells = ''.join(_row_cells(canonical, col) for col in cols)
                    body_rows.append(
                        '<tr><td><strong>'
                        + html_module.escape(str(display_name))
                        + '</strong>' + member_note + '</td>' + cells + '</tr>')
                return ('<div class="sticky-table-container" '
                        'style="overflow-x: auto; max-height: none;">'
                        '<table class="presence-table"><thead><tr>'
                        '<th>Type (' + html_module.escape(role_label) + ')</th>'
                        + header_cells + '</tr></thead><tbody>'
                        + ''.join(body_rows) + '</tbody></table></div>')

            source_traces = _build_traces(source_cols, top_source_types)
            target_traces = _build_traces(target_cols, top_target_types)
            chart_payload = json.dumps(
                {'source': source_traces, 'target': target_traces})

            # One shared chip legend for the whole card (issue 5b): the
            # color maps to the dataset in BOTH charts, so per-chart Plotly
            # legends (which used to cover the bars) are dropped.
            chip_items = []
            legend_cols = source_cols
            for i, col in enumerate(legend_cols):
                nick = (col.replace('_source', '').replace('_target', '')
                        .replace('_v', ':v').replace('_', ' '))
                chip_items.append(
                    f'<span style="display:inline-flex; align-items:center; '
                    f'gap:5px; margin-right:16px; font-size:0.85em;">'
                    f'<span style="width:12px; height:12px; border-radius:2px; '
                    f'display:inline-block; background:'
                    f'{colors[i % len(colors)]};"></span>'
                    f'{html_module.escape(nick)}</span>')
            chip_legend = (
                '<div style="display:flex; flex-wrap:wrap; align-items:center; '
                'margin: 0 0 10px 0;">'
                '<span style="font-size:0.85em; color:var(--secondary-color); '
                'margin-right:8px;">Dataset (shared legend):</span>'
                + ''.join(chip_items) + '</div>')

            html_parts.append(
                chip_legend
                + '<div class="counts-split">'
                '<style>'
                '.counts-split { display: flex; flex-wrap: wrap; gap: 30px; }'
                '.counts-split-col { flex: 1 1 44%; min-width: 0; }'
                '.counts-split-col h4 { color: #1d4ed8; margin: 4px 0 10px 0; }'
                '</style>'
                + '<div class="counts-split-col"><h4>Source neuron counts '
                'by type</h4><div id="combinedTypeChartSource" '
                'class="chart-container" style="height: 420px;"></div>'
                + _table_html(source_cols, 'Source', source_sorted_types) + '</div>'
                '<div class="counts-split-col"><h4>Target neuron counts '
                'by type</h4><div id="combinedTypeChartTarget" '
                'class="chart-container" style="height: 420px;"></div>'
                + _table_html(target_cols, 'Target', target_sorted_types) + '</div>'
                '</div>')
            html_parts.append("""
            <script>
            (function() {
                const payload = __PAYLOAD__;
                Plotly.newPlot('combinedTypeChartSource', payload.source, {
                    barmode: 'group',
                    xaxis: { title: 'Neuron Type', tickangle: -45, automargin: true },
                    yaxis: { title: 'Count' },
                    showlegend: false,
                    margin: { b: 150 }
                }, { responsive: true });
                Plotly.newPlot('combinedTypeChartTarget', payload.target, {
                    barmode: 'group',
                    xaxis: { title: 'Neuron Type', tickangle: -45, automargin: true },
                    yaxis: { title: 'Count' },
                    showlegend: false,
                    margin: { b: 150 }
                }, { responsive: true });
            })();
            </script>
            """.replace('__PAYLOAD__', chart_payload))
    
    # Group counts table (if custom groups exist)
    if group_df is not None and not group_df.empty:
        html_parts.append('<div class="card"><h3>Neuron Counts by Custom Group</h3>')
        html_parts.append('<div class="sticky-table-container" style="overflow-x: auto;"><table class="presence-table"><thead><tr><th>Custom Group</th>')
        
        data_cols = [c for c in group_df.columns if c not in ['custom_group', 'role']]
        for col in data_cols:
            display_col = col.replace('_v', ':v').replace('_', ' ')
            html_parts.append(f'<th>{display_col}</th>')
        html_parts.append('</tr></thead><tbody>')
        
        for _, row in group_df.iterrows():
            html_parts.append(f'<tr><td><strong>{row.get("custom_group", "")}</strong></td>')
            for col in data_cols:
                val = row.get(col, 0)
                if pd.isna(val) or val == 0:
                    html_parts.append('<td class="absent">-</td>')
                else:
                    html_parts.append(f'<td class="present">{int(val)}</td>')
            html_parts.append('</tr>')
        
        html_parts.append('</tbody></table></div></div>')
    
    # If no data available, show message
    if (summary_df is None or summary_df.empty) and (type_df is None or type_df.empty):
        html_parts.append('''
            <div class="card">
                <p style="color: var(--secondary-color);">
                    ℹ️ Neuron count data not available. Run export_results() to generate this comparison.
                </p>
            </div>
        ''')
    
    html_parts.append('</div></div>')
    return ''.join(html_parts)


def _similarity_matrices_from_frame(similarities: pd.DataFrame,
                                    available: List[str]) -> Dict[str, list]:
    """Build symmetric per-metric matrices from pairwise similarity rows.

    v2.2 (plan-similarity-matrix-schema-v2): the four representatives are
    Jaccard + Cosine (edge level), Path Jaccard (path level) and
    NetSimile-lite (graph level). Shared by the Standard per-threshold
    section and the Custom per-query section so both render the identical
    four-panel set.
    """
    n = len(available)
    jaccard = [[1.0 if i == j else None for j in range(n)] for i in range(n)]
    cosine_sim = [[1.0 if i == j else None for j in range(n)] for i in range(n)]
    path_jac = [[1.0 if i == j else None for j in range(n)] for i in range(n)]
    netsimile = [[1.0 if i == j else None for j in range(n)] for i in range(n)]
    if similarities is not None and not similarities.empty:
        for _, row in similarities.iterrows():
            d1, d2 = row['dataset_1'], row['dataset_2']
            if d1 in available and d2 in available:
                i1, i2 = available.index(d1), available.index(d2)
                jac = row.get('jaccard_similarity', None)
                cos_val = row.get('cosine_similarity', None)
                pj_val = row.get('path_jaccard_similarity', None)
                ns_val = row.get('netsimile_similarity', None)
                if pd.isna(jac): jac = None
                if pd.isna(cos_val): cos_val = None
                if pd.isna(pj_val): pj_val = None
                if pd.isna(ns_val): ns_val = None
                jaccard[i1][i2] = jaccard[i2][i1] = jac
                cosine_sim[i1][i2] = cosine_sim[i2][i1] = cos_val
                path_jac[i1][i2] = path_jac[i2][i1] = pj_val
                netsimile[i1][i2] = netsimile[i2][i1] = ns_val
    return {
        'jaccard': jaccard,
        'cosine': cosine_sim,
        'path_jaccard': path_jac,
        'netsimile': netsimile,
    }


def _similarity_detail_table(similarities: pd.DataFrame,
                             available: List[str],
                             nickname_map: Dict[str, str]) -> str:
    """Per-pair detail metrics for the v2.2 panel (plan §3.1): coverage,
    edge top-20, guarded Spearman (+ shared count), path diagnostics and
    strength-W1. One row per dataset pair; — for undefined values."""
    if similarities is None or similarities.empty:
        return ''
    rows = []
    for _, row in similarities.iterrows():
        d1, d2 = row.get('dataset_1'), row.get('dataset_2')
        if d1 not in available or d2 not in available:
            continue
        def fmt(value, nd=3):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return '—'
            return '—' if pd.isna(value) else f'{value:.{nd}f}'
        nick = lambda d: nickname_map.get(d, d)
        rows.append(
            f'<tr><td>{html.escape(nick(d1))} ↔ {html.escape(nick(d2))}</td>'
            f'<td>{fmt(row.get("coverage_min"))}</td>'
            f'<td>{fmt(row.get("top20_overlap"))}</td>'
            f'<td>{fmt(row.get("spearman_rank_correlation"))} '
            f'({fmt(row.get("common_edges"), 0)})</td>'
            f'<td>{fmt(row.get("path_jaccard_similarity"))}</td>'
            f'<td>{fmt(row.get("path_top20_overlap"))}</td>'
            f'<td>{fmt(row.get("hop_profile_w1"))}</td>'
            f'<td>{fmt(row.get("netsimile_similarity"))}</td>'
            f'<td>{fmt(row.get("strength_w1_out"))} / '
            f'{fmt(row.get("strength_w1_in"))}</td>'
            '</tr>')
    if not rows:
        return ''
    return (
        '<div class="sticky-table-container" style="overflow-x: auto; margin-top: 8px;">'
        '<table style="font-size:0.85em;"><thead><tr>'
        '<th>Pair</th><th>Coverage (min)</th><th>Edge top-20</th>'
        '<th>Spearman (shared, ≥10)</th><th>Path Jaccard</th>'
        '<th>Path top-20</th><th>Hop W1</th><th>NetSimile</th>'
        '<th>Strength W1 (out/in)</th>'
        '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')


def _similarity_heatmap_card(dom_key: str, title: str, labels: List[str],
                             matrices: Dict[str, list]) -> str:
    """Render the v2.2 four-representative similarity heatmap card.

    ``dom_key`` must be a DOM-safe string unique within the report (the
    Standard mode passes the scalar threshold, Custom mode the safe query
    slug). Panels are colored by comparison LEVEL (plan-similarity-matrix-
    schema-v2 §3.1.1): edge 🔷 blue, path 🟣 violet, graph 🔶 amber; all
    four run the green [0,1] scale and the per-level captions replace the
    old all-edge/set-based legend. Shared verbatim by both modes.
    """
    cell_size = 50
    n = len(labels)
    chart_size = min(n * cell_size + 80, 200)
    num_metrics = 4
    max_width_pct = f"{100 // num_metrics}%"
    jaccard = matrices['jaccard']
    cosine_sim = matrices['cosine']
    path_jac = matrices['path_jaccard']
    net_sim = matrices['netsimile']
    return f"""
                <div class="card">
                    <h3>{html.escape(title)}</h3>

                    <!-- Representatives in one row, colored by level -->
                    <div style="display: flex; flex-wrap: nowrap; gap: 10px; overflow-x: auto; padding: 8px 0;">
                        <!-- EDGE LEVEL: Jaccard (presence over the union) -->
                        <div style="flex: 1; min-width: {chart_size}px; max-width: {max_width_pct}; background: #eff6ff; border-radius: 6px; padding: 8px;">
                            <h5 style="font-size: 10px; margin: 0 0 4px 0; color: #1e40af; text-align: center;">🔷 Jaccard</h5>
                            <div id="jaccard_{dom_key}" style="width: 100%; height: {chart_size}px;"></div>
                        </div>
                        <!-- EDGE LEVEL: Cosine (weights over the union) -->
                        <div style="flex: 1; min-width: {chart_size}px; max-width: {max_width_pct}; background: #eff6ff; border-radius: 6px; padding: 8px;">
                            <h5 style="font-size: 10px; margin: 0 0 4px 0; color: #1e40af; text-align: center;">🔷 Cosine</h5>
                            <div id="cosine_{dom_key}" style="width: 100%; height: {chart_size}px;"></div>
                        </div>
                        <!-- PATH LEVEL: Path Jaccard (canonical path union) -->
                        <div style="flex: 1; min-width: {chart_size}px; max-width: {max_width_pct}; background: #f5f3ff; border-radius: 6px; padding: 8px;">
                            <h5 style="font-size: 10px; margin: 0 0 4px 0; color: #5b21b6; text-align: center;">🟣 Path Jaccard</h5>
                            <div id="path_jaccard_{dom_key}" style="width: 100%; height: {chart_size}px;"></div>
                        </div>
                        <!-- GRAPH LEVEL: NetSimile-lite (alignment-free) -->
                        <div style="flex: 1; min-width: {chart_size}px; max-width: {max_width_pct}; background: #fef3c7; border-radius: 6px; padding: 8px;">
                            <h5 style="font-size: 10px; margin: 0 0 4px 0; color: #92400e; text-align: center;">🔶 NetSimile</h5>
                            <div id="netsimile_{dom_key}" style="width: 100%; height: {chart_size}px;"></div>
                        </div>
                    </div>
                    <p style="font-size: 0.75em; color: #64748b; margin-top: 8px; text-align: center;">
                        🔷 <strong>Edge level</strong> — Jaccard: presence over the edge union · Cosine: weights over the union (0 for missing)<br>
                        🟣 <strong>Path level</strong> — Path Jaccard: presence over the canonical path union (in-memory; N/A below 5 paths per side)<br>
                        🔶 <strong>Graph level</strong> — NetSimile-lite: alignment-free node-feature signature (no edge support)
                    </p>
                </div>
                <script>
                    (function() {{
                        const labels = {json.dumps(labels)};
                        const jaccard = {json.dumps(jaccard)};
                        const cosineSim = {json.dumps(cosine_sim)};
                        const pathJac = {json.dumps(path_jac)};
                        const netSim = {json.dumps(net_sim)};
                        const layout = {{
                            margin: {{ l: 45, r: 10, t: 10, b: 45 }},
                            xaxis: {{ tickangle: -45, scaleanchor: 'y', constrain: 'domain', tickfont: {{size: 8}} }},
                            yaxis: {{ autorange: 'reversed', constrain: 'domain', tickfont: {{size: 8}} }}
                        }};
                        // Annotation function for [0, 1] metrics
                        const makeAnnotations = (data, labels) => data.flatMap((row, i) =>
                            row.map((val, j) => ({{
                                x: labels[j], y: labels[i],
                                text: val === null ? 'N/A' : val.toFixed(2),
                                showarrow: false,
                                font: {{ color: (val === null || val > 0.5) ? 'white' : 'black', size: 10 }}
                            }})));
                        // Use consistent green colorscale for every
                        // representative: higher value = darker green
                        const greenScale = [[0, '#ffffff'], [0.3, '#c6efce'], [0.6, '#22c55e'], [1, '#166534']];
                        // EDGE LEVEL (blue cards)
                        Plotly.newPlot('jaccard_{dom_key}', [{{
                            z: jaccard, x: labels, y: labels, type: 'heatmap',
                            colorscale: greenScale, zmin: 0, zmax: 1, showscale: false
                        }}], {{...layout, annotations: makeAnnotations(jaccard, labels)}}, {{responsive: true}});
                        Plotly.newPlot('cosine_{dom_key}', [{{
                            z: cosineSim, x: labels, y: labels, type: 'heatmap',
                            colorscale: greenScale, zmin: 0, zmax: 1, showscale: false
                        }}], {{...layout, annotations: makeAnnotations(cosineSim, labels)}}, {{responsive: true}});
                        // PATH LEVEL (violet card)
                        Plotly.newPlot('path_jaccard_{dom_key}', [{{
                            z: pathJac, x: labels, y: labels, type: 'heatmap',
                            colorscale: greenScale, zmin: 0, zmax: 1, showscale: false
                        }}], {{...layout, annotations: makeAnnotations(pathJac, labels)}}, {{responsive: true}});
                        // GRAPH LEVEL (amber card)
                        Plotly.newPlot('netsimile_{dom_key}', [{{
                            z: netSim, x: labels, y: labels, type: 'heatmap',
                            colorscale: greenScale, zmin: 0, zmax: 1, showscale: false
                        }}], {{...layout, annotations: makeAnnotations(netSim, labels)}}, {{responsive: true}});
                    }})();
                </script>
"""


def _generate_similarity_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                  nickname_map: Dict[str, str]) -> str:
    """Generate similarity matrices section with square cell heatmaps for all metrics."""
    from .metrics import ComparisonMetrics
    metrics = ComparisonMetrics()
    
    html_parts = []
    html_parts.append("""
        <div id="similarity" class="section">
            <div class="section-header">🔢 Similarity Matrices</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Pairwise similarity between datasets, presented at three levels (card colors = level):<br>
                    <strong>🔷 Edge level</strong> — <strong>Jaccard</strong> [0, 1]: binary edge <em>presence overlap</em> |A∩B|/|A∪B| ·
                    <strong>Cosine</strong> [0, 1]: <em>directional similarity</em> of weight vectors over the union (0 for missing)<br>
                    <strong>🟣 Path level</strong> — <strong>Path Jaccard</strong> [0, 1]: overlap of the canonical <em>multi-hop path sets</em>
                    (N/A below 5 paths per side)<br>
                    <strong>🔶 Graph level</strong> — <strong>NetSimile-lite</strong> [0, 1]: alignment-free comparison of node-feature
                    distributions (degree/strength profile) — no type correspondence needed<br>
                    Detail metrics per pair (coverage, edge top-20, guarded Spearman ≥10 shared, path top-20, hop/strength W1) sit in the
                    table under each card; legacy columns (Edge Rank, Path Rank, Pearson, RV, Ruzicka) remain in the CSV exports only.
                </p>
""")
    
    # Note: Connectivity Profile Similarity is shown in the separate connectivity_profile_comparison.html
    # report since it uses different methodology (graph-based rank correlation vs threshold-sensitive edge comparison)
    
    for threshold in thresholds:
        aligned = analyzer.get_aligned_data(threshold)
        
        # Always include ALL datasets, even if they have no connections at this threshold
        # This ensures consistent matrix dimensions across thresholds
        available = dataset_names  # Use all datasets, not just those with data
        
        if len(available) < 2:
            continue
        
        labels = [nickname_map[d] for d in available]
        
        # Ensure aligned data has columns for all datasets (fill missing with 0)
        for d in dataset_names:
            if d not in aligned.columns:
                aligned[d] = 0
        
        # Get path data for this threshold (for path rank correlation)
        path_data = None
        if hasattr(analyzer, '_get_path_data_for_threshold'):
            try:
                path_data = analyzer._get_path_data_for_threshold(threshold)
            except Exception:
                pass
        
        similarities = metrics.calculate_all_pairwise_similarities(
            aligned, dataset_names, threshold=1, include_advanced_metrics=True, path_data=path_data
        )
        
        # Save similarity data to CSV (inside comparison_results folder)
        try:
            full_output_path = getattr(analyzer.parameters, 'full_output_path', None)
            if full_output_path:
                csv_dir = os.path.join(full_output_path, 'similarity_matrices')
                os.makedirs(csv_dir, exist_ok=True)
                # Add threshold to the DataFrame before saving
                similarities_with_threshold = similarities.copy()
                similarities_with_threshold['threshold'] = threshold
                similarities_with_threshold.to_csv(os.path.join(csv_dir, f'similarity_threshold_{threshold}.csv'), index=False)
        except Exception:
            pass
        
        # Calculate cell size for square cells - smaller to fit 4 in a row
        # (handled inside the shared heatmap card helper)

        # Always display the four v2.2 representatives through the shared
        # card so the Custom query section stays in lockstep, followed by
        # the per-pair detail table.
        matrices = _similarity_matrices_from_frame(similarities, available)
        html_parts.append(_similarity_heatmap_card(
            str(threshold), _threshold_display_label(analyzer, threshold, dataset_names), labels, matrices))
        html_parts.append(_similarity_detail_table(
            similarities, available, nickname_map))
    
    html_parts.append('</div></div>')
    return ''.join(html_parts)


def _generate_hemisphere_symmetry_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                          nickname_map: Dict[str, str]) -> str:
    """Generate hemisphere symmetry summary section."""
    html_parts = []
    
    # Check if separate_hemispheres is enabled
    separate_hemispheres = bool(getattr(getattr(analyzer, 'parameters', None), 'separate_hemispheres', False))
    
    summaries = analyzer.get_hemisphere_symmetry_summaries()

    html_parts.append("""
        <div id="hemisphere-symmetry" class="section">
            <div class="section-header">🪞 Hemisphere Symmetry</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Hemisphere symmetry summaries per dataset and threshold (ipsilateral vs contralateral).
                </p>
    """)

    # Show notice if separate_hemispheres=False
    if not separate_hemispheres:
        html_parts.append('''
            <div class="card" style="background: #fef3c7; border: 1px solid #f59e0b;">
                <p style="color: #92400e; margin: 0;">
                    <strong>⚠️ Hemisphere analysis unavailable:</strong> 
                    <code>separate_hemispheres=False</code> in this comparison. 
                    To enable hemisphere symmetry analysis, set <code>separate_hemispheres=True</code> 
                    in ComparisonParameters. This adds _L/_R/_U suffixes to neuron type labels, 
                    allowing comparison of left vs right hemisphere connectivity.
                </p>
            </div>
        ''')
        html_parts.append("</div></div>")
        return ''.join(html_parts)

    if not summaries:
        html_parts.append('<div class="card"><p style="color:#999; text-align:center;">No hemisphere symmetry summaries found.</p></div>')
        html_parts.append("</div></div>")
        return ''.join(html_parts)

    html_parts.append('<div class="tabs"><div class="tab-buttons">')
    threshold_dom_keys = _unique_dom_keys(thresholds)
    for i, t in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(
            f'<button class="tab-btn {active}" '
            f'data-sym-dom-key="{threshold_dom_keys[i]}" '
            f'onclick="showSymTab({_html_js_arg(t)}, this)">{_threshold_tab_label(analyzer, t, dataset_names, nickname_map)}</button>')
    html_parts.append('</div>')

    for i, threshold in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(
            f'<div id="sym_tab_{threshold_dom_keys[i]}" '
            f'class="tab-content {active}">')
        html_parts.append('<div class="card">')
        html_parts.append(f'<h3>Hemisphere Symmetry at {_threshold_display_label(analyzer, threshold, dataset_names)}</h3>')
        html_parts.append('<table><thead><tr>'
                          '<th>Dataset</th>'
                          '<th>Ipsi Jaccard</th>'
                          '<th>Contra Jaccard</th>'
                          '<th>Ipsi Conserved/Union</th>'
                          '<th>Contra Conserved/Union</th>'
                          '<th>Types Conserved/Union</th>'
                          '<th>Counts L/R</th>'
                          '</tr></thead><tbody>')

        for dataset in dataset_names:
            summary = summaries.get(threshold, {}).get(dataset)
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
            ds_label = nickname_map.get(dataset, dataset)
            html_parts.append(
                f'<tr><td><strong>{ds_label}</strong></td>'
                f'<td>{ipsi_j:.3f}</td>'
                f'<td>{contra_j:.3f}</td>'
                f'<td>{ipsi_cons}</td>'
                f'<td>{contra_cons}</td>'
                f'<td>{types_cons}</td>'
                f'<td>{lr_counts}</td></tr>'
            )

        html_parts.append('</tbody></table></div></div>')

    html_parts.append("""
                </div>
                <script>
                    function showSymTab(threshold, button) {
                        document.querySelectorAll('#hemisphere-symmetry .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#hemisphere-symmetry .tab-btn').forEach(el => el.classList.remove('active'));
                        const domKey = button && button.dataset.symDomKey;
                        const panel = domKey ? document.getElementById('sym_tab_' + domKey) : null;
                        if (!panel) return;
                        panel.classList.add('active');
                        if (button) button.classList.add('active');
                    }
                </script>
            </div>
        </div>
    """)

    return ''.join(html_parts)


def _generate_networks_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                nickname_map: Dict[str, str],
                                point_keys: List = None,
                                point_labels: List[str] = None,
                                aligned_network_getter=None,
                                aligned_getter=None,
                                path_getter=None,
                                mode_labels: Tuple[str, str] = ('Threshold', 'Dataset'),
                                tab_label_prefix: str = 't = ',
                                key_noun_plural: str = 'thresholds',
                                key_noun_singular: str = 'threshold',
                                section_id: str = 'networks',
                                section_note: str = '') -> str:
    """Generate networks section with conservation-colored edges and role-colored nodes.

    Both report modes share this generator. Standard mode passes only the
    scalar ``thresholds``; Custom combination mode additionally passes
    ``point_keys`` (query ids used for DOM/JS state), ``point_labels`` and
    query-aware data getters, and the mode/button wording adapts through
    ``mode_labels``/``tab_label_prefix``/``key_noun_*``.
    """
    point_keys = list(thresholds) if point_keys is None else list(point_keys)
    if point_labels is None:
        point_labels = [f"{tab_label_prefix}{k}" for k in point_keys]
    js_keys = [json.dumps(k) for k in point_keys]
    dom_keys = _unique_dom_keys(point_keys)
    dom_key_by_key = {key: dom_keys[index]
                      for index, key in enumerate(point_keys)}
    key_label_by_key = {k: point_labels[i] for i, k in enumerate(point_keys)}

    def resolve_network_aligned(k):
        if aligned_network_getter is not None:
            return aligned_network_getter(k)
        return analyzer.get_aligned_data_for_network(k)

    def resolve_aligned(k):
        if aligned_getter is not None:
            return aligned_getter(k)
        return analyzer.get_aligned_data(k)

    def resolve_paths(k):
        if path_getter is not None:
            try:
                return path_getter(k)
            except Exception:
                return None
        return None

    thresholds_json = json.dumps(point_keys)
    dom_keys_json = json.dumps(dom_keys)
    num_networks = len(point_keys)
    # Responsive grid: 1 col on small, 2 cols if 2+ networks
    grid_cols = min(num_networks, 2)
    separate_hemispheres = bool(getattr(getattr(analyzer, 'parameters', None), 'separate_hemispheres', False))
    mirror_disabled_attr = '' if separate_hemispheres else 'disabled'
    mirror_btn_style = (
        'padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border-color); '
        'background: #64748b; color: white; font-size: 12px; white-space: nowrap; '
        + ('cursor: pointer;' if separate_hemispheres else 'cursor: not-allowed; opacity: 0.5;')
    )
    mirror_btn_title = '' if separate_hemispheres else 'Hemisphere mirroring requires separate_hemispheres=True'
    
    # Self-edges in the edge matrix (type→same type).  They are excluded
    # from path analysis (FindAllPath requires source≠target), so call them
    # out when present.  They arise from intra-type synapses of
    # intermediate/target populations, or from an identical source/target
    # query — the message distinguishes the two.
    self_edge_warning = ""
    try:
        if hasattr(analyzer, 'label_mapper') and analyzer.label_mapper:
            src_set = set(analyzer.label_mapper.get_all_std_labels('source'))
            tgt_set = set(analyzer.label_mapper.get_all_std_labels('target'))
        else:
            src_list = analyzer.parameters._ensure_flat_list(analyzer.parameters.source_neurons)
            tgt_list = analyzer.parameters._ensure_flat_list(analyzer.parameters.target_neurons)
            src_set = set(src_list)
            tgt_set = set(tgt_list)

        # Empty mappers must not read as "identical lists".
        lists_identical = bool(src_set) and bool(tgt_set) \
            and src_set == tgt_set
        overlap = sorted(str(t) for t in (src_set & tgt_set)
                         ) if (src_set and tgt_set) else []

        # Count self-edges across all thresholds
        self_edge_count = 0
        for k in point_keys:
            aligned = resolve_network_aligned(k)
            if not aligned.empty:
                for edge_key in aligned.index:
                    if ' -> ' in str(edge_key):
                        parts = str(edge_key).split(' -> ')
                        if len(parts) == 2 and parts[0] == parts[1]:
                            self_edge_count += 1
                break  # Only count once (same edges at all thresholds)
        if self_edge_count > 0:
            if lists_identical:
                reason = ('The source and target query lists are identical, '
                          'so every intra-list connection is a self-edge.')
            elif overlap:
                shown = ', '.join(overlap[:5]) + ('…' if len(overlap) > 5 else '')
                reason = (f'The source and target queries share '
                          f'{len(overlap)} token(s) ({shown}); the rest are '
                          'intra-type synapses of intermediate or target '
                          'populations.')
            else:
                reason = ('These are intra-type synapses of intermediate or '
                          'target populations (e.g. a recurrent '
                          's-LNv→s-LNv connection) — the query lists '
                          'themselves do not overlap.')
            self_edge_warning = f'''
                <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
                    <span style="color: #92400e;">⚠️ <strong>Self-edges detected:</strong>
                    <strong>{self_edge_count}</strong> self-edges (type→same type) exist in the edge matrix
                    but are excluded from path analysis (FindAllPath requires source≠target). {reason}</span>
                </div>'''
    except Exception:
        pass
    
    html_parts = []
    html_parts.append(f"""
        <div id="{html.escape(section_id, quote=True)}" class="section networks-section">
            <div class="section-header">🕸️ Network Visualizations</div>
            {f'<div style="padding: 0 18px;">{section_note}</div>' if section_note else ''}
            <div class="section-content">
                {self_edge_warning}
                <div style="margin-bottom: 15px;">
                    <p style="color: var(--secondary-color); margin: 0 0 8px 0;">
                        <strong>Edge conservation:</strong> 
                        <span style="color: #22c55e;">● Conserved (all)</span> 
                        <span style="color: #f59e0b; margin-left: 12px;">● Partial (some)</span>
                        <span style="color: #94a3b8; margin-left: 12px;">● Unique (one)</span>
                    </p>
                    <p style="color: var(--secondary-color); margin: 0;">
                        <strong>Node role:</strong> 
                        <span style="color: #ef4444;">● Source</span> 
                        <span style="color: #3b82f6; margin-left: 12px;">● Intermediate</span>
                        <span style="color: #8b5cf6; margin-left: 12px;">● Target</span>
                        <span style="color: #9ca3af; margin-left: 12px;">● Dead-end (no path through)</span>
                    </p>
                </div>
                <script>
                    // Auto mode renders TWO networks sections (per-threshold +
                    // density-matched), each emitting this script. A second
                    // top-level `const networkDomKeys` is a parse-time
                    // SyntaxError that silently kills the whole block for the
                    // second section, so every declaration lives in this IIFE
                    // and the shared window.* state MERGES instead of resetting.
                    (function() {{
                    // Global network mode state (default: static)
                    // Filter mode: 0 = Show All, 1 = Conserved Only, 2 = No Unique
                    window.networkFilterMode = window.networkFilterMode || {{}}; // Per-key filter mode
                    window.networkPhysicsEnabled = window.networkPhysicsEnabled || {{}}; // Per-key physics state
                    window.hideDeadEndNodes = window.hideDeadEndNodes || {{}}; // Per-key dead-end state
                    window.allNetworks = window.allNetworks || {{}};
                    const sectionKeys = {thresholds_json};
                    window.allThresholds = (window.allThresholds || []).concat(sectionKeys);
                    window.networkDomKeys = window.networkDomKeys || {{}};
                    const networkDomKeys = {dom_keys_json};

                    // Initialize this section's keys only (merge-safe across sections)
                    sectionKeys.forEach((t, index) => {{
                        window.networkFilterMode[t] = 0;  // 0=All, 1=Conserved, 2=NoUnique
                        window.networkPhysicsEnabled[t] = false;
                        window.hideDeadEndNodes[t] = false;
                        window.networkDomKeys[String(t)] = networkDomKeys[index];
                    }});

                    function networkDomKey(threshold) {{
                        return window.networkDomKeys[String(threshold)] || String(threshold);
                    }}
                    
                    // Per-network toggle functions
                    function toggleNetworkFilter(threshold) {{
                        // Cycle through modes: 0 (All) -> 1 (Conserved) -> 2 (No Unique) -> 0
                        window.networkFilterMode[threshold] = (window.networkFilterMode[threshold] + 1) % 3;
                        const mode = window.networkFilterMode[threshold];
                        const btn = document.getElementById('filter_btn_' + networkDomKey(threshold));
                        
                        const modeLabels = ['🌐 Show All', '✅ Conserved Only', '🚫 No Unique'];
                        const modeColors = ['var(--secondary-color)', '#22c55e', '#f59e0b'];
                        btn.innerHTML = modeLabels[mode];
                        btn.style.background = modeColors[mode];
                        
                        updateNetworkDisplay(threshold);
                    }}
                    
                    function toggleDeadEndNodes(threshold) {{
                        window.hideDeadEndNodes[threshold] = !window.hideDeadEndNodes[threshold];
                        const btn = document.getElementById('deadend_btn_' + networkDomKey(threshold));
                        
                        if (window.hideDeadEndNodes[threshold]) {{
                            btn.innerHTML = '🚫 Hide Dead-ends';
                            btn.style.background = '#f59e0b';
                        }} else {{
                            btn.innerHTML = '👁️ Show Dead-ends';
                            btn.style.background = 'var(--secondary-color)';
                        }}
                        
                        updateNetworkDisplay(threshold);
                    }}
                    
                    function toggleNetworkPhysics(threshold) {{
                        window.networkPhysicsEnabled[threshold] = !window.networkPhysicsEnabled[threshold];
                        const btn = document.getElementById('physics_btn_' + networkDomKey(threshold));
                        
                        if (window.networkPhysicsEnabled[threshold]) {{
                            btn.innerHTML = '💥 Duang Mode';
                            btn.style.background = 'var(--primary-color)';
                        }} else {{
                            btn.innerHTML = '📌 Static Mode';
                            btn.style.background = 'var(--secondary-color)';
                        }}
                        
                        applyNetworkPhysics(threshold);
                    }}
                    
                    // Hemisphere mirroring toggle
                    window.hemisphereMirrorEnabled = window.hemisphereMirrorEnabled || {{}};  // Per-key mirror state
                    sectionKeys.forEach(t => {{
                        // Mirror auto-enables when Separate Hemispheres (L/R) is checked
                        window.hemisphereMirrorEnabled[t] = {'true' if separate_hemispheres else 'false'};
                    }});
                    
                    function toggleHemisphereMirror(threshold) {{
                        window.hemisphereMirrorEnabled[threshold] = !window.hemisphereMirrorEnabled[threshold];
                        const btn = document.getElementById('mirror_btn_' + networkDomKey(threshold));
                        
                        if (window.hemisphereMirrorEnabled[threshold]) {{
                            btn.innerHTML = '🪞 Mirrored';
                            btn.style.background = '#0ea5e9';
                            applyHemisphereMirror(threshold);
                        }} else {{
                            btn.innerHTML = '🪞 Mirror Hemispheres';
                            btn.style.background = '#64748b';
                            resetHemisphereMirror(threshold);
                        }}
                    }}
                    
                    function getBaseName(label) {{
                        // Extract base name without hemisphere suffix (handles parentheses)
                        const base = label.split('(')[0].trim();
                        if (base.endsWith('_L') || base.endsWith('_R') || base.endsWith('_U')) {{
                            return base.slice(0, -2);
                        }}
                        return base;
                    }}
                    
                    function getHemisphere(label) {{
                        // Get hemisphere from label suffix (handles parentheses)
                        const base = label.split('(')[0].trim();
                        if (base.endsWith('_L')) return 'L';
                        if (base.endsWith('_R')) return 'R';
                        if (base.endsWith('_U')) return 'U';
                        return 'U';  // Unknown
                    }}
                    
                    function applyHemisphereMirror(threshold) {{
                        if (!window.allNetworks[threshold]) return;
                        const netData = window.allNetworks[threshold];
                        const net = netData.network;
                        const nodes = netData.nodes;
                        
                        // Get current visible node IDs from the DataSet
                        const visibleNodeIds = new Set(nodes.getIds());
                        
                        // Store original positions if not already stored
                        if (!netData.originalPositions) {{
                            netData.originalPositions = {{}};
                            const positions = net.getPositions();
                            for (const [id, pos] of Object.entries(positions)) {{
                                netData.originalPositions[id] = {{ x: pos.x, y: pos.y }};
                            }}
                        }}
                        
                        // Update original positions with current visible positions
                        // (hierarchical layout may have changed since last mirror toggle)
                        const currentPositions = net.getPositions();
                        for (const [id, pos] of Object.entries(currentPositions)) {{
                            netData.originalPositions[id] = {{ x: pos.x, y: pos.y }};
                        }}
                        
                        // Use current positions as the layout baseline
                        const positions = currentPositions;
                        const nodeIds = Object.keys(positions);
                        if (nodeIds.length === 0) return;
                        
                        // Calculate bounds of current visible layout
                        let minX = Infinity, maxX = -Infinity;
                        for (const id of nodeIds) {{
                            minX = Math.min(minX, positions[id].x);
                            maxX = Math.max(maxX, positions[id].x);
                        }}
                        const centerX = (minX + maxX) / 2;
                        
                        // Build base name -> hemisphere node mapping (only for visible nodes)
                        const baseNodeMap = {{}};
                        const allNodes = netData.allNodes;
                        allNodes.forEach(node => {{
                            // Only include nodes that are currently visible
                            if (!visibleNodeIds.has(node.id)) return;
                            
                            const baseName = getBaseName(node.label);
                            const hemi = getHemisphere(node.label);
                            if (!baseNodeMap[baseName]) baseNodeMap[baseName] = {{}};
                            baseNodeMap[baseName][hemi] = node;
                        }});
                        
                        // Apply mirrored layout: enforce symmetric X for L/R pairs
                        const updates = [];
                        const minOffset = 60;
                        for (const [baseName, hemiNodes] of Object.entries(baseNodeMap)) {{
                            const nodeL = hemiNodes['L'];
                            const nodeR = hemiNodes['R'];
                            const nodeU = hemiNodes['U'];
                            const candidates = [nodeL, nodeR, nodeU].filter(Boolean);
                            if (candidates.length === 0) continue;
                            
                            let sumY = 0;
                            let countY = 0;
                            candidates.forEach(n => {{
                                if (positions[n.id]) {{
                                    sumY += positions[n.id].y;
                                    countY++;
                                }}
                            }});
                            const baseY = countY > 0 ? sumY / countY : 0;
                            
                            if (nodeL && nodeR && positions[nodeL.id] && positions[nodeR.id]) {{
                                const leftPos = positions[nodeL.id];
                                const rightPos = positions[nodeR.id];
                                let offset = (Math.abs(leftPos.x - centerX) + Math.abs(rightPos.x - centerX)) / 2;
                                if (!isFinite(offset) || offset < minOffset) offset = minOffset;
                                updates.push({{ id: nodeL.id, x: centerX + offset, y: baseY }});
                                updates.push({{ id: nodeR.id, x: centerX - offset, y: baseY }});
                            }} else if (nodeL && positions[nodeL.id]) {{
                                let offset = Math.abs(positions[nodeL.id].x - centerX);
                                if (!isFinite(offset) || offset < minOffset) offset = minOffset;
                                updates.push({{ id: nodeL.id, x: centerX + offset, y: baseY }});
                            }} else if (nodeR && positions[nodeR.id]) {{
                                let offset = Math.abs(positions[nodeR.id].x - centerX);
                                if (!isFinite(offset) || offset < minOffset) offset = minOffset;
                                updates.push({{ id: nodeR.id, x: centerX - offset, y: baseY }});
                            }}
                            
                            if (nodeU && positions[nodeU.id]) {{
                                updates.push({{ id: nodeU.id, x: centerX, y: baseY }});
                            }}
                        }}
                        
                        // Apply updates (only to visible nodes)
                        updates.forEach(u => {{
                            nodes.update({{ id: u.id, x: u.x, y: u.y }});
                        }});
                        
                        // Fit the view
                        net.fit({{ animation: true }});
                    }}
                    
                    function resetHemisphereMirror(threshold) {{
                        if (!window.allNetworks[threshold]) return;
                        const netData = window.allNetworks[threshold];
                        const net = netData.network;
                        const nodes = netData.nodes;
                        
                        // Restore original positions if available
                        if (netData.originalPositions) {{
                            const updates = [];
                            for (const [id, pos] of Object.entries(netData.originalPositions)) {{
                                updates.push({{ id: parseInt(id), x: pos.x, y: pos.y }});
                            }}
                            updates.forEach(u => {{
                                nodes.update(u);
                            }});
                            net.fit({{ animation: true }});
                        }}
                    }}
                    
                    function updateNetworkDisplay(threshold) {{
                        if (!window.allNetworks[threshold]) return;
                        
                        const netData = window.allNetworks[threshold];
                        const net = netData.network;
                        const edges = netData.edges;
                        const nodes = netData.nodes;
                        const allEdges = netData.allEdges;
                        const allNodes = netData.allNodes;
                        const conservedEdgeIds = netData.conservedEdgeIds;
                        const uniqueEdgeIds = netData.uniqueEdgeIds || new Set();
                        const nodeRoles = netData.nodeRoles || {{}};
                        
                        const mode = window.networkFilterMode[threshold];
                        
                        // Filter edges based on mode
                        let filteredEdges = allEdges;
                        if (mode === 1) {{
                            // Conserved Only
                            filteredEdges = allEdges.filter(e => conservedEdgeIds.has(e.id));
                        }} else if (mode === 2) {{
                            // No Unique (show conserved + partial, hide unique)
                            filteredEdges = allEdges.filter(e => !uniqueEdgeIds.has(e.id));
                        }}
                        
                        // Get connected nodes from filtered edges
                        let connectedNodeIds = new Set();
                        filteredEdges.forEach(e => {{
                            connectedNodeIds.add(e.from);
                            connectedNodeIds.add(e.to);
                        }});
                        
                        // Filter nodes based on mode
                        let filteredNodes;
                        if (mode === 0) {{
                            filteredNodes = allNodes;
                        }} else {{
                            filteredNodes = allNodes.filter(n => connectedNodeIds.has(n.id));
                        }}
                        
                        // Recalculate dead-ends based on CURRENT filtered edges
                        // Dead-end = intermediate node with only incoming OR only outgoing edges in current view
                        // OR node that only connects to dead-ends (recursive)
                        // IMPORTANT: Source and target nodes are NEVER dead-ends
                        let dynamicDeadEndNodeIds = new Set();
                        
                        if (filteredEdges.length > 0) {{
                            // Build adjacency for current filtered graph
                            let adjOut = {{}};
                            let adjIn = {{}};
                            let activeNodes = new Set();
                            
                            filteredEdges.forEach(e => {{
                                if (!adjOut[e.from]) adjOut[e.from] = [];
                                adjOut[e.from].push(e.to);
                                
                                if (!adjIn[e.to]) adjIn[e.to] = [];
                                adjIn[e.to].push(e.from);
                                
                                activeNodes.add(e.from);
                                activeNodes.add(e.to);
                            }});
                            
                            // Iterative dead-end detection
                            let changed = true;
                            while (changed) {{
                                changed = false;
                                activeNodes.forEach(nodeId => {{
                                    if (dynamicDeadEndNodeIds.has(nodeId)) return;
                                    
                                    const role = nodeRoles[nodeId] || 'intermediate';
                                    if (role !== 'intermediate') return;
                                    
                                    const outNeighbors = adjOut[nodeId] || [];
                                    const inNeighbors = adjIn[nodeId] || [];
                                    
                                    // Condition 1: All outgoing paths lead to dead-ends (or no outgoing)
                                    // every() returns True for empty sequence
                                    const leadsToDeadEnd = outNeighbors.every(n => dynamicDeadEndNodeIds.has(n));
                                    
                                    // Condition 2: All incoming paths come from dead-ends (or no incoming)
                                    const comesFromDeadEnd = inNeighbors.every(n => dynamicDeadEndNodeIds.has(n));
                                    
                                    if (leadsToDeadEnd || comesFromDeadEnd) {{
                                        dynamicDeadEndNodeIds.add(nodeId);
                                        changed = true;
                                    }}
                                }});
                            }}
                        }}
                        
                        // Update node colors to reflect current dead-end status
                        // Source and target always keep their colors, never become dead-end
                        const roleColors = {{
                            'source': {{ background: '#ef4444', border: '#b91c1c' }},
                            'intermediate': {{ background: '#3b82f6', border: '#1d4ed8' }},
                            'target': {{ background: '#8b5cf6', border: '#6d28d9' }},
                            'dead-end': {{ background: '#9ca3af', border: '#6b7280' }}
                        }};
                        
                        filteredNodes = filteredNodes.map(n => {{
                            const role = nodeRoles[n.id] || 'intermediate';
                            // Source and target are NEVER dead-ends - they keep their role
                            const isDeadEnd = (role === 'intermediate') && dynamicDeadEndNodeIds.has(n.id);
                            const displayRole = isDeadEnd ? 'dead-end' : role;
                            const colors = roleColors[displayRole];
                            return {{
                                ...n,
                                color: colors,
                                title: n.label + ' (' + displayRole + ')'
                            }};
                        }});
                        
                        // Apply dead-end filter if enabled
                        // NEVER hide source or target nodes, even if dead-end toggle is on
                        if (window.hideDeadEndNodes[threshold]) {{
                            filteredNodes = filteredNodes.filter(n => {{
                                const role = nodeRoles[n.id] || 'intermediate';
                                // Keep source and target nodes always
                                if (role === 'source' || role === 'target') return true;
                                // Filter out dead-end intermediates
                                return !dynamicDeadEndNodeIds.has(n.id);
                            }});
                            filteredEdges = filteredEdges.filter(e => {{
                                const fromRole = nodeRoles[e.from] || 'intermediate';
                                const toRole = nodeRoles[e.to] || 'intermediate';
                                // Keep edges to/from source or target
                                if (fromRole === 'source' || fromRole === 'target') return true;
                                if (toRole === 'source' || toRole === 'target') return true;
                                // Filter out edges involving dead-end intermediates
                                return !dynamicDeadEndNodeIds.has(e.from) && !dynamicDeadEndNodeIds.has(e.to);
                            }});
                        }}
                        
                        // Apply to network
                        edges.clear();
                        edges.add(filteredEdges);
                        nodes.clear();
                        nodes.add(filteredNodes);
                        
                        // Reinitialize layout
                        net.setOptions({{
                            nodes: {{ size: 18, font: {{ size: 11 }} }},
                            edges: {{ smooth: {{ type: 'curvedCW', roundness: 0.1 }} }},
                            layout: {{
                                hierarchical: {{
                                    enabled: true,
                                    direction: 'UD',
                                    sortMethod: 'directed',
                                    levelSeparation: 120,
                                    nodeSpacing: 80,
                                    treeSpacing: 100
                                }}
                            }},
                            physics: {{ enabled: false }}
                        }});
                        setTimeout(() => {{
                            net.setOptions({{ layout: {{ hierarchical: false }} }});
                            // Re-apply mirror positions if mirror mode is enabled
                            if (window.hemisphereMirrorEnabled[threshold]) {{
                                applyHemisphereMirror(threshold);
                            }}
                            net.fit({{ animation: true }});
                        }}, 200);
                    }}
                    
                    function applyNetworkPhysics(threshold) {{
                        if (!window.allNetworks[threshold]) return;
                        
                        const net = window.allNetworks[threshold].network;
                        
                        if (window.networkPhysicsEnabled[threshold]) {{
                            // Duang mode
                            net.setOptions({{
                                nodes: {{ size: 20, font: {{ size: 12 }} }},
                                edges: {{ smooth: {{ type: 'continuous', roundness: 0.2 }} }},
                                layout: {{ hierarchical: false }},
                                physics: {{
                                    enabled: true,
                                    solver: 'forceAtlas2Based',
                                    forceAtlas2Based: {{
                                        gravitationalConstant: -80,
                                        centralGravity: 0.005,
                                        springLength: 120,
                                        springConstant: 0.06,
                                        damping: 0.5,
                                        avoidOverlap: 0.8
                                    }},
                                    stabilization: {{ enabled: true, iterations: 100, updateInterval: 25 }},
                                    minVelocity: 0.5
                                }}
                            }});
                            net.once('stabilized', () => {{ net.fit({{ animation: true }}); }});
                        }} else {{
                            // Static mode
                            net.setOptions({{
                                nodes: {{ size: 18, font: {{ size: 11 }} }},
                                edges: {{ smooth: {{ type: 'curvedCW', roundness: 0.1 }} }},
                                layout: {{
                                    hierarchical: {{
                                        enabled: true,
                                        direction: 'UD',
                                        sortMethod: 'directed',
                                        levelSeparation: 120,
                                        nodeSpacing: 80,
                                        treeSpacing: 100
                                    }}
                                }},
                                physics: {{ enabled: false }}
                            }});
                            setTimeout(() => {{
                                net.setOptions({{ layout: {{ hierarchical: false }} }});
                                net.fit({{ animation: true }});
                            }}, 200);
                        }}
                    }}

                    // Inline onclick handlers resolve these at click time
                    window.toggleNetworkFilter = toggleNetworkFilter;
                    window.toggleDeadEndNodes = toggleDeadEndNodes;
                    window.toggleNetworkPhysics = toggleNetworkPhysics;
                    window.toggleHemisphereMirror = toggleHemisphereMirror;
                    }})();
                </script>
""")
    
    # Mode toggle (Threshold vs Dataset)
    nicknames = [nickname_map[d] for d in dataset_names]
    dataset_dom_keys = [f'{section_id}__{k}' for k in _unique_dom_keys(nicknames)]
    dataset_dom_key_by_nick = {
        nick: dataset_dom_keys[index]
        for index, nick in enumerate(nicknames)
    }
    nicknames_json = json.dumps(nicknames)

    html_parts.append(f'''
                <!-- Toggle Mode Selector -->
                <div style="margin-bottom: 15px;">
                    <span style="font-weight: 600; margin-right: 10px;">View by:</span>
                    <button class="tab-btn active" id="{section_id}__network_mode_threshold" onclick="switchNetworkMode('{section_id}', 'threshold')">{mode_labels[0]}</button>
                    <button class="tab-btn" id="{section_id}__network_mode_dataset" onclick="switchNetworkMode('{section_id}', 'dataset')">{mode_labels[1]}</button>
                </div>
''')

    # By Threshold View
    html_parts.append(f'<div id="{section_id}__network_by_threshold" class="tabs"><div class="tab-buttons">')
    for i, (k, js_key, k_label) in enumerate(zip(point_keys, js_keys, point_labels)):
        active = 'active' if i == 0 else ''
        dom_key = dom_keys[i]
        html_parts.append(
            f'<button class="tab-btn {active}" '
            f'data-network-dom-key="{dom_key}" '
            f'onclick="showNetworkTab({_html_js_arg(k)}, this)">'
            f'{html.escape(k_label)}</button>')
    html_parts.append('</div>')

    # Network containers (only first visible initially)
    for i, k in enumerate(point_keys):
        active = 'active' if i == 0 else ''
        dom_key = dom_keys[i]
        html_parts.append(
            f'<div id="network_tab_{dom_key}" '
            f'data-network-key="{html.escape(str(k), quote=True)}" '
            f'class="tab-content {active}">')
        html_parts.append(_generate_conservation_network(
            analyzer, dataset_names, k, nickname_map,
            key=k, dom_key=dom_key, display_label=point_labels[i],
            aligned_override=(resolve_network_aligned(k)
                              if aligned_network_getter else None),
            path_override=(resolve_paths(k) if path_getter else None)))
        html_parts.append(_generate_type_coverage_card(
            analyzer, k, dataset_names, nickname_map,
            display_label=point_labels[i]))
        html_parts.append('</div>')

    html_parts.append('</div>')  # Close network_by_threshold tabs div

    # By Dataset View
    html_parts.append(f'<div id="{section_id}__network_by_dataset" class="tabs" style="display: none;"><div class="tab-buttons">')
    for i, d in enumerate(dataset_names):
        active = 'active' if i == 0 else ''
        nick = nickname_map[d]
        dataset_dom_key = dataset_dom_keys[i]
        html_parts.append(
            f'<button class="tab-btn {active}" '
            f'data-network-dom-key="{dataset_dom_key}" '
            f'onclick="showNetworkDatasetTab({_html_js_arg(nick)}, this)">'
            f'{html.escape(str(nick))}</button>')
    html_parts.append('</div>')

    # Dataset-centric network containers (showing all points for one dataset)
    for i, d in enumerate(dataset_names):
        nick = nickname_map[d]
        active = 'active' if i == 0 else ''
        dataset_dom_key = dataset_dom_keys[i]
        html_parts.append(
            f'<div id="network_dataset_tab_{dataset_dom_key}" '
            f'data-network-dataset-key="{html.escape(str(nick), quote=True)}" '
            f'data-network-dom-key="{html.escape(str(dataset_dom_key), quote=True)}" '
            f'class="tab-content {active}">')
        html_parts.append(_generate_dataset_network(
            analyzer, d, thresholds, nickname_map,
            keys=point_keys, key_labels=point_labels,
            aligned_getter=aligned_getter, path_getter=path_getter,
            key_noun_plural=key_noun_plural,
            key_noun_singular=key_noun_singular,
            dom_key=dataset_dom_key))
        html_parts.append('</div>')
    
    html_parts.append('</div>')  # Close network_by_dataset tabs div
    
    # JavaScript for tab switching
    html_parts.append("""
                <script>
                    // Same namespace discipline as the state block above: this
                    // script is emitted once per networks section, so everything
                    // lives in an IIFE and only the handlers the inline onclick
                    // attributes resolve are published on window.
                    (function() {
                    var sectionEl = document.currentScript
                        ? document.currentScript.closest('.networks-section') : null;

                    function switchNetworkMode(scope, mode) {
                        var pfx = scope + '__';
                        document.getElementById(pfx + 'network_mode_threshold').classList.toggle('active', mode === 'threshold');
                        document.getElementById(pfx + 'network_mode_dataset').classList.toggle('active', mode === 'dataset');
                        document.getElementById(pfx + 'network_by_threshold').style.display = mode === 'threshold' ? 'block' : 'none';
                        document.getElementById(pfx + 'network_by_dataset').style.display = mode === 'dataset' ? 'block' : 'none';
                        
                        // Re-fit ONLY VISIBLE networks (must call redraw() first to recalculate canvas dimensions)
                        // vis.js cannot render in hidden containers
                        setTimeout(function() {
                            if (mode === 'threshold') {
                                // Only redraw the active threshold network
                                const qsel = '#' + scope + '__network_by_threshold .tab-content.active';
                        const activeTab = document.querySelector(qsel);
                                if (activeTab) {
                                    const threshold = activeTab.dataset.networkKey;
                                    if (window.allNetworks && window.allNetworks[threshold]) {
                                        window.allNetworks[threshold].network.redraw();
                                        window.allNetworks[threshold].network.fit({ animation: false });
                                    }
                                }
                            } else {
                                // Only redraw the active dataset networks
                                const dsel = '#' + scope + '__network_by_dataset .tab-content.active';
                        const activeTab = document.querySelector(dsel);
                                if (activeTab) {
                                    const dataset = activeTab.dataset.networkDatasetKey;
                                    const netData = window.allNetworks && window.allNetworks[(activeTab.dataset.networkDomKey || '') + '__' + dataset + '_dataset'];
                                    if (netData && netData.network) {
                                        netData.network.redraw();
                                        netData.network.fit({ animation: false });
                                    }
                                }
                            }
                        }, 100);
                    }
                    
                    // Re-draw and fit THIS section's active network after page
                    // load. Each section emits its own copy of this script, so
                    // every per-threshold / density-matched section fits itself
                    // (window.allThresholds[0] only ever named the first one).
                    // IMPORTANT: Only redraw networks in VISIBLE containers;
                    // vis.js cannot calculate dimensions for hidden containers.
                    window.addEventListener('load', function() {
                        setTimeout(function() {
                            if (!sectionEl || !sectionEl.id) return;
                            const activeTab = document.querySelector(
                                '#' + sectionEl.id + '__network_by_threshold .tab-content.active');
                            if (!activeTab) return;
                            const threshold = activeTab.dataset.networkKey;
                            const netData = window.allNetworks && window.allNetworks[threshold];
                            if (netData && netData.network) {
                                netData.network.redraw();
                                netData.network.fit({ animation: false });
                            }
                        }, 300);
                    });
                    
                    function showNetworkTab(threshold, button) {
                        // Section scope comes from the button's section root
                        const scope = button ? button.closest('.networks-section').id : '';
                        const domKey = button && button.dataset.networkDomKey;
                        // Hide all network tabs in threshold view
                        document.querySelectorAll('#' + scope + '__network_by_threshold .tab-content').forEach(el => el.classList.remove('active'));
                        // Remove active from all buttons in threshold view
                        document.querySelectorAll('#' + scope + '__network_by_threshold .tab-btn').forEach(el => el.classList.remove('active'));
                        // Show selected tab
                        const panel = domKey ? document.getElementById('network_tab_' + domKey) : null;
                        if (!panel) return;
                        panel.classList.add('active');
                        // Mark button as active
                        if (button) button.classList.add('active');
                        
                        // Re-fit the network since it may have been hidden
                        // Must call redraw() first to recalculate canvas dimensions
                        if (window.allNetworks && window.allNetworks[threshold]) {
                            setTimeout(function() {
                                const netData = window.allNetworks[threshold];
                                netData.network.redraw();
                                netData.network.fit({ animation: false });
                            }, 100);
                        }
                    }
                    
                    function showNetworkDatasetTab(dataset, button) {
                        // Section scope comes from the button's section root
                        const scope = button ? button.closest('.networks-section').id : '';
                        const domKey = button && button.dataset.networkDomKey;
                        // Hide all network tabs in dataset view
                        document.querySelectorAll('#' + scope + '__network_by_dataset .tab-content').forEach(el => el.classList.remove('active'));
                        // Remove active from all buttons in dataset view
                        document.querySelectorAll('#' + scope + '__network_by_dataset .tab-btn').forEach(el => el.classList.remove('active'));
                        // Show selected tab (panel ids embed the section
                        // prefix themselves; the scope must NOT be applied
                        // again — that double prefix matched nothing).
                        const panel = domKey ? document.getElementById('network_dataset_tab_' + domKey) : null;
                        if (!panel) return;
                        panel.classList.add('active');
                        // Mark button as active
                        if (button) button.classList.add('active');
                        
                        // Re-fit the network since it may have been hidden
                        // Must call redraw() first to recalculate canvas dimensions
                        setTimeout(function() {
                            // Registration key: domKey + '__' + dataset + '_dataset'
                            // (domKey comes from the clicked button; activeTab is
                            // not in scope here).
                            const netData = window.allNetworks && window.allNetworks[(domKey || '') + '__' + dataset + '_dataset'];
                            if (netData && netData.network) {
                                netData.network.redraw();
                                netData.network.fit({ animation: false });
                            }
                        }, 100);
                    }

                    // Inline onclick handlers resolve these at click time
                    window.switchNetworkMode = switchNetworkMode;
                    window.showNetworkTab = showNetworkTab;
                    window.showNetworkDatasetTab = showNetworkDatasetTab;
                    })();
                </script>
            </div>
        </div>
""")
    
    return ''.join(html_parts)


def _generate_reciprocal_visualizations_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                                nickname_map: Dict[str, str]) -> str:
    """Generate reciprocal visualizations section with links to VisPath outputs."""
    html_parts = []

    find_reciprocal = bool(getattr(getattr(analyzer, 'parameters', None), 'find_reciprocal', False))

    html_parts.append("""
        <div id="reciprocal-visualizations" class="section">
            <div class="section-header">🔁 Reciprocal Visualizations</div>
            <div class="section-content">
    """)

    if not find_reciprocal:
        html_parts.append("""
            <div class="card">
                <h3>Reciprocal analysis not enabled</h3>
                <p>This report was generated without <strong>find_reciprocal</strong>. Enable it to generate reciprocal network and heatmap visualizations.</p>
            </div>
        """)
        html_parts.append("</div></div>")
        return ''.join(html_parts)

    for t in thresholds:
        # Dead-code section (Phase 9 rewrites it); keep the historical title.
        html_parts.append(f'<div class="card"><h3>Threshold t = {t}</h3>')
        html_parts.append('<table><thead><tr>'
                  '<th>Dataset</th>'
                  '<th>Type Network</th>'
                  '<th>Type Heatmap</th>'
                  '<th>Group Network</th>'
                  '<th>Group Heatmap</th>'
                  '<th>BodyId Network</th>'
                  '<th>BodyId Heatmap</th>'
                  '<th>Type CSV</th>'
                  '<th>Group CSV</th>'
                  '<th>BodyId CSV</th>'
                  '</tr></thead><tbody>')

        for d in dataset_names:
            nick = nickname_map.get(d, d)
            base_dir = os.path.join(
                analyzer.parameters.full_output_path,
                'dataset_data',
                d,
                f'minsyn_{t}',
                'find_reciprocal',
                'visualizations'
            )

            type_network = _make_link(os.path.join(base_dir, 'reciprocal_type_network.html'), analyzer.parameters.full_output_path)
            type_heatmap = _make_link(os.path.join(base_dir, 'reciprocal_type_heatmap.html'), analyzer.parameters.full_output_path)
            group_network = _make_link(os.path.join(base_dir, 'reciprocal_groups_network.html'), analyzer.parameters.full_output_path)
            group_heatmap = _make_link(os.path.join(base_dir, 'reciprocal_groups_heatmap.html'), analyzer.parameters.full_output_path)
            body_network = _make_link(os.path.join(base_dir, 'reciprocal_bodyId_network.html'), analyzer.parameters.full_output_path)
            body_heatmap = _make_link(os.path.join(base_dir, 'reciprocal_bodyId_heatmap.html'), analyzer.parameters.full_output_path)

            type_csv = _make_link(os.path.join(base_dir, 'reciprocal_connection_type.csv'), analyzer.parameters.full_output_path)
            group_csv = _make_link(os.path.join(base_dir, 'reciprocal_connection_custom_groups.csv'), analyzer.parameters.full_output_path)
            body_csv = _make_link(os.path.join(base_dir, 'reciprocal_connection_bodyId.csv'), analyzer.parameters.full_output_path)

            html_parts.append('<tr>'
                              f'<td><strong>{nick}</strong></td>'
                              f'<td>{type_network}</td>'
                              f'<td>{type_heatmap}</td>'
                              f'<td>{group_network}</td>'
                              f'<td>{group_heatmap}</td>'
                              f'<td>{body_network}</td>'
                              f'<td>{body_heatmap}</td>'
                              f'<td>{type_csv}</td>'
                              f'<td>{group_csv}</td>'
                              f'<td>{body_csv}</td>'
                              '</tr>')

        html_parts.append('</tbody></table></div>')

    html_parts.append('</div></div>')
    return ''.join(html_parts)


def _extract_edges_from_paths(path_data: pd.DataFrame, dataset_names: List[str], max_paths: int = 100) -> set:
    """
    Extract edges from top-M paths ranked by total min_weight across datasets.
    
    This builds a set of edges from the top paths, which ensures complete path
    connectivity rather than selecting disconnected top edges.
    
    Args:
        path_data: DataFrame with path strings as index and dataset columns containing min_weight
        dataset_names: List of dataset names
        max_paths: Maximum number of top paths to include
        
    Returns:
        Set of edge keys (e.g., "A -> B") including BOTH canonical names AND display names
        to ensure matching works regardless of which format the aligned data uses.
    """
    if path_data is None or path_data.empty:
        return set()
    
    # Calculate total weight across datasets for each path
    available_cols = [d for d in dataset_names if d in path_data.columns]
    if not available_cols:
        return set()
    
    # Helper to extract canonical name from display name
    # Handles format: "GNG588(CB0038)" -> "GNG588"
    path_data = path_data.copy()
    path_data['_total'] = path_data[available_cols].sum(axis=1)
    
    # Get top paths by total weight
    top_paths_df = path_data.nlargest(max_paths, '_total')
    
    # Extract edges from each path
    edges = set()
    for path_str in top_paths_df.index:
        path_str = str(path_str)
        # Parse path nodes: "A -> B -> C" or "A → B → C"
        if ' -> ' in path_str:
            nodes = [n.strip() for n in path_str.split(' -> ')]
        elif ' → ' in path_str:
            nodes = [n.strip() for n in path_str.split(' → ')]
        else:
            continue
        
        # Create edge keys for consecutive node pairs
        # Add BOTH display format AND canonical format to ensure matching
        # regardless of which format aligned data uses
        for i in range(len(nodes) - 1):
            src_display = nodes[i]
            dst_display = nodes[i+1]
            src_canonical = get_canonical_name(src_display)
            dst_canonical = get_canonical_name(dst_display)
            
            # Add display name format (e.g., "GNG588(CB0038) -> GNG458(CB0890)")
            display_edge_key = f"{src_display} -> {dst_display}"
            edges.add(display_edge_key)
            
            # Add canonical name format (e.g., "GNG588 -> GNG458")
            canonical_edge_key = f"{src_canonical} -> {dst_canonical}"
            edges.add(canonical_edge_key)
    
    return edges


def _filter_aligned_by_paths(aligned: pd.DataFrame, path_data: pd.DataFrame, 
                              dataset_names: List[str], max_edges: int = 500,
                              max_paths_multiplier: float = 2.0) -> pd.DataFrame:
    """
    Filter aligned edge data to include edges from top paths.
    
    Strategy: Start with max_paths = max_edges * multiplier, extract edges from those paths.
    If not enough edges, expand the number of paths considered.
    
    Args:
        aligned: DataFrame with edge keys as index and dataset columns
        path_data: DataFrame with path keys as index and dataset columns containing min_weight
        dataset_names: List of dataset names
        max_edges: Target number of edges to include
        max_paths_multiplier: Initial multiplier for number of paths to consider
        
    Returns:
        Filtered aligned DataFrame
    """
    if path_data is None or path_data.empty or aligned.empty:
        # Fallback to original top-edges approach if no path data
        if len(aligned) > max_edges:
            available_cols = [d for d in dataset_names if d in aligned.columns]
            if available_cols:
                aligned = aligned.copy()
                aligned['_total'] = aligned[available_cols].sum(axis=1)
                aligned = aligned.nlargest(max_edges, '_total').drop(columns=['_total'])
        return aligned
    
    # Start with initial number of paths
    max_paths = int(max_edges * max_paths_multiplier)
    edges_from_paths = _extract_edges_from_paths(path_data, dataset_names, max_paths)
    
    # Expand if needed (paths share edges, so may need more paths for enough edges)
    attempts = 0
    while len(edges_from_paths) < max_edges and max_paths < len(path_data) and attempts < 5:
        max_paths = min(int(max_paths * 1.5), len(path_data))
        edges_from_paths = _extract_edges_from_paths(path_data, dataset_names, max_paths)
        attempts += 1
    
    # Filter aligned to only include edges from paths
    # Note: We keep all edges from selected paths even if slightly over max_edges
    # This preserves complete path connectivity
    if edges_from_paths:
        aligned = aligned[aligned.index.isin(edges_from_paths)]
    
    return aligned


def _analyzer_type_coverage(analyzer, point_key) -> Dict[Tuple[str, str], Any]:
    """Union type-resolution coverage for one comparison point.

    Returns {} for analyzers without the pass (older pickles, tests with
    fake analyzers) — the report must render unchanged, just unannotated.
    """
    getter = getattr(analyzer, '_type_coverage_for_query', None)
    if not callable(getter):
        return {}
    try:
        return getter(point_key) or {}
    except Exception:
        return {}


def _generate_type_coverage_card(analyzer, point_key, dataset_names: List[str],
                                 nickname_map: Dict[str, str],
                                 display_label: str = None) -> str:
    """Per-query table of union-resolved types that a dataset lacks.

    The union of types that appeared in ANY dataset, resolved into EVERY
    dataset by the type mapper; rows list only the absent side so the
    "—" cells in the matrices have an explicit verdict.
    """
    coverage = _analyzer_type_coverage(analyzer, point_key)
    if not coverage:
        return ''
    try:
        from .type_coverage import STATUS_LABELS
    except Exception:  # noqa: BLE001 - never block the report
        try:
            from src.comparison.type_coverage import STATUS_LABELS
        except ImportError:
            return ''

    datasets = [d for d in dataset_names if d in nickname_map]
    per_type: Dict[str, Dict[str, Any]] = {}
    for (type_name, dataset), entry in coverage.items():
        if entry.present:
            continue
        per_type.setdefault(type_name, {})[dataset] = entry
    if not per_type:
        return ''

    def _status_cell(entry) -> str:
        label = STATUS_LABELS.get(entry.status, entry.status)
        detail = html.escape(str(entry.detail)) if entry.detail else ''
        title = f' title="{detail}"' if detail else ''
        return (f'<span{title} style="font-size:0.82em;'
                f'color:var(--secondary-color);">'
                f'{html.escape(label)}</span>')

    rows = []
    for type_name in sorted(per_type):
        cells = []
        for dataset in datasets:
            entry = per_type[type_name].get(dataset)
            cells.append(
                '<td style="padding:3px 8px;border:1px solid '
                'var(--border-color);">✓</td>' if entry is None else
                '<td style="padding:3px 8px;border:1px solid '
                f'var(--border-color);">{_status_cell(entry)}</td>')
        rows.append(
            '<tr><td style="padding:3px 8px;border:1px solid '
            f'var(--border-color);font-weight:600;">'
            f'{html.escape(type_name)}</td>' + ''.join(cells) + '</tr>')
    if len(rows) > 200:
        rows = rows[:200] + [
            f'<tr><td colspan="{len(datasets) + 1}" '
            f'style="padding:3px 8px;border:1px solid '
            f'var(--border-color);">… {len(per_type) - 200} more types '
            f'(see comparison_results/type_resolution_union.csv)'
            f'</td></tr>']

    title = display_label or point_key
    header = ''.join(
        f'<th style="padding:3px 8px;border:1px solid var(--border-color);'
        f'font-size:0.85em;text-align:left;">{html.escape(str(nickname_map[d]))}</th>'
        for d in datasets)
    return (
        '<details class="card" style="margin:10px 0;">'
        '<summary style="cursor:pointer; font-weight:600; '
        'color:var(--primary-color);">🌀 Type coverage — absent types '
        'resolved per dataset</summary>'
        '<div style="margin-top:8px;">'
        '<p style="font-size:0.88em;">Union of types that appeared in any '
        'dataset for query <strong>'
        f'{html.escape(str(title))}</strong>, resolved into every dataset '
        'via the type mapper. Only datasets lacking the type are filled; '
        '✓ = present. Full table: '
        '<code>comparison_results/type_resolution_union.csv</code></p>'
        '<table style="border-collapse:collapse;margin-top:6px;">'
        f'<thead><tr><th style="padding:3px 8px;border:1px solid '
        f'var(--border-color);font-size:0.85em;text-align:left;">Type'
        f'</th>{header}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div></details>')


def _generate_conservation_network(analyzer, dataset_names: List[str], threshold: int,
                                    nickname_map: Dict[str, str], max_edges: int = 500,
                                    key=None, display_label: str = None,
                                    aligned_override: pd.DataFrame = None,
                                    path_override: pd.DataFrame = None,
                                    dom_key: str = None) -> str:
    """Generate network with conservation-based edge coloring and role-based node coloring.

    Args:
        max_edges: Maximum number of edges to show in the network (default 500).
                   Edges are selected from top paths to preserve path connectivity.
        key: Comparison-point key used in DOM ids and JS state. Defaults to the
             scalar ``threshold`` so Standard reports are unchanged; Custom
             combination mode passes the safe query slug.
        display_label: Card title. Defaults to the historical
             "Network at Threshold = {threshold}".
        aligned_override: Pre-resolved aligned edges (query-aware callers);
             when omitted the Standard threshold accessor is used.
        path_override: Pre-resolved path data (query-aware callers).
    """
    point_key = threshold if key is None else key
    dom_key = dom_key or _js_ident(point_key)
    js_key = json.dumps(point_key)
    card_title = display_label if display_label is not None \
        else f"Network at Threshold = {threshold}"
    if aligned_override is not None:
        aligned = aligned_override
    else:
        aligned = analyzer.get_aligned_data_for_network(threshold)

    # Get path data for this threshold to filter by top paths
    if path_override is not None:
        path_data = path_override
    else:
        path_data = None
        try:
            path_data = analyzer._get_path_data_for_threshold(threshold)
        except Exception:
            pass
    
    # Filter aligned data using path-based approach
    aligned = _filter_aligned_by_paths(aligned, path_data, dataset_names, max_edges)
    
    # Always include ALL datasets, even if they have no connections at this threshold
    # This ensures consistent behavior and proper conservation coloring
    available = dataset_names  # Use all datasets, not just those with data
    
    # Ensure aligned data has columns for all datasets (fill missing with 0)
    for d in dataset_names:
        if d not in aligned.columns:
            aligned[d] = 0
    
    num_datasets = len(available)
    nicknames = [nickname_map[d] for d in available]

    # Union type-resolution coverage (type_coverage.py): lets an absent
    # dataset say WHY it has no weight (below threshold / not in dataset /
    # unmapped) instead of a bare "—".
    type_coverage = _analyzer_type_coverage(analyzer, point_key)

    # Get source and target neurons (including all mapped type names across datasets)
    source_neurons = set()
    target_neurons = set()
    
    # Always start with parameters (these are the user's intent)
    try:
        src_list = analyzer.parameters._ensure_flat_list(analyzer.parameters.source_neurons)
        tgt_list = analyzer.parameters._ensure_flat_list(analyzer.parameters.target_neurons)
        source_neurons.update(src_list)
        target_neurons.update(tgt_list)
    except Exception:
        pass
    
    # If label mapper is available, ALSO include the mapped keys
    if hasattr(analyzer, 'label_mapper') and analyzer.label_mapper:
        source_neurons.update(analyzer.label_mapper.get_all_std_labels('source'))
        target_neurons.update(analyzer.label_mapper.get_all_std_labels('target'))
    
    # CRITICAL: If auto_type_mapping is enabled, ALSO include the resolved type names
    # for EACH dataset. This is necessary because edge data uses mapped/canonical names
    # (e.g., 'GNG588') not the original query types (e.g., 'CB0038').
    if hasattr(analyzer, 'parameters') and analyzer.parameters.auto_type_mapping:
        for dataset in dataset_names:
            # Get resolved source neurons for this dataset
            src_resolved = analyzer.parameters.get_source_neurons_for_dataset(dataset)
            source_neurons.update(src_resolved)
            # Get resolved target neurons for this dataset
            tgt_resolved = analyzer.parameters.get_target_neurons_for_dataset(dataset)
            target_neurons.update(tgt_resolved)
    
    # Build display name map for cross-dataset type names
    # Format: {canonical_name: "mcns_name (F:fafb_name/H:hemi_name)"}
    display_name_map = {}
    dataset_legend = {}  # {short_code: full_dataset_name}
    type_mapper = None
    if hasattr(analyzer, 'parameters') and analyzer.parameters.auto_type_mapping:
        type_mapper = analyzer.parameters._auto_type_mapper
        if type_mapper:
            dataset_legend = type_mapper.get_all_dataset_short_codes(dataset_names)
    
    # Build nodes and edges with conservation coloring
    nodes = []
    edges = []
    node_ids = {}
    node_roles = {}  # Track role for each node label
    node_counter = 0
    edge_data = {}  # {(source, target): {dataset: weight}}

    # Build adjacency maps for recursive dead-end detection
    outgoing_map = {}  # label -> list of target labels
    incoming_map = {}  # label -> list of source labels
    
    # Collect edge data per dataset
    for edge_key, row in aligned.iterrows():
        if ' -> ' not in str(edge_key):
            continue
        parts = str(edge_key).split(' -> ')
        source, target = parts[0], parts[1] if len(parts) > 1 else ''
        
        edge_tuple = (source, target)
        if edge_tuple not in edge_data:
            edge_data[edge_tuple] = {}
        
        # Update adjacency maps
        if source not in outgoing_map: outgoing_map[source] = []
        outgoing_map[source].append(target)
        
        if target not in incoming_map: incoming_map[target] = []
        incoming_map[target].append(source)
        
        for dataset in available:
            weight = row.get(dataset, 0)
            if weight > 0:
                edge_data[edge_tuple][dataset] = int(weight)
    
    if not edge_data:
        return '<div class="card"><p>No connections at this threshold.</p></div>'
    
    # Detect nodes with display names (format: "Name(alt1/alt2)") to enable legend
    # Note: aligned data from metrics.py already has display names applied
    import re
    # New format: "MeVPLo2(MTe07)" - canonical name followed by (alternatives) without space
    display_name_pattern = re.compile(r'^[^\(]+\([^\)]+\)$')  # Matches "Name(X)" or "Name(X/Y)" pattern
    
    # Store dataset mappings for hover labels: node -> {dataset_code: name_in_that_dataset}
    node_dataset_mappings = {}
    
    # Collect all unique node names
    all_nodes_raw = set()
    for (src, tgt) in edge_data.keys():
        all_nodes_raw.add(src)
        all_nodes_raw.add(tgt)
    
    # Detect nodes that have display names (to enable legend display)
    for node in all_nodes_raw:
        if display_name_pattern.match(node):
            # Node already has display name format - mark it for legend
            canonical = node.split('(')[0]  # Extract canonical name (no space before paren)
            display_name_map[canonical] = node  # Store mapping for legend trigger
    
    # Apply additional display name transformation if type_mapper available
    # This handles any nodes that might not have been transformed by metrics.py
    if type_mapper:
        for node in all_nodes_raw:
            # Skip nodes that already have display names
            if display_name_pattern.match(node):
                # Already has display name, but get dataset mappings for hover
                display_name, dataset_info = type_mapper.get_display_name_with_dataset_info(node.split('(')[0], dataset_names)
                node_dataset_mappings[node] = dataset_info
                continue
            display_name, dataset_info = type_mapper.get_display_name_with_dataset_info(node, dataset_names)
            if display_name != node:
                display_name_map[node] = display_name
            if dataset_info:
                node_dataset_mappings[display_name] = dataset_info
        
        # Transform edge_data keys to use display names (only for non-display nodes)
        if display_name_map:
            new_edge_data = {}
            new_outgoing_map = {}
            new_incoming_map = {}
            
            for (src, tgt), weights in edge_data.items():
                new_src = display_name_map.get(src, src)
                new_tgt = display_name_map.get(tgt, tgt)
                new_edge_data[(new_src, new_tgt)] = weights
                
                # Update adjacency maps with display names
                if new_src not in new_outgoing_map:
                    new_outgoing_map[new_src] = []
                new_outgoing_map[new_src].append(new_tgt)
                
                if new_tgt not in new_incoming_map:
                    new_incoming_map[new_tgt] = []
                new_incoming_map[new_tgt].append(new_src)
            
            edge_data = new_edge_data
            outgoing_map = new_outgoing_map
            incoming_map = new_incoming_map
    
    # Helper to extract canonical name from display name
    # Handles format: "MeVPaMe1(MTe46)" -> "MeVPaMe1"
    # Helper function to check if a node label matches any pattern
    # For merged display names like "MeVPaMe1(MTe46)", also check the canonical part
    # Also handles hemisphere suffixes (_L/_R/_U) for separate_hemispheres mode
    # First pass: collect all nodes and their initial roles (intermediate)
    all_node_labels = set()
    for (source, target) in edge_data.keys():
        all_node_labels.add(source)
        all_node_labels.add(target)
    
    # Initialize all nodes as intermediate
    for label in all_node_labels:
        node_roles[label] = 'intermediate'
    
    # Override with source/target roles (priority: source > target > intermediate)
    for label in all_node_labels:
        if matches_patterns(label, target_neurons):
            node_roles[label] = 'target'
    for label in all_node_labels:
        if matches_patterns(label, source_neurons):
            node_roles[label] = 'source'
    
    # Detect dead-end nodes: intermediate nodes that only have incoming OR only outgoing edges
    # (i.e., they're not on a complete path from source to target)
    # Recursive detection: also include nodes that only connect to dead-ends
    dead_end_nodes = set()
    
    # Iterative dead-end detection
    changed = True
    while changed:
        changed = False
        for label in all_node_labels:
            if label in dead_end_nodes:
                continue
                
            role = node_roles.get(label, 'intermediate')
            if role != 'intermediate':
                continue
            
            # Get neighbors
            out_neighbors = outgoing_map.get(label, [])
            in_neighbors = incoming_map.get(label, [])
            
            # Condition 1: All outgoing paths lead to dead-ends (or no outgoing)
            # every() returns True for empty sequence
            leads_to_dead_end = all(n in dead_end_nodes for n in out_neighbors)
            
            # Condition 2: All incoming paths come from dead-ends (or no incoming)
            comes_from_dead_end = all(n in dead_end_nodes for n in in_neighbors)
            
            if leads_to_dead_end or comes_from_dead_end:
                dead_end_nodes.add(label)
                changed = True
    
    # Node colors by role
    role_colors = {
        'source': {'background': '#ef4444', 'border': '#b91c1c'},      # Red
        'intermediate': {'background': '#3b82f6', 'border': '#1d4ed8'}, # Blue
        'target': {'background': '#8b5cf6', 'border': '#6d28d9'},      # Purple
        'dead-end': {'background': '#9ca3af', 'border': '#6b7280'}     # Gray for dead-ends
    }
    
    # Create nodes and edges
    edge_id = 0
    conserved_edge_ids = []  # Track IDs of conserved edges
    unique_edge_ids = []  # Track IDs of unique edges (only in 1 dataset)
    conserved_node_ids = set()  # Track node IDs that are part of conserved edges
    
    for (source, target), weights in edge_data.items():
        # Helper to build node hover title with dataset mapping info
        def build_node_title(node_label: str, role: str) -> str:
            lines = [f"{node_label}", f"Role: {role}"]
            if node_label in node_dataset_mappings:
                ds_info = node_dataset_mappings[node_label]
                if ds_info:
                    lines.append("Names by dataset:")
                    for code, name in sorted(ds_info.items()):
                        lines.append(f"  {code}: {name}")
            coverage_line = (
                _coverage_node_line(
                    type_coverage, node_label, available,
                    lambda d: nickname_map.get(d, d))
                if _coverage_node_line is not None else '')
            if coverage_line:
                lines.append(coverage_line)
            return '\n'.join(lines)
        
        # Add nodes with role-based coloring
        if source not in node_ids:
            node_ids[source] = node_counter
            role = node_roles.get(source, 'intermediate')
            is_dead_end = source in dead_end_nodes
            if is_dead_end:
                colors = role_colors['dead-end']
                display_role = 'dead-end'
            else:
                colors = role_colors[role]
                display_role = role
            nodes.append({
                'id': node_counter, 
                'label': source, 
                'title': build_node_title(source, display_role),
                'color': colors
            })
            node_counter += 1
        if target not in node_ids:
            node_ids[target] = node_counter
            role = node_roles.get(target, 'intermediate')
            is_dead_end = target in dead_end_nodes
            if is_dead_end:
                colors = role_colors['dead-end']
                display_role = 'dead-end'
            else:
                colors = role_colors[role]
                display_role = role
            nodes.append({
                'id': node_counter, 
                'label': target, 
                'title': build_node_title(target, display_role),
                'color': colors
            })
            node_counter += 1
        
        # Determine conservation level
        present_count = len(weights)
        is_conserved = (present_count == num_datasets)
        is_unique = (present_count == 1)
        if is_conserved:
            color = '#22c55e'  # Conserved - green
            conservation = 'Conserved'
            conserved_edge_ids.append(edge_id)
            # Track nodes that are part of conserved edges
            conserved_node_ids.add(node_ids[source])
            conserved_node_ids.add(node_ids[target])
        elif present_count > 1:
            color = '#f59e0b'  # Partial - orange
            conservation = 'Partial'
        else:
            color = '#94a3b8'  # Unique - gray
            conservation = 'Unique'
            unique_edge_ids.append(edge_id)
        
        # Build hover label with all weights
        hover_lines = [f"{source} → {target}", f"Conservation: {conservation} ({present_count}/{num_datasets})"]
        for i, d in enumerate(available):
            w = weights.get(d, 0)
            if w > 0:
                hover_lines.append(f"{nicknames[i]}: {int(w)}")
                continue
            reason = (
                _coverage_absence_note(type_coverage, d, source, target)
                if _coverage_absence_note is not None else '')
            hover_lines.append(
                f"{nicknames[i]}: —" + (f" ({reason})" if reason else ""))
        
        edges.append({
            'id': edge_id,
            'from': node_ids[source],
            'to': node_ids[target],
            'color': {'color': color, 'highlight': color},
            'width': 2 + min(present_count, 3),
            'title': '\n'.join(hover_lines)  # Real newline for vis.js tooltip
        })
        edge_id += 1
    
    # Ensure ALL source and target neurons are present in the network (even if isolated)
    # This fixes the issue where target neurons drop out if they have no connections
    # BUT: only add exact neuron names, NOT patterns with wildcards (*, .*)
    # ALSO: Don't add if the neuron is already represented in a merged node (e.g., MeVPLo2 in MeVPLo2(MTe07))
    
    def is_represented_in_nodes(label: str, node_ids: dict) -> bool:
        """Check if a label is already represented in existing nodes.
        
        Returns True if:
        - label is directly in node_ids
        - label with hemisphere suffix (_L/_R/_U) is in node_ids (e.g., aMe12 represented by aMe12_L)
        - label appears at the start of a display name like 'MeVPLo2 (F:MTe07)'
        - label appears inside parentheses of a display name (dataset-prefixed like F:MTe07)
        """
        if label in node_ids:
            return True
        # Check for hemisphere-suffixed versions (e.g., aMe12 is represented by aMe12_L or aMe12_R)
        for suffix in ['_L', '_R', '_U']:
            if (label + suffix) in node_ids:
                return True
        # Check for display names like 'MeVPLo2 (F:MTe07)' where label could be MeVPLo2 or MTe07
        for existing_label in node_ids.keys():
            # Check if label is the base name (before space and parentheses)
            # New format: "MeVPLo2 (F:MTe07)"
            if existing_label.startswith(label + ' ('):
                return True
            # Legacy format: "MeVPLo2(MTe07)"
            if existing_label.startswith(label + '('):
                return True
            # Check for hemisphere suffix before parentheses (e.g., 'aMe12_L (F:aMe12)')
            for suffix in ['_L', '_R', '_U']:
                if existing_label.startswith(label + suffix + ' (') or existing_label.startswith(label + suffix + '('):
                    return True
            # Check if label is inside parentheses (with or without dataset code)
            if '(' in existing_label and ')' in existing_label:
                paren_start = existing_label.index('(')
                paren_end = existing_label.index(')')
                inner = existing_label[paren_start + 1:paren_end]
                # Handle multiple dataset-prefixed names separated by /
                # Format: "F:MTe07/H:MTe07_variant"
                inner_names = [n.strip() for n in inner.split('/')]
                for name in inner_names:
                    # Remove dataset prefix if present (e.g., "F:MTe07" -> "MTe07")
                    if ':' in name:
                        unprefixed = name.split(':', 1)[1]
                        if label == unprefixed:
                            return True
                    elif label == name:
                        return True
        return False
    
    # Check if separate_hemispheres is enabled
    separate_hemispheres = getattr(getattr(analyzer, 'parameters', None), 'separate_hemispheres', False)
    
    def has_hemisphere_suffix(name: str) -> bool:
        """Check if a name already has a hemisphere suffix (_L/_R/_U)."""
        return name.endswith('_L') or name.endswith('_R') or name.endswith('_U')
    
    # Track which canonical names have been added as isolated nodes to avoid duplicates
    # This handles the case where CB0038 (FAFB/BANC) and GNG588 (MCNS) are the same type
    added_canonical_sources = set()
    added_canonical_targets = set()

    # B6 (query-anchored merge policy): raw per-dataset names resolve to
    # their group label before the represented-check — one group = one
    # node, and an isolated duplicate can never resurface for a member
    # whose group node already exists.
    merge_policy = None
    try:
        merge_policy = analyzer._merge_policy_or_none()
    except Exception:
        merge_policy = None

    for label in source_neurons:
        # Skip patterns - they're not actual neuron names
        if '*' in label or '.*' in label:
            continue
        # When separate_hemispheres is True, skip adding isolated nodes for base names
        # (without _L/_R/_U suffix). The actual hemisphere-suffixed nodes will be added
        # from edge data if they have connections.
        if separate_hemispheres and not has_hemisphere_suffix(label):
            continue
        if not is_represented_in_nodes(label, node_ids):
            # B6: the group's node supersedes this raw name.
            _group_label = (
                merge_policy.label_for_name(label)
                if merge_policy is not None else None)
            if _group_label is not None and (
                    _group_label in node_ids
                    or is_represented_in_nodes(_group_label, node_ids)):
                continue
            # Get the merged display name if type_mapper is available
            # This ensures CB0038 and GNG588 become one node: GNG588(CB0038)
            display_label = label
            dataset_info = {}
            if _group_label is not None:
                display_label = _group_label
                dataset_info = {
                    ds: '/'.join(names)
                    for ds, names in
                    merge_policy.names_by_dataset(_group_label).items()
                }
            elif type_mapper:
                display_label, dataset_info = type_mapper.get_display_name_with_dataset_info(label, dataset_names)
            
            # Extract canonical name to avoid adding duplicates
            canonical = display_label.split('(')[0] if '(' in display_label else display_label
            if canonical in added_canonical_sources:
                continue  # Already added this type with merged display name
            added_canonical_sources.add(canonical)
            
            # Also check if display_label is already in node_ids (may have been added by another name)
            if display_label in node_ids:
                continue
            
            # Build hover title with dataset info
            title_lines = [f"{display_label}", "Role: source (isolated)"]
            if dataset_info:
                title_lines.append("Names by dataset:")
                for code, name in sorted(dataset_info.items()):
                    title_lines.append(f"  {code}: {name}")
                node_dataset_mappings[display_label] = dataset_info
            _cov_line = (
                _coverage_node_line(
                    type_coverage, display_label, available,
                    lambda d: nickname_map.get(d, d))
                if _coverage_node_line is not None else '')
            if _cov_line:
                title_lines.append(_cov_line)

            node_ids[display_label] = node_counter
            node_roles[display_label] = 'source'
            nodes.append({
                'id': node_counter,
                'label': display_label,
                'title': '\n'.join(title_lines),
                'color': role_colors['source']
            })
            node_counter += 1
            
    for label in target_neurons:
        # Skip patterns - they're not actual neuron names
        if '*' in label or '.*' in label:
            continue
        # When separate_hemispheres is True, skip adding isolated nodes for base names
        # (without _L/_R/_U suffix). The actual hemisphere-suffixed nodes will be added
        # from edge data if they have connections.
        if separate_hemispheres and not has_hemisphere_suffix(label):
            continue
        if not is_represented_in_nodes(label, node_ids):
            # B6: the group's node supersedes this raw name.
            _group_label = (
                merge_policy.label_for_name(label)
                if merge_policy is not None else None)
            if _group_label is not None and (
                    _group_label in node_ids
                    or is_represented_in_nodes(_group_label, node_ids)):
                continue
            # Get the merged display name if type_mapper is available
            display_label = label
            dataset_info = {}
            if _group_label is not None:
                display_label = _group_label
                dataset_info = {
                    ds: '/'.join(names)
                    for ds, names in
                    merge_policy.names_by_dataset(_group_label).items()
                }
            elif type_mapper:
                display_label, dataset_info = type_mapper.get_display_name_with_dataset_info(label, dataset_names)
            
            # Extract canonical name to avoid adding duplicates
            canonical = display_label.split('(')[0] if '(' in display_label else display_label
            if canonical in added_canonical_targets:
                continue  # Already added this type with merged display name
            added_canonical_targets.add(canonical)
            
            # Also check if display_label is already in node_ids
            if display_label in node_ids:
                continue
            
            # Build hover title with dataset info
            title_lines = [f"{display_label}", "Role: target (isolated)"]
            if dataset_info:
                title_lines.append("Names by dataset:")
                for code, name in sorted(dataset_info.items()):
                    title_lines.append(f"  {code}: {name}")
                node_dataset_mappings[display_label] = dataset_info
            _cov_line = (
                _coverage_node_line(
                    type_coverage, display_label, available,
                    lambda d: nickname_map.get(d, d))
                if _coverage_node_line is not None else '')
            if _cov_line:
                title_lines.append(_cov_line)

            node_ids[display_label] = node_counter
            node_roles[display_label] = 'target'
            nodes.append({
                'id': node_counter,
                'label': display_label,
                'title': '\n'.join(title_lines),
                'color': role_colors['target']
            })
            node_counter += 1
    
    div_id = f"network_{dom_key}"
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)
    conserved_ids_json = json.dumps(conserved_edge_ids)
    unique_ids_json = json.dumps(unique_edge_ids)
    conserved_node_ids_json = json.dumps(list(conserved_node_ids))
    dead_end_node_ids = [node_ids[label] for label in dead_end_nodes if label in node_ids]
    dead_end_node_ids_json = json.dumps(dead_end_node_ids)
    # Map node IDs to roles for dynamic dead-end calculation
    node_roles_by_id = {node_ids[label]: role for label, role in node_roles.items() if label in node_ids}
    node_roles_json = json.dumps(node_roles_by_id)
    
    # Count dead-end nodes and conserved edges for display
    dead_end_count = len(dead_end_nodes)
    dead_end_info = f' | <span style="color: #9ca3af;"><strong>{dead_end_count}</strong> dead-end</span>' if dead_end_count > 0 else ''
    conserved_count = len(conserved_edge_ids)
    unique_count = len(unique_edge_ids)
    conserved_info = f' | <span style="color: #22c55e;"><strong>{conserved_count}</strong> conserved</span>'
    unique_info = f' | <span style="color: #94a3b8;"><strong>{unique_count}</strong> unique</span>' if unique_count > 0 else ''
    
    # Generate dataset legend HTML for cross-dataset type name display
    dataset_legend_html = ""
    if dataset_legend and display_name_map:
        legend_parts = [f'<span title="{full_name}" style="cursor: help;"><b>{code}</b>={full_name}</span>' 
                        for code, full_name in sorted(dataset_legend.items())]
        if legend_parts:
            dataset_legend_html = f'''
            <div style="margin-top: 8px; padding: 8px; background: #f8f9fa; border-radius: 6px; font-size: 11px;">
                <span style="color: #666;">📝 Dataset codes in node names:</span> {" | ".join(legend_parts)}
            </div>
            '''

    mirror_disabled_attr = '' if separate_hemispheres else 'disabled'
    mirror_btn_style = (
        'padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border-color); '
        'background: #64748b; color: white; font-size: 12px; white-space: nowrap; '
        + ('cursor: pointer;' if separate_hemispheres else 'cursor: not-allowed; opacity: 0.5;')
    )
    mirror_btn_title = '' if separate_hemispheres else 'Hemisphere mirroring requires separate_hemispheres=True'
    
    return f'''
        <div class="card">
            <h3>{html.escape(card_title)}</h3>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div style="color: var(--secondary-color);">
                    <strong>{len(nodes)}</strong> neurons | <strong>{len(edges)}</strong> edges{conserved_info}{unique_info}{dead_end_info}
                </div>
                <div style="display: flex; gap: 8px;">
                    <button id="filter_btn_{dom_key}" onclick="toggleNetworkFilter({_html_js_arg(point_key)})"
                        style="padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border-color);
                               background: var(--secondary-color); color: white; cursor: pointer; font-size: 12px; white-space: nowrap;">
                        🌐 Show All
                    </button>
                    <button id="deadend_btn_{dom_key}" onclick="toggleDeadEndNodes({_html_js_arg(point_key)})"
                        style="padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border-color);
                               background: var(--secondary-color); color: white; cursor: pointer; font-size: 12px; white-space: nowrap;">
                        👁️ Show Dead-ends
                    </button>
                    <button id="physics_btn_{dom_key}" onclick="toggleNetworkPhysics({_html_js_arg(point_key)})"
                        style="padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border-color);
                               background: var(--secondary-color); color: white; cursor: pointer; font-size: 12px; white-space: nowrap;">
                        📌 Static Mode
                    </button>
                    <button id="mirror_btn_{dom_key}" onclick="toggleHemisphereMirror({_html_js_arg(point_key)})" {mirror_disabled_attr}
                        title="{mirror_btn_title}"
                        style="{mirror_btn_style}">
                        🪞 Mirror Hemispheres
                    </button>
                </div>
            </div>
            {dataset_legend_html}
            <div id="{div_id}" style="height: 900px; border: 1px solid var(--border-color); border-radius: 8px;"></div>
        </div>
        <script>
            (function() {{
                if (typeof vis === 'undefined') return;
                const allNodes = {nodes_json};
                const allEdges = {edges_json};
                const conservedEdgeIds = new Set({conserved_ids_json});
                const uniqueEdgeIds = new Set({unique_ids_json});
                const conservedNodeIds = new Set({conserved_node_ids_json});
                const deadEndNodeIds = new Set({dead_end_node_ids_json});
                const nodeRoles = {node_roles_json};
                
                const nodes = new vis.DataSet(allNodes);
                const edges = new vis.DataSet(allEdges);
                const container = document.getElementById({json.dumps(div_id)});
                const data = {{ nodes: nodes, edges: edges }};
                
                // Initialize with hierarchical layout for proper layer-like positioning
                // Direction 'UD' = top-to-bottom flow (rotated 90° clockwise from LR)
                const options = {{
                    nodes: {{
                        shape: 'dot',
                        size: 18,
                        font: {{ size: 11 }},
                        borderWidth: 2
                    }},
                    edges: {{
                        arrows: {{ to: {{ enabled: true, scaleFactor: 0.7 }} }},
                        smooth: {{ type: 'curvedCW', roundness: 0.1 }}
                    }},
                    layout: {{
                        hierarchical: {{
                            enabled: true,
                            direction: 'UD',
                            sortMethod: 'directed',
                            levelSeparation: 120,
                            nodeSpacing: 100,
                            treeSpacing: 120
                        }}
                    }},
                    physics: {{
                        enabled: false
                    }},
                    interaction: {{
                        hover: true,
                        tooltipDelay: 50,
                        multiselect: true,
                        dragNodes: true
                    }}
                }};
                
                const network = new vis.Network(container, data, options);
                
                // After initial layout, disable hierarchical for free movement
                setTimeout(function() {{
                    network.setOptions({{ layout: {{ hierarchical: false }} }});
                    network.fit({{ animation: true }});
                }}, 200);
                
                // Register with global toggle (store original data for filtering)
                window.allNetworks[{js_key}] = {{
                    network: network,
                    nodes: nodes,
                    edges: edges,
                    allNodes: allNodes,
                    allEdges: allEdges,
                    conservedEdgeIds: conservedEdgeIds,
                    uniqueEdgeIds: uniqueEdgeIds,
                    conservedNodeIds: conservedNodeIds,
                    deadEndNodeIds: deadEndNodeIds,
                    nodeRoles: nodeRoles
                }};

                // Auto-enable the mirror layout when Separate Hemispheres (L/R) is on
                if (window.hemisphereMirrorEnabled[{js_key}]) {{
                    applyHemisphereMirror({js_key});
                }}
            }})();
        </script>
'''


def _generate_dataset_network(analyzer, dataset: str, thresholds: List[int],
                               nickname_map: Dict[str, str], max_edges: int = 500,
                               keys: List = None, key_labels: List[str] = None,
                               aligned_getter=None, path_getter=None,
                               key_noun_plural: str = 'thresholds',
                               key_noun_singular: str = 'threshold',
                               dom_key: str = None) -> str:
    """Generate network visualization for a single dataset across all points.

    Standard mode iterates the scalar ``thresholds``; Custom combination mode
    passes ``keys`` (query ids), ``key_labels`` and query-aware getters so the
    same cross-point view is rendered over the query axis.
    """
    point_keys = list(thresholds) if keys is None else list(keys)
    if key_labels is None:
        key_labels = [f"t={k}" for k in point_keys]
    key_label_by_key = {k: key_labels[i] for i, k in enumerate(point_keys)}
    point_dom_keys = _unique_dom_keys(point_keys)
    point_dom_key_by_key = {
        key: point_dom_keys[index]
        for index, key in enumerate(point_keys)
    }
    num_thresholds = len(point_keys)

    nick = nickname_map[dataset]
    dataset_dom_key = dom_key or _js_ident(nick)

    # Get path data for filtering (use first point with data)
    all_path_edges = set()
    for k in point_keys:
        try:
            if path_getter is not None:
                path_data = path_getter(k)
            else:
                path_data = analyzer._get_path_data_for_threshold(k)
            if path_data is not None and not path_data.empty:
                # Get edges from top paths for this point
                edges_from_paths = _extract_edges_from_paths(path_data, [dataset], max_paths=int(max_edges * 2))
                all_path_edges.update(edges_from_paths)
        except Exception:
            pass
    
    # Collect edges: {(source, target): {threshold: weight}}
    edge_weights = {}  # {(source, target): {t: weight}}
    all_nodes = set()
    
    # First pass: get total weights per edge for prioritization
    edge_total_weights = {}
    
    for k in point_keys:
        if aligned_getter is not None:
            aligned = aligned_getter(k)
        else:
            aligned = analyzer.get_aligned_data(k)
        if dataset not in aligned.columns:
            continue

        # Vectorized extraction of edges
        valid_mask = aligned[dataset] > 0
        for edge_key in aligned.loc[valid_mask].index:
            if ' -> ' not in str(edge_key):
                continue

            # If we have path edges, filter to only those
            if all_path_edges and edge_key not in all_path_edges:
                continue

            parts = str(edge_key).split(' -> ')
            source, target = parts[0], parts[1] if len(parts) > 1 else ''
            weight = aligned.loc[edge_key, dataset]
            # Handle Series (duplicate indices)
            if hasattr(weight, 'iloc'):
                weight = weight.iloc[0]
            weight = int(weight)
            if weight > 0:
                edge_tuple = (source, target)
                if edge_tuple not in edge_weights:
                    edge_weights[edge_tuple] = {}
                    edge_total_weights[edge_tuple] = 0
                edge_weights[edge_tuple][k] = weight
                edge_total_weights[edge_tuple] += weight
    
    # Limit to top edges if still too many
    if len(edge_weights) > max_edges:
        top_edges = sorted(edge_total_weights.items(), key=lambda x: -x[1])[:max_edges]
        top_edge_set = set(e[0] for e in top_edges)
        edge_weights = {k: v for k, v in edge_weights.items() if k in top_edge_set}
    
    # Build node set
    for (source, target) in edge_weights:
        all_nodes.add(source)
        all_nodes.add(target)
    
    if not all_nodes:
        return (f'<div class="card"><p>No connections for {nick} at any '
                f'{key_noun_singular}.</p></div>')
    
    # Get source and target neurons for role coloring
    source_neurons = set()
    target_neurons = set()
    
    # If label mapper is available, use standardized labels
    if hasattr(analyzer, 'label_mapper') and analyzer.label_mapper:
        source_neurons = set(analyzer.label_mapper.get_all_std_labels('source'))
        target_neurons = set(analyzer.label_mapper.get_all_std_labels('target'))
    else:
        # Otherwise use parameters
        try:
            src_list = analyzer.parameters._ensure_flat_list(analyzer.parameters.source_neurons)
            tgt_list = analyzer.parameters._ensure_flat_list(analyzer.parameters.target_neurons)
            source_neurons = set(src_list)
            target_neurons = set(tgt_list)
        except Exception:
            pass
    
    # Helper to extract canonical name from display name like "MeVPaMe1(MTe46)" -> "MeVPaMe1"
    # Determine node roles
    node_roles = {}
    for label in all_nodes:
        node_roles[label] = 'intermediate'
    for label in all_nodes:
        if matches_patterns(label, target_neurons):
            node_roles[label] = 'target'
    for label in all_nodes:
        if matches_patterns(label, source_neurons):
            node_roles[label] = 'source'
    
    # Track incoming/outgoing for dead-end detection
    has_outgoing = set()
    has_incoming = set()
    for (source, target) in edge_weights.keys():
        has_outgoing.add(source)
        has_incoming.add(target)
    
    # Detect dead-end nodes
    dead_end_nodes = set()
    for label in all_nodes:
        role = node_roles.get(label, 'intermediate')
        if role == 'intermediate':
            only_incoming = label in has_incoming and label not in has_outgoing
            only_outgoing = label in has_outgoing and label not in has_incoming
            if only_incoming or only_outgoing:
                dead_end_nodes.add(label)
    
    # Node colors by role
    role_colors = {
        'source': {'background': '#ef4444', 'border': '#b91c1c'},
        'intermediate': {'background': '#3b82f6', 'border': '#1d4ed8'},
        'target': {'background': '#8b5cf6', 'border': '#6d28d9'},
        'dead-end': {'background': '#9ca3af', 'border': '#6b7280'}
    }
    
    # Create nodes
    nodes = []
    node_ids = {}
    node_counter = 0
    for label in sorted(all_nodes):
        node_ids[label] = node_counter
        role = node_roles.get(label, 'intermediate')
        is_dead_end = label in dead_end_nodes
        display_role = 'dead-end' if is_dead_end else role
        colors = role_colors[display_role]
        nodes.append({
            'id': node_counter,
            'label': label,
            'title': f"{label} ({display_role})",
            'color': colors
        })
        node_counter += 1
    
    # Create unique edges with conservation-style coloring
    # gray (1 threshold) -> orange (some) -> green (all thresholds)
    edges = []
    edge_id = 0
    edges_by_key = {k: [] for k in point_keys}  # Track which edges appear at each point
    conserved_edge_ids = []  # Edges at ALL points
    unique_edge_ids = []  # Edges at only 1 point

    for (source, target), t_weights in edge_weights.items():
        thresholds_present = len(t_weights)

        # Conservation-style coloring
        if thresholds_present == num_thresholds:
            color = '#22c55e'  # All points - green
            conservation = f'All {key_noun_plural}'
            conserved_edge_ids.append(edge_id)
        elif thresholds_present > 1:
            color = '#f59e0b'  # Partial - orange
            conservation = f'{thresholds_present}/{num_thresholds} {key_noun_plural}'
        else:
            color = '#94a3b8'  # Unique - gray
            conservation = f'1 {key_noun_singular} only'
            unique_edge_ids.append(edge_id)

        # Build hover with all point weights
        hover_lines = [f"{source} → {target}", f"Conservation: {conservation}"]
        for k in point_keys:
            w = t_weights.get(k, 0)
            status = f"{int(w)}" if w > 0 else "—"
            hover_lines.append(f"{key_label_by_key[k]}: {status}")

        edges.append({
            'id': edge_id,
            'from': node_ids[source],
            'to': node_ids[target],
            'color': {'color': color, 'highlight': color},
            'width': 2 + min(thresholds_present, 3),
            'title': '\n'.join(hover_lines),
            'thresholds': list(t_weights.keys())  # Store which points this edge appears at
        })

        # Track edges by point for filtering
        for k in t_weights.keys():
            edges_by_key[k].append(edge_id)

        edge_id += 1

    div_id = f"network_{dataset_dom_key}_dataset"
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)
    # Runtime filter state uses the raw comparison-point key.  DOM ids use a
    # sanitized key, so keep the two namespaces separate.
    edges_by_threshold_json = json.dumps({str(k): ids for k, ids in edges_by_key.items()})
    thresholds_json = json.dumps(point_keys)
    conserved_ids_json = json.dumps(conserved_edge_ids)
    unique_ids_json = json.dumps(unique_edge_ids)
    dead_end_node_ids = [node_ids[label] for label in dead_end_nodes if label in node_ids]
    dead_end_node_ids_json = json.dumps(dead_end_node_ids)
    node_roles_by_id = {node_ids[label]: role for label, role in node_roles.items() if label in node_ids}
    node_roles_json = json.dumps(node_roles_by_id)
    
    # Count statistics
    total_edges = len(edges)
    conserved_count = len(conserved_edge_ids)
    unique_count = len(unique_edge_ids)
    dead_end_count = len(dead_end_nodes)
    
    conserved_info = f'<span style="color: #22c55e;"><strong>{conserved_count}</strong> all-t</span>'
    partial_count = total_edges - conserved_count - unique_count
    partial_info = f'<span style="color: #f59e0b;"><strong>{partial_count}</strong> partial</span>' if partial_count > 0 else ''
    unique_info = f'<span style="color: #94a3b8;"><strong>{unique_count}</strong> single-t</span>' if unique_count > 0 else ''
    dead_end_info = f'<span style="color: #9ca3af;"><strong>{dead_end_count}</strong> dead-end</span>' if dead_end_count > 0 else ''
    
    stats_parts = [s for s in [conserved_info, partial_info, unique_info, dead_end_info] if s]
    stats_str = ' | '.join(stats_parts)
    
    # Build point filter buttons
    threshold_buttons = []
    for k in point_keys:
        count = len(edges_by_key[k])
        k_dom = point_dom_key_by_key[k]
        btn_html = (f'<button id="t_btn_{dataset_dom_key}_{k_dom}" class="threshold-filter-btn active" '
                    f'onclick="toggleDatasetThreshold({_html_js_arg(nick)}, {_html_js_arg(k)}, '
                    f'{_html_js_arg(dataset_dom_key)}, {_html_js_arg(k_dom)})" '
                    f'style="padding: 4px 8px; border-radius: 4px; border: 1px solid var(--border-color); '
                    f'background: #22c55e; color: white; cursor: pointer; font-size: 11px; margin-right: 4px;">'
                    f'{html.escape(key_label_by_key[k])} ({count})</button>')
        threshold_buttons.append(btn_html)

    return f'''
        <div class="card">
            <h3>{html.escape(str(nick))}: Cross-{key_noun_plural.title()} Network</h3>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div style="color: var(--secondary-color);">
                    <strong>{len(nodes)}</strong> neurons | <strong>{total_edges}</strong> edges | {stats_str}
                </div>
                <div style="display: flex; gap: 8px;">
                    <button id="deadend_btn_{dataset_dom_key}" onclick="toggleDatasetDeadEnd({_html_js_arg(nick)}, {_html_js_arg(dataset_dom_key)})"
                        style="padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border-color); 
                               background: var(--secondary-color); color: white; cursor: pointer; font-size: 12px;">
                        👁️ Show Dead-ends
                    </button>
                    <button id="physics_btn_{dataset_dom_key}" onclick="toggleDatasetNetworkPhysics({_html_js_arg(nick)}, {_html_js_arg(dataset_dom_key)})"
                        style="padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border-color); 
                               background: var(--secondary-color); color: white; cursor: pointer; font-size: 12px;">
                        📌 Static Mode
                    </button>
                </div>
            </div>
            <div style="margin-bottom: 10px;">
                <strong>Show {key_noun_plural}:</strong> {''.join(threshold_buttons)}
                <span style="margin-left: 10px; color: var(--secondary-color);">
                    <span style="color: #22c55e;">● All</span>
                    <span style="color: #f59e0b; margin-left: 8px;">● Partial</span>
                    <span style="color: #94a3b8; margin-left: 8px;">● Single</span>
                </span>
            </div>
            <div id="{div_id}" style="height: 900px; border: 1px solid var(--border-color); border-radius: 8px;"></div>
        </div>
        <script>
            (function() {{
                if (typeof vis === 'undefined') return;
                const allNodes = {nodes_json};
                const allEdges = {edges_json};
                const edgesByThreshold = {edges_by_threshold_json};
                const thresholds = {thresholds_json};
                const conservedEdgeIds = new Set({conserved_ids_json});
                const uniqueEdgeIds = new Set({unique_ids_json});
                const deadEndNodeIds = new Set({dead_end_node_ids_json});
                const nodeRoles = {node_roles_json};
                
                const nodes = new vis.DataSet(allNodes);
                const edges = new vis.DataSet(allEdges);
                const container = document.getElementById({json.dumps(div_id)});
                const data = {{ nodes: nodes, edges: edges }};
                
                const options = {{
                    nodes: {{
                        shape: 'dot',
                        size: 18,
                        font: {{ size: 11 }},
                        borderWidth: 2
                    }},
                    edges: {{
                        arrows: {{ to: {{ enabled: true, scaleFactor: 0.7 }} }},
                        smooth: {{ type: 'curvedCW', roundness: 0.1 }}
                    }},
                    layout: {{
                        hierarchical: {{
                            enabled: true,
                            direction: 'UD',
                            sortMethod: 'directed',
                            levelSeparation: 120,
                            nodeSpacing: 100,
                            treeSpacing: 120
                        }}
                    }},
                    physics: {{ enabled: false }},
                    interaction: {{
                        hover: true,
                        tooltipDelay: 50,
                        multiselect: true,
                        dragNodes: true
                    }}
                }};
                
                const network = new vis.Network(container, data, options);
                
                setTimeout(function() {{
                    network.setOptions({{ layout: {{ hierarchical: false }} }});
                    network.fit({{ animation: true }});
                }}, 200);
                
                // Register with global toggle
                window.allNetworks[{json.dumps(dom_key)} + '__' + {json.dumps(nick)} + '_dataset'] = {{
                    network: network,
                    nodes: nodes,
                    edges: edges,
                    allNodes: allNodes,
                    allEdges: allEdges,
                    edgesByThreshold: edgesByThreshold,
                    thresholds: thresholds,
                    conservedEdgeIds: conservedEdgeIds,
                    uniqueEdgeIds: uniqueEdgeIds,
                    deadEndNodeIds: deadEndNodeIds,
                    nodeRoles: nodeRoles,
                    activeThresholds: new Set(thresholds.map(String)),
                    hideDeadEnds: false
                }};
            }})();
            
            // Dataset network state
            if (!window.datasetNetworkPhysicsEnabled) {{
                window.datasetNetworkPhysicsEnabled = {{}};
            }}
            window.datasetNetworkPhysicsEnabled[{json.dumps(dom_key)} + '__' + {json.dumps(nick)}] = false;
            
            function toggleDatasetThreshold(dataset, threshold, datasetDomKey, thresholdDomKey) {{
                const netData = window.allNetworks[datasetDomKey + '__' + dataset + '_dataset'];
                if (!netData) return;
                
                const tStr = String(threshold);
                const btn = document.getElementById('t_btn_' + datasetDomKey + '_' + thresholdDomKey);
                
                if (netData.activeThresholds.has(tStr)) {{
                    netData.activeThresholds.delete(tStr);
                    btn.style.background = '#94a3b8';
                }} else {{
                    netData.activeThresholds.add(tStr);
                    btn.style.background = '#22c55e';
                }}
                
                updateDatasetNetworkDisplay(dataset);
            }}
            
            function toggleDatasetDeadEnd(dataset, datasetDomKey) {{
                const netData = window.allNetworks[datasetDomKey + '__' + dataset + '_dataset'];
                if (!netData) return;
                
                netData.hideDeadEnds = !netData.hideDeadEnds;
                const btn = document.getElementById('deadend_btn_' + datasetDomKey);
                
                if (netData.hideDeadEnds) {{
                    btn.innerHTML = '🚫 Hide Dead-ends';
                    btn.style.background = '#f59e0b';
                }} else {{
                    btn.innerHTML = '👁️ Show Dead-ends';
                    btn.style.background = 'var(--secondary-color)';
                }}
                
                updateDatasetNetworkDisplay(dataset);
            }}
            
            function updateDatasetNetworkDisplay(dataset) {{
                const netData = window.allNetworks[datasetDomKey + '__' + dataset + '_dataset'];
                if (!netData) return;
                
                const net = netData.network;
                const edges = netData.edges;
                const nodes = netData.nodes;
                const allEdges = netData.allEdges;
                const allNodes = netData.allNodes;
                const edgesByThreshold = netData.edgesByThreshold;
                const activeThresholds = netData.activeThresholds;
                const nodeRoles = netData.nodeRoles;
                
                // Get edge IDs that appear in at least one active threshold
                let activeEdgeIds = new Set();
                activeThresholds.forEach(t => {{
                    (edgesByThreshold[t] || []).forEach(id => activeEdgeIds.add(id));
                }});
                
                // Filter edges to only those with at least one active threshold
                let filteredEdges = allEdges.filter(e => {{
                    const edgeThresholds = e.thresholds || [];
                    return edgeThresholds.some(t => activeThresholds.has(String(t)));
                }});
                
                // Get connected nodes
                let connectedNodeIds = new Set();
                filteredEdges.forEach(e => {{
                    connectedNodeIds.add(e.from);
                    connectedNodeIds.add(e.to);
                }});
                
                let filteredNodes = allNodes.filter(n => connectedNodeIds.has(n.id));
                
                // Recalculate dead-ends based on filtered edges
                // IMPORTANT: Source and target nodes are NEVER dead-ends
                let hasOutgoing = new Set();
                let hasIncoming = new Set();
                filteredEdges.forEach(e => {{
                    hasOutgoing.add(e.from);
                    hasIncoming.add(e.to);
                }});
                
                let dynamicDeadEndNodeIds = new Set();
                filteredNodes.forEach(n => {{
                    const role = nodeRoles[n.id] || 'intermediate';
                    // Only intermediate nodes can be dead-ends
                    // Source and target nodes are NEVER dead-ends
                    if (role === 'intermediate') {{
                        const onlyIn = hasIncoming.has(n.id) && !hasOutgoing.has(n.id);
                        const onlyOut = hasOutgoing.has(n.id) && !hasIncoming.has(n.id);
                        if (onlyIn || onlyOut) {{
                            dynamicDeadEndNodeIds.add(n.id);
                        }}
                    }}
                }});
                
                // Update node colors
                // Source and target always keep their colors, never become dead-end
                const roleColors = {{
                    'source': {{ background: '#ef4444', border: '#b91c1c' }},
                    'intermediate': {{ background: '#3b82f6', border: '#1d4ed8' }},
                    'target': {{ background: '#8b5cf6', border: '#6d28d9' }},
                    'dead-end': {{ background: '#9ca3af', border: '#6b7280' }}
                }};
                
                filteredNodes = filteredNodes.map(n => {{
                    const role = nodeRoles[n.id] || 'intermediate';
                    // Source and target are NEVER dead-ends - they keep their role
                    const isDeadEnd = (role === 'intermediate') && dynamicDeadEndNodeIds.has(n.id);
                    const displayRole = isDeadEnd ? 'dead-end' : role;
                    const colors = roleColors[displayRole];
                    return {{
                        ...n,
                        color: colors,
                        title: n.label + ' (' + displayRole + ')'
                    }};
                }});
                
                // Apply dead-end filter if enabled
                // NEVER hide source or target nodes, even if dead-end toggle is on
                if (netData.hideDeadEnds) {{
                    filteredNodes = filteredNodes.filter(n => {{
                        const role = nodeRoles[n.id] || 'intermediate';
                        // Keep source and target nodes always
                        if (role === 'source' || role === 'target') return true;
                        // Filter out dead-end intermediates
                        return !dynamicDeadEndNodeIds.has(n.id);
                    }});
                    filteredEdges = filteredEdges.filter(e => {{
                        const fromRole = nodeRoles[e.from] || 'intermediate';
                        const toRole = nodeRoles[e.to] || 'intermediate';
                        // Keep edges to/from source or target
                        if (fromRole === 'source' || fromRole === 'target') return true;
                        if (toRole === 'source' || toRole === 'target') return true;
                        // Filter out edges involving dead-end intermediates
                        return !dynamicDeadEndNodeIds.has(e.from) && !dynamicDeadEndNodeIds.has(e.to);
                    }});
                }}
                
                // Apply to network
                edges.clear();
                edges.add(filteredEdges);
                nodes.clear();
                nodes.add(filteredNodes);
                
                // Reinitialize layout
                net.setOptions({{
                    nodes: {{ size: 18, font: {{ size: 11 }} }},
                    edges: {{ smooth: {{ type: 'curvedCW', roundness: 0.1 }} }},
                    layout: {{
                        hierarchical: {{
                            enabled: true,
                            direction: 'UD',
                            sortMethod: 'directed',
                            levelSeparation: 120,
                            nodeSpacing: 80,
                            treeSpacing: 100
                        }}
                    }},
                    physics: {{ enabled: false }}
                }});
                setTimeout(() => {{
                    net.setOptions({{ layout: {{ hierarchical: false }} }});
                    net.fit({{ animation: true }});
                }}, 200);
            }}
            
            function toggleDatasetNetworkPhysics(dataset, datasetDomKey) {{
                window.datasetNetworkPhysicsEnabled[datasetDomKey + '__' + dataset] = !window.datasetNetworkPhysicsEnabled[datasetDomKey + '__' + dataset];
                const btn = document.getElementById('physics_btn_' + datasetDomKey);
                const netData = window.allNetworks[datasetDomKey + '__' + dataset + '_dataset'];
                
                if (!netData) return;
                
                const net = netData.network;
                
                if (window.datasetNetworkPhysicsEnabled[datasetDomKey + '__' + dataset]) {{
                    btn.innerHTML = '💥 Duang Mode';
                    btn.style.background = 'var(--primary-color)';
                    net.setOptions({{
                        nodes: {{ size: 20, font: {{ size: 12 }} }},
                        edges: {{ smooth: {{ type: 'continuous', roundness: 0.2 }} }},
                        layout: {{ hierarchical: false }},
                        physics: {{
                            enabled: true,
                            solver: 'forceAtlas2Based',
                            forceAtlas2Based: {{
                                gravitationalConstant: -80,
                                centralGravity: 0.005,
                                springLength: 120,
                                springConstant: 0.06,
                                damping: 0.5,
                                avoidOverlap: 0.8
                            }},
                            stabilization: {{ enabled: true, iterations: 100, updateInterval: 25 }},
                            minVelocity: 0.5
                        }}
                    }});
                    net.once('stabilized', () => {{ net.fit({{ animation: true }}); }});
                }} else {{
                    btn.innerHTML = '📌 Static Mode';
                    btn.style.background = 'var(--secondary-color)';
                    net.setOptions({{
                        nodes: {{ size: 18, font: {{ size: 11 }} }},
                        edges: {{ smooth: {{ type: 'curvedCW', roundness: 0.1 }} }},
                        layout: {{
                            hierarchical: {{
                                enabled: true,
                                direction: 'UD',
                                sortMethod: 'directed',
                                levelSeparation: 120,
                                nodeSpacing: 80,
                                treeSpacing: 100
                            }}
                        }},
                        physics: {{ enabled: false }}
                    }});
                    setTimeout(() => {{
                        net.setOptions({{ layout: {{ hierarchical: false }} }});
                        net.fit({{ animation: true }});
                    }}, 200);
                }}
            }}
        </script>
'''


def _generate_edge_matrices_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                     nickname_map: Dict[str, str]) -> str:
    """Generate edge presence matrices section with dual toggle (by threshold and by dataset)."""
    html_parts = []
    nicknames = [nickname_map[d] for d in dataset_names]
    threshold_dom_keys = _unique_dom_keys(thresholds)
    dataset_dom_keys = _unique_dom_keys(nicknames)
    nicknames_json = json.dumps(nicknames)
    thresholds_json = json.dumps(thresholds)
    
    html_parts.append(f"""
        <div id="edge-matrices" class="section">
            <div class="section-header">🔗 Edge Presence Matrices</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Edge presence across datasets. ✔️ = present with weight, ❌ = absent.
                </p>
                
                <!-- Toggle Mode Selector -->
                <div style="margin-bottom: 15px;">
                    <span style="font-weight: 600; margin-right: 10px;">View by:</span>
                    <button class="tab-btn active" id="edge_mode_threshold" onclick="switchEdgeMode('threshold')">Threshold</button>
                    <button class="tab-btn" id="edge_mode_dataset" onclick="switchEdgeMode('dataset')">Dataset</button>
                </div>
                
                <!-- By Threshold View -->
                <div id="edge_by_threshold" class="tabs">
                    <div class="tab-buttons">
""")
    
    for i, t in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(
            f'<button class="tab-btn {active}" '
            f'data-edge-dom-key="{threshold_dom_keys[i]}" '
            f'onclick="showEdgeTab({_html_js_arg(t)}, this)">{_threshold_tab_label(analyzer, t, dataset_names, nickname_map)}</button>')
    html_parts.append('</div>')
    
    for i, threshold in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(
            f'<div id="edge_tab_{threshold_dom_keys[i]}" '
            f'class="tab-content {active}">')
        html_parts.append(_generate_presence_table(analyzer.get_aligned_data(threshold), dataset_names, nickname_map, threshold))
        html_parts.append('</div>')
    
    html_parts.append('</div>')  # Close edge_by_threshold
    
    # By Dataset View
    html_parts.append(f'''
                <!-- By Dataset View -->
                <div id="edge_by_dataset" class="tabs" style="display: none;">
                    <div class="tab-buttons">
''')
    
    for i, d in enumerate(dataset_names):
        active = 'active' if i == 0 else ''
        nick = nickname_map[d]
        html_parts.append(
            f'<button class="tab-btn {active}" '
            f'data-edge-dataset-dom-key="{dataset_dom_keys[i]}" '
            f'onclick="showEdgeDatasetTab({_html_js_arg(nick)}, this)">'
            f'{html.escape(str(nick))}</button>')
    html_parts.append('</div>')
    
    # Generate dataset-centric tables (showing all thresholds for one dataset)
    for i, d in enumerate(dataset_names):
        nick = nickname_map[d]
        active = 'active' if i == 0 else ''
        html_parts.append(
            f'<div id="edge_dataset_tab_{dataset_dom_keys[i]}" '
            f'class="tab-content {active}">')
        html_parts.append(_generate_edge_dataset_table(analyzer, d, thresholds, nickname_map))
        html_parts.append('</div>')
    
    html_parts.append(f"""
                </div>
                
                <script>
                    function switchEdgeMode(mode) {{
                        document.getElementById('edge_mode_threshold').classList.toggle('active', mode === 'threshold');
                        document.getElementById('edge_mode_dataset').classList.toggle('active', mode === 'dataset');
                        document.getElementById('edge_by_threshold').style.display = mode === 'threshold' ? 'block' : 'none';
                        document.getElementById('edge_by_dataset').style.display = mode === 'dataset' ? 'block' : 'none';
                    }}
                    
                    function showEdgeTab(threshold, button) {{
                        document.querySelectorAll('#edge_by_threshold .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#edge_by_threshold .tab-btn').forEach(el => el.classList.remove('active'));
                        const domKey = button && button.dataset.edgeDomKey;
                        const panel = domKey ? document.getElementById('edge_tab_' + domKey) : null;
                        if (!panel) return;
                        panel.classList.add('active');
                        if (button) button.classList.add('active');
                    }}
                    
                    function showEdgeDatasetTab(dataset, button) {{
                        document.querySelectorAll('#edge_by_dataset .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#edge_by_dataset .tab-btn').forEach(el => el.classList.remove('active'));
                        const domKey = button && button.dataset.edgeDatasetDomKey;
                        const panel = domKey ? document.getElementById('edge_dataset_tab_' + domKey) : null;
                        if (!panel) return;
                        panel.classList.add('active');
                        if (button) button.classList.add('active');
                    }}
                </script>
            </div>
        </div>
""")
    
    return ''.join(html_parts)


def _generate_edge_dataset_table(analyzer, dataset: str, thresholds: List[int], 
                                  nickname_map: Dict[str, str]) -> str:
    """Generate edge presence table for a single dataset across all thresholds."""
    nick = nickname_map[dataset]
    max_edges_to_show = 50
    
    # First, identify top edges from the first threshold to limit iteration
    top_edges_set = set()
    first_threshold = thresholds[0] if thresholds else None
    if first_threshold is not None:
        aligned_first = analyzer.get_aligned_data(first_threshold)
        if not aligned_first.empty and dataset in aligned_first.columns:
            top_indices = aligned_first.nlargest(max_edges_to_show * 2, dataset).index
            top_edges_set = set(top_indices)
    
    # Collect data only for top edges across thresholds
    edge_data = {}  # edge_key -> {threshold: weight}
    
    for threshold in thresholds:
        aligned = analyzer.get_aligned_data(threshold)
        if aligned.empty or dataset not in aligned.columns:
            continue
        
        # Only iterate over top edges
        if top_edges_set:
            edges_to_process = [e for e in top_edges_set if e in aligned.index]
        else:
            edges_to_process = list(aligned.index[:max_edges_to_show * 2])
        
        for edge_key in edges_to_process:
            weight = aligned.loc[edge_key, dataset]
            # Handle Series (duplicate indices)
            if hasattr(weight, 'iloc'):
                weight = weight.iloc[0]
            if edge_key not in edge_data:
                edge_data[edge_key] = {}
            edge_data[edge_key][threshold] = weight
    
    if not edge_data:
        return '<p>No data available.</p>'
    
    # Sort edges by lowest threshold weight
    sorted_edges = sorted(
        edge_data.items(), 
        key=lambda x: (-x[1].get(thresholds[0], 0), str(x[0]))
    )[:50]
    
    html = [f'<div style="margin-bottom: 8px; color: var(--primary-color); font-weight: 600;">Dataset: {nick} (all thresholds)</div>']
    html.append('<div class="sticky-table-container" style="overflow-x: auto;"><table><thead><tr><th>Edge</th>')
    for t in thresholds:
        html.append(f'<th>t={t}</th>')
    html.append('</tr></thead><tbody>')
    
    for edge_key, weights in sorted_edges:
        html.append(f'<tr><td><strong>{edge_key}</strong></td>')
        for t in thresholds:
            w = weights.get(t, 0)
            if w > 0:
                html.append(f'<td><span class="presence-check">✔️</span> {int(w)}</td>')
            else:
                html.append('<td><span class="presence-cross">❌</span></td>')
        html.append('</tr>')
    
    html.append('</tbody></table></div>')
    return ''.join(html)


def _canonical_appearance_ranks(analyzer, dataset_names, thresholds,
                                resolve_canonical) -> Dict[str, int]:
    """First-appearance rank per canonical type from the ordered presence
    matrices (plan §7B).

    Walks the path-presence rows (falling back to edge-presence) in the
    report's own conservation/weight order and assigns each canonical type
    the index of the first row containing it. ``resolve_canonical(name)``
    maps a display name to its canonical row key. Types never seen get no
    rank (they sort last, alphabetically).
    """
    ranks: Dict[str, int] = {}
    order = 0

    def _record(name):
        nonlocal order
        if name is None:
            return
        key = resolve_canonical(name)
        if key is None:
            return
        key = str(key)
        if key not in ranks:
            ranks[key] = order
            order += 1

    def _consume_matrix(matrix):
        if matrix is None or getattr(matrix, 'empty', True):
            return
        available = [d for d in dataset_names if d in matrix.columns]
        if not available:
            return
        try:
            copy = matrix.copy()
            copy['_conservation'] = (copy[available] > 0).sum(axis=1)
            copy['_total'] = copy[available].sum(axis=1)
            copy = copy.sort_values(['_conservation', '_total'],
                                    ascending=[False, False])
        except Exception:
            copy = matrix
        for key in copy.index:
            for node in _normalize_path_key(key).split(' -> '):
                _record(node.strip())

    # Path presence first (the user's named criterion), then edges.
    for t in thresholds:
        try:
            _consume_matrix(analyzer._get_path_data_for_threshold(t))
        except Exception:
            pass
    for t in thresholds:
        try:
            _consume_matrix(analyzer.get_aligned_data(t))
        except Exception:
            pass
    return ranks


def _render_merge_policy_topology(policy, dataset_names: List[str]) -> str:
    """B5 subsection: the query-anchored resolution tree per group —
    per-dataset members, split branches with their row-based pools and
    vote provenance, fan-in keys, and the policy warnings.  Evidence
    only; nothing here changes merge membership."""
    top = policy.topology_dict()
    # Collapsed by default: the branch trees and pool disclosures are
    # evidence detail, not something every reader must scroll past.
    parts = ['<details class="card" style="margin:10px 0;">',
             '<summary style="cursor:pointer; font-weight:600; '
             'color:var(--primary-color);">Resolution topology '
             '(query-anchored merge)</summary>',
             '<div style="margin-top:8px;">',
             f'<p>{html_module.escape(str(top.get("summary", "")))}</p>']
    warnings = top.get('warnings') or []
    if warnings:
        parts.append('<ul style="font-size:0.88em;">' + ''.join(
            f'<li>{html_module.escape(str(w))}</li>' for w in warnings)
            + '</ul>')
    for group in top.get('groups') or []:
        members = group.get('members') or {}
        member_txt = ' · '.join(
            f'{html_module.escape(str(ds))}: '
            f'{html_module.escape(", ".join(names))}'
            for ds, names in sorted(members.items()))
        parts.append(
            '<div style="border:1px solid var(--border-color);'
            'border-radius:6px;padding:6px 10px;margin:6px 0;">'
            f'<strong>{html_module.escape(str(group.get("label", "")))}</strong> '
            '<span style="color:var(--secondary-color);">'
            f'[{html_module.escape(str(group.get("kind", "")))} '
            f'{html_module.escape(str(group.get("group_id", "")))}]</span>'
            f'<br>{member_txt}')
        for branch in group.get('branches') or []:
            b_members = branch.get('members') or {}
            b_txt = ' · '.join(
                f'{html_module.escape(str(ds))}: '
                f'{html_module.escape(", ".join(names) if isinstance(names, list) else str(names))}'
                for ds, names in sorted(b_members.items()))
            parts.append(
                '<div style="margin:4px 0 0 14px;">↳ '
                f'<strong>{html_module.escape(str(branch.get("label", "")))}</strong>'
                f' — {b_txt}')
            pool_info = branch.get('pools') or {}
            if pool_info.get('support'):
                parts.append(
                    '<br><span style="font-size:0.85em;">'
                    f'{html_module.escape(str(pool_info["support"]))}</span>')
            for pool in pool_info.get('pools') or []:
                parts.append(
                    '<br><span style="font-size:0.85em;'
                    'color:var(--secondary-color);">'
                    f"src {html_module.escape(str(pool.get('source_basis')))}"
                    f" {html_module.escape(str(pool.get('source_pool_size')))}"
                    f" / tgt {html_module.escape(str(pool.get('target_basis')))}"
                    f" {html_module.escape(str(pool.get('target_pool_size')))}"
                    f" — {html_module.escape(str(pool.get('status')))}"
                    '</span>')
            parts.append('</div>')
        parts.append('</div>')
    fan = top.get('fan_in') or {}
    if fan:
        items = ', '.join(
            f"{html_module.escape(str(key))} ← "
            f"{' / '.join(html_module.escape(str(x)) for x in values)}"
            for key, values in fan.items())
        parts.append(
            '<p style="font-size:0.88em;">Fan-in keys (merge with '
            f'neither claimant): {items}</p>')
    parts.append('</div></details>')
    return ''.join(parts)


def _filtered_result_types(raw_results, only_thresholds=None):
    """``{dataset: type set}`` scan of ``raw_results`` (dataset → threshold
    → result frame), optionally restricted to ``only_thresholds`` (ints).

    Mirrors ``ComparisonAnalyzer._collect_result_types_by_dataset`` (string
    values from the known type columns) with an added threshold filter —
    the type-mapping grid passes the compared schedule so its universe is
    identical between a fresh export and a later re-export (a fresh run's
    raw_results additionally holds the auto-bootstrap probe threshold,
    which re-exports never load)."""
    types_by_dataset: Dict[str, set] = {}
    for dataset, thresh_results in raw_results.items():
        if not isinstance(thresh_results, dict):
            continue
        dataset_types = types_by_dataset.setdefault(dataset, set())
        for threshold, result in thresh_results.items():
            if only_thresholds is not None and threshold not in \
                    only_thresholds:
                continue
            if isinstance(result, pd.DataFrame) and not result.empty:
                for col in ['type_pre', 'type_post', 'from_type', 'to_type',
                            'std_label_pre', 'std_label_post']:
                    if col in result.columns:
                        dataset_types.update(
                            result[col].dropna().unique())
            elif isinstance(result, dict):
                for key in ['type_level', 'edge_level']:
                    df = result.get(key)
                    if df is not None and hasattr(df, 'empty') \
                            and not df.empty:
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


def _rival_evidence_of(detail):
    """Per-rival evidence rows of one same-name detail record (or []) —
    tolerates both the dict shape and older list shapes."""
    if isinstance(detail, dict):
        ev = detail.get('rival_evidence')
        return list(ev) if isinstance(ev, (list, tuple)) else []
    return []


def _votes_text(ev):
    """`N curated / M auto` for one rival evidence row (or an em dash)."""
    verified = ev.get('verified_votes') or {}
    auto = ev.get('auto_votes') or {}
    try:
        curated_n = sum(int(v or 0) for v in verified.values())
        auto_n = sum(int(v or 0) for v in auto.values())
    except Exception:
        return '—'
    if not (curated_n or auto_n):
        return '—'
    return f'{curated_n} curated / {auto_n} auto'


def _same_name_first_aggregate(mapper, dataset_names, filter_types=None):
    """Count same-name fan-outs among the run's datasets, by disposition,
    plus the multi-value `type` cell count.

    Backs the Type Mapping panel's suspects card
    (plan-samename-first-fanout-resolution §3;
    plan-type-column-multivalue-normalization Stage 1).  Read-only; returns
    ``{}`` when the mapper has neither."""
    if mapper is None:
        return {}
    try:
        counts = mapper.same_name_first_summary(
            filter_types=(set(filter_types) if filter_types else None),
            datasets=dataset_names)
    except Exception:
        counts = None
    if not isinstance(counts, dict):
        return {}
    out = {k: v for k, v in counts.items() if v}
    try:
        mv = mapper.multivalue_summary(datasets=dataset_names)
        if mv:
            out['multivalue_cells'] = sum(mv.values())
    except Exception:
        pass
    return out


def _generate_type_mapping_section(analyzer, dataset_names: List[str]) -> str:
    """🔗 Type Mapping (auto) — the run's type resolution in three tables.

    **Sources** — the priority-selected ``(dataset: type)`` observation per
    canonical row (``canonical_source_rank``: male-cns → FAFB → other
    neuprint → BANC); remaining observations listed beside. **Targets** —
    the resolved name in each dataset (nicknamed columns); names differing
    from the canonical are colored (split-branch members share one color),
    lists beyond three names collapse, and the source dataset's own name is
    repeated muted. **Intermediates** — mid-chain hops of bridge routes
    from the resolver's chain evidence. Rows are capped at 100, ordered by
    first appearance in the presence matrices with an alphabetical toggle,
    and the collapsed Resolution topology card rides on top. Rendered only
    when auto_type_mapping is enabled."""
    params = getattr(analyzer, 'parameters', None)
    mapper = getattr(params, '_auto_type_mapper', None)
    if params is None or not getattr(params, 'auto_type_mapping', False) \
            or mapper is None:
        return ''
    def _compared_threshold_set():
        """Per-dataset thresholds the compared queries actually read.

        A fresh run's ``raw_results`` also holds the auto-bootstrap probe
        (t=floor, density capture only — no compared matrix uses it); a
        later re-export never loads it.  Restricting the type universe to
        the schedule keeps the grid identical between the two."""
        try:
            combos = getattr(params, 'threshold_combinations', None)
            if combos:
                return {int(v) for q in combos
                        for v in (q.get('thresholds') or {}).values()}
            thresholds = getattr(params, 'thresholds', None)
            return {int(t) for t in (thresholds or [])} or None
        except Exception:
            return None

    def _compared_threshold_set():
        """Per-dataset thresholds the compared queries actually read.

        A fresh run's ``raw_results`` also holds the auto-bootstrap probe
        (t=floor, density capture only — no compared matrix uses it); a
        later re-export never loads it.  Restricting the type universe to
        the schedule keeps the grid identical between the two."""
        try:
            combos = getattr(params, 'threshold_combinations', None)
            if combos:
                return {int(v) for q in combos
                        for v in (q.get('thresholds') or {}).values()}
            thresholds = getattr(params, 'thresholds', None)
            return {int(t) for t in (thresholds or [])} or None
        except Exception:
            return None

    raw_results = getattr(analyzer, 'raw_results', None)
    if isinstance(raw_results, dict) and raw_results:
        result_types_by_dataset = _filtered_result_types(
            raw_results, _compared_threshold_set())
    else:
        try:
            result_types_by_dataset = \
                analyzer._collect_result_types_by_dataset()
        except Exception:
            try:
                result_types = list(analyzer._collect_result_types())
            except Exception:
                return ''
            result_types_by_dataset = {None: set(result_types)}

    raw_type_count = sum(
        len(types) for types in result_types_by_dataset.values())
    if not raw_type_count:
        return ''

    def _source_rank(dataset):
        """Global canonical-source priority (plan F1): male-cns → FAFB →
        other neuprint → BANC; unknown datasets rank with neuprint."""
        try:
            from comparison.comparison_parameters import canonical_source_rank
            return canonical_source_rank(dataset)
        except Exception:
            return 2

    def _canonical_for(type_name, source_dataset):
        """Canonical row key via the shared resolver's merge keys
        (licensed renames map; a conflicted type keys per dataset;
        display unaffected).  ``source_dataset=None`` auto-detects."""
        try:
            from comparison.type_resolver import canonical_merge_key
            return canonical_merge_key(
                mapper, type_name, source_dataset,
                cache=canonical_for_cache).key
        except Exception:
            return type_name

    def _resolve_for(type_name, source_dataset):
        """Per-dataset valid targets through the shared validity resolver
        (``comparison.type_resolver``), so the report's labels and statuses
        agree with the panel and the analysis that produced it.

        Unique renames/bridges yield one name; a valid split yields ALL its
        branches (the row lists every target rather than an arbitrary
        one); conflicts, evidence-only relations, unmapped types, and an
        unavailable mapper yield nothing for that dataset.

        A same-name identity result is dropped when resolving into a
        DIFFERENT dataset: the mapper's tier-6 name echo must not bleed one
        dataset's name into another dataset's cell (the ``aMe24``/FAFB
        leak). The name is still shown under its own canonical row.

        Returns ``(resolved, intermediates)`` where ``intermediates`` are
        the mid-chain hops (``dataset:value``) of the bridge evidence —
        the Intermediates table's data (plan F2)."""
        from .type_resolver import (
            STATUS_BRIDGED, STATUS_MAPPED, STATUS_VALID_SPLIT,
            resolve_valid_targets,
        )
        resolved: Dict[str, object] = {}
        intermediates: set = set()
        for dataset in dataset_names:
            try:
                res = resolve_valid_targets(
                    mapper, type_name, source_dataset, dataset)
            except Exception:
                continue
            if (res.status in (STATUS_MAPPED, STATUS_BRIDGED)
                    and res.equivalence_key):
                value = str(res.equivalence_key)
                # Drop cross-dataset same-NAME-only echoes (mapper tier 6):
                # that name belongs to another dataset's namespace and must
                # not bleed into this cell. A real rename that happens to
                # share the name (kind 'renamed') is kept, and so is a
                # curated identity (Phase 1b ``curated_identity``: a unique
                # tabular/label relation confirms this exact pairing).
                if (dataset != source_dataset
                        and str(getattr(res, 'kind', '')).strip()
                        == 'same name'
                        and not getattr(res, 'curated_identity', False)):
                    continue
                resolved[dataset] = value
                for chain in (getattr(res, 'evidence', ()) or ()):
                    steps = list(chain)
                    if len(steps) < 3:
                        continue  # direct relation: no intermediate hops
                    for step in steps[1:-1]:
                        hop_ds = str(step.get('dataset') or '')
                        hop_val = str(step.get('value') or '')
                        if hop_val:
                            intermediates.add(
                                f'{hop_ds}:{hop_val}' if hop_ds else hop_val)
            elif res.status == STATUS_VALID_SPLIT and res.target_types:
                resolved[dataset] = tuple(res.target_types)
        return resolved, intermediates

    # B5/B3: policy-governed raw types consume the merge policy's data
    # directly (the per-run resolution the analysis used); non-policy
    # types keep the canonical/resolve derivation.
    policy = None
    try:
        policy = analyzer._merge_policy_or_none()
    except Exception:
        policy = None
    policy_seen: set = set()
    policy_done: set = set()

    # Canonicalize first, then cap.  A raw type can occur in several source
    # namespaces, so each canonical row unions its per-dataset names.
    canonical_for_cache: dict = {}
    canonical_rows: Dict[str, Dict[str, set]] = {}
    # Per-row provenance for the report's Source column: (dataset, raw type)
    # observations the row was anchored on.  Policy rows use the group
    # anchor; derived rows record the observed (dataset, raw type).
    row_sources: Dict[str, set] = {}
    # Per-row branch membership (raw name -> branch index) so split-branch
    # names can share one color across dataset columns.
    row_branch_colors: Dict[str, Dict[str, int]] = {}
    # Per-row bridge-chain intermediate hops (plan F2, Intermediates table).
    row_intermediates: Dict[str, set] = {}
    for source_dataset, source_types in result_types_by_dataset.items():
        source_hint = (source_dataset
                       if source_dataset in dataset_names else None)
        for raw_type in sorted(source_types):
            # Policy-governed types consume the MERGE POLICY's data
            # directly (real-data finding 2026-09-15: re-deriving via
            # resolve_valid_targets leaked FAFB aMe24 into the
            # 5thsLNv_LNd6 row through the curated aMe24=aMe24 identity —
            # the merge policy's key_map is the per-run resolution the
            # analysis actually used, so the row must match it).
            if policy is not None and source_hint in dataset_names:
                policy_label = policy.key_for(source_hint, raw_type)
                if policy_label:
                    policy_seen.add(policy_label)
                    if policy_label not in policy_done:
                        policy_done.add(policy_label)
                        group_row = canonical_rows.setdefault(
                            policy_label,
                            {dataset: set() for dataset in dataset_names})
                        for ds, names in policy.names_by_dataset(
                                policy_label).items():
                            if ds in group_row:
                                group_row[ds].update(names)
                                # Every member observation feeds the
                                # Source column's GLOBAL priority pick
                                # (plan F1/V1: the displayed source is the
                                # highest-priority dataset where the row
                                # resolves — male-cns first — never the
                                # per-name merge anchor alone).
                                row_sources.setdefault(
                                    policy_label, set()).update(
                                    (str(ds), str(nm)) for nm in names)
                        group = policy.group_by_label(policy_label)
                        if group is not None:
                            anchor_ds, anchor_type = getattr(
                                group, 'anchor', (None, None)) or (None, None)
                            if anchor_ds:
                                row_sources.setdefault(policy_label, set()).add(
                                    (str(anchor_ds), str(anchor_type)))
                            if group.branches:
                                cmap: Dict[str, int] = {}
                                for branch_idx, branch in enumerate(
                                        group.branches):
                                    for _ds, names in sorted(
                                            (branch.members or {}).items()):
                                        for nm in names:
                                            cmap.setdefault(str(nm),
                                                            branch_idx)
                                row_branch_colors[policy_label] = cmap
                    continue
            canonical = str(_canonical_for(raw_type, source_hint)
                            or raw_type)
            row = canonical_rows.setdefault(
                canonical, {dataset: set() for dataset in dataset_names})
            if source_hint is not None:
                row_sources.setdefault(canonical, set()).add(
                    (str(source_hint), str(raw_type)))
            mappings, intermediates = _resolve_for(raw_type, source_hint)
            if intermediates:
                row_intermediates.setdefault(canonical, set()).update(
                    intermediates)
            for dataset in dataset_names:
                value = mappings.get(dataset)
                if not value:
                    continue
                if isinstance(value, (tuple, list, set)):
                    # A valid split: list every licensed branch instead of
                    # an arbitrary single target.
                    row[dataset].update(str(v) for v in value)
                else:
                    row[dataset].add(str(value))
            # A mapper may not return a same-namespace value for a source
            # hint.  Preserve the observed raw name in that dataset rather
            # than displaying a misleading blank cell.
            if source_dataset in dataset_names and not row[source_dataset]:
                row[source_dataset].add(str(raw_type))

    # Appearance order (plan §7B): rank canonical rows by their first
    # appearance in the ordered presence matrices; alphabetical is the
    # tiebreak/fallback. A client-side toggle lets the user switch.
    try:
        thresholds_for_order = list(getattr(
            analyzer, '_analysis_thresholds', lambda: [])())
    except Exception:
        thresholds_for_order = []
    appearance = _canonical_appearance_ranks(
        analyzer, dataset_names, thresholds_for_order,
        lambda name: _canonical_for(name, None))
    if appearance:
        ordered = sorted(
            canonical_rows.items(),
            key=lambda item: (appearance.get(item[0], 10 ** 9), item[0]))
    else:
        ordered = sorted(canonical_rows.items(), key=lambda item: item[0])
    cap = 100
    shown = ordered[:cap]

    # Persist the order with the mapping outputs (plan §7B.4) so the CSV
    # and the report agree.
    try:
        analyzer._type_appearance_ranks = dict(appearance)
    except Exception:
        pass

    conflicts_total = len(getattr(mapper, '_conflicts', []) or [])
    topology_html = (
        _render_merge_policy_topology(policy, dataset_names)
        if policy is not None else '')
    auto_only_rows = []
    try:
        from .merge_policy import auto_only_edges
        _section_types: set = set()
        for _src_ds, _src_types in result_types_by_dataset.items():
            _section_types.update(str(t) for t in _src_types)
        auto_only_rows = list(auto_only_edges(mapper, _section_types,
                                              dataset_names))
    except Exception:
        auto_only_rows = []
    # Per-(canonical row, dataset) auto-only counts: the badge attaches to
    # the AUTO-LABEL SOURCE cell (plan F3) — the dataset whose transferred
    # annotations the row's mappings rest on (in a BANC-anchored run that
    # is the BANC column) — instead of to the canonical label. The badge
    # lives ONLY there (plan-cross-dataset-report-mapping-grid-and-role-
    # tables Item 1): the canonical cell carries none, so the fact renders
    # exactly once per donor dataset.
    auto_only_badge: Dict[Tuple[str, str], int] = {}

    def _row_key_for(name, ds):
        if policy is not None and ds in dataset_names:
            label = policy.key_for(ds, name)
            if label:
                return str(label)
        try:
            return str(_canonical_for(name, ds if ds in dataset_names
                                      else None) or name)
        except Exception:
            return str(name)

    for _row in auto_only_rows:
        src_key = _row_key_for(str(_row['source_type']),
                               str(_row['source_dataset']))
        key = (src_key, str(_row['source_dataset']))
        auto_only_badge[key] = auto_only_badge.get(key, 0) + 1

    html = ['<div id="type-mapping" class="section">',
            '<div class="section-header">🔗 Type Mapping (auto)</div>',
            '<div class="section-content">']
    if topology_html:
        html.append(topology_html)
    # ---- Same-name-first suspects aggregate card (plan-samename-first-
    # fanout-resolution §3): how many fan-outs carried a same-name
    # candidate, by disposition, so the user can gauge the adjudication
    # workload before opening the suspects CSV.
    try:
        # Use the SAME type universe as the suspects CSV and the run note
        # (the analyzer's collected result types) so every surface agrees.
        _run_types = set(analyzer._collect_result_types())
        _agg = _same_name_first_aggregate(
            mapper, dataset_names, filter_types=_run_types)
    except Exception:
        try:
            _agg = _same_name_first_aggregate(mapper, dataset_names)
        except Exception:
            _agg = None
    if _agg:
        # Boundary-clean labels (plan-ui-type-mapper-alignment §2.5): the
        # card states the mapper's observations; "duplicate"/"confirmed" are
        # verification-stage words and must not appear here.
        _label = {
            'selected': 'selected (same-name candidate chosen)',
            'rivals': 'rival relations exported (suspects CSV)',
            'rivals_exported': 'rival relations exported (suspects CSV)',
            'selected_rivals': 'rivals of the selected fan-outs',
            'gated_held': 'kept unmapped (rivals not all separably paired)',
            'excluded_evidence_only': 'evidence-only (N-to-1) — excluded',
            'multivalue_cells': 'multi-value type cells (🧩, kept atomic)',
        }
        _rows_txt = ''.join(
            f'<tr><td>{html_module.escape(_label.get(k, k))}</td>'
            f'<td>{v}</td></tr>'
            for k, v in sorted(_agg.items()))
        html.append(
            '<div class="card" style="margin:10px 0;">'
            '<strong>Same-name-first suspects</strong> '
            '<span style="color:var(--secondary-color);font-size:0.9em;">'
            '— fan-outs whose candidate set contained the source\'s own '
            'name; the same-name candidate is selected, the rivals are '
            'suspects to verify (see '
            '<code>auto_type_mapping_suspects.csv</code>).</span>'
            '<table style="margin-top:6px;"><thead><tr>'
            '<th>disposition</th><th>count</th></tr></thead><tbody>'
            f'{_rows_txt}</tbody></table></div>')
    try:
        _nicks = analyzer.parameters.get_dataset_nicknames()
        section_nicks = {ds: _nicks[i] for i, ds in enumerate(dataset_names)}
    except Exception:
        section_nicks = {}

    # ---- Query-role tables (plan round-4 F-2): the user reads this
    # section by QUERY ROLE — (1) the queried source tokens, (2) the
    # queried target tokens, (3) the intermediate types that appear inside
    # paths — not by name provenance.
    try:
        records = analyzer.resolve_query_inputs() or []
    except Exception:
        records = []

    # Merged query-role table (plan round-4 F-2; unified per plan-cross-
    # dataset-report-mapping-grid-and-role-tables Item 5): ONE aligned
    # table with three section groups — queried sources, queried targets,
    # path intermediates — sharing one column geometry, one coloring
    # schema, and a #paths column on every group.
    status_colors = {
        'same_name_identity': '#15803d', 'mapped': '#15803d',
        'bridged': '#15803d', 'taxonomy': '#1d4ed8',
        'taxonomy_mapped': '#1d4ed8', 'valid_split': '#7c3aed',
        'same_name_fallback': '#b45309', 'evidence_only': '#b45309',
        'conflict': '#b91c1c', 'unmapped': '#6b7280',
    }
    _traversed_color = '#15803d'
    _muted_color = '#9ca3af'
    _not_traversed_tip = ('resolves here, not traversed in this '
                          'run&#39;s paths')

    def _status_legend_html() -> str:
        """Explicit visual legend for the shared coloring schema.

        Colors are read back through ``status_colors`` (never retyped) so
        the legend cannot drift from the cell rendering; the muted chip
        and the em-dash sample cover the two non-color marks.
        """
        def chip(color: str, label: str) -> str:
            return (
                '<span style="display:inline-flex; align-items:center; '
                'gap:4px; margin:2px 14px 2px 0;" '
                f'title="resolution status color: {color}">'
                '<span style="width:12px; height:12px; border-radius:3px; '
                f'display:inline-block; background:{color};"></span>'
                f'<span style="font-size:0.78em; color:#475569;">{label}'
                '</span></span>')

        entries = [
            (status_colors['same_name_identity'], 'identity / mapped / bridged'),
            (status_colors['taxonomy'], 'taxonomy'),
            (status_colors['valid_split'], 'valid split'),
            (status_colors['same_name_fallback'], 'fallback / evidence only'),
            (status_colors['conflict'], 'conflict'),
            (status_colors['unmapped'], 'unmapped'),
        ]
        muted = (
            '<span style="display:inline-flex; align-items:center; '
            'gap:4px; margin:2px 14px 2px 0;" '
            f'title="{_not_traversed_tip}">'
            '<span style="width:12px; height:12px; border-radius:3px; '
            f'display:inline-block; background:{_muted_color};"></span>'
            '<span style="font-size:0.78em; color:#9ca3af;">muted</span>'
            '<span style="font-size:0.78em; color:#475569;">= resolves '
            'here but was not traversed in this run&#39;s paths</span>'
            '</span>')
        dash = (
            '<span style="display:inline-flex; align-items:center; '
            'gap:4px; margin:2px 0;" '
            'title="no resolution in this dataset">'
            '<span style="font-size:0.9em; color:#6b7280;">&#8212;</span>'
            '<span style="font-size:0.78em; color:#475569;">= no '
            'resolution in this dataset</span></span>')
        return (
            '<div class="status-legend" style="display:flex; '
            'flex-wrap:wrap; align-items:center; margin:4px 0 2px 0;">'
            + ''.join(chip(color, label) for color, label in entries)
            + muted + dash + '</div>')

    def _tokens_for(role):
        toks, seen = [], set()
        for rec in records:
            if str(rec.get('role') or '') != role:
                continue
            token = str(rec.get('token') or '')
            if token and token not in seen:
                seen.add(token)
                toks.append(token)
        return toks

    def _record_row(token, role, present_datasets):
        """One queried token's per-dataset resolved names + status.

        ``present_datasets`` is the set of datasets whose paths start
        (source) / end (target) at the type; a dataset that resolves the
        token but never participates renders muted — the same semantics
        the intermediates table always had (Item 5). ``None`` = no path
        data at all, so nothing is muted."""
        cells = []
        for d in dataset_names:
            recs = [r for r in records
                    if str(r.get('token')) == token
                    and str(r.get('dataset')) == d
                    and str(r.get('role')) == role]
            if not recs:
                cells.append('<td>—</td>')
                continue
            parts_cell = []
            for rec in recs:
                status = str(rec.get('status') or '')
                color = status_colors.get(status, '#6b7280')
                _targets = rec.get('target_types')
                targets = ('; '.join(str(t) for t in _targets)
                           if isinstance(_targets, (list, tuple, set))
                           else str(_targets or '—'))
                if present_datasets is not None and \
                        d not in present_datasets:
                    tip = (html_module.escape(status) + '; '
                           + _not_traversed_tip)
                    parts_cell.append(
                        f'<div><span style="color:{_muted_color};" '
                        f'title="{tip}">'
                        f'{html_module.escape(targets)}</span></div>')
                else:
                    parts_cell.append(
                        f'<div><span style="color:{color};" '
                        f'title="{html_module.escape(status)}">'
                        f'{html_module.escape(targets)}</span></div>')
            cells.append('<td>' + ''.join(parts_cell) + '</td>')
        return ('<tr><td><strong>' + html_module.escape(token)
                + '</strong></td>' + ''.join(cells))

    def _token_row(token, role, hop_stats):
        present = hop_stats.get(token, {}).get('datasets')
        n_paths = hop_stats.get(token, {}).get('paths', 0)
        return (_record_row(token, role, present)
                + f'<td>{n_paths}</td></tr>')

    src_toks = _tokens_for('source')
    tgt_toks = _tokens_for('target')

    # ---- Path participation: distinct types INSIDE paths (intermediates)
    # plus first/last hops — the queried sources' and targets' #paths and
    # their not-traversed muting come from the same walk. ----
    inter_stats: Dict[str, Dict] = {}
    start_stats: Dict[str, Dict] = {}
    end_stats: Dict[str, Dict] = {}

    def _hop_entry(store, name):
        return store.setdefault(str(name), {'paths': 0, 'datasets': set()})

    try:
        if getattr(params, 'threshold_mode', '') == 'combinations':
            path_iters = [(str(q.get('id')),
                           analyzer._get_path_data_for_query(q))
                          for q in (params.threshold_combinations or [])]
        else:
            path_iters = [(t, analyzer._get_path_data_for_threshold(t))
                          for t in (params.thresholds or [])]
        for point_key, pdf in path_iters:
            if pdf is None or getattr(pdf, 'empty', True):
                continue
            ds_cols = [d for d in dataset_names if d in pdf.columns]
            for key in pdf.index:
                hops = _normalize_path_key(key).split(' -> ')
                if len(hops) >= 2:
                    for hop, store in ((hops[0], start_stats),
                                       (hops[-1], end_stats)):
                        entry = _hop_entry(store, hop)
                        entry['paths'] += 1
                        for d in ds_cols:
                            try:
                                if pdf.loc[key, d] > 0:
                                    entry['datasets'].add(d)
                            except (TypeError, ValueError, KeyError):
                                continue
                for mid in hops[1:-1]:
                    entry = _hop_entry(inter_stats, mid)
                    entry['paths'] += 1
                    for d in ds_cols:
                        try:
                            if pdf.loc[key, d] > 0:
                                entry['datasets'].add(d)
                        except (TypeError, ValueError, KeyError):
                            continue
    except Exception:
        inter_stats, start_stats, end_stats = {}, {}, {}

    def _intermediate_cells(mid, present_datasets):
        """Per-dataset resolved names for one path intermediate (parity
        with the queried-role tables).  Resolution lanes: the merge
        policy's group members first (the per-run resolution the analysis
        actually used), then the shared validity resolver; a dataset that
        resolved nowhere but carried the type in its paths keeps the
        observed canonical label.  Resolved-but-never-traversed datasets
        render muted."""
        per_ds: Dict[str, List[str]] = {}
        if policy is not None:
            try:
                label = policy.label_for_name(mid)
                if label:
                    per_ds = {
                        ds: [str(n) for n in names]
                        for ds, names in (
                            policy.names_by_dataset(label).items())
                        if ds in dataset_names}
            except Exception:
                per_ds = {}
        if not per_ds:
            try:
                mappings, _ = _resolve_for(mid, None)
            except Exception:
                mappings = {}
            for ds, value in mappings.items():
                if not value:
                    continue
                per_ds[str(ds)] = (
                    [str(v) for v in value]
                    if isinstance(value, (tuple, list, set))
                    else [str(value)])
        cells = []
        for d in dataset_names:
            names = per_ds.get(d) or (
                [mid] if d in present_datasets else [])
            if not names:
                cells.append('<td>—</td>')
                continue
            shown = ' '.join(html_module.escape(n) for n in names)
            if d in present_datasets:
                cells.append(
                    f'<td><span style="color:{_traversed_color};" '
                    f'title="traversed in this run&#39;s paths">'
                    f'{shown}</span></td>')
            else:
                cells.append(
                    f'<td><span style="color:{_muted_color};" title="resolves '
                    f'here, not traversed in this run&#39;s paths">'
                    f'{shown}</span></td>')
        return cells

    sections = []
    if src_toks:
        sections.append(('Queried sources', [
            _token_row(tok, 'source', start_stats) for tok in src_toks]))
    if tgt_toks:
        sections.append(('Queried targets', [
            _token_row(tok, 'target', end_stats) for tok in tgt_toks]))
    if inter_stats:
        ordered_inter = sorted(inter_stats.items(),
                               key=lambda kv: (-kv[1]['paths'], kv[0]))
        irows = []
        for mid, info in ordered_inter[:50]:
            cells = ''.join(_intermediate_cells(mid, info['datasets']))
            irows.append(
                f'<tr><td><strong>{html_module.escape(mid)}</strong></td>'
                f'{cells}'
                f'<td>{info["paths"]}</td></tr>')
        sections.append(('Path intermediates', irows))

    if sections:
        head_cells = ''.join(
            f'<th>{html_module.escape(str(section_nicks.get(d, d)))}</th>'
            for d in dataset_names)
        body = []
        for title, rows in sections:
            body.append(
                f'<tr><td colspan="{len(dataset_names) + 2}" '
                f'style="background:#f1f5f9; font-weight:600; '
                f'color:#334155;">{html_module.escape(title)}</td></tr>')
            body.extend(rows)
        html.append(
            '<h4>Queried types &amp; path participation</h4>'
            + _status_legend_html()
            + '<p style="color:#666; font-size:0.85em;">Queried source and '
            'target types plus the distinct intermediates inside this '
            'run&#39;s multi-hop paths — one shared column layout and one '
            'coloring schema. Color = resolution status (legend above); '
            'muted = resolves here but was not traversed in this '
            'run&#39;s paths; &#8212; = no resolution. #paths = paths '
            'starting at a source, ending at a target, or traversing an '
            'intermediate, summed over all query points.</p>'
            '<div class="sticky-table-container" style="overflow-x: auto;">'
            '<table><thead><tr><th>Type</th>' + head_cells
            + '<th>#paths</th></tr></thead><tbody>'
            + ''.join(body) + '</tbody></table></div>')
    else:
        html.append(
            '<h4>Queried types &amp; path participation</h4>'
            '<p style="color:#666; font-size:0.9em;">No queried types or '
            'path intermediates to show (single-hop paths only, or no '
            'paths).</p>')

    # ---- Appendix: the full canonical mapping grid (collapsed) ----
    html.append('<details><summary style="cursor:pointer; font-weight:600; '
                'color:var(--primary-color);">All types — full mapping grid '
                '(appendix)</summary><div style="margin-top:8px;">')

    # Cell rendering: names ordered by first appearance, names that differ
    # from the canonical colored (split-branch members share one color so
    # branch membership is readable across dataset columns), and long lists
    # collapsed behind an inline expander.
    _name_palette = ['#1d4ed8', '#b45309', '#7c3aed', '#0f766e',
                     '#be185d', '#4d7c0f']

    def _ordered_names(t, values):
        def rank(nm):
            try:
                key = str(_canonical_for(nm, None) or nm)
            except Exception:
                key = str(nm)
            return appearance.get(key, 10 ** 9)
        return sorted(values, key=lambda nm: (rank(nm), str(nm)))

    def _cell_html(t, values):
        values = _ordered_names(t, values)
        cmap = row_branch_colors.get(t) or {}
        unbranched = [nm for nm in values if nm != t and nm not in cmap]

        def color_of(nm):
            if nm == t:
                return None
            branch_idx = cmap.get(nm)
            if branch_idx is None:
                branch_idx = len(cmap) + unbranched.index(nm)
            return _name_palette[branch_idx % len(_name_palette)]

        def span(nm):
            color = color_of(nm)
            text = html_module.escape(str(nm))
            if color is None:
                return f'<span style="white-space:nowrap;">{text}</span>'
            return (f'<span style="color:{color}; font-weight:600; '
                    f'white-space:nowrap;" title="differs from the '
                    f'canonical name">{text}</span>')

        inline, rest = values[:3], values[3:]
        cell = ' '.join(span(nm) for nm in inline)
        if rest:
            more = ''.join(f'<div>{span(nm)}</div>' for nm in rest)
            cell += (f' <details style="display:inline-block;">'
                     f'<summary style="cursor:pointer; color:var(--secondary-color); '
                     f'white-space:nowrap;">+{len(rest)} more</summary>'
                     f'{more}</details>')
        return cell

    html.extend([
            '<p style="color: var(--secondary-color);">',
            'Canonical row → per-dataset target names for the types involved '
            'in this run. Rows are ordered by their first appearance in the '
            'path-presence matrix (most-connected first); use the toggle for '
            'alphabetical lookup. Names that differ from the canonical name '
            'are colored — names sharing a split branch share a color — and '
            'target lists beyond three names collapse. Names without a '
            'male-cns counterpart remain under their observed name as an '
            'explicit fallback row.</p>',
            '<div style="margin-bottom:8px;">'
            '<button class="tab-btn active" onclick="sortTypeMapping(\'appearance\')">Appearance</button>'
            '<button class="tab-btn" onclick="sortTypeMapping(\'alpha\')">Alphabetical</button>'
            '</div>',
            '<div class="sticky-table-container" style="overflow-x: auto;">'
            '<table id="type-mapping-table"><thead><tr><th>#</th>'
            '<th>Canonical type</th><th>Source (priority)</th>'])
    for d in dataset_names:
        nick = section_nicks.get(d, d)
        html.append(f'<th>Target in {html_module.escape(str(nick))}</th>')
    html.append('</tr></thead><tbody>')
    for rank, (t, mappings) in enumerate(shown, start=1):
        data_name = html_module.escape(str(t))
        # A conflicted type's canonical KEY is namespaced ('ds:type') to
        # keep each dataset's row separate — DISPLAY the plain type name;
        # the namespaced key stays on data-canonical for the sorter.
        display_t = str(t)
        for _ds in dataset_names:
            if display_t.startswith(_ds + ':'):
                display_t = display_t[len(_ds) + 1:]
                break
        display_name = html_module.escape(display_t)
        # Same-name-first SUSPECTS badge (plan-samename-first-fanout-
        # resolution): this row's source had a fan-out whose candidate set
        # contained its own name — the rivals are suspects to verify.
        suspects_badge = ''
        _suspect_rows: Dict[str, Dict] = {}
        for _obs_ds, _obs_type in sorted(row_sources.get(t) or ()):
            try:
                _details = mapper.same_name_suspects_for_source(
                    _obs_type, _obs_ds, dataset_names)
            except Exception:
                _details = []
            if _details:
                _suspect_rows[_obs_ds] = _details
                break
        if _suspect_rows:
            # Collapsed IN-CELL details (plan-ui-type-mapper-alignment
            # §4.5 / D4): a short always-visible badge plus a native
            # <details> holding the per-rival table — the same four facts
            # the panel's expander and the suspects CSV carry.  No JS.
            _all_ev = [ev for _details in _suspect_rows.values()
                       for _d in _details
                       for ev in _rival_evidence_of(_d)]
            if not _all_ev:
                # Evidence unavailable (stub mappers / older shapes): count
                # the recorded rival names so the badge is still truthful.
                _nv = sum(len(_d.get('rivals') or ())
                          for _details in _suspect_rows.values()
                          for _d in _details if isinstance(_d, dict))
                _all_ev = [{'rival': r}
                           for _details in _suspect_rows.values()
                           for _d in _details if isinstance(_d, dict)
                           for r in (_d.get('rivals') or ())][:max(_nv, 0)]
            _n_rivals = len(_all_ev)
            _disp = html_module.escape(', '.join(sorted({
                str(_d.get('disposition'))
                for _details in _suspect_rows.values()
                for _d in _details
                if isinstance(_d, dict) and _d.get('disposition')})))
            _ev_rows = ''.join(
                '<tr>'
                f'<td>{html_module.escape(str(ev.get("rival") or ""))}</td>'
                f'<td>{"yes" if ev.get("rival_has_own_clean_pair") else "no"}</td>'
                f'<td>{html_module.escape(_votes_text(ev))}</td>'
                f'<td>{html_module.escape(str(ev.get("reverse_target") or "—"))}</td>'
                f'<td>{html_module.escape(str(ev.get("rival_pair_status") or ""))}</td>'
                '</tr>'
                for ev in _all_ev)
            suspects_badge = (
                # Item 4: hover popover instead of an inline <details> —
                # the expander grew the row inside .sticky-table-container
                # and the browser re-anchored, scrolling the table. The
                # popover is absolutely positioned (zero layout shift) and
                # stays open while the pointer is over badge or panel, so
                # the rival table's text is selectable and copiable.
                f' <span class="suspects-wrap">'
                f'<span class="suspects-trigger" tabindex="0" '
                f'style="color:#b45309;font-size:0.8em;">'
                f'⚠️ suspects ({_n_rivals})</span>'
                '<span class="suspects-pop">'
                '<div style="color:#666;margin-bottom:4px;">same-name-first '
                f'selection over {_n_rivals} rival candidate(s) — rivals are '
                f'never merged; pair-level disposition: {_disp}. '
                '`own 1-to-1 pair` states that the crosswalk also pairs that '
                'rival\'s own name 1-to-1 in this direction (an observation, '
                'not a verdict). See auto_type_mapping_suspects.csv.</div>'
                '<table style="border-collapse:collapse;">'
                '<thead><tr><th style="text-align:left;padding:2px 8px 2px 0;">'
                'rival</th><th style="text-align:left;padding:2px 8px 2px 0;">'
                'own 1-to-1 pair</th><th style="text-align:left;'
                'padding:2px 8px 2px 0;">votes (curated/auto)</th>'
                '<th style="text-align:left;padding:2px 8px 2px 0;">reverse '
                'target</th><th style="text-align:left;">rival pair status'
                '</th></tr></thead>'
                f'<tbody>{_ev_rows}</tbody></table></span></span>')
        # Multi-value `type` cell marker (plan-type-column-multivalue-
        # normalization): the source cell lists several candidate names.
        mv_badge = ''
        for _obs_ds, _obs_type in sorted(row_sources.get(t) or ()):
            try:
                _parts = mapper.multivalue_parts(_obs_type, _obs_ds)
            except Exception:
                _parts = None
            if _parts:
                mv_badge = (
                    ' <span style="color:#0e7490;font-size:0.8em;" '
                    f'title="multi-value type cell: this release annotates '
                    f'several candidate names in one type cell">'
                    f'🧩 multi ({html_module.escape("|".join(_parts))})'
                    '</span>')
                break
        # Global priority pick (plan F1/V1): the row's source is the
        # highest-priority dataset where it resolves — male-cns → FAFB →
        # other neuprint → BANC — independent of which dataset the merge
        # policy anchored on.  A name present in one dataset only keeps
        # that dataset automatically; remaining observations collapse
        # into a tooltip.
        sources = sorted(
            row_sources.get(t) or (),
            key=lambda pair: (_source_rank(pair[0]), pair[0], pair[1]))
        if sources:
            top_ds, top_type = sources[0]
            rest = sources[1:]
            source_cell = (f'{html_module.escape(top_ds)}: '
                           f'{html_module.escape(top_type)}')
            if rest:
                tip = html_module.escape('; '.join(
                    f'{ds}: {rt}' for ds, rt in rest))
                # Item 3: the count alone forced a hover to be meaningful —
                # name the datasets inline too.
                nick_list = ', '.join(
                    section_nicks.get(ds, ds) for ds, _ in rest)
                source_cell += (
                    f' <span style="color:#94a3b8; white-space:nowrap;" '
                    f'title="also observed: {tip}">(+{len(rest)}: '
                    f'{html_module.escape(nick_list)})</span>')
        else:
            source_cell = '—'
        html.append(f'<tr data-canonical="{data_name}" '
                    f'data-rank="{appearance.get(t, "")}">'
                    f'<td>{rank}</td>'
                    f'<td><strong>{display_name}</strong>'
                    f'{suspects_badge}{mv_badge}</td>'
                    f'<td>{source_cell}</td>')
        for d in dataset_names:
            values = sorted(mappings.get(d) or ())
            badge_count = auto_only_badge.get((str(t), str(d)))
            badge = ''
            if badge_count:
                badge = (
                    f' <span style="color:#b45309;font-size:0.8em;" '
                    f'title="{badge_count} mapping(s) on this row rest on '
                    f'auto-transferred labels transferred FROM this '
                    f'dataset (no curated vote) — see '
                    f'user_warning_notes.">&#9888;&#65039; auto-only'
                    f'</span>')
            # Pass the DISPLAY name so canonical-vs-member comparisons
            # (coloring, marker) work on the plain name for conflicted
            # rows whose key is namespaced.
            val = (_cell_html(display_t, values) if values else "—")
            html.append(f'<td>{val}{badge}</td>')
        html.append('</tr>')
    html.append('</tbody></table></div>')
    html.append("'''\n        <script>\n        function sortTypeMapping(mode) {\n"
                "            const table = document.getElementById('type-mapping-table');\n"
                "            if (!table) return;\n"
                "            const tbody = table.querySelector('tbody');\n"
                "            const rows = Array.from(tbody.querySelectorAll('tr'));\n"
                "            rows.sort((a, b) => mode === 'alpha'\n"
                "                ? a.dataset.canonical.localeCompare(b.dataset.canonical)\n"
                "                : (parseFloat(a.dataset.rank || 1e9) - parseFloat(b.dataset.rank || 1e9))\n"
                "                  || a.dataset.canonical.localeCompare(b.dataset.canonical));\n"
                "            rows.forEach(r => tbody.appendChild(r));\n"
                "            document.querySelectorAll('#type-mapping .tab-btn').forEach(b => b.classList.remove('active'));\n"
                "        }\n        </script>'''")
    html.append('</div></details>')
    html.append(f'<p style="color:#666; font-size:0.9em;">Showing {len(shown)} '
                f'of {len(canonical_rows)} canonical rows from '
                f'{raw_type_count} involved types'
                + (f'; {conflicts_total} mapping conflicts recorded — see '
                   f'auto_type_mapping_conflicts.csv' if conflicts_total else '; no mapping conflicts')
                + '.</p></div></div>')
    return ''.join(html)


def _normalize_path_key(key) -> str:
    """§3: canonical display-key normalization for hop-weight lookups.

    Unifies separator variants ('->' vs ' → ') and strips display suffixes
    ('Name(OtherName)') so the path-presence table rows and the hop-weight
    dict — built through slightly different key forms — always meet."""
    s = str(key).replace(' → ', ' -> ').replace('->', ' -> ')
    nodes = []
    for node in s.split(' -> '):
        node = node.strip()
        if '(' in node:
            node = node.split('(', 1)[0].strip()
        nodes.append(node)
    return ' -> '.join(nodes)


def _path_length_from_key(key) -> int:
    """Return the number of edges represented by a displayed path key."""
    # A path with N nodes has N - 1 edges.  Path rows are expected to contain
    # at least one edge; keep the lower bound defensive for malformed keys.
    return max(1, _normalize_path_key(key).count(' -> '))


def _generate_path_matrices_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                     nickname_map: Dict[str, str]) -> str:
    """Generate path presence matrices section with dual toggle (by threshold and by dataset)."""
    html_parts = []
    nicknames = [nickname_map[d] for d in dataset_names]
    threshold_dom_keys = _unique_dom_keys(thresholds)
    dataset_dom_keys = _unique_dom_keys(nicknames)
    
    html_parts.append(f"""
        <div id="path-matrices" class="section">
            <div class="section-header">🛤️ Path Presence Matrices</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Path presence across datasets. Weight = minimum edge weight along path.<br>
                    <em>Hop weights shown as <code>-w₁-w₂-</code> with <strong>min weight bolded</strong>.</em>
                </p>
                
                <!-- Toggle Mode Selector -->
                <div style="margin-bottom: 15px;">
                    <span style="font-weight: 600; margin-right: 10px;">View by:</span>
                    <button class="tab-btn active" id="path_mode_threshold" onclick="switchPathMode('threshold')">Threshold</button>
                    <button class="tab-btn" id="path_mode_dataset" onclick="switchPathMode('dataset')">Dataset</button>
                </div>
                
                <!-- By Threshold View -->
                <div id="path_by_threshold" class="tabs">
                    <div class="tab-buttons">
""")
    
    for i, t in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(
            f'<button class="tab-btn {active}" '
            f'data-path-dom-key="{threshold_dom_keys[i]}" '
            f'onclick="showPathTab({_html_js_arg(t)}, this)">{_threshold_tab_label(analyzer, t, dataset_names, nickname_map)}</button>')
    html_parts.append('</div>')
    
    for i, threshold in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(
            f'<div id="path_tab_{threshold_dom_keys[i]}" '
            f'class="tab-content {active}">')
        path_data = analyzer._get_path_data_for_threshold(threshold)
        if path_data is not None and not path_data.empty:
            html_parts.append(_generate_path_presence_table(analyzer, path_data, dataset_names, nickname_map, threshold))
        else:
            html_parts.append('<p>No path data available.</p>')
        html_parts.append('</div>')
    
    html_parts.append('</div>')  # Close path_by_threshold
    
    # By Dataset View
    html_parts.append(f'''
                <!-- By Dataset View -->
                <div id="path_by_dataset" class="tabs" style="display: none;">
                    <div class="tab-buttons">
''')
    
    for i, d in enumerate(dataset_names):
        active = 'active' if i == 0 else ''
        nick = nickname_map[d]
        html_parts.append(
            f'<button class="tab-btn {active}" '
            f'data-path-dataset-dom-key="{dataset_dom_keys[i]}" '
            f'onclick="showPathDatasetTab({_html_js_arg(nick)}, this)">'
            f'{html.escape(str(nick))}</button>')
    html_parts.append('</div>')
    
    # Generate dataset-centric tables (showing all thresholds for one dataset)
    for i, d in enumerate(dataset_names):
        nick = nickname_map[d]
        active = 'active' if i == 0 else ''
        html_parts.append(
            f'<div id="path_dataset_tab_{dataset_dom_keys[i]}" '
            f'class="tab-content {active}">')
        html_parts.append(_generate_path_dataset_table(analyzer, d, thresholds, nickname_map))
        html_parts.append('</div>')
    
    html_parts.append(f"""
                </div>
                
                <script>
                    function switchPathMode(mode) {{
                        document.getElementById('path_mode_threshold').classList.toggle('active', mode === 'threshold');
                        document.getElementById('path_mode_dataset').classList.toggle('active', mode === 'dataset');
                        document.getElementById('path_by_threshold').style.display = mode === 'threshold' ? 'block' : 'none';
                        document.getElementById('path_by_dataset').style.display = mode === 'dataset' ? 'block' : 'none';
                    }}
                    
                    function showPathTab(threshold, button) {{
                        document.querySelectorAll('#path_by_threshold .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#path_by_threshold .tab-btn').forEach(el => el.classList.remove('active'));
                        const domKey = button && button.dataset.pathDomKey;
                        const panel = domKey ? document.getElementById('path_tab_' + domKey) : null;
                        if (!panel) return;
                        panel.classList.add('active');
                        if (button) button.classList.add('active');
                    }}
                    
                    function showPathDatasetTab(dataset, button) {{
                        document.querySelectorAll('#path_by_dataset .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#path_by_dataset .tab-btn').forEach(el => el.classList.remove('active'));
                        const domKey = button && button.dataset.pathDatasetDomKey;
                        const panel = domKey ? document.getElementById('path_dataset_tab_' + domKey) : null;
                        if (!panel) return;
                        panel.classList.add('active');
                        if (button) button.classList.add('active');
                    }}
                </script>
            </div>
        </div>
""")
    
    return ''.join(html_parts)


def _generate_path_presence_table(analyzer, data: pd.DataFrame, dataset_names: List[str],
                                   nickname_map: Dict[str, str], threshold=None,
                                   caption_override: str = None) -> str:
    """Generate path presence table with hop weights shown as -w1-w2- with min bolded."""
    if data is None or data.empty:
        return '<p>No data available.</p>'

    available = [d for d in dataset_names if d in data.columns]
    if not available:
        return '<p>No datasets available.</p>'

    # Get path hop weights from analyzer.  A query report passes the complete
    # per-dataset threshold map so each cell can retain its hop annotation.
    path_hop_weights = {}
    if threshold is not None and hasattr(analyzer, '_get_path_hop_weights_for_threshold'):
        try:
            path_hop_weights = analyzer._get_path_hop_weights_for_threshold(threshold) or {}
        except Exception:
            path_hop_weights = {}
    # §3: normalized lookup so display/canonical key variants meet
    hop_by_norm = {_normalize_path_key(k): v for k, v in path_hop_weights.items()}

    # Add threshold caption if provided
    if caption_override is not None:
        threshold_caption = (f'<div style="margin-bottom: 8px; color: var(--primary-color); '
                             f'font-weight: 600;">{html.escape(caption_override)}</div>')
    else:
        threshold_caption = f'<div style="margin-bottom: 8px; color: var(--primary-color); font-weight: 600;">Threshold = {threshold}</div>' if threshold is not None else ''

    parts = [f'{threshold_caption}<div class="sticky-table-container" style="overflow-x: auto;"><table><thead><tr><th>Item</th><th>Len</th>']
    for d in available:
        parts.append(f'<th>{nickname_map[d]}</th>')
    parts.append('<th>Conservation</th><th>Surprise</th></tr></thead><tbody>')

    # Sort by density-adjusted surprise (plan §12 item 1), then total weight;
    # conservation is retained as a column for backward compatibility.
    data_copy = data.copy()
    data_copy['_conservation'] = (data_copy[available] > 0).sum(axis=1)
    data_copy['_total'] = data_copy[available].sum(axis=1)
    surprise_by_count = _presence_surprise_by_count(data_copy, available)
    data_copy['_surprise'] = data_copy['_conservation'].map(
        lambda k: surprise_by_count.get(int(k), 0.0))
    data_copy = data_copy.sort_values(['_surprise', '_total'], ascending=[False, False])
    # Plan round-4 F-6: keep the full set in the scroll viewport for
    # normal runs; cap only very large tables (DOM size guard).
    render_total = len(data_copy)
    if render_total > 2000:
        top = data_copy.head(1000)
    else:
        top = data_copy

    for key, row in top.iterrows():
        count = sum(1 for d in available if row.get(d, 0) > 0)
        badge = 'badge-success' if count == len(available) else 'badge-warning' if count > 1 else 'badge-danger'
        surprise = row.get('_surprise', 0.0)
        surprise_txt = '—' if surprise in (None, 0) else f'{surprise:g}'

        # §3: path length = hop count (nodes - 1); prefer the hop-weight
        # list length (max across datasets), fall back to the key's arrow
        # count.
        _norm = _normalize_path_key(key)
        _hop_hits = list(hop_by_norm.get(_norm, {}).values())
        if any(_hop_hits):
            length = max(len(h) for h in _hop_hits if h)
        else:
            length = _path_length_from_key(key)

        parts.append(f'<tr><td><strong>{key}</strong></td><td>{length}</td>')
        for d in available:
            w = row.get(d, 0)
            try:
                safe_name = analyzer.parameters._sanitize_name(d)
            except AttributeError:
                import re as _re
                safe_name = _re.sub(r'[^A-Za-z0-9_-]+', '_', str(d))
            if w > 0:
                # Get hop weights if available (normalized lookup, §3)
                hop_weights_str = ''
                hop_weights = hop_by_norm.get(_norm, {}).get(safe_name)
                if hop_weights:
                        min_w = min(hop_weights)
                        # Format with bolded minimum
                        formatted = []
                        for hw in hop_weights:
                            if hw == min_w:
                                formatted.append(f'<strong>{int(hw)}</strong>')
                            else:
                                formatted.append(str(int(hw)))
                        hop_weights_str = f'<br><span style="font-size: 0.8em; color: #666;">-{"-".join(formatted)}-</span>'
                
                parts.append(f'<td><span class="presence-check">✔️</span> {int(w)}{hop_weights_str}</td>')
            else:
                parts.append('<td><span class="presence-cross">❌</span></td>')
        parts.append(f'<td><span class="badge {badge}">{count}/{len(available)}</span></td>'
                     f'<td>{surprise_txt}</td></tr>')
    
    parts.append('</tbody></table></div>')
    if render_total > 2000:
        parts.append(
            f'<p class="note">Showing the first 1,000 of {render_total:,} rows — the full set is in the CSV export.</p>')
    return ''.join(parts)


def _generate_path_dataset_table(analyzer, dataset: str, thresholds: List[int], 
                                  nickname_map: Dict[str, str]) -> str:
    """Generate path presence table for a single dataset across all thresholds."""
    nick = nickname_map[dataset]
    safe_name = analyzer.parameters._sanitize_name(dataset)
    
    # First, identify top paths from the first threshold to limit iteration
    max_paths_to_show = 50
    top_paths_set = set()
    
    first_threshold = thresholds[0] if thresholds else None
    if first_threshold is not None:
        pdata_first = analyzer._get_path_data_for_threshold(first_threshold)
        if pdata_first is not None and not pdata_first.empty and dataset in pdata_first.columns:
            # Get top paths by weight
            top_indices = pdata_first.nlargest(max_paths_to_show * 2, dataset).index
            top_paths_set = set(top_indices)
    
    # Collect data only for top paths across thresholds
    path_data = {}  # path_key -> {threshold: (weight, hop_weights)}
    
    for threshold in thresholds:
        pdata = analyzer._get_path_data_for_threshold(threshold)
        if pdata is None or pdata.empty or dataset not in pdata.columns:
            continue
        
        # Get hop weights for this threshold
        hop_weights_dict = analyzer._get_path_hop_weights_for_threshold(threshold) if hasattr(analyzer, '_get_path_hop_weights_for_threshold') else {}
        hop_weights_norm = {_normalize_path_key(k): v for k, v in hop_weights_dict.items()}
        
        # Only iterate over top paths (using vectorized access)
        if top_paths_set:
            paths_to_process = [p for p in top_paths_set if p in pdata.index]
        else:
            paths_to_process = list(pdata.index[:max_paths_to_show * 2])
        
        for path_key in paths_to_process:
            weight = pdata.loc[path_key, dataset]
            # Handle Series (duplicate indices)
            if hasattr(weight, 'iloc'):
                weight = weight.iloc[0]
            if path_key not in path_data:
                path_data[path_key] = {}
            
            hop_weights = hop_weights_dict.get(path_key, {}).get(safe_name, []) if hop_weights_dict else []
            if not hop_weights and hop_weights_dict:
                # §3: normalized lookup — display/canonical key variants
                _norm = _normalize_path_key(path_key)
                hop_weights = hop_weights_norm.get(_norm, {}).get(safe_name, [])
            path_data[path_key][threshold] = (weight, hop_weights)
    
    if not path_data:
        return '<p>No data available.</p>'
    
    # Sort paths by lowest threshold weight
    sorted_paths = sorted(
        path_data.items(), 
        key=lambda x: (-x[1].get(thresholds[0], (0, []))[0], str(x[0]))
    )[:50]
    
    html = [f'<div style="margin-bottom: 8px; color: var(--primary-color); font-weight: 600;">Dataset: {nick} (all thresholds)</div>']
    html.append('<div class="sticky-table-container" style="overflow-x: auto;"><table><thead><tr><th>Path</th><th>Len</th>')
    for t in thresholds:
        html.append(f'<th>t={t}</th>')
    html.append('</tr></thead><tbody>')
    
    for path_key, weights in sorted_paths:
        # §3: path length = max hop-list length across thresholds, else
        # derived from the key's hop count.
        _lens = [len(h) for (_w, h) in weights.values() if h]
        length = max(_lens) if _lens else _path_length_from_key(path_key)
        html.append(f'<tr><td><strong>{path_key}</strong></td><td>{length}</td>')
        for t in thresholds:
            w, hop_weights = weights.get(t, (0, []))
            if w > 0:
                hop_weights_str = ''
                if hop_weights:
                    min_w = min(hop_weights)
                    formatted = []
                    for hw in hop_weights:
                        if hw == min_w:
                            formatted.append(f'<strong>{int(hw)}</strong>')
                        else:
                            formatted.append(str(int(hw)))
                    hop_weights_str = f'<br><span style="font-size: 0.8em; color: #666;">-{"-".join(formatted)}-</span>'
                html.append(f'<td><span class="presence-check">✔️</span> {int(w)}{hop_weights_str}</td>')
            else:
                html.append('<td><span class="presence-cross">❌</span></td>')
        html.append('</tr>')
    
    html.append('</tbody></table></div>')
    return ''.join(html)


def _presence_surprise_by_count(data: pd.DataFrame,
                                available: List[str]) -> Dict[int, float]:
    """Density-adjusted surprise per observed presence count.

    Null model: each dataset's marginal presence rate is ``p_i``; a given
    item is independently present in dataset ``i`` with probability
    ``p_i``. The p-value of observing at least ``k`` presences is the tail
    sum over all subsets of size >= k, and ``surprise = -log10(p)``. This
    is coarser than a per-item model (it depends only on ``k``) but is far
    more informative than the raw count because it accounts for datasets
    with very different edge densities, and it never claims significance
    from a raw conservation count alone.
    """
    import math
    probs = []
    for d in available:
        try:
            col = data[d]
            p = float((col > 0).mean())
        except Exception:
            p = 0.0
        probs.append(min(max(p, 1e-9), 1 - 1e-9))
    n = len(probs)
    surprise: Dict[int, float] = {}
    for k in range(n + 1):
        tail = 0.0
        for mask in range(1 << n):
            if bin(mask).count('1') < k:
                continue
            prob = 1.0
            for i in range(n):
                prob *= probs[i] if (mask >> i) & 1 else (1 - probs[i])
            tail += prob
        surprise[k] = round(-math.log10(tail), 3) if tail > 0 else float('inf')
    return surprise


def _generate_presence_table(data: pd.DataFrame, dataset_names: List[str],
                              nickname_map: Dict[str, str], threshold: int = None,
                              caption_override: str = None) -> str:
    """Generate a presence matrix table with optional threshold indication."""
    if data is None or data.empty:
        return '<p>No data available.</p>'

    available = [d for d in dataset_names if d in data.columns]
    if not available:
        return '<p>No datasets available.</p>'

    # Add threshold caption if provided
    if caption_override is not None:
        threshold_caption = (f'<div style="margin-bottom: 8px; color: var(--primary-color); '
                             f'font-weight: 600;">{html.escape(caption_override)}</div>')
    else:
        threshold_caption = f'<div style="margin-bottom: 8px; color: var(--primary-color); font-weight: 600;">Threshold = {threshold}</div>' if threshold is not None else ''
    
    parts = [f'{threshold_caption}<div class="sticky-table-container" style="overflow-x: auto;"><table><thead><tr><th>Item</th>']
    for d in available:
        parts.append(f'<th>{nickname_map[d]}</th>')
    parts.append('<th>Conservation</th><th>Surprise</th></tr></thead><tbody>')
    
    # Sort by conservation (count of datasets present), then by total weight
    data_copy = data.copy()
    data_copy['_conservation'] = (data_copy[available] > 0).sum(axis=1)
    data_copy['_total'] = data_copy[available].sum(axis=1)
    surprise_by_count = _presence_surprise_by_count(data_copy, available)
    data_copy['_surprise'] = data_copy['_conservation'].map(
        lambda k: surprise_by_count.get(int(k), 0.0))
    # Sort by surprise descending (plan §12 item 1), then total weight.
    data_copy = data_copy.sort_values(['_surprise', '_total'],
                                      ascending=[False, False])
    # Plan round-4 F-6: keep the full set in the scroll viewport for
    # normal runs; cap only very large tables (DOM size guard).
    render_total = len(data_copy)
    if render_total > 2000:
        top = data_copy.head(1000)
    else:
        top = data_copy
    
    for key, row in top.iterrows():
        count = sum(1 for d in available if row.get(d, 0) > 0)
        badge = 'badge-success' if count == len(available) else 'badge-warning' if count > 1 else 'badge-danger'
        surprise = row.get('_surprise', 0.0)
        surprise_txt = '—' if surprise in (None, 0) else f'{surprise:g}'
        
        parts.append(f'<tr><td><strong>{key}</strong></td>')
        for d in available:
            w = row.get(d, 0)
            if w > 0:
                parts.append(f'<td><span class="presence-check">✔️</span> {int(w)}</td>')
            else:
                parts.append('<td><span class="presence-cross">❌</span></td>')
        parts.append(f'<td><span class="badge {badge}">{count}/{len(available)}</span></td>'
                     f'<td>{surprise_txt}</td></tr>')
    
    parts.append('</tbody></table></div>')
    if render_total > 2000:
        parts.append(
            f'<p class="note">Showing the first 1,000 of {render_total:,} rows '
            '— the full set is in the CSV export.</p>')
    return ''.join(parts)


def _render_conservation_donut_card(dom_key: str, title: str,
                                    edge_values: List, edge_labels: List[str],
                                    edge_colors: List[str],
                                    path_values: List, path_labels: List[str],
                                    path_colors: List[str],
                                    total_edges: int, common_edges: int,
                                    total_paths: int, common_paths: int) -> str:
    """Render one conservation-distribution card (edge/path donuts).

    Shared by the Standard per-threshold section (``dom_key`` = scalar
    threshold) and the Custom per-query section (``dom_key`` = safe query
    slug).
    """
    er = (common_edges / total_edges * 100) if total_edges > 0 else 0
    pr = (common_paths / total_paths * 100) if total_paths > 0 else 0
    return f'''
            <div class="card" style="min-width: 0;">
                <h3 style="font-size: 1rem; margin-bottom: 10px; overflow-wrap: anywhere; line-height: 1.35;">{html.escape(title)}</h3>
                <div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;">
                    <div id="cons_edge_{dom_key}" style="flex: 1; min-width: 160px; max-width: 200px; height: 220px;"></div>
                    <div id="cons_path_{dom_key}" style="flex: 1; min-width: 160px; max-width: 200px; height: 220px;"></div>
                </div>
                <div style="text-align: center; color: var(--secondary-color); font-size: 0.8rem; margin-top: 8px;">
                    Edges: {common_edges}/{total_edges} ({er:.1f}%) | Paths: {common_paths}/{total_paths} ({pr:.1f}%)
                </div>
            </div>
            <script>
                (function() {{
                    const edgeValues = {json.dumps(edge_values)};
                    const edgeLabels = {json.dumps(edge_labels)};
                    const edgeColors = {json.dumps(edge_colors)};
                    const pathValues = {json.dumps(path_values)};
                    const pathLabels = {json.dumps(path_labels)};
                    const pathColors = {json.dumps(path_colors)};

                    if (edgeValues.length > 0) {{
                        Plotly.newPlot('cons_edge_{dom_key}', [{{
                            values: edgeValues,
                            labels: edgeLabels,
                            type: 'pie',
                            hole: 0.4,
                            marker: {{ colors: edgeColors }},
                            textinfo: 'percent',
                            textposition: 'inside',
                            textfont: {{ size: 10 }},
                            hoverinfo: 'label+value+percent'
                        }}], {{
                            title: {{ text: 'Edges', font: {{ size: 12 }} }},
                            showlegend: false,
                            margin: {{ t: 30, b: 10, l: 10, r: 10 }}
                        }}, {{responsive: true}});
                    }} else {{
                        document.getElementById('cons_edge_{dom_key}').innerHTML = '<p style="text-align:center;color:#999;">No edge data</p>';
                    }}

                    if (pathValues.length > 0) {{
                        Plotly.newPlot('cons_path_{dom_key}', [{{
                            values: pathValues,
                            labels: pathLabels,
                            type: 'pie',
                            hole: 0.4,
                            marker: {{ colors: pathColors }},
                            textinfo: 'percent',
                            textposition: 'inside',
                            textfont: {{ size: 10 }},
                            hoverinfo: 'label+value+percent'
                        }}], {{
                            title: {{ text: 'Paths', font: {{ size: 12 }} }},
                            showlegend: false,
                            margin: {{ t: 30, b: 10, l: 10, r: 10 }}
                        }}, {{responsive: true}});
                    }} else {{
                        document.getElementById('cons_path_{dom_key}').innerHTML = '<p style="text-align:center;color:#999;">No path data</p>';
                    }}
                }})();
            </script>
'''


def _generate_conservation_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                    key_findings: Dict, nickname_map: Dict[str, str]) -> str:
    """Generate conservation analysis section with distribution by dataset count."""
    num_datasets = len(dataset_names)
    
    # Get path_presence_matrix from comparison_report
    path_presence_matrix = None
    if hasattr(analyzer, 'comparison_report') and analyzer.comparison_report:
        path_presence_matrix = analyzer.comparison_report.get('path_presence_matrix', None)
    
    # Generate Plotly JSON
    plotly_json = "null"
    try:
        from .visualizations import ComparisonVisualizer
        vis = ComparisonVisualizer()
        # Use type-mapped results for proper cross-dataset comparison
        mapped_results = analyzer.get_mapped_results()
        plotly_json = vis.plot_conservation_across_thresholds_plotly(
            mapped_results,
            thresholds,
            align_func=analyzer.get_aligned_data,
            nickname_map=nickname_map,
            path_presence_matrix=path_presence_matrix
        )
    except Exception as e:
        print(f"Warning: Could not generate Plotly chart: {e}")

    html_parts = []
    html_parts.append(f"""
        <div id="conservation" class="section">
            <div class="section-header">🏆 Conservation Analysis</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Edge and path conservation across all {num_datasets} datasets. Shows distribution by how many datasets each edge/path appears in.
                </p>
                
                <!-- Conservation Across Thresholds Plot (Plotly) -->
                <div id="conservation_plot" style="width:100%; height:950px; margin-bottom: 50px; border: 1px solid var(--border-color); border-radius: 8px; overflow: hidden;"></div>
                <script>
                    try {{
                        var plotData = {plotly_json};
                        Plotly.newPlot('conservation_plot', plotData.data, plotData.layout);
                    }} catch (e) {{
                        console.error("Failed to render Plotly chart:", e);
                        document.getElementById('conservation_plot').innerHTML = '<p style="text-align:center; padding:20px;">Interactive chart failed to load. See static image below.</p>';
                    }}
                </script>
                
                <!-- Fallback Static Image -->
                <div style="text-align: center; margin-bottom: 30px; display: none;" id="conservation_static">
                    <img src="comparison_visualizations/conservation_across_thresholds.png" 
                         alt="Conservation Across Thresholds" 
                         style="max-width: 100%; border: 1px solid var(--border-color); border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);"
                         onerror="this.style.display='none'">
                </div>
                <script>
                    // Show static image if Plotly fails or is missing
                    if (typeof Plotly === 'undefined' || !document.getElementById('conservation_plot').innerHTML) {{
                        document.getElementById('conservation_static').style.display = 'block';
                    }}
                </script>
                
                <!-- Conservation Pie Charts Section -->
                <div style="clear: both; margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border-color);">
                    <h3 style="margin-bottom: 20px; color: var(--text-color);">Conservation Distribution by Threshold</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 20px;">
""")
    
    # Color palette for conservation levels (from conserved to unique)
    # Using a gradient from green (conserved in all) to gray (unique)
    conservation_colors = [
        '#22c55e',  # All datasets (green)
        '#84cc16',  # N-1 (lime)
        '#eab308',  # N-2 (yellow)
        '#f59e0b',  # N-3 (orange)
        '#f97316',  # N-4 (orange-red)
        '#ef4444',  # Lower (red)
        '#94a3b8',  # Unique (gray)
    ]
    
    for threshold in thresholds:
        # Get aligned data to compute distribution
        aligned = analyzer.get_aligned_data(threshold)
        available = [d for d in dataset_names if d in aligned.columns]
        n = len(available)
        
        if aligned.empty or n == 0:
            continue
        
        # Compute edge distribution using vectorized operations
        edge_counts = {}  # {count: number_of_edges}
        # Count non-zero columns per row
        presence_matrix = (aligned[available] > 0).astype(int)
        counts_per_row = presence_matrix.sum(axis=1)
        edge_counts = counts_per_row[counts_per_row > 0].value_counts().to_dict()
        
        # Get path data from path_presence_matrix
        path_counts = {}  # {count: number_of_paths}
        try:
            # Use path_presence_matrix from comparison_report
            path_presence_matrix = None
            if hasattr(analyzer, 'comparison_report') and analyzer.comparison_report:
                path_presence_matrix = analyzer.comparison_report.get('path_presence_matrix', None)
            
            if path_presence_matrix is not None and not path_presence_matrix.empty:
                # Find columns for this threshold (sanitized names with _t{threshold} suffix)
                cols_for_threshold = []
                for d in dataset_names:
                    safe_name = analyzer.parameters._sanitize_name(d)
                    col_name = f'{safe_name}_t{threshold}'
                    if col_name in path_presence_matrix.columns:
                        cols_for_threshold.append(col_name)
                
                if cols_for_threshold:
                    # Vectorized: count how many datasets each path appears in at this threshold
                    sub_df = path_presence_matrix[cols_for_threshold]
                    # Convert to boolean - handle 'True', True, and numeric > 0
                    presence_bool = sub_df.apply(lambda col: col.map(lambda v: v == 'True' or v == True or (isinstance(v, (int, float)) and v > 0)))
                    counts_per_path = presence_bool.sum(axis=1)
                    path_counts = counts_per_path[counts_per_path > 0].value_counts().to_dict()
        except Exception as e:
            pass
        
        # Build data for pie charts
        edge_values = []
        edge_labels = []
        edge_colors = []
        
        for count in range(n, 0, -1):
            if count in edge_counts and edge_counts[count] > 0:
                edge_values.append(edge_counts[count])
                if count == n:
                    edge_labels.append(f'All {n} datasets')
                elif count == 1:
                    edge_labels.append('Unique (1)')
                else:
                    edge_labels.append(f'In {count} datasets')
                # Assign color based on position from top
                color_idx = min(n - count, len(conservation_colors) - 1)
                edge_colors.append(conservation_colors[color_idx])
        
        path_values = []
        path_labels = []
        path_colors = []
        
        for count in range(n, 0, -1):
            if count in path_counts and path_counts[count] > 0:
                path_values.append(path_counts[count])
                if count == n:
                    path_labels.append(f'All {n} datasets')
                elif count == 1:
                    path_labels.append('Unique (1)')
                else:
                    path_labels.append(f'In {count} datasets')
                color_idx = min(n - count, len(conservation_colors) - 1)
                path_colors.append(conservation_colors[color_idx])
        
        # Summary stats
        kf = key_findings.get(threshold, {})
        te, ce = kf.get('total_edges', 0), kf.get('common_edges', 0)
        tp, cp = kf.get('total_paths', 0), kf.get('common_paths', 0)

        html_parts.append(_render_conservation_donut_card(
            str(threshold), f"Conservation at {_threshold_display_label(analyzer, threshold, dataset_names)}",
            edge_values, edge_labels, edge_colors,
            path_values, path_labels, path_colors,
            te, ce, tp, cp))
    
    # Links to conserved graph visualizations
    html_parts.append('''
                    </div>
                    <div class="card" style="margin-top: 30px;">
                        <h3>Conserved Graph Visualizations</h3>
                        <table>
                            <thead>
                                <tr>
                                    <th>Threshold</th>
                                    <th>Conserved Paths</th>
                                    <th>Conserved Reciprocal Graph</th>
                                </tr>
                            </thead>
                            <tbody>
    ''')

    for threshold in thresholds:
        conserved_path_file = os.path.join(
            analyzer.parameters.full_output_path,
            'conserved_paths',
            f'conserved_network_t{threshold}_network.html'
        )
        conserved_recip_file = os.path.join(
            analyzer.parameters.full_output_path,
            'conserved_reciprocal_graph',
            f'conserved_reciprocal_t{threshold}_network.html'
        )
        html_parts.append(
            f'<tr><td><strong>t={threshold}</strong></td>'
            f'<td>{_make_link(conserved_path_file, analyzer.parameters.full_output_path)}</td>'
            f'<td>{_make_link(conserved_recip_file, analyzer.parameters.full_output_path)}</td></tr>'
        )

    html_parts.append('''
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    ''')
    return ''.join(html_parts)


def _js_ident(key) -> str:
    """Sanitize a comparison-point key into a valid JS identifier."""
    import re
    ident = re.sub(r'[^A-Za-z0-9_]', '_', str(key))
    return ident or 'point'


def _html_js_arg(value) -> str:
    """Encode a JSON value for use inside a double-quoted HTML attribute.

    Generated reports contain a few inline handlers for backwards-compatible
    controls.  Raw ``json.dumps`` output is not safe in an attribute because
    string arguments contain double quotes, which terminate the attribute
    before the browser can hand the handler to JavaScript.
    """
    return html.escape(json.dumps(value), quote=True)


def _unique_dom_keys(values) -> List[str]:
    """Return stable, unique DOM-safe keys for generated controls."""
    used = set()
    result = []
    for value in values:
        base = _js_ident(value)
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f'{base}_{suffix}'
            suffix += 1
        used.add(candidate)
        result.append(candidate)
    return result


def _render_overlap_point_card(dom_key: str, title: str, labels: List[str],
                               edge_overlap: list, path_overlap: list,
                               chart_size: int = 450,
                               active: bool = False) -> str:
    """Render one overlap tab: edge/path Plotly heatmaps with a
    count/proportion toggle.

    Shared by the Standard per-threshold section (``dom_key`` = scalar
    threshold) and the Custom per-query section (``dom_key`` = safe query
    slug). DOM ids are ``overlap_tab_{dom_key}``, ``edge_overlap_{dom_key}``,
    ``path_overlap_{dom_key}``; the toggle updater is
    ``window.updateOverlapMode_{js_ident}``.
    """
    js_ident = _js_ident(dom_key)
    edge_overlap_json = json.dumps(edge_overlap)
    path_overlap_json = json.dumps(path_overlap)
    labels_json = json.dumps(labels)
    active_cls = 'active' if active else ''
    return f'''
            <div id="overlap_tab_{dom_key}" class="tab-content {active_cls}">
                <div class="card">
                    <h3>{html.escape(title)}</h3>
                    <div style="margin-bottom: 15px; text-align: center;">
                        <label style="margin-right: 20px; cursor: pointer;">
                            <input type="radio" name="overlap_mode_{dom_key}" value="count" checked
                                   onclick="updateOverlapMode_{js_ident}('count')"> Show Count
                        </label>
                        <label style="cursor: pointer;">
                            <input type="radio" name="overlap_mode_{dom_key}" value="proportion"
                                   onclick="updateOverlapMode_{js_ident}('proportion')"> Show Proportion
                        </label>
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 30px; justify-content: center;">
                        <div style="width: {chart_size}px;">
                            <h5 style="text-align: center; margin-bottom: 8px; color: var(--secondary-color);">Edge Overlap</h5>
                            <div id="edge_overlap_{dom_key}" style="width: {chart_size}px; height: {chart_size}px;"></div>
                        </div>
                        <div style="width: {chart_size}px;">
                            <h5 style="text-align: center; margin-bottom: 8px; color: var(--secondary-color);">Path Overlap</h5>
                            <div id="path_overlap_{dom_key}" style="width: {chart_size}px; height: {chart_size}px;"></div>
                        </div>
                    </div>
                    <p style="font-size: 0.8em; color: #64748b; margin-top: 10px; text-align: center;">
                        Read as: edges/paths from <strong>row</strong> dataset found in <strong>column</strong> dataset.
                        Diagonal = total in that dataset.
                    </p>
                </div>
            </div>
            <script>
                (function() {{
                    const labels = {labels_json};
                    const edgeOverlap = {edge_overlap_json};
                    const pathOverlap = {path_overlap_json};

                    // Compute proportion matrices (row-normalized: what % of row's edges/paths are in col)
                    const edgeProportion = edgeOverlap.map((row, i) =>
                        row.map((val, j) => {{
                            const diag = edgeOverlap[i][i];
                            return diag > 0 ? val / diag : 0;
                        }})
                    );
                    const pathProportion = pathOverlap.map((row, i) =>
                        row.map((val, j) => {{
                            const diag = pathOverlap[i][i];
                            return diag > 0 ? val / diag : 0;
                        }})
                    );

                    // Create text annotations for count mode
                    const edgeTextCount = edgeOverlap.map((row, i) =>
                        row.map((val, j) => {{
                            const diag = edgeOverlap[i][i];
                            const pct = diag > 0 ? (val / diag * 100).toFixed(0) : 0;
                            return i === j ? String(val) : val + ' (' + pct + '%)';
                        }})
                    );
                    const pathTextCount = pathOverlap.map((row, i) =>
                        row.map((val, j) => {{
                            const diag = pathOverlap[i][i];
                            const pct = diag > 0 ? (val / diag * 100).toFixed(0) : 0;
                            return i === j ? String(val) : val + ' (' + pct + '%)';
                        }})
                    );

                    // Create text annotations for proportion mode
                    const edgeTextProp = edgeProportion.map((row, i) =>
                        row.map((val, j) => (val * 100).toFixed(1) + '%')
                    );
                    const pathTextProp = pathProportion.map((row, i) =>
                        row.map((val, j) => (val * 100).toFixed(1) + '%')
                    );

                    // Square matrix layout with fixed aspect ratio and no grid
                    const baseLayout = {{
                        xaxis: {{
                            tickangle: -45,
                            side: 'bottom',
                            tickfont: {{size: 11}},
                            constrain: 'domain',
                            showgrid: false,
                            zeroline: false
                        }},
                        yaxis: {{
                            autorange: 'reversed',
                            tickfont: {{size: 11}},
                            scaleanchor: 'x',
                            scaleratio: 1,
                            showgrid: false,
                            zeroline: false
                        }},
                        margin: {{ l: 100, r: 30, t: 30, b: 100 }},
                        width: {chart_size},
                        height: {chart_size},
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)'
                    }};

                    // Store current mode
                    window.overlapMode_{js_ident} = 'count';

                    // Update function for toggling modes
                    window.updateOverlapMode_{js_ident} = function(mode) {{
                        window.overlapMode_{js_ident} = mode;

                        if (mode === 'count') {{
                            // Count mode
                            Plotly.react('edge_overlap_{dom_key}', [{{
                                z: edgeOverlap,
                                x: labels,
                                y: labels,
                                type: 'heatmap',
                                colorscale: [[0, '#f8fafc'], [0.5, '#93c5fd'], [1, '#2563eb']],
                                text: edgeTextCount,
                                texttemplate: '%{{text}}',
                                textfont: {{ size: 10 }},
                                hovertemplate: '%{{y}} in %{{x}}: %{{z}} edges<extra></extra>',
                                showscale: true,
                                colorbar: {{ title: 'Count', len: 0.5 }}
                            }}], baseLayout);

                            Plotly.react('path_overlap_{dom_key}', [{{
                                z: pathOverlap,
                                x: labels,
                                y: labels,
                                type: 'heatmap',
                                colorscale: [[0, '#f8fafc'], [0.5, '#c4b5fd'], [1, '#8b5cf6']],
                                text: pathTextCount,
                                texttemplate: '%{{text}}',
                                textfont: {{ size: 10 }},
                                hovertemplate: '%{{y}} in %{{x}}: %{{z}} paths<extra></extra>',
                                showscale: true,
                                colorbar: {{ title: 'Count', len: 0.5 }}
                            }}], baseLayout);
                        }} else {{
                            // Proportion mode (row-normalized)
                            Plotly.react('edge_overlap_{dom_key}', [{{
                                z: edgeProportion,
                                x: labels,
                                y: labels,
                                type: 'heatmap',
                                colorscale: [[0, '#f8fafc'], [0.5, '#93c5fd'], [1, '#2563eb']],
                                text: edgeTextProp,
                                texttemplate: '%{{text}}',
                                textfont: {{ size: 10 }},
                                hovertemplate: '%{{y}} in %{{x}}: %{{z:.1%}} of row edges<extra></extra>',
                                showscale: true,
                                colorbar: {{ title: 'Proportion', len: 0.5, tickformat: '.0%' }},
                                zmin: 0, zmax: 1
                            }}], baseLayout);

                            Plotly.react('path_overlap_{dom_key}', [{{
                                z: pathProportion,
                                x: labels,
                                y: labels,
                                type: 'heatmap',
                                colorscale: [[0, '#f8fafc'], [0.5, '#c4b5fd'], [1, '#8b5cf6']],
                                text: pathTextProp,
                                texttemplate: '%{{text}}',
                                textfont: {{ size: 10 }},
                                hovertemplate: '%{{y}} in %{{x}}: %{{z:.1%}} of row paths<extra></extra>',
                                showscale: true,
                                colorbar: {{ title: 'Proportion', len: 0.5, tickformat: '.0%' }},
                                zmin: 0, zmax: 1
                            }}], baseLayout);
                        }}
                    }};

                    // Initial plot in count mode
                    Plotly.newPlot('edge_overlap_{dom_key}', [{{
                        z: edgeOverlap,
                        x: labels,
                        y: labels,
                        type: 'heatmap',
                        colorscale: [[0, '#f8fafc'], [0.5, '#93c5fd'], [1, '#2563eb']],
                        text: edgeTextCount,
                        texttemplate: '%{{text}}',
                        textfont: {{ size: 10 }},
                        hovertemplate: '%{{y}} in %{{x}}: %{{z}} edges<extra></extra>',
                        showscale: true,
                        colorbar: {{ title: 'Count', len: 0.5 }}
                    }}], baseLayout, {{responsive: false}});

                    Plotly.newPlot('path_overlap_{dom_key}', [{{
                        z: pathOverlap,
                        x: labels,
                        y: labels,
                        type: 'heatmap',
                        colorscale: [[0, '#f8fafc'], [0.5, '#c4b5fd'], [1, '#8b5cf6']],
                        text: pathTextCount,
                        texttemplate: '%{{text}}',
                        textfont: {{ size: 10 }},
                        hovertemplate: '%{{y}} in %{{x}}: %{{z}} paths<extra></extra>',
                        showscale: true,
                        colorbar: {{ title: 'Count', len: 0.5 }}
                    }}], baseLayout, {{responsive: false}});
                }})();
            </script>
'''


def _generate_overlap_matrices_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                        nickname_map: Dict[str, str]) -> str:
    """Generate dataset overlap matrices section.
    
    Shows asymmetric N×N matrices where cell (i,j) = 
    number of edges/paths in dataset i that are also found in dataset j.
    Diagonal shows total count for each dataset.
    """
    html_parts = []
    nicknames = [nickname_map[d] for d in dataset_names]
    threshold_dom_keys = _unique_dom_keys(thresholds)
    thresholds_json = json.dumps(thresholds)
    
    html_parts.append(f"""
        <div id="overlap-matrices" class="section">
            <div class="section-header">🔀 Dataset Overlap Matrices</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Asymmetric overlap matrices. Cell (row, col) shows how many edges/paths from the <strong>row</strong> dataset 
                    are also found in the <strong>column</strong> dataset. Diagonal = total count per dataset.
                </p>
                <div class="tabs">
                    <div class="tab-buttons">
""")
    
    for i, t in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(
            f'<button class="tab-btn {active}" '
            f'data-overlap-dom-key="{threshold_dom_keys[i]}" '
            f'onclick="showOverlapTab({_html_js_arg(t)}, this)">{_threshold_tab_label(analyzer, t, dataset_names, nickname_map)}</button>')
    html_parts.append('</div>')
    
    for i, threshold in enumerate(thresholds):
        aligned = analyzer.get_aligned_data(threshold)
        
        # Always include ALL datasets, even if they have no connections at this threshold
        # This ensures consistent matrix dimensions across thresholds
        available = dataset_names  # Use all datasets, not just those with data
        n = len(available)
        
        # Ensure aligned data has columns for all datasets (fill missing with 0)
        for d in dataset_names:
            if d not in aligned.columns:
                aligned[d] = 0
        
        if n == 0:
            active = 'active' if i == 0 else ''
            html_parts.append(
                f'<div id="overlap_tab_{threshold_dom_keys[i]}" '
                f'class="tab-content {active}"><p>No datasets configured.</p></div>')
            continue
        
        # Compute edge overlap matrix (asymmetric)
        edge_overlap = [[0 for _ in range(n)] for _ in range(n)]
        for i1, d1 in enumerate(available):
            edges_in_d1 = set(aligned.index[aligned[d1] > 0]) if d1 in aligned.columns else set()
            edge_overlap[i1][i1] = len(edges_in_d1)  # Diagonal = total
            for i2, d2 in enumerate(available):
                if i1 != i2:
                    edges_in_d2 = set(aligned.index[aligned[d2] > 0]) if d2 in aligned.columns else set()
                    # Edges in d1 that are also in d2
                    edge_overlap[i1][i2] = len(edges_in_d1 & edges_in_d2)
        
        # Compute path overlap matrix (asymmetric)
        path_overlap = [[0 for _ in range(n)] for _ in range(n)]
        try:
            path_data = analyzer._get_path_data_for_threshold(threshold)
            if path_data is not None and not path_data.empty:
                for i1, d1 in enumerate(available):
                    paths_in_d1 = set(path_data.index[path_data[d1] > 0]) if d1 in path_data.columns else set()
                    path_overlap[i1][i1] = len(paths_in_d1)
                    for i2, d2 in enumerate(available):
                        if i1 != i2:
                            paths_in_d2 = set(path_data.index[path_data[d2] > 0]) if d2 in path_data.columns else set()
                            path_overlap[i1][i2] = len(paths_in_d1 & paths_in_d2)
        except Exception as e:
            pass  # Path data may not be available
        
        # Render the count/proportion heatmap card through the shared
        # helper so the Custom query section stays in lockstep.
        labels = [nickname_map[d] for d in available]
        html_parts.append(_render_overlap_point_card(
            threshold_dom_keys[i], f"Dataset Overlap at {_threshold_display_label(analyzer, threshold, dataset_names)}",
            labels, edge_overlap, path_overlap, active=(i == 0)))
    
    html_parts.append(f"""
                </div>
                <script>
                    function showOverlapTab(threshold, button) {{
                        document.querySelectorAll('#overlap-matrices .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#overlap-matrices .tab-btn').forEach(el => el.classList.remove('active'));
                        const domKey = button && button.dataset.overlapDomKey;
                        const panel = domKey ? document.getElementById('overlap_tab_' + domKey) : null;
                        if (!panel) return;
                        panel.classList.add('active');
                        if (button) button.classList.add('active');
                    }}
                </script>
            </div>
        </div>
""")
    
    return ''.join(html_parts)


def _generate_statistics_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                  nickname_map: Dict[str, str]) -> str:
    """Generate statistics section."""
    html_parts = []
    threshold_dom_keys = _unique_dom_keys(thresholds)
    html_parts.append("""
        <div id="statistics" class="section">
            <div class="section-header">📉 Statistics</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Detailed statistics per dataset and threshold.
                </p>
""")
    
    html_parts.append('<div class="tabs"><div class="tab-buttons">')
    for i, t in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(
            f'<button class="tab-btn {active}" '
            f'data-stats-dom-key="{threshold_dom_keys[i]}" '
            f'onclick="showStatsTab({_html_js_arg(t)}, this)">{_threshold_tab_label(analyzer, t, dataset_names, nickname_map)}</button>')
    html_parts.append('</div>')
    
    for i, threshold in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(
            f'<div id="stats_tab_{threshold_dom_keys[i]}" '
            f'class="tab-content {active}">')
        html_parts.append(_generate_stats_table(analyzer, dataset_names, threshold, nickname_map))
        html_parts.append('</div>')
    
    html_parts.append("""
                </div>
                <script>
                    function showStatsTab(threshold, button) {
                        document.querySelectorAll('#statistics .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#statistics .tab-btn').forEach(el => el.classList.remove('active'));
                        const domKey = button && button.dataset.statsDomKey;
                        const panel = domKey ? document.getElementById('stats_tab_' + domKey) : null;
                        if (!panel) return;
                        panel.classList.add('active');
                        if (button) button.classList.add('active');
                    }
                </script>
""")
    
    # Generate unified 2x2 Similarity Trends plot across thresholds
    similarity_trends_html = _generate_similarity_trends_2x2_plot(analyzer, dataset_names, thresholds, nickname_map)
    html_parts.append(similarity_trends_html)
    
    html_parts.append("""
            </div>
        </div>
""")
    
    return ''.join(html_parts)


def _generate_similarity_trends_2x2_plot(analyzer, dataset_names: List[str], thresholds: List[int],
                                          nickname_map: Dict[str, str],
                                          point_keys: List = None,
                                          point_labels: List[str] = None,
                                          point_similarities: Dict = None,
                                          axis_title: str = 'Threshold',
                                          card_title: str = 'Similarity Trends Across Thresholds') -> str:
    """
    Generate the similarity-trends grid for the four pairwise metrics.

    Layout (plan round-4 F-7): 4 ROWS (one per metric — Jaccard, Edge Rank,
    Cosine, Spearman) x n COLUMNS (one per analysis family). When the run
    has both row families, the left column shows the per-threshold rows and
    the right column the per-density rows (levels descending, i.e.
    thresholds increasing); each panel has its own category x axis in
    explicit query order — never a numeric threshold schedule, and never a
    combined interleaved axis. Runs with a single family render one column.

    Custom combination mode passes ``point_keys``/``point_similarities``
    (query id -> cached pairwise similarity frame).

    Empty-state contract: a metric row with NO plottable value in any
    query (e.g. Spearman when every pair falls below the >=10-shared-edges
    gate) still gets its axes for grid consistency, plus a centered grey
    note naming the gate and — for Spearman, when the frames carry
    ``common_edges`` — the run's maximum shared-edge count with the pair
    and query that produced it.
    """
    from .metrics import ComparisonMetrics
    import json

    metrics = ComparisonMetrics()

    # Collect data for all 4 metrics
    # Structure: {metric_name: {(d1, d2): {threshold: value}}}
    all_pair_data = {
        'jaccard': {},
        'edge_rank': {},
        'cosine': {},
        'spearman': {}
    }

    available_pairs = []
    for i, d1 in enumerate(dataset_names):
        for d2 in dataset_names[i+1:]:
            pair_key = (d1, d2)
            available_pairs.append(pair_key)
            for metric in all_pair_data:
                all_pair_data[metric][pair_key] = {}

    if point_similarities is not None:
        point_keys = list(thresholds) if point_keys is None else list(point_keys)
        if point_labels is None:
            point_labels = [str(k) for k in point_keys]
        for k in point_keys:
            sim_df = point_similarities.get(k)
            if sim_df is None or sim_df.empty:
                continue
            for _, row in sim_df.iterrows():
                pair_key = (row['dataset_1'], row['dataset_2'])
                if pair_key not in all_pair_data['jaccard']:
                    pair_key = (row['dataset_2'], row['dataset_1'])
                if pair_key not in all_pair_data['jaccard']:
                    continue
                all_pair_data['jaccard'][pair_key][k] = row.get('jaccard_similarity')
                all_pair_data['edge_rank'][pair_key][k] = row.get('edge_rank_correlation')
                all_pair_data['cosine'][pair_key][k] = row.get('cosine_similarity')
                all_pair_data['spearman'][pair_key][k] = row.get('spearman_rank_correlation')
    else:
        for threshold in thresholds:
            aligned = analyzer.get_aligned_data(threshold)
            if aligned.empty:
                continue

            available = [d for d in dataset_names if d in aligned.columns]

            for i, d1 in enumerate(available):
                for d2 in available[i+1:]:
                    pair_key = (d1, d2)
                    if pair_key not in all_pair_data['jaccard']:
                        pair_key = (d2, d1)

                    # Jaccard (set overlap)
                    c1, c2 = aligned[d1], aligned[d2]
                    s1, s2 = set(aligned.index[c1 > 0]), set(aligned.index[c2 > 0])
                    inter, union = len(s1 & s2), len(s1 | s2)
                    jac = inter / union if union > 0 else 0
                    all_pair_data['jaccard'][pair_key][threshold] = jac

                    # Get edge weights as Series
                    weights_a = aligned[d1].dropna()
                    weights_b = aligned[d2].dropna()

                    # Edge Rank (all edges, 0 for missing)
                    edge_rank = metrics.calculate_edge_list_rank_correlation(weights_a, weights_b)
                    all_pair_data['edge_rank'][pair_key][threshold] = edge_rank

                    # Cosine (all edges, 0 for missing)
                    cosine = metrics.calculate_cosine_similarity(weights_a, weights_b)
                    all_pair_data['cosine'][pair_key][threshold] = cosine

                    # Spearman (shared edges only)
                    spearman = metrics.calculate_spearman_rank_correlation(weights_a, weights_b)
                    all_pair_data['spearman'][pair_key][threshold] = spearman
    
    # Plan round-4 F-7: split the comparison axis into PER-ANALYSIS
    # COLUMNS — left column = per-threshold rows, right column =
    # per-density rows — in a 4-row (metrics) x n-col grid. Each panel has
    # its own category x axis so tick labels never collide, and density
    # panels read descending by level (thresholds increasing).
    axis_keys = point_keys if point_similarities is not None else thresholds

    thr_keys = [k for k in axis_keys if str(k).startswith('threshold=')]
    dens_keys = [k for k in axis_keys if str(k).startswith('aligned_density=')]
    other_keys = [k for k in axis_keys
                  if k not in thr_keys and k not in dens_keys]

    columns = []  # (axis_keys, column_title)
    if thr_keys:
        columns.append((thr_keys, 'Per-threshold'))
    if dens_keys:
        columns.append((dens_keys, 'Per-density'))
    if other_keys:
        columns.append((other_keys, 'Other queries'))
    if not columns:
        columns.append((list(axis_keys), ''))

    n_cols = max(1, len(columns))
    n_rows = 4
    metric_rows = [
        ('jaccard', 'Jaccard', '(set overlap) [0,1]', [0, 1], False),
        ('edge_rank', 'Edge Rank', '(all edges) [-1,1]', [-1, 1], True),
        ('cosine', 'Cosine', '(all edges) [0,1]', [0, 1], False),
        ('spearman', 'Spearman', '(shared only) [-1,1]', [-1, 1], True),
    ]
    colors = ['#3b82f6', '#f97316', '#22c55e', '#ef4444', '#8b5cf6',
              '#06b6d4', '#ec4899', '#eab308']

    col_gap = 0.08
    row_gap = 0.11
    col_w = (1.0 - (n_cols - 1) * col_gap) / n_cols
    row_h = (1.0 - (n_rows - 1) * row_gap) / n_rows

    def _col_domain(c):
        x0 = (c - 1) * (col_w + col_gap)
        return [x0, x0 + col_w]

    def _row_domain(r):
        y1 = 1.0 - (r - 1) * (row_h + row_gap)
        return [y1 - row_h, y1]

    all_traces = []
    annotations = []
    shapes = []
    xaxis_layout = {}
    yaxis_layout = {}

    def make_traces(metric_data, axis_keys_subset, x_i, y_i, show_legend,
                    y_range):
        # x-axes are numbered column-major ((r-1)*n_cols + c) while y-axes
        # are numbered row-major (r) — the two suffixes must be derived
        # independently, or every panel right of (1,1) binds to the WRONG
        # y-domain and rows 3+ reference y-axes the layout never defines
        # (Plotly then renders them full-height over the grid).
        x_suffix = '' if x_i == 1 else str(x_i)
        y_suffix = '' if y_i == 1 else str(y_i)
        traces = []
        for idx, pair_key in enumerate(available_pairs):
            d1, d2 = pair_key
            n1 = nickname_map.get(d1, d1)
            n2 = nickname_map.get(d2, d2)
            x_vals, y_vals = [], []
            for k in axis_keys_subset:
                if k in metric_data[pair_key]:
                    val = metric_data[pair_key][k]
                    if val is not None and not (
                            isinstance(val, float) and np.isnan(val)):
                        x_vals.append(k)
                        y_vals.append(val)
            if not x_vals:
                continue
            traces.append({
                'x': x_vals, 'y': y_vals, 'type': 'scatter',
                'mode': 'lines+markers', 'name': f'{n1} vs {n2}',
                'line': {'color': colors[idx % len(colors)], 'width': 2},
                'marker': {'size': 6},
                'legendgroup': f'{n1} vs {n2}',
                'showlegend': show_legend,
                'xaxis': f'x{x_suffix}', 'yaxis': f'y{y_suffix}',
            })
        avg_x, avg_y = [], []
        for k in axis_keys_subset:
            vals = [metric_data[pk].get(k) for pk in available_pairs
                    if k in metric_data[pk]]
            vals = [v for v in vals if v is not None and not (
                isinstance(v, float) and np.isnan(v))]
            if vals:
                avg_x.append(k)
                avg_y.append(sum(vals) / len(vals))
        if avg_x:
            traces.append({
                'x': avg_x, 'y': avg_y, 'type': 'scatter',
                'mode': 'lines+markers', 'name': 'Average',
                'line': {'color': '#64748b', 'width': 3, 'dash': 'dash'},
                'marker': {'size': 8, 'symbol': 'star'},
                'legendgroup': 'Average', 'showlegend': show_legend,
                'xaxis': f'x{x_suffix}', 'yaxis': f'y{y_suffix}',
            })
        return traces

    show_legend = True

    def _metric_has_data(metric_data) -> bool:
        for pair_vals in metric_data.values():
            for val in pair_vals.values():
                if val is not None and not (
                        isinstance(val, float) and np.isnan(val)):
                    return True
        return False

    def _max_shared_suffix() -> str:
        """The run's best shared-edge count behind the Spearman gate."""
        best_n, best_text = None, ''
        for k, sim_df in (point_similarities or {}).items():
            if sim_df is None or sim_df.empty \
                    or 'common_edges' not in sim_df.columns:
                continue
            for _, row in sim_df.iterrows():
                try:
                    n = int(row.get('common_edges'))
                except (TypeError, ValueError):
                    continue
                if best_n is None or n > best_n:
                    d1 = nickname_map.get(row.get('dataset_1'),
                                          row.get('dataset_1'))
                    d2 = nickname_map.get(row.get('dataset_2'),
                                          row.get('dataset_2'))
                    best_n = n
                    best_text = f' ({d1} vs {d2}, {k})'
        if best_n is None:
            return ''
        return f'; max shared in this run: {best_n}{best_text}'

    def _empty_row_note(metric: str, m_title: str) -> str:
        if metric == 'spearman':
            return (f'no {m_title} data — gated at \u226510 shared edges'
                    f'{_max_shared_suffix()}')
        return f'no {m_title} data in any query'

    for r, (metric, m_title, m_sub, y_range, zero_line) in enumerate(
            metric_rows, start=1):
        for c, (col_keys, col_title) in enumerate(columns, start=1):
            x_i = (r - 1) * n_cols + c
            y_i = r
            x_dom = _col_domain(c)
            y_dom = _row_domain(r)
            xaxis_layout[f'xaxis{"" if x_i == 1 else x_i}'] = {
                # no axis titles here: Plotly mispositions a shared-domain
                # x-axis title to the FIRST row when several x-axes span
                # the same column with automargin — the column caption is
                # emitted as a paper annotation below instead.
                'type': 'category', 'domain': x_dom,
                'tickangle': -40, 'automargin': True,
            }
            yaxis_layout[f'yaxis{"" if y_i == 1 else y_i}'] = {
                'title': 'Similarity' if c == 1 else '',
                'range': y_range, 'domain': y_dom,
            }
            all_traces.extend(make_traces(
                all_pair_data[metric], col_keys, x_i, y_i,
                show_legend and r == 1 and c == 1, y_range))
            annotations.append({
                'text': (f'<b>{m_title}</b><br>{m_sub} \u00b7 {col_title}'),
                'x': (x_dom[0] + x_dom[1]) / 2, 'y': y_dom[1] + 0.004,
                'xref': 'paper', 'yref': 'paper', 'xanchor': 'center',
                'yanchor': 'bottom', 'showarrow': False, 'font': {'size': 10},
            })
            if zero_line:
                shapes.append({
                    'type': 'line', 'x0': 0, 'x1': 1, 'y0': 0, 'y1': 0,
                    'xref': f'x{"" if x_i == 1 else x_i} domain',
                    'yref': f'y{"" if y_i == 1 else y_i}',
                    'line': {'color': 'gray', 'width': 1, 'dash': 'dot'},
                })
        # Empty-state: a metric with no plottable value anywhere must say
        # WHY instead of rendering bare axes (the APL-style confusion
        # this report keeps learning to avoid).
        if not _metric_has_data(all_pair_data[metric]):
            note_dom = _row_domain(r)
            annotations.append({
                'text': _empty_row_note(metric, m_title),
                'x': 0.5, 'y': (note_dom[0] + note_dom[1]) / 2,
                'xref': 'paper', 'yref': 'paper', 'xanchor': 'center',
                'yanchor': 'middle', 'showarrow': False,
                'font': {'size': 11, 'color': '#94a3b8'},
            })
        show_legend = False

    # One x-axis caption per column, under the last row's tick labels
    # (paper coords — deterministic, unlike axis titles here).
    for c, (_keys, _col_title) in enumerate(columns, start=1):
        x_dom = _col_domain(c)
        annotations.append({
            'text': axis_title,
            'x': (x_dom[0] + x_dom[1]) / 2, 'y': -0.052,
            'xref': 'paper', 'yref': 'paper', 'xanchor': 'center',
            'yanchor': 'top', 'showarrow': False,
            'font': {'size': 12},
        })

    height = 340 * n_rows + 200
    layout = {
        'annotations': annotations,
        **xaxis_layout,
        **yaxis_layout,
        'legend': {'orientation': 'h', 'y': -0.02 / max(n_rows - 1, 1) - 0.075,
                   'x': 0.5, 'xanchor': 'center'},
        'margin': {'t': 30, 'b': 110, 'l': 60, 'r': 30},
        'hovermode': 'x unified',
        'shapes': shapes,
    }
    figure_height = height

    plotly_data = json.dumps({'data': all_traces, 'layout': layout})
    
    return f'''
        <div class="card" style="margin-top: 30px;">
            <h3>{html.escape(card_title)}</h3>
            <p style="color: var(--secondary-color); font-size: 0.85rem; margin-bottom: 15px;">
                How similarity metrics change across the comparison axis. <strong>Edge Rank</strong> and <strong>Cosine</strong> compare
                all edges (assigning 0 to missing edges), while <strong>Spearman (shared)</strong> only compares edges present in both datasets.
                The dashed line shows the average across all dataset pairs.
            </p>
            <div id="similarity_trends_2x2" style="width: 100%; height: {figure_height}px;"></div>
            <script>
                (function() {{
                    try {{
                        var plotData = {plotly_data};
                        Plotly.newPlot('similarity_trends_2x2', plotData.data, plotData.layout, {{responsive: true}});
                    }} catch(e) {{
                        console.error("Failed to render similarity trends plot:", e);
                        document.getElementById('similarity_trends_2x2').innerHTML = '<p style="text-align:center;color:#999;">Chart failed to load</p>';
                    }}
                }})();
            </script>
        </div>
    '''


def _generate_jaccard_similarity_plot(analyzer, dataset_names: List[str], thresholds: List[int],
                                       nickname_map: Dict[str, str]) -> str:
    """Generate a line plot showing Jaccard similarity across thresholds for all dataset pairs."""
    
    # Collect Jaccard similarities for each pair at each threshold
    pair_data = {}  # {(d1, d2): {threshold: jaccard}}
    available_pairs = []
    
    for i, d1 in enumerate(dataset_names):
        for d2 in dataset_names[i+1:]:
            pair_key = (d1, d2)
            pair_data[pair_key] = {}
            available_pairs.append(pair_key)
    
    for threshold in thresholds:
        aligned = analyzer.get_aligned_data(threshold)
        if aligned.empty:
            continue
        
        available = [d for d in dataset_names if d in aligned.columns]
        
        for i, d1 in enumerate(available):
            for d2 in available[i+1:]:
                pair_key = (d1, d2)
                if pair_key not in pair_data:
                    pair_key = (d2, d1)  # Try reverse
                
                c1, c2 = aligned[d1], aligned[d2]
                s1, s2 = set(aligned.index[c1 > 0]), set(aligned.index[c2 > 0])
                inter, union = len(s1 & s2), len(s1 | s2)
                jac = inter / union if union > 0 else 0
                pair_data[pair_key][threshold] = jac
    
    # Build traces for Plotly
    traces = []
    colors = ['#3b82f6', '#f97316', '#22c55e', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#eab308']
    
    for idx, pair_key in enumerate(available_pairs):
        d1, d2 = pair_key
        n1, n2 = nickname_map.get(d1, d1), nickname_map.get(d2, d2)
        
        x_vals = []
        y_vals = []
        for t in thresholds:
            if t in pair_data[pair_key]:
                x_vals.append(t)
                y_vals.append(pair_data[pair_key][t])
        
        if x_vals:
            color = colors[idx % len(colors)]
            traces.append({
                'x': x_vals,
                'y': y_vals,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': f'{n1} vs {n2}',
                'line': {'color': color, 'width': 2},
                'marker': {'size': 8}
            })
    
    # Calculate average Jaccard across all pairs
    avg_x = []
    avg_y = []
    for t in thresholds:
        vals = [pair_data[pk].get(t) for pk in available_pairs if t in pair_data[pk]]
        vals = [v for v in vals if v is not None]
        if vals:
            avg_x.append(t)
            avg_y.append(sum(vals) / len(vals))
    
    if avg_x:
        traces.append({
            'x': avg_x,
            'y': avg_y,
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Average',
            'line': {'color': '#64748b', 'width': 3, 'dash': 'dash'},
            'marker': {'size': 10, 'symbol': 'star'}
        })
    
    layout = {
        'title': {'text': 'Jaccard Similarity Across Thresholds', 'font': {'size': 16}},
        'xaxis': {'title': 'Threshold', 'type': 'category'},
        'yaxis': {'title': 'Jaccard Index', 'range': [0, 1]},
        'legend': {'orientation': 'h', 'y': -0.2, 'x': 0.5, 'xanchor': 'center'},
        'margin': {'t': 50, 'b': 80, 'l': 60, 'r': 30},
        'hovermode': 'x unified'
    }
    
    plotly_data = json.dumps({'data': traces, 'layout': layout})
    
    return f'''
        <div class="card" style="margin-top: 30px;">
            <h3>Jaccard Similarity Trend Across Thresholds</h3>
            <p style="color: var(--secondary-color); font-size: 0.85rem; margin-bottom: 15px;">
                How edge overlap between dataset pairs changes with increasing threshold. The dashed line shows the average across all pairs.
            </p>
            <div id="jaccard_trend_plot" style="width: 100%; height: 1050px;"></div>
            <script>
                (function() {{
                    try {{
                        var plotData = {plotly_data};
                        Plotly.newPlot('jaccard_trend_plot', plotData.data, plotData.layout, {{responsive: true}});
                    }} catch(e) {{
                        console.error("Failed to render Jaccard trend plot:", e);
                        document.getElementById('jaccard_trend_plot').innerHTML = '<p style="text-align:center;color:#999;">Chart failed to load</p>';
                    }}
                }})();
            </script>
        </div>
    '''


def _generate_edge_rank_correlation_plot(analyzer, dataset_names: List[str], thresholds: List[int],
                                          nickname_map: Dict[str, str]) -> str:
    """Generate a line plot showing Edge Rank Correlation across thresholds for all dataset pairs."""
    from .metrics import ComparisonMetrics
    
    metrics = ComparisonMetrics()
    
    # Collect edge rank correlations for each pair at each threshold
    pair_data = {}  # {(d1, d2): {threshold: correlation}}
    available_pairs = []
    
    for i, d1 in enumerate(dataset_names):
        for d2 in dataset_names[i+1:]:
            pair_key = (d1, d2)
            pair_data[pair_key] = {}
            available_pairs.append(pair_key)
    
    for threshold in thresholds:
        aligned = analyzer.get_aligned_data(threshold)
        if aligned.empty:
            continue
        
        available = [d for d in dataset_names if d in aligned.columns]
        
        for i, d1 in enumerate(available):
            for d2 in available[i+1:]:
                pair_key = (d1, d2)
                if pair_key not in pair_data:
                    pair_key = (d2, d1)  # Try reverse
                
                # Get edge weights as Series
                weights_a = aligned[d1].dropna()
                weights_b = aligned[d2].dropna()
                
                # Calculate edge rank correlation
                corr = metrics.calculate_edge_list_rank_correlation(weights_a, weights_b)
                pair_data[pair_key][threshold] = corr
    
    # Build traces for Plotly
    traces = []
    colors = ['#3b82f6', '#f97316', '#22c55e', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#eab308']
    
    for idx, pair_key in enumerate(available_pairs):
        d1, d2 = pair_key
        n1, n2 = nickname_map.get(d1, d1), nickname_map.get(d2, d2)
        
        x_vals = []
        y_vals = []
        for t in thresholds:
            if t in pair_data[pair_key]:
                val = pair_data[pair_key][t]
                # Skip NaN values
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    x_vals.append(t)
                    y_vals.append(val)
        
        if x_vals:
            color = colors[idx % len(colors)]
            traces.append({
                'x': x_vals,
                'y': y_vals,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': f'{n1} vs {n2}',
                'line': {'color': color, 'width': 2},
                'marker': {'size': 8}
            })
    
    # Calculate average across all pairs, ignoring NaN values
    avg_x = []
    avg_y = []
    for t in thresholds:
        vals = [pair_data[pk].get(t) for pk in available_pairs if t in pair_data[pk]]
        # Filter out None and NaN values
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if vals:
            avg_x.append(t)
            avg_y.append(sum(vals) / len(vals))
    
    if avg_x:
        traces.append({
            'x': avg_x,
            'y': avg_y,
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Average',
            'line': {'color': '#64748b', 'width': 3, 'dash': 'dash'},
            'marker': {'size': 10, 'symbol': 'star'}
        })
    
    layout = {
        'title': {'text': 'Edge Rank Correlation Across Thresholds', 'font': {'size': 16}},
        'xaxis': {'title': 'Threshold', 'type': 'category'},
        'yaxis': {'title': 'Edge Rank Correlation', 'range': [-1, 1]},
        'legend': {'orientation': 'h', 'y': -0.2, 'x': 0.5, 'xanchor': 'center'},
        'margin': {'t': 50, 'b': 80, 'l': 60, 'r': 30},
        'hovermode': 'x unified'
    }
    
    plotly_data = json.dumps({'data': traces, 'layout': layout})
    
    return f'''
        <div class="card" style="margin-top: 30px;">
            <h3>Edge Rank Correlation Trend Across Thresholds</h3>
            <p style="color: var(--secondary-color); font-size: 0.85rem; margin-bottom: 15px;">
                Compares the ranking of edges by weight (using union of edges). Values range from -1 (inverse ranking) to +1 (identical ranking). The dashed line shows the average across all pairs.
            </p>
            <div id="edge_rank_trend_plot" style="width: 100%; height: 1050px;"></div>
            <script>
                (function() {{
                    try {{
                        var plotData = {plotly_data};
                        Plotly.newPlot('edge_rank_trend_plot', plotData.data, plotData.layout, {{responsive: true}});
                    }} catch(e) {{
                        console.error("Failed to render Edge Rank Correlation trend plot:", e);
                        document.getElementById('edge_rank_trend_plot').innerHTML = '<p style="text-align:center;color:#999;">Chart failed to load</p>';
                    }}
                }})();
            </script>
        </div>
    '''


def _generate_cosine_similarity_trend_plot(analyzer, dataset_names: List[str], thresholds: List[int],
                                          nickname_map: Dict[str, str]) -> str:
    """Generate a line plot showing Cosine Similarity across thresholds for all dataset pairs."""
    from .metrics import ComparisonMetrics
    
    metrics = ComparisonMetrics()
    
    # Collect cosine similarities for each pair at each threshold
    pair_data = {}  # {(d1, d2): {threshold: cosine_sim}}
    available_pairs = []
    
    for i, d1 in enumerate(dataset_names):
        for d2 in dataset_names[i+1:]:
            pair_key = (d1, d2)
            pair_data[pair_key] = {}
            available_pairs.append(pair_key)
    
    for threshold in thresholds:
        try:
            aligned = analyzer.get_aligned_data(threshold)
        except Exception:
            continue
        
        if aligned is None or aligned.empty:
            continue
        
        available = [d for d in dataset_names if d in aligned.columns]
        
        for i, d1 in enumerate(available):
            for d2 in available[i+1:]:
                pair_key = (d1, d2)
                if pair_key not in pair_data:
                    pair_key = (d2, d1)  # Try reverse
                
                # Get edge weights as Series
                weights_a = aligned[d1].dropna()
                weights_b = aligned[d2].dropna()
                
                # Calculate cosine similarity
                cosine = metrics.calculate_cosine_similarity(weights_a, weights_b)
                pair_data[pair_key][threshold] = cosine
    
    # Build traces for Plotly
    traces = []
    colors = ['#3b82f6', '#f97316', '#22c55e', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#eab308']
    
    for idx, pair_key in enumerate(available_pairs):
        d1, d2 = pair_key
        n1, n2 = nickname_map.get(d1, d1), nickname_map.get(d2, d2)
        
        x_vals = []
        y_vals = []
        for t in thresholds:
            if t in pair_data[pair_key]:
                val = pair_data[pair_key][t]
                # Skip NaN values
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    x_vals.append(t)
                    y_vals.append(val)
        
        if x_vals:
            color = colors[idx % len(colors)]
            traces.append({
                'x': x_vals,
                'y': y_vals,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': f'{n1} vs {n2}',
                'line': {'color': color, 'width': 2},
                'marker': {'size': 8}
            })
    
    # Calculate average across all pairs, ignoring NaN values
    avg_x = []
    avg_y = []
    for t in thresholds:
        vals = [pair_data[pk].get(t) for pk in available_pairs if t in pair_data[pk]]
        # Filter out None and NaN values
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if vals:
            avg_x.append(t)
            avg_y.append(sum(vals) / len(vals))
    
    if avg_x:
        traces.append({
            'x': avg_x,
            'y': avg_y,
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Average',
            'line': {'color': '#64748b', 'width': 3, 'dash': 'dash'},
            'marker': {'size': 10, 'symbol': 'star'}
        })
    
    layout = {
        'title': {'text': 'Cosine Similarity Across Thresholds', 'font': {'size': 16}},
        'xaxis': {'title': 'Threshold', 'type': 'category'},
        'yaxis': {'title': 'Cosine Similarity', 'range': [0, 1]},
        'legend': {'orientation': 'h', 'y': -0.2, 'x': 0.5, 'xanchor': 'center'},
        'margin': {'t': 50, 'b': 80, 'l': 60, 'r': 30},
        'hovermode': 'x unified'
    }
    
    plotly_data = json.dumps({'data': traces, 'layout': layout})
    
    return f'''
        <div class="card" style="margin-top: 30px;">
            <h3>Cosine Similarity Trend Across Thresholds</h3>
            <p style="color: var(--secondary-color); font-size: 0.85rem; margin-bottom: 15px;">
                Compares edge weight distributions using cosine similarity (scale-invariant). Higher values indicate more similar connectivity patterns. The dashed line shows the average across all pairs (N/A values excluded).
            </p>
            <div id="cosine_trend_plot" style="width: 100%; height: 1050px;"></div>
            <script>
                (function() {{
                    try {{
                        var plotData = {plotly_data};
                        Plotly.newPlot('cosine_trend_plot', plotData.data, plotData.layout, {{responsive: true}});
                    }} catch(e) {{
                        console.error("Failed to render Cosine Similarity trend plot:", e);
                        document.getElementById('cosine_trend_plot').innerHTML = '<p style="text-align:center;color:#999;">Chart failed to load</p>';
                    }}
                }})();
            </script>
        </div>
    '''


def _generate_path_rank_correlation_plot(analyzer, dataset_names: List[str], thresholds: List[int],
                                          nickname_map: Dict[str, str]) -> str:
    """Generate a line plot showing Path Rank Correlation across thresholds for all dataset pairs."""
    from .metrics import ComparisonMetrics
    
    metrics = ComparisonMetrics()
    
    # Collect path rank correlations for each pair at each threshold
    pair_data = {}  # {(d1, d2): {threshold: correlation}}
    available_pairs = []
    
    for i, d1 in enumerate(dataset_names):
        for d2 in dataset_names[i+1:]:
            pair_key = (d1, d2)
            pair_data[pair_key] = {}
            available_pairs.append(pair_key)
    
    for threshold in thresholds:
        try:
            path_df = analyzer._get_path_data_for_threshold(threshold)
        except Exception:
            continue
        
        if path_df is None or path_df.empty:
            continue
        
        available = [d for d in dataset_names if d in path_df.columns]
        
        for i, d1 in enumerate(available):
            for d2 in available[i+1:]:
                pair_key = (d1, d2)
                if pair_key not in pair_data:
                    pair_key = (d2, d1)  # Try reverse
                
                # Get path weights as Series
                paths_a = path_df[d1].dropna()
                paths_b = path_df[d2].dropna()
                
                # Calculate path rank correlation
                corr = metrics.calculate_path_list_rank_correlation(paths_a, paths_b)
                pair_data[pair_key][threshold] = corr
    
    # Build traces for Plotly
    traces = []
    colors = ['#3b82f6', '#f97316', '#22c55e', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#eab308']
    
    for idx, pair_key in enumerate(available_pairs):
        d1, d2 = pair_key
        n1, n2 = nickname_map.get(d1, d1), nickname_map.get(d2, d2)
        
        x_vals = []
        y_vals = []
        for t in thresholds:
            if t in pair_data[pair_key]:
                val = pair_data[pair_key][t]
                # Skip NaN values in plotting
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    x_vals.append(t)
                    y_vals.append(val)
        
        if x_vals:
            color = colors[idx % len(colors)]
            traces.append({
                'x': x_vals,
                'y': y_vals,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': f'{n1} vs {n2}',
                'line': {'color': color, 'width': 2},
                'marker': {'size': 8}
            })
    
    # Calculate average across all pairs (excluding NaN values)
    avg_x = []
    avg_y = []
    for t in thresholds:
        vals = [pair_data[pk].get(t) for pk in available_pairs if t in pair_data[pk]]
        # Filter out None and NaN values
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if vals:
            avg_x.append(t)
            avg_y.append(sum(vals) / len(vals))
    
    if avg_x:
        traces.append({
            'x': avg_x,
            'y': avg_y,
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Average',
            'line': {'color': '#64748b', 'width': 3, 'dash': 'dash'},
            'marker': {'size': 10, 'symbol': 'star'}
        })
    
    layout = {
        'title': {'text': 'Path Rank Correlation Across Thresholds', 'font': {'size': 16}},
        'xaxis': {'title': 'Threshold', 'type': 'category'},
        'yaxis': {'title': 'Path Rank Correlation', 'range': [-1, 1]},
        'legend': {'orientation': 'h', 'y': -0.2, 'x': 0.5, 'xanchor': 'center'},
        'margin': {'t': 50, 'b': 80, 'l': 60, 'r': 30},
        'hovermode': 'x unified',
        'shapes': [{
            'type': 'line',
            'x0': 0, 'x1': 1, 'xref': 'paper',
            'y0': 0, 'y1': 0,
            'line': {'color': 'gray', 'width': 1, 'dash': 'dot'}
        }]
    }
    
    plotly_data = json.dumps({'data': traces, 'layout': layout})
    
    return f'''
        <div class="card" style="margin-top: 30px;">
            <h3>Path Rank Correlation Trend Across Thresholds</h3>
            <p style="color: var(--secondary-color); font-size: 0.85rem; margin-bottom: 15px;">
                Compares the ranking of multi-hop paths by min_weight (using union of paths). Higher values indicate more similar path importance rankings. The dashed line shows the average across all pairs.
            </p>
            <div id="path_rank_trend_plot" style="width: 100%; height: 1050px;"></div>
            <script>
                (function() {{
                    try {{
                        var plotData = {plotly_data};
                        Plotly.newPlot('path_rank_trend_plot', plotData.data, plotData.layout, {{responsive: true}});
                    }} catch(e) {{
                        console.error("Failed to render Path Rank Correlation trend plot:", e);
                        document.getElementById('path_rank_trend_plot').innerHTML = '<p style="text-align:center;color:#999;">Chart failed to load</p>';
                    }}
                }})();
            </script>
        </div>
    '''


def _stats_blocks_html(aligned: pd.DataFrame, dataset_names: List[str],
                       nickname_map: Dict[str, str], title_suffix: str) -> str:
    """Render the shared per-point statistics blocks.

    Both modes use identical row semantics: the aligned edge count (>0),
    total/mean-positive/max weight, and pairwise Jaccard + shared-edge rank
    correlation computed from the aligned frame.
    """
    if aligned is None or aligned.empty:
        return '<p>No data available.</p>'

    available = [d for d in dataset_names if d in aligned.columns]
    if not available:
        return '<p>No datasets available.</p>'

    parts = [f'''<div class="card"><h3>Per-Dataset Statistics <span style="color: var(--primary-color);">{html.escape(title_suffix)}</span></h3>
        <table><thead><tr><th>Dataset</th><th>Edge Count</th><th>Total Weight</th><th>Mean Weight</th><th>Max Weight</th></tr></thead><tbody>''']

    for d in available:
        col = aligned[d]
        ec = int((col > 0).sum())
        tw = int(col.sum())
        mw = float(col[col > 0].mean()) if (col > 0).any() else 0
        mx = int(col.max()) if len(col) > 0 else 0
        parts.append(f'<tr><td>{nickname_map[d]}</td><td>{ec}</td><td>{tw}</td><td>{mw:.2f}</td><td>{mx}</td></tr>')

    parts.append('</tbody></table></div>')

    # Pairwise similarities
    parts.append(f'''<div class="card"><h3>Pairwise Similarities <span style="color: var(--primary-color);">{html.escape(title_suffix)}</span></h3>
        <table><thead><tr><th>Dataset 1</th><th>Dataset 2</th><th>Jaccard</th><th>Rank Corr</th><th>Common</th></tr></thead><tbody>''')

    from scipy.stats import spearmanr
    for i, d1 in enumerate(available):
        for d2 in available[i+1:]:
            c1, c2 = aligned[d1], aligned[d2]
            s1, s2 = set(aligned.index[c1 > 0]), set(aligned.index[c2 > 0])
            inter, union = len(s1 & s2), len(s1 | s2)
            jac = inter / union if union > 0 else 0
            # Use rank correlation instead of cosine
            w1, w2 = c1.values, c2.values
            # Only compute rank correlation where both have values
            mask = (w1 > 0) & (w2 > 0)
            if mask.sum() >= 2:
                rank_corr, _ = spearmanr(w1[mask], w2[mask])
                rank_corr = rank_corr if not np.isnan(rank_corr) else 0
            else:
                rank_corr = 0
            parts.append(f'<tr><td>{nickname_map[d1]}</td><td>{nickname_map[d2]}</td><td>{jac:.3f}</td><td>{rank_corr:.3f}</td><td>{inter}</td></tr>')

    parts.append('</tbody></table></div>')
    return ''.join(parts)


def _generate_stats_table(analyzer, dataset_names: List[str], threshold: int,
                           nickname_map: Dict[str, str]) -> str:
    """Generate statistics tables with threshold indication."""
    aligned = analyzer.get_aligned_data(threshold)
    return _stats_blocks_html(aligned, dataset_names, nickname_map,
                              f"(Threshold = {threshold})")


def _generate_footer() -> str:
    """Generate footer."""
    return """
        <button class="print-btn" onclick="window.print()">🖨️ Print Report</button>
        <script>
            // Plan round-4 F-6: catch-all viewport — wrap any data table not
            // already inside a scroll container so every table gets the
            // ~20-row scroll viewport.
            window.addEventListener('load', function() {
                document.querySelectorAll('table').forEach(function(t) {
                    var p = t.parentElement;
                    if (!p) return;
                    if (p.classList.contains('sticky-table-container') ||
                        p.classList.contains('table-scroll') ||
                        p.tagName === 'TEMPLATE') return;
                    var w = document.createElement('div');
                    w.className = 'sticky-table-container';
                    p.insertBefore(w, t);
                    w.appendChild(t);
                });
            });
        </script>
        <script>
            // Final initialization: ensure visible network is properly rendered
            // This runs after all network scripts have executed
            // IMPORTANT: Only redraw networks in visible containers
            // vis.js cannot calculate dimensions for hidden containers (display:none)
            window.addEventListener('load', function() {
                setTimeout(function() {
                    // Only redraw the active/visible network on load
                    // The first threshold network is visible by default
                    if (window.allThresholds && window.allThresholds.length > 0) {
                        const firstThreshold = window.allThresholds[0];
                        if (window.allNetworks && window.allNetworks[firstThreshold]) {
                            const netData = window.allNetworks[firstThreshold];
                            if (netData && netData.network) {
                                netData.network.redraw();
                                netData.network.fit({ animation: false });
                            }
                        }
                    }
                }, 500);
            });
        </script>
    </div>
</body>
</html>
"""
