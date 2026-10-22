"""Presence-driven corridor lane section for the cross-dataset report
(plan-corridor-lane.md §8 / §23.4).

Rendered ONLY when at least one compared per-dataset run folder carries
``route_scope_paths/`` (the corridor lane's own namespace) — absence of
the artifacts returns '' and every existing section keeps its bytes.
Datasets without corridor artifacts get explicit "—" cells, never silent
omission. Corridor walks are compared across datasets through the
analyzer's canonical type mapping (``_get_canonical_type``) — dataset
type name spaces are NOT assumed equal. Offline discipline: no external
resources are referenced.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path


def _is_nan(v):
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _load_lane(analyzer, dataset_name: str, threshold):
    """(provenance, walk rows) for one dataset x threshold, or None."""
    try:
        base = Path(analyzer.parameters.get_dataset_output_path(
            dataset_name, threshold))
    except Exception:
        return None
    # Issue 2026-10-22 §2.1: the typed lane owns its own namespace. Prefer
    # it when present, then fall back to the Fast lane; a run can have both.
    loaded = []
    for lane_dir in ("route_scope_paths_type_pruned", "route_scope_paths"):
        prov_file = base / lane_dir / "route_provenance.json"
        if not prov_file.exists():
            continue
        try:
            prov = json.loads(prov_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        loaded.append({"prov": prov, "base": base, "lane_dir": lane_dir})
    if not loaded:
        return None

    def _rank(item):
        prov = item["prov"]
        finished = str(prov.get("status") or "") == "complete"
        block = prov.get("type_prune")
        exhaustive = (not isinstance(block, dict)
                      or block.get("type_pruned_complete") is not False)
        typed = item["lane_dir"] == "route_scope_paths_type_pruned"
        # A complete/exhaustive lane wins; typed stays first within a tier.
        # An incomplete typed lane must not hide a complete Fast lane.
        return (not (finished and exhaustive), not finished, not typed)

    return min(loaded, key=_rank)


def _walk_key(analyzer, dataset_name: str, path_str: str):
    """Canonical walk key: tuple of canonical types + hop count."""
    types = [t for t in path_str.split("->") if t]
    canon = []
    for t in types:
        try:
            canon.append(analyzer._get_canonical_type(t, dataset_name))
        except Exception:
            canon.append(t)
    return tuple(canon)


def generate_corridor_section(analyzer, dataset_names, thresholds,
                              nickname_map=None) -> str:
    """Render the corridor section, or '' when no run has lane artifacts."""
    nickname_map = nickname_map or {}
    lanes = {}
    refused = {}
    for ds in dataset_names:
        for t in thresholds:
            lane = _load_lane(analyzer, ds, t)
            if lane is None:
                continue
            if lane["prov"].get("status") == "complete":
                # a dataset's lowest-tier lane wins (sweeps fire there only)
                if ds not in lanes or t < lanes[ds][0]:
                    lanes[ds] = (t, lane)
            elif lane["prov"].get("status") == "refused":
                # a REFUSED lane is disclosed, never silently "—"
                if ds not in refused or t < refused[ds][0]:
                    refused[ds] = (t, lane)

    if not lanes and not refused:
        return ""

    typed_only = bool(lanes) and all(
        isinstance(lane["prov"].get("type_prune"), dict)
        for _t, lane in lanes.values())
    incomplete = []
    for ds, (t, lane) in sorted(lanes.items()):
        block = lane["prov"].get("type_prune")
        if (isinstance(block, dict)
                and block.get("type_pruned_complete") is False):
            incomplete.append((ds, t, block))
    parts = ['<div class="card">']
    parts.append(
        '<h2>Type-pruned lane (INCOMPLETE &mdash; see notes)</h2>'
        if typed_only and incomplete else
        '<h2>Type-pruned complete lane (every type walk over the pruned '
        'graph, bodyId-validated)</h2>' if typed_only else
        '<h2>Fast mode (corridor lane — strongest type-level routes at the '
        'asked threshold)</h2>')
    for ds, t, block in incomplete:
        reason = str(block.get("completeness_reason") or "incomplete")
        parts.append(
            '<p><b>INCOMPLETE</b> typed lane for '
            + html.escape(str(nickname_map.get(ds, ds))) + ' @ ' + str(t)
            + ': <code>type_pruned_complete=false</code> ('
            + html.escape(reason) + '). The numbers cover only the '
            'enumerated walks.</p>')
    for ds, (t, lane) in sorted(refused.items()):
        reason = str(lane["prov"].get("reason") or "refused")
        parts.append(
            '<p style="color:var(--secondary-color);font-size:0.9em;">'
            + html.escape(str(nickname_map.get(ds, ds))) + ' @ ' + str(t)
            + ': Fast mode refused &mdash; ' + html.escape(reason)
            + '</p>')
    parts.append(
        '<p>The type graph is pruned coarsely, EVERY type walk over the '
        'pruned graph is enumerated, each is validated at bodyId level, '
        'and only realized paths clearing the exact <code>min_ratio_adj</code> '
        'gate are exported (<code>route_scope_paths_type_pruned/</code> '
        'per run folder; '
        'never merged into the compared paths). The lane is single-floor and '
        'its <code>type_pruned_complete</code> is separate from the Complete '
        'lane\'s <code>paths_complete</code>. Datasets without lane '
        'artifacts are listed with "—" cells.</p>'
        if typed_only else
        '<p>Standalone route-scoped re-enumeration at each run\'s ASKED '
        'threshold, restricted to validated top-M* type corridors '
        '(<code>route_scope_paths/</code> per run folder; never merged '
        'into the compared paths). Datasets without lane artifacts are '
        'listed with "—" cells.</p>')

    # ---- per-dataset plan summary --------------------------------------
    parts.append('<div class="card"><h3>Per-dataset plan summary</h3>')
    parts.append('<table><thead><tr><th>Dataset</th><th>Threshold</th>'
                 '<th>Validated walks</th><th>bodyId paths</th>'
                 '<th>Scope pairs</th><th>Class floors (min..max)</th>'
                 '<th>Runtime (s)</th><th>Notes</th></tr></thead><tbody>')
    for ds in dataset_names:
        nick = nickname_map.get(ds, ds)
        if ds not in lanes:
            parts.append(f'<tr><td>{html.escape(str(nick))}</td>'
                         '<td>—</td><td>—</td><td>—</td><td>—</td>'
                         '<td>—</td><td>—</td><td>no corridor artifacts'
                         '</td></tr>')
            continue
        t, lane = lanes[ds]
        prov = lane["prov"]
        floors = [v["floor"] for v in prov.get("classes", {}).values()
                  if v.get("floor") is not None]
        notes = []
        if prov.get("time_capped"):
            notes.append("time-capped")
        type_block = prov.get("type_prune")
        if (isinstance(type_block, dict)
                and type_block.get("type_pruned_complete") is False):
            notes.append("typed lane INCOMPLETE: "
                         + str(type_block.get("completeness_reason")
                               or "incomplete"))
        bitten = sum(1 for v in prov.get("classes", {}).values()
                     if v.get("stop") == "budget")
        notes.append(f"{bitten} class(es) budget-bound")
        floor_txt = (f"{min(floors)}..{max(floors)}" if floors else "—")
        parts.append(
            f'<tr><td>{html.escape(str(nick))}</td><td>{t}</td>'
            f'<td>{prov.get("validated_walks", "—")}</td>'
            f'<td>{prov.get("bodyId_paths", "—")}</td>'
            f'<td>{prov.get("scope_pairs", "—")}</td>'
            f'<td>{floor_txt}</td>'
            f'<td>{prov.get("runtime_s", "—")}</td>'
            f'<td>{html.escape(", ".join(notes))}</td></tr>')
    parts.append('</tbody></table></div>')

    # ---- type-pruned complete lane (P2-N/P2-O) --------------------------
    typed_rows = []
    for ds, (t, lane) in sorted(lanes.items()):
        block = lane["prov"].get("type_prune")
        if not isinstance(block, dict):
            continue
        typed_rows.append((nickname_map.get(ds, ds), t, block,
                           lane["prov"].get("type_pruned_complete")))
    if typed_rows:
        parts.append('<div class="card"><h3>Type-pruned complete lane</h3>')
        parts.append(
            '<p>Separate SINGLE-FLOOR lane: the type graph is pruned on the '
            'pair-level basis, EVERY type walk over the pruned graph is '
            'enumerated, each is validated at bodyId level, and only paths '
            'clearing the exact realized <code>min_ratio_adj</code> gate are '
            'exported. <code>type_pruned_complete</code> is a different '
            'claim from the Complete lane\'s <code>paths_complete</code>; a '
            'cap or an unverified walk forbids it. When a '
            '<code>scope_mode</code> is present, the claim is complete '
            'WITHIN the per-pair scope only: each pair is bounded at '
            '<code>min(Lt + pad, hard cap)</code>, and unreached pairs are '
            'listed, never given a global bound.</p>')
        parts.append('<table><thead><tr><th>Dataset</th><th>Threshold</th>'
                     '<th>Basis / floor</th><th>Scope</th><th>Pairs</th>'
                     '<th>Candidate walks</th><th>Validated</th>'
                     '<th>Phantoms</th><th>Unverified</th>'
                     '<th>Weak-filtered</th><th>Exported</th>'
                     '<th>type_pruned_complete</th>'
                     '<th>filter_complete</th><th>Stop</th>'
                     '</tr></thead><tbody>')
        for nick, t, block, complete in typed_rows:
            pair_txt = (f"{block.get('pairs_before', '—')} → "
                        f"{block.get('pairs_after', '—')}")
            basis_txt = (f"{block.get('basis', '—')} @ "
                         f"{block.get('min_ratio_adj', '—')}")
            scope_txt = ('—' if not block.get('scope_mode') else
                         f"{block.get('scope_mode')} (Lt "
                         f"{block.get('scope_lt', '—')}, pad "
                         f"{block.get('scope_pad', '—')}, "
                         f"scope_complete={block.get('scope_complete')})")
            parts.append(
                f'<tr><td>{html.escape(str(nick))}</td><td>{t}</td>'
                f'<td>{html.escape(basis_txt)}</td>'
                f'<td>{html.escape(scope_txt)}</td>'
                f'<td>{pair_txt}</td>'
                f'<td>{block.get("candidate_walks", "—")}</td>'
                f'<td>{block.get("validated_walks", "—")}</td>'
                f'<td>{block.get("phantoms", "—")}</td>'
                f'<td>{block.get("unverified", "—")}</td>'
                f'<td>{block.get("filtered_weak", "—")}</td>'
                f'<td>{block.get("accepted_paths", "—")}</td>'
                f'<td>{complete}</td>'
                f'<td>{block.get("filter_complete", "—")}</td>'
                f'<td>{html.escape(str(block.get("completeness_reason", "—")))}'
                f'</td></tr>')
        parts.append('</tbody></table></div>')

    # ---- canonical-mapped corridor matrix -------------------------------
    parts.append('<div class="card"><h3>Corridor comparison '
                 '(canonical type mapping)</h3>')
    per_ds_walks = {}
    tau_by_key = {}
    for ds, (_t, lane) in lanes.items():
        csv = lane["base"] / lane["lane_dir"] / "route_paths_type.csv"
        keys = set()
        try:
            import polars as pl
            rows = pl.read_csv(csv)
            adj_col = ('min_ratio_adj' in rows.columns)
            for path_str, tau, min_adj in zip(
                    rows["path"].to_list(),
                    rows["applied_tau"].to_list(),
                    (rows["min_ratio_adj"].to_list() if adj_col
                     else [None] * rows.height)):
                k = _walk_key(analyzer, ds, path_str)
                keys.add(k)
                tau_by_key.setdefault(k, {})[ds] = (tau, min_adj)
        except Exception:
            pass
        per_ds_walks[ds] = keys
    lane_datasets = sorted(per_ds_walks)
    all_keys = sorted(set().union(*per_ds_walks.values())) if \
        per_ds_walks else []
    shared = [k for k in all_keys
              if all(k in per_ds_walks[ds] for ds in lane_datasets)]
    parts.append('<table><thead><tr><th>Corridor walk (canonical)</th>' + ''.join(
        f'<th>{html.escape(str(nickname_map.get(ds, ds)))}</th>'
        for ds in lane_datasets) + '</tr></thead><tbody>')
    for k in sorted(set(all_keys), key=lambda x: (-sum(
            1 for ds in lane_datasets if x in per_ds_walks[ds]), x)):
        cells = []
        for ds in lane_datasets:
            if k in per_ds_walks[ds]:
                entry = tau_by_key.get(k, {}).get(ds)
                if isinstance(entry, tuple):
                    tau, min_adj = entry
                    adj_txt = (f', adj {min_adj:.3g}'
                               if min_adj is not None
                               and not _is_nan(min_adj) else '')
                    cells.append(f'✓ (τ={tau:g}{adj_txt})'
                                 if tau is not None else '✓')
                else:
                    cells.append('✓')
            else:
                cells.append('—')
        parts.append('<tr><td>' + ' → '.join(
            html.escape(str(t)) for t in k) + '</td>' + ''.join(
            f'<td>{c}</td>' for c in cells) + '</tr>')
    parts.append('</tbody></table>')
    if len(lane_datasets) > 1:
        parts.append(f'<p>{len(shared)} corridor walk(s) shared by all '
                     f'{len(lane_datasets)} lane datasets (canonical '
                     'mapping); "—" marks a route absent in that dataset\'s '
                     'validated corridors.</p>')
    parts.append('</div>')

    # ---- gap partition ---------------------------------------------------
    gap_rows = []
    for ds, (t, lane) in lanes.items():
        gap = lane["prov"].get("gap_partition")
        if gap:
            gap_rows.append((nickname_map.get(ds, ds), t, gap))
    if gap_rows:
        parts.append('<div class="card"><h3>Corridor vs main-run paths '
                     '(gap partition)</h3>')
        parts.append('<table><thead><tr><th>Dataset</th><th>Threshold</th>'
                     '<th>Main type paths</th><th>Corridor walks</th>'
                     '<th>Shared</th><th>Corridor-only</th></tr></thead>'
                     '<tbody>')
        for nick, t, gap in gap_rows:
            parts.append(
                f'<tr><td>{html.escape(str(nick))}</td><td>{t}</td>'
                f'<td>{gap.get("main_paths", "—")}</td>'
                f'<td>{gap.get("corridor_walks", "—")}</td>'
                f'<td>{gap.get("both", "—")}</td>'
                f'<td>{gap.get("corridor_only", "—")}</td></tr>')
        parts.append('</tbody></table></div>')

    # ---- refill totals per dataset (X1: cross-dataset refill carry;
    # shared basis with the refill tab, independent scope) --------------
    refill_rows = []
    for ds, (t, lane) in lanes.items():
        try:
            base = Path(analyzer.parameters.get_dataset_output_path(ds, t))
        except Exception:
            continue
        ref_csv = base / 'type_level_refill' / 'refill_connection_type.csv'
        if not ref_csv.exists():
            continue
        try:
            import polars as _pl
            rf = _pl.read_csv(ref_csv)
            total_col = ('refilled_total' if 'refilled_total' in rf.columns
                         else None)
            if total_col:
                refill_rows.append((
                    nickname_map.get(ds, ds), t, rf.height,
                    int(rf[total_col].sum()), len(refill_rows)))
        except Exception:
            continue
    if refill_rows:
        parts.append('<div class="card"><h3>Refill totals per dataset '
                     '(shared basis; refill scope)</h3>')
        parts.append('<table><thead><tr><th>Dataset</th><th>Threshold</th>'
                     '<th>Refilled pairs</th><th>Refilled total (synapses)'
                     '</th></tr></thead><tbody>')
        for nick, t, pairs_n, total, _ in refill_rows:
            parts.append(f'<tr><td>{html.escape(str(nick))}</td>'
                         f'<td>{t}</td><td>{pairs_n}</td>'
                         f'<td>{total:,}</td></tr>')
        parts.append('</tbody></table>'
                     '<p style="color:var(--secondary-color);font-size:0.9em;">'
                     'Same ratio conventions as the corridor tables above '
                     '(plain + adj); refill scope stays independent of the '
                     'corridor scope.</p></div>')

    parts.append('</div>')
    return ''.join(parts)
