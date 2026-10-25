"""Presence-driven Connection-strength section for the cross-dataset
report (2026-10-11 user directive: "the comparison report should include
the refill, ratios, adj ratios").

Per dataset x threshold: the top-K type pairs with the FULL strength
quartet — weight (synapses), connection_ratio (all-inputs share),
connection_ratio_adj (>= threshold share), refilled_total (refill mass,
present only when the delegate's refill mirror exists). Sourced from
each delegate's ``data_details/connection_type.csv`` + the refill pair
table; K = the run's ``top_edges`` cap. Presence-driven: a dataset
without delegate frames renders an explicit "—" row; total absence of
strength data renders '' so existing sections keep their bytes.
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


def _dataset_strength(base: Path, top_k: int):
    """(rows, available_cols) for one delegate folder, or None."""
    conn_file = base / "data_details" / "connection_type.csv"
    if not conn_file.exists():
        return None
    try:
        df = pd.read_csv(conn_file)
    except Exception:
        return None
    if df.empty or "type_pre" not in df.columns:
        return None
    # collapse conn_layer slices to the pair's cross-depth totals (the
    # canonical connection-table reading; per-layer rows stay in the CSV)
    agg_cols = {"weight": "sum"}
    for c in ("connection_ratio", "connection_ratio_adj"):
        if c in df.columns:
            # Per-conn-layer shares share one denominator; the pair's share
            # is their sum, not the mean (which understated multi-layer pairs).
            agg_cols[c] = "sum"
    df = df.groupby(["type_pre", "type_post"], dropna=False).agg(
        agg_cols).reset_index()
    available = ["weight"]
    for c in ("connection_ratio", "connection_ratio_adj"):
        if c in df.columns:
            available.append(c)
    # refill totals from the delegate's refill pair table (pair-level;
    # the connection mirror has no refilled_total)
    ref_file = base / "data_details" / "type_level_refill" / \
        "refill_type_pairs.csv"
    refilled_col_present = False
    if ref_file.exists():
        try:
            ref = pd.read_csv(ref_file)
            if {"type_pre", "type_post", "refilled_total"} <= set(
                    ref.columns):
                ref = ref[["type_pre", "type_post", "refilled_total"]]
                df = df.merge(ref, on=["type_pre", "type_post"],
                              how="left")
                available.append("refilled_total")
                refilled_col_present = True
        except Exception:
            pass
    df = df.sort_values("weight", ascending=False).head(top_k)
    if "refilled_total" not in df.columns:
        # the Refill column is ALWAYS rendered (explicit "—" cells when
        # the delegate has no refill mirror — never a silent omission,
        # and the table keeps a stable column set across datasets)
        df["refilled_total"] = float("nan")
    return df, available, refilled_col_present


def generate_strength_section(analyzer, dataset_names, thresholds,
                              nickname_map=None) -> str:
    """Render the Connection-strength section, or '' when no delegate
    frame exists anywhere."""
    nickname_map = nickname_map or {}
    top_k = int(getattr(analyzer.parameters, "top_edges", 50) or 50)
    blocks = []
    any_data = False
    for ds in dataset_names:
        nick = nickname_map.get(ds, ds)
        ds_rows = []
        available_union = []
        for t in thresholds:
            try:
                base = Path(analyzer.parameters.get_dataset_output_path(
                    ds, t))
            except Exception:
                continue
            got = _dataset_strength(base, top_k)
            if got is None:
                continue
            df, available, _has_refill = got
            any_data = True
            for c in available:
                if c not in available_union:
                    available_union.append(c)
            ds_rows.append((t, df))
        if not ds_rows:
            continue
        head = (f'<div class="card"><h3>{html.escape(str(nick))} — '
                'strength table</h3>')
        parts = [head]
        if "connection_ratio_adj" not in available_union:
            parts.append('<p style="color:var(--secondary-color);'
                         'font-size:0.9em;">Adjusted ratios absent from '
                         'this delegate\'s export.</p>')
        if "refilled_total" not in available_union:
            parts.append('<p style="color:var(--secondary-color);'
                         'font-size:0.9em;">No refill mirror for this '
                         'dataset (Refill column omitted).</p>')
        for t, df in ds_rows:
            parts.append(f'<p style="margin:6px 0 2px;">threshold '
                         f'<b>{t}</b> — top {len(df)} pairs by weight'
                         f' (of the delegate\'s exported pairs)</p>')
            cols = [c for c in ("weight", "connection_ratio",
                                "connection_ratio_adj", "refilled_total")
                    if c in df.columns]
            parts.append('<table><thead><tr><th>source</th><th>target</th>'
                         + ''.join(f'<th>{html.escape(c)}</th>'
                                   for c in cols)
                         + '</tr></thead><tbody>')
            for _, r in df.iterrows():
                cells = []
                for c in cols:
                    v = r.get(c)
                    if c == "refilled_total":
                        cells.append(f"{int(v):,}"
                                     if pd.notna(v) else "—")
                    elif c in ("connection_ratio", "connection_ratio_adj"):
                        cells.append(f"{v:.4g}" if pd.notna(v) else "—")
                    else:
                        cells.append(f"{v:,.0f}"
                                     if pd.notna(v) else "—")
                parts.append('<tr><td>' + html.escape(str(r["type_pre"]))
                             + '</td><td>' + html.escape(str(r["type_post"]))
                             + '</td>' + ''.join(f'<td>{c}</td>'
                                                 for c in cells)
                             + '</tr>')
            parts.append('</tbody></table>')
        parts.append('</div>')
        blocks.append(''.join(parts))
    if not any_data:
        return ""
    intro = ('<div class="card"><h2>Connection strength (Complete mode '
             'exports)</h2>'
             '<p>Per delegate: top pairs with weight (synapses), '
             'connection_ratio (all-inputs share), connection_ratio_adj '
             '(&ge;-threshold share) and refilled_total (Fast-refill mass '
             'when the mirror exists). Same ratio conventions as the '
             'Refill/Fast tabs. Rows come from each delegate\'s '
             'connection_type.csv + refill pair table; the per-pair '
             'explorers remain in the delegate pair reports under '
             'dataset_data/.</p></div>')
    return intro + ''.join(blocks)
