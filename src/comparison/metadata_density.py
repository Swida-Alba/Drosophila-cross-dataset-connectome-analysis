"""Synapse-density metadata enrichment (threshold-alignment spec, Feature A).

Computes the per-neuron synapse-density block added to every
``datasets/<safe>/<safe>_metadata.json``:

- ``per_neuron_median`` — median over ALL neurons of (pre + post): the
  "median synapse density per neuron" used by the cross-dataset
  threshold-equivalence note.
- ``per_neuron_mean``, ``per_neuron_median_post``, ``per_neuron_median_pre``.

Neurons without any synapse annotation count toward the denominator by
design (``neuron_universe: "all_neurons"`` documents the choice).

Per-neuron ``pre`` availability differs by source: NeuPrint-hosted neuron
tables carry ``pre`` + ``post`` (``pre_source: "neuron_table"``), while
local FAFB/BANC tables carry ``post`` only — there ``pre`` is derived
offline from ``merged_connections.parquet`` (sum of weight grouped by
``bodyId_pre``) and flagged ``pre_source: "derived_from_connections"``.
"""

from typing import Dict, Optional

import os

import numpy as np
import pandas as pd


def _find_connections_file(dataset_path: str) -> Optional[str]:
    """Locate the dataset's merged connections parquet (name carries the
    safe_name prefix in local downloads, e.g.
    ``flywire_FAFB_v783_merged_connections.parquet``)."""
    direct = os.path.join(dataset_path, 'merged_connections.parquet')
    if os.path.exists(direct):
        return direct
    try:
        for entry in os.listdir(dataset_path):
            if entry.endswith('merged_connections.parquet'):
                return os.path.join(dataset_path, entry)
    except OSError:
        pass
    return None


def _derive_pre_from_connections(dataset_path: str) -> Optional[pd.Series]:
    """Per-neuron presynaptic weights from merged_connections.parquet.

    Returns a Series indexed by bodyId (as string, matching neuron-table
    bodyId columns after str-normalization), or None when the connections
    file is missing/unreadable.
    """
    conn_path = _find_connections_file(dataset_path)
    if conn_path is None:
        return None
    try:
        conn_df = pd.read_parquet(
            conn_path, columns=['bodyId_pre', 'weight'])
    except Exception:
        try:
            import polars as pl
            conn_df = pl.read_csv(conn_path).select(
                ['bodyId_pre', 'weight']).to_pandas()
        except Exception:
            return None
    if conn_df.empty or 'bodyId_pre' not in conn_df.columns:
        return None
    pre = conn_df.groupby(conn_df['bodyId_pre'].astype(str))['weight'].sum()
    return pre


def compute_synapse_density(
    neuron_df: pd.DataFrame,
    dataset_path: Optional[str] = None,
    is_flywire_source: Optional[bool] = None,
) -> Dict:
    """Build the ``synapse_density`` metadata block from a neuron table.

    Args:
        neuron_df: neuron table with a ``post`` column (and ``pre`` when
            the source provides it). ``bodyId`` is used to join derived
            presynaptic sums for FAFB/BANC local-release datasets.
        dataset_path: dataset folder holding ``merged_connections.parquet``
            (needed only when ``pre`` must be derived).
        is_flywire_source: force the derivation decision; auto-detected
            from the table columns when None (derive only when ``pre`` is
            absent AND a connections file is available).

    Returns:
        Dict with per-neuron density stats, or an empty dict when the
        table has no usable synapse columns.
    """
    if neuron_df is None or neuron_df.empty:
        return {}

    post_col = 'post' if 'post' in neuron_df.columns else None
    pre_col = 'pre' if 'pre' in neuron_df.columns else None
    if post_col is None and pre_col is None:
        return {}

    n_neurons = len(neuron_df)
    if n_neurons == 0:
        return {}

    post = pd.to_numeric(neuron_df[post_col], errors='coerce').fillna(0) \
        if post_col else pd.Series(0.0, index=neuron_df.index)
    pre = pd.to_numeric(neuron_df[pre_col], errors='coerce').fillna(0) \
        if pre_col else None

    pre_source = 'neuron_table'
    derived_total = None
    if pre is None:
        # No per-neuron pre in the table: derive from the connections file
        # when the dataset is a local FAFB/BANC release download.
        derive = is_flywire_source
        if derive is None:
            derive = _find_connections_file(dataset_path or '') is not None
        derived = _derive_pre_from_connections(dataset_path) \
            if derive and dataset_path else None
        if derived is not None:
            ids = neuron_df['bodyId'].astype(str)
            pre = ids.map(derived).fillna(0).astype(float)
            derived_total = float(derived.sum())
            pre_source = 'derived_from_connections'
        else:
            pre = pd.Series(0.0, index=neuron_df.index)
            pre_source = 'unavailable'

    total = pre + post
    return {
        'per_neuron_median': float(np.median(total)) if n_neurons else 0.0,
        'per_neuron_mean': float(total.mean()) if n_neurons else 0.0,
        'per_neuron_median_post': float(np.median(post)) if n_neurons else 0.0,
        'per_neuron_median_pre': float(np.median(pre)) if n_neurons else 0.0,
        # Dataset totals, so a source table without a native `pre` column
        # does not report total_presynaptic=0 while its density block says
        # pre was derived from connections.
        # The derived pre total counts every connection row (including a
        # pre bodyId absent from the neuron table); the per-neuron join
        # above cannot. Both totals are the same on a complete table.
        'total_pre': float(pre.sum() if derived_total is None
                           else derived_total),
        'total_post': float(post.sum()),
        'total': float((pre.sum() if derived_total is None
                        else derived_total) + post.sum()),
        'neuron_universe': 'all_neurons',
        'pre_source': pre_source,
    }


def merge_density_into_metadata(
    metadata: Dict,
    neuron_df: pd.DataFrame,
    dataset_name: str,
    dataset_path: str,
) -> Dict:
    """Add/update the ``synapse_density`` block on a metadata dict in place.

    Preserves every existing key (including ``fetched_at``) and stamps
    ``density_generated_at`` when the block changed.
    """
    density = compute_synapse_density(neuron_df, dataset_path=dataset_path)
    if not density:
        return metadata
    from datetime import datetime
    metadata['synapse_density'] = density
    metadata['density_generated_at'] = datetime.now().isoformat()
    return metadata


def load_neuron_table(dataset_path: str, safe_name: str) -> Optional[pd.DataFrame]:
    """Read the all-neurons table for a dataset (parquet preferred)."""
    parquet_path = os.path.join(
        dataset_path, f'{safe_name}_allneurons_neuron_df.parquet')
    csv_path = os.path.join(
        dataset_path, f'{safe_name}_allneurons_neuron_df.csv')
    if os.path.exists(parquet_path):
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path, low_memory=False)
        except Exception:
            pass
    return None
