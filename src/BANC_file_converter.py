import json
import os
from datetime import datetime, timezone
import pandas as pd

# Round-7 report: this module prints non-ASCII status glyphs on paths that
# never import coana (its own __main__, converter CLIs), so install the
# UTF-8 stdio guard at the source — idempotent, never raises.
try:
    from utils.console_encoding import ensure_utf8_stdio
except ImportError:
    try:
        from .utils.console_encoding import ensure_utf8_stdio
    except ImportError:
        ensure_utf8_stdio = None
if ensure_utf8_stdio is not None:
    ensure_utf8_stdio()

try:
    from .utils.parquet_utils import (
        parquet_is_reusable, write_file_atomic, write_parquet_atomic)
except ImportError:  # pragma: no cover - src laid bare on sys.path
    from utils.parquet_utils import (
        parquet_is_reusable, write_file_atomic, write_parquet_atomic)

try:
    from .flywire_ids import (
        canonicalize_flywire_id_expr,
        dataset_folder,
        normalize_flywire_id_columns,
    )
except ImportError:
    from flywire_ids import (
        canonicalize_flywire_id_expr,
        dataset_folder,
        normalize_flywire_id_columns,
    )

try:
    from .utils.flywire_readiness import print_download_instructions
except ImportError:
    try:
        from utils.flywire_readiness import print_download_instructions
    except ImportError:
        print_download_instructions = None

def _read_banc_csv(read_path, string_columns=()):
    """Read a (possibly gzipped) BANC CSV via Polars and return pandas.

    Polars decompresses gzip with multiple threads and parses CSV with
    vectorized kernels, while the returned frame keeps the same column
    semantics (string columns keep the nullable pandas 'string' dtype).
    """
    import polars as pl

    # Only override columns that actually exist: polars would otherwise
    # create null placeholder columns for missing names.
    header = pl.read_csv(read_path, n_rows=0)
    overrides = {
        column: pl.Utf8
        for column in string_columns
        if column in header.columns
    }
    frame = pl.read_csv(read_path, schema_overrides=overrides)
    result = frame.to_pandas()
    for column in overrides:
        result[column] = result[column].astype('string')
    return result


def process_neurons_dataframe(df, save_path, save_csv_path=None):
    """
    Write an already-mapped neuron DataFrame into the parquet/CSV tables.

    The Codex CSV path renames columns first and lands here; the public
    bucket preparation (banc_public_data) builds the final-schema frame
    directly.  Both share the same canonicalization and column ordering.
    """
    try:
        print(f"  ⏳ Writing neuron tables -> {save_path}...")

        # Work on an explicit copy: callers may pass slices of larger
        # frames, and the column writes below must not operate on views.
        df = df.copy()

        normalize_flywire_id_columns(df, ['bodyId'])

        # Deduplicate if needed
        if df.duplicated(subset=['bodyId']).any():
            print("  Deduplicating dataframe...")
            df = df.drop_duplicates(subset=['bodyId'])

        # Fill missing standard columns
        if 'type' not in df.columns: df['type'] = 'Unknown'
        df['type'] = df['type'].fillna('Unknown')

        # Instance: use type if no specific name column
        df['instance'] = df['type']

        df['post'] = 0 # Placeholder

        # Select columns (keep all renamed + others)
        standard_cols = ['bodyId', 'type', 'instance', 'post', 'super_class', 'Class', 'Sub Class', 'Soma side', 'hemilineage', 'nerve', 'flow', 'nt_type']

        # Add any other columns
        all_cols = list(df.columns)
        ordered_cols = [c for c in standard_cols if c in all_cols] + [c for c in all_cols if c not in standard_cols]
        df = df[ordered_cols]

        # Sort by bodyId
        df = df.sort_values('bodyId')

        print(f"  Saving to Parquet: {save_path}...")
        write_parquet_atomic(
            save_path,
            lambda temp: df.to_parquet(temp, index=False,
                                       compression='snappy'))

        if save_csv_path:
            print(f"  Saving to CSV: {save_csv_path}...")
            write_file_atomic(
                save_csv_path,
                lambda temp: df.to_csv(temp, index=False))

        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"  ✓ Conversion complete ({len(df):,} neurons). Output size: {file_size_mb:.2f} MB")
        return True

    except Exception as e:
        print(f"  ⚠️ Error writing neuron tables: {e}")
        return False


def process_neurons_to_parquet(read_path, save_path, save_csv_path=None):
    """
    Process neurons.csv.gz into neuron_df parquet format for BANC.
    """
    if parquet_is_reusable(save_path):
        print(f"  ✓ Found existing converted file: {save_path}")
        return True

    print(f"  ⏳ Processing {read_path} -> {save_path}...")
    
    if not os.path.exists(read_path):
        print(f"  ⚠️ Error: Input file not found: {read_path}")
        return False

    try:
        # Read neurons
        print("  Reading neuron data...")
        df = _read_banc_csv(read_path, string_columns=('Root ID', 'bodyId'))
        
        # Rename columns to match coana expectations
        # BANC columns: ['Root ID', 'Top in/out region', 'Community labels', 'Predicted NT type', 
        # 'Predicted NT confidence', 'Verified NT type', 'Verified Neuropeptide', 'Body Part', 
        # 'Function', 'Flow', 'Super Class', 'Class', 'Sub Class', 'Hemilineage', 'Nerve', 
        # 'Soma side', 'Primary Cell Type', 'Alternative Cell Type(s)', 'Cable length (nm)', 
        # 'Surface area (nm^2)', 'Volume (nm^3)']
        
        rename_map = {
            'Root ID': 'bodyId',
            'Primary Cell Type': 'type',
            'Super Class': 'super_class',
            'Flow': 'flow',
            'Nerve': 'nerve',
            'Hemilineage': 'hemilineage',
            'Predicted NT type': 'nt_type',
            # The Codex CSV mislabeled micrometre cable lengths as "(nm)";
            # emit the honest unit so both prep paths agree.
            'Cable length (nm)': 'Cable length (µm)',
        }
        
        # Check which columns actually exist
        existing_cols = df.columns.tolist()
        actual_rename = {k: v for k, v in rename_map.items() if k in existing_cols}
        
        df = df.rename(columns=actual_rename)
        return process_neurons_dataframe(df, save_path, save_csv_path=save_csv_path)

    except Exception as e:
        print(f"  ⚠️ Error processing neurons: {e}")
        return False

def process_connections_dataframe(read_frame, save_path):
    """
    Aggregate an already-loaded connections frame into merged_connections
    parquet format (weights summed, ROIs joined).

    Accepts a path (CSV/parquet) or an in-memory polars/pandas frame with
    the Codex schema: ``pre_root_id``, ``post_root_id``,
    ``syn_count``/``num_synapses``, optional ``neuropil``.  Rows with
    placeholder id 0 are dropped.
    """
    try:
        import polars as pl

        if isinstance(read_frame, (str, os.PathLike)):
            read_path = str(read_frame)
            if read_path.endswith('.parquet'):
                df = pl.read_parquet(read_path)
            else:
                header = pl.read_csv(read_path, n_rows=0)
                overrides = {
                    column: pl.Utf8
                    for column in ('pre_root_id', 'post_root_id',
                                   'bodyId_pre', 'bodyId_post')
                    if column in header.columns
                }
                df = pl.read_csv(read_path, schema_overrides=overrides)
        elif isinstance(read_frame, pl.DataFrame):
            df = read_frame
        else:
            df = pl.from_pandas(read_frame)

        # Numeric ids (parquet products) become exact decimal strings before
        # the canonicalization below.
        for column in ('pre_root_id', 'post_root_id', 'bodyId_pre',
                       'bodyId_post'):
            if column in df.columns and df.schema[column] != pl.Utf8:
                df = df.with_columns(pl.col(column).cast(pl.Utf8))

        # Placeholder roots (id 0) carry no connectivity.
        id_columns = [c for c in ('pre_root_id', 'post_root_id',
                                  'bodyId_pre', 'bodyId_post')
                      if c in df.columns]
        for column in id_columns:
            df = df.filter(
                pl.col(column).is_not_null()
                & (pl.col(column).str.strip_chars() != '0')
            )

        # Rename for consistency (only columns that exist - polars raises
        # for missing names, where pandas silently ignored them)
        rename_map = {
            'pre_root_id': 'bodyId_pre',
            'post_root_id': 'bodyId_post',
            'syn_count': 'weight',
            'num_synapses': 'weight',
            'neuropil': 'roi'
        }
        df = df.rename({
            old: new for old, new in rename_map.items() if old in df.columns
        })

        # Canonical ID strings (strip whitespace, drop leading zeros, accept
        # integral '123.0' spellings) - same semantics as
        # normalize_flywire_id_columns, vectorized.
        def _canonical_ids(column: str):
            return canonicalize_flywire_id_expr(column)

        for column in ('bodyId_pre', 'bodyId_post'):
            if column in df.columns:
                df = df.with_columns(_canonical_ids(column).alias(column))

        # Aggregate weights and ROIs (sum weights, join ROIs)
        print("  Aggregating connections across ROIs...")
        if 'roi' in df.columns:
            roi_expr = (
                pl.col('roi').cast(pl.Utf8)
                .filter(
                    pl.col('roi').cast(pl.Utf8).is_not_null()
                    & (pl.col('roi').cast(pl.Utf8) != 'nan')
                )
                .unique()
                .sort()
                .str.join('|')
            )
            df = df.group_by(['bodyId_pre', 'bodyId_post']).agg([
                pl.col('weight').sum().alias('weight'),
                roi_expr.alias('roi'),
            ])
        else:
            print("  Note: 'roi' column not found, aggregating weights only.")
            df = df.group_by(['bodyId_pre', 'bodyId_post']).agg(
                pl.col('weight').sum().alias('weight')
            ).with_columns(pl.lit('WholeBrain').alias('roi'))

        # Sort by pre, post
        print("  Sorting connections...")
        df = df.sort(['bodyId_pre', 'bodyId_post'])

        print(f"  Saving to Parquet: {save_path}...")
        write_parquet_atomic(
            save_path,
            lambda temp: df.write_parquet(temp, compression='snappy'))

        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"  ✓ Conversion complete. Output size: {file_size_mb:.2f} MB")
        return True

    except Exception as e:
        print(f"  ⚠️ Error processing connections: {e}")
        return False


def process_connections_to_parquet(read_path, save_path):
    """
    Process connections_princeton.csv.gz into merged_connections parquet format.
    Aggregates weights across ROIs.

    The read + aggregation run in Polars: multi-threaded gzip and vectorized
    groupby are far faster than the pandas pipeline on large tables, with
    identical results.
    """
    if parquet_is_reusable(save_path):
        print(f"  ✓ Found existing converted file: {save_path}")
        return True

    print(f"  ⏳ Processing {read_path} -> {save_path}...")

    if not os.path.exists(read_path):
        print(f"  ⚠️ Error: Input file not found: {read_path}")
        return False

    return process_connections_dataframe(read_path, save_path)

def update_neuron_post_counts(neuron_path, conn_path, save_csv_path=None):
    """
    Update the 'post' column in the neuron DataFrame by summing weights from the connections DataFrame.
    """
    print("  Updating neuron post-synaptic counts from connections...")
    try:
        # Load neurons
        if neuron_path.endswith('.parquet'):
            df_neuron = pd.read_parquet(neuron_path)
        else:
            df_neuron = pd.read_csv(neuron_path, dtype={'bodyId': str})
            
        # Load connections (only need bodyId_post and weight)
        print("  Loading connections for count calculation...")
        if conn_path.endswith('.parquet'):
            df_conn = pd.read_parquet(conn_path, columns=['bodyId_post', 'weight'])
        else:
            df_conn = pd.read_csv(conn_path, usecols=['bodyId_post', 'weight'], dtype={'bodyId_post': str})
            
        # Calculate post counts
        print("  Calculating post counts...")
        # Ensure bodyId_post is string
        normalize_flywire_id_columns(df_conn, ['bodyId_post'])
        post_counts = df_conn.groupby('bodyId_post')['weight'].sum()
        
        # Update df_neuron
        # Ensure bodyId is string
        normalize_flywire_id_columns(df_neuron, ['bodyId'])
        
        # Map counts
        print("  Mapping counts to neurons...")
        df_neuron['post'] = df_neuron['bodyId'].map(post_counts).fillna(0).astype(int)
        
        # Save
        print(f"  Saving updated neurons to {neuron_path}...")
        if neuron_path.endswith('.parquet'):
            write_parquet_atomic(
                neuron_path,
                lambda temp: df_neuron.to_parquet(
                    temp, index=False, compression='snappy'))
        else:
            write_file_atomic(
                neuron_path, lambda temp: df_neuron.to_csv(temp, index=False))

        if save_csv_path:
            print(f"  Saving updated neurons to {save_csv_path}...")
            write_file_atomic(
                save_csv_path, lambda temp: df_neuron.to_csv(temp, index=False))
            
        print("  ✓ Post counts updated.")
        return True
    except Exception as e:
        print(f"  ⚠️ Error updating post counts: {e}")
        return False


# Fields of the failure placeholder a cross-dataset metadata collection used
# to persist (``comparison_analyzer._create_empty_metadata``): none of them
# describe real data, so they must never be merged forward into a prepared
# sidecar (2026-09-18 retest, F9).
_ERROR_PLACEHOLDER_KEYS = ("source", "error", "fetched_at", "neuron_counts",
                           "synapse_counts", "roi_coverage", "coverage_notes")


def _strip_error_placeholder(data):
    """Drop a stale error placeholder from an existing sidecar payload.

    Returns *data* untouched when it carries no error marker; otherwise the
    placeholder fields (``source='error'``, its ``error`` text and the
    all-zero stat blocks written with them) are removed so the real
    provenance and table-derived stats replace them instead of sitting
    beside them.
    """
    if not isinstance(data, dict):
        return {}
    if data.get("source") != "error" and "error" not in data:
        return data
    return {k: v for k, v in data.items() if k not in _ERROR_PLACEHOLDER_KEYS}


def _patch_dataset_metadata(dataset_dir, dataset_name, source, notes=None):
    """Record the preparation provenance in the dataset metadata.json."""
    dataset_name = dataset_folder(dataset_name)
    meta_path = os.path.join(dataset_dir, f"{dataset_name}_metadata.json")
    try:
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8-sig') as handle:
                data = _strip_error_placeholder(json.load(handle))
        else:
            data = {"dataset": dataset_name, "fetched_at": None}
        data["dataset"] = dataset_name
        data["source"] = source
        if notes:
            data.setdefault("preparation_notes", []).append(notes)
        from datetime import datetime, timezone
        data["fetched_at"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S")
        with open(meta_path, 'w', encoding='utf-8') as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"  ⚠️ Could not update dataset metadata: {exc}")


def _update_post_counts_if_zero(neuron_pq, conn_pq, save_csv_path=None):
    """Populate the neuron table's ``post`` column when it is all zeros.

    Shared by the public-bucket and Codex preparation paths: both write
    ``post = 0`` placeholders that only the merged-connections table can
    fill.
    """
    if not (parquet_is_reusable(neuron_pq)
            and parquet_is_reusable(conn_pq)):
        return
    try:
        # Read just the post column to check if it's all zeros
        df_check = pd.read_parquet(neuron_pq, columns=['post'])
        if df_check['post'].sum() == 0:
            print("  ℹ️  Post counts are 0. Updating from connections...")
            update_neuron_post_counts(neuron_pq, conn_pq,
                                      save_csv_path=save_csv_path)
        else:
            print("  ✓ Post counts already populated.")
    except Exception as e:
        print(f"  ⚠️ Could not check post counts: {e}")


def build_connection_cache_from_tables(dataset_dir, cache_dir=None):
    """(Re)build the BANC connectivity cache files from the final tables.

    The pathfinding/connectivity layer reads the per-neuron connection
    cache — never the merged table directly — and BANC has no API to fill
    that cache online.  For bucket-prepared datasets both cache artifacts
    are therefore derived locally whenever the tables are newer:

    - ``connections.parquet``: the merged table rows (schema ``bodyId_pre,
      bodyId_post, weight, roi, cached_date``), atomically replaced;
    - ``neuron_index_state.parquet``: one row per neuron table id (stale
      or foreign ids from earlier eras are dropped), with
      ``downstream_complete=True`` — every neuron's connectivity is fully
      known from the local table, so no cache-miss re-fetch is needed.

    Returns True when the cache exists and is current afterwards.
    """
    import polars as pl

    dataset_dir = str(dataset_dir)
    dataset_name = dataset_folder(os.path.basename(os.path.normpath(dataset_dir)))
    merged = os.path.join(dataset_dir,
                          f"{dataset_name}_merged_connections.parquet")
    if not os.path.exists(merged):
        return False
    cache_dir = cache_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.normpath(dataset_dir))),
        "cache", dataset_name)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "connections.parquet")

    merged_stat = os.stat(merged)
    merged_sig = f"{merged_stat.st_mtime_ns}:{merged_stat.st_size}"
    marker = cache_file + ".src"

    def _cache_is_current() -> bool:
        """True when the cache was built from the CURRENT merged table."""
        if not (os.path.exists(cache_file) and os.path.exists(marker)):
            return False
        try:
            with open(marker, "r", encoding="ascii") as handle:
                return handle.read().strip() == merged_sig
        except OSError:
            return False

    if _cache_is_current():
        return True  # cache already covers the current table generation
    try:
        os.remove(cache_file)  # stale generation: never mix
    except OSError:
        pass

    neuron_pq = os.path.join(dataset_dir,
                             f"{dataset_name}_allneurons_neuron_df.parquet")
    table_ids = None
    if os.path.exists(neuron_pq):
        # Strictly per-version: the state rows must be a subset of this
        # release's neuron table (stale ids from earlier eras are dropped).
        table_ids = pd.read_parquet(neuron_pq, columns=["bodyId"])
        table_ids["bodyId"] = table_ids["bodyId"].astype(str)

    print(f"  ⏳ Rebuilding the BANC connection cache from "
          f"{os.path.basename(merged)}...")
    frame = (
        pl.scan_parquet(merged)
        .select(
            pl.col("bodyId_pre").cast(pl.Utf8),
            pl.col("bodyId_post").cast(pl.Utf8),
            pl.col("weight"),
            pl.col("roi"),
        )
        .filter(pl.col("bodyId_pre") != "0", pl.col("bodyId_post") != "0")
        .with_columns(
            pl.lit(datetime.now(timezone.utc).strftime("%Y-%m-%d"))
            .alias("cached_date")
        )
        .collect()
    )
    temp_file = cache_file + f".{os.getpid()}.tmp"
    frame.write_parquet(temp_file, compression="snappy")
    os.replace(temp_file, cache_file)
    print(f"  ✓ Connection cache rebuilt: {frame.height:,} connections "
          f"-> {cache_file}")

    # Rebuild the neuron-index state sidecar from the same generation so
    # stale/foreign ids cannot leak into the neuron-index dict.
    if table_ids is not None:
        outgoing = frame.group_by("bodyId_pre").agg(
            # Partner-row COUNT (the dominant convention: the NeuPrint pull
            # counts partner rows, the FlyWire import counts posts), NOT the
            # weight sum — a third unit variant invited misuse (BANC-05).
            pl.len().alias("connection_count"))
        state = pl.from_pandas(table_ids).join(
            outgoing, left_on="bodyId", right_on="bodyId_pre", how="left"
        ).with_columns(
            pl.col("connection_count").fill_null(0).cast(pl.Int64),
            pl.lit(datetime.now(timezone.utc).strftime("%Y-%m-%d"))
            .alias("last_fetched"),
            pl.lit(True).alias("downstream_complete"),
        ).select(
            "bodyId", "downstream_complete", "last_fetched",
            "connection_count",
        )
        state_file = os.path.join(cache_dir, "neuron_index_state.parquet")
        state_temp = state_file + f".{os.getpid()}.tmp"
        state.write_parquet(state_temp, compression="snappy")
        os.replace(state_temp, state_file)
    # The .src marker goes LAST: an interruption anywhere above leaves no
    # marker, so the (idempotent) rebuild simply re-runs next time instead
    # of a current-looking signature short-circuiting a stale sidecar
    # (issue-report BANC-06).
    with open(marker, "w", encoding="ascii") as handle:
        handle.write(merged_sig)
    return True


def _regenerate_banc_metadata(dataset_name, dataset_dir,
                              source="banc_public_gcs"):
    """Recompute the stats block of ``<ds>_metadata.json`` from the tables.

    Bucket preparation must produce table-true statistics: the previous
    patch-only flow left stale values behind (a FAFB coverage note, neuron
    counts derived from an old connectivity-cache sidecar, ``type_coverage``
    counting ``'Unknown'`` as typed).  Only tables are read here — never
    cache state.
    """
    dataset_name = dataset_folder(dataset_name)
    meta_path = os.path.join(dataset_dir, f"{dataset_name}_metadata.json")
    data = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8-sig") as handle:
                data = _strip_error_placeholder(json.load(handle))
        except Exception:
            data = {}

    neuron_pq = os.path.join(
        dataset_dir, f"{dataset_name}_allneurons_neuron_df.parquet")
    conn_pq = os.path.join(
        dataset_dir, f"{dataset_name}_merged_connections.parquet")
    if not parquet_is_reusable(neuron_pq):
        return False

    neurons = pd.read_parquet(neuron_pq)
    total = int(len(neurons))
    type_vals = neurons["type"].astype("string").fillna("") if \
        "type" in neurons.columns else pd.Series([""] * total)
    typed = int(((type_vals.str.strip() != "")
                 & (type_vals.str.strip() != "Unknown")).sum())

    total_synapses = 0
    if parquet_is_reusable(conn_pq):
        conn = pd.read_parquet(conn_pq, columns=["weight"])
        # Every merged row carries ``weight`` synapses; each synapse has one
        # pre- and one post-synaptic site, so both totals equal the sum.
        total_synapses = int(conn["weight"].fillna(0).sum())

    data.update({
        "dataset": dataset_name,
        "source": source,
        "neuron_counts": {
            "total": total,
            "typed": typed,
            "untyped": total - typed,
            "type_coverage": (typed / total) if total else 0,
        },
        "synapse_counts": {
            "total_presynaptic": total_synapses,
            "total_postsynaptic": total_synapses,
            "total": total_synapses,
        },
        "roi_coverage": {
            "status": "not_available_for_banc",
            "roi_list": [],
            "roi_count": 0,
            "neuron_counts_per_roi": {},
        },
        "coverage_notes": "Full brain and VNC connectome.",
    })
    try:
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"  ⚠️ Could not regenerate BANC metadata stats: {exc}")
        return False
    return True


def ensure_banc_data(dataset_name, dataset_dir):
    """
    Ensure BANC data is available and converted for the given dataset.
    """
    # Legacy ``flywire_BANC_*`` and hidden ``banc:v888`` spellings must share
    # the same safe file namespace as their canonical dataset directories.
    dataset_name = dataset_folder(dataset_name)
    # Round-7 R7-1a: an explicit-but-unknown release (banc_v999) must refuse
    # BEFORE any download, not serve the default release under that name.
    try:
        import banc_public_data
        banc_public_data.reject_unknown_banc_release(dataset_name)
    except ValueError:
        raise
    except Exception:  # noqa: BLE001 — the bucket module may be unavailable
        pass
    print(f"\nChecking BANC data for {dataset_name}...")
    print("  ℹ️  One-time preparation: raw downloads in downloads/ are converted "
          "into the local parquet tables used by every DROCAT workflow. "
          "Already-converted files are skipped on re-runs.")

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
        print(f"Created dataset folder: {dataset_dir}")

    downloads_dir = os.path.join(dataset_dir, "downloads")
    if not os.path.exists(downloads_dir):
        os.makedirs(downloads_dir, exist_ok=True)

    # Define target files
    neuron_pq = os.path.join(dataset_dir, f"{dataset_name}_allneurons_neuron_df.parquet")
    neuron_csv = os.path.join(dataset_dir, f"{dataset_name}_allneurons_neuron_df.csv")
    conn_pq = os.path.join(dataset_dir, f"{dataset_name}_merged_connections.parquet")

    # Define raw source files (manual Codex download remains the fallback)
    neurons_raw = os.path.join(downloads_dir, "neurons.csv.gz")
    conn_raw = os.path.join(downloads_dir, "connections_princeton.csv.gz")

    # --- 0. Public bucket preparation (no manual download needed) ---
    needs_neurons = not parquet_is_reusable(neuron_pq) and not os.path.exists(neurons_raw)
    needs_connections = not parquet_is_reusable(conn_pq) and not os.path.exists(conn_raw)
    if needs_neurons or needs_connections:
        print("  🌐 Preparing from the public BANC release bucket "
              "(metadata ~58 MB + connections ~76 MB, one-time, no token)...")
        try:
            import banc_public_data
            if banc_public_data.prepare_dataset_tables(dataset_name, dataset_dir):
                _patch_dataset_metadata(
                    dataset_dir, dataset_name, source="banc_public_gcs",
                    notes=("neuron metadata + connections fetched from the "
                           "public release bucket; connection thresholds "
                           "(synapse size>=3, count>=3) are baked into the "
                           "product"))
                # Bucket-prepared neuron tables carry post = 0: fill the
                # column from the merged connections exactly like the
                # Codex path below (the early return must not skip it).
                _update_post_counts_if_zero(neuron_pq, conn_pq,
                                            save_csv_path=neuron_csv)
                _regenerate_banc_metadata(dataset_name, dataset_dir)
                return True
            print("  ⚠️ Public bucket preparation incomplete; falling back "
                  "to the manual Codex download path.")
        except Exception as exc:
            print(f"  ⚠️ Public bucket preparation failed ({exc}); falling "
                  "back to the manual Codex download path.")

    all_critical_present = True

    # --- 1. Neurons ---
    if parquet_is_reusable(neuron_pq):
        print(f"  ✓ Found existing neurons: {os.path.basename(neuron_pq)}")
    else:
        print("  Checking neuron source files...")
        neurons_raw = os.path.join(downloads_dir, "neurons.csv.gz")
        
        if not os.path.exists(neurons_raw):
            print(f"  ❌ Missing required file: neurons.csv.gz")
            all_critical_present = False
        else:
            if not process_neurons_to_parquet(neurons_raw, neuron_pq, save_csv_path=neuron_csv):
                all_critical_present = False

    # --- 2. Connections ---
    if parquet_is_reusable(conn_pq):
        print(f"  ✓ Found existing connections: {os.path.basename(conn_pq)}")
    else:
        print("  Checking connection source files...")
        conn_raw = os.path.join(downloads_dir, "connections_princeton.csv.gz")
        
        if not os.path.exists(conn_raw):
            print(f"  ❌ Missing required file: connections_princeton.csv.gz")
            all_critical_present = False
        else:
            if not process_connections_to_parquet(conn_raw, conn_pq):
                all_critical_present = False

    # --- Post Counts Update ---
    _update_post_counts_if_zero(neuron_pq, conn_pq, save_csv_path=neuron_csv)

    # Regenerate the metadata stats from the final tables (table-true
    # neuron/synapse counts, banc coverage note) on every successful prep.
    # The Codex-manual layout records its own provenance (BANC-04).
    if parquet_is_reusable(neuron_pq) and parquet_is_reusable(conn_pq):
        _regenerate_banc_metadata(dataset_name, dataset_dir,
                                  source="banc_codex_manual")

    if not all_critical_present:
        print()
        if print_download_instructions is not None:
            print_download_instructions(dataset_name, dataset_dir)
        else:
            print("=" * 60)
            print("MISSING CRITICAL FILES")
            print("Please download missing files to:", downloads_dir)
            print("Standard BANC preparation uses the public release bucket; "
                  "see docs/BANC_INTEGRATION.md")
            print("=" * 60 + "\n")
        return False

    return True

if __name__ == "__main__":
    # Standard run for installation
    dataset_name = "banc_v626"
    # Determine project root (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, "datasets", dataset_name)
    
    print(f"Running BANC data preparation for {dataset_name}...")
    print(f"Target directory: {dataset_dir}")
    
    ensure_banc_data(dataset_name, dataset_dir)
