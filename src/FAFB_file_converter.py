
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
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import zipfile
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import concurrent.futures
import multiprocessing


# Columns of the synapse table no DROCAT reader consumes (audited
# 2026-09: see docs/audits/SYNAPSE_TABLE_COLUMN_AUDIT_2026-09-10.md).
# They are dropped at conversion time and compacted away in tables
# converted by older DROCAT versions.
FAFB_UNREAD_SYNAPSE_COLUMNS = ("ctr_x", "ctr_y", "ctr_z", "size", "neuropil")

# Lossless storage profile for the converted synapse table: the string
# id columns stay strings (user choice) and take DELTA_LENGTH_BYTE_ARRAY,
# the coordinate columns narrow to int32 (guarded by footer statistics;
# observed ranges stay below ~1e6) with BYTE_STREAM_SPLIT.  Measured on
# the v783 release: 1.01 GB -> 0.91 GB (-10%), values bit-identical.
# Both id naming generations are listed; absent entries are ignored.
FAFB_SYNAPSE_PROFILE = {
    "drop_columns": FAFB_UNREAD_SYNAPSE_COLUMNS,
    "casts": {column: "int32" for column in
              ("pre_x", "pre_y", "pre_z", "post_x", "post_y", "post_z")},
    "column_encoding": {
        "pre_root_id": "DELTA_LENGTH_BYTE_ARRAY",
        "post_root_id": "DELTA_LENGTH_BYTE_ARRAY",
        "pre_root_id_720575940": "DELTA_LENGTH_BYTE_ARRAY",
        "post_root_id_720575940": "DELTA_LENGTH_BYTE_ARRAY",
        "pre_x": "BYTE_STREAM_SPLIT",
        "pre_y": "BYTE_STREAM_SPLIT",
        "pre_z": "BYTE_STREAM_SPLIT",
        "post_x": "BYTE_STREAM_SPLIT",
        "post_y": "BYTE_STREAM_SPLIT",
        "post_z": "BYTE_STREAM_SPLIT",
    },
    "use_dictionary": False,
    "compression": "zstd",
    "compression_level": 7,
}

try:
    from utils.parquet_utils import (
        parquet_is_reusable, parquet_readable, reencode_parquet_lossless,
        reclaim_writer_temps, temp_sibling, write_file_atomic,
        write_parquet_atomic)
except ImportError:  # pragma: no cover - src laid bare on sys.path
    from utils.parquet_utils import (
        parquet_is_reusable, parquet_readable, reencode_parquet_lossless,
        reclaim_writer_temps, temp_sibling, write_file_atomic,
        write_parquet_atomic)


def compact_synapse_table(syn_pq, progress_callback=None) -> bool:
    """Bring a converted FAFB synapse table to the lossless layout.

    One streaming pass drops the unread legacy columns when present,
    applies the measured encodings and stamps the footer marker; tables
    already carrying the marker are left untouched.  Values are
    preserved exactly — the id columns stay strings.
    """
    return reencode_parquet_lossless(
        syn_pq, FAFB_SYNAPSE_PROFILE,
        progress_callback=progress_callback)


try:
    from .flywire_ids import canonicalize_flywire_id_expr, normalize_flywire_id_columns
except ImportError:
    from flywire_ids import canonicalize_flywire_id_expr, normalize_flywire_id_columns

try:
    from .utils.flywire_readiness import print_download_instructions as _print_flywire_download_instructions
except ImportError:
    try:
        from utils.flywire_readiness import print_download_instructions as _print_flywire_download_instructions
    except ImportError:
        _print_flywire_download_instructions = None

def _parse_swc_batch(zip_path, filenames):
    """Helper function to parse a batch of SWC files from a zip.

    Bytes-based parsing: the legacy path decoded every file to text and
    rebuilt stripped line lists before splitting; operating on the raw
    bytes is measurably faster on the ~140k-file healed bundle.
    """
    # Use column-oriented storage for memory efficiency
    data = {
        'bodyId': [], 'node_id': [], 'type': [], 
        'x': [], 'y': [], 'z': [], 'radius': [], 'parent_id': []
    }
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for filename in filenames:
                try:
                    # Handle potential folder prefixes in zip
                    body_id = os.path.basename(filename).split('.')[0]
                    
                    content = z.read(filename)  # bytes
                    for raw in content.splitlines():
                        # Skip empty lines and comments (35 = '#')
                        if not raw or raw.lstrip()[:1] == b'#':
                            continue
                        parts = raw.split()
                        if len(parts) >= 7:
                            # SWC: id type x y z radius parent
                            data['bodyId'].append(body_id)
                            data['node_id'].append(int(parts[0]))
                            data['type'].append(int(parts[1]))
                            data['x'].append(float(parts[2]))
                            data['y'].append(float(parts[3]))
                            data['z'].append(float(parts[4]))
                            data['radius'].append(float(parts[5]))
                            data['parent_id'].append(int(parts[6]))
                except Exception:
                    continue
    except Exception as e:
        print(f"Error in batch: {e}")
    return data

def _read_flywire_csv(read_path, string_columns=(), string_dtype='string'):
    """Read a (possibly gzipped) FlyWire CSV via Polars and return pandas.

    Polars decompresses gzip with multiple threads and parses CSV with
    vectorized kernels (~5-9x faster than pandas on the multi-GB FAFB
    tables), while the returned frame keeps the same column semantics.
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
        result[column] = result[column].astype(string_dtype)
    return result


def process_neurons_to_parquet(read_path, save_path, save_csv_path=None, enrichment_files=None):
    """
    Process classification.csv.gz and optional enrichment files into neuron_df parquet format.
    Enrichment files: names, coordinates, neurons (neurotransmitters), cell_stats.
    """
    if parquet_is_reusable(save_path):
        print(f"  ✓ Found existing converted file: {save_path}")
        return True

    print(f"  ⏳ Processing {read_path} -> {save_path}...")
    
    if not os.path.exists(read_path):
        print(f"  ⚠️ Error: Input file not found: {read_path}")
        return False

    try:
        # Read classifications (Base)
        print("  Reading classification data...")
        df = _read_flywire_csv(read_path, string_columns=('root_id',))
        
        # Rename columns to match coana expectations
        rename_map = {
            'root_id': 'bodyId',
            'class': 'cell_class',
            'sub_class': 'cell_type',
            'side': 'hemisphere'
        }
        df = df.rename(columns=rename_map)
        normalize_flywire_id_columns(df, ['bodyId'])
        
        # Deduplicate base if needed
        if df.duplicated(subset=['bodyId']).any():
            print("  Deduplicating base dataframe...")
            df = df.drop_duplicates(subset=['bodyId'])

        # --- Enrichment ---
        if enrichment_files:
            # 1. Names
            if enrichment_files.get('names'):
                fpath = enrichment_files['names']
                print(f"  Merging names from {os.path.basename(fpath)}...")
                df_names = _read_flywire_csv(fpath, string_columns=('root_id',), string_dtype=object)
                df_names = df_names.rename(columns={'root_id': 'bodyId', 'name': 'instance'})
                normalize_flywire_id_columns(df_names, ['bodyId'])
                if 'group' in df_names.columns: df_names = df_names.drop(columns=['group'])
                df = pd.merge(df, df_names, on='bodyId', how='left')

            # 2. Coordinates
            if enrichment_files.get('coordinates'):
                fpath = enrichment_files['coordinates']
                print(f"  Merging coordinates from {os.path.basename(fpath)}...")
                df_coords = _read_flywire_csv(fpath, string_columns=('root_id',), string_dtype=object)
                df_coords = df_coords.rename(columns={'root_id': 'bodyId'})
                normalize_flywire_id_columns(df_coords, ['bodyId'])
                if 'supervoxel_id' in df_coords.columns: df_coords = df_coords.drop(columns=['supervoxel_id'])
                df_coords = df_coords.drop_duplicates(subset=['bodyId'])
                df = pd.merge(df, df_coords, on='bodyId', how='left')

            # 3. Neurotransmitters (neurons.csv)
            if enrichment_files.get('neurons'):
                fpath = enrichment_files['neurons']
                print(f"  Merging neurotransmitters from {os.path.basename(fpath)}...")
                df_nt = _read_flywire_csv(fpath, string_columns=('root_id',), string_dtype=object)
                df_nt = df_nt.rename(columns={'root_id': 'bodyId'})
                normalize_flywire_id_columns(df_nt, ['bodyId'])
                if 'group' in df_nt.columns: df_nt = df_nt.drop(columns=['group'])
                df = pd.merge(df, df_nt, on='bodyId', how='left')

            # 4. Cell Stats
            if enrichment_files.get('cell_stats'):
                fpath = enrichment_files['cell_stats']
                print(f"  Merging cell stats from {os.path.basename(fpath)}...")
                df_stats = _read_flywire_csv(fpath, string_columns=('root_id',), string_dtype=object)
                df_stats = df_stats.rename(columns={'root_id': 'bodyId'})
                normalize_flywire_id_columns(df_stats, ['bodyId'])
                df = pd.merge(df, df_stats, on='bodyId', how='left')

            # 5. Cell Types (consolidated_cell_types.csv.gz)
            if enrichment_files.get('cell_types'):
                fpath = enrichment_files['cell_types']
                print(f"  Merging cell types from {os.path.basename(fpath)}...")
                df_types = _read_flywire_csv(fpath, string_columns=('root_id',), string_dtype=object)
                
                # Rename root_id -> bodyId, primary_type -> type
                rename_dict = {'root_id': 'bodyId'}
                if 'primary_type' in df_types.columns:
                    rename_dict['primary_type'] = 'type'
                
                df_types = df_types.rename(columns=rename_dict)
                normalize_flywire_id_columns(df_types, ['bodyId'])
                
                # Merge all columns (keep all original col names, use suffix for conflicts)
                df = pd.merge(df, df_types, on='bodyId', how='left', suffixes=(None, '_enriched'))

        # Logic for type: type (from consolidated) > cell_type > cell_class > super_class
        if 'type' not in df.columns: df['type'] = np.nan
        if 'cell_type' not in df.columns: df['cell_type'] = np.nan
        if 'cell_class' not in df.columns: df['cell_class'] = np.nan
        if 'super_class' not in df.columns: df['super_class'] = np.nan
        
        df['type'] = df['type'].fillna(df['cell_type']).fillna(df['cell_class']).fillna(df['super_class']).fillna('Unknown')
        
        # Instance: usually same as type or specific name. 
        if 'instance' not in df.columns:
            df['instance'] = df['type']
        else:
            df['instance'] = df['instance'].fillna(df['type'])
        
        df['post'] = 0 # Placeholder
        
        # Select columns (keep all enriched columns + standard ones)
        standard_cols = ['bodyId', 'type', 'instance', 'post', 'super_class', 'cell_class', 'cell_type', 'hemisphere', 'nucleus_id', 'hemilineage', 'nerve', 'flow']
        # Add any other columns found in enrichment (e.g. x, y, z, nt_type, etc.)
        all_cols = list(df.columns)
        # Prioritize standard columns order
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
        print(f"  ✓ Conversion complete. Output size: {file_size_mb:.2f} MB")
        return True
        
    except Exception as e:
        print(f"  ⚠️ Error processing neurons: {e}")
        return False

def process_connections_to_parquet(read_path, save_path):
    """
    Process connections.csv.gz into merged_connections parquet format.
    Logic matches coana.py _prepare_flywire_data.

    The read + aggregation run in Polars: multi-threaded gzip and vectorized
    groupby are ~9x / ~26x faster than the pandas pipeline on the multi-GB
    FAFB tables, with identical results.
    """
    if parquet_is_reusable(save_path):
        print(f"  ✓ Found existing converted file: {save_path}")
        return True

    print(f"  ⏳ Processing {read_path} -> {save_path}...")
    
    if not os.path.exists(read_path):
        print(f"  ⚠️ Error: Input file not found: {read_path}")
        return False

    try:
        import polars as pl

        # Read IDs as strings so the exact decimal digits survive.  Only
        # override columns that actually exist (polars would otherwise
        # create null placeholder columns for missing names).
        header = pl.read_csv(read_path, n_rows=0)
        overrides = {
            column: pl.Utf8
            for column in ('pre_root_id', 'post_root_id', 'bodyId_pre', 'bodyId_post')
            if column in header.columns
        }
        df = pl.read_csv(read_path, schema_overrides=overrides)
        
        # Rename for consistency (only columns that exist - polars raises
        # for missing names, where pandas silently ignored them)
        rename_map = {
            'pre_root_id': 'bodyId_pre',
            'post_root_id': 'bodyId_post',
            'syn_count': 'weight',
            'neuropil': 'roi'
        }
        df = df.rename({
            old: new for old, new in rename_map.items() if old in df.columns
        })
        
        # Canonical ID strings (strip whitespace, drop leading zeros, accept
        # integral '123.0' spellings) - same semantics as
        # normalize_flywire_id_columns, vectorized.
        def _canonical_ids(column: str):
            s = pl.col(column).cast(pl.Utf8).str.strip_chars()
            s = s.str.replace(r'^([0-9]+)\.0+$', '${1}')
            s = s.str.strip_chars_start('0')
            return pl.when(s.str.len_chars() == 0).then(pl.lit('0')).otherwise(s)

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

def process_synapse_table_to_parquet(read_path, save_path, chunksize=100000):
    """
    Process synapse table CSV to Parquet.
    Includes logic to fix short IDs (FlyWire specific).

    Polars reads and fixes the whole table in one vectorized pass instead
    of the chunked pandas loop (``chunksize`` is kept for compatibility);
    every row is checked for the 9-digit short form, where the legacy
    chunked code only inspected the first row of each chunk.

    The cleft-centre columns (``ctr_*``), the ``size`` score and the
    ``neuropil`` label are dropped: no DROCAT reader consumes them
    (``_read_flywire_connection_frame`` reads pre/post sites, root ids
    and a weight-threshold column only), and they were ~38% of the
    converted file.  Re-run the conversion from the CSV if they are ever
    needed again.

    The output goes to a temporary sibling and is atomically renamed, so
    an interrupted conversion never leaves a truncated parquet behind.
    """
    if parquet_is_reusable(save_path):
        print(f"  ✓ Found existing converted file: {save_path}")
        return True

    print(f"  ⏳ Processing {read_path} -> {save_path}...")
    
    if not os.path.exists(read_path):
        print(f"  ⚠️ Error: Input file not found: {read_path}")
        return False

    try:
        import polars as pl
        compression = 'gzip' if read_path.endswith('.gz') else None

        # Discover the dynamic root-ID columns before parsing data. Reading
        # them as strings at parser time is essential: an after-the-fact
        # ``astype(str)`` cannot repair a rounded float.
        header = pd.read_csv(read_path, compression=compression, nrows=0)
        pre_cols = [c for c in header.columns if c.startswith('pre_root_id')]
        post_cols = [c for c in header.columns if c.startswith('post_root_id')]
        if not pre_cols or not post_cols:
            print("  ⚠️ Error: Could not find root_id columns in CSV")
            return False
        pre_root_id_col = pre_cols[0]
        post_root_id_col = post_cols[0]

        df = pl.read_csv(
            read_path,
            schema_overrides={
                pre_root_id_col: pl.Utf8,
                post_root_id_col: pl.Utf8,
            },
        )
        if df.is_empty():
            print("  ⚠️ Error: No data found.")
            return False

        # Normalize IDs and fix short IDs (FlyWire specific issue) in one
        # vectorized pass: every 9-digit ID gets the 720575940 prefix, then
        # the exact-string canonicalization runs like the rest of DROCAT.
        def _fix_ids(column: str):
            return canonicalize_flywire_id_expr(column)

        for column in (pre_root_id_col, post_root_id_col):
            df = df.with_columns(_fix_ids(column).alias(column))

        # Drop columns with no consumer anywhere in DROCAT (audited
        # 2026-09: zero readers of ctr_*/size/neuropil in src, ui,
        # scripts and tests).
        unread = [c for c in ("ctr_x", "ctr_y", "ctr_z", "size", "neuropil")
                  if c in df.columns]
        if unread:
            df = df.drop(unread)

        print("  Sorting by root IDs...")
        df = df.sort([pre_root_id_col, post_root_id_col])

        print(f"  Saving to Parquet: {save_path}...")
        write_parquet_atomic(
            save_path,
            lambda temp: df.write_parquet(temp, compression='snappy'))

        # Born encoded: apply the lossless layout right away so no later
        # upgrade pass is ever needed for fresh conversions.
        reencode_parquet_lossless(save_path, FAFB_SYNAPSE_PROFILE)

        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"  ✓ Conversion complete. Output size: {file_size_mb:.2f} MB")
        return True
        
    except Exception as e:
        print(f"  ⚠️ Error processing synapse table: {e}")
        return False

def process_skeletons_to_parquet(zip_path, save_path, batch_size=500):
    """
    Process skeletons from a ZIP of SWC files to a single Parquet file.
    Optimized for space and speed using parallel processing and batched writing.
    """
    # Reclaim crashed-conversion temps BEFORE the reuse early-return: a
    # present final means a prior run finished, but its crashed temp (or a
    # pre-R7-2 naked '{save_path}.tmp') would otherwise sit forever — this
    # function never reaches the writer that owns the temp again.
    reclaim_writer_temps(save_path, 'fafb-table')

    if os.path.exists(save_path):
        print(f"  ✓ Found existing converted file: {save_path}")
        return True

    print(f"  ⏳ Processing {zip_path} -> {save_path}...")

    if not os.path.exists(zip_path):
        print(f"  ⚠️ Error: Input file not found: {zip_path}")
        return False

    # Use a temporary file to avoid corrupting the destination if
    # interrupted — through the shared temp-sibling scheme so an
    # interrupted write is reclaimable (round-7 R7-2), not a naked .tmp.
    temp_path = temp_sibling(save_path, 'fafb-table')
    writer = None
    
    try:
        # Get list of files first
        with zipfile.ZipFile(zip_path, 'r') as z:
            swc_files = [f for f in z.namelist() if f.endswith('.swc')]
            swc_files.sort()
            
        if not swc_files:
            print("  ⚠️ No SWC files found in zip.")
            return False
            
        print(f"  Found {len(swc_files)} skeletons. Starting parallel processing...")
        
        # Define schema
        schema = pa.schema([
            ('bodyId', pa.string()),
            ('node_id', pa.int32()),
            ('type', pa.int8()),
            ('x', pa.int32()),
            ('y', pa.int32()),
            ('z', pa.int32()),
            ('radius', pa.int32()),
            ('parent_id', pa.int32())
        ])
        
        # Prepare chunks
        chunks = [swc_files[i:i + batch_size] for i in range(0, len(swc_files), batch_size)]
        
        # Cap workers to avoid OOM on high-core machines with limited RAM per core
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        num_workers = min(num_workers, 8) 
        
        print(f"  Using {num_workers} workers for {len(chunks)} batches (batch_size={batch_size})...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all jobs
            futures = {executor.submit(_parse_swc_batch, zip_path, chunk): chunk for chunk in chunks}
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(chunks), desc="  Converting"):
                batch_nodes = future.result()
                
                # Check if batch has data (using column-oriented check)
                if batch_nodes and batch_nodes['bodyId']:
                    df_batch = pd.DataFrame(batch_nodes)
                    
                    # Optimize types
                    df_batch['bodyId'] = df_batch['bodyId'].astype(str)
                    df_batch['node_id'] = df_batch['node_id'].astype('int32')
                    df_batch['type'] = df_batch['type'].astype('int8')
                    df_batch['parent_id'] = df_batch['parent_id'].astype('int32')
                    
                    # Optimize coords
                    for col in ['x', 'y', 'z']:
                        df_batch[col] = df_batch[col].round().astype('int32')
                    # Radius follows the same rounding convention as coords
                    # (the legacy truncation made e.g. 1.5 -> 1).
                    df_batch['radius'] = df_batch['radius'].round().astype('int32')
                    
                    # Sort batch
                    df_batch = df_batch.sort_values(['bodyId', 'node_id'])
                    
                    # Convert to Table
                    table = pa.Table.from_pandas(df_batch, schema=schema)
                    
                    # Initialize writer if first batch
                    if writer is None:
                        writer = pq.ParquetWriter(temp_path, schema=schema, compression='snappy')
                    
                    writer.write_table(table)
                    
        if writer:
            writer.close()
            # Atomic replace: no remove-first window (an interruption after
            # the old os.remove lost the original table), and it works
            # where the rename needed the removal first.
            os.replace(temp_path, save_path)
            
            file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
            print(f"  ✓ Conversion complete. Output size: {file_size_mb:.2f} MB")
            return True
        else:
            print("  ⚠️ No valid skeleton data found.")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
        
    except Exception as e:
        print(f"  ⚠️ Error processing skeletons: {e}")
        if writer:
            writer.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def update_neuron_post_counts(neuron_path, conn_path, save_csv_path=None):
    """
    Update the 'post' column in the neuron DataFrame by summing weights from the connections DataFrame.
    This is necessary because FlyWire classification files don't include post-synaptic counts.
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
            df_neuron.to_parquet(neuron_path, index=False, compression='snappy')
        else:
            df_neuron.to_csv(neuron_path, index=False)
            
        if save_csv_path:
            print(f"  Saving updated neurons to {save_csv_path}...")
            df_neuron.to_csv(save_csv_path, index=False)
            
        print("  ✓ Post counts updated.")
        return True
    except Exception as e:
        print(f"  ⚠️ Error updating post counts: {e}")
        return False

def print_download_instructions(downloads_dir, dataset_name="flywire_FAFB_v783"):
    # The dataset name is passed through so the printed folder and dataset
    # identifier always match the dataset being prepared.
    if _print_flywire_download_instructions is not None:
        _print_flywire_download_instructions(dataset_name, Path(downloads_dir).parent)
    else:
        print(f"\033[31mError: Missing converted files or source files.\033[0m")
        print(f"\033[33mPlease download the following files from: \033[34mhttps://codex.flywire.ai/api/download?dataset=fafb\033[0m")
        print(f"\033[33mAnd save them to: \033[34m{downloads_dir}\033[0m")
    
    print("\nFile status:")
    files_info = [
        ("classification.csv.gz", "Neuron Classification"),
        ("names.csv.gz", "Optional neuron metadata: names"),
        ("coordinates.csv.gz", "Optional neuron metadata: soma coordinates"),
        ("neurons.csv.gz", "Optional neuron metadata: neurotransmitters"),
        ("cell_stats.csv.gz", "Optional neuron metadata: cell statistics"),
        ("consolidated_cell_types.csv.gz", "Optional neuron metadata: consolidated cell types"),
        ("connections_princeton_no_threshold.csv.gz", "Connectivity Data"),
        ("fafb_v783_princeton_synapse_table.csv.gz", "Optional synapse coordinates for visualization"),
        ("sk_lod1_783_healed.zip", "Optional skeletons for visualization")
    ]

    for fname, desc in files_info:
        fpath = os.path.join(downloads_dir, fname)
        exists = os.path.exists(fpath)
        
        # Check uncompressed version if .gz
        if not exists and fname.endswith('.gz'):
            fpath_uncompressed = os.path.join(downloads_dir, fname[:-3])
            if os.path.exists(fpath_uncompressed):
                exists = True
        
        # Check fallbacks for connections
        if not exists and fname == "connections_princeton_no_threshold.csv.gz":
            if os.path.exists(os.path.join(downloads_dir, "connections_princeton.csv.gz")) or \
               os.path.exists(os.path.join(downloads_dir, "connections.csv.gz")):
                exists = True

        is_optional = desc.startswith("Optional") or "Visualization" in desc
        fname_display = f"[optional] {fname}" if is_optional else fname

        if exists:
            print(f"  ✅ [existed] {fname_display} ({desc})")
        else:
            if is_optional:
                print(f"  - {fname_display} ({desc})")
            else:
                print(f"  ❌ {fname_display} ({desc})")

def ensure_flywire_data(dataset_name, dataset_dir):
    """
    Ensure FlyWire data is available and converted for the given dataset.
    Checks each component (neurons, connections, synapses, skeletons) independently.
    """
    print(f"\nChecking FlyWire data for {dataset_name}...")
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
    syn_pq = os.path.join(dataset_dir, f"{dataset_name}_synapse_table.parquet")
    sk_zip_dst = os.path.join(dataset_dir, "sk_lod1_783_healed.zip")

    all_critical_present = True

    # --- 1. Neurons ---
    if parquet_is_reusable(neuron_pq):
        print(f"  ✓ Found existing neurons: {os.path.basename(neuron_pq)}")
    else:
        print("  Checking neuron source files...")
        class_raw = os.path.join(downloads_dir, "classification.csv.gz")
        
        if not os.path.exists(class_raw):
            print(f"  ❌ Missing required file: classification.csv.gz")
            all_critical_present = False
        else:
            # Check enrichment files
            required_enrichment = {
                'names': os.path.join(downloads_dir, "names.csv.gz"),
                'coordinates': os.path.join(downloads_dir, "coordinates.csv.gz"),
                'neurons': os.path.join(downloads_dir, "neurons.csv.gz"),
                'cell_stats': os.path.join(downloads_dir, "cell_stats.csv.gz"),
                'cell_types': os.path.join(downloads_dir, "consolidated_cell_types.csv.gz")
            }
            
            enrichment_files = {}
            missing_enrichment = []

            for key, path in required_enrichment.items():
                if os.path.exists(path):
                    enrichment_files[key] = path
                else:
                    uncompressed = path.replace('.gz', '')
                    if os.path.exists(uncompressed):
                        enrichment_files[key] = uncompressed
                    else:
                        missing_enrichment.append(os.path.basename(path))
            
            if missing_enrichment:
                print(f"  ⚠️ Missing neuron metadata files: {', '.join(missing_enrichment)}")
                print("  (Will proceed without them, but neuron info will be incomplete)")
            
            if not process_neurons_to_parquet(class_raw, neuron_pq, save_csv_path=neuron_csv, enrichment_files=enrichment_files):
                all_critical_present = False

    # --- 2. Connections ---
    if parquet_is_reusable(conn_pq):
        print(f"  ✓ Found existing connections: {os.path.basename(conn_pq)}")
    else:
        print("  Checking connection source files...")
        conn_raw = os.path.join(downloads_dir, "connections_princeton_no_threshold.csv.gz")
        if not os.path.exists(conn_raw):
            # Fallbacks
            conn_raw = os.path.join(downloads_dir, "connections_princeton.csv.gz")
            if not os.path.exists(conn_raw):
                conn_raw = os.path.join(downloads_dir, "connections.csv.gz")

        if os.path.exists(conn_raw):
            if not process_connections_to_parquet(conn_raw, conn_pq):
                all_critical_present = False
        else:
            print(f"  ❌ Missing required file: connections_princeton_no_threshold.csv.gz")
            all_critical_present = False

    # --- 3. Synapses (Optional) ---
    if os.path.exists(syn_pq) and parquet_readable(syn_pq):
        print(f"  ✓ Found existing synapse table: {os.path.basename(syn_pq)}")
        # Slim tables converted by older DROCAT versions in place; a
        # no-op for the current column set.
        compact_synapse_table(syn_pq)
    else:
        if os.path.exists(syn_pq):
            # A truncated file has no valid parquet footer: the previous
            # conversion run was interrupted mid-write.  Discard it and
            # re-convert from the raw CSV below.
            print(f"  ⚠️ Existing synapse table is incomplete (the previous "
                  "run was interrupted); re-converting it from the raw "
                  "download...")
            try:
                os.remove(syn_pq)
            except OSError:
                pass
        syn_raw = os.path.join(downloads_dir, "fafb_v783_princeton_synapse_table.csv.gz")
        if not os.path.exists(syn_raw):
            for f in os.listdir(downloads_dir):
                if 'synapse' in f and f.endswith('.csv.gz'):
                    syn_raw = os.path.join(downloads_dir, f)
                    break

        if syn_raw and os.path.exists(syn_raw):
            process_synapse_table_to_parquet(syn_raw, syn_pq)
        else:
            print("  ℹ️  Synapse table source not found (optional).")

    # --- 4. Skeletons (Optional) ---
    if os.path.exists(sk_zip_dst):
        print(f"  ✓ Found existing skeletons: {os.path.basename(sk_zip_dst)}")
    else:
        sk_raw = os.path.join(downloads_dir, "sk_lod1_783_healed.zip")
        if not os.path.exists(sk_raw):
            for f in os.listdir(downloads_dir):
                if f.endswith('.zip') and ('sk' in f or 'skeleton' in f):
                    sk_raw = os.path.join(downloads_dir, f)
                    break

        if sk_raw and os.path.exists(sk_raw):
            print(f"  Moving {os.path.basename(sk_raw)} -> {os.path.basename(sk_zip_dst)}...")
            shutil.move(sk_raw, sk_zip_dst)
        else:
            print("  ℹ️  Skeletons zip not found (optional).")

    # --- Post Counts Update ---
    if parquet_is_reusable(neuron_pq) and parquet_is_reusable(conn_pq):
        try:
            # Read just the post column to check if it's all zeros
            df_check = pd.read_parquet(neuron_pq, columns=['post'])
            if df_check['post'].sum() == 0:
                print("  ℹ️  Post counts are 0. Updating from connections...")
                update_neuron_post_counts(neuron_pq, conn_pq, save_csv_path=neuron_csv)
            else:
                print("  ✓ Post counts already populated.")
        except Exception as e:
            print(f"  ⚠️ Could not check post counts: {e}")

    if not all_critical_present:
        print("\n" + "="*60)
        print("MISSING CRITICAL FILES")
        print("="*60)
        print_download_instructions(downloads_dir, dataset_name)
        return False

    return True

if __name__ == "__main__":
    # Standard run for installation
    dataset_name = "flywire_FAFB_v783"
    # Determine project root (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, "datasets", dataset_name)
    
    print(f"Running FAFB data preparation for {dataset_name}...")
    print(f"Target directory: {dataset_dir}")
    
    ensure_flywire_data(dataset_name, dataset_dir)
