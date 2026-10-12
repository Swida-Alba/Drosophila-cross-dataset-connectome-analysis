"""
ThresholdedConnectionMap: one synapse cutoff of a connection cache as an
independent dataset.

The connection cache stores the COMPLETE graph (every edge with weight >= 1).
A query with ``min_synapse_num = t`` works on the derived graph

    D_t = { edges with weight >= t }

which is treated as an independent dataset: its totals (total incoming per
post neuron / per post type) are computed from D_t alone and cached per
cutoff, so ratios at different thresholds are never mixed.

The map owns both aggregate tables (bodyId-level and type-level) plus the
neuron-index lookup, so numerator and denominator always come from the same
D_t. Aggregates are computed lazily with vectorized Polars and cached for
the lifetime of the map (one map per cutoff; stale maps are discarded when
the underlying cache files change).
"""

import threading
from typing import Optional, Union

import pandas as pd
import polars as pl


class ThresholdedConnectionMap:
    """The connection graph at synapse cutoff ``t``, with its own aggregates."""

    def __init__(
        self,
        db_path: str,
        neuron_index_path: str,
        min_weight: int = 1,
        conn_frame: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
        source_signature: tuple = None,
    ):
        self._db_path = db_path
        self._index_path = neuron_index_path
        self._min_weight = int(min_weight)
        self._conn_frame = conn_frame
        self.source_signature = source_signature
        self._lock = threading.Lock()
        self._cache: dict = {}

    @property
    def min_weight(self) -> int:
        return self._min_weight

    def _load_neuron_index(self) -> pl.DataFrame:
        """bodyId -> type lookup from the cache's neuron index.

        Preserves the legacy dict semantics: the LAST entry wins for
        duplicate bodyIds; untyped neurons are excluded (they are grouped by
        their bodyId at type level and served by the bodyId-level table).
        """
        index = pl.read_parquet(self._index_path, columns=["bodyId", "type"])
        # Connection caches are normalized to string body IDs, but older
        # parquet files may have stored them as integers.  Normalize the
        # lookup side as well so type-level totals work for both schemas.
        index = index.with_columns(
            pl.col("bodyId").cast(pl.Utf8, strict=False).alias("bodyId")
        )
        index = index.unique(subset=["bodyId"], keep="last")
        if "type" in index.columns:
            index = index.filter(
                pl.col("type").is_not_null() & (pl.col("type") != "")
            )
        return index

    def _thresholded_source(self) -> pl.LazyFrame:
        """The D_t edge source (lazy scan for disk, eager frame in memory)."""
        if self._conn_frame is not None:
            conn = self._conn_frame
            if isinstance(conn, pd.DataFrame):
                conn = pl.from_pandas(conn)
            lazy = conn.lazy()
        else:
            # A cancelled/resumable cache build can leave only batch parquet
            # files until consolidation runs.  Read those files too, and
            # tolerate the optional-column differences found in older cache
            # generations.
            import os
            from pathlib import Path

            paths = []
            if self._db_path and os.path.exists(self._db_path):
                paths.append(self._db_path)
            # LAYOUT-AWARE batch scoping (2026-10-12, plan §12.3): a
            # pair-collapsed main cache (marked by the connections_by_roi
            # sidecar) already CONTAINS every consolidated batch's rows
            # (summed) — batches coexisting with it are stale
            # consolidation leftovers whose per-ROI slices would
            # double-count after the per-pair sum, so they are skipped.
            # Legacy per-ROI caches keep the exact-row crash-window
            # dedup below (batches re-list the main rows verbatim there).
            # Batches WITHOUT a main file (a fetch interrupted before the
            # first consolidation) are per-fetch disjoint — they are the
            # only data and are always included.
            pair_level = (
                self._db_path
                and (Path(self._db_path).parent
                     / "connections_by_roi.parquet").exists()
            )
            include_batches = not (paths and pair_level)
            if self._db_path and include_batches:
                batch_dir = Path(self._db_path).parent / "_batch_files"
                paths.extend(str(path) for path in sorted(batch_dir.glob("batch_*.parquet")))
            if not paths:
                return pl.LazyFrame({
                    "bodyId_pre": pl.Series([], dtype=pl.Utf8),
                    "bodyId_post": pl.Series([], dtype=pl.Utf8),
                    "weight": pl.Series([], dtype=pl.Int64),
                })
            # Scan each file separately.  Passing a list of parquet files to
            # scan_parquet makes Polars choose one physical schema for the
            # whole scan before the casts below run; a legacy numeric batch
            # followed by a string-ID batch can therefore still fail during
            # the union.  Selecting/casting each file first keeps mixed cache
            # generations readable.
            required = {"bodyId_pre", "bodyId_post", "weight"}
            lazy_frames = []
            for path in paths:
                frame = pl.scan_parquet(path)
                names = set(frame.collect_schema().names())
                missing = required - names
                if missing:
                    continue
                select_exprs = [
                    pl.col("bodyId_pre").cast(pl.Utf8, strict=False).alias("bodyId_pre"),
                    pl.col("bodyId_post").cast(pl.Utf8, strict=False).alias("bodyId_post"),
                    pl.col("weight").cast(pl.Int64, strict=False).alias("weight"),
                    # roi rides along ONLY for the crash-window dedup key
                    # (dropped right after) — see the guard below.
                    pl.col("roi").cast(pl.Utf8, strict=False).alias("_roi")
                    if "roi" in names else pl.lit(None, dtype=pl.Utf8).alias("_roi"),
                ]
                lazy_frames.append(frame.select(select_exprs))
            if not lazy_frames:
                return pl.LazyFrame({
                    "bodyId_pre": pl.Series([], dtype=pl.Utf8),
                    "bodyId_post": pl.Series([], dtype=pl.Utf8),
                    "weight": pl.Series([], dtype=pl.Int64),
                })
            lazy = pl.concat(lazy_frames, how="vertical_relaxed")
            # Crash-window guard, SCOPED to the crash window (2026-10-12
            # fix of an f7df023 regression): consolidation writes the
            # main parquet BEFORE removing _batch_files — an interrupted
            # rmtree leaves the batches re-listing the main file's rows,
            # which would double every D_t denominator. That matters
            # ONLY when main and batches coexist; and the collapse must
            # be EXACT-ROW (the batches re-list verbatim), not
            # (pre, post, roi) — a keep-one-row dedup silently dropped
            # legitimate same-pair rows in roi-less caches (the
            # T_DUP/T0 under-counts in test_performance_fixes). A
            # single consolidated file is one row per (pre, post, roi)
            # by construction, so plain summation is already exact.
            if (not pair_level and len(paths) > 1
                    and os.path.exists(self._db_path)):
                lazy = lazy.unique(
                    subset=["bodyId_pre", "bodyId_post", "_roi", "weight"],
                    keep="any")
            lazy = lazy.drop("_roi")

        # Normalize IDs at the read boundary.  This covers both the on-disk
        # cache and callers that provide an in-memory pandas frame.
        lazy = lazy.with_columns([
            pl.col("bodyId_pre").cast(pl.Utf8, strict=False).alias("bodyId_pre"),
            pl.col("bodyId_post").cast(pl.Utf8, strict=False).alias("bodyId_post"),
            pl.col("weight").cast(pl.Int64, strict=False).alias("weight"),
        ])
        # PER-PAIR COLLAPSE BEFORE THE TIER FILTER (2026-10-12, plan §12 —
        # the same doctrine as the refill's 2026-10-10 collapse): the
        # cache stores per-ROI slices, so the PHYSICAL pair weight is the
        # sum of its slices and D_t's tier filter must see that sum. The
        # previous row-level filter dropped sub-threshold SLICES of
        # supra-threshold pairs (a 2+5 pair at t=3 counted 5) and made a
        # pair whose every slice is sub-threshold vanish (1+2 at t=3)
        # — on hemibrain-scale caches (9% of pairs multi-slice) that
        # under-counted every ratio denominator. Sum per pair first,
        # then apply the cutoff to the pair's physical weight.
        lazy = (
            lazy.group_by(["bodyId_pre", "bodyId_post"])
            .agg(pl.col("weight").sum().alias("weight"))
        )
        if self._min_weight > 1:
            lazy = lazy.filter(pl.col("weight") >= self._min_weight)
        return lazy

    def total_incoming_by_bodyid(self) -> pl.DataFrame:
        """Full-D_t aggregate ``bodyId_post -> total_incoming_weight``."""
        cached = self._cache.get("by_bodyid")
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get("by_bodyid")
            if cached is not None:
                return cached
            result = (
                self._thresholded_source()
                .group_by("bodyId_post")
                .agg(pl.col("weight").sum().alias("total_incoming_weight"))
                .collect()
            )
            self._cache["by_bodyid"] = result
            return result

    def total_incoming_by_type(self) -> pl.DataFrame:
        """Full-D_t aggregate ``type_post -> total_incoming_weight``."""
        cached = self._cache.get("by_type")
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get("by_type")
            if cached is not None:
                return cached
            result = (
                self._thresholded_source()
                .join(
                    self._load_neuron_index().lazy(),
                    left_on="bodyId_post",
                    right_on="bodyId",
                    how="inner",
                )
                .group_by("type")
                .agg(pl.col("weight").sum().alias("total_incoming_weight"))
                .rename({"type": "type_post"})
                .collect()
            )
            self._cache["by_type"] = result
            return result
