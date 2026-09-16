"""NeuronBridge Find Lines with cross-dataset name expansion.

A thin orchestration over :class:`neuronbridge_finder.NeuronBridgeFinder`
(plan `_plan/plan-nb-find-lines-dataset-aware-queries.md` §7).  The finder
is never modified: names resolve ONLY against local dataset tables and
bodyIds are validated against NeuronBridge by strict dataset identity, so
the orchestration's whole job is to hand the finder the RIGHT names at the
RIGHT releases:

1. A coverage snapshot (:mod:`neuronbridge_coverage`) decides which of the
   selected datasets NeuronBridge can serve, and at which hosted release
   (``male-cns:v1.0`` → ``male-cns:v0.9``; BANC/optic-lobe → advisory
   warnings, never blocks).
2. Each name chip expands through the cross-dataset type mapper's shared
   policy API (``type_resolver.resolve_valid_targets`` →
   ``expansion_targets``) into every covered dataset's namespace — the
   "combination input": one name in, every dataset-local equivalent out.
   The male-cns v1.0 → v0.9 release alignment needs no extra code; it IS
   the mapper's curated shared-name alias, reached through the same call.
3. Each chip runs one ``find_lines_batch`` call over its deduplicated
   expanded names at the covered releases, keeping per-chip coverage
   weighting and giving every result row its original chip as
   ``source_query``.

All output lands in ONE per-run folder (``NB-find-lines-expanded_*``
under the requested output dir, announced via the standard folder marker
so the results panel links to the whole run).  Single-chip runs are flat;
multi-chip runs group one ``chip_{query}/`` folder per query:

* per-chip files (the finder's own ``line_summary.csv``, split summaries,
  and per-query CSVs — ``parameters.json`` in each records the hosted
  releases),
* ``expansion_map.csv`` — chip → expanded name → hosted release → mapping
  status/kind,
* ``expansion_summary.json`` — the original selection, the coverage
  routing that shaped the run, and the output-detail cleanup audit,
* ``user_warning_notes.txt`` — advisory notes for unavailable datasets
  (``coverage:`` prefix), the same warning-file convention as the other
  NeuronBridge tools.

The **output-detail flags** prune the run folder after all chips complete
(Compact = ``keep_per_match_csv=False, cleanup_source_images=True``):
bodyId-level ``*_lines.csv`` tables go once their ``line_summary.csv``
exists, and ``images/`` goes once the integrated PDF/PPTX exists — every
removal audited in ``cleanup_audit.json``.  Under the default-off match
cache, a pruned match table regenerates only by re-running the query.

Degraded modes stay honest: an unavailable mapper, a failed coverage
refresh, or a conflict-only chip fall back to the ordinary query path (or
to no targets for that dataset) with a printed reason — never a silent
wrong answer.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:  # ``src/`` on sys.path (runner scripts) vs package imports
    from src.neuronbridge_coverage import (
        CoverageSnapshot,
        load_snapshot,
        refresh as refresh_coverage,
        warnings_for,
    )
    from src.neuronbridge_output_policy import (
        enforce_match_table_retention,
        prune_find_lines_run,
    )
except ImportError:  # pragma: no cover - bare ``src`` layout
    from neuronbridge_coverage import (
        CoverageSnapshot,
        load_snapshot,
        refresh as refresh_coverage,
        warnings_for,
    )
    from neuronbridge_output_policy import (
        enforce_match_table_retention,
        prune_find_lines_run,
    )

try:
    from comparison.cross_dataset_type_mapper import get_type_mapper
    from comparison.type_resolver import (
        MapperSnapshot,
        expansion_targets,
        resolve_valid_targets,
    )
    HAS_TYPE_MAPPER = True
except ImportError:  # pragma: no cover - ``src`` layout
    try:
        from src.comparison.cross_dataset_type_mapper import get_type_mapper
        from src.comparison.type_resolver import (
            MapperSnapshot,
            expansion_targets,
            resolve_valid_targets,
        )
        HAS_TYPE_MAPPER = True
    except ImportError:
        HAS_TYPE_MAPPER = False


EXPANSION_MAP_FILENAME = "expansion_map.csv"
EXPANSION_SUMMARY_FILENAME = "expansion_summary.json"
WARNINGS_FILENAME = "user_warning_notes.txt"


def _build_finder(verbose: bool, separate_splitgal4: bool, region: str,
                  max_workers: int, use_cache: bool = False):
    """Construct the inner Finder (seam for tests)."""
    try:
        from neuronbridge_finder import NeuronBridgeFinder
    except ImportError:  # pragma: no cover - ``src`` layout
        from src.neuronbridge_finder import NeuronBridgeFinder
    return NeuronBridgeFinder(
        verbose=verbose,
        separate_splitgal4=separate_splitgal4,
        region=region,
        max_workers=max_workers,
        use_cache=use_cache,
    )


def _expand_targets(
    mapper: Any,
    snapshot: Any,
    chip: str,
    hosted_dataset: str,
) -> Tuple[Tuple[str, str, str], ...]:
    """`(name, status, kind)` targets one chip is licensed to query at one
    hosted release.  Policy is the Type Mapping panel's: splits expand to
    all branch targets, conflicts/evidence-only expand to nothing, unmapped
    keeps the raw name as the counted long-tail fallback."""
    resolution = resolve_valid_targets(
        mapper, chip, None, hosted_dataset, snapshot=snapshot)
    status = str(getattr(resolution, "status", "unmapped"))
    kind = str(getattr(resolution, "kind", "") or "")
    targets = expansion_targets(resolution) if resolution is not None else ()
    return tuple(
        (str(target), status, kind) for target in targets
    )


def expand_chip(
    chip: str,
    covered: Dict[str, str],
    mapper: Any = None,
    mapper_snapshot: Any = None,
) -> List[Dict[str, str]]:
    """Expand one name chip into `(expanded_name, nb_dataset)` pairs.

    ``covered`` maps a selected dataset to the hosted release to query
    (``CoverageSnapshot.covered_datasets``).  When the type mapper is not
    loaded the chip passes through unexpanded — the ordinary query path.
    """
    if mapper is None:
        return [
            {"expanded_name": chip, "nb_dataset": hosted,
             "mapping_status": "mapper unavailable", "mapping_kind": ""}
            for hosted in dict.fromkeys(covered.values())
        ]

    snap = mapper_snapshot
    if snap is None and HAS_TYPE_MAPPER:
        snap = MapperSnapshot(mapper)
    if snap is not None and not getattr(snap, "loaded", True):
        return [
            {"expanded_name": chip, "nb_dataset": hosted,
             "mapping_status": "mapper unavailable", "mapping_kind": ""}
            for hosted in dict.fromkeys(covered.values())
        ]

    expanded: List[Dict[str, str]] = []
    seen: set = set()
    for hosted in dict.fromkeys(covered.values()):
        try:
            targets = _expand_targets(mapper, snap, chip, hosted)
        except Exception:
            targets = ((chip, "unmapped", ""),)
        if not targets:
            # Conflict/evidence-only at this release: nothing is licensed.
            expanded.append({
                "expanded_name": "", "nb_dataset": hosted,
                "mapping_status": "conflict", "mapping_kind": "no licensed target",
            })
            continue
        for target, status, kind in targets:
            key = (target, hosted)
            if key in seen:
                continue
            seen.add(key)
            expanded.append({
                "expanded_name": target,
                "nb_dataset": hosted,
                "mapping_status": status,
                "mapping_kind": kind,
            })
    return expanded


class ExpandedLineFinder:
    """Run NeuronBridge Find Lines with coverage routing + name expansion."""

    def __init__(
        self,
        verbose: bool = True,
        separate_splitgal4: bool = True,
        region: str = "Brain",
        max_workers: int = 8,
        use_cache: bool = False,
    ) -> None:
        self._finder = _build_finder(
            verbose=verbose,
            separate_splitgal4=separate_splitgal4,
            region=region,
            max_workers=max_workers,
            use_cache=use_cache,
        )
        self._verbose = bool(verbose)
        # The orchestration owns the run protocol: the inner per-chip
        # find_lines_batch calls each restart their own 3/4-step protocol,
        # which would make the results panel's determinate bar bounce, and
        # each announces its own chip subfolder (the results panel would
        # link to the last chip only).  Both instance-level events are
        # therefore filtered and replaced by the orchestration's own.
        self._finder._progress = lambda *args, **kwargs: None
        inner_vprint = self._finder._vprint

        def _filtered_vprint(message: Any, *args: Any, **kwargs: Any) -> None:
            if isinstance(message, str) and "Output folder:" in message:
                return
            inner_vprint(message, *args, **kwargs)

        self._finder._vprint = _filtered_vprint

    def _progress(self, step: int, total: int, label: str = "") -> None:
        """Orchestration-level step events for the results panel."""
        if self._verbose:
            print(
                f"[DROCAT][progress] {int(step)}/{int(total)} {label}".rstrip(),
                flush=True,
            )

    def _vprint(self, message: str) -> None:
        if self._verbose:
            print(message, flush=True)

    # ------------------------------------------------------------------
    # coverage + expansion plumbing
    # ------------------------------------------------------------------

    def _coverage_snapshot(
        self,
        coverage_datasets: Optional[Sequence[str]],
        coverage_refresh: bool,
        cache_root: Optional[os.PathLike] = None,
    ) -> Optional[CoverageSnapshot]:
        if not coverage_datasets:
            return load_snapshot(cache_root)
        if not coverage_refresh:
            return load_snapshot(cache_root)
        try:
            return refresh_coverage(coverage_datasets, cache_root=cache_root)
        except Exception as exc:
            self._vprint(f"   ⚠️ Coverage refresh failed ({exc}); "
                         "using the persisted snapshot if any.")
            return load_snapshot(cache_root)

    def _covered(
        self,
        scope: List[str],
        snapshot: Optional[CoverageSnapshot],
    ) -> Tuple[Dict[str, str], List[str]]:
        """Covered `{selected: hosted}` mapping + advisory warning strings.

        Warn-don't-block contract: an unavailable snapshot or a failed
        refresh degrades to the ordinary (unrouted) query path, never to a
        refusal or a fabricated verdict.
        """
        if snapshot is None:
            return {}, []
        unavailable = snapshot.unavailable_datasets(scope)
        warnings = warnings_for(scope, snapshot)
        covered = snapshot.covered_datasets(scope)
        if not covered:
            # Nothing verified (offline refresh, all-unknown): run the
            # ordinary path over the user's own selection instead.
            return {}, warnings
        return covered, warnings

    # ------------------------------------------------------------------
    # public entry point (runner: finder.run(**method_params))
    # ------------------------------------------------------------------

    def run(
        self,
        queries: Any,
        dataset: Optional[Any] = None,
        expand_names: bool = True,
        coverage_datasets: Optional[Sequence[str]] = None,
        coverage_refresh: bool = True,
        match_type: Optional[str] = None,
        sort_by: str = "max",
        output_dir: Optional[str] = None,
        download_images: Optional[str] = None,
        download_img_for_top_n_lines: Optional[int] = 30,
        image_formats: Any = None,
        image_types: Any = "all",
        max_download_images_per_line: Optional[int] = 12,
        flylight_category: Optional[Any] = None,
        organize_by_region: bool = False,
        simple_mode: bool = False,
        pdf_images_per_page: Tuple[int, int] = (3, 2),
        pdf_landscape: bool = True,
        summary_format: Any = "pdf",
        summary_background_color: Any = "black",
        keep_per_match_csv: bool = True,
        cleanup_source_images: bool = False,
        compact_keep_last_n: int = 1,
        cache_root: Optional[os.PathLike] = None,
        _coverage_snapshot: Optional[CoverageSnapshot] = None,
    ) -> pd.DataFrame:
        """Expanded Find Lines; parameters match ``find_lines_batch`` with
        the addition of ``expand_names`` / ``coverage_*`` and the
        output-detail flags (Full default; Compact prunes bodyId-level
        match tables and downloaded images after summarization — audited
        in ``cleanup_audit.json``).  ``compact_keep_last_n`` (default 1)
        keeps the newest N expanded runs' match tables instead of deleting
        the current run's: older Compact runs are swept, Full/unknown-mode
        runs are immune.  Passing an explicit ``_coverage_snapshot``
        (tests) skips the network refresh.
        """
        if isinstance(queries, str):
            chips: List[Any] = [
                q.strip() for q in queries.split(",") if q.strip()]
        elif isinstance(queries, int):
            chips = [queries]
        else:
            try:
                chips = list(queries)
            except TypeError:
                chips = [queries]
        if not chips:
            self._vprint("❌ No queries provided")
            return pd.DataFrame()

        batch_params = dict(
            match_type=match_type,
            sort_by=sort_by,
            download_images=download_images,
            download_img_for_top_n_lines=download_img_for_top_n_lines,
            image_formats=image_formats,
            image_types=image_types,
            max_download_images_per_line=max_download_images_per_line,
            flylight_category=flylight_category,
            organize_by_region=organize_by_region,
            simple_mode=simple_mode,
            pdf_images_per_page=pdf_images_per_page,
            pdf_landscape=pdf_landscape,
            summary_format=summary_format,
            summary_background_color=summary_background_color,
        )

        if not expand_names:
            return self._finder.find_lines_batch(
                queries=queries, dataset=dataset, output_dir=output_dir,
                keep_per_match_csv=keep_per_match_csv,
                cleanup_source_images=cleanup_source_images,
                **batch_params)

        scope = [str(value) for value in (
            dataset if isinstance(dataset, (list, tuple, set))
            else ([dataset] if dataset else (coverage_datasets or []))
        ) if str(value)]
        self._progress(1, 3, "Resolve coverage and expand queries")
        snapshot = (
            _coverage_snapshot
            if _coverage_snapshot is not None
            else self._coverage_snapshot(
                coverage_datasets or scope, coverage_refresh, cache_root)
        )
        covered, warnings = self._covered(scope, snapshot)

        mapper = None
        mapper_snapshot = None
        if HAS_TYPE_MAPPER:
            try:
                mapper = get_type_mapper()
            except Exception as exc:
                self._vprint(f"   ⚠️ Type mapper failed to load ({exc}); "
                             "querying names unexpanded.")
                mapper = None
            if mapper is not None:
                mapper_snapshot = MapperSnapshot(mapper)
                if not mapper_snapshot.loaded:
                    self._vprint("   ⚠️ Type mapper not loaded; "
                                 "querying names unexpanded.")
                    mapper = None

        expansion_ready = bool(covered) and mapper is not None
        if not expansion_ready:
            # Ordinary path, unchanged behavior.
            for warning in warnings:
                self._vprint(f"   ⚠️ {warning}")
            if output_dir:
                self._write_warnings(output_dir, warnings)
            return self._finder.find_lines_batch(
                queries=chips, dataset=dataset, output_dir=output_dir,
                keep_per_match_csv=keep_per_match_csv,
                cleanup_source_images=cleanup_source_images,
                **batch_params)

        for warning in warnings:
            self._vprint(f"   ⚠️ {warning}")
        hosted_list = list(dict.fromkeys(covered.values()))
        self._vprint(
            f"🔀 Expanding {len(chips)} query chip(s) across "
            f"{len(hosted_list)} NeuronBridge-hosted release(s): "
            f"{', '.join(hosted_list)}"
        )

        # One coherent per-run folder: every chip's result folder and the
        # orchestration reports live inside it, and the announced marker
        # lets the results panel link to the WHOLE run instead of the last
        # chip's folder.
        run_root: Optional[str] = None
        if output_dir:
            run_root = self._make_run_root(output_dir, chips, hosted_list)
            self._vprint(f"📁 Output folder: {run_root}")

        name_chips: List[str] = []
        body_id_chips: List[Any] = []
        for chip in chips:
            try:
                int(chip)
                body_id_chips.append(chip)
            except (ValueError, TypeError):
                name_chips.append(str(chip))

        expansion_rows: List[Dict[str, str]] = []
        all_frames: List[pd.DataFrame] = []
        chip_dirs: List[str] = []
        self._progress(2, 3, "Run Find Lines per query chip")

        def _chip_dir_for(chip: Any) -> Optional[str]:
            if not run_root:
                return None
            sanitized = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in str(chip)
            ) or "chip"
            chip_dir = os.path.join(run_root, f"chip_{sanitized}")
            os.makedirs(chip_dir, exist_ok=True)
            chip_dirs.append(chip_dir)
            return chip_dir

        for chip in name_chips:
            expanded = expand_chip(chip, covered, mapper, mapper_snapshot)
            for row in expanded:
                expansion_rows.append({"source_query": chip, **row})
            # The same expanded name at several hosted releases resolves
            # against every dataset of the shared query list, so the chip's
            # call needs each distinct name once.
            query_names = list(dict.fromkeys(
                row["expanded_name"] for row in expanded
                if row["expanded_name"]
            ))
            if not query_names:
                self._vprint(
                    f"   ⛔ {chip}: no NeuronBridge-hosted target "
                    "(mapping conflict, or no hosted release carries it).")
                continue
            self._vprint(
                f"   ➡️ {chip}: {len(query_names)} expanded name(s) "
                f"across {len(set(r['nb_dataset'] for r in expanded if r['expanded_name']))} release(s)"
            )
            chip_dir = _chip_dir_for(chip)
            frame = self._finder.find_lines_batch(
                queries=query_names, dataset=hosted_list,
                output_dir=chip_dir, **batch_params)
            if chip_dir:
                self._hoist_finder_folder(chip_dir)
            if frame is not None and not frame.empty:
                frame = frame.copy()
                frame["source_query"] = chip
                all_frames.append(frame)

        if body_id_chips:
            chip_dir = _chip_dir_for("_".join(str(c) for c in body_id_chips))
            frame = self._finder.find_lines_batch(
                queries=body_id_chips, dataset=dataset,
                output_dir=chip_dir, **batch_params)
            if chip_dir:
                self._hoist_finder_folder(chip_dir)
            if frame is not None and not frame.empty:
                all_frames.append(frame)

        # Single-chip runs are flat: the chip folder IS the run folder.
        if run_root and len(chip_dirs) == 1:
            self._hoist_folder(chip_dirs[0], run_root)

        # Output-detail policy (Compact): drop bodyId-level match tables
        # and images now that every summary / PDF exists; audited.
        cleanup_audit: Optional[Dict[str, Any]] = None
        retention_info: Optional[Dict[str, Any]] = None
        if run_root and prune_find_lines_run is not None:
            try:
                if not keep_per_match_csv and compact_keep_last_n > 0:
                    # Belt-and-braces (D2 refinement): keep the newest N
                    # runs' match tables — the in-progress run included —
                    # and sweep older Compact runs instead of deleting this
                    # run's tables outright.
                    retention_info = enforce_match_table_retention(
                        output_dir, run_root, compact_keep_last_n)
                    cleanup_audit = prune_find_lines_run(
                        run_root,
                        # Match tables stay (retention window); the
                        # downloaded images still go once the PDF exists.
                        keep_per_match_csv=True,
                        cleanup_source_images=True,
                    )
                else:
                    cleanup_audit = prune_find_lines_run(
                        run_root,
                        keep_per_match_csv=keep_per_match_csv,
                        cleanup_source_images=cleanup_source_images,
                    )
                if retention_info and retention_info["pruned_runs"]:
                    self._vprint(
                        f"   🗄️ Retention: kept match tables in the newest "
                        f"{compact_keep_last_n} run(s); pruned "
                        f"{len(retention_info['pruned_runs'])} older "
                        "Compact run(s)."
                    )
            except Exception as exc:
                self._vprint(f"   ⚠️ Output cleanup skipped: {exc}")

        self._progress(3, 3, "Write expansion report")
        if run_root:
            # The inner finder calls ran with the library defaults (the
            # orchestration owns the prune), so stamp the EFFECTIVE output
            # detail into their parameter records — a Compact run must not
            # read back as Full.
            self._stamp_output_detail(
                [run_root] + chip_dirs,
                keep_per_match_csv=keep_per_match_csv,
                cleanup_source_images=cleanup_source_images,
                compact_keep_last_n=compact_keep_last_n,
            )
            self._write_expansion_map(run_root, expansion_rows)
            self._write_summary(
                run_root, chips=chips, dataset=dataset, scope=scope,
                covered=covered, warnings=warnings,
                expansion_rows=expansion_rows,
                output_detail={
                    "keep_per_match_csv": keep_per_match_csv,
                    "cleanup_source_images": cleanup_source_images,
                    "compact_keep_last_n": compact_keep_last_n,
                },
                cleanup=cleanup_audit,
                retention=retention_info,
            )
            if warnings:
                self._write_warnings(run_root, warnings)

        if not all_frames:
            return pd.DataFrame()
        combined = pd.concat(all_frames, ignore_index=True)
        return combined

    @staticmethod
    def _stamp_output_detail(
        folders: Sequence[str],
        *,
        keep_per_match_csv: bool,
        cleanup_source_images: bool,
        compact_keep_last_n: int = 0,
    ) -> None:
        """Record the EFFECTIVE output-detail flags in every run-level
        ``parameters.json``.  The inner finder calls use the library
        defaults (the orchestration prunes), so without this stamp a
        Compact run would read back as Full."""
        for folder in folders:
            path = os.path.join(folder, "parameters.json")
            try:
                if not os.path.exists(path):
                    continue
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                function_params = payload.setdefault("function_params", {})
                function_params["keep_per_match_csv"] = keep_per_match_csv
                function_params["cleanup_source_images"] = cleanup_source_images
                function_params["compact_keep_last_n"] = compact_keep_last_n
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
            except Exception:
                continue

    @staticmethod
    def _hoist_finder_folder(chip_dir: str) -> None:
        """Collapse the finder's timestamped output folder into ``chip_dir``.

        The inner ``find_lines_batch`` always creates its own
        ``NB-find-lines_*`` folder; for per-chip folders the extra level is
        pure noise, so its contents move up one level.
        """
        try:
            children = os.listdir(chip_dir)
        except OSError:
            return
        if len(children) != 1:
            return
        inner = os.path.join(chip_dir, children[0])
        if not os.path.isdir(inner) or not children[0].startswith("NB-find-lines"):
            return
        for name in os.listdir(inner):
            os.rename(os.path.join(inner, name), os.path.join(chip_dir, name))
        os.rmdir(inner)

    @staticmethod
    def _hoist_folder(src: str, dst: str) -> None:
        """Move every child of ``src`` into ``dst`` and drop ``src``."""
        for name in os.listdir(src):
            os.rename(os.path.join(src, name), os.path.join(dst, name))
        os.rmdir(src)

    def _make_run_root(
        self,
        output_dir: str,
        chips: Sequence[Any],
        hosted_list: Sequence[str],
    ) -> str:
        """Create and return this run's folder under ``output_dir``."""
        try:
            from neuronbridge_finder import dataset_abbrev
        except ImportError:  # pragma: no cover - ``src`` layout
            from src.neuronbridge_finder import dataset_abbrev
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_info = "_".join(str(q) for q in list(chips)[:3])
        if len(chips) > 3:
            query_info += "_etc"
        query_info = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in query_info)
        abbrev = dataset_abbrev("_".join(hosted_list)) \
            if len(hosted_list) == 1 else "ALL"
        run_root = os.path.join(
            output_dir,
            f"NB-find-lines-expanded_{abbrev}_{query_info}_{timestamp}",
        )
        os.makedirs(run_root, exist_ok=True)
        return run_root

    # ------------------------------------------------------------------
    # report writers
    # ------------------------------------------------------------------

    def _write_expansion_map(
        self, output_dir: str, rows: List[Dict[str, str]]
    ) -> None:
        try:
            frame = pd.DataFrame(
                rows,
                columns=["source_query", "expanded_name", "nb_dataset",
                         "mapping_status", "mapping_kind"],
            )
            path = os.path.join(output_dir, EXPANSION_MAP_FILENAME)
            frame.to_csv(path, index=False)
            self._vprint(f"   💾 Expansion map: {path}")
        except Exception as exc:
            self._vprint(f"   ⚠️ Could not write the expansion map: {exc}")

    def _write_summary(
        self,
        output_dir: str,
        *,
        chips: Sequence[Any],
        dataset: Any,
        scope: List[str],
        covered: Dict[str, str],
        warnings: List[str],
        expansion_rows: List[Dict[str, str]],
        output_detail: Optional[Dict[str, Any]] = None,
        cleanup: Optional[Dict[str, Any]] = None,
        retention: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Self-describing run record: the user's ORIGINAL selection, the
        coverage routing that shaped the run, and the output-detail cleanup
        audit.  The per-chip ``parameters.json`` files only show the hosted
        releases, so this file is the provenance for what was selected, why
        releases were added (alignment) or dropped (unavailable), and what
        Compact removed."""
        payload = {
            "tool": "nb_find_lines_expanded",
            "created_at": datetime.now(
                timezone.utc).isoformat(timespec="seconds"),
            "queries": [str(chip) for chip in chips],
            "selected_datasets": dataset,
            "scope_datasets": scope,
            "covered_datasets": dict(covered),
            "unavailable_datasets": [
                ds for ds in scope if ds not in covered
            ],
            "warnings": warnings,
            "expansion": expansion_rows,
            "output_detail": output_detail or {},
            "cleanup": cleanup or {},
            "retention": retention or {},
        }
        try:
            path = os.path.join(output_dir, EXPANSION_SUMMARY_FILENAME)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            self._vprint(f"   💾 Expansion summary: {path}")
        except Exception as exc:
            self._vprint(f"   ⚠️ Could not write the expansion summary: {exc}")

    def _write_warnings(self, output_dir: str, warnings: List[str]) -> None:
        """Append coverage notes to the run's ``user_warning_notes.txt``.

        Same warning-file convention as the other NeuronBridge tools; each
        line carries a ``coverage:`` prefix so the notes stay attributable.
        """
        try:
            path = os.path.join(output_dir, WARNINGS_FILENAME)
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(f"[{stamp}]\n")
                for warning in warnings:
                    handle.write(f"WARNING: coverage: {warning}\n")
        except OSError:
            pass


def find_lines_expanded(**kwargs: Any) -> pd.DataFrame:
    """Module-level convenience matching the runner's method-call style."""
    finder = ExpandedLineFinder(
        verbose=bool(kwargs.pop("verbose", True)),
        separate_splitgal4=bool(kwargs.pop("separate_splitgal4", True)),
        region=str(kwargs.pop("region", "Brain")),
        max_workers=int(kwargs.pop("max_workers", 8)),
    )
    return finder.run(**kwargs)
