#!/usr/bin/env python
"""Purge legacy simplified skeletons from the local skeleton caches.

Design contract (plan-raw-basis-vectorization.md): the on-disk skeleton
cache stores RAW skeletons; simplification is a visualization-time concern.
The V2 vector cache builds from raw trees only (simplified files are
skipped), so a level-90 file left in the cache makes its body unavailable
to vector builds and renders until it is re-fetched raw.

This tool scans ``cache/<dataset>/skeletons/**`` for ``.swc.zst`` /
``.swc.gz`` files whose recorded simplification level is > 0 and either
reports them (dry-run, default), moves them to a quarantine folder
(``--quarantine``, recoverable), or deletes them (``--delete``).  It also
reports/removes stale morph-cross vector sidecars (``--purge-sidecars``)
whose signature predates the raw-basis flip.

Headerless legacy ``.swc.gz`` files read as raw (level 0) and are never
touched.

Usage:
    python scripts/maintenance/purge_legacy_simp90_cache.py            # dry-run
    python scripts/maintenance/purge_legacy_simp90_cache.py --quarantine
    python scripts/maintenance/purge_legacy_simp90_cache.py --delete
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKELETON_ROOTS = ("skeletons",)


def _read_stamp(path: Path) -> int:
    try:
        if path.suffix == ".gz":
            import gzip
            with gzip.open(path, "rb") as handle:
                head = handle.read(400).decode("utf-8", errors="ignore")
        else:
            import zstandard
            dctx = zstandard.ZstdDecompressor()
            with open(path, "rb") as fh:
                head = dctx.stream_reader(fh).read(400).decode(
                    "utf-8", errors="ignore")
        for line in head.splitlines():
            if line.startswith("# DROCAT simpl:"):
                return int(line.split(":", 1)[1].strip() or 0)
        return -1  # headerless legacy: reads as raw (level 0)
    except Exception:
        return -2  # unreadable: report, do not touch


def _iter_skeleton_files(project_root: Path):
    cache_root = project_root / "cache"
    if not cache_root.is_dir():
        return
    for ds_dir in sorted(cache_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        for root in SKELETON_ROOTS:
            base = ds_dir / root
            if not base.is_dir():
                continue
            for pattern in ("*.swc.zst", "*.swc.gz"):
                for path in sorted(base.rglob(pattern)):
                    yield ds_dir, path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report/quarantine/delete legacy simplified (>0) "
                    "skeleton cache files. Dry-run report by default.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--quarantine", action="store_true",
                        help="move simp90 files to "
                             "<ds>/skeletons/_legacy_simp90/ (recoverable)")
    parser.add_argument("--delete", action="store_true",
                        help="permanently delete simp90 files")
    parser.add_argument("--purge-sidecars", action="store_true",
                        help="also delete morph-cross target-vector "
                             "sidecars (cross_dataset_nullvec_*.npz, "
                             "cross_dataset_targetvec_*.npz)")
    args = parser.parse_args()
    if args.delete and args.quarantine:
        parser.error("--delete and --quarantine are mutually exclusive")
    root = Path(args.project_root)

    legacy_total = 0
    raw_total = 0
    print(f"Scanning {root / 'cache'} for skeleton simplification stamps...\n")
    for ds_dir, path in _iter_skeleton_files(root):
        stamp = _read_stamp(path)
        if stamp == -2:
            print(f"  [unreadable] {path}")
            continue
        if stamp <= 0:
            raw_total += 1
            continue
        legacy_total += 1
        rel = path.relative_to(root)
        print(f"  [simp{stamp}] {rel}  ({path.stat().st_size / 1e6:.2f} MB)")
        if args.delete:
            path.unlink()
        elif args.quarantine:
            # Quarantine OUTSIDE the cache's skeleton_dir (a sibling of the
            # store folder): find_skeleton_file rglobs subdirectories of the
            # store when any exist, so an in-store quarantine folder would
            # keep the files discoverable (and re-seed simp90 into loads).
            quarantine = path.parents[1] / "_legacy_simp90"
            quarantine.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(quarantine / path.name))

    print(f"\nraw files: {raw_total}   legacy simplified files: "
          f"{legacy_total}")
    if legacy_total and not (args.delete or args.quarantine):
        print("Dry run only — re-run with --quarantine (recoverable) or "
              "--delete to act.")

    if args.purge_sidecars:
        removed = 0
        # both generations of the sidecar: the target-vector store was
        # named for its first use (the null sample), then widened
        patterns = ("*/find_similar/morphology/cross_dataset_nullvec_*.npz",
                    "*/find_similar/morphology/cross_dataset_targetvec_*.npz")
        sidecars = sorted(p for pat in patterns
                          for p in (root / "cache").glob(pat))
        for sidecar in sidecars:
            sidecar.unlink()
            removed += 1
            print(f"  removed sidecar: {sidecar.relative_to(root)}")
        print(f"vector sidecars removed: {removed}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
