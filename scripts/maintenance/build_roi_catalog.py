#!/usr/bin/env python3
"""Regenerate the tracked skeleton ROI snapshot (``ui/data/roi_catalogs.json``).

The skeleton UI's ROI select reads local ``datasets/`` and ``cache/`` sidecars,
which are gitignored and absent on a fresh install.  This script captures the
primary + available ROI inventories from a machine that HAS the data so a new
clone can still offer sub-primary ROIs offline.  The snapshot is the UI's last
fallback; live local data always wins.

Run on a data machine (NeuPrint caches / local releases present):

    PYTHONPATH="src:." ~/anaconda3/envs/drocat-4.5.0/bin/python \
        scripts/maintenance/build_roi_catalog.py

``--check`` regenerates in memory and exits non-zero if the committed file has
drifted from the local data (useful after refreshing caches; not a CI gate,
since CI has no dataset sidecars).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ui.roi_options import BUNDLED_ROI_CATALOG, _local_roi_catalog  # noqa: E402


def _dataset_folders(root: Path) -> list[str]:
    folders: set[str] = set()
    for base in (root / "datasets", root / "cache"):
        if base.is_dir():
            folders.update(p.name for p in base.iterdir() if p.is_dir())
    return sorted(folders)


def build_catalog(root: Path) -> dict[str, dict[str, list[str]]]:
    catalog: dict[str, dict[str, list[str]]] = {}
    for folder in _dataset_folders(root):
        local = _local_roi_catalog(root, [folder])
        primary = local["primary"]
        available = local["available"]
        if not primary and not available:
            continue
        # Canonical, run-stable order for the snapshot (case-equal names such
        # as AL(R)/aL(R) must not reorder between regenerations).
        available = sorted(available, key=lambda s: (s.casefold(), s))
        catalog[folder] = {"primary": primary, "available": available}
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="exit non-zero if the committed snapshot differs from local data",
    )
    parser.add_argument(
        "--output", type=Path, default=BUNDLED_ROI_CATALOG,
        help="snapshot path (default: ui/data/roi_catalogs.json)",
    )
    args = parser.parse_args()

    catalog = build_catalog(ROOT)
    if not catalog:
        print(
            f"ERROR: no local ROI sidecars found under {ROOT}; refusing to "
            "overwrite the shipped snapshot with an empty catalog.",
            file=sys.stderr,
        )
        return 1
    payload = json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        try:
            current = args.output.read_text(encoding="utf-8")
        except OSError:
            current = ""
        if current != payload:
            print(f"DRIFT: {args.output} differs from local data; "
                  "re-run without --check to regenerate.")
            return 1
        print(f"OK: {args.output} matches local data "
              f"({len(catalog)} datasets).")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    total = sum(len(v["available"]) for v in catalog.values())
    print(f"Wrote {args.output} ({len(catalog)} datasets, "
          f"{total} available names, {args.output.stat().st_size} bytes).")
    for folder, entry in catalog.items():
        print(f"  {folder}: primary={len(entry['primary'])} "
              f"available={len(entry['available'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())