#!/usr/bin/env python
"""Optional recompression of the FAFB v783 healed skeleton bundle.

Bulk-converts ``sk_lod1_783_healed.zip`` (13.87 GB, 139,273 SWC entries)
into the appendable columnar container ``sk_lod1_783_healed.zst``
(≈ 8.8 GB, ~5 GB saved, ≈ 1 h single-threaded / ≈ 10-15 min with 8
workers).  The same container is also produced lazily by the reader on the
ZIP fallback path (every loaded skeleton converts on the fly); this script
executes the optional bulk conversion, verification, compaction and the
interactive first-run question.

Run (from the repository root):
    python scripts/maintenance/flywire_fafb_v783_skeleton_recompression.py pack
    python scripts/maintenance/flywire_fafb_v783_skeleton_recompression.py verify
    python scripts/maintenance/flywire_fafb_v783_skeleton_recompression.py info
    python scripts/maintenance/flywire_fafb_v783_skeleton_recompression.py compact
    python scripts/maintenance/flywire_fafb_v783_skeleton_recompression.py prompt

Options:
    --dataset-dir PATH   dataset folder (default: datasets/flywire_FAFB_v783)
    --workers N          parallel pack workers (default: auto, up to 8)
    --delete-source      delete the ZIP after a successful verified pack
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fafb_bundle import (  # noqa: E402
    FAFBSkeletonBundle,
    info as bundle_info,
    pack,
    verify,
)

DEFAULT_DATASET_DIR = ROOT / "datasets" / "flywire_FAFB_v783"
ZIP_NAME = "sk_lod1_783_healed.zip"
BUNDLE_NAME = "sk_lod1_783_healed.zst"


def _paths(dataset_dir: Path):
    dataset_dir = Path(dataset_dir)
    return dataset_dir / ZIP_NAME, dataset_dir / BUNDLE_NAME


def _pack_now(zip_path: Path, bundle_path: Path, workers: int,
              delete_source: bool) -> int:
    print(f"Packing {zip_path} -> {bundle_path} ...")
    started = time.time()
    stats = pack(zip_path, bundle_path, n_workers=workers)
    elapsed = time.time() - started
    print(json.dumps(stats, indent=2))
    print(f"packed in {elapsed / 60:.1f} min "
          f"({stats['bytes'] / 1e9:.2f} GB, {stats['neurons']} neurons)")
    print("verifying...")
    result = verify(bundle_path, zip_path=zip_path, sample=200)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        print("VERIFY FAILED - the bundle was kept, the ZIP was NOT deleted")
        return 1
    if delete_source:
        zip_path.unlink(missing_ok=True)
        print(f"source ZIP deleted (reclaimed {stats['bytes'] / 1e9:.2f} GB "
              "of bundle + ZIP space)")
    return 0


def _cmd_pack(args) -> int:
    zip_path, bundle_path = _paths(args.dataset_dir)
    if not zip_path.exists():
        print(f"ZIP not found: {zip_path}")
        return 1
    return _pack_now(zip_path, bundle_path, args.workers, args.delete_source)


def _cmd_verify(args) -> int:
    zip_path, bundle_path = _paths(args.dataset_dir)
    if not bundle_path.exists():
        print(f"bundle not found: {bundle_path}")
        return 1
    result = verify(
        bundle_path,
        zip_path=zip_path if zip_path.exists() else None,
        sample=args.sample)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def _cmd_info(args) -> int:
    zip_path, bundle_path = _paths(args.dataset_dir)
    result = bundle_info(
        bundle_path, zip_path=zip_path if zip_path.exists() else None)
    print(json.dumps(result, indent=2))
    return 0


def _cmd_compact(args) -> int:
    zip_path, bundle_path = _paths(args.dataset_dir)
    if not zip_path.exists() or not bundle_path.exists():
        print("both the ZIP and the bundle must exist to compact")
        return 1
    reader = FAFBSkeletonBundle(bundle_path, zip_path=zip_path,
                                lazy_convert=False)
    try:
        removed = reader.compact_zip()
    finally:
        reader.close()
    print(f"removed {removed} converted entries from the ZIP")
    if removed == 0:
        print("nothing to remove (no converted entries)")
    return 0


def _cmd_prompt(args) -> int:
    """The interactive first-run question (also invoked by the converter)."""
    zip_path, bundle_path = _paths(args.dataset_dir)
    if not zip_path.exists():
        print(f"ZIP not found: {zip_path}")
        return 1
    if bundle_path.exists():
        print(f"bundle already exists: {bundle_path}")
        return 0
    print()
    print("=" * 70)
    print("OPTIONAL SKELETON RECOMPRESSION")
    print("=" * 70)
    print("The FAFB healed skeleton bundle can be recompressed into the")
    print("columnar .zst container:")
    print(f"  - saves ~5 GB of storage (13.87 GB -> ~8.8 GB)")
    print(f"  - ~1 h of extra work (or ~10-15 min with parallel workers)")
    print("Skipping enables LAZY conversion: every skeleton loaded from the")
    print("ZIP is converted on the fly and its ZIP entry is removed.")
    print()
    answer = input("Recompress now? [y]es now / [l]azy conversion: ").strip().lower()
    if answer in ("y", "yes"):
        return _pack_now(zip_path, bundle_path, args.workers, False)
    print("lazy conversion enabled: the reader converts on first load")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optional FAFB v783 healed skeleton recompression")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--workers", type=int, default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    p_pack = sub.add_parser("pack", help="bulk-convert the healed ZIP now")
    p_pack.add_argument("--delete-source", action="store_true")

    p_verify = sub.add_parser("verify", help="round-trip checks vs the ZIP")
    p_verify.add_argument("--sample", type=int, default=200)

    sub.add_parser("info", help="entry counts and sizes")
    sub.add_parser("compact", help="physically drop converted ZIP entries")
    sub.add_parser("prompt", help="ask whether to recompress now or lazily")

    args = parser.parse_args()
    if args.command == "pack":
        return _cmd_pack(args)
    if args.command == "verify":
        return _cmd_verify(args)
    if args.command == "info":
        return _cmd_info(args)
    if args.command == "compact":
        return _cmd_compact(args)
    if args.command == "prompt":
        return _cmd_prompt(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
