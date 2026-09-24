"""One-time migration: rename on-disk BANC artifacts to the canonical names.

DROCAT now analyzes BANC from its own public release data instead of through
FlyWire, so the dataset identifiers dropped the ``flywire_`` prefix
(``flywire_BANC_v626`` -> ``banc_v626``, ``flywire_BANC_v888`` ->
``banc_v888``).  Code accepts both spellings (see
``utils.naming_utils.canonical_dataset_name``), but the on-disk folders and
name-embedded files are renamed here so paths stay consistent.

Handled locations (all regenerable except ``datasets/``):
- ``datasets/flywire_BANC_v*/``          -> renamed dirs, inner files, metadata.json
- ``neuron_indexes/flywire_BANC_v*/``    -> renamed dirs + manifest.json entries
- ``cache/flywire_BANC_v*/``             -> renamed dirs (warm caches)
- ``cache/neuronbridge/**/id_to_lines/`` -> renamed per-neuron parquet files

(``cache/dataset_availability.json`` now stores only the server dimension
with canonical keys, so it needs no re-keying.)

Frozen user outputs (``homolog_param_benchmark/`` etc.) are intentionally not
touched.  Idempotent: already-renamed targets are skipped.  Run with
``--dry-run`` to preview.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RENAMES = [
    ("flywire_BANC_v626", "banc_v626"),
    ("flywire_BANC_v888", "banc_v888"),
    ("flywire_BANC_v999", "banc_v999"),
]

TARGET_DIRS = ["datasets", "neuron_indexes", "cache"]


def new_name(name: str):
    for old, new in RENAMES:
        if old in name:
            return name.replace(old, new)
    return None


def rename_path(path: Path, dry_run: bool) -> bool:
    target = new_name(path.name)
    if target is None:
        return False
    dest = path.with_name(target)
    if dest.exists():
        print(f"  = skip (target exists): {path.name}")
        return False
    if dry_run:
        print(f"  -> would rename: {path} -> {dest}")
        return True
    path.rename(dest)
    print(f"  renamed: {path.name} -> {target}")
    return True


def patch_json_values(path: Path, dry_run: bool) -> bool:
    """Rewrite legacy BANC names inside a JSON document (manifests)."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, ValueError):
        return False
    changed = False
    for old, new in RENAMES:
        if old in text:
            text = text.replace(old, new)
            changed = True
    if not changed:
        return False
    if dry_run:
        print(f"  -> would patch values: {path}")
        return True
    path.write_text(text, encoding="utf-8")
    print(f"  patched values: {path}")
    return True


def patch_metadata_json(dataset_dir: Path, dry_run: bool) -> bool:
    meta = dataset_dir / f"{dataset_dir.name}_metadata.json"
    if not meta.exists():
        return False
    try:
        data = json.loads(meta.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    if data.get("dataset") != dataset_dir.name:
        if dry_run:
            print(f"  -> would set dataset={dataset_dir.name} in {meta.name}")
            return True
        data["dataset"] = dataset_dir.name
        meta.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  patched: {meta.name} (dataset={dataset_dir.name})")
    return True


def migrate_dir_children(path: Path, dry_run: bool) -> None:
    for child in sorted(path.iterdir()):
        if new_name(child.name):
            rename_path(child, dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="print planned actions without changing anything")
    args = parser.parse_args()
    dry_run = args.dry_run

    moved = 0
    for dir_name in TARGET_DIRS:
        base = PROJECT_ROOT / dir_name
        if not base.is_dir():
            continue
        print(f"[{dir_name}/]")
        for child in sorted(base.iterdir()):
            if not child.is_dir() or new_name(child.name) is None:
                continue
            if rename_path(child, dry_run):
                moved += 1
            # Inner files of a renamed (or already-canonical) dataset dir.
            target_dir = base / new_name(child.name)
            if target_dir.is_dir():
                if dir_name == "datasets":
                    migrate_dir_children(target_dir, dry_run)
                    patch_metadata_json(target_dir, dry_run)
                elif dir_name == "neuron_indexes":
                    manifest = base / "manifest.json"
                    if manifest.exists():
                        patch_json_values(manifest, dry_run)

    # NeuronBridge per-neuron caches embed the dataset name in the filename.
    nb_root = PROJECT_ROOT / "cache" / "neuronbridge"
    if nb_root.is_dir():
        print("[cache/neuronbridge/]")
        for path in sorted(nb_root.rglob("*flywire_BANC*")):
            if path.is_file() and rename_path(path, dry_run):
                moved += 1

    avail = PROJECT_ROOT / "cache" / "dataset_availability.json"
    if avail.exists():
        # The availability file now stores only the server dimension (canonical
        # NeuPrint/bucket keys), so it holds no BANC namespaces to migrate.
        print("[cache/dataset_availability.json] server-only; nothing to rename")

    print(f"\n{'would rename' if dry_run else 'renamed'} {moved} path(s).")
    if not dry_run:
        print("Code accepts legacy spellings via canonical_dataset_name(), so "
              "old configs and scripts keep working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
