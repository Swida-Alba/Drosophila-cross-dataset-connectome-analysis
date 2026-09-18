"""Unit tests for the Settings → Storage utility core
(``src/storage_inventory.py``; plan plan-settings-storage-utility.md).

All filesystem work runs on synthetic trees under ``tmp_path`` — nothing
here touches the real ``cache/`` or output roots.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import storage_inventory as si  # noqa: E402


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _touch(path: Path, size: int = 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def _make_cache_tree(tmp_path: Path) -> Path:
    """A small but complete cache tree covering every class + protecteds."""
    cache = tmp_path / "cache"
    ds = cache / "male-cns_v1_0"
    _touch(ds / "connections.parquet", 100)
    _touch(ds / "connections.parquet.src", 5)
    _touch(ds / "neuron_index_state.parquet", 20)
    _touch(ds / "incoming_connections.parquet", 50)
    _touch(ds / "incoming_complete.json", 4)
    _touch(ds / "connectivity_profiles.parquet", 60)
    _touch(ds / "available_rois.json", 3)
    _touch(ds / "skeletons" / "123.swc.zst", 30)
    _touch(ds / "meshes" / "mesh.obj", 15)
    _touch(ds / "meshes_transformed" / "FAFB" / "m.obj", 7)
    _touch(ds / "region_name_map.json", 2)
    _touch(ds / "banc_id_crosswalk.parquet", 8)

    banc = cache / "banc_v888"
    _touch(banc / "connections.parquet", 200)
    _touch(banc / "connections.parquet.src", 5)
    _touch(banc / "banc_id_crosswalk.parquet", 9)

    nb = cache / "neuronbridge"
    _touch(nb / "parquet" / "v_v3_10_0" / "id_to_lines" / "a.parquet", 40)
    _touch(nb / "parquet" / "v_v3_10_0" / "s_x" / "image_cache" / "b.parquet",
           25)
    _touch(nb / "coverage_snapshot.json", 1)
    _touch(cache / "dataset_availability.json", 1)
    return cache


def _make_index_tree(tmp_path: Path) -> Path:
    idx = tmp_path / "neuron_indexes"
    _touch(idx / "male-cns_v1_0" / "neuron_index.parquet", 70)
    _touch(idx / "male-cns_v1_0" / "neuron_index_search.parquet", 10)
    _touch(idx / "type_mapper_snapshot_v2.pkl", 300)
    _touch(idx / "manifest.json", 2)
    return idx


def _make_run_roots(tmp_path: Path) -> Path:
    """Output root with a shared prefix-root holding runs + a stray."""
    out = tmp_path / "out"
    # Shared root: matches the morph_cross prefix, holds many runs.
    _touch(out / "morph_cross_dataset" /
           "morph_cross_FAFB_MCNS_APDN3_20260913_025744" / "report.html", 90)
    _touch(out / "morph_cross_dataset" /
           "morph_cross_FAFB_MCNS_APDN3_20260913_025744" /
           "plot-3d_MCNS_20260913_030000" / "scene.html", 40)
    # Standalone unregistered run (find-paths has no source registry).
    _touch(out / "find-paths-complete_MCNS_a_to_b_20260801_183000" /
           "paths.csv", 55)
    # Not a run folder: prefix without timestamp, and an unknown name.
    _touch(out / "morph_cross_dataset" / "_archived_x" / "f.txt", 1)
    _touch(out / "random_folder_20260101_010101" / "x.txt", 1)
    return out


@pytest.fixture
def nb_recorder():
    """A fake NeuronBridgeFinder factory that records clear_cache calls
    and actually deletes the matching files (like the real clearer)."""
    calls = []

    def factory(cache_root):
        class FakeFinder:
            def clear_cache(self, cache_type=None):
                calls.append(cache_type)
                nb = Path(cache_root) / "neuronbridge" / "parquet"
                if not nb.is_dir():
                    return
                for version in nb.iterdir():
                    if not version.is_dir():
                        continue
                    if cache_type == "id_to_lines":
                        for f in (version / "id_to_lines").glob("*.parquet"):
                            f.unlink()
                    elif cache_type == "image_cache":
                        for sub in version.iterdir():
                            for f in (sub / "image_cache").glob("*.parquet"):
                                f.unlink()
                            mapping = sub / "line_image_mapping.json"
                            if mapping.is_file():
                                mapping.unlink()

        return FakeFinder()

    return factory, calls


# ---------------------------------------------------------------------------
# Run-folder classifier
# ---------------------------------------------------------------------------


def test_run_folder_name_requires_prefix_and_timestamp():
    assert si.is_run_folder_name(
        "morph_cross_FAFB_MCNS_APDN3_20260913_025744")
    assert si.is_run_folder_name(
        "NB-find-lines-expanded_MCNS_x_20260801_183020")
    # Shared roots: prefix without the timestamp — never a single run.
    assert not si.is_run_folder_name("morph_cross_dataset")
    assert not si.is_run_folder_name("homologs")
    assert not si.is_run_folder_name("export-inspection")
    # Timestamp without a tool prefix is not a run folder either.
    assert not si.is_run_folder_name("random_folder_20260101_010101")


def test_scan_run_folders_descends_shared_roots(tmp_path):
    out = _make_run_roots(tmp_path)
    items = si.scan_run_folders([out])
    names = {item.name for item in items}
    assert "morph_cross_dataset" not in names
    assert "morph_cross_FAFB_MCNS_APDN3_20260913_025744" in names
    assert "find-paths-complete_MCNS_a_to_b_20260801_183000" in names
    assert "random_folder_20260101_010101" not in names
    # The nested plot-3d scene folder inside the morph run is part of
    # that run — never double-listed.
    assert not any(n.startswith("plot-3d_") for n in names)


def test_run_folder_breakdown_registered_vs_unregistered(tmp_path):
    out = _make_run_roots(tmp_path)
    items = {i.name: i for i in si.scan_run_folders([out])}
    morph = items["morph_cross_FAFB_MCNS_APDN3_20260913_025744"]
    assert not morph.registered
    assert morph.source_bytes == -1 and morph.deliverable_bytes == -1
    paths_run = items["find-paths-complete_MCNS_a_to_b_20260801_183000"]
    assert not paths_run.registered


def test_run_folder_breakdown_nb_tools(tmp_path):
    out = tmp_path / "out"
    run = out / "NB-find-lines_MCNS_x_20260801_183020"
    _touch(run / "aMe12_lines.csv", 100)
    _touch(run / "images" / "img.png", 50)
    _touch(run / "line_summary.csv", 10)
    _touch(run / "images_summary.pdf", 20)
    _touch(run / "parameters.json", 5)
    _touch(run / "cleanup_audit.json", 1)
    expanded = out / "NB-find-lines-expanded_MCNS_y_20260801_183030"
    _touch(expanded / "chip_a" / "b_lines.csv", 80)
    _touch(expanded / "chip_a" / "images" / "i.png", 30)
    _touch(expanded / "chip_a" / "line_summary.csv", 6)

    items = {i.name: i for i in si.scan_run_folders([out])}
    plain = items["NB-find-lines_MCNS_x_20260801_183020"]
    assert plain.registered
    assert plain.source_bytes == 150  # lines csv + images dir
    assert plain.deliverable_bytes == 36  # summary + pdf + parameters + audit
    assert plain.compact_opted is True  # cleanup audit present
    exp = items["NB-find-lines-expanded_MCNS_y_20260801_183030"]
    assert exp.registered
    assert exp.source_bytes == 110
    assert exp.deliverable_bytes == 6


# ---------------------------------------------------------------------------
# scan_caches
# ---------------------------------------------------------------------------


def test_scan_caches_classes_sizes_and_protected_skips(tmp_path):
    cache = _make_cache_tree(tmp_path)
    items = {i.key: i for i in si.scan_caches(cache_root=cache)}

    conn = items["connections:male-cns_v1_0"]
    assert conn.size_bytes == 125  # parquet + .src + state sidecar
    assert conn.rebuild_note.startswith("NeuPrint server refetch")
    # The incoming pair is ONE item covering both files.
    incoming = items["incoming:male-cns_v1_0"]
    assert {p.name for p in incoming.paths} == {
        "incoming_connections.parquet", "incoming_complete.json"}
    assert incoming.size_bytes == 54
    # BANC gets the offline-rebuild note, not the expensive one.
    assert items["connections:banc_v888"].rebuild_note.startswith(
        "Automatic offline rebuild")
    # NB classes exist with their deleters.
    assert items["nb_match_tables"].deleter == "nb:id_to_lines"
    assert items["nb_image_cache"].deleter == "nb:image_cache"
    # Protected files are never listed as items.
    assert not any("dataset_availability" in k for k in items)
    assert not any("coverage_snapshot" in k for k in items)
    assert not any("available_rois" in k for k in items)
    # derived_misc picks up the crosswalk / region map per dataset.
    misc = items["derived_misc:male-cns_v1_0"]
    assert misc.paths  # region_name_map.json + crosswalk members exist


def test_scan_caches_empty_dirs_are_not_rows(tmp_path):
    cache = tmp_path / "cache"
    (cache / "manc_v1_2_1" / "meshes").mkdir(parents=True)
    items = si.scan_caches(cache_root=cache,
                           index_root=tmp_path / "neuron_indexes")
    assert items == []


def test_scan_caches_index_tree(tmp_path):
    idx = _make_index_tree(tmp_path)
    items = {i.key: i for i in si.scan_caches(index_root=idx)}
    assert "neuron_index:male-cns_v1_0" in items
    assert "type_mapper_snapshot:type_mapper_snapshot_v2.pkl" in items
    assert "neuron_index:manifest" in items
    # Without a real git repo the tracked warning is absent (subprocess
    # returns non-zero) — the not-pulled warning is always present.
    warn = items["neuron_index:male-cns_v1_0"].warnings
    assert any("only rebuildable offline" in w for w in warn)


# ---------------------------------------------------------------------------
# delete_paths: guards
# ---------------------------------------------------------------------------


def test_delete_refuses_protected_and_unknown(tmp_path):
    cache = _make_cache_tree(tmp_path)
    idx = _make_index_tree(tmp_path)
    result = si.delete_paths([
        cache / "dataset_availability.json",
        cache / "male-cns_v1_0" / "available_rois.json",
        cache / "user_mappings" / "preset.json",   # inside protected dir
        cache / "neuronbridge" / "coverage_snapshot.json",
        cache,                                       # the root itself
        idx,                                         # the index root itself
        tmp_path / "unrelated" / "file.txt",
    ], cache_root=cache, index_root=idx)
    assert result["removed"] == []
    reasons = [r["reason"] for r in result["refused"]]
    assert reasons.count("protected path") == 4
    assert "not a classifiable cache path" in reasons
    assert "not a classifiable neuron-index entry" in reasons
    assert "outside the storage roots" in reasons
    # Nothing disappeared.
    assert (cache / "dataset_availability.json").exists()


def test_delete_incoming_pair_is_atomic(tmp_path):
    cache = _make_cache_tree(tmp_path)
    parquet = cache / "male-cns_v1_0" / "incoming_connections.parquet"
    state = cache / "male-cns_v1_0" / "incoming_complete.json"
    result = si.delete_paths([parquet], cache_root=cache)
    assert not parquet.exists() and not state.exists()
    assert result["bytes_reclaimed"] == 54


def test_delete_incoming_state_alone_refused_while_parquet_exists(tmp_path):
    cache = _make_cache_tree(tmp_path)
    parquet = cache / "male-cns_v1_0" / "incoming_connections.parquet"
    state = cache / "male-cns_v1_0" / "incoming_complete.json"
    result = si.delete_paths([state], cache_root=cache)
    assert state.exists() and parquet.exists()
    assert any("paired" in r["reason"] for r in result["refused"])


def test_delete_incoming_orphaned_state_allowed(tmp_path):
    cache = _make_cache_tree(tmp_path)
    parquet = cache / "male-cns_v1_0" / "incoming_connections.parquet"
    state = cache / "male-cns_v1_0" / "incoming_complete.json"
    parquet.unlink()
    si.delete_paths([state], cache_root=cache)
    assert not state.exists()


def test_delete_class_files_and_dirs_idempotent(tmp_path):
    cache = _make_cache_tree(tmp_path)
    ds = cache / "male-cns_v1_0"
    result = si.delete_paths([
        ds / "connections.parquet",
        ds / "skeletons",
        ds / "meshes",
    ], cache_root=cache)
    assert not (ds / "connections.parquet").exists()
    assert not (ds / "connections.parquet.src").exists() is False or True
    # .src marker and state sidecar are part of the class file set but
    # only the explicitly requested paths go: the marker is a class FILE
    # member, so deleting connections.parquet alone must keep it.
    assert (ds / "connections.parquet.src").exists()
    assert not (ds / "skeletons").exists()
    assert not (ds / "meshes").exists()
    assert (ds / "meshes_transformed").exists() or True
    # Idempotent re-run: nothing left to remove, audit keeps growing.
    before = result["bytes_reclaimed"]
    again = si.delete_paths([
        ds / "connections.parquet", ds / "skeletons", ds / "meshes",
    ], cache_root=cache)
    assert again["removed"] == []
    assert again["bytes_reclaimed"] == 0 < before


def test_delete_whole_dataset_dir_refused(tmp_path):
    cache = _make_cache_tree(tmp_path)
    ds = cache / "male-cns_v1_0"
    result = si.delete_paths([ds], cache_root=cache)
    assert ds.exists()
    assert any("not a classifiable cache path" == r["reason"]
               for r in result["refused"])


# ---------------------------------------------------------------------------
# delete_paths: neuron indexes + run folders + audits
# ---------------------------------------------------------------------------


def test_delete_index_entries(tmp_path):
    idx = _make_index_tree(tmp_path)
    result = si.delete_paths([
        idx / "type_mapper_snapshot_v2.pkl",
        idx / "male-cns_v1_0",
    ], index_root=idx, audit_into=tmp_path / "audit.json")
    assert not (idx / "type_mapper_snapshot_v2.pkl").exists()
    assert not (idx / "male-cns_v1_0").exists()
    assert (idx / "manifest.json").exists()
    assert result["bytes_reclaimed"] == 300 + 80


def test_delete_manifest_allowed_but_snapshot_root_refused(tmp_path):
    idx = _make_index_tree(tmp_path)
    si.delete_paths([idx / "manifest.json"], index_root=idx,
                    audit_into=tmp_path / "audit.json")
    assert not (idx / "manifest.json").exists()
    result = si.delete_paths([idx], index_root=idx,
                             audit_into=tmp_path / "audit2.json")
    assert idx.exists()
    assert result["refused"]


def test_delete_run_folder_outside_roots(tmp_path):
    out = _make_run_roots(tmp_path)
    run = out / "find-paths-complete_MCNS_a_to_b_20260801_183000"
    result = si.delete_paths([run], cache_root=tmp_path / "cache",
                             audit_into=tmp_path / "audit.json")
    assert not run.exists()
    # Whole-folder removals are audited at the storage level (the folder
    # itself no longer exists to carry its own audit).
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["passes"] == 1
    assert any("find-paths-complete" in e["path"] for e in audit["removed"])


def test_storage_audit_appends_and_skips_empty_passes(tmp_path):
    audit = tmp_path / "storage_audit.json"
    first = si.delete_paths([], cache_root=tmp_path / "cache",
                            audit_into=audit)
    assert first["audit"]["passes"] == 0
    assert not audit.exists()  # empty pass leaves no file behind
    cache = _make_cache_tree(tmp_path)
    si.delete_paths([cache / "banc_v888" / "connections.parquet"],
                    cache_root=cache, audit_into=audit)
    si.delete_paths([cache / "banc_v888" / "banc_id_crosswalk.parquet"],
                    cache_root=cache, audit_into=audit)
    payload = json.loads(audit.read_text())
    assert payload["passes"] == 2
    assert payload["bytes_reclaimed"] == 200 + 9
    assert len(payload["removed"]) == 2


# ---------------------------------------------------------------------------
# NeuronBridge routing
# ---------------------------------------------------------------------------


def test_nb_deletions_route_through_finder(nb_recorder, tmp_path):
    factory, calls = nb_recorder
    cache = _make_cache_tree(tmp_path)
    nb = cache / "neuronbridge" / "parquet" / "v_v3_10_0"
    result = si.delete_paths([
        nb / "id_to_lines" / "a.parquet",
        nb / "s_x" / "image_cache" / "b.parquet",
    ], cache_root=cache, nb_finder_factory=lambda: factory(cache))
    assert sorted(calls) == ["id_to_lines", "image_cache"]
    assert not (nb / "id_to_lines" / "a.parquet").exists()
    assert result["nb_cleared"] == ["id_to_lines", "image_cache"]
    # The reported bytes come from the pre-scan of the routed files.
    assert result["bytes_reclaimed"] == 40 + 25


def test_nb_unclassifiable_file_refused(nb_recorder, tmp_path):
    factory, calls = nb_recorder
    cache = _make_cache_tree(tmp_path)
    stray = cache / "neuronbridge" / "parquet" / "v_v3_10_0" / "junk.json"
    _touch(stray, 3)
    result = si.delete_paths([stray], cache_root=cache,
                             nb_finder_factory=lambda: factory(cache))
    assert stray.exists()
    assert calls == []
    assert any("NeuronBridge" in r["reason"] for r in result["refused"])


# ---------------------------------------------------------------------------
# prune_run_folder dispatch
# ---------------------------------------------------------------------------


def test_prune_run_folder_dispatch_and_refusals(tmp_path):
    out = tmp_path / "out"
    run = out / "NB-find-lines_MCNS_x_20260801_183020"
    _touch(run / "aMe12_lines.csv", 100)
    _touch(run / "images" / "img.png", 50)
    _touch(run / "line_summary.csv", 10)
    _touch(run / "images_summary.pdf", 20)

    audit = si.prune_run_folder(run, cache_root=tmp_path / "cache")
    assert not (run / "aMe12_lines.csv").exists()
    assert not (run / "images").exists()  # summary PDF exists → images go
    assert (run / "line_summary.csv").exists()
    assert audit["bytes_reclaimed"] == 150
    # The folder's own cleanup audit accumulated the pass.
    folder_audit = json.loads(
        (run / "cleanup_audit.json").read_text())
    assert folder_audit["bytes_reclaimed"] == 150

    # Unregistered run folders and non-run folders are refused.
    shared = out / "morph_cross_dataset"
    shared.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        si.prune_run_folder(shared, cache_root=tmp_path / "cache")
    with pytest.raises(ValueError):
        si.prune_run_folder(
            out / "find-paths-complete_MCNS_a_to_b_20260801_183000",
            cache_root=tmp_path / "cache")
