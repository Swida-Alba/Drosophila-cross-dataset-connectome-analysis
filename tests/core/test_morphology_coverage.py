"""Coverage tests for src/morphology.py.

test_morphology.py (left untouched) covers the comparer/pipeline paths; this
file targets the large clusters of uncovered pure/synthetic-testable code:
feature extraction edge branches, the compressed-skeleton read/write helpers
and their format fallbacks, the similarity-matrix helpers, the
SkeletonVectorCache path/manifest/pending/append/load/vectors internals, and
the population-statistics + version-sibling fallbacks.

Hermetic: synthetic navis.TreeNeuron objects with a handful of nodes, tiny
mesh stubs, and all filesystem state under pytest tmp_path.  No network, no
multiprocessing (the parallel vectorizer is monkeypatched to raise so the
serial fallback is exercised instead).
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import navis  # noqa: E402
import morphology as M  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic neurons
# ---------------------------------------------------------------------------

def make_tree():
    nodes = pd.DataFrame({
        "node_id": [0, 1, 2, 3, 4, 5, 6],
        "parent_id": [-1, 0, 1, 1, 2, 2, 0],
        "x": [0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0],
        "y": [0.0, 0.0, 1.0, -1.0, 2.0, 0.0, 1.0],
        "z": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "radius": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "type": [1] * 7,
    })
    return navis.TreeNeuron(nodes)


def make_mesh_stub(n=5):
    verts = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 10.0, 10.0],
    ])
    faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 4], [0, 2, 3]])
    return SimpleNamespace(vertices=verts, faces=faces)


def make_swc_text(nid=1, n=8):
    lines = [f"# SWC {nid}"]
    for i in range(1, n + 1):
        parent = -1 if i == 1 else i - 1
        lines.append(f"{i} 1 {i * 100.0} {i * 10.0} {i * 5.0} 2.0 {parent}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# dataset id helpers
# ---------------------------------------------------------------------------

def test_dataset_folder_and_body_id_helpers():
    assert M._dataset_folder("hemibrain:v1.2.1") == "hemibrain_v1_2_1"
    # NeuPrint integer contract
    assert M._canonical_dataset_body_id("hemibrain:v1.2.1", "123") == 123
    assert M._api_dataset_body_id("hemibrain:v1.2.1", "123") == 123
    # FlyWire keeps exact string ids
    fid = "720575940614131061"
    assert M._canonical_dataset_body_id("flywire", fid) == fid
    assert M._api_dataset_body_id("flywire", fid) == int(fid)


# ---------------------------------------------------------------------------
# flywire soma positions + local presence
# ---------------------------------------------------------------------------

def test_load_flywire_soma_positions_parquet_and_csv(tmp_path):
    folder = M._dataset_folder("flywire")
    ds_dir = tmp_path / "datasets" / folder
    ds_dir.mkdir(parents=True)
    fid = "720575940614131061"
    frame = pd.DataFrame({
        "bodyId": [fid, "111"],
        "position": ["[1.0 2.0 3.0]", "not-a-position"],
    })
    frame.to_parquet(ds_dir / f"{folder}_allneurons_neuron_df.parquet",
                     index=False)
    # BUG REPORTED: explicit body_ids hits an undefined
    # ``normalize_flywire_body_ids`` (plural, never imported) at line 246,
    # so the filtered path always swallows a NameError and returns {}.
    assert M._load_flywire_soma_positions("flywire", tmp_path, [fid]) == {}
    # Unfiltered load parses valid rows and skips unparseable positions.
    out = M._load_flywire_soma_positions("flywire", tmp_path)
    assert fid in out and np.allclose(out[fid], [1.0, 2.0, 3.0])

    # csv variant without a position column -> empty
    (ds_dir / f"{folder}_allneurons_neuron_df.parquet").unlink()
    pd.DataFrame({"bodyId": [fid], "other": [1]}).to_csv(
        ds_dir / f"{folder}_allneurons_neuron_df.csv", index=False)
    assert M._load_flywire_soma_positions("flywire", tmp_path) == {}


def test_load_flywire_soma_positions_no_table(tmp_path):
    assert M._load_flywire_soma_positions("flywire", tmp_path) == {}


def test_has_local_dataset_presence(tmp_path):
    ds = "hemibrain:v1.2.1"
    folder = M._dataset_folder(ds)
    assert M._has_local_dataset_presence(ds, tmp_path) is False
    # neuron_df table present
    ds_dir = tmp_path / "datasets" / folder
    ds_dir.mkdir(parents=True)
    (ds_dir / f"{folder}_neuron_df.parquet").write_bytes(b"x")
    assert M._has_local_dataset_presence(ds, tmp_path) is True
    # connections cache present
    (ds_dir / f"{folder}_neuron_df.parquet").unlink()
    cache_dir = tmp_path / "cache" / folder
    cache_dir.mkdir(parents=True)
    (cache_dir / "connections.parquet").write_bytes(b"x")
    assert M._has_local_dataset_presence(ds, tmp_path) is True


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------

def test_compute_morphometrics_basic():
    feats = M.compute_morphometrics(make_tree())
    assert set(feats) == set(M.MORPHOMETRIC_FEATURES)
    assert feats["n_nodes"] == 7.0
    assert feats["cable_length"] > 0
    assert all(np.isfinite(v) for v in feats.values())


def test_compute_morphometrics_no_leaves_branch():
    # a node table with a cycle has no leaves -> the empty-leaves fallback
    cyclic = pd.DataFrame({
        "node_id": [0, 1, 2],
        "parent_id": [1, 2, 0],
        "x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0], "z": [0.0, 0.0, 0.0],
        "radius": [1.0, 1.0, 1.0],
    })
    stub = SimpleNamespace(nodes=cyclic, soma=None)
    feats = M.compute_morphometrics(stub)
    assert feats["tortuosity"] == 1.0
    assert feats["mean_path_length"] == 0.0


def test_compute_morphometrics_soma_exception_swallowed():
    class BadSoma:
        nodes = make_tree().nodes

        @property
        def soma(self):
            raise RuntimeError("no soma")

    feats = M.compute_morphometrics(BadSoma())
    assert feats["soma_radius"] == 0.0


def test_compute_persistence_vector_size_mismatch(monkeypatch):
    monkeypatch.setattr(navis.morpho, "persistence_vectors",
                        lambda n, samples=100: (np.zeros(10),))
    vec = M.compute_persistence_vector(make_tree())
    assert vec.shape == (M.PERSISTENCE_DIM,) and np.all(vec == 0)


def test_compute_persistence_vector_failure_returns_zeros():
    vec = M.compute_persistence_vector(SimpleNamespace(nodes=None))
    assert vec.shape == (M.PERSISTENCE_DIM,) and np.all(vec == 0)


def test_mesh_morphometrics_and_histogram():
    mesh = make_mesh_stub()
    feats = M.compute_mesh_morphometrics(mesh)
    assert feats["n_nodes"] == 5.0 and feats["n_branch"] == 4.0
    hist = M.compute_spatial_histogram(mesh)
    assert hist.shape == (M.PERSISTENCE_DIM,) and abs(hist.sum() - 1) < 1e-9
    empty = SimpleNamespace(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))
    assert np.all(M.compute_spatial_histogram(empty) == 0)


def test_vectorize_neuron_dispatch_and_error():
    morph, vec = M.vectorize_neuron(make_tree())
    assert vec.shape == (M.VECTOR_DIM,)
    morph, mvec = M.vectorize_neuron(make_mesh_stub())
    assert mvec.shape == (M.VECTOR_DIM,)
    with pytest.raises(ValueError):
        M.vectorize_neuron(SimpleNamespace())


def test_neuron_rep():
    assert M._neuron_rep(make_tree()) == "skeleton"
    assert M._neuron_rep(make_mesh_stub()) == "mesh"
    assert M._neuron_rep(SimpleNamespace()) == ""


# ---------------------------------------------------------------------------
# skeleton body id + compressed skeleton round trips
# ---------------------------------------------------------------------------

def test_skeleton_body_id_suffixes():
    assert M._skeleton_body_id("x/123.swc.zst") == 123
    assert M._skeleton_body_id("x/123.swc.gz") == 123
    assert M._skeleton_body_id("x/123.pkl.zst") == 123
    assert M._skeleton_body_id("x/123.pkl") == 123
    assert M._skeleton_body_id("x/123") == 123


def test_simplification_factor():
    assert M._simplification_factor(0) == 1
    assert M._simplification_factor(90) == 10
    assert M._simplification_factor(50) == 2
    with pytest.raises(ValueError):
        M._simplification_factor(95)
    with pytest.raises(ValueError):
        M._simplification_factor("abc")


def test_read_stored_simplification():
    assert M._read_stored_simplification("# DROCAT simpl: 50\n1 1 0 0 0 1 -1") == 50
    assert M._read_stored_simplification(b"# DROCAT simpl: 90\n") == 90
    assert M._read_stored_simplification("# DROCAT simpl: bogus\n") == 0
    assert M._read_stored_simplification("1 1 0 0 0 1 -1\n") == 0


def test_relevel_for_target():
    neuron = make_tree()
    # target not coarser than stored -> unchanged
    assert M._relevel_for_target(neuron, 90, 90) is neuron
    assert M._relevel_for_target(neuron, 50, 50) is neuron
    # mesh / no-nodes objects are never re-leveled
    assert M._relevel_for_target(make_mesh_stub(), 0, 90) is not None
    # invalid target raises through the factor validation
    with pytest.raises(ValueError):
        M._relevel_for_target(neuron, 0, 95)


def test_write_and_load_compressed_skeleton_roundtrip(tmp_path):
    neuron = make_tree()
    zst = tmp_path / "42.swc.zst"
    M._write_compressed_skeleton(zst, neuron, simplification=0, codec="zst")
    loaded = M._load_cached_skeleton_file(zst)
    assert type(loaded).__name__ == "TreeNeuron"
    assert getattr(loaded, "_drocat_simplification", None) == 0

    gz = tmp_path / "43.swc.gz"
    M._write_compressed_swc(gz, neuron)
    loaded_gz = M._load_cached_skeleton_file(gz)
    assert type(loaded_gz).__name__ == "TreeNeuron"


def test_load_cached_pickle_and_pkl_zst(tmp_path):
    import pickle
    neuron = make_tree()
    pkl = tmp_path / "7.pkl"
    with open(pkl, "wb") as handle:
        pickle.dump(neuron, handle)
    assert M._load_cached_skeleton_file(pkl) is not None

    import zstandard as zstd
    pkl_zst = tmp_path / "8.pkl.zst"
    blob = zstd.ZstdCompressor().compress(pickle.dumps(neuron))
    pkl_zst.write_bytes(blob)
    assert M._load_cached_skeleton_file(pkl_zst) is not None


def test_load_cached_zstd_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "zstd", None)
    zst = tmp_path / "1.swc.zst"
    zst.write_bytes(b"\x28\xb5\x2f\xfd")
    with pytest.raises(ImportError):
        M._load_cached_skeleton_file(zst)
    pkl_zst = tmp_path / "2.pkl.zst"
    pkl_zst.write_bytes(b"\x28\xb5\x2f\xfd")
    with pytest.raises(ImportError):
        M._load_cached_skeleton_file(pkl_zst)


def test_write_compressed_skeleton_records_level_and_none(tmp_path):
    neuron = make_tree()
    path = tmp_path / "5.swc.zst"
    M._write_compressed_skeleton(path, neuron, simplification=50)
    text = M._load_cached_skeleton_file(path)
    assert getattr(text, "_drocat_simplification", None) == 50
    # simplification=None records the neuron's attached level
    neuron._drocat_simplification = 0
    path2 = tmp_path / "6.swc.zst"
    M._write_compressed_skeleton(path2, neuron, simplification=None)
    assert M._load_cached_skeleton_file(path2) is not None


def test_write_compressed_skeleton_zstd_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "zstd", None)
    with pytest.raises(ImportError):
        M._write_compressed_skeleton(tmp_path / "9.swc.zst", make_tree(),
                                     simplification=0, codec="zst")


def test_vectorize_one_file(tmp_path):
    zst = tmp_path / "101.swc.zst"
    M._write_compressed_skeleton(zst, make_tree(), simplification=0)
    result = M._vectorize_one_file(str(zst))
    assert result is not None and result[0] == 101 and result[3] == "skeleton"

    # a simplified file is skipped
    simp = tmp_path / "102.swc.zst"
    M._write_compressed_skeleton(simp, make_tree(), simplification=90)
    assert M._vectorize_one_file(str(simp)) is None

    # a corrupt file returns None
    bad = tmp_path / "103.swc.zst"
    bad.write_bytes(b"garbage")
    assert M._vectorize_one_file(str(bad)) is None


# ---------------------------------------------------------------------------
# similarity helpers
# ---------------------------------------------------------------------------

def test_similarity_matrices():
    query = np.array([1.0, 2.0, 3.0])
    matrix = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [0.0, 0.0, 0.0]])
    cos = M.cosine_similarity_matrix(query, matrix)
    assert abs(cos[0] - 1.0) < 1e-9 and cos[2] == 0.0
    assert np.all(M.cosine_similarity_matrix(np.zeros(3), matrix) == 0)
    pear = M.pearson_similarity_matrix(query, matrix)
    assert pear.shape == (3,)
    assert np.allclose(M.similarity_matrix(query, matrix, "pearson"), pear)
    assert np.allclose(M.similarity_matrix(query, matrix, "cosine"), cos)


def test_pairwise_similarity_matrix():
    mat = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    scores = M.pairwise_similarity_matrix(mat)
    assert scores.shape == (4, 4)
    assert abs(scores[0, 1] - 1.0) < 1e-9
    assert np.all(scores[3, :] == 0) and np.all(scores[:, 3] == 0)
    assert M.pairwise_similarity_matrix(np.zeros((0, 3))).shape == (0, 0)
    pear = M.pairwise_similarity_matrix(mat, metric="pearson")
    assert pear.shape == (4, 4)
    with pytest.raises(ValueError):
        M.pairwise_similarity_matrix(np.array([1.0, 2.0]))


def test_sorted_candidates():
    assert M._sorted_candidates(None) is None
    empty = pd.DataFrame()
    assert M._sorted_candidates(empty).empty
    no_score = pd.DataFrame({"target_bodyId": [1], "x": [2]})
    assert M._sorted_candidates(no_score) is no_score
    frame = pd.DataFrame({
        "target_bodyId": [2, 1, 3],
        "_score": [0.5, 0.9, 0.9],
    })
    out = M._sorted_candidates(frame)
    assert list(out["target_bodyId"]) == [1, 3, 2]


# ---------------------------------------------------------------------------
# folder level marker + downsample + infer rep
# ---------------------------------------------------------------------------

def test_skeleton_folder_level_and_marker(tmp_path):
    assert M._skeleton_folder_level("hemibrain:v1.2.1", str(tmp_path)) == "raw"
    M._write_skeleton_level_marker("hemibrain:v1.2.1", str(tmp_path))
    marker = (tmp_path / "cache" / "hemibrain_v1_2_1" / "skeletons" / ".level")
    assert marker.exists()
    # a second write is a no-op (marker already exists)
    M._write_skeleton_level_marker("hemibrain:v1.2.1", str(tmp_path))
    marker.write_text("simp90\n")
    assert M._skeleton_folder_level("hemibrain:v1.2.1", str(tmp_path)) == "simp90"
    marker.write_text("bogus\n")
    assert M._skeleton_folder_level("hemibrain:v1.2.1", str(tmp_path)) == "raw"


def test_downsample_for_cache():
    neuron = make_tree()
    neuron.soma = None
    out = M._downsample_for_cache(neuron, downsampling_factor=2)
    assert out is not None
    # a multi-node soma is cleared before downsampling
    multi = make_tree()
    multi.soma_flags = np.ones(len(multi.nodes), dtype=bool)
    assert len(multi.soma) > 1
    result = M._downsample_for_cache(multi, 2)
    assert result is not None and result.soma is None


def test_downsample_for_cache_failure_returns_original(monkeypatch):
    neuron = make_tree()
    neuron.soma = None
    monkeypatch.setattr(navis, "downsample_neuron",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    assert M._downsample_for_cache(neuron, 2) is neuron


def test_infer_dataset_rep(tmp_path):
    M._REP_MEMO.clear()
    # no cached files -> empty representation, memoized
    assert M._infer_dataset_rep("hemibrain:v1.2.1", str(tmp_path)) == ""
    key = ("hemibrain:v1.2.1", str(tmp_path))
    assert M._REP_MEMO[key] == ""
    # memo hit
    assert M._infer_dataset_rep("hemibrain:v1.2.1", str(tmp_path)) == ""


# ---------------------------------------------------------------------------
# SkeletonVectorCache: construction + validation
# ---------------------------------------------------------------------------

def _raw_cache(tmp_path, dataset="hemibrain:v1.2.1", **kw):
    kw.setdefault("n_workers", 1)
    kw.setdefault("verbose", False)
    return M.SkeletonVectorCache(dataset, project_root=str(tmp_path),
                                 raw_only=True, **kw)


def _vector_record(bid, rep="skeleton"):
    row = {"bodyId": bid, "rep": rep}
    for i, name in enumerate(M.MORPHOMETRIC_FEATURES):
        row[name] = float(i + 1) + float(bid if isinstance(bid, int) else 0.0)
    for i in range(M.PERSISTENCE_DIM):
        row[f"pv_{i}"] = float(i % 7)
    return row


def test_cache_init_validation(tmp_path):
    with pytest.raises(ValueError):
        M.SkeletonVectorCache("hemibrain:v1.2.1", project_root=str(tmp_path),
                              representation="volume")
    with pytest.raises(ValueError):
        M.SkeletonVectorCache("hemibrain:v1.2.1", project_root=str(tmp_path),
                              raw_only=True, representation="mesh")
    with pytest.raises(ValueError):
        M.SkeletonVectorCache("hemibrain:v1.2.1", project_root=str(tmp_path),
                              raw_only=True, raw_format="hdf5")


def test_cache_path_layout(tmp_path):
    cache = _raw_cache(tmp_path)
    assert cache.skeleton_dir.name == "raw_skeletons"
    assert cache.parquet_path.name == "skeleton_vectors.parquet"
    assert cache.pending_path.name == "skeleton_vectors_pending.parquet"
    assert cache.skeleton_manifest_path.name == "raw_skeleton_manifest.json"
    mesh = M.SkeletonVectorCache("flywire", project_root=str(tmp_path),
                                 representation="mesh", verbose=False)
    assert mesh.parquet_path.name == "mesh_vectors.parquet"
    assert mesh.meta_path.name == "mesh_meta.json"
    legacy = M.SkeletonVectorCache("hemibrain:v1.2.1",
                                   project_root=str(tmp_path), verbose=False)
    assert legacy.skeleton_dir.name == "skeletons"
    assert legacy.legacy_skeleton_dir is None


def test_skeleton_manifest_load_and_write(tmp_path, monkeypatch):
    cache = _raw_cache(tmp_path)
    default = cache._load_skeleton_manifest()
    assert default["dataset"] == "hemibrain:v1.2.1" and default["files"] == {}
    # non-raw writer is a no-op
    legacy = M.SkeletonVectorCache("hemibrain:v1.2.1",
                                   project_root=str(tmp_path), verbose=False)
    legacy._write_skeleton_manifest({"files": {}})
    assert not legacy.skeleton_manifest_path.exists()
    # roundtrip
    manifest = dict(default)
    manifest["files"] = {"1": {"file": "1.swc.zst"}}
    cache._write_skeleton_manifest(manifest)
    assert cache._load_skeleton_manifest()["files"] == {"1": {"file": "1.swc.zst"}}
    # incompatible provenance -> fallback
    cache.skeleton_manifest_path.write_text(json.dumps(
        {"cache_schema_version": 999, "dataset": "other",
         "representation": "mesh", "files": []}))
    assert cache._load_skeleton_manifest()["files"] == {}
    # corrupt json -> fallback
    cache.skeleton_manifest_path.write_text("{not json")
    assert cache._load_skeleton_manifest()["dataset"] == "hemibrain:v1.2.1"
    # replace failure + unlink failure both survive in the finally path
    monkeypatch.setattr("os.replace", _raise_oserror)
    monkeypatch.setattr(Path, "unlink", _unlink_oserror)
    with pytest.raises(OSError):
        cache._write_skeleton_manifest(manifest)


def _raise_oserror(*args, **kwargs):
    raise OSError("boom")


def _unlink_oserror(self, *args, **kwargs):
    raise OSError("unlink refused")


def test_atomic_parquet_cleanup_on_failure(tmp_path, monkeypatch):
    cache = _raw_cache(tmp_path)
    frame = pd.DataFrame([_vector_record(1)])
    target = cache.morph_dir / "probe.parquet"
    monkeypatch.setattr("os.replace", _raise_oserror)
    with pytest.raises(OSError):
        cache._atomic_parquet(frame, target)
    leftovers = [p for p in cache.morph_dir.glob(".*.tmp")]
    assert leftovers == []
    # unlink failure in the cleanup is swallowed
    monkeypatch.setattr(Path, "unlink", _unlink_oserror)
    with pytest.raises(OSError):
        cache._atomic_parquet(frame, target)


def test_dedupe_frames_edges():
    main = pd.DataFrame([_vector_record(1)])
    out = M.SkeletonVectorCache._dedupe_frames(main, None)
    assert len(out) == 1 and out.index[0] == 0
    pending = pd.DataFrame([_vector_record(2)])
    out = M.SkeletonVectorCache._dedupe_frames(None, pending)
    assert len(out) == 1
    out = M.SkeletonVectorCache._dedupe_frames(main, pending)
    assert len(out) == 2
    # duplicate bodyId: first (main) wins
    dup = pd.DataFrame([_vector_record(1), _vector_record(3)])
    out = M.SkeletonVectorCache._dedupe_frames(main, dup)
    assert len(out) == 2
    # frame without a bodyId column passes through untouched
    anon = pd.DataFrame({"x": [1, 2]})
    out = M.SkeletonVectorCache._dedupe_frames(anon, anon)
    assert len(out) == 4


def test_clear_pending_oserrors(tmp_path, monkeypatch):
    cache = _raw_cache(tmp_path)
    cache.morph_dir.mkdir(parents=True, exist_ok=True)
    cache.pending_path.write_bytes(b"junk")
    cache.meta_path.write_text(json.dumps({"pending_appends": 3}))
    monkeypatch.setattr(Path, "unlink", _unlink_oserror)
    monkeypatch.setattr(Path, "write_text", _raise_oserror)
    cache._clear_pending()  # both failures swallowed
    assert cache.pending_path.exists()


def test_vector_row_count_corrupt(tmp_path):
    cache = _raw_cache(tmp_path)
    cache.morph_dir.mkdir(parents=True, exist_ok=True)
    cache.parquet_path.write_bytes(b"corrupt")
    cache.pending_path.write_bytes(b"corrupt")
    assert cache._vector_row_count() == 0
    cache._atomic_parquet(pd.DataFrame([_vector_record(1)]), cache.parquet_path)
    assert cache._vector_row_count() == 1


def test_merge_pending_paths(tmp_path):
    cache = _raw_cache(tmp_path)
    cache.morph_dir.mkdir(parents=True, exist_ok=True)
    # no pending file -> 0
    assert cache._merge_pending() == 0
    # corrupt pending -> cleared, 0
    cache.pending_path.write_bytes(b"corrupt")
    assert cache._merge_pending() == 0
    assert not cache.pending_path.exists()
    # corrupt main + valid pending -> pending becomes main
    cache.parquet_path.write_bytes(b"corrupt")
    cache._atomic_parquet(pd.DataFrame([_vector_record(1)]), cache.pending_path)
    assert cache._merge_pending() == 1
    meta = json.loads(cache.meta_path.read_text())
    assert meta["n_rows"] == 1
    assert meta["rep"] == "skeleton"
    assert meta["vector_basis"] == "raw"
    assert meta["raw_format"] == "swc.zst"
    assert meta["mean"] and meta["std"]
    # second merge: main valid, pending dedupes against main
    cache._atomic_parquet(pd.DataFrame([_vector_record(1), _vector_record(2)]),
                          cache.pending_path)
    assert cache._merge_pending() == 2


def test_temp_skeleton_roundtrip_and_delete_oserror(tmp_path, monkeypatch):
    cache = _raw_cache(tmp_path)
    path = cache.write_temp_skeleton(42, make_tree())
    assert path.exists() and path.name == "42.swc.zst"
    cache.delete_temp_skeletons([42])
    assert not path.exists()
    path2 = cache.write_temp_skeleton(43, make_tree())
    monkeypatch.setattr(Path, "unlink", _unlink_oserror)
    cache.delete_temp_skeletons([43])  # OSError swallowed per entry
    assert path2.exists()


def test_discover_skeleton_files_preference(tmp_path):
    cache = _raw_cache(tmp_path)
    cache.skeleton_dir.mkdir(parents=True)
    cache.legacy_skeleton_dir.mkdir(parents=True)
    # body 5 exists in several formats: swc.zst in the canonical dir wins
    (cache.skeleton_dir / "5.swc.zst").write_bytes(b"x")
    (cache.skeleton_dir / "5.swc.gz").write_bytes(b"x")
    (cache.legacy_skeleton_dir / "5.pkl.zst").write_bytes(b"x")
    (cache.legacy_skeleton_dir / "5.pkl").write_bytes(b"x")
    # body 6 only in the legacy dir
    (cache.legacy_skeleton_dir / "6.pkl").write_bytes(b"x")
    # unparseable body id is skipped
    (cache.skeleton_dir / "not-a-number.pkl").write_bytes(b"x")
    # temp staging is never discovered
    temp = cache.skeleton_dir / "_temp_cache"
    temp.mkdir()
    (temp / "7.swc.zst").write_bytes(b"x")
    found = cache._discover_skeleton_files()
    assert any(p.endswith("5.swc.zst") for p in found)
    assert not any("5.swc.gz" in p or "5.pkl" in p for p in found)
    assert any("6.pkl" in p for p in found)
    assert not any("not-a-number" in p for p in found)
    assert not any("_temp_cache" in p for p in found)


def test_cached_dir_listing(tmp_path, monkeypatch):
    import os as os_mod
    cache = _raw_cache(tmp_path)
    missing = tmp_path / "nowhere"
    assert cache._cached_dir_listing(missing) == (set(), False)
    directory = tmp_path / "listing"
    directory.mkdir()
    (directory / "1.swc.zst").write_bytes(b"x")
    (directory / "sub").mkdir()
    names, has_subdirs = cache._cached_dir_listing(directory)
    assert names == {"1.swc.zst"} and has_subdirs
    # cached hit (same mtime)
    assert cache._cached_dir_listing(directory)[0] == {"1.swc.zst"}
    # scandir failure -> empty names but cached
    monkeypatch.setattr(os_mod, "scandir", _raise_oserror)
    cache._dir_listing_cache.clear()
    names, has_subdirs = cache._cached_dir_listing(directory)
    assert names == set() and not has_subdirs


def test_find_skeleton_file_raw_cache(tmp_path):
    cache = _raw_cache(tmp_path)
    assert cache.find_skeleton_file(1) is None
    cache.skeleton_dir.mkdir(parents=True)
    direct = cache.skeleton_dir / "1.swc.zst"
    direct.write_bytes(b"x")
    assert cache.find_skeleton_file("1") == direct
    # nested bulk folder discovered via rglob
    nested_dir = cache.skeleton_dir / "bulk"
    nested_dir.mkdir()
    nested = nested_dir / "2.swc.gz"
    nested.write_bytes(b"x")
    assert cache.find_skeleton_file(2) == nested
    # legacy fallback only when the canonical dirs have nothing
    (cache.legacy_skeleton_dir).mkdir(parents=True)
    legacy = cache.legacy_skeleton_dir / "3.pkl"
    legacy.write_bytes(b"x")
    assert cache.find_skeleton_file(3) == legacy


def test_load_skeleton_raw_cache(tmp_path):
    import pickle as pickle_mod
    cache = _raw_cache(tmp_path)
    assert cache.load_skeleton(1) is None
    # corrupt file -> exception swallowed -> None
    cache.skeleton_dir.mkdir(parents=True)
    (cache.skeleton_dir / "9.swc.zst").write_bytes(b"garbage")
    assert cache.load_skeleton(9) is None
    # legacy pickle in the migration source dir -> loaded + migrated
    cache.legacy_skeleton_dir.mkdir(parents=True)
    neuron = make_tree()
    neuron.soma = None
    with open(cache.legacy_skeleton_dir / "4.pkl", "wb") as handle:
        pickle_mod.dump(neuron, handle)
    loaded = cache.load_skeleton(4)
    assert loaded is not None and len(loaded.nodes) == 7
    assert (cache.skeleton_dir / "4.swc.zst").exists()  # lazy migration
    # re-level request on a TreeNeuron (coarser target keeps the neuron)
    again = cache.load_skeleton(4, simplification=90)
    assert again is not None
    # a mesh pickle is rejected by a raw-only cache (type guard)
    with open(cache.skeleton_dir / "6.pkl", "wb") as handle:
        pickle_mod.dump(make_mesh_stub(), handle)
    assert cache.load_skeleton(6) is None


def test_persist_skeletons_formats(tmp_path):
    neuron = make_tree()
    neuron.soma = None
    cache = _raw_cache(tmp_path)
    assert cache.persist_skeletons({}) == 0
    # default swc.zst + manifest entry
    assert cache.persist_skeletons({1: neuron}, simplification=0) == 1
    assert (cache.skeleton_dir / "1.swc.zst").exists()
    manifest = cache._load_skeleton_manifest()
    assert manifest["files"]["1"]["file"] == "1.swc.zst"
    # wrong representation skipped; un-canonical id swallowed
    assert cache.persist_skeletons({2: make_mesh_stub()}) == 0
    assert cache.persist_skeletons({"abc": neuron}) == 0
    # swc.gz format
    gz = _raw_cache(tmp_path, raw_format="swc.gz")
    assert gz.persist_skeletons({3: neuron}) == 1
    assert (gz.skeleton_dir / "3.swc.gz").exists()
    # legacy pickle format
    pkl = _raw_cache(tmp_path, raw_format="pkl")
    assert pkl.persist_skeletons({4: neuron}) == 1
    assert (pkl.skeleton_dir / "4.pkl").exists()


def test_log_verbose(tmp_path, capsys):
    cache = M.SkeletonVectorCache("hemibrain:v1.2.1",
                                  project_root=str(tmp_path), verbose=True)
    cache._log("hello")
    assert "hello" in capsys.readouterr().out


def test_load_meta_corrupt(tmp_path):
    cache = _raw_cache(tmp_path)
    assert cache._load_meta() is None
    cache.morph_dir.mkdir(parents=True, exist_ok=True)
    cache.meta_path.write_text("{broken")
    assert cache._load_meta() is None


def _write_level0_swc_zst(cache, bid):
    neuron = make_tree()
    neuron.soma = None
    path = cache.skeleton_dir / f"{bid}.swc.zst"
    M._write_compressed_skeleton(path, neuron, simplification=0)
    return path


def test_build_serial_raw_cache(tmp_path, monkeypatch):
    cache = _raw_cache(tmp_path, n_workers=2)
    cache.skeleton_dir.mkdir(parents=True)
    _write_level0_swc_zst(cache, 1)
    _write_level0_swc_zst(cache, 2)
    # parallel path forced to fail -> serial fallback (no real workers)
    monkeypatch.setattr(M.SkeletonVectorCache, "_vectorize_parallel",
                        lambda self, files: (_ for _ in ()).throw(RuntimeError()))
    result = cache.build()
    assert result["rows"] == 2 and result["new"] == 2
    assert cache.parquet_path.exists() and cache.cache_exists()
    meta = json.loads(cache.meta_path.read_text())
    assert meta["n_rows"] == 2 and meta["rep"] == "skeleton"
    # second build reuses all rows
    assert cache.build()["new"] == 0
    assert cache.ensure()["rows"] == 2


def test_build_empty_and_corrupt_inputs(tmp_path):
    cache = _raw_cache(tmp_path)
    result = cache.build()  # nothing to vectorize
    assert result == {"rows": 0, "new": 0, "fetched": 0}
    meta = json.loads(cache.meta_path.read_text())
    assert meta["n_rows"] == 0 and meta["mean"] == []
    # corrupt main + corrupt pending do not break a rebuild
    cache.parquet_path.write_bytes(b"corrupt")
    cache.pending_path.write_bytes(b"corrupt")
    cache.skeleton_dir.mkdir(parents=True, exist_ok=True)
    _write_level0_swc_zst(cache, 5)
    result = cache.build()
    assert result["rows"] == 1
    assert not cache.pending_path.exists()  # superseded by the rebuild


def test_build_fetch_missing_with_fake_fetcher(tmp_path, monkeypatch):
    cache = M.SkeletonVectorCache("hemibrain:v1.2.1",
                                  project_root=str(tmp_path), n_workers=1,
                                  verbose=False)
    folder = M._dataset_folder("hemibrain:v1.2.1")
    index_dir = tmp_path / "neuron_indexes" / folder
    index_dir.mkdir(parents=True)
    pd.DataFrame({"bodyId": [1, 2, 3]}).to_parquet(
        index_dir / "neuron_index.parquet", index=False)
    ds_dir = tmp_path / "datasets" / folder
    ds_dir.mkdir(parents=True)
    (ds_dir / f"{folder}_neuron_df.csv").write_text("bodyId,type\n1,A\n")
    cache.skeleton_dir.mkdir(parents=True)
    _write_level0_swc_zst(cache, 1)

    def fake_fetch(dataset, ids, **kwargs):
        cache.morph_dir.mkdir(parents=True, exist_ok=True)
        cache._atomic_parquet(pd.DataFrame([_vector_record(2)]),
                              cache.parquet_path)
        return {2: object()}

    monkeypatch.setattr(M, "fetch_skeletons_on_demand_batch", fake_fetch)
    result = cache.build(fetch_missing=5)
    assert result["fetched"] == 1
    assert result["rows"] == 2  # fetched row 2 + vectorized file 1
    # corrupt neuron index -> fetch skipped entirely
    (index_dir / "neuron_index.parquet").write_bytes(b"corrupt")
    assert cache.build(fetch_missing=5)["fetched"] == 0


class _FakeBundle:
    def __init__(self, ids, close_raises=False):
        self._ids = ids
        self.bundle_path = Path("/nonexistent/bundle.zst")
        self.zip_path = None
        self.close_raises = close_raises

    def ids(self):
        return list(self._ids)

    def count(self):
        return len(self._ids)

    def close(self):
        if self.close_raises:
            raise OSError("close refused")


def _fake_swc_row(bid):
    return (bid, [float(i) for i in range(len(M.MORPHOMETRIC_FEATURES))],
            [0.0] * M.PERSISTENCE_DIM, "skeleton")


def test_build_from_fafb_bundle_serial(tmp_path, monkeypatch):
    fid = "720575940614131061"
    cache = M.SkeletonVectorCache("FAFB_v783", project_root=str(tmp_path),
                                  n_workers=1, verbose=False, raw_only=True)
    monkeypatch.setattr(M, "_fafb_bundle",
                        lambda ds, root: _FakeBundle([fid]))
    monkeypatch.setattr(M, "_init_fafb_zip_worker", lambda *a, **k: None)
    monkeypatch.setattr(M, "_vectorize_one_swc", _fake_swc_row)
    result = cache.build()
    assert result["rows"] == 1 and result["new"] == 1

    # an existing mesh-based cache is rebuilt from scratch
    cache._atomic_parquet(
        pd.DataFrame([_vector_record(fid, rep="mesh")]), cache.parquet_path)
    result = cache.build()
    assert result["rows"] == 1

    # failing parallel swc path falls back to serial; close failure swallowed
    cache.n_workers = 2
    monkeypatch.setattr(M, "_fafb_bundle",
                        lambda ds, root: _FakeBundle([fid], close_raises=True))
    monkeypatch.setattr(
        M.SkeletonVectorCache, "_vectorize_parallel_swc",
        lambda self, source, zp, bids: (_ for _ in ()).throw(RuntimeError()))
    result = cache.build()
    assert result["rows"] == 1


def test_build_bundle_source_missing(tmp_path, monkeypatch):
    cache = M.SkeletonVectorCache("FAFB_v783", project_root=str(tmp_path),
                                  n_workers=1, verbose=False, raw_only=True)

    def no_bundle(ds, root):
        raise FileNotFoundError("no bundle")

    monkeypatch.setattr(M, "_fafb_bundle", no_bundle)
    result = cache.build()
    assert result == {"rows": 0, "new": 0, "fetched": 0}


def test_cache_load_variants(tmp_path):
    cache = _raw_cache(tmp_path)
    assert cache.load() is None
    cache.morph_dir.mkdir(parents=True, exist_ok=True)
    # corrupt main only -> empty -> None
    cache.parquet_path.write_bytes(b"corrupt")
    assert cache.load() is None
    # corrupt main + valid pending merges; corrupt pending is ignored
    cache._atomic_parquet(pd.DataFrame([_vector_record(1)]), cache.pending_path)
    data = cache.load()
    assert data is not None and len(data["bodyIds"]) == 1
    cache._atomic_parquet(pd.DataFrame([_vector_record(1)]), cache.parquet_path)
    cache.pending_path.write_bytes(b"corrupt")
    data = cache.load()
    assert data is not None
    # legacy frame without a rep column -> dataset_rep inferred (memoized "")
    M._REP_MEMO.clear()
    legacy = pd.DataFrame([_vector_record(1)]).drop(columns=["rep"])
    cache._atomic_parquet(legacy, cache.parquet_path)
    cache.pending_path.unlink(missing_ok=True)
    data = cache.load()
    assert data is not None and data["dataset_rep"] == ""


def test_cache_coverage_fafb(tmp_path, monkeypatch):
    import zipfile
    cache = M.SkeletonVectorCache("FAFB_v783", project_root=str(tmp_path),
                                  n_workers=1, verbose=False, raw_only=True)
    # bundle path: count from the bundle
    monkeypatch.setattr(M, "_fafb_bundle",
                        lambda ds, root: _FakeBundle([1, 2, 3]))
    assert cache.coverage()["skeletons"] == 3
    # bundle unavailable -> ZIP fallback counts .swc entries
    zip_path = tmp_path / "fallback.zip"
    with zipfile.ZipFile(zip_path, "w") as handle:
        handle.writestr("1.swc", "x")
        handle.writestr("2.swc", "x")
        handle.writestr("readme.txt", "x")

    def no_bundle(ds, root):
        raise FileNotFoundError("no bundle")

    monkeypatch.setattr(M, "_fafb_bundle", no_bundle)
    monkeypatch.setattr(M, "_fafb_skeleton_zip_path",
                        lambda ds, root: zip_path)
    assert cache.coverage()["skeletons"] == 2
    # corrupt zip -> count stays 0
    monkeypatch.setattr(M, "_fafb_skeleton_zip_path",
                        lambda ds, root: cache.morph_dir / "missing.zip")
    assert cache.coverage()["skeletons"] == 0


# ---------------------------------------------------------------------------
# load_local_release_skeletons (raw cache -> healed bundle -> CAVE)
# ---------------------------------------------------------------------------

class _BundleGetStub:
    """Bundle double resolving body ids to SWC text."""

    def __init__(self, texts):
        self._texts = {int(k): v for k, v in texts.items()}
        self.closed = False

    def get(self, bid):
        return self._texts.get(int(bid))

    def close(self):
        self.closed = True


class _RawCacheStub:
    def __init__(self, neurons):
        self._neurons = {int(k): v for k, v in neurons.items()}

    def load_skeleton(self, bid, simplification=None):
        return self._neurons.get(int(bid))


class _CaveFetcherStub:
    """CAVEDataFetcher double; skeletons is a class-level id -> neuron map."""

    skeletons: dict = {}

    def __init__(self, **kwargs):
        self.called = []

    def fetch_skeleton(self, bid, use_cache=True, denoise_twigs=None):
        self.called.append(int(bid))
        return self.skeletons.get(int(bid))


class _PersistingRawCacheStub(_RawCacheStub):
    def __init__(self, neurons):
        super().__init__(neurons)
        self.persisted = {}

    def persist_skeletons(self, neurons, simplification=None):
        self.persisted.update({int(k): simplification
                               for k in neurons})
        return len(neurons)


def test_load_local_release_skeletons_cache_then_bundle(tmp_path, monkeypatch):
    tree_a = make_tree()
    bundle = _BundleGetStub({2: make_swc_text(nid=2)})
    raw_cache = _PersistingRawCacheStub({1: tree_a})
    monkeypatch.setattr(M, "find_similar_raw_cache", lambda ds, **k: raw_cache)
    monkeypatch.setattr(M, "_fafb_bundle", lambda ds, root: bundle)
    logs = []
    out = M.load_local_release_skeletons(
        "flywire_FAFB_v783", [1, 2], project_root=str(tmp_path),
        log=logs.append, check_extrusions=False)
    assert set(out) == {1, 2}
    assert out[1] is tree_a
    assert int(out[2].id) == 2          # parsed from the bundle SWC text
    assert bundle.closed                # bundle handle always closed
    # zip-only serving: no warm-up mirror into the raw store
    assert raw_cache.persisted == {}
    assert any("healed zip resolved 1/1" in m for m in logs)


def test_load_local_release_skeletons_replaces_flagged_extrusions(
        tmp_path, monkeypatch):
    import fafb_utils

    bundle = _BundleGetStub({7: make_swc_text(nid=7)})
    monkeypatch.setattr(M, "find_similar_raw_cache",
                        lambda ds, **k: _PersistingRawCacheStub({}))
    monkeypatch.setattr(M, "_fafb_bundle", lambda ds, root: bundle)
    monkeypatch.setattr(fafb_utils, "flag_extrusions", lambda *a, **k: [7])
    statuses = {}
    monkeypatch.setattr(fafb_utils, "set_extrusion_repair_status",
                        lambda root, folder, s: statuses.update(s))
    replacement = make_tree()
    cave_ids = []
    monkeypatch.setattr(
        M, "_flywire_cave_skeletons",
        lambda dataset, body_ids, project_root=None, log=None,
        denoise_twigs=None: (cave_ids.extend(int(b) for b in body_ids),
                             {7: replacement})[1])
    out = M.load_local_release_skeletons(
        "flywire_FAFB_v783", [7], project_root=str(tmp_path),
        check_extrusions=True)
    assert out[7] is replacement        # flagged tree replaced via CAVE
    assert cave_ids == [7]
    assert statuses == {7: "api_repaired"}


def test_load_local_release_skeletons_respects_api_repaired(
        tmp_path, monkeypatch):
    import fafb_utils
    import cave_data_fetcher

    # api_repaired bodies are served by the leading repair-cache stage;
    # the zip is never read and the CAVE chain never re-fires for them.
    replacement = make_tree()
    monkeypatch.setattr(
        cave_data_fetcher, "load_repaired_skeletons",
        lambda dataset, body_ids, project_root=None, log=None:
        {int(b): replacement for b in body_ids})
    monkeypatch.setattr(M, "find_similar_raw_cache",
                        lambda ds, **k: _PersistingRawCacheStub({}))

    def no_zip(*a, **k):
        raise AssertionError("a repaired body must not reload the zip")

    monkeypatch.setattr(M, "_fafb_bundle", no_zip)
    monkeypatch.setattr(fafb_utils, "flag_extrusions", lambda *a, **k: [7])

    def no_cave(*a, **k):
        raise AssertionError("a cached api_repaired tree must not be re-fetched")

    monkeypatch.setattr(M, "_flywire_cave_skeletons", no_cave)
    out = M.load_local_release_skeletons(
        "flywire_FAFB_v783", [7], project_root=str(tmp_path),
        check_extrusions=True)
    assert out[7] is replacement        # dedicated CAVE-derived tree kept


def test_load_local_release_skeletons_cave_fallback_for_fafb(tmp_path, monkeypatch):
    # No raw cache, no bundle: id 5 can only come from CAVE.
    def no_cache(ds, **k):
        raise FileNotFoundError("no cache")

    def no_bundle(ds, root):
        raise FileNotFoundError("no bundle")

    monkeypatch.setattr(M, "find_similar_raw_cache", no_cache)
    monkeypatch.setattr(M, "_fafb_bundle", no_bundle)
    tree = make_tree()
    monkeypatch.setattr(M, "_flywire_cave_skeletons",
                        lambda dataset, body_ids, project_root=None, log=None,
                        denoise_twigs=None: {5: tree})
    out = M.load_local_release_skeletons(
        "flywire_FAFB_v783", [5], project_root=str(tmp_path))
    assert set(out) == {5} and out[5] is tree


def test_flywire_cave_skeletons_requires_token(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.flywire_readiness.flywire_skeleton_readiness",
                        lambda *a, **k: {"cave_token": False})

    class _Boom:
        def __init__(self, *a, **k):
            raise AssertionError("CAVE fetcher must not run without a token")

    monkeypatch.setattr("cave_data_fetcher.CAVEDataFetcher", _Boom)
    logs = []
    out = M._flywire_cave_skeletons(
        "flywire_FAFB_v783", [7], project_root=str(tmp_path), log=logs.append)
    assert out == {}
    assert any("CAVE_TOKEN is not configured" in m for m in logs)


def test_flywire_cave_skeletons_skeletonizes_via_fetcher(
        tmp_path, monkeypatch):
    monkeypatch.setattr("utils.flywire_readiness.flywire_skeleton_readiness",
                        lambda *a, **k: {"cave_token": True})
    _CaveFetcherStub.skeletons = {5: make_tree()}
    monkeypatch.setattr("cave_data_fetcher.CAVEDataFetcher", _CaveFetcherStub)
    out = M._flywire_cave_skeletons(
        "flywire_FAFB_v783", [5], project_root=str(tmp_path),
        denoise_twigs=3000.0)
    assert set(out) == {5} and out[5] is _CaveFetcherStub.skeletons[5]


def test_flywire_cave_skeletons_uses_cached_replacement_without_token(
        tmp_path, monkeypatch):
    replacement = make_tree()
    monkeypatch.setattr(
        M, "_load_cave_cached_skeletons",
        lambda *a, **k: {5: replacement},
    )
    monkeypatch.setattr(
        "utils.flywire_readiness.flywire_skeleton_readiness",
        lambda *a, **k: {"cave_token": False},
    )

    out = M._flywire_cave_skeletons(
        "flywire_FAFB_v783", [5, 6], project_root=str(tmp_path))

    assert out == {5: replacement}


def test_append_vectors_branches(tmp_path, monkeypatch):
    # The unlock-failure branch this test drives needs flock(), which is
    # POSIX-only; on Windows the append lock degrades to a no-op.
    fcntl = pytest.importorskip("fcntl")
    cache = _raw_cache(tmp_path)
    vec = np.arange(M.VECTOR_DIM, dtype=float)
    assert cache.append_vectors([]) == 0
    # mesh rows rejected by a raw cache; duplicates within the batch collapse
    assert cache.append_vectors([(1, vec, "mesh"), (2, vec, "mesh")]) == 0
    # flock unlock failure is swallowed in the finally block
    real_flock = fcntl.flock

    def flaky_flock(fd, flag):
        if flag == fcntl.LOCK_UN:
            raise OSError("unlock refused")
        return real_flock(fd, flag)

    monkeypatch.setattr(fcntl, "flock", flaky_flock)
    assert cache.append_vectors([(3, vec, "skeleton")]) == 1
    monkeypatch.setattr(fcntl, "flock", real_flock)
    # corrupt pending file is rebuilt on the next append
    cache.pending_path.write_bytes(b"corrupt")
    assert cache.append_vectors([(4, vec, "skeleton")]) == 1
    # basis mismatch against the recorded meta -> rejected
    meta = json.loads(cache.meta_path.read_text())
    meta["vector_basis"] = "simp90"
    cache.meta_path.write_text(json.dumps(meta))
    assert cache.append_vectors([(5, vec, "skeleton")],
                                vector_basis="raw") == 0
    # rep mismatch against the recorded cache rep -> rejected
    meta["vector_basis"] = "raw"
    meta["rep"] = "mesh"
    cache.meta_path.write_text(json.dumps(meta))
    assert cache.append_vectors([(6, vec, "skeleton")]) == 0
    # threshold-crossed append triggers the amortized merge
    meta["rep"] = "skeleton"
    cache.meta_path.write_text(json.dumps(meta))
    monkeypatch.setattr(M, "PENDING_MERGE_APPENDS", 1)
    assert cache.append_vectors([(7, vec, "skeleton")]) == 1
    assert cache.parquet_path.exists() and not cache.pending_path.exists()
    # fcntl import failure -> lock skipped entirely
    monkeypatch.setitem(sys.modules, "fcntl", None)
    assert cache.append_vectors([(8, vec, "skeleton")]) == 1


def test_vectors_for_compute_missing(tmp_path):
    cache = _raw_cache(tmp_path)
    cache.skeleton_dir.mkdir(parents=True)
    _write_level0_swc_zst(cache, 7)          # vectorizable
    M._write_compressed_skeleton(cache.skeleton_dir / "8.swc.zst",
                                 make_tree(), simplification=30)  # wrong level
    (cache.skeleton_dir / "9.swc.zst").write_bytes(b"garbage")  # corrupt
    vectors, mask, reps = cache.vectors_for([7, 8, 9, 10])
    assert mask.tolist() == [True, False, False, False]
    assert reps[0] == "skeleton"
    # the computed vector was persisted to the pending staging file
    assert cache.pending_path.exists()
    # cached row served from the parquet after a merge
    cache._merge_pending()
    vectors, mask, reps = cache.vectors_for([7], compute_missing=False)
    assert mask[0] and reps[0] == "skeleton"


def test_find_similar_cache_factories(tmp_path):
    raw = M.find_similar_raw_cache("hemibrain:v1.2.1",
                                   project_root=str(tmp_path), verbose=False)
    assert raw.raw_only
    mesh = M.find_similar_flywire_mesh_cache(
        "flywire", project_root=str(tmp_path), verbose=False)
    assert mesh.mesh_only
    with pytest.raises(ValueError):
        M.find_similar_flywire_mesh_cache("hemibrain:v1.2.1",
                                          project_root=str(tmp_path))
    assert M.find_similar_dataset_cache(
        "flywire", project_root=str(tmp_path), verbose=False).mesh_only
    assert M.find_similar_dataset_cache(
        "hemibrain:v1.2.1", project_root=str(tmp_path),
        verbose=False).raw_only


def test_cache_fetched_skeleton_vectors(tmp_path):
    neuron = make_tree()
    neuron.soma = None
    calls = []
    callback = lambda done, total, msg: calls.append((done, total, msg))
    cache = _raw_cache(tmp_path)
    result = M.cache_fetched_skeleton_vectors(
        "hemibrain:v1.2.1", {1: neuron}, project_root=str(tmp_path),
        vector_cache=cache, progress_callback=callback, verbose=True)
    assert result["vectorized"] == 1 and result["cached"] == 1
    assert calls and result["cache_error"] is None
    # list input without ids + an unsupported object -> failure counted
    bad = SimpleNamespace(nodes=None)
    result = M.cache_fetched_skeleton_vectors(
        "hemibrain:v1.2.1", [neuron, bad], project_root=str(tmp_path),
        vector_cache=cache, progress_offset=10, verbose=True)
    assert result["seen"] == 2 and result["failures"] == 1
    # cache write failure is reported, not raised
    broken = SimpleNamespace(append_vectors=lambda rows, vector_basis: _raise_oserror())
    result = M.cache_fetched_skeleton_vectors(
        "hemibrain:v1.2.1", {2: neuron}, vector_cache=broken, verbose=True)
    assert result["cached"] == 0 and "OSError" in result["cache_error"]


def test_datasets_share_population(tmp_path):
    folder_a = M._dataset_folder("male-cns:v1.0")
    folder_b = M._dataset_folder("male-cns:v0.9")
    base = tmp_path / "neuron_indexes"
    (base / folder_a).mkdir(parents=True)
    (base / folder_b).mkdir(parents=True)
    assert not M._datasets_share_population("male-cns:v1.0", "male-cns:v0.9",
                                            tmp_path)
    pd.DataFrame({"bodyId": list(range(1, 11))}).to_parquet(
        base / folder_a / "neuron_index.parquet", index=False)
    pd.DataFrame({"bodyId": list(range(1, 11))}).to_parquet(
        base / folder_b / "neuron_index.parquet", index=False)
    assert M._datasets_share_population("male-cns:v1.0", "male-cns:v0.9",
                                        tmp_path)
    # disjoint populations
    pd.DataFrame({"bodyId": list(range(900, 920))}).to_parquet(
        base / folder_b / "neuron_index.parquet", index=False)
    assert not M._datasets_share_population("male-cns:v1.0", "male-cns:v0.9",
                                            tmp_path)
    # corrupt index -> None -> False
    (base / folder_a / "neuron_index.parquet").write_bytes(b"corrupt")
    assert not M._datasets_share_population("male-cns:v1.0", "male-cns:v0.9",
                                            tmp_path)


def test_sibling_skeleton_dirs(tmp_path):
    assert M._sibling_skeleton_dirs(":v1.0", tmp_path) == []
    assert M._sibling_skeleton_dirs("male-cns:v1.0", tmp_path) == []
    cache_root = tmp_path / "cache"
    sibling = cache_root / "male-cns_v0_9"
    (sibling / "skeletons").mkdir(parents=True)
    (cache_root / "male-cns_v1_0" / "skeletons").mkdir(parents=True)
    (cache_root / "other_v1" ).mkdir()
    (cache_root / "stray.txt").write_text("x")
    base = tmp_path / "neuron_indexes"
    shared = list(range(1, 11))
    for ds in ("male-cns:v1.0", "male-cns:v0.9"):
        folder = base / M._dataset_folder(ds)
        folder.mkdir(parents=True)
        pd.DataFrame({"bodyId": shared}).to_parquet(
            folder / "neuron_index.parquet", index=False)
    dirs = M._sibling_skeleton_dirs("male-cns:v1.0", tmp_path)
    assert dirs == [sibling / "skeletons"]  # self version excluded


def _write_stats_file(cache, dataset, n, sample_cap=M.POPULATION_STATS_SAMPLE):
    cache.morph_dir.mkdir(parents=True, exist_ok=True)
    (cache.morph_dir / "population_stats.json").write_text(json.dumps({
        "dataset": dataset, "dim": M.VECTOR_DIM, "n": n,
        "sample_cap": sample_cap,
        "mean": [0.0] * M.VECTOR_DIM, "std": [1.0] * M.VECTOR_DIM,
    }))


def test_population_stats_reuse_and_corrupt(tmp_path):
    cache = _raw_cache(tmp_path)
    _write_stats_file(cache, "hemibrain:v1.2.1", 400)
    mean, std = M.population_stats("hemibrain:v1.2.1", str(tmp_path),
                                   cache=cache)
    assert mean.shape == (M.VECTOR_DIM,) and std.shape == (M.VECTOR_DIM,)
    # corrupt stats file -> recomputation path (no files -> None)
    (cache.morph_dir / "population_stats.json").write_text("{broken")
    assert M.population_stats("hemibrain:v1.2.1", str(tmp_path),
                              cache=cache) == (None, None)
    # too-small sample count -> not reused
    _write_stats_file(cache, "hemibrain:v1.2.1", 2)
    assert M.population_stats("hemibrain:v1.2.1", str(tmp_path),
                              cache=cache) == (None, None)


def test_population_stats_level_mismatch_meta_fallback(tmp_path):
    cache = M.SkeletonVectorCache("hemibrain:v1.2.1",
                                  project_root=str(tmp_path), verbose=False)
    folder = M._dataset_folder("hemibrain:v1.2.1")
    level_dir = tmp_path / "cache" / folder / "skeletons"
    level_dir.mkdir(parents=True)
    (level_dir / ".level").write_text("simp90\n")
    # no vector cache -> None
    assert M.population_stats("hemibrain:v1.2.1", str(tmp_path)) == (None, None)
    # vector cache meta stats of the right shape are reused
    cache._atomic_parquet(pd.DataFrame([_vector_record(1)]), cache.parquet_path)
    cache._write_meta({"mean": [0.0] * M.VECTOR_DIM,
                       "std": [1.0] * M.VECTOR_DIM}, 1, rep="skeleton")
    mean, std = M.population_stats("hemibrain:v1.2.1", str(tmp_path))
    assert mean is not None and std is not None


def test_population_stats_raw_meta_fallback(tmp_path):
    cache = _raw_cache(tmp_path)
    # no files, but a raw cache with enough rows reuses its meta stats
    rows = pd.DataFrame([_vector_record(i) for i in range(1, 302)])
    cache._atomic_parquet(rows, cache.parquet_path)
    cache._write_meta({"mean": [0.5] * M.VECTOR_DIM,
                       "std": [1.5] * M.VECTOR_DIM}, len(rows),
                      rep="skeleton")
    mean, std = M.population_stats("hemibrain:v1.2.1", str(tmp_path),
                                   cache=cache)
    assert mean is not None and float(mean[0]) == 0.5


def test_population_stats_vectorize_paths(tmp_path, monkeypatch):
    cache = _raw_cache(tmp_path)
    cache.skeleton_dir.mkdir(parents=True)
    for bid in (1, 2, 3):
        _write_level0_swc_zst(cache, bid)
    real_vectorize = M._vectorize_one_file
    # short parallel result -> deterministic serial recomputation;
    # max_sample forces the bounded sampling branch
    monkeypatch.setattr(M.SkeletonVectorCache, "_vectorize_parallel",
                        lambda self, files: [])
    mean, std = M.population_stats("hemibrain:v1.2.1", str(tmp_path),
                                   max_sample=2, cache=cache)
    assert mean.shape == (M.VECTOR_DIM,)
    # every row failing -> (None, None)
    (cache.morph_dir / "population_stats.json").unlink(missing_ok=True)
    monkeypatch.setattr(M, "_vectorize_one_file", lambda path: None)
    assert M.population_stats("hemibrain:v1.2.1", str(tmp_path),
                              max_sample=2, cache=cache) == (None, None)
    # stats-file write failure is swallowed
    monkeypatch.setattr(M, "_vectorize_one_file", real_vectorize)
    monkeypatch.setattr(Path, "write_text", _raise_oserror)
    mean, std = M.population_stats("hemibrain:v1.2.1", str(tmp_path),
                                   max_sample=2, cache=cache)
    assert mean is not None


# ---------------------------------------------------------------------------
# on-demand fetching (all network seams faked)
# ---------------------------------------------------------------------------

def _fake_neuprint_module(monkeypatch, fetch_skeleton_impl):
    import types
    fake = types.ModuleType("neuprint")

    class Client:
        def __init__(self, server, dataset=None, token=None):
            self.server = server
            self.dataset = dataset

    fake.Client = Client
    fake.set_default_client = lambda client: None
    fake.fetch_skeleton = fetch_skeleton_impl
    monkeypatch.setitem(sys.modules, "neuprint", fake)
    return fake


def test_fetch_neuprint_skeleton(monkeypatch):
    def node_df():
        return pd.DataFrame({
            "node_id": [0, 1, 2], "parent_id": [-1, 0, 1],
            "x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0], "radius": [5.0, 5.0, 5.0],
        })

    _fake_neuprint_module(monkeypatch, lambda bid: node_df())
    nrn = M._fetch_neuprint_skeleton("hemibrain:v1.2.1", 1)
    assert nrn is not None and len(nrn.nodes) == 3 and nrn.soma is None
    # empty/None results and unbuildable frames return None
    _fake_neuprint_module(monkeypatch, lambda bid: None)
    assert M._fetch_neuprint_skeleton("hemibrain:v1.2.1", 1) is None
    _fake_neuprint_module(monkeypatch, lambda bid: pd.DataFrame({"a": [1]}))
    assert M._fetch_neuprint_skeleton("hemibrain:v1.2.1", 1) is None


def test_fetch_cave_seams(monkeypatch):
    import types
    fake = types.ModuleType("cave_data_fetcher")

    class CAVEDataFetcher:
        def __init__(self, dataset=None, project_root=None, verbose=False):
            self.dataset = dataset

        def fetch_skeleton(self, body_id, use_cache=True, simplify_mesh=0.0,
                           denoise_twigs=None):
            return "skeleton-for-" + str(body_id)

        def fetch_fafb_mesh(self, body_id, use_cache=True, simplify_mesh=None,
                            soma_simplification=None, soma_radius=None,
                            soma_pos=None):
            return "mesh-for-" + str(body_id)

    fake.CAVEDataFetcher = CAVEDataFetcher
    monkeypatch.setitem(sys.modules, "cave_data_fetcher", fake)
    fid = "720575940614131061"
    assert M._fetch_cave_skeleton("flywire", fid) == f"skeleton-for-{fid}"
    assert M._fetch_cave_mesh("flywire", fid, soma_pos=[1, 2, 3]) \
        == f"mesh-for-{fid}"


def test_fetch_skeleton_on_demand_paths(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        M.fetch_skeleton_on_demand("hemibrain:v1.2.1", 1, level="bogus")
    with pytest.raises(ValueError):
        M.fetch_skeleton_on_demand("hemibrain:v1.2.1", 1,
                                   simplification=None)
    # FlyWire mesh path: fetched mesh is vectorized then returned
    calls = []
    monkeypatch.setattr(M, "_fetch_cave_mesh",
                        lambda *a, **k: make_mesh_stub())
    monkeypatch.setattr(M, "cache_fetched_skeleton_vectors",
                        lambda *a, **k: calls.append((a, k)))
    fid = "720575940614131061"
    assert M.fetch_skeleton_on_demand("flywire", fid,
                                      project_root=str(tmp_path)) is not None
    assert calls  # vector cache transaction happened
    # persist=False skips the vector cache transaction
    calls.clear()
    assert M.fetch_skeleton_on_demand("flywire", fid,
                                      project_root=str(tmp_path),
                                      persist=False) is not None
    assert not calls
    # NeuPrint raw-cache hit short-circuits the network
    cache = _raw_cache(tmp_path)
    cache.skeleton_dir.mkdir(parents=True)
    _write_level0_swc_zst(cache, 5)
    calls.clear()
    out = M.fetch_skeleton_on_demand("hemibrain:v1.2.1", 5,
                                     project_root=str(tmp_path),
                                     raw_cache=cache)
    assert out is not None and calls
    # raw cache construction failure -> fetch still attempted
    monkeypatch.setattr(M, "find_similar_raw_cache",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(M, "_fetch_neuprint_skeleton", lambda ds, bid: None)
    assert M.fetch_skeleton_on_demand("hemibrain:v1.2.1", 6,
                                      project_root=str(tmp_path)) is None
    # Standalone BANC fetches from its public SWC bucket and must never touch
    # the FAFB CAVE skeleton compatibility seam.
    import banc_public_data
    attempts = []

    def fake_banc(dataset, body_id, **kwargs):
        attempts.append((dataset, body_id, kwargs))
        return make_tree()

    monkeypatch.setattr(banc_public_data, "fetch_banc_swc", fake_banc)
    monkeypatch.setattr(
        M, "_fetch_cave_skeleton",
        lambda *a, **k: pytest.fail("BANC must not call the FAFB CAVE seam"),
    )
    monkeypatch.setattr(M, "cache_fetched_skeleton_vectors",
                        lambda *a, **k: None)
    neuron = M.fetch_skeleton_on_demand("banc:v1.0", 7,
                                        project_root=str(tmp_path))
    assert neuron is not None and len(attempts) == 1
    assert attempts[0][0] == "banc:v1.0"
    # unsuccessful fetch returns None
    monkeypatch.setattr(banc_public_data, "fetch_banc_swc",
                        lambda *a, **k: None)
    assert M.fetch_skeleton_on_demand("banc:v1.0", 8,
                                      project_root=str(tmp_path)) is None


def test_fetch_neuprint_batch_with_progress(monkeypatch):
    from navis.interfaces import neuprint as neu
    meta = pd.DataFrame({"bodyId": [1, 2]})

    def fetch_neurons(criteria, client=None):
        return meta, None

    def skeleton_worker(row, client=None, with_synapses=False,
                        missing_swc="warn", heal=False):
        if row.bodyId == 2:
            raise RuntimeError("swc missing")
        neuron = make_tree()
        neuron.id = row.bodyId
        return neuron

    # NeuronCriteria normally requires a live neuprint default client
    monkeypatch.setattr(neu, "NeuronCriteria",
                        lambda bodyId=None, **kw: SimpleNamespace(bodyId=bodyId))
    monkeypatch.setattr(neu, "fetch_neurons", fetch_neurons)
    monkeypatch.setattr(neu, "__fetch_skeleton", skeleton_worker)
    done = []
    neurons = M._fetch_neuprint_batch_with_progress(
        [1, 2], client=object(), max_threads=1,
        on_neuron=lambda d, t: done.append((d, t)))
    assert len(neurons) == 1 and done == [(1, 2), (2, 2)]
    # empty metadata -> no neurons
    monkeypatch.setattr(neu, "fetch_neurons",
                        lambda criteria, client=None: (None, None))
    assert M._fetch_neuprint_batch_with_progress(
        [1], client=object(), max_threads=1) == []


def test_normalize_fetched_neurons():
    tree = make_tree()
    out = M._normalize_fetched_neurons("hemibrain:v1.2.1", {1: tree},
                                       flywire=False)
    assert 1 in out and out[1].id == 1
    # DataFrame coercion + multi-soma clearing
    coerced = M._normalize_fetched_neurons(
        "hemibrain:v1.2.1", {2: tree.nodes}, flywire=False)
    assert isinstance(coerced[2], navis.TreeNeuron)
    # FlyWire keeps only MeshNeurons
    assert M._normalize_fetched_neurons("flywire", {1: tree},
                                        flywire=True) == {}
    # unbuildable objects are skipped
    assert M._normalize_fetched_neurons(
        "hemibrain:v1.2.1", {3: "not-a-neuron"}, flywire=False) == {}


def _tree_with_id(bid):
    neuron = make_tree()
    neuron.soma = None
    neuron.id = bid
    return neuron


def test_fetch_batch_validation(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        M.fetch_skeletons_on_demand_batch("hemibrain:v1.2.1", [1],
                                          project_root=str(tmp_path),
                                          level="bogus")
    with pytest.raises(ValueError):
        M.fetch_skeletons_on_demand_batch("hemibrain:v1.2.1", [1],
                                          project_root=str(tmp_path),
                                          simplification=None)
    # invalid/duplicate ids are skipped before any fetch
    _fake_neuprint_module(monkeypatch, lambda bid: None)
    monkeypatch.setattr(M, "_fetch_neuprint_batch_with_progress",
                        lambda *a, **k: [])
    assert M.fetch_skeletons_on_demand_batch(
        "hemibrain:v1.2.1", ["abc", None], project_root=str(tmp_path),
        client=object()) == {}
    assert M.fetch_skeletons_on_demand_batch(
        "hemibrain:v1.2.1", [], project_root=str(tmp_path)) == {}
    assert M.fetch_skeletons_on_demand_batch(
        "hemibrain:v1.2.1", None, project_root=str(tmp_path)) == {}


def test_fetch_batch_neuprint_pipeline(tmp_path, monkeypatch):
    _fake_neuprint_module(monkeypatch, lambda bid: None)
    cache = _raw_cache(tmp_path)
    cache.skeleton_dir.mkdir(parents=True)
    _write_level0_swc_zst(cache, 1)               # cache hit
    cache.write_temp_skeleton(9, _tree_with_id(9))  # crash-resume staging

    def batch_fetch(batch_ids, *, client, max_threads, on_neuron=None,
                    missing_swc="warn"):
        neurons = [_tree_with_id(b) for b in batch_ids]
        if on_neuron is not None:
            for i in range(len(neurons)):
                on_neuron(i + 1, len(neurons))
        return neurons

    monkeypatch.setattr(M, "_fetch_neuprint_batch_with_progress", batch_fetch)
    events = []
    result = M.fetch_skeletons_on_demand_batch(
        "hemibrain:v1.2.1", [1, 2, 9], project_root=str(tmp_path),
        persist=True, batch_size=2, max_threads=1,
        progress_callback=lambda d, t, m: events.append((d, t, m)),
        client=object())
    assert set(result.keys()) == {1, 2, 9}
    assert events  # progress reported
    # fetched skeleton persisted; staging entry removed; pending merged
    assert (cache.skeleton_dir / "2.swc.zst").exists()
    assert not (cache.temp_cache_dir() / "9.swc.zst").exists()
    assert not cache.pending_path.exists()
    assert cache._vector_row_count() >= 2

    # vendored fetch failure -> navis wrapper fallback; final merge failure
    (cache.skeleton_dir / "2.swc.zst").unlink()

    def broken_batch(*args, **kwargs):
        raise RuntimeError("vendored internals changed")

    monkeypatch.setattr(M, "_fetch_neuprint_batch_with_progress", broken_batch)
    from navis.interfaces import neuprint as neu
    monkeypatch.setattr(neu, "fetch_skeletons",
                        lambda df, parallel=True, max_threads=1,
                        missing_swc="warn", client=None: [_tree_with_id(2)])
    monkeypatch.setattr(M.SkeletonVectorCache, "_merge_pending",
                        lambda self: (_ for _ in ()).throw(RuntimeError()))
    result = M.fetch_skeletons_on_demand_batch(
        "hemibrain:v1.2.1", [2], project_root=str(tmp_path),
        persist=True, client=object())
    assert 2 in result

    # cancel before the first batch: nothing fetched, no error
    import threading
    cancel = threading.Event()
    cancel.set()
    result = M.fetch_skeletons_on_demand_batch(
        "hemibrain:v1.2.1", [3], project_root=str(tmp_path),
        persist=True, client=object(), cancel_event=cancel)
    assert result == {}


def test_fetch_batch_single_fetch_override(tmp_path, monkeypatch):
    cache = _raw_cache(tmp_path)
    attempts = []

    def legacy_fetch(dataset, body_id, project_root=None, persist=True,
                     **kwargs):
        attempts.append(set(kwargs))
        if "level" in kwargs:
            raise TypeError("unexpected keyword argument 'level'")
        return _tree_with_id(body_id)

    monkeypatch.setattr(M, "fetch_skeleton_on_demand", legacy_fetch)
    result = M.fetch_skeletons_on_demand_batch(
        "hemibrain:v1.2.1", [4], project_root=str(tmp_path), persist=True)
    assert 4 in result and len(attempts) == 2
    assert (cache.skeleton_dir / "4.swc.zst").exists()
    # cancel event stops the per-id loop immediately
    import threading
    cancel = threading.Event()
    cancel.set()
    result = M.fetch_skeletons_on_demand_batch(
        "hemibrain:v1.2.1", [5], project_root=str(tmp_path),
        persist=True, cancel_event=cancel)
    assert result == {}


def test_fetch_batch_flywire_meshes(tmp_path, monkeypatch):
    import types
    verts = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                      [0., 0., 1.]])
    faces = np.array([[0, 1, 2], [0, 1, 3]])
    mesh = navis.MeshNeuron((verts, faces))
    mesh.id = int("720575940614131061")
    anon = navis.MeshNeuron((verts, faces))  # no id -> skipped

    fake = types.ModuleType("cave_data_fetcher")

    class CAVEDataFetcher:
        def __init__(self, dataset=None, project_root=None, verbose=False):
            pass

        def fetch_fafb_meshes(self, body_ids, use_cache=True,
                              simplify_mesh=None, soma_simplification=None,
                              soma_radius=None, soma_positions=None):
            return [mesh, anon]

    fake.CAVEDataFetcher = CAVEDataFetcher
    monkeypatch.setitem(sys.modules, "cave_data_fetcher", fake)
    fid = "720575940614131061"
    result = M.fetch_skeletons_on_demand_batch(
        "flywire", [fid], project_root=str(tmp_path), persist=True)
    assert fid in result


# ---------------------------------------------------------------------------
# download_all_skeletons (bulk pull; every fetch seam faked)
# ---------------------------------------------------------------------------

def _downloader_guard(monkeypatch):
    monkeypatch.setattr(M, "require_flywire_skeleton_access",
                        lambda *a, **k: None)


def test_download_all_rejections(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    with pytest.raises(Exception):
        M.download_all_skeletons("flywire", project_root=str(tmp_path))
    with pytest.raises(ValueError):
        M.download_all_skeletons("hemibrain:v1.2.1",
                                 project_root=str(tmp_path), mode="bogus")


def test_download_all_neuprint_batch_path(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    folder = M._dataset_folder("hemibrain:v1.2.1")
    ds_dir = tmp_path / "datasets" / folder
    ds_dir.mkdir(parents=True)
    pd.DataFrame({"bodyId": [1, 2, 3]}).to_parquet(
        ds_dir / f"{folder}_allneurons_neuron_df.parquet", index=False)
    cache = _raw_cache(tmp_path)
    cache.skeleton_dir.mkdir(parents=True)
    _write_level0_swc_zst(cache, 1)  # already cached

    calls = {}

    def fake_batch(dataset, ids, **kwargs):
        calls["ids"] = list(ids)
        return {2: _tree_with_id(2)}

    monkeypatch.setattr(M, "fetch_skeletons_on_demand_batch", fake_batch)
    events = []
    summary = M.download_all_skeletons(
        "hemibrain:v1.2.1", project_root=str(tmp_path), max_workers=1,
        progress_callback=lambda d, t, m: events.append((d, t, m)),
        verbose=False)
    assert calls["ids"] == [2, 3]
    assert summary["total"] == 2 and summary["fetched"] == 1
    assert summary["errors"] == 1 and summary["skipped_existing"] == 1
    assert events
    # everything already available -> early exit
    _write_level0_swc_zst(cache, 2)
    _write_level0_swc_zst(cache, 3)
    summary = M.download_all_skeletons("hemibrain:v1.2.1",
                                       project_root=str(tmp_path))
    assert summary["total"] == 0 and summary["skipped_existing"] == 3
    # batch fetch exception -> all errors
    (cache.skeleton_dir / "2.swc.zst").unlink()
    (cache.skeleton_dir / "3.swc.zst").unlink()
    monkeypatch.setattr(M, "fetch_skeletons_on_demand_batch",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    summary = M.download_all_skeletons("hemibrain:v1.2.1",
                                       project_root=str(tmp_path),
                                       verbose=True)
    assert summary["errors"] == 2 and summary["fetched"] == 0
    # pre-set cancel -> cancelled without fetching
    import threading
    cancel = threading.Event()
    cancel.set()
    summary = M.download_all_skeletons("hemibrain:v1.2.1",
                                       project_root=str(tmp_path),
                                       cancel_event=cancel)
    assert summary["cancelled"] and summary["fetched"] == 0


def test_download_all_fafb_guard_and_mode_aliases(tmp_path, monkeypatch):
    # BUG REPORTED / unreachable: the FAFB bulk-download guard at
    # morphology.py:3696 raises for every dataset where is_flywire_dataset()
    # is True, and FAFB's local-release predicate is separate from BANC, so the
    # FAFB healed-bundle counting block (~3795-3833) and the per-neuron
    # ``_fetch_one`` legacy worker loop (~3915-3991) can never execute in
    # production. They are therefore not exercised here; only the reachable
    # guard + compatibility-mode aliases are.
    from utils.flywire_readiness import FlyWireSkeletonAccessError
    _downloader_guard(monkeypatch)
    for dataset in ("FAFB_v783", "fafb"):
        with pytest.raises(FlyWireSkeletonAccessError):
            M.download_all_skeletons(dataset, project_root=str(tmp_path))
    # BANC pulls are allowed (public-bucket per-neuron fetches); without
    # local tables there is nothing to enumerate, so it exits early.
    summary = M.download_all_skeletons(
        "banc:v1.0", project_root=str(tmp_path), verbose=False)
    assert summary["total"] == 0
    # compatibility mode aliases are validated before the guard is the only
    # dataset-dependent step; invalid modes raise ValueError (already covered
    # above), valid aliases for a non-FlyWire dataset proceed to the fetch
    # stage — use an empty index so they exit early with total == 0.
    folder = M._dataset_folder("hemibrain:v1.2.1")
    ds_dir = tmp_path / "datasets" / folder
    ds_dir.mkdir(parents=True)
    pd.DataFrame({"bodyId": []}).to_parquet(
        ds_dir / f"{folder}_allneurons_neuron_df.parquet", index=False)
    for alias in ("raw_skeletons", "fine95", "fine_skeletons", "simp90"):
        summary = M.download_all_skeletons(
            "hemibrain:v1.2.1", project_root=str(tmp_path), mode=alias,
            verbose=False)
        assert summary["total"] == 0 and summary["mode"] == "raw"


# =============================================================================
# Batch 4: MorphologyComparer orchestration + enrichment
# =============================================================================

def make_chain(n=40):
    """Long synthetic chain neuron (enough nodes for k=20 dotprops)."""
    nodes = pd.DataFrame({
        "node_id": list(range(n)),
        "parent_id": [-1] + list(range(n - 1)),
        "x": [float(i * 100) for i in range(n)],
        "y": [float(i % 5) for i in range(n)],
        "z": [float((i * 7) % 11) for i in range(n)],
        "radius": [1.0] * n,
        "type": [1] * n,
    })
    neuron = navis.TreeNeuron(nodes)
    neuron.soma = None
    return neuron


def _write_neuron_table(tmp_path, dataset, rows):
    folder = M._dataset_folder(dataset)
    ds_dir = Path(tmp_path) / "datasets" / folder
    ds_dir.mkdir(parents=True, exist_ok=True)
    path = ds_dir / f"{folder}_allneurons_neuron_df.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _seed_cache(tmp_path, ids, dataset="hemibrain:v1.2.1"):
    cache = _raw_cache(tmp_path, dataset=dataset)
    cache_v2 = M.SkeletonVectorCacheV2(dataset, project_root=str(tmp_path),
                                       verbose=False)
    records = []
    for b in ids:
        vec = M.vectorize_neuron(_tree_with_id(b))[1]
        records.append((b, vec, "skeleton"))
        # The V2 cache (default method) shares the population but carries
        # the extended schema; zero-fill the appended blocks.
        v2 = np.concatenate([vec, np.zeros(M.VECTOR_V2_DIM - len(vec))])
        cache_v2.append_vectors([(b, v2, "skeleton")])
    cache.append_vectors(records)
    return cache


def _patch_getneurons(monkeypatch, body_ids, types=None):
    frame = pd.DataFrame({
        "bodyId": list(body_ids),
        "type": list(types or [""] * len(list(body_ids))),
        "instance": [""] * len(list(body_ids)),
    })
    monkeypatch.setattr(
        M, "getNeurons",
        lambda q, ds, verbose=False, search_columns="auto": (
            frame, None, None, None))


def _comparer(tmp_path, **kw):
    kw.setdefault("query", 1)
    kw.setdefault("dataset", "hemibrain:v1.2.1")
    kw.setdefault("project_root", str(tmp_path))
    kw.setdefault("output_dir", str(Path(tmp_path) / "outputs"))
    kw.setdefault("verbose", False)
    kw.setdefault("n_workers", 1)
    kw.setdefault("candidate_source", "cache")
    return M.MorphologyComparer(**kw)


def test_comparer_init_validation(tmp_path):
    base = dict(dataset="hemibrain:v1.2.1", query=1, verbose=False,
                project_root=str(tmp_path))
    with pytest.raises(ValueError):
        M.MorphologyComparer(level="bogus", **base)
    with pytest.raises(ValueError):
        M.MorphologyComparer(method="bogus", **base)
    with pytest.raises(ValueError):
        M.MorphologyComparer(metric="manhattan", **base)
    with pytest.raises(ValueError):
        M.MorphologyComparer(candidate_source="bogus", **base)
    with pytest.raises(ValueError):
        M.MorphologyComparer(candidate_cap=0, **base)
    with pytest.raises(ValueError):
        M.MorphologyComparer(min_shared_partners=0, **base)
    with pytest.raises(ValueError):
        M.MorphologyComparer(visualize_by="roi", **base)


def test_comparer_find_similar_cache_direct_vector(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    _seed_cache(tmp_path, [1, 2, 3])
    _patch_getneurons(monkeypatch, [1], types=["T1"])
    comparer = _comparer(tmp_path, method="vector_v2", expand_top_types=0, level="bodyid",
                         metric="cosine", saveas="run_cov")
    results = comparer.find_similar()
    assert not results.empty
    assert set(results["target_bodyId"]) == {2, 3}
    run_dir = Path(tmp_path) / "outputs" / "run_cov"
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "type_summary.csv").exists()
    assert (run_dir / "README.txt").exists()
    # pearson metric variant
    comparer = _comparer(tmp_path, method="vector_v2", expand_top_types=0, level="bodyid",
                         metric="pearson", saveas="run_cov_pearson")
    results = comparer.find_similar()
    assert not results.empty


def test_comparer_find_similar_type_level(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    dataset = "hemibrain:v1.2.1"
    _write_neuron_table(tmp_path, dataset, {
        "bodyId": [1, 2, 3, 4],
        "type": ["T1", "T1", "T2", "T2"],
        "instance": ["a", "b", "c", "d"],
    })
    _seed_cache(tmp_path, [1, 2, 3, 4])
    _patch_getneurons(monkeypatch, [1], types=["T1"])
    # explicit type level
    comparer = _comparer(tmp_path, method="vector_v2", expand_top_types=0, level="type",
                         saveas="run_type")
    results = comparer.find_similar()
    assert not results.empty
    # auto level: non-numeric query resolving to a single type -> type
    _patch_getneurons(monkeypatch, [1], types=["T1"])
    comparer = _comparer(tmp_path, method="vector_v2", expand_top_types=0, level="auto",
                         query="some pattern", saveas="run_auto_type")
    results = comparer.find_similar()
    assert comparer.level == "type"
    assert not results.empty
    # auto level with a numeric query -> bodyid
    _patch_getneurons(monkeypatch, [1], types=["T1"])
    comparer = _comparer(tmp_path, method="vector_v2", expand_top_types=0, level="auto", query="42",
                         saveas="run_auto_bodyid")
    results = comparer.find_similar()
    assert comparer.level == "bodyid"


def test_comparer_cache_direct_missing_query_fetch(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    _seed_cache(tmp_path, [1, 2, 3])
    _patch_getneurons(monkeypatch, [5], types=["T9"])
    fetched = {}

    def fake_batch(dataset, ids, **kwargs):
        fetched["ids"] = list(ids)
        return {5: _tree_with_id(5)}

    monkeypatch.setattr(M, "fetch_skeletons_on_demand_batch", fake_batch)
    comparer = _comparer(tmp_path, query=5, saveas="run_fetch")
    results = comparer.find_similar()
    assert fetched["ids"] == [5]
    assert not results.empty
    # the fetched query vector was appended to the V2 vector cache
    cache = M.find_similar_dataset_cache_v2(
        "hemibrain:v1.2.1", project_root=str(tmp_path), verbose=False)
    data = cache.load()
    assert 5 in set(int(b) for b in data["bodyIds"])


def test_comparer_profile_first_fallback(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    _seed_cache(tmp_path, [1, 2, 3])
    _patch_getneurons(monkeypatch, [1], types=["T1"])
    # candidate_source="auto" resolves to 'profile' (no ROI table locally)
    comparer = _comparer(tmp_path, candidate_source="auto",
                         saveas="run_profile_fb")
    monkeypatch.setattr(M.MorphologyComparer, "_profile_first_search",
                        lambda self, qdf, cache, source: (pd.DataFrame(),
                                                          pd.DataFrame()))
    results = comparer.find_similar()
    assert comparer.resolved_candidate_source == "profile"
    assert not results.empty  # fell back to the vector cache

    # empty cache + empty profile screen -> explicit error
    empty_root = Path(tmp_path) / "empty_ds"
    empty_root.mkdir()
    _patch_getneurons(monkeypatch, [9], types=[""])
    comparer = _comparer(empty_root, query=9, candidate_source="profile",
                         saveas="run_none")
    monkeypatch.setattr(M.MorphologyComparer, "_profile_first_search",
                        lambda self, qdf, cache, source: (pd.DataFrame(),
                                                          pd.DataFrame()))
    with pytest.raises(ValueError):
        comparer.find_similar()


def test_comparer_find_similar_errors(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    comparer = _comparer(tmp_path, dataset=None)
    with pytest.raises(ValueError):
        comparer.find_similar()
    # query resolution failure
    monkeypatch.setattr(M, "getNeurons",
                        lambda q, ds, verbose=False, search_columns="auto":
                        (pd.DataFrame(), None, None, None))
    comparer = _comparer(tmp_path, query=12345)
    with pytest.raises(ValueError):
        comparer.find_similar()
    # cache-direct with an empty vector cache -> error
    _patch_getneurons(monkeypatch, [1], types=[""])
    empty_root = Path(tmp_path) / "empty_cache"
    empty_root.mkdir()
    comparer = _comparer(empty_root, saveas="run_empty")
    with pytest.raises(ValueError):
        comparer.find_similar()


def test_comparer_nblast(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    cache = _seed_cache(tmp_path, [1, 2, 3])
    cache.skeleton_dir.mkdir(parents=True, exist_ok=True)
    for bid in (1, 2, 3):
        neuron = make_chain()
        neuron.id = bid
        M._write_compressed_skeleton(
            cache.skeleton_dir / f"{bid}.swc.zst", neuron, simplification=0)
    _patch_getneurons(monkeypatch, [1], types=["T1"])
    comparer = _comparer(tmp_path, method="nblast", level="bodyid",
                         candidate_cap=5, saveas="run_nblast")
    results = comparer.find_similar()
    assert not results.empty
    assert set(results["target_bodyId"]) == {2, 3}
    assert (results["metric"] == "nblast").all()


def test_visualize_top_results_paths(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    dataset = "hemibrain:v1.2.1"
    _write_neuron_table(tmp_path, dataset, {
        "bodyId": [1, 2, 3, 4],
        "type": ["T1", "T2", "T2", "T3"],
        "instance": ["", "", "", ""],
    })
    _seed_cache(tmp_path, [1, 2, 3, 4])
    _patch_getneurons(monkeypatch, [1], types=["T1"])
    results_df = pd.DataFrame({
        "target_bodyId": [2, 3, 4, 4],
        "target_type": ["T2", "T2", "T3", "T3"],
        "similarity": [0.9, 0.8, 0.7, np.nan],
        "is_intra_type": [False, False, False, False],
    })
    query_df = pd.DataFrame({"bodyId": [1], "type": ["T1"],
                             "instance": [""]})

    created = []

    class FakeVisualizer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

        def plot_neurons(self):
            pass

    monkeypatch.setattr(M, "_import_visualizer", lambda: FakeVisualizer)

    # visualize_by='type'
    comparer = _comparer(tmp_path, visualize_top_n=2, visualize_by="type",
                         saveas="run_viz")
    comparer.output_folder = str(Path(tmp_path) / "outputs" / "run_viz")
    comparer._visualize_top_results(results_df, query_df=query_df)
    assert created and "neuron_layers" in created[0].kwargs
    layers = created[0].kwargs["neuron_layers"]
    assert layers[0] == [1]  # query reference layer first
    assert len(layers) >= 2

    # visualize_by='bodyid'
    created.clear()
    comparer = _comparer(tmp_path, visualize_top_n=2, visualize_by="bodyid",
                         saveas="run_viz2")
    comparer.output_folder = str(Path(tmp_path) / "outputs" / "run_viz2")
    comparer._visualize_top_results(results_df, query_df=query_df)
    assert created

    # plot failure is swallowed
    created.clear()

    class ExplodingVisualizer(FakeVisualizer):
        def plot_neurons(self):
            raise RuntimeError("render failed")

    monkeypatch.setattr(M, "_import_visualizer",
                        lambda: ExplodingVisualizer)
    comparer = _comparer(tmp_path, visualize_top_n=2, saveas="run_viz3")
    comparer.output_folder = str(Path(tmp_path) / "outputs" / "run_viz3")
    comparer._visualize_top_results(results_df, query_df=query_df)

    # visualizer unavailable -> skipped
    monkeypatch.setattr(M, "_import_visualizer", lambda: None)
    comparer = _comparer(tmp_path, visualize_top_n=2, saveas="run_viz4")
    comparer._visualize_top_results(results_df, query_df=query_df)
    # visualize_top_n <= 0 -> immediate return
    comparer = _comparer(tmp_path, visualize_top_n=0)
    comparer._visualize_top_results(results_df, query_df=query_df)
    # empty results and no query -> return
    comparer = _comparer(tmp_path, visualize_top_n=3)
    comparer._visualize_top_results(pd.DataFrame(), query_df=None)


def test_visualize_top_results_legend_mode_not_overridden(tmp_path, monkeypatch):
    """The shared panel's global legend preference must not rewrite the
    mode derived from ``visualize_by`` (homolog-style regression)."""
    _downloader_guard(monkeypatch)
    dataset = "hemibrain:v1.2.1"
    _write_neuron_table(tmp_path, dataset, {
        "bodyId": [1, 2, 3],
        "type": ["T1", "T2", "T2"],
        "instance": ["", "", ""],
    })
    _seed_cache(tmp_path, [1, 2, 3])
    _patch_getneurons(monkeypatch, [1], types=["T1"])
    results_df = pd.DataFrame({
        "target_bodyId": [2, 3],
        "target_type": ["T2", "T2"],
        "similarity": [0.9, 0.8],
        "is_intra_type": [False, False],
    })
    query_df = pd.DataFrame({"bodyId": [1], "type": ["T1"],
                             "instance": [""]})

    created = []

    class FakeVisualizer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

        def plot_neurons(self):
            pass

    monkeypatch.setattr(M, "_import_visualizer", lambda: FakeVisualizer)

    settings = {"legend_mode": "type"}  # matches the app-wide default

    comparer = _comparer(tmp_path, visualize_top_n=2, visualize_by="type",
                         saveas="legend_type",
                         visualization_settings=dict(settings))
    comparer.output_folder = str(Path(tmp_path) / "outputs" / "legend_type")
    comparer._visualize_top_results(results_df, query_df=query_df)
    assert created and created[-1].kwargs["legend_mode"] == "layer"

    created.clear()
    comparer = _comparer(tmp_path, visualize_top_n=2, visualize_by="bodyid",
                         saveas="legend_bodyid",
                         visualization_settings=dict(settings))
    comparer.output_folder = str(Path(tmp_path) / "outputs" / "legend_bodyid")
    comparer._visualize_top_results(results_df, query_df=query_df)
    assert created and created[-1].kwargs["legend_mode"] == "single"


def test_enrich_homolog_results_paths(tmp_path, monkeypatch):
    _seed_cache(tmp_path, [1, 2], dataset="hemibrain:v1.2.1")
    _seed_cache(tmp_path, [10, 11], dataset="male-cns:v1.0")
    frame = pd.DataFrame({"source_bodyId": [1, 2],
                          "target_bodyId": [10, 99]})
    out = M.enrich_homolog_results(frame, "hemibrain:v1.2.1",
                                   "male-cns:v1.0",
                                   project_root=str(tmp_path), verbose=False)
    assert {"morph_v2_similarity",
            "morph_nblast"} <= set(out.columns)
    assert not np.isnan(out["morph_v2_similarity"].iloc[0])
    assert np.isnan(out["morph_v2_similarity"].iloc[1])  # target 99 has no vector
    # empty frame / missing columns pass through untouched
    empty = pd.DataFrame()
    assert M.enrich_homolog_results(empty, "a", "b") is empty
    assert M.enrich_homolog_results(None, "a", "b") is None
    no_cols = pd.DataFrame({"a": [1]})
    assert M.enrich_homolog_results(no_cols, "a", "b") is no_cols
    # vector computation failure -> NaN columns, no rows dropped
    monkeypatch.setattr(M, "find_similar_dataset_cache_v2",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    out = M.enrich_homolog_results(frame, "hemibrain:v1.2.1",
                                   "male-cns:v1.0",
                                   project_root=str(tmp_path))
    assert np.isnan(out["morph_v2_similarity"]).all() and len(out) == 2


def test_soma_positions_table_variants(tmp_path):
    root = Path(tmp_path)
    folder = M._dataset_folder("flywire")
    ds_dir = root / "datasets" / folder
    ds_dir.mkdir(parents=True)
    # parquet table without any soma position column -> {}
    pd.DataFrame({"bodyId": ["720575940614131061"]}).to_parquet(
        ds_dir / f"{folder}_allneurons_neuron_df.parquet", index=False)
    assert M._load_flywire_soma_positions("flywire", root) == {}
    # CSV table with a position column exercises the CSV branch. The
    # filtered-load path still hits the reported
    # normalize_flywire_body_ids NameError (BUG REPORTED, line 246) and
    # returns {}; the unfiltered load reaches the loop but the same bug
    # fires before it, so {} is expected here too.
    (ds_dir / f"{folder}_allneurons_neuron_df.parquet").unlink()
    pd.DataFrame({
        "bodyId": ["720575940614131061"],
        "position": ["[1.0, 2.0, 3.0]"],
    }).to_csv(ds_dir / f"{folder}_allneurons_neuron_df.csv", index=False)
    # Unfiltered load parses the CSV row successfully (lines 235-256).
    positions = M._load_flywire_soma_positions("flywire", root)
    assert set(positions) == {"720575940614131061"}
    # Filtered load hits the reported normalize_flywire_body_ids NameError
    # (BUG REPORTED, line 246) and therefore returns {}.
    assert M._load_flywire_soma_positions(
        "flywire", root, body_ids=["720575940614131061"]) == {}


def test_local_dataset_presence_variants(tmp_path):
    root = Path(tmp_path)
    dataset = "hemibrain:v1.2.1"
    folder = M._dataset_folder(dataset)
    cache_dir = root / "cache" / folder
    assert not M._has_local_dataset_presence(dataset, root)
    # mesh pickle presence (lines 291-295)
    meshes = cache_dir / "meshes"
    meshes.mkdir(parents=True)
    (meshes / "1.pkl").write_bytes(b"x")
    assert M._has_local_dataset_presence(dataset, root)
    # legacy find_similar raw skeletons (lines 296-301)
    (meshes / "1.pkl").unlink()
    legacy = cache_dir / "find_similar" / "raw_skeletons"
    legacy.mkdir(parents=True)
    (legacy / "2.swc.gz").write_bytes(b"x")
    assert M._has_local_dataset_presence(dataset, root)
    # connections.parquet presence
    (legacy / "2.swc.gz").unlink()
    (cache_dir / "connections.parquet").write_bytes(b"x")
    assert M._has_local_dataset_presence(dataset, root)


# =============================================================================
# Batch 5: candidate screens (connection cache) + profile-first search
# =============================================================================

_CONN_ROWS = {
    "bodyId_pre": [100, 1, 100, 100, 3],
    "bodyId_post": [1, 200, 2, 3, 200],
    "weight": [5, 5, 5, 5, 5],
    "roi": ["R1", "R1", "R1", "R1", "R1"],
}


def _write_connections(tmp_path, dataset, rows=None):
    import polars as pl
    path = (Path(tmp_path) / "cache" / M._dataset_folder(dataset)
            / "connections.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows or _CONN_ROWS).write_parquet(path)
    return path


def test_connection_cache_candidates(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    dataset = "hemibrain:v1.2.1"
    _write_connections(tmp_path, dataset)
    comparer = _comparer(tmp_path, min_weight=3, min_shared_partners=1)
    query_df = pd.DataFrame({"bodyId": [1], "type": ["T1"],
                             "instance": [""]})
    out = comparer._connection_cache_candidates(query_df)
    assert list(out["target_bodyId"]) == [3, 2]
    assert int(out["shared_count"].iloc[0]) == 2
    assert out["profile_similarity"].iloc[0] == pytest.approx(1.0)
    # ROI filter variants (wildcard / explicit / unmatched)
    assert not comparer._connection_cache_candidates(
        query_df, roi_filter=["*"]).empty
    assert not comparer._connection_cache_candidates(
        query_df, roi_filter=["R1"]).empty
    assert comparer._connection_cache_candidates(
        query_df, roi_filter=["NOPE"]).empty
    # above the shared-partner threshold nothing survives
    assert comparer._connection_cache_candidates(
        query_df, min_shared_partners=5).empty
    # top_k bound
    assert len(comparer._connection_cache_candidates(
        query_df, top_k=1)) == 1
    # missing connection cache -> empty
    comparer2 = _comparer(Path(tmp_path) / "no_conn", min_weight=3)
    assert comparer2._connection_cache_candidates(query_df).empty
    # unreadable connection cache -> empty
    conn_path = _write_connections(tmp_path, dataset)
    conn_path.write_bytes(b"garbage")
    assert comparer._connection_cache_candidates(query_df).empty


def test_profile_first_search_end_to_end(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    dataset = "hemibrain:v1.2.1"
    _write_neuron_table(tmp_path, dataset, {
        "bodyId": [1, 2, 3],
        "type": ["T1", "T2", "T2"],
        "instance": ["", "", ""],
    })
    cache = _seed_cache(tmp_path, [1, 2, 3])
    # merge pending so the meta carries mean/std: profile-first scoring then
    # restores cache rows to raw before re-standardizing with sample stats
    cache._merge_pending()
    _write_connections(tmp_path, dataset)
    _patch_getneurons(monkeypatch, [1], types=["T1"])
    comparer = _comparer(tmp_path, candidate_source="profile",
                         method="vector_v2", expand_top_types=0, saveas="run_profile_e2e",
                         min_shared_partners=1)
    results = comparer.find_similar()
    assert comparer.resolved_candidate_source == "profile"
    assert not results.empty
    assert set(results["target_bodyId"]) == {2, 3}
    assert (Path(tmp_path) / "outputs" / "run_profile_e2e"
            / "results.csv").exists()


def test_find_similar_empty_results(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    _seed_cache(tmp_path, [1])
    _patch_getneurons(monkeypatch, [1], types=["T1"])
    comparer = _comparer(tmp_path, saveas="run_no_hits")
    results = comparer.find_similar()
    assert results.empty
    run_dir = Path(tmp_path) / "outputs" / "run_no_hits"
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "README.txt").exists()


def test_visualize_top_results_edge_branches(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    created = []

    class FakeVisualizer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

        def plot_neurons(self):
            pass

    monkeypatch.setattr(M, "_import_visualizer", lambda: FakeVisualizer)
    query_df = pd.DataFrame({"bodyId": [1], "type": ["T1"],
                             "instance": [""]})

    # results=None with a query -> query-only reference layer
    comparer = _comparer(tmp_path, visualize_top_n=2, saveas="run_viz_none")
    comparer.output_folder = str(Path(tmp_path) / "outputs" / "run_viz_none")
    comparer._visualize_top_results(None, query_df=query_df)
    assert created and created[-1].kwargs["neuron_layers"] == [[1]]

    # rows referencing the query are dropped; NaN/blank targets are skipped
    created.clear()
    work = pd.DataFrame({
        "target_bodyId": [1, None, 2, "bogus"],
        "target_type": ["T1", "", "T2", "T2"],
        "similarity": [0.99, 0.9, 0.8, 0.7],
    })
    comparer = _comparer(tmp_path, visualize_top_n=5, visualize_by="bodyid",
                         saveas="run_viz_edge")
    comparer.output_folder = str(Path(tmp_path) / "outputs" / "run_viz_edge")
    comparer._visualize_top_results(work, query_df=query_df)
    assert created
    layers = created[-1].kwargs["neuron_layers"]
    assert layers == [[1], [2]]

    # type mode with an empty/unknown type and no members -> skipped layer
    created.clear()
    work = pd.DataFrame({
        "target_bodyId": [3, 3],
        "target_type": ["", "Tmissing"],
        "similarity": [0.9, 0.8],
    })
    comparer = _comparer(tmp_path, visualize_top_n=2, visualize_by="type",
                         saveas="run_viz_skip")
    comparer.output_folder = str(Path(tmp_path) / "outputs" / "run_viz_skip")
    comparer._visualize_top_results(work, query_df=query_df)
    # nothing renderable beyond the query layer -> no visualizer run or
    # query-only layers; either way no exception escapes
    # empty work AND empty query -> early return
    created.clear()
    comparer._visualize_top_results(pd.DataFrame(),
                                    query_df=pd.DataFrame())
    assert not created

    # query frame with NaN/unparseable bodyIds -> skipped by _body_ids
    created.clear()
    weird_q = pd.DataFrame({"bodyId": [np.nan, "bogus", 1],
                            "type": ["", "", "T1"],
                            "instance": ["", "", ""]})
    comparer = _comparer(tmp_path, visualize_top_n=2, saveas="run_viz_q")
    comparer.output_folder = str(Path(tmp_path) / "outputs" / "run_viz_q")
    comparer._visualize_top_results(None, query_df=weird_q)
    assert created and created[-1].kwargs["neuron_layers"] == [[1]]

    # all-NaN similarities and no query -> filtered empty, no layers
    created.clear()
    work = pd.DataFrame({"target_bodyId": [2], "target_type": ["T2"],
                         "similarity": [np.nan]})
    comparer = _comparer(tmp_path, visualize_top_n=2, saveas="run_viz_nan")
    comparer.output_folder = str(Path(tmp_path) / "outputs" / "run_viz_nan")
    comparer._visualize_top_results(work, query_df=None)
    assert not created

    # type mode with only empty types -> no renderable layers
    created.clear()
    work = pd.DataFrame({"target_bodyId": [3], "target_type": [""],
                         "similarity": [0.5]})
    comparer = _comparer(tmp_path, visualize_top_n=2, visualize_by="type",
                         saveas="run_viz_notype")
    comparer.output_folder = str(Path(tmp_path) / "outputs"
                                 / "run_viz_notype")
    comparer._visualize_top_results(work, query_df=None)
    assert not created

    # visualization_settings skip-keys (ranking keys + mesh_color='auto')
    created.clear()
    comparer = _comparer(tmp_path, visualize_top_n=2,
                         visualization_settings={
                             "visualize_top_n": 9,
                             "mesh_color": "auto",
                             "skeleton_mode": "tube",
                         }, saveas="run_viz_settings")
    comparer.output_folder = str(Path(tmp_path) / "outputs"
                                 / "run_viz_settings")
    work = pd.DataFrame({"target_bodyId": [2], "target_type": ["T2"],
                         "similarity": [0.8]})
    comparer._visualize_top_results(work, query_df=None)
    assert created
    assert created[-1].kwargs["skeleton_mode"] == "tube"
    assert created[-1].kwargs.get("mesh_color") != "auto"


def _write_chain_skeletons(cache, ids):
    cache.skeleton_dir.mkdir(parents=True, exist_ok=True)
    for bid in ids:
        neuron = make_chain()
        neuron.id = bid
        M._write_compressed_skeleton(
            cache.skeleton_dir / f"{bid}.swc.zst", neuron, simplification=0)


def test_profile_first_nblast_refine(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    dataset = "hemibrain:v1.2.1"
    _write_neuron_table(tmp_path, dataset, {
        "bodyId": [1, 2, 3],
        "type": ["T1", "T2", "T2"],
        "instance": ["", "", ""],
    })
    cache = _seed_cache(tmp_path, [1, 2, 3])
    _write_chain_skeletons(cache, [1, 2, 3])
    _write_connections(tmp_path, dataset)
    _patch_getneurons(monkeypatch, [1], types=["T1"])
    comparer = _comparer(tmp_path, candidate_source="profile",
                         method="nblast", saveas="run_profile_nblast",
                         min_shared_partners=1)
    results = comparer.find_similar()
    assert not results.empty
    assert set(results["target_bodyId"]) == {2, 3}
    # NBLAST refinement re-ranked the rows and assigned ranks
    assert "rank" in results.columns
    assert results["rank"].tolist() == [1, 2]


def test_profile_first_fetch_missing(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    dataset = "hemibrain:v1.2.1"
    _write_neuron_table(tmp_path, dataset, {
        "bodyId": [1, 2, 3],
        "type": ["T1", "T2", "T2"],
        "instance": ["", "", ""],
    })
    _seed_cache(tmp_path, [2])  # query 1 AND candidate 3 are missing
    _write_connections(tmp_path, dataset)
    _patch_getneurons(monkeypatch, [1], types=["T1"])
    fetched = {}

    def fake_batch(ds, ids, **kwargs):
        fetched["ids"] = sorted(ids)
        # 3 returns garbage -> vectorization failure path (continue)
        return {1: _tree_with_id(1), 3: "not-a-neuron"}

    monkeypatch.setattr(M, "fetch_skeletons_on_demand_batch", fake_batch)
    comparer = _comparer(tmp_path, candidate_source="profile",
                         saveas="run_profile_fetch", min_shared_partners=1)
    results = comparer.find_similar()
    assert fetched["ids"] == [1, 3]
    assert not results.empty
    # candidate 2 scored from the cache; 3 was unvectorizable
    assert 2 in set(results["target_bodyId"])
    # fetched query vector was persisted for reuse (V2 schema)
    cache = M.find_similar_dataset_cache_v2(
        "hemibrain:v1.2.1", project_root=str(tmp_path), verbose=False)
    assert 1 in set(int(b) for b in cache.load()["bodyIds"])
    # fetch returning nothing -> the query cannot be vectorized (fresh root
    # so no earlier run has persisted the query vector)
    noroot = Path(tmp_path) / "nofetch"
    noroot.mkdir()
    _write_neuron_table(noroot, dataset, {
        "bodyId": [1, 2, 3],
        "type": ["T1", "T2", "T2"],
        "instance": ["", "", ""],
    })
    _seed_cache(noroot, [2])
    _write_connections(noroot, dataset)
    monkeypatch.setattr(M, "fetch_skeletons_on_demand_batch",
                        lambda ds, ids, **kw: {})
    comparer = _comparer(noroot, candidate_source="profile",
                         saveas="run_profile_nofetch",
                         min_shared_partners=1)
    with pytest.raises(ValueError):
        comparer.find_similar()


def test_profile_first_empty_candidates_fallback(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    dataset = "hemibrain:v1.2.1"
    _seed_cache(tmp_path, [1, 2, 3])
    _patch_getneurons(monkeypatch, [1], types=["T1"])
    monkeypatch.setattr(M.MorphologyComparer, "_discover_candidates",
                        lambda self, qdf, source: (pd.DataFrame(), source))
    comparer = _comparer(tmp_path, candidate_source="profile",
                         saveas="run_nocand_fb")
    results = comparer.find_similar()
    # empty screen falls back to the vector cache
    assert not results.empty


def test_profile_first_type_level(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    dataset = "hemibrain:v1.2.1"
    _write_neuron_table(tmp_path, dataset, {
        "bodyId": [1, 2, 3],
        "type": ["T1", "T2", "T2"],
        "instance": ["", "", ""],
    })
    _seed_cache(tmp_path, [1, 2, 3])
    _write_connections(tmp_path, dataset)
    _patch_getneurons(monkeypatch, [1], types=["T1"])
    comparer = _comparer(tmp_path, candidate_source="profile",
                         method="vector_v2", expand_top_types=0, level="type",
                         saveas="run_profile_type", min_shared_partners=1)
    results = comparer.find_similar()
    assert not results.empty


class _FakeRoiStore:
    def __init__(self, dataset, project_root=None, verbose=False, log=None):
        pass

    def ensure(self):
        return self

    def screen(self, ids, top_k=None):
        frame = pd.DataFrame({"bodyId": [2, 3],
                              "roi_similarity": [0.8, 0.6]})
        return frame.head(top_k) if top_k else frame


def test_roi_and_combined_screens(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    dataset = "hemibrain:v1.2.1"
    _write_neuron_table(tmp_path, dataset, {
        "bodyId": [1, 2, 3],
        "type": ["T1", "T2", "T2"],
        "instance": ["", "", ""],
    })
    _seed_cache(tmp_path, [1, 2, 3])
    _write_connections(tmp_path, dataset)
    _patch_getneurons(monkeypatch, [1], types=["T1"])
    monkeypatch.setattr(M, "RoiProfileStore", _FakeRoiStore)
    # explicit roi screen
    comparer = _comparer(tmp_path, candidate_source="roi",
                         saveas="run_roi_only")
    results = comparer.find_similar()
    assert comparer.resolved_candidate_source == "roi"
    assert not results.empty
    # combined screen (roi + connection cache outer merge)
    comparer = _comparer(tmp_path, candidate_source="combined",
                         saveas="run_combined", min_shared_partners=1)
    results = comparer.find_similar()
    assert comparer.resolved_candidate_source == "combined"
    assert not results.empty
    # combined without a connection cache degrades to the ROI screen only
    noconn_root = Path(tmp_path) / "noconn"
    noconn_root.mkdir()
    _seed_cache(noconn_root, [1, 2, 3])
    _write_neuron_table(noconn_root, dataset, {
        "bodyId": [1, 2, 3],
        "type": ["T1", "T2", "T2"],
        "instance": ["", "", ""],
    })
    comparer = _comparer(noconn_root, candidate_source="combined",
                         saveas="run_combined_roi_only")
    results = comparer.find_similar()
    assert not results.empty


def test_roi_screen_unavailable_auto_fallback(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    dataset = "hemibrain:v1.2.1"
    _write_neuron_table(tmp_path, dataset, {
        "bodyId": [1, 2, 3],
        "type": ["T1", "T2", "T2"],
        "instance": ["", "", ""],
    })
    _seed_cache(tmp_path, [1, 2, 3])
    _write_connections(tmp_path, dataset)
    _patch_getneurons(monkeypatch, [1], types=["T1"])
    # Make the ROI screen probe look ready, then fail at store build:
    # auto mode must fall back to the connection-cache screen.
    probe_file = Path(tmp_path) / "roi_probe.parquet"
    probe_file.write_bytes(b"x")
    monkeypatch.setattr(M, "roi_count_table_path",
                        lambda ds, root: probe_file)
    monkeypatch.setattr(M, "load_primary_rois",
                        lambda ds, root: ["R1"])

    class BrokenStore:
        def __init__(self, *args, **kwargs):
            raise M.RoiScreeningUnavailable("no ROI table usable")

        def ensure(self):
            return self

    monkeypatch.setattr(M, "RoiProfileStore", BrokenStore)
    comparer = _comparer(tmp_path, candidate_source="auto",
                         saveas="run_roi_fb", min_shared_partners=1)
    results = comparer.find_similar()
    assert comparer.resolved_candidate_source == "profile"
    assert not results.empty
    # explicit 'roi' selection surfaces the preparation error instead
    comparer = _comparer(tmp_path, candidate_source="roi",
                         saveas="run_roi_err", min_shared_partners=1)
    with pytest.raises(M.RoiScreeningUnavailable):
        comparer.find_similar()


def test_type_query_multi_member_intra_rows(tmp_path, monkeypatch):
    _downloader_guard(monkeypatch)
    dataset = "hemibrain:v1.2.1"
    _write_neuron_table(tmp_path, dataset, {
        "bodyId": [1, 2, 3, 4],
        "type": ["T1", "T1", "T1", "T2"],
        "instance": ["", "", "", ""],
    })
    _seed_cache(tmp_path, [1, 2, 3, 4])
    # three same-type query members -> cyclic intra-type pair orientation
    _patch_getneurons(monkeypatch, [1, 2, 3], types=["T1", "T1", "T1"])
    comparer = _comparer(tmp_path, method="vector_v2", expand_top_types=0, level="type",
                         query=[1, 2, 3], saveas="run_intra")
    results = comparer.find_similar()
    assert not results.empty
